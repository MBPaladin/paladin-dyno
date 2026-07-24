"""Long-duration (24 h) EtherCAT soak / jitter test for the Linux dyno host.

Purpose
-------
Exercise the *production* real-time pacing path (dyno/src/timing.py:
hybrid_sleep_until + SCHED_FIFO + gc.disable) against the real bus at 1 kHz for
a full day, with every actuator command held at zero, to prove out two things
the short bench captures can't:

  1. Logging durability  -- the capture path keeps writing correct, loadable
     files for 24 h without unbounded growth or corruption.
  2. GC / memory stability -- resident memory and Python object counts stay
     flat over 86.4 million cycles with the garbage collector disabled in the
     hot loop (leaks there are refcount bugs, and this is how we'd catch one).

Why not one big .npz
--------------------
collect_jitter.py preallocates one array per stream for the whole run and calls
np.savez_compressed once at the end. At 24 h * 1 kHz that is 86.4M cycles, i.e.
~2.4 GB of arrays held live for the entire day plus a multi-GB compression
spike at the finish -- and a crash at hour 23 loses the lot. That is the wrong
structure for a durability test.

Instead this script segments the capture: a small, fixed-size buffer
(--segment-min minutes, default 5) is filled by the real-time thread and handed
to a separate writer thread that compresses it to one seg#####.npz. Buffers are
drawn from a fixed pool and recycled, so the hot loop performs no per-cycle or
per-segment allocation, RAM stays flat regardless of total duration, each
segment is independently loadable, and a crash costs at most one segment. This
mirrors the production Logger, which likewise runs in its own thread off a queue
and writes in bounded chunks rather than buffering the whole run.

Output layout (under --out-dir, default jitter_bench/soak_<epoch>/):
  seg00000.npz, seg00001.npz, ...   per-segment captures (see arrays below)
  health.csv                        one row per segment: RSS, gc object count,
                                    gc collections, wkc errors, jitter summary
  meta.json                         run-level metadata

Each seg#####.npz holds, for the cycles in that segment:
  send_t  int64 (n,)    master send timestamp, CLOCK_MONOTONIC ns
  dc_t    int64 (n,)    slave DC hardware timestamp, ns
  wkc     int32 (n,)    working counter
  nsamp   uint8 (n,4)   ELM3004 per-channel sample counter (PAI)
  status  uint8 (n,4)   ELM3004 per-channel status byte (PAI)
  meta    json string   segment index, start/stop times, cycles, wkc errors

Usage (on the dyno host; use the venv python, which setup.sh grants
cap_net_raw + cap_sys_nice so no sudo is needed):
  DYNO_ROOT=/path/to/paladin-dyno \
    .venv/bin/python dyno/jitter_bench/soak_test.py --hours 24

  # unattended: survives the SSH session dropping, Ctrl-C / SIGTERM flush cleanly
  nohup .venv/bin/python dyno/jitter_bench/soak_test.py --hours 24 \
       > soak.out 2>&1 &

  # post-run analysis streams the segments (never loads 24 h into RAM):
  python3 dyno/jitter_bench/soak_test.py --analyze jitter_bench/soak_1737000000

Options:
  --hours 24            run length in hours (float; e.g. 0.05 for a 3-min smoke test)
  --segment-min 5       minutes of data per .npz segment
  --interface enp86s0   NIC (default read from dyno/config/master_config.yaml)
  --cycle-us 1000       cycle time (default from master_config.yaml)
  --out-dir PATH        output directory (default jitter_bench/soak_<epoch>)
  --no-rt               skip SCHED_FIFO (dry run without root; jitter will be worse)
  --analyze DIR         analysis mode: summarize an existing soak directory

The ELM3004 needs ~2.2 s to settle after entering OP; the first few seconds of
segment 0 are start-up transient and are excluded by the analysis.
"""
import argparse
import csv
import gc
import json
import os
import queue
import signal
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.environ.get('DYNO_ROOT', os.path.dirname(os.path.dirname(HERE)))

CYCLE_NS_DEFAULT = 1_000_000            # 1 kHz
TARGET_DC_MODULO = 500_000             # DC-servo phase target (matches master.py)
ELM3004_PRODUCT_CODE = 1344368073      # 0x50222e11, from dyno/src/devices.py
_PAGE = os.sysconf('SC_PAGE_SIZE') if hasattr(os, 'sysconf') else 4096


# --------------------------------------------------------------------------- #
# Segment buffer pool
# --------------------------------------------------------------------------- #
class Segment:
    """One reusable, fixed-size capture buffer. Recycled between the RT thread
    and the writer thread so the hot loop never allocates."""

    def __init__(self, n):
        self.send_t = np.zeros(n, dtype=np.int64)
        self.dc_t = np.zeros(n, dtype=np.int64)
        self.wkc = np.zeros(n, dtype=np.int32)
        self.nsamp = np.zeros((n, 4), dtype=np.uint8)
        self.status = np.zeros((n, 4), dtype=np.uint8)
        self.reset()

    def reset(self):
        self.count = 0
        self.index = -1
        self.wall_start = 0.0
        self.mono_start = 0
        self.wkc_errors = 0


def read_rss_kb():
    """Resident set size in KiB, from /proc/self/statm (no psutil dependency)."""
    try:
        with open('/proc/self/statm') as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * _PAGE // 1024
    except (OSError, IndexError, ValueError):
        return -1


# --------------------------------------------------------------------------- #
# Writer thread: compresses full segments to disk, samples process health
# --------------------------------------------------------------------------- #
class Writer(threading.Thread):
    """Consumes filled Segments off full_q, writes seg#####.npz, appends a
    health row, and returns the buffer to free_q. Runs off the RT thread so
    compression (hundreds of ms) never lands in a cycle."""

    def __init__(self, out_dir, free_q, full_q, expected_wkc, cycle_ns):
        super().__init__(name='soak-writer', daemon=True)
        self.out_dir = out_dir
        self.free_q = free_q
        self.full_q = full_q
        self.expected_wkc = expected_wkc
        self.cycle_ns = cycle_ns
        self.cycles_total = 0
        self.wkc_errors_total = 0
        self.overruns_total = 0
        self._health_path = os.path.join(out_dir, 'health.csv')
        with open(self._health_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'wall_iso', 'elapsed_s', 'segment', 'cycles_total', 'rss_kb',
                'gc_objects', 'gc_gen0', 'gc_gen1', 'gc_gen2',
                'wkc_errors_total', 'overruns_total',
                'seg_period_max_us', 'seg_period_p999_us'])
        self._t0 = time.time()

    def run(self):
        while True:
            item = self.full_q.get()
            if item is None:           # shutdown sentinel
                break
            seg, overruns = item
            self.overruns_total += overruns
            self._write_segment(seg)
            seg.reset()
            self.free_q.put(seg)

    def _write_segment(self, seg):
        n = seg.count
        if n == 0:
            return
        self.cycles_total += n
        self.wkc_errors_total += int(seg.wkc_errors)

        # per-segment jitter summary from DC timestamps (drop lost-frame gaps)
        d = np.diff(seg.dc_t[:n]).astype(np.float64) / 1000.0
        d = d[d > 0]
        if len(d):
            p_max = float(d.max())
            p_999 = float(np.percentile(d, 99.9))
        else:
            p_max = p_999 = float('nan')

        meta = dict(segment=seg.index, cycles=n,
                    wall_start=seg.wall_start,
                    wall_stop=time.time(),
                    wkc_errors=int(seg.wkc_errors),
                    expected_wkc=self.expected_wkc,
                    cycle_ns=self.cycle_ns)
        path = os.path.join(self.out_dir, f'seg{seg.index:05d}.npz')
        tmp = path + '.tmp'
        # write to a file handle so numpy doesn't re-append .npz to the name,
        # then rename atomically -- a crash mid-write can't corrupt a segment
        with open(tmp, 'wb') as fh:
            np.savez_compressed(
                fh,
                send_t=seg.send_t[:n], dc_t=seg.dc_t[:n], wkc=seg.wkc[:n],
                nsamp=seg.nsamp[:n], status=seg.status[:n],
                meta=json.dumps(meta))
        os.replace(tmp, path)

        # process-health row (all /proc + gc walks stay off the RT thread)
        c0, c1, c2 = gc.get_count()
        row = [time.strftime('%Y-%m-%dT%H:%M:%S'),
               round(time.time() - self._t0, 1), seg.index, self.cycles_total,
               read_rss_kb(), len(gc.get_objects()), c0, c1, c2,
               self.wkc_errors_total, self.overruns_total,
               round(p_max, 1), round(p_999, 1)]
        with open(self._health_path, 'a', newline='') as f:
            csv.writer(f).writerow(row)
            f.flush()
            os.fsync(f.fileno())
        print(f'  seg{seg.index:05d}: {n} cycles, rss={row[4] / 1024:.0f}MB, '
              f'objs={row[5]}, wkc_err={self.wkc_errors_total}, '
              f'p99.9={p_999:.0f}us max={p_max:.0f}us', flush=True)


# --------------------------------------------------------------------------- #
# EtherCAT bring-up (generic: any layout with an ELM3004 present)
# --------------------------------------------------------------------------- #
def elm3004_setup(slave):
    """SM-synchron SDO setup, identical to ELM3004.setup() in devices.py."""
    slave.sdo_write(0x1C12, 0, (0x0000).to_bytes(2, 'little'))
    slave.sdo_write(0x1C13, 0, (0x0000).to_bytes(2, 'little'))
    slave.sdo_write(0x1C33, 1, (0x01).to_bytes(2, 'little'))
    for ui, obj in enumerate([0x1A00, 0x1A01, 0x1A21, 0x1A22,
                              0x1A42, 0x1A43, 0x1A63, 0x1A64]):
        slave.sdo_write(0x1C13, ui + 1, obj.to_bytes(2, 'little'))
    slave.sdo_write(0x1C13, 0, (0x0008).to_bytes(2, 'little'))


def bring_up(pysoem, interface, cycle_ns):
    master = pysoem.Master()
    master.open(interface)
    n = master.config_init()
    if n < 1:
        master.close()
        raise RuntimeError(f'no EtherCAT slaves found on {interface} -- '
                           'is the bus powered and the NIC correct?')
    print('slaves:', [s.name for s in master.slaves], flush=True)

    master.state = pysoem.PREOP_STATE
    master.write_state()
    master.state_check(pysoem.PREOP_STATE, timeout=500_000)

    elm_index = None
    for i, s in enumerate(master.slaves):
        if s.id == ELM3004_PRODUCT_CODE:
            elm3004_setup(s)
            elm_index = i
    if elm_index is None:
        print('WARNING: no ELM3004 on the bus; nsamp/status will stay zero',
              flush=True)

    master.config_map()
    master.config_dc()
    if master.state_check(pysoem.SAFEOP_STATE, timeout=2_000_000) != pysoem.SAFEOP_STATE:
        master.close()
        raise RuntimeError('failed to reach SAFEOP')

    # Hold every output at zero for the whole run ("all commands set to zero").
    for s in master.slaves:
        if len(s.output):
            s.output = bytes(len(s.output))

    master.state = pysoem.OP_STATE
    master.send_processdata()
    master.receive_processdata(timeout=2000)
    master.write_state()
    for _ in range(200):
        master.send_processdata()
        master.receive_processdata(timeout=2000)
        if master.state_check(pysoem.OP_STATE, timeout=50_000) == pysoem.OP_STATE:
            break
    else:
        master.close()
        raise RuntimeError('failed to reach OP')
    print(f'OP reached, expected_wkc={master.expected_wkc}', flush=True)
    return master, elm_index


# --------------------------------------------------------------------------- #
# Capture (real-time acquisition loop)
# --------------------------------------------------------------------------- #
def capture(args):
    import pysoem  # noqa: local import so --analyze works off the dyno
    sys.path.insert(0, REPO_ROOT)
    from dyno.src.timing import hybrid_sleep_until, set_cyclic_thread_priority

    cycle_ns = args.cycle_us * 1000
    seg_cycles = int(args.segment_min * 60 * 1e9 / cycle_ns)
    total_cycles = int(args.hours * 3600 * 1e9 / cycle_ns)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f'soak: {args.hours} h, {total_cycles} cycles, '
          f'{seg_cycles} cycles/segment, out={args.out_dir}', flush=True)

    master, elm_index = bring_up(pysoem, args.interface, cycle_ns)
    expected_wkc = master.expected_wkc
    elm = master.slaves[elm_index] if elm_index is not None else None

    # buffer pool: 3 segments is enough for the writer to stay ahead
    free_q = queue.Queue()
    full_q = queue.Queue()
    for _ in range(3):
        free_q.put(Segment(seg_cycles))
    writer = Writer(args.out_dir, free_q, full_q, expected_wkc, cycle_ns)
    writer.start()

    # graceful stop for a 24 h unattended run: flush the partial segment
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    # production real-time regime: SCHED_FIFO + GC off in the hot loop
    if args.no_rt:
        print('WARNING: --no-rt, running at normal priority', flush=True)
    elif set_cyclic_thread_priority():
        print('SCHED_FIFO priority set', flush=True)
    else:
        print('WARNING: could not set SCHED_FIFO (need root); jitter will be '
              'worse under load', flush=True)
    gc_was_enabled = gc.isenabled()
    gc.disable()

    monotonic_ns = time.clock_gettime_ns
    seg = free_q.get()
    seg.index = 0
    seg.wall_start = time.time()
    seg.mono_start = monotonic_ns(time.CLOCK_MONOTONIC)
    seg_idx = 0
    overruns = 0

    master_time_offset = None
    cycle_start = monotonic_ns(time.CLOCK_MONOTONIC)
    total = 0
    try:
        while total < total_cycles and not stop.is_set():
            t0 = monotonic_ns(time.CLOCK_MONOTONIC)
            master.send_processdata()
            wkc = master.receive_processdata(timeout=int(cycle_ns / 1000 * 0.9))
            dc = master.dc_time

            i = seg.count
            seg.send_t[i] = t0
            seg.dc_t[i] = dc
            seg.wkc[i] = wkc
            if elm is not None:
                buf = bytes(elm.input)   # per ch: nsamp u8, status u8, pad u16, val i32
                for ch in range(4):
                    seg.nsamp[i, ch] = buf[ch * 8]
                    seg.status[i, ch] = buf[ch * 8 + 1]
            if wkc < expected_wkc:
                seg.wkc_errors += 1
            seg.count += 1
            total += 1

            # DC drift servo -- identical algorithm to master.py
            dc_modulo = dc % cycle_ns
            if master_time_offset is None:
                master_time_offset = dc - t0
            shift = int(((dc - t0) - master_time_offset) / 30) \
                + int((dc_modulo - TARGET_DC_MODULO) / 30)
            master_time_offset += shift
            cycle_start += cycle_ns - shift

            # hand off a full segment; grab a fresh buffer without blocking if
            # one is free, otherwise count the stall and wait (writer fell behind)
            if seg.count >= seg_cycles:
                full_q.put((seg, overruns))
                overruns = 0
                try:
                    seg = free_q.get_nowait()
                except queue.Empty:
                    overruns += 1
                    seg = free_q.get()
                seg_idx += 1
                seg.index = seg_idx
                seg.wall_start = time.time()
                seg.mono_start = monotonic_ns(time.CLOCK_MONOTONIC)

            hybrid_sleep_until(cycle_start)
    finally:
        if gc_was_enabled:
            gc.enable()
        # flush the final partial segment, then drain the writer
        if seg.count > 0:
            full_q.put((seg, overruns))
        full_q.put(None)
        writer.join()
        master.state = pysoem.INIT_STATE
        master.write_state()
        master.close()

    meta = dict(hours=args.hours, cycle_ns=cycle_ns, interface=args.interface,
                segment_min=args.segment_min, seg_cycles=seg_cycles,
                requested_cycles=total_cycles, captured_cycles=total,
                segments=seg_idx + 1, expected_wkc=expected_wkc,
                wkc_errors_total=writer.wkc_errors_total,
                overruns_total=writer.overruns_total,
                stopped_early=bool(stop.is_set()),
                started=time.strftime('%Y-%m-%d %H:%M:%S',
                                      time.localtime(writer._t0)),
                finished=time.strftime('%Y-%m-%d %H:%M:%S'),
                python=sys.version.split()[0], numpy=np.__version__)
    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'\nDone: {total} cycles in {seg_idx + 1} segments -> {args.out_dir}',
          flush=True)
    print(f'  wkc errors: {writer.wkc_errors_total}, '
          f'writer overruns: {writer.overruns_total}', flush=True)
    analyze(argparse.Namespace(analyze=args.out_dir))


# --------------------------------------------------------------------------- #
# Analysis (streams segments; never holds the whole run in RAM)
# --------------------------------------------------------------------------- #
def analyze(args):
    out_dir = args.analyze
    segs = sorted(f for f in os.listdir(out_dir)
                  if f.startswith('seg') and f.endswith('.npz'))
    if not segs:
        print(f'no segments in {out_dir}')
        return

    # stream period stats with a running histogram (constant memory)
    edges = np.arange(0, 5000.5, 1.0)          # 0..5 ms in 1 us bins
    hist = np.zeros(len(edges) - 1, dtype=np.int64)
    n_cycles = wkc_bad = 0
    dup = skip = invalid = 0
    p_max = 0.0
    for name in segs:
        z = np.load(os.path.join(out_dir, name), allow_pickle=True)
        dc, wkc, status = z['dc_t'], z['wkc'], z['status']
        seg_meta = json.loads(str(z['meta']))
        expected_wkc = seg_meta.get('expected_wkc', int(wkc.max(initial=0)))
        n_cycles += len(wkc)
        wkc_bad += int((wkc < expected_wkc).sum())
        d = np.diff(dc).astype(np.float64) / 1000.0
        d = d[d > 0]
        if len(d):
            hist += np.histogram(d, bins=edges)[0]
            p_max = max(p_max, float(d.max()))
        # ADC sync health on channel 0's sample counter
        cc = (status[:, 0].astype(np.int16) >> 6) & 3
        dcc = (cc[1:] - cc[:-1]) % 4
        dup += int((dcc == 0).sum())
        skip += int((dcc >= 2).sum())
        invalid += int(((status[:, 0] >> 5) & 1).sum())

    total = hist.sum()
    cum = np.cumsum(hist)
    centers = (edges[:-1] + edges[1:]) / 2

    def pct(p):
        if total == 0:
            return float('nan')
        return float(centers[np.searchsorted(cum, p / 100.0 * total)])

    print(f'\n=== soak analysis: {out_dir} ===')
    print(f'segments={len(segs)} cycles={n_cycles:,} '
          f'(~{n_cycles / 3.6e6:.1f} h at 1 kHz)')
    print(f'DC cycle period us: p50={pct(50):.1f} p99={pct(99):.1f} '
          f'p99.9={pct(99.9):.1f} max={p_max:.1f}')
    print(f'wkc errors: {wkc_bad:,} / {n_cycles:,}')
    print(f'ADC ch1: duplications={dup} dropouts={skip} txpdo_invalid={invalid}')

    hpath = os.path.join(out_dir, 'health.csv')
    if os.path.exists(hpath):
        rows = list(csv.DictReader(open(hpath)))
        if rows:
            r0, r1 = rows[0], rows[-1]
            drss = int(r1['rss_kb']) - int(r0['rss_kb'])
            dobj = int(r1['gc_objects']) - int(r0['gc_objects'])
            print(f'RSS: {int(r0["rss_kb"]) // 1024} MB -> '
                  f'{int(r1["rss_kb"]) // 1024} MB (drift {drss / 1024:+.1f} MB)')
            print(f'gc objects: {r0["gc_objects"]} -> {r1["gc_objects"]} '
                  f'({dobj:+d})')
            print('  -> leak check: RSS/object drift should be ~flat over the run')


def default_interface_and_cycle():
    """Read NIC + cycle time from dyno/config/master_config.yaml, matching the
    real master. Falls back to enp86s0 / 1000 us if unavailable."""
    interface, cycle_us = 'enp86s0', CYCLE_NS_DEFAULT // 1000
    try:
        import yaml
        with open(os.path.join(REPO_ROOT, 'dyno', 'config',
                               'master_config.yaml')) as f:
            cfg = yaml.safe_load(f)
        interface = cfg.get('interface', interface)
        cycle_us = int(cfg.get('cycle_time_us', cycle_us))
    except Exception:
        pass
    return interface, cycle_us


if __name__ == '__main__':
    iface, cyc = default_interface_and_cycle()
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--hours', type=float, default=24.0)
    p.add_argument('--segment-min', type=float, default=5.0,
                   help='minutes of data per .npz segment (default 5)')
    p.add_argument('--interface', default=iface,
                   help=f'EtherCAT NIC (default from master_config.yaml: {iface})')
    p.add_argument('--cycle-us', type=int, default=cyc)
    p.add_argument('--out-dir', default=None)
    p.add_argument('--no-rt', action='store_true',
                   help='skip SCHED_FIFO (dry run without root)')
    p.add_argument('--analyze', metavar='DIR',
                   help='analyze an existing soak directory instead of capturing')
    args = p.parse_args()

    if args.analyze:
        analyze(args)
    else:
        if args.out_dir is None:
            args.out_dir = os.path.join(HERE, f'soak_{int(time.time())}')
        if args.segment_min <= 0:
            p.error('--segment-min must be > 0')
        capture(args)
