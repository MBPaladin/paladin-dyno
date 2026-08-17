"""Collect EtherCAT 1 kHz jitter / ADC-sync benchmark data on the bench rig
(EK1100 + ELM3004 on the ASIX USB adapter, Windows).

Runs two pacing methods and writes one .npz per method, consumed by
plot_jitter.py in this directory:

  bad   -> baseline.npz  old production-style pacing: OS sleep, normal
                         priority, garbage collector on, 900 us rx timeout
  fixed -> fixed.npz     patched pacing: absolute-deadline busy-wait, HIGH
                         priority class, GC off, 5 ms rx timeout

Every run logs, per cycle: master send timestamp, slave DC hardware
timestamp, working counter, and the ELM3004 per-channel PAI status bytes
(TxPDO State / sample counters) so one capture serves both the jitter
figure and the ADC-sync figure.

Usage (Windows anaconda python has all dependencies):
  "C:\\Users\\Nathan Justus\\anaconda3\\python.exe" collect_jitter.py --duration 300
  ... --method bad         run only one method
  ... --out-prefix trial2_ write trial2_baseline.npz etc.

The ELM3004 needs ~2.2 s to settle after entering OP; keep duration >= 10 s.
Requires the bench segment powered and plugged into the ASIX adapter, and
nothing else (e.g. Wireshark on that NIC is fine; a second master is not).
"""
import argparse
import ctypes
import gc
import json
import os
import sys
import time

import numpy as np
import pysoem

HERE = os.path.dirname(os.path.abspath(__file__))
ADAPTER = r'\Device\NPF_{64AEA1B5-FD73-40C8-995D-48C4FF29D07F}'  # ASIX USB GbE
CYCLE_NS = 1_000_000  # 1 kHz
TARGET_DC_MODULO = 500_000  # DC-servo phase target (matches dyno master.py)

kernel32 = ctypes.windll.kernel32
CUR_PROC = ctypes.c_void_p(-1)
CUR_THREAD = ctypes.c_void_p(-2)

METHODS = {
    'bad': dict(outfile='baseline.npz', rx_timeout_us=900),
    'fixed': dict(outfile='fixed.npz', rx_timeout_us=5000),
}


def elm3004_setup(slave):
    """Same SDO setup as ELM3004.setup() in dyno/src/devices.py (SM-synchron)."""
    slave.sdo_write(0x1C12, 0, (0x0000).to_bytes(2, 'little'))
    slave.sdo_write(0x1C13, 0, (0x0000).to_bytes(2, 'little'))
    slave.sdo_write(0x1C33, 1, (0x01).to_bytes(2, 'little'))
    for ui, obj in enumerate([0x1A00, 0x1A01, 0x1A21, 0x1A22,
                              0x1A42, 0x1A43, 0x1A63, 0x1A64]):
        slave.sdo_write(0x1C13, ui + 1, obj.to_bytes(2, 'little'))
    slave.sdo_write(0x1C13, 0, (0x0008).to_bytes(2, 'little'))


def bring_up():
    master = pysoem.Master()
    master.open(ADAPTER)
    n = master.config_init()
    if n < 2:
        master.close()
        raise RuntimeError(f'expected 2 slaves on {ADAPTER}, found {n} — '
                           'is the bench rig powered and plugged in?')
    print('slaves:', [s.name for s in master.slaves])
    master.state = pysoem.PREOP_STATE
    master.write_state()
    master.state_check(pysoem.PREOP_STATE, timeout=500_000)
    elm3004_setup(master.slaves[1])
    master.config_map()
    master.config_dc()
    if master.state_check(pysoem.SAFEOP_STATE, timeout=2_000_000) != pysoem.SAFEOP_STATE:
        raise RuntimeError('failed to reach SAFEOP')
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
        raise RuntimeError('failed to reach OP')
    print(f'OP reached, expected_wkc={master.expected_wkc}')
    return master


def run_method(method, duration_s, out_path):
    cfg = METHODS[method]
    n = int(duration_s * 1000)
    send_t = np.zeros(n, dtype=np.int64)
    dc_t = np.zeros(n, dtype=np.int64)
    wkc_arr = np.zeros(n, dtype=np.int32)
    nsamp = np.zeros((n, 4), dtype=np.uint8)
    status = np.zeros((n, 4), dtype=np.uint8)

    master = bring_up()
    expected_wkc = master.expected_wkc
    elm = master.slaves[1]
    rx_timeout = cfg['rx_timeout_us']

    if method == 'fixed':
        kernel32.SetPriorityClass(CUR_PROC, ctypes.c_uint32(0x0080))  # HIGH
        kernel32.SetThreadPriority(CUR_THREAD, ctypes.c_int(15))  # TIME_CRITICAL
        gc.disable()
    print(f'method={method} duration={duration_s:.0f}s gc={gc.isenabled()} '
          f'rx_timeout={rx_timeout}us')

    perf = time.perf_counter_ns
    servo_offset = None
    cycle_start = perf()
    i = 0
    t_report = time.time()
    try:
        while i < n:
            t0 = perf()
            master.send_processdata()
            wkc = master.receive_processdata(timeout=rx_timeout)
            dc = master.dc_time

            buf = bytes(elm.input)  # per ch: nsamp u8, status u8, pad u16, val i32
            send_t[i] = t0
            dc_t[i] = dc
            wkc_arr[i] = wkc
            for ch in range(4):
                nsamp[i, ch] = buf[ch * 8]
                status[i, ch] = buf[ch * 8 + 1]

            # DC drift servo (same algorithm as dyno master.py)
            dc_modulo = dc % CYCLE_NS
            if servo_offset is None:
                servo_offset = dc - t0
            shift = int(((dc - t0) - servo_offset) / 30) \
                + int((dc_modulo - TARGET_DC_MODULO) / 30)
            servo_offset += shift
            cycle_start += CYCLE_NS - shift
            i += 1

            if i % 60000 == 0:
                print(f'  ...{i // 1000}s of {duration_s:.0f}s '
                      f'({time.time() - t_report:.0f}s wall)')

            if method == 'bad':
                remaining = cycle_start - perf()
                if remaining > 0:
                    time.sleep(remaining / 1e9)
            else:
                while perf() < cycle_start:
                    pass
    finally:
        gc.enable()
        master.state = pysoem.INIT_STATE
        master.write_state()
        master.close()

    meta = dict(method=method, duration=duration_s, rx_timeout_us=rx_timeout,
                cycles=i, expected_wkc=expected_wkc, adapter=ADAPTER,
                collected=time.strftime('%Y-%m-%d %H:%M:%S'),
                python=sys.version.split()[0])
    np.savez_compressed(out_path, send_t=send_t[:i], dc_t=dc_t[:i],
                        wkc=wkc_arr[:i], nsamp=nsamp[:i], status=status[:i],
                        meta=json.dumps(meta))
    print(f'saved {i} cycles -> {out_path}')
    summarize(send_t[:i], dc_t[:i], wkc_arr[:i], status[:i], expected_wkc)


def summarize(send_t, dc_t, wkc, status, expected_wkc):
    for label, src in [('send', send_t), ('dc', dc_t)]:
        d = np.diff(src).astype(np.float64) / 1000.0
        d = d[d > 0]
        print(f'  {label}-period us: mean={d.mean():.1f} std={d.std():.1f} '
              f'p99={np.percentile(d, 99):.1f} p99.9={np.percentile(d, 99.9):.1f} '
              f'max={d.max():.1f}')
    print(f'  bad wkc: {int((wkc < expected_wkc).sum())} / {len(wkc)}')

    # ADC sync health (skip ELM3004 settling window)
    inv0 = (status[:, 0] >> 5) & 1
    settled = np.where(inv0 == 0)[0]
    start = int(settled[0]) + 100 if len(settled) else 0
    ok = wkc >= expected_wkc
    ok[:start] = False
    idx = np.where(ok)[0]
    for ch in range(4):
        cc = (status[idx, ch].astype(np.int16) >> 6) & 3
        dcc = (cc[1:] - cc[:-1]) % 4
        dup, skip = int((dcc == 0).sum()), int((dcc >= 2).sum())
        invalid = int(((status[idx, ch] >> 5) & 1).sum())
        if ch == 0 or dup or skip or invalid:
            print(f'  ch{ch + 1}: duplications={dup} dropouts={skip} '
                  f'txpdo_invalid={invalid}  (of {len(idx)} analyzed cycles)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=60.0,
                   help='seconds per method (e.g. 300 for 5-minute runs)')
    p.add_argument('--method', choices=['bad', 'fixed', 'both'], default='both')
    p.add_argument('--out-prefix', default='',
                   help='prefix for output filenames (e.g. trial2_)')
    args = p.parse_args()
    if args.duration < 10:
        p.error('duration must be >= 10 s (ELM3004 needs ~2.2 s to settle)')

    methods = ['bad', 'fixed'] if args.method == 'both' else [args.method]
    for m in methods:
        out = os.path.join(HERE, args.out_prefix + METHODS[m]['outfile'])
        print(f'\n=== {m} -> {os.path.basename(out)} ===')
        run_method(m, args.duration, out)
    print('\nDone. Regenerate figures with plot_jitter.py')
