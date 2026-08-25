"""Segment: a read-only view onto one contiguous span of one log.

All HDF5 access in the framework goes through here. Processors never open a
file, which is what makes them testable against synthetic arrays.

Behavior IDs encode a three-level hierarchy that the runner uses for grouping:

    SEG1-RUN0                 test_trace   -> one span per execution
    SEG1-RUN0-SETPOINT7       grid_search  -> one span per grid setpoint
                                              (already trimmed to the settled
                                              dwell; see test_manager.py:374)

Behavior names may themselves contain hyphens (CSV-BACKLASH-HYSTERESIS), so the
suffixes are stripped from the right rather than split on the first hyphen.
"""

import json
import os
import re

import h5py
import numpy as np

_SETPOINT_RE = re.compile(r'-SETPOINT(\d+)$')
_RUN_RE = re.compile(r'-RUN(\d+)$')


def parse_behavior_id(raw):
    """'SEG1-RUN0-SETPOINT7' -> ('SEG1', 0, 7). Missing parts come back None."""
    setpoint = run = None
    m = _SETPOINT_RE.search(raw)
    if m:
        setpoint = int(m.group(1))
        raw = raw[:m.start()]
    m = _RUN_RE.search(raw)
    if m:
        run = int(m.group(1))
        raw = raw[:m.start()]
    return raw, run, setpoint


class Segment:
    """One span. Channel reads are sliced, float64, and cached."""

    def __init__(self, f, log_dir, raw_id, start, end, config, tare):
        self._f = f
        self._cache = {}
        self.log_dir = log_dir
        self.raw_id = raw_id
        self.behavior, self.run, self.setpoint = parse_behavior_id(raw_id)
        self.start = int(start)
        self.end = int(end)
        self.config = config
        self.tare = tare

    # -- identity ----------------------------------------------------------
    def __len__(self):
        return self.end - self.start

    def __repr__(self):
        return f'<Segment {self.raw_id} [{self.start}:{self.end}] n={len(self)}>'

    @property
    def group_key(self):
        """(behavior, run) -- the grouping key for granularity='run'."""
        return (self.behavior, self.run)

    # -- channels ----------------------------------------------------------
    def channels(self):
        return [k for k in self._f.keys()
                if isinstance(self._f[k], h5py.Dataset) and self._f[k].ndim == 1
                and not k.startswith('behavior')]

    def has(self, channel):
        return channel in self._f

    def __getitem__(self, channel):
        if channel not in self._cache:
            self._cache[channel] = np.asarray(
                self._f[channel][self.start:self.end], dtype=float)
        return self._cache[channel]

    def is_active(self, channel):
        """A command channel is NaN whenever that mode is not active, so
        'fully NaN' is a reliable data-driven test for 'this mode was never
        commanded'. Cheaper and more honest than reasoning from config."""
        if not self.has(channel):
            return False
        return not np.all(np.isnan(self[channel]))

    @property
    def dt(self):
        t = self['time']
        return float(np.median(np.diff(t))) if len(t) > 1 else float('nan')

    @property
    def duration(self):
        t = self['time']
        return float(t[-1] - t[0]) if len(t) > 1 else 0.0

    # -- config helpers ----------------------------------------------------
    def devices(self):
        return self.config.get('devices', {})

    def device_params(self, name):
        return self.devices().get(name, {}).get('params', {})

    # A bench's `ports:` block (carried into resolved_config.json) maps each
    # mechanical port to: `role` (what commands and the UI target), `prefix`
    # (what this bench's log channels are called), `device` (which EtherCAT
    # slave), `attached` (free text for figure titles -- nothing branches on
    # it), and optionally `cell` (the torque sensor on that shaft).
    def ports(self):
        return self.config.get('ports') or []

    def port_by_role(self, role):
        for port in self.ports():
            if port.get('role') == role:
                return port
        return None

    def attached_label(self, role):
        """Human label for what is bolted to a port, for figure titles."""
        port = self.port_by_role(role)
        attached = (port or {}).get('attached')
        return str(attached) if attached and attached != 'none' else ''

    def current_units(self, device_name):
        """How this drive reports current: ('peak'|'rms', declared?).

        A bench can carry drives that disagree -- the Elmo reports the
        quadrature-axis amplitude, the AKDs report A_rms -- and nothing in the
        analysis path converts between them. This is therefore a *labelling*
        fact: it decides which unit a report prints beside a kt, and nothing
        else reads it. Unifying the convention means touching every processor's
        current handling, which is a separate job from saying which one a given
        number is in.

        The second element says whether the config declared it or whether the
        rms default was assumed, so a caller can warn once rather than printing
        a confident unit it guessed.
        """
        declared = (self.device_params(device_name)
                    .get('drive_params', {}).get('current_units'))
        if declared is None:
            return 'rms', False
        return str(declared).strip().lower(), True

    def command_channel(self, kind, prefer=None):
        """The active `<prefix>_<kind>_command` channel on this span, or None.

        Guessing the command channel from the channel a processor measured is
        wrong whenever the two shafts differ, which on a two-motor bench is the
        normal case rather than the exception: the inertia test measures
        `dut_velocity` while the absorber is the motor actually holding the
        velocity command, so `dut_velocity_command` is all-NaN and the
        commanded speed appears not to exist.

        `prefer` is a channel prefix to try first, so a processor that does know
        which port it means still gets that one.
        """
        prefixes = [prefer] if prefer else []
        prefixes += [p.get('prefix') for p in self.ports() if p.get('prefix')]
        for prefix in prefixes:
            channel = f'{prefix}_{kind}_command'
            if self.is_active(channel):
                return channel
        return None

    def cmd_span(self, channel, levels_cap=16):
        """What was actually commanded on `channel` over this span.

        The test YAML is not carried in the log -- resolved_config holds the
        bench, not the trace -- so the commanded magnitudes a report quotes
        ("a torque sawtooth of +/-5 Nm") have to come from the command channel
        itself. That is the better source anyway: it is what went out on the
        wire after limit clipping, not what someone asked for.

        Returns None when the mode was never commanded (a command channel is
        all-NaN whenever its mode is inactive). `max_rate` is the 95th
        percentile of |d/dt|, not the maximum: the maximum is a single
        differentiated sample and lands on whatever noise the channel has.
        """
        if not self.is_active(channel):
            return None
        v = self[channel]
        finite = np.isfinite(v)
        if not finite.any():
            return None
        vals = v[finite]
        levels = np.unique(np.round(vals, 6))
        rate, reversals = None, None
        t = self['time'][finite]
        if len(vals) > 2:
            dv, dt = np.diff(vals), np.diff(t)
            moving = dt > 0
            if moving.any():
                slope = np.abs(dv[moving] / dt[moving])
                if slope.size:
                    rate = float(np.percentile(slope, 95))
            # Direction reversals of the command, which is how many times a
            # sawtooth turned around. Counted on a sign sequence with the flat
            # samples removed: a dwell at the peak is a run of zeros, and
            # counting sign changes through it would score one reversal per
            # sample of dwell.
            sign = np.sign(dv[np.abs(dv) > 1e-12])
            if sign.size > 1:
                reversals = int(np.count_nonzero(np.diff(sign)))
        return {
            'min': float(vals.min()),
            'max': float(vals.max()),
            'amplitude': float(max(abs(vals.min()), abs(vals.max()))),
            'n_levels': int(levels.size),
            # A sawtooth visits thousands of distinct values; only a genuine
            # setpoint grid is worth listing, so a long list is dropped rather
            # than truncated into something that looks like a complete grid.
            'levels': ([float(x) for x in levels]
                       if levels.size <= levels_cap else None),
            'max_rate': rate,
            'n_reversals': reversals,
        }

    def prefix(self, device_name):
        """Config device name -> log channel prefix.

        Read from the bench's `ports:` block when one exists. The lowercase
        fallback is only correct on benches whose prefixes happen to equal
        their device names (`DUT`/`LOAD` -> `dut_*`/`load_*`); the gearbox
        bench logs `input_*`/`output_*` and silently matched nothing under it.
        """
        for port in self.ports():
            if port.get('device') == device_name and port.get('prefix'):
                return str(port['prefix'])
        return device_name.lower()


def find_log_file(log_dir):
    """Return the single .hdf5 in log_dir. Accepts a parent folder and resolves
    to the most recent run beneath it, matching post_processor.py's behavior."""
    log_dir = os.path.abspath(log_dir.rstrip('/'))
    if not os.path.isdir(log_dir):
        raise NotADirectoryError(log_dir)

    here = sorted(f for f in os.listdir(log_dir) if f.endswith('.hdf5'))
    if len(here) == 1:
        return log_dir, os.path.join(log_dir, here[0])
    if len(here) > 1:
        raise ValueError(
            f'{log_dir} holds {len(here)} .hdf5 files; point at one run folder: '
            + ', '.join(here))

    candidates = []
    for d in os.listdir(log_dir):
        sub = os.path.join(log_dir, d)
        if os.path.isdir(sub) and any(x.endswith('.hdf5') for x in os.listdir(sub)):
            candidates.append(sub)
    if not candidates:
        raise FileNotFoundError(f'No .hdf5 found in {log_dir} or its subfolders')
    newest = max(candidates, key=os.path.getmtime)
    print(f'Resolved to most recent run: {newest}')
    return find_log_file(newest)


def _load_json_attr(f, key):
    raw = f.attrs.get(key)
    if raw is None:
        return {}
    try:
        return json.loads(raw)
    except (ValueError, TypeError):
        print(f'  WARNING: could not parse {key!r} attribute as JSON')
        return {}


class Log:
    """Open log + its segments. Use as a context manager."""

    def __init__(self, log_dir, path):
        self.log_dir = log_dir
        self.path = path
        self._f = h5py.File(path, 'r')
        self.config = _load_json_attr(self._f, 'resolved_config')
        self.tare = _load_json_attr(self._f, 'tare')
        self.segments = self._build_segments()

    def _build_segments(self):
        segs = []
        ids = self._f['behavior_ids'][:]
        idx = self._f['behavior_indices'][:]
        for i, raw in enumerate(ids):
            raw = raw.decode('utf-8') if isinstance(raw, bytes) else str(raw)
            if not raw:
                continue
            start, end = int(idx[i, 0]), int(idx[i, 1])
            if end <= start:
                continue
            segs.append(Segment(self._f, self.log_dir, raw, start, end,
                                self.config, self.tare))
        return segs

    def grouped(self, granularity):
        """Group segments per the processor's declared granularity. Returns a
        list of (label, [Segment]) preserving log order."""
        if granularity == 'setpoint':
            return [(s.raw_id, [s]) for s in self.segments]
        if granularity == 'log':
            # One group, every span. The label becomes an output filename
            # prefix, so it is a fixed word rather than a joined list of
            # behavior names -- those run to dozens of characters on a real
            # test and would make every file unreadable.
            return [('ALL', list(self.segments))] if self.segments else []

        keyfn = (lambda s: s.group_key) if granularity == 'run' else (lambda s: s.behavior)
        order, buckets = [], {}
        for s in self.segments:
            k = keyfn(s)
            if k not in buckets:
                buckets[k] = []
                order.append(k)
            buckets[k].append(s)

        out = []
        for k in order:
            members = buckets[k]
            label = members[0].raw_id if len(members) == 1 else (
                f'{members[0].behavior}-RUN{members[0].run}'
                if granularity == 'run' else members[0].behavior)
            out.append((label, members))
        return out

    def close(self):
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def open_log(log_dir):
    resolved_dir, path = find_log_file(log_dir)
    return Log(resolved_dir, path)
