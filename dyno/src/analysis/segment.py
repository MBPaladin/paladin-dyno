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

    def prefix(self, device_name):
        """Config device name -> log channel prefix ('LOAD' -> 'load')."""
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
