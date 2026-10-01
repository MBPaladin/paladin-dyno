"""Fit the throw stop decel from logged velocity-0 stops.

    PYTHONPATH=. .venv/bin/python -m dyno.src.archimedes.fit_stop_decel <log dir or .hdf5> [...]
    ... --csv stops.csv         also write every stop as a row

Works on the raw logs of the `archimedes_stop_decel_*` plans (see
dyno/tests/traces/generate_archimedes_stop_decel_tests.py), and on any log with a
velocity command stepped to zero from a moving shaft -- `velocity_mode_6` is one.

A STOP is a sample where the drive motor's velocity command goes from non-zero to
exactly zero while the OUTPUT shaft is moving. Everything is read from the
output encoder, whichever motor drove, so the result is in the coordinates the
throw's lookahead uses:

    v0      output velocity at the first zero-command sample
    travel  output angle covered from there to the first velocity zero-crossing
    a_eff   v0^2 / (2 * travel)

Reported per drive motor and hold-torque condition:

    a (one-param)   least squares of travel = v0^2 / (2a)
    t_d, a          least squares of travel = v0*t_d + v0^2 / (2a): the lag term
                    separated out (shown when the data support it)
    a_safe          the lowest a_eff among stops at >= half the top speed. A
                    lookahead using this is at least as conservative as every
                    observed fast stop.
    t_lag (safe)    travel = v0 * t_lag: the stop as a pure lag, for a drive whose
                    travel grows LINEARLY with speed so that a_eff climbs with
                    it (the output drive under a PI velocity loop). The safe
                    figure is the largest travel / v0 among stops at >= half the
                    top speed. Use it as `throw.stop_lag_s.<drive>` when a_eff
                    rises several-fold across the ladder; it replaces the decel.

WHICH NUMBER TO USE. Size from the worst hold condition, not the zero-torque
one: assisting torque lengthens the stop. Take a_safe of the worst group, round
DOWN, and put it under `throw.stop_decel_rad_s2.<drive>` in the rig config.
The hold-torque sign shown is hold command x throw direction, so a group is
'+' or '-' relative to the motion; whichever has the smaller decel is the
assisting one.
"""
import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

MIN_V0 = 0.2            # rad/s at the output: slower than this is not a stop worth fitting
MAX_STOP_S = 3.0        # give up looking for the zero crossing after this
HOLD_ZERO_NM = 1e-3


def _read(f, key):
    if key not in f:
        return None
    return np.nan_to_num(f[key][:].astype(float), nan=0.0)


def find_logs(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, '**', '*.hdf5'), recursive=True))
        else:
            out.append(p)
    return out


def stops_in(path):
    """Every velocity-0 stop in one log, as a list of dicts."""
    import h5py
    with h5py.File(path, 'r') as f:
        t = f['time'][:].astype(float)
        pos, vel = f['load_position'][:].astype(float), f['load_velocity'][:].astype(float)
        cmds = {'input': _read(f, 'dut_velocity_command'),
                'output': _read(f, 'load_velocity_command')}
        holds = {'input': _read(f, 'load_torque_command'),     # input drives, output holds
                 'output': _read(f, 'dut_torque_command')}     # output drives, input holds
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1e-3
    rows = []
    for drive, cmd in cmds.items():
        if cmd is None:
            continue
        fell = np.where((np.abs(cmd[:-1]) > 1e-9) & (np.abs(cmd[1:]) < 1e-9))[0] + 1
        for i0 in fell:
            v0 = vel[i0]
            if abs(v0) < MIN_V0 or np.isnan(v0):
                continue
            d = np.sign(v0)
            n = int(MAX_STOP_S / dt)
            seg = d * vel[i0:i0 + n]
            crossed = np.where(seg <= 0)[0]
            if len(crossed) == 0:
                continue
            i1 = i0 + int(crossed[0])
            travel = d * (pos[i1] - pos[i0])
            if travel <= 0:
                continue
            hold = holds[drive][i0] if holds[drive] is not None else 0.0
            rows.append({'log': os.path.basename(os.path.dirname(path)) or path,
                         'drive': drive, 't': t[i0], 'v0': abs(v0), 'dir': int(d),
                         'travel': travel, 'stop_s': (i1 - i0) * dt,
                         'a_eff': v0 * v0 / (2.0 * travel),
                         'hold_nm': hold, 'hold_signed': hold * d})
    return rows


def hold_class(row):
    if abs(row['hold_signed']) < HOLD_ZERO_NM:
        return 'hold 0'
    return 'hold %s' % ('+' if row['hold_signed'] > 0 else '-')


def fit(rows):
    v = np.array([r['v0'] for r in rows])
    d = np.array([r['travel'] for r in rows])
    out = {'n': len(rows), 'vmin': v.min(), 'vmax': v.max()}
    out['a'] = 1.0 / (2.0 * ((d * v * v).sum() / (v ** 4).sum()))
    fast = v >= 0.5 * v.max()
    out['a_safe'] = float((v[fast] ** 2 / (2.0 * d[fast])).min())
    out['t_lag'] = float((d * v).sum() / (v * v).sum())
    out['t_lag_safe'] = float((d[fast] / v[fast]).max())
    out['t_d'] = out['a_lag'] = None
    if len(rows) >= 4 and len(set(np.round(v, 1))) >= 3:
        (td, k), *_ = np.linalg.lstsq(np.column_stack([v, v * v]), d, rcond=None)
        if k > 0 and td >= 0:
            out['t_d'], out['a_lag'] = float(td), float(1.0 / (2.0 * k))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('paths', nargs='+', help='log directories or .hdf5 files')
    ap.add_argument('--csv', help='write every stop to this csv')
    args = ap.parse_args(argv)

    rows = []
    for path in find_logs(args.paths):
        found = stops_in(path)
        print(f'{path}: {len(found)} stop(s)')
        rows += found
    if not rows:
        sys.exit('no velocity-0 stops found')

    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    print(f"\n{'drive':7s} {'condition':9s} {'n':>3s} {'v range':>13s} {'a':>7s} "
          f"{'t_d [ms]':>9s} {'a (lag)':>8s} {'a_safe':>7s} {'t_lag [ms]':>10s} "
          f"{'safe':>6s}")
    worst, worst_lag = {}, {}
    for drive in ('output', 'input'):
        mine = [r for r in rows if r['drive'] == drive]
        for cond in sorted({hold_class(r) for r in mine}):
            grp = [r for r in mine if hold_class(r) == cond]
            r = fit(grp)
            lag = '' if r['t_d'] is None else f"{r['t_d'] * 1e3:9.1f} {r['a_lag']:8.1f}"
            print(f"{drive:7s} {cond:9s} {r['n']:3d} {r['vmin']:5.2f}-{r['vmax']:5.2f} "
                  f"{r['a']:7.1f} {lag if lag else '        -        -'} {r['a_safe']:7.1f} "
                  f"{r['t_lag'] * 1e3:10.1f} {r['t_lag_safe'] * 1e3:6.1f}")
            worst[drive] = min(worst.get(drive, 1e18), r['a_safe'])
            worst_lag[drive] = max(worst_lag.get(drive, 0.0), r['t_lag_safe'])
    print('\nSpeed ladder (all conditions pooled, by drive):')
    for drive in ('output', 'input'):
        mine = [r for r in rows if r['drive'] == drive]
        for v in sorted({round(r['v0'], 1) for r in mine}):
            at = [r for r in mine if round(r['v0'], 1) == v]
            a = [r['a_eff'] for r in at]
            print(f'  {drive:6s} {v:5.1f} rad/s  n={len(at):2d}  a_eff {min(a):6.1f} .. '
                  f'{max(a):6.1f}  travel {np.mean([r["travel"] for r in at]):.3f} rad')
    print('\nSuggested throw.stop_decel_rad_s2 (worst hold condition, a_safe, rounded down):')
    for drive in ('output', 'input'):
        if drive in worst:
            print(f'  {drive}: {int(worst[drive])}')
    print('\nAlternative, when a_eff climbs with speed in the ladder above '
          '(travel ~ proportional to v):')
    print('Suggested throw.stop_lag_s (worst hold condition, fast stops, rounded UP to 1 ms;')
    print('it replaces the decel for that drive):')
    for drive in ('output', 'input'):
        if drive in worst_lag:
            print(f'  {drive}: {math.ceil(worst_lag[drive] * 1e3) / 1e3:g}')


if __name__ == '__main__':
    main()
