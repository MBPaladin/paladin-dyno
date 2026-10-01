"""Print the creep ratio of every efficiency segment in a log folder.

Same definition as the report generator (analyzers/efficiency.py::_creep):
creep = |1 - ratio x w_out / w_in| x 100 on the MEAN speeds of each settled
constant-speed leg, legs shorter than --min-leg-s dropped (default 0.1 s here; the report's CREEP_MIN_LEG_S is 0.25 s), legs that do not
track (> TRACKING_MAX_PCT) counted but excluded, median over the rest.

Why this is not just a call to the analyzer: throw-pattern logs (the
_alt_programmable tests) store every constant-speed leg as its own
`<level>-RUN0-SETPOINTn` segment, driven by a velocity command, and
dataset.load skips those ("no shaft is commanded"). The report therefore finds
nothing in them today. Here each SETPOINT segment IS one leg, and the legs of a
level are pooled the way _creep pools a shuttle's legs.

`ratio` must be the KINEMATIC ratio from a no-load sweep (a loaded fit would
hide the creep). Taken from, in order: --ratio, --ratio-from <folder>, a
sibling velocity_ramp/ folder, the 43.88 default.

    PYTHONPATH=. .venv/bin/python dyno/utilities/archimedes_creep.py \\
        dyno/logs/archimedes/gbx_1p2p2/forward_driving/efficiency_extended_85
"""
import argparse
import glob
import os
import sys

import numpy as np

from dyno.src.analysis.segment import Log
from dyno.src.archimedes import dataset, naming
from dyno.src.archimedes.analyzers import efficiency as eff


def measured_ratio(folder):
    """Dataset._ratio_votes, run straight off the segments: a throw-format
    velocity ramp logs SETPOINT segments that dataset.load refuses to classify,
    so the loader cannot be used here. Same rule -- |w_out| > 0.02 rad/s, at
    least 200 samples, median over segments -- BUT the per-segment figure is
    mean(w_in)/mean(w_out), not Dataset's regression slope. A SETPOINT segment
    is one constant-speed leg, so w_out spans almost nothing and a slope of
    w_in on w_out is flattened by the w_out noise (40.6 against a true ~43.2
    here); the ratio of means has no such bias."""
    votes = []
    for path in sorted(glob.glob(os.path.join(folder, '**', '*.hdf5'), recursive=True)):
        with Log(os.path.dirname(path), path) as log:
            for seg in log.segments:
                if naming.parse(seg.raw_id).kind != naming.VELOCITY:
                    continue
                wi, wo = seg['dut_velocity'], seg['load_velocity']
                m = np.isfinite(wi) & np.isfinite(wo) & (np.abs(wo) > 0.02)
                if m.sum() >= 200:
                    votes.append(abs(np.mean(wi[m]) / np.mean(wo[m])))
    return float(np.median(votes)) if votes else None


MIN_LEG_S = 0.1     # the report uses eff.CREEP_MIN_LEG_S (0.25); 3000 rpm legs only last ~0.16 s


def leg_creeps(log_root, ratio, min_leg_s=MIN_LEG_S):
    """{point: [creep % per leg]} for every efficiency segment under log_root."""
    levels, short = {}, {}
    for path in sorted(glob.glob(os.path.join(log_root, '**', '*.hdf5'), recursive=True)):
        with Log(os.path.dirname(path), path) as log:
            for seg in log.segments:
                pt = naming.parse(seg.raw_id)
                if pt.kind != naming.EFFICIENCY:
                    continue
                dt = seg.dt if np.isfinite(seg.dt) and seg.dt > 0 else 1e-3
                if len(seg) * dt < min_leg_s:
                    short[(pt.behavior, pt.rpm, pt.torque_nm)] = \
                        short.get((pt.behavior, pt.rpm, pt.torque_nm), 0) + 1
                    continue
                wi = float(np.nanmean(seg['dut_velocity']))
                wo = float(np.nanmean(seg['load_velocity']))
                if not (np.isfinite(wi) and np.isfinite(wo)) or abs(wi) < 1e-9:
                    continue
                key = (pt.behavior, pt.rpm, pt.torque_nm)
                levels.setdefault(key, []).append(abs(1.0 - ratio * wo / wi) * 100.0)
    return levels, short


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('log', help='folder holding the efficiency .hdf5 log(s)')
    ap.add_argument('--ratio', type=float, help='kinematic ratio, input rad per output rad')
    ap.add_argument('--ratio-from', metavar='DIR',
                    help='folder with a no-load velocity ramp to measure the ratio on')
    ap.add_argument('--min-leg-s', type=float, default=MIN_LEG_S,
                    help=f'shortest leg kept, seconds (default {MIN_LEG_S:g}; the report uses '
                         f'{eff.CREEP_MIN_LEG_S:g})')
    args = ap.parse_args()

    log = os.path.abspath(args.log)
    ratio, how = None, None
    if args.ratio:
        ratio, how = args.ratio, '--ratio'
    else:
        src = args.ratio_from or os.path.join(os.path.dirname(log), 'velocity_ramp')
        if os.path.isdir(src):
            ratio = measured_ratio(src)
            how = f'measured on the no-load ramp in {os.path.relpath(src)}'
    if ratio is None:
        ratio, how = 43.88, 'DEFAULT (no no-load sweep found; pass --ratio or --ratio-from)'

    levels, short = leg_creeps(log, ratio, args.min_leg_s)
    for key in short:
        levels.setdefault(key, [])
    if not levels:
        sys.exit(f'no efficiency segments under {log}')

    print(f'ratio {ratio:.4f}  ({how})')
    print(f'{"segment":<12}{"rpm":>6}{"T_cmd":>8}{"legs":>6}{"track":>6}{"creep %":>10}{"max %":>9}')
    over = 0
    for (name, rpm, torque), vals in sorted(levels.items(), key=lambda kv: (kv[0][1], -kv[0][2])):
        tracking = [v for v in vals if v <= eff.TRACKING_MAX_PCT]
        med = float(np.median(tracking)) if tracking else None
        over += bool(med is not None and med > eff.CREEP_LIMIT_PCT)
        f = lambda v: '       n/a' if v is None else f'{v:>9.3f}'
        note = (f'   ({short[(name, rpm, torque)]} leg(s) under {args.min_leg_s:g} s dropped)'
                if (name, rpm, torque) in short else '')
        print(f'{name:<12}{rpm:>6g}{torque:>8g}{len(vals):>6}{len(tracking):>6}'
              f'{f(med):>10}{f(max(tracking) if tracking else None):>9}{note}')
    print(f'\n{len(levels)} segment(s); {over} above the {eff.CREEP_LIMIT_PCT:g} % limit')


if __name__ == '__main__':
    main()
