"""Regenerate ONE forward alternating-efficiency test at a chosen torque.

Reproduces archimedes_fwd_efficiency_alt_80 with 80 Nm replaced by --torque:
at each efficiency speed, a +T segment then a -T segment, the input driving
velocity throws while the output holds the torque. The throw geometry, throw
counts, ramps and hold ramp come from generate_archimedes_tests (same
functions, same constants), so nothing here can drift from the main generator.

The result is always written to the same name, archimedes_fwd_efficiency_alt_programmable
(override with --name), so each run replaces the last rather than adding a file.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python dyno/tests/traces/generate_archimedes_fwd_efficiency_programmable.py --torque 80
    ... --torque 135 --output-cap 140       past the customer's 100 Nm limit
    ... --torque 105:130:5 --rpm 20,300 --output-cap 130 --name archimedes_fwd_efficiency_alt_105_130
"""
import argparse
import sys

from deployment import dyno_paths
import generate_archimedes_tests as g   # sibling script (its directory is sys.path[0])

DEFAULT_NAME = 'archimedes_fwd_efficiency_alt_programmable'
LEVEL_RATE_NM_S = 50.0      # hold-torque ramp rate, same literal build_efficiency uses
SETTLE_S = 1.0


def level_tag(torque):
    """80 -> '80' (matches the hand-generated ids); 12.5 -> '12p5'."""
    return f'{torque:g}'.replace('.', 'p').zfill(2)


def torque_list(text):
    """'80' -> [80.0]; '105,110' -> [105.0, 110.0]; '105:130:5' -> 105..130 step 5."""
    out = []
    for part in text.split(','):
        if ':' in part:
            lo, hi, step = (float(x) for x in part.split(':'))
            n = int(round((hi - lo) / step))
            out.extend(lo + i * step for i in range(n + 1))
        else:
            out.append(float(part))
    return out


def build(torques, rpms, name, cfg):
    motor = 'input'
    notes, segs = [], []
    for rpm in rpms:
        for torque in torques:
            for sign, letter in ((1, 'P'), (-1, 'N')):
                sid = f'E{rpm:04d}_{letter}{level_tag(torque)}'
                seg = g.plan_throw(sid, motor, rpm, cfg, notes,
                                   hold_level=sign * torque,
                                   level_rate=LEVEL_RATE_NM_S,
                                   settle_s=SETTLE_S,
                                   hold_ramp=g.THROW_HOLD_RAMP_OUT_NM_S)
                if seg is None:
                    sys.exit(f'{sid}: {rpm} rpm cannot be thrown with the current '
                             'constants (see notes above); this generator does not '
                             'fall back to shuttles')
                seg.pop('_geo', None)
                segs.append(seg)
    notes.append(f'programmable efficiency: +/-{", ".join(f"{x:g}" for x in torques)} Nm '
                 f'output at {", ".join(str(x) for x in rpms)} rpm input, '
                 f'{len(segs)} throw segments')
    return {'name': name, 'segments': segs}, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--torque', type=torque_list, required=True, metavar='LIST',
                    help='output torque magnitude(s) held during the throws: 80, or 105,110, or '
                         'lo:hi:step like 105:130:5 (segments '
                         'run at +T then -T)')
    ap.add_argument('--rpm', type=g._rpm_list, default=None, metavar='LIST',
                    help=f'input speeds, comma separated (default '
                         f'{",".join(str(x) for x in g.EFF_RPM)})')
    ap.add_argument('--output-cap', type=float, default=None, metavar='NM',
                    help=f'raise the output torque cap for this run (default '
                         f'{g.OUTPUT_TORQUE_CAP_NM:g}, the customer limit in '
                         'generate_archimedes_tests.py); needed for --torque above it')
    ap.add_argument('--name', default=DEFAULT_NAME,
                    help=f'test name to write, overwriting any previous (default {DEFAULT_NAME})')
    args = ap.parse_args()

    if args.output_cap is not None:
        g.OUTPUT_TORQUE_CAP_NM = args.output_cap    # one run only; the file is untouched
    lo, hi = min(args.torque), max(args.torque)
    if lo <= 0:
        sys.exit('--torque is a magnitude and must be > 0')
    if hi > g.OUTPUT_TORQUE_CAP_NM:
        sys.exit(f'--torque {hi:g} is over the {g.OUTPUT_TORQUE_CAP_NM:g} Nm '
                 'output cap in generate_archimedes_tests.py '
                 '(pass --output-cap NM to raise it for this run)')

    cfg = g.load_config()
    if hi > cfg['output_torque_safety']:
        sys.exit(f'--torque {hi:g} is over safeties.output_torque.limit '
                 f'({cfg["output_torque_safety"]:g} Nm) in the rig config')
    if g.T_SLIP_OUT_NM and hi > g.EFF_SLIP_MARGIN * g.T_SLIP_OUT_NM:
        print(f'WARNING: {hi:g} Nm is above {g.EFF_SLIP_MARGIN:g} x the measured '
              f'{g.T_SLIP_OUT_NM:g} Nm slip torque; the unit may slip rather than load')

    rpms = args.rpm or list(g.EFF_RPM)
    recipe, notes = build(args.torque, rpms, args.name, cfg)
    result = g.write_plan(recipe, notes, cfg, dyno_paths.dyno_test_directory)
    return 0 if result and result[3] else 1


if __name__ == '__main__':
    sys.exit(main())
