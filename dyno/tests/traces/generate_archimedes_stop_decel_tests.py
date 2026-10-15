"""Stop-decel calibration plans for throw mode, as builder recipes.

Writes two plans, one per drive motor:

    archimedes_stop_decel_output   the OUTPUT (absorber) drives and brakes
    archimedes_stop_decel_input    the INPUT drives and brakes

Each is a speed ladder of throw segments. A throw fires its velocity-0 stop on a
predicted turnaround angle, so every throw is one measured stop from a known
speed. Fit the data with

    PYTHONPATH=. .venv/bin/python -m dyno.src.archimedes.fit_stop_decel <log dir>

and put the result in the rig config under `throw.stop_decel_rad_s2` (see
docs/archimedes_throw_mode_design.md, section 3).

WHAT THE LADDER COVERS
  * Speeds 0.5 .. 8 rad/s at the output (input 22 .. 351 rad/s). The fit needs
    several speeds because a lag term makes one v^2/(2a) figure speed-dependent.
  * Zero hold torque at every speed, then +T and -T at a few speeds, T being
    5 Nm referred to the output shaft. The hold sign follows the throw
    direction, so both assisting and opposing torque are exercised in both
    directions. Assisting torque lengthens the stop, so the worst case is the
    one to size from.
  * Four throws per segment, alternating direction, so each speed has stops in
    both directions and the shaft ends where it started.

WHY THE STOP DECEL IN THE PLAN IS LOW ON PURPOSE
  Each segment overrides `stop_decel_rad_s2` with a deliberately low guess
  (ASSUMED below). The lookahead then fires EARLY, so the real stop finishes
  well short of the target and short of the window, whatever the drive really
  does. The number being measured is the true decel, so the guess only has to
  be safe. The input's is lower still, because nothing has been measured for it.
  Climb the ladder in order: a stop that runs longer than the guess shows at low
  speed first, where it costs nothing.

RUNNING IT
  Declare centre and pass touch-off as for any test. The window stays armed
  behind the throws. The plan ends with a recentre.

  PYTHONPATH=. .venv/bin/python dyno/tests/traces/generate_archimedes_stop_decel_tests.py
"""
import math
import sys

from deployment import dyno_paths
from dyno.src import test_builder, test_preview

MODE = 'inhouse_archimedes'

# Output-shaft speeds [rad/s]. 8 is close to the 8.8 rad/s top of the test range.
SPEEDS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
LOADED_SPEEDS = (1.0, 3.0, 6.0)
LOAD_NM_AT_OUTPUT = 5.0         # hold torque, referred to the output shaft

# Deliberately low stop decel guesses [rad/s^2, output frame]. The measured
# output-drive stop in velocity_mode_6 was ~67.
ASSUMED = {'output': 40.0, 'input': 30.0}

RAMP_OUT = 60.0                 # speed-up accel [rad/s^2, output frame]
N_THROWS = 4                    # even, so each segment ends where it began
MARGIN = 0.05
WINDOW_SLACK = 0.02             # extra room under the builder's window check
MAX_TARGET = 1.0
MIN_TARGET = 0.3


def target_for(v, half_window, window_decel):
    """Largest turnaround the window's look-ahead allows at speed v, capped."""
    reach = v * v / (2.0 * window_decel) if window_decel > 0 else 0.0
    return math.floor(min(MAX_TARGET, half_window - MARGIN - WINDOW_SLACK - reach) * 100) / 100


def feasible(v, target, assumed):
    """A throw must have room to speed up and to brake from the fire point."""
    need = v * v / (2.0 * RAMP_OUT) + v * v / (2.0 * assumed) + 0.1
    return 2.0 * target >= need


def segment(seg_id, drive, v, target, hold_level, follows, ratio):
    k = ratio if drive == 'input' else 1.0
    hold = 'output' if drive == 'input' else 'input'
    seg = test_builder.default_segment(seg_id)
    seg['pattern'] = 'throw'
    seg['params'] = {key: spec[1] for key, spec in test_builder.PATTERNS['throw'].items()}
    seg['params'].update(speed_rad_s=round(v * k, 3), target_rad=target,
                         ramp_accel=round(RAMP_OUT * k, 3), n_throws=N_THROWS,
                         start_dir=1, hold_follows_dir=follows,
                         stop_decel_rad_s2=ASSUMED[drive])
    seg['primary'] = {'motor': drive, 'control_mode': 'velocity', 'accel': 1.0}
    seg['secondary'] = {'control_mode': 'torque', 'levels': [hold_level],
                        'rate': 1.0, 'settle_s': 0.0}
    seg['_hold'] = hold
    return seg


def build(drive, limits):
    rig = limits['throw_rig']
    half, window_decel = float(rig['half_window']), rig['window_decel']
    ratio = limits['input']['gear_ratio']
    # Hold torque for the loaded stops, in the HOLD motor's own units.
    load_nm = LOAD_NM_AT_OUTPUT / ratio if drive == 'output' else LOAD_NM_AT_OUTPUT
    load_nm = round(load_nm, 3)

    segs, skipped = [], []
    plan = [('Z', v, 0.0, False) for v in SPEEDS]
    for sign, tag in ((+1, 'P'), (-1, 'N')):
        plan += [(tag, v, sign * load_nm, True) for v in LOADED_SPEEDS]
    for tag, v, hold, follows in plan:
        target = target_for(v, half, window_decel)
        if target < MIN_TARGET or not feasible(v, target, ASSUMED[drive]):
            skipped.append((tag, v))
            continue
        segs.append(segment(f'{tag}{int(round(v * 10)):02d}', drive, v, target, hold,
                            follows, ratio))
    for seg in segs:
        seg.pop('_hold')

    # Back to centre, then done. Recentre drives the input whichever drove.
    rc = test_builder.default_segment('RC')
    rc['pattern'] = 'recentre'
    rc['params'] = {key: spec[1] for key, spec in test_builder.PATTERNS['recentre'].items()}
    rc['primary'] = {'motor': 'input', 'control_mode': 'velocity', 'accel': 1.0}
    rc['secondary'] = {'control_mode': 'torque', 'levels': [0.0], 'rate': 1.0,
                       'settle_s': 0.0}
    segs.append(rc)
    return {'name': f'archimedes_stop_decel_{drive}', 'segments': segs}, skipped


def main():
    limits = test_preview.limits_from_config(MODE)
    rig = limits['throw_rig']
    if not (rig.get('window_enabled') and rig.get('half_window')):
        sys.exit('position_window is not enabled in the rig config; the ladder is '
                 'sized against it')
    tests_dir = dyno_paths.dyno_test_directory
    for drive in ('output', 'input'):
        recipe, skipped = build(drive, limits)
        problems = []
        for seg in recipe['segments']:
            problems += [f"[{seg['id']}] {m}"
                         for m in test_builder.validate_segment(seg, limits)]
        if problems:
            sys.exit('\n'.join([f'{drive} plan does not validate:'] + problems))
        rel = test_builder.save_test(recipe, tests_dir)
        result = test_preview.expand_test(rel, MODE, limits)
        print(f"{drive:6s} drive: {rel}  ({len(recipe['segments']) - 1} throw segments, "
              f"{result['t'][-1]:.0f} s nominal)")
        for seg in recipe['segments'][:-1]:
            p = seg['params']
            print(f"    {seg['id']:5s} drive {p['speed_rad_s']:8.2f} rad/s  target "
                  f"{p['target_rad']:.2f} rad  hold {seg['secondary']['levels'][0]:+.3f} Nm")
        for tag, v in skipped:
            print(f'    skipped {tag} {v:g} rad/s: no room to speed up and stop inside the window')


if __name__ == '__main__':
    main()
