"""Throw behavior: logic only, no bus.

`throw` runs one motor in velocity mode limit to limit, firing a stop on a
predicted turnaround angle (x + v^2/(2*stop_decel) >= target, live output
encoder) and reversing. What a bus cannot check cheaply is checked here against
a stand-in plant: the direction it picks, that the turnaround lands on the
target, that it uses LIVE velocity, one ratio_reset per stop, what it refuses to
do, and that nothing ever gets near the position window.

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/throw_logic_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from throw_sim_plant import (HALF_WINDOW, LIMITS, RATIO, Plant, make,  # noqa: E402
                             run)

fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


def refuses(**kw):
    try:
        make(**kw)
    except AssertionError as e:
        return str(e)
    return None


print('--- refuses rather than guesses ---')
for label, reader in (('no centre declared', Plant().reader(centre=None)),
                      ('output position NaN',
                       lambda: {'position': {'output': float('nan')}, 'centre': 0.0,
                                'velocity': {'output': 0.0}, 'input_sign': 1})):
    cmds = run(make(reader=reader))
    check(all(c['output_command'] == 0.0 and c['input_command'] == 0.0 for c in cmds),
          f'{label}: never commands motion')

cmds = run(make(reader=Plant(drive='input').reader(input_sign=None), drive_motor='input'))
check(all(c['input_command'] == 0.0 for c in cmds),
      'input drive without input_sign: never commands motion')

p = Plant(drive='output')
cmds = run(make(reader=p.reader(input_sign=None)), p)
check(any(c['output_command'] != 0.0 for c in cmds),
      'output drive does not need input_sign')

print('--- load-time refusals ---')
check(refuses(rig={'window_enabled': True, 'half_window': 1.5, 'window_decel': 70.0,
                   'stop_decel': {'output': None}}) is not None,
      'a null stop_decel for the drive motor refuses the plan')
check(refuses(drive_motor='input', rig={'window_enabled': False,
                                        'stop_decel': {'output': 70.0, 'input': None}})
      is not None, 'input drive refused while only the output is measured')
check(refuses(target_rad=1.45) is not None,
      'target too close to the window (with its look-ahead + margin) is refused')
check(refuses(speed_rad_s=100.0) is not None, 'speed over the motor limit is refused')
check(refuses(ramp_accel=1e5) is not None, 'ramp_accel over the motor limit is refused')
check(refuses(hold_level=500.0) is not None, 'hold torque over the limit is refused')
check(refuses(stop_decel_rad_s2=50.0, target_rad=1.0) is None,
      'settings stop_decel_rad_s2 overrides the rig value (calibration runs)')

print('--- direction and turnaround ---')
p = Plant()
beh = make(p)
cmds = run(beh, p)
moving = [c['output_command'] for c in cmds if c['output_command'] != 0.0]
check(len(beh.stops) == 4, f'ran all {len(beh.stops)}/4 throws')
check([s['direction'] for s in beh.stops] == [1, -1, 1, -1],
      'directions alternate, starting with start_dir=+1')
check(moving[0] > 0, 'first throw goes positive')
check(all(abs(abs(s['peak_rel']) - 1.0) < 0.02 for s in beh.stops),
      'turnaround lands on target_rad (fitted decel == true decel): '
      + ', '.join('%+.3f' % s['peak_rel'] for s in beh.stops))
check(p.peak < HALF_WINDOW - 0.3,
      f'the position window is never approached (peak {p.peak:.3f} of {HALF_WINDOW})')

p = Plant()
beh = make(p, start_dir=-1)
run(beh, p)
check(beh.stops[0]['direction'] == -1, 'start_dir=-1 goes negative first')

p = Plant(start_rel=1.2)
beh = make(p)
run(beh, p)
check(beh.stops[0]['direction'] == -1,
      'starting past the target on the start_dir side reverses the first throw')

print('--- lookahead uses LIVE velocity ---')
# The shaft runs 40% over its command, as in velocity_mode_6. A trigger built on
# the commanded speed would stop late; live v stops on target.
p = Plant(gain=1.4)
beh = make(p)
run(beh, p)
check(all(abs(abs(s['peak_rel']) - 1.0) < 0.03 for s in beh.stops),
      'command overshoot does not move the turnaround: '
      + ', '.join('%+.3f' % s['peak_rel'] for s in beh.stops))
check(all(abs(s['fire_v']) > 3.6 for s in beh.stops), 'fired at the live (overshot) speed')

# Fitted decel optimistic vs the plant: overshoots by the difference, and the
# per-stop record says so (this is what the calibration reads).
p = Plant(true_decel=50.0)
beh = make(p)
run(beh, p)
s = beh.stops[0]
check(abs(s['peak_rel']) > 1.0 + 0.02 and s['travel'] > s['predicted_travel'] + 0.02,
      'a fitted decel above the plant overshoots the target, and the record shows '
      f"travel {s['travel']:.3f} vs predicted {s['predicted_travel']:.3f}")

print('--- ratio reset ---')
p = Plant()
beh = make(p)
cmds = run(beh, p)
resets = [c for c in cmds if c.get('ratio_reset')]
check(len(resets) == 4, f'ratio_reset emitted once per stop ({len(resets)} for 4)')

print('--- what it commands ---')
p = Plant()
cmds = run(make(p), p)
check(all(c['output_mode'] == 'velocity' and c['input_mode'] == 'torque' for c in cmds),
      'output drive: output in velocity, input holds torque')
check(cmds[-1]['output_command'] == 0.0, 'ends at zero speed')
peak = max(abs(c['output_command']) for c in cmds)
check(peak <= 3.0 + 1e-9, f'never exceeds the configured speed ({peak:.2f})')
ramp = max(abs(b['output_command'] - a['output_command'])
           for a, b in zip(cmds, cmds[1:]) if b['output_command'] != 0.0)
check(ramp <= 100.0 * 0.001 + 1e-6, f'speed-up is slewed at ramp_accel ({ramp:.4f}/cycle)')
flagged = [c for c in cmds if 'log_flag' in c]
check(flagged and all(abs(abs(c['output_command']) - 3.0) < 1e-6 for c in flagged),
      'only the steady-speed part of a throw is log-flagged')
check(len({c['log_flag'] for c in flagged}) == 4, 'one flag per throw')

p = Plant()
cmds = run(make(p, hold_level=5.0, hold_follows_dir=True), p)
holds = {c['input_command'] for c in cmds}
check(holds == {5.0, -5.0}, 'hold torque follows the throw direction when asked')
p = Plant()
cmds = run(make(p, hold_level=5.0), p)
check({c['input_command'] for c in cmds} == {5.0}, 'hold torque is fixed otherwise')

print('--- input drive: output-encoder stopping through the gearbox ---')
for sign in (1, -1):
    p = Plant(drive='input', input_sign=sign)
    beh = make(p, drive_motor='input', speed_rad_s=3.0 * RATIO)
    cmds = run(beh, p)
    first = next(c['input_command'] for c in cmds if c['input_command'] != 0.0)
    check(first * sign > 0 and len(beh.stops) == 4,
          f'input_sign {sign:+d}: first throw goes output-positive, 4 throws')
    check(all(abs(abs(s['peak_rel']) - 1.0) < 0.03 for s in beh.stops),
          f'input_sign {sign:+d}: turnaround lands on target in OUTPUT coordinates')
    check(all(c['input_mode'] == 'velocity' and c['output_mode'] == 'torque'
              for c in cmds), 'input drive: input in velocity, output holds torque')

print('--- always ends ---')
p = Plant()
p.frozen = True
beh = make(p)
cmds = run(beh, p)
check(len(cmds) < 3.0 / 0.001 * 4 + 1000 and cmds[-1]['output_command'] == 0.0,
      f'frozen feedback: a throw that never fires times out and stops ({len(cmds)} cycles)')

p = Plant(true_decel=0.5)      # a stop that will not come to rest inside the bound
beh = make(p)
cmds = run(beh, p)
check(cmds[-1]['output_command'] == 0.0 and len(beh.stops) < 4,
      'a stop that never reaches rest ends the series instead of throwing on')

print('--- progress contract ---')
p = Plant()
beh = make(p, n_throws=5)
seen = []
for c in beh.commands():
    p.step(c)
    seen.append(beh.repeat)
check(beh.repeats == 5 and max(seen) == 5 and seen == sorted(seen),
      f'repeats={beh.repeats}, repeat climbs 1..{max(seen)}')

print('--- offline preview (no sensor_reader) ---')
beh = make()
cmds = run(beh)
speeds = [c['output_command'] for c in cmds]
signs = []
for v in speeds:
    if v != 0 and (not signs or signs[-1] != (v > 0)):
        signs.append(v > 0)
check(signs == [True, False, True, False],
      'dead-reckoned preview shows four alternating throws')
check(abs(max(speeds) - 3.0) < 1e-9 and abs(min(speeds) + 3.0) < 1e-9,
      'preview reaches +/- the configured speed')
check(speeds[-1] == 0.0, 'preview ends at rest')

print('\n' + ('ALL OK' if not fails else f'{len(fails)} FAILED'))
sys.exit(1 if fails else 0)
