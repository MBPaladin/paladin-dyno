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

print('--- stop lag (travel grows linearly with speed) ---')
LAG_RIG = {'window_enabled': True, 'half_window': HALF_WINDOW, 'window_decel': 70.0,
           'stop_decel': {'output': 70.0, 'input': 70.0}, 'stop_lag': {'output': 0.06}}
for speed in (0.7, 2.0, 4.0):
    p = Plant(true_lag=0.06)
    beh = make(p, rig=LAG_RIG, speed_rad_s=speed, ramp_accel=40.0, target_rad=1.0)
    run(beh, p)
    check(len(beh.stops) == 4 and all(abs(abs(s['peak_rel']) - 1.0) < 0.03
                                      for s in beh.stops),
          f'{speed:g} rad/s: lag stop lands on target at every speed: '
          + ', '.join('%+.3f' % s['peak_rel'] for s in beh.stops))
# A constant decel fitted at the fast end (a_eff rises with speed under a PI
# velocity loop) under-predicts the slow stops, so they turn late.
q = Plant(true_lag=0.06)
slow = make(q, speed_rad_s=0.7, ramp_accel=40.0, stop_decel_rad_s2=4.0 / (2 * 0.06))
run(slow, q)
check(all(abs(s['peak_rel']) > 1.02 for s in slow.stops),
      'a constant decel fitted at speed turns a slow throw late (the lag does not): '
      + ', '.join('%+.3f' % s['peak_rel'] for s in slow.stops))
p = Plant(true_lag=0.06)
beh = make(p, rig=LAG_RIG, speed_rad_s=3.0)
run(beh, p)
s = beh.stops[0]
check(abs(s['travel'] - s['predicted_travel']) < 0.01,
      f"the stop record predicts the lag travel ({s['travel']:.3f} vs {s['predicted_travel']:.3f})")
check(refuses(rig={'window_enabled': False, 'stop_decel': {'output': None},
                   'stop_lag': {'output': 0.06}}) is None,
      'a lag alone is enough: no constant decel needed for the drive')
check(refuses(rig={'window_enabled': False, 'stop_decel': {'output': None},
                   'stop_lag': {}}) is not None,
      'neither a decel nor a lag refuses the plan')
p = Plant(true_decel=70.0)
beh = make(p, rig=LAG_RIG, stop_decel_rad_s2=70.0)
run(beh, p)
check(beh.stop_lag is None and beh.stop_decel == 70.0,
      'a stop_decel_rad_s2 setting (calibration) wins over the rig lag')
beh = make(Plant(), rig=LAG_RIG)
check(beh.stop_lag == 0.06 and beh.stop_decel is None,
      'a rig lag replaces the constant decel for that drive')
check(make(Plant(drive='input'), rig=LAG_RIG, drive_motor='input',
           speed_rad_s=3.0 * RATIO).stop_lag is None,
      'the lag is per drive: the input keeps its constant decel')

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

print('--- hold torque meets zero between segments ---')
p = Plant()
beh = make(p, n_throws=3, hold_level=4.0, hold_ramp_nm_s=20.0, zero_hold_s=0.1)
cmds = run(beh, p)
hold = [c['input_command'] for c in cmds]           # output drives, input holds
speed = [c['output_command'] for c in cmds]
first_move = next(i for i, v in enumerate(speed) if v != 0.0)
last_move = max(i for i, v in enumerate(speed) if v != 0.0)
check(hold[0] == 0.0 and abs(hold[first_move - 1] - 4.0) < 1e-9,
      'torque starts at 0 and has reached the level before the first motion')
check(abs(max(abs(b - a) for a, b in zip(hold, hold[1:])) - 20.0 * beh.dt) < 1e-6,
      f'torque never changes faster than the ramp ({20.0 * beh.dt:.4f} Nm/cycle)')
check(all(abs(h - 4.0) < 1e-9 for h in hold[first_move:last_move]),
      'torque is constant through the throws')
tail = hold[last_move + 1:]
check(abs(tail[-1]) < 1e-9 and len(tail) >= int(0.1 / beh.dt) and tail[-int(0.1 / beh.dt):] == [0.0] * int(0.1 / beh.dt),
      'ends ramped to 0 with the zero-torque hold')
check(all(c['ratio_reset'] for c in cmds if c.get('ratio_reset')), 'ratio resets still happen at every stop')
check(sum(1 for c in cmds if c.get('ratio_reset')) == 3, 'one ratio_reset per throw')

print('--- position hold through the torque ramps ---')


class Walker(Plant):
    """Plant whose shaft walks OPPOSITE to any change in the hold torque, by
    `walk` rad per Nm of the holding (input) motor. 0.0115 is the LOAD AKD's
    measured rad per OUTPUT Nm; here it is applied per hold Nm, which only
    sets the scale of the disturbance (0.23 rad/s at the 20 Nm/s ramp)."""

    def __init__(self, walk, **kw):
        super().__init__(**kw)
        self.walk, self.last = walk, 0.0

    def step(self, cmd):
        super().step(cmd)
        hold = cmd['input_command']
        self.pos -= self.walk * (hold - self.last)
        self.last = hold


def ramp_walk(kp):
    p = Walker(0.0115, drive='output')
    beh = make(p, n_throws=2, hold_level=12.0, hold_ramp_nm_s=20.0, zero_hold_s=0.3,
               hold_position_kp=kp)
    worst = 0.0
    for c in beh.commands():
        p.step(c)
        if c['input_command'] != 12.0 and beh.repeat == 2:
            worst = max(worst, abs(p.pos - p.centre - beh.stops[-1]['rest_rel']))
    return worst, beh


w0, _ = ramp_walk(0.0)
w4, beh4 = ramp_walk(4.0)
check(w0 > 0.05, f'without the hold the shaft walks off the turnaround ({w0:.3f} rad)')
check(w4 < 0.6 * w0, f'kp 4 /s cuts the walk to {w4:.3f} rad (vs {w0:.3f})')
check(abs(beh4.stops[-1]['rest_rel']) > 0.9, 'turnaround still lands on the target with the hold on')

p = Plant()
cmds = run(make(p, n_throws=2, hold_level=4.0, hold_ramp_nm_s=20.0, zero_hold_s=0.1,
                hold_position_kp=4.0), p)
check(abs(max(abs(c['output_command']) for c in cmds[:150])) < 1e-9,
      'undisturbed shaft: the hold commands no speed (nothing to correct)')
cmds = run(make(reader=Plant().reader(centre=None), n_throws=2, hold_level=4.0,
                hold_ramp_nm_s=20.0, hold_position_kp=4.0))
check(all(c['output_command'] == 0.0 for c in cmds), 'no feedback: still never commands motion')

print('--- default (no ramp) is the original step ---')
p = Plant()
cmds = run(make(p, n_throws=2, hold_level=4.0), p)
check(cmds[0]['input_command'] == 4.0, 'hold_ramp_nm_s = 0 commands the level at once')

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
