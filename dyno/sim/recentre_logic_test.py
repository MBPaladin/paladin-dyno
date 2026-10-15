"""Recentre behavior: logic only, no bus.

`recentre` drives the output back to the declared centre using the INPUT shaft,
between efficiency levels, so creep is undone instead of being paid for out of
the throw. Everything here is the part a bus cannot check cheaply: the direction
it picks, what it refuses to do, and that it leaves the input in velocity mode --
which is what makes the move survive the next segment's position-mode re-zero.

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/recentre_logic_test.py
"""
import sys

from dyno.src.test_manager import Recentre

fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


def reader(positions, centre=0.0, input_sign=1):
    """A sensor_reader whose output position walks toward centre whenever the
    behavior commands the right way -- a stand-in plant, one sample per call."""
    state = {'i': 0}

    def read():
        i = min(state['i'], len(positions) - 1)
        state['i'] += 1
        return {'torque': {}, 'velocity': {'input': 0.0, 'output': 0.0},
                'position': {'input': 0.0, 'output': positions[i]},
                'centre': centre, 'input_sign': input_sign, 'ratio': 43.88}
    return read


def behavior(reader_fn=None, **settings):
    s = {'tolerance_rad': 0.02, 'slow_within_rad': 0.05, 'velocity_rad_s': 4.0,
         'approach_velocity_rad_s': 1.0, 'acceleration_rad_s2': 20.0,
         'timeout_s': 0.5, 'settle_s': 0.02}
    s.update(settings)
    limits = {'input': {'velocity': 500.0, 'acceleration': 5500.0, 'torque': 15.0},
              'output': {'velocity': 12.0, 'acceleration': 150.0, 'torque': 200.0},
              'coupled': True}
    return Recentre({'id': 'RC', 'settings': s}, 'inhouse_archimedes', limits,
                    reader_fn)


print('--- refuses rather than guesses ---')
cmds = list(behavior(None).commands())
check(all(c['input_command'] == 0.0 for c in cmds),
      'no sensor_reader: never commands motion')

cmds = list(behavior(reader([0.5], centre=None)).commands())
check(all(c['input_command'] == 0.0 for c in cmds),
      'no centre declared: never commands motion')

cmds = list(behavior(reader([0.5], input_sign=None)).commands())
check(all(c['input_command'] == 0.0 for c in cmds),
      'input_sign unmeasured: never commands motion')

print('--- direction ---')
# Output sits BELOW centre (error = centre - position > 0), input_sign +1:
# needs a positive input command.
cmds = list(behavior(reader([-0.5] * 2000, centre=0.0, input_sign=1)).commands())
moving = [c['input_command'] for c in cmds if c['input_command'] != 0.0]
check(moving and all(v > 0 for v in moving),
      f'output below centre, input_sign +1 -> positive input ({max(moving):+.2f} peak)')

cmds = list(behavior(reader([+0.5] * 2000, centre=0.0, input_sign=1)).commands())
moving = [c['input_command'] for c in cmds if c['input_command'] != 0.0]
check(moving and all(v < 0 for v in moving),
      f'output above centre, input_sign +1 -> negative input ({min(moving):+.2f} peak)')

cmds = list(behavior(reader([-0.5] * 2000, centre=0.0, input_sign=-1)).commands())
moving = [c['input_command'] for c in cmds if c['input_command'] != 0.0]
check(moving and all(v < 0 for v in moving),
      'input_sign -1 flips the command for the same error')

print('--- stopping ---')
cmds = list(behavior(reader([0.001] * 2000, centre=0.0)).commands())
check(all(c['input_command'] == 0.0 for c in cmds),
      'already inside tolerance: stops without moving')

# Error growing: the direction is wrong in a way input_sign did not predict.
walk_away = [0.10 + 0.01 * i for i in range(2000)]
cmds = list(behavior(reader(walk_away, centre=0.0)).commands())
grew = len(cmds)
cmds_far = list(behavior(reader([0.5] * 2000, centre=0.0)).commands())
check(grew < len(cmds_far),
      f'a growing error stops early ({grew} cycles vs {len(cmds_far)} on a steady one)')

print('--- what it leaves behind ---')
cmds = list(behavior(reader([-0.5] * 2000, centre=0.0)).commands())
check(all(c['input_mode'] == 'velocity' for c in cmds),
      'input stays in VELOCITY mode throughout (so the next segment re-zeros)')
check(all(c['output_mode'] == 'torque' and c['output_command'] == 0.0 for c in cmds),
      'output held at 0 Nm throughout')
check(cmds[-1]['input_command'] == 0.0, 'ends at zero speed')
peak = max(abs(c['input_command']) for c in cmds)
ramp = max(abs(b['input_command'] - a['input_command'])
           for a, b in zip(cmds, cmds[1:]))
check(ramp <= 20.0 * 0.001 + 1e-9,
      f'speed is slewed at the configured acceleration, not stepped ({ramp:.4f} per cycle)')
check(peak <= 4.0 + 1e-9, f'never exceeds the configured speed ({peak:.2f} rad/s)')

print('\n' + ('ALL OK' if not fails else f'{len(fails)} FAILED'))
sys.exit(1 if fails else 0)
