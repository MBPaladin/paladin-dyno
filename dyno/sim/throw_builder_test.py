"""Throw pattern in the test builder: validation, preview, compile. No bus, no Qt.

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/throw_builder_test.py
"""
import copy
import sys

from dyno.src import test_builder as tb

fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


LIMITS = {'input': {'torque': 15.0, 'velocity': 500.0, 'acceleration': 7500.0,
                    'rotatum': 5000.0, 'gear_ratio': 43.88},
          'output': {'torque': 200.0, 'velocity': 60.0, 'acceleration': 2000.0,
                     'rotatum': 5000.0, 'gear_ratio': 1.0},
          'coupled': True,
          'throw_rig': {'window_enabled': True, 'half_window': 1.5,
                        'window_decel': 70.0,
                        'stop_decel': {'output': 70.0, 'input': None}}}


def segment(**params):
    seg = tb.default_segment('TH')
    seg['pattern'] = 'throw'
    seg['params'] = {k: v[1] for k, v in tb.PATTERNS['throw'].items()}
    seg['params'].update(params)
    seg['primary'] = {'motor': 'output', 'control_mode': 'velocity', 'accel': 1.0}
    seg['secondary'] = {'control_mode': 'torque', 'levels': [5.0], 'rate': 1.0,
                        'settle_s': 1.0}
    return seg


print('--- validation ---')
check(tb.validate_segment(segment(speed_rad_s=4.0), LIMITS) == [],
      'a sensible output-drive throw is clean')
issues = tb.validate_segment(segment(speed_rad_s=4.0, target_rad=1.4), LIMITS)
check(len(issues) == 1 and 'half_window_rad' in issues[0],
      'a target the window would abort is flagged')
inp = segment(speed_rad_s=100.0, ramp_accel=5000.0)
inp['primary']['motor'] = 'input'
issues = tb.validate_segment(inp, LIMITS)
check(len(issues) == 1 and 'stop decel' in issues[0],
      'an input drive with no measured stop decel is flagged')
bad = segment()
bad['primary']['control_mode'] = 'torque'
check(any('velocity mode' in i for i in tb.validate_segment(bad, LIMITS)),
      'primary must be velocity mode')
bad = segment()
bad['secondary']['control_mode'] = 'velocity'
check(any('torque mode' in i for i in tb.validate_segment(bad, LIMITS)),
      'secondary must be torque mode')
check(any('velocity limit' in i for i in
          tb.validate_segment(segment(speed_rad_s=100.0), LIMITS)),
      'speed over the motor limit is flagged')

print('--- settings and compile ---')
s = tb.throw_settings(segment(speed_rad_s=4.0, start_dir=-1))
check(s['drive_motor'] == 'output' and s['hold_level'] == 5.0 and s['start_dir'] == -1,
      'drive motor from primary, hold level from the secondary, start_dir kept')
recipe = {'name': 'x', 'segments': [segment(), dict(copy.deepcopy(segment()), id='T2',
                                                    repeats=3)]}
doc = tb.build_yaml_dict(recipe)
types = [(b['id'], b['type']) for b in doc['behaviors']]
check(types == [('TH', 'throw'), ('T2_LOOP', 'loop')],
      f'compiles to throw behaviors, repeats wrapped in a loop {types}')
check(tb.is_generative(segment()), 'throw is generative (no trace csv)')

print('--- preview ---')
cols, rows = tb.throw_preview_rows(segment(speed_rad_s=4.0, n_throws=4), LIMITS)
check(cols == ['time', 'output_motor_velocity', 'input_motor_torque'],
      f'channels are the driven velocity and the hold torque {cols}')
peaks = [r[1] for r in rows if abs(r[1]) > 3.9]
check(max(peaks) == 4.0 and min(peaks) == -4.0, 'reaches +/- the drive speed')
check(all(b[0] > a[0] for a, b in zip(rows, rows[1:])), 'time increases')
check(rows[-1][1] == 0.0, 'ends at rest')
_c, rin = tb.throw_preview_rows(inp, LIMITS)
check(abs(max(r[1] for r in rin) - 100.0) < 1e-9,
      'input drive: preview speed is in the input shaft frame')
_c, short = tb.throw_preview_rows(segment(speed_rad_s=8.0, target_rad=0.05), LIMITS)
check(max(r[1] for r in short) < 8.0, 'a throw too short to reach speed peaks lower')

print('\n' + ('ALL OK' if not fails else f'{len(fails)} FAILED'))
sys.exit(1 if fails else 0)
