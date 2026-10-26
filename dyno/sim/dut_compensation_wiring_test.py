"""Bus-free check of the dut_compensation wiring: GridSearch profile cycling and
Controller._comp_step / _comp_load_feedforward on a stubbed controller.

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/dut_compensation_wiring_test.py
"""
import math
import os
import types

os.environ.setdefault('DYNO_SIM', 'inhouse_archimedes')

from dyno.src.test_manager import GridSearch          # noqa: E402
from dyno.src.dyno_controller import Controller       # noqa: E402
from dyno.src.dut_compensation import compile_dut_compensation  # noqa: E402
import yaml                                           # noqa: E402
from deployment import dyno_paths                     # noqa: E402

ok = True


def check(cond, msg):
    global ok
    if not cond:
        ok = False
        print('FAIL:', msg)


LIMITS = {'input': {'torque': 1.0, 'rotatum': 200.0, 'velocity': 1200.0, 'acceleration': 1000.0,
                    'gear_ratio': 26},
          'output': {'torque': 30.0, 'rotatum': 200.0, 'velocity': 10.0, 'acceleration': 10.0}}


def grid(profiles, torques=(12.0, -12.0)):
    s = {'input_motor': {'control_mode': 'torque', 'command_list': list(torques)},
         'output_motor': {'control_mode': 'velocity', 'command_list': [0.1]},
         'loop_order': ['velocity', 'torque'], 'duration_per_point_s': 0.005,
         'min_output_revs': 0.0, 'max_hold_s': 1.0, 'settle_time_s': 0.002,
         'transition_rate': 1.0, 'continuous_torque': 20.0, 'profile_settle_s': 0.002}
    if profiles is not None:
        s['dut_compensation_profiles'] = profiles
    return GridSearch({'id': 'SEG', 'settings': s}, 'inhouse', LIMITS)


# --- GridSearch -----------------------------------------------------------------------
g = grid(['C1_base', 'C6_all'])
check(g.repeats == 4, f'repeats {g.repeats}, expected 2 points x 2 profiles')
cmds = list(g.commands())
check(all(c.get('dut_comp_profile') in ('C1_base', 'C6_all') for c in cmds),
      'every command must carry a profile')
flags = []
for c in cmds:
    f = c.get('log_flag')
    if f and (not flags or flags[-1] != f):
        flags.append(f)
check(flags == ['SEG-RUN0-SETPOINT0-C1_base', 'SEG-RUN0-SETPOINT0-C6_all',
                'SEG-RUN0-SETPOINT1-C1_base', 'SEG-RUN0-SETPOINT1-C6_all'], f'log flags {flags}')
for f in flags:
    span = [c for c in cmds if c.get('log_flag') == f]
    check(all(c['dut_comp_profile'] == f.rsplit('-', 1)[1] for c in span), f'profile mismatch in {f}')
plain = list(grid(None, torques=(0.25,)).commands())
check(all('dut_comp_profile' not in c for c in plain), 'no profiles -> no stamp')
check(any(c.get('log_flag') == 'SEG-RUN0-SETPOINT0' for c in plain), 'plain flags unchanged')
for bad, why in ((['C9_nope'], 'unknown profile'), (['C1_base'], 'over max_output_nm')):
    try:
        grid(bad, torques=(12.0,) if why == 'unknown profile' else (31.0,))
        check(False, f'{why} should be refused')
    except AssertionError:
        pass
try:
    grid(None, torques=(12.0,))
    check(False, 'without profiles, 12 must still be refused against the DUT torque limit')
except AssertionError:
    pass

# --- Controller._comp_step ------------------------------------------------------------
with open(f'{dyno_paths.dyno_config_directory}/inhouse_dyno_config.yaml') as f:
    rig = yaml.safe_load(f)
dut_params = next(s['params'] for s in rig['expected_slave_layout'] if s['name'] == 'DUT')


class Dev(types.SimpleNamespace):
    pass


ctl = Controller.__new__(Controller)
ctl._comp_cfg = compile_dut_compensation(rig)
ctl._comp = None
ctl._comp_was_active = False
ctl._comp_dut_command = 0.0
ctl._comp_warned = False
ctl._load_only = False
ctl.process_data_cycle_time_us = 1000
ctl._ff = {'mode': 'fixed', 'ratio': 0.8, 'max_nm': 30}
ctl._ff_signs = {'dut_push': 1, 'load_push': -1, 'cross': 1}
ctl.devices = Dev(DUT=Dev(mode='torque', switching_modes=False, params=dut_params, position=0.0),
                  LOAD=Dev(mode='velocity', torque_limit=30.0))
ctl._comp_clear_log()

ctl.current_cmd = {'input_command': 0.25, 'output_command': 0.1}
check(ctl._comp_step() is None, 'no profile -> inactive')
check(math.isnan(ctl.dut_comp_target), 'inactive logs NaN')

kt = dut_params['motor_params']['kt']
ctl.current_cmd = {'input_command': 12.0, 'output_command': 0.1, 'dut_comp_profile': 'C1_base'}
th = 0.0
for _ in range(200):          # let the speed estimate settle at 0.1 rad/s output
    th += 2.6e-3
    ctl.devices.DUT.position = th
    tau = ctl._comp_step()
I = ctl.dut_comp_current
check(tau == 12.0, f'returned target {tau}')
check(20 < I < 26, f'12 Nm forward at +0.1 rad/s should need ~23 A, got {I:.2f}')
check(abs(ctl._comp_dut_command - kt * I) < 1e-3 * kt * abs(I), 'command = kt x I (k_tanh negligible)')
ff = ctl._comp_load_feedforward(12.0, 0.1)
check(abs(ff - 12.0) < 1e-9, f'LOAD ff should be +12 (ratio 1, cross +1), got {ff}')
print(f'12 Nm forward, C1_base: {I:.2f} A -> DUT command {ctl._comp_dut_command:.4f} Nm at kt {kt}')

ctl.devices.DUT.switching_modes = True
check(ctl._comp_step() is None, 'mode switch in flight -> inactive')
ctl.devices.DUT.switching_modes = False

ctl.current_cmd = {'input_command': 12.0, 'output_command': 0.1, 'dut_comp_profile': 'nope'}
check(ctl._comp_step() == 0.0 and ctl._comp_dut_command == 0.0, 'unknown profile -> zero, no raise')

print('PASS' if ok else 'FAILED')
raise SystemExit(0 if ok else 1)
