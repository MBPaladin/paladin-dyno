"""Logic-only exercise of the end-of-test brake and logging tail (`post_test:`).

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/post_test_logic_test.py

No bus, no drives, no sleeps -- a stubbed Controller and a hand-wound clock, so
it runs in under a second and covers the cases the fake bus cannot reach on
demand: every sign-flag combination, all three drive modes, the settle timer,
the timeout, NaN feedback, the polarity abort, torque clipping, shutdown
mid-brake, and every ending that must NOT brake. dyno/sim/post_test_brake_test.py
is the end-to-end counterpart on the fake bus.
"""
import math
import os
import sys
import time
import types

os.environ.setdefault('DYNO_SIM', 'inhouse_archimedes')

from dyno.src.dyno_controller import Controller  # noqa: E402

ok = True
def check(cond, msg):
    global ok
    if not cond:
        ok = False; print('FAIL:', msg)

POST = {'log_tail_s': 2.0,
        'brake': {'enabled': True, 'torque_nm': 4.0,
                  'velocity_source': 'devices.LOAD.velocity',
                  'stop_velocity_rad_s': 0.05, 'settle_s': 0.05,
                  'min_velocity_rad_s': 0.05, 'timeout_s': 0.60,
                  'verify_s': 0.010, 'verify_rise_frac': 0.10}}

def stub(post=POST, flip_torque=False, flip_dir=True, mode='torque',
         v_out=4.0, v_in=175.0, torque_limit=15.0):
    c = Controller.__new__(Controller)
    c.mode = 'inhouse_archimedes'
    c.dyno_params = {}
    c._load_only = False
    c.shutdown = False
    c.time = 1.0
    c._stop_reason = None
    c._test_active = True
    c._post_test = None
    c.test_definition = None
    c._jog = None
    c._tare_state = None
    c._window = None
    c._window_centre = None
    c._window_message = None
    c._safe_default_command = {'input_mode': 'torque', 'output_mode': 'torque',
                               'input_command': 0, 'output_command': 0}
    c.devices = types.SimpleNamespace(
        LOAD=types.SimpleNamespace(position=0.0, velocity=v_out, fault=False,
                                   sw_enable=True, mode='position',
                                   switching_modes=False,
                                   command_operating_mode=lambda m: None),
        # position 10.0 with a command-frame origin of 1.9: the shaft reads
        # 10.0 rad from bring-up, but the command that HOLDS it there is 8.1,
        # because position commands are relative to the position-mode entry
        # point (AKD.position_command_frame). The two are deliberately
        # different here -- they were on the rig too (1.4-1.9 rad at the input),
        # and anything that holds the shaft by commanding `position` steps it
        # by the gap instead.
        DUT=types.SimpleNamespace(position=10.0, position_cmd_origin=1.9,
                                  position_command_frame=8.1,
                                  position_command=math.nan,
                                  velocity=v_in, fault=False,
                                  sw_enable=True, mode=mode,
                                  switching_modes=False,
                                  torque_limit=torque_limit,
                                  flip_torque_sign=flip_torque,
                                  flip_direction_sign=flip_dir,
                                  command_operating_mode=lambda m: None))
    c._post = c._compile_post_test(post)
    return c

# --- polarity, over all four flag combinations -------------------------------
# A positive torque command accelerates the REPORTED velocity in
# sign(cmd) * polarity. Current goes through flip_torque_sign, velocity through
# flip_direction_sign, so polarity is +1 only when the two agree.
print('--- polarity from the flag pair ---')
for ft, fd, want in [(False, False, +1), (True, True, +1),
                     (False, True, -1), (True, False, -1)]:
    got = stub(flip_torque=ft, flip_dir=fd)._brake_polarity()
    print(f'  flip_torque={ft!s:5} flip_direction={fd!s:5} -> polarity {got:+d}')
    check(got == want, f'polarity for ({ft},{fd}) was {got}, expected {want}')

# The rig's DUT today, and after the fix Nathan is considering.
c = stub(flip_torque=False, flip_dir=True)
check(c._brake_polarity() == -1, 'todays DUT config should give polarity -1')
# Commanding +0.5 Nm logged -192 rad/s in the coastdowns: acceleration direction
# is sign(cmd)*polarity = -1. Consistent.

# --- torque-mode brake: sign opposes motion ----------------------------------
print('--- torque-mode brake command sign ---')
for v_in, ft, fd in [(175.0, False, True), (-175.0, False, True),
                     (175.0, True, True), (-175.0, True, True)]:
    c = stub(flip_torque=ft, flip_dir=fd, v_in=v_in)
    c._stop_test({'kind': 'position_window', 'detail': 'x'})
    cmd = c.current_cmd = c._post_test_step()
    pol = c._brake_polarity()
    # The physical test: the command must accelerate the reported velocity
    # AGAINST its current sign.
    accel_dir = math.copysign(1, cmd['input_command']) * pol
    print(f'  v_in={v_in:+7.1f} flags=({ft},{fd}) -> cmd {cmd["input_command"]:+.2f} Nm, '
          f'accelerates {accel_dir:+.0f}')
    check(accel_dir == -math.copysign(1, v_in),
          f'brake command does not oppose motion for v_in={v_in}, flags=({ft},{fd})')
    check(abs(cmd['input_command']) == 4.0, 'brake did not use torque_nm')
    check(c.devices.DUT.sw_enable, 'input drive was disabled during the brake')
    check(not c.devices.LOAD.sw_enable, 'absorber was left enabled during the brake')

# --- velocity / position mode are sign-free ----------------------------------
print('--- velocity and position mode brakes ---')
c = stub(mode='velocity')
c._stop_test({'kind': 'position_window'})
cmd = c._post_test_step()
print(f'  velocity mode -> {cmd["input_mode"]} command {cmd["input_command"]}')
check(cmd['input_mode'] == 'velocity' and cmd['input_command'] == 0.0,
      'velocity-mode brake did not command zero speed')
c = stub(mode='position')
c._stop_test({'kind': 'position_window'})
cmd = c._post_test_step()
print(f'  position mode -> {cmd["input_mode"]} command {cmd["input_command"]}')
check(cmd['input_mode'] == 'position' and cmd['input_command'] == 8.1,
      'position-mode brake did not hold the shaft position')
check(cmd['input_command'] != 10.0,
      'position-mode brake held the FEEDBACK position, not the command-frame '
      'one -- that is a step of the frame offset, which is what put 20 A into '
      'the input drive at the start of every brake on 2026-09-17')
check(c.devices.DUT.mode == 'position', 'brake switched the drive mode (it must not)')

# --- reaching standstill ends the brake and starts the tail ------------------
print('--- stop detection, settle and tail handover ---')
c = stub()
c._stop_test({'kind': 'position_window', 'detail': 'x'})
check(c._post_test['stage'] == 'brake', 'did not enter the brake stage')
t0 = c._post_test['brake']['started']
c.devices.LOAD.velocity = 0.01   # stopped
c.devices.DUT.velocity = 0.4
c._brake_step(c._post_test, t0 + 0.001)      # first sample below -> arms settle
check(c._post_test['brake']['below_since'] is not None, 'settle timer did not arm')
check(c._brake_step(c._post_test, t0 + 0.02) is not None,
      'brake ended before settle_s elapsed')
check(c._brake_step(c._post_test, t0 + 0.06) is None, 'brake did not end after settle_s')
c._end_brake(c._post_test, t0 + 0.06)
print(f"  outcome={c._stop_reason['brake']['outcome']} "
      f"after {c._stop_reason['brake'].get('stopped_after_s')} s")
check(c._stop_reason['brake']['outcome'] == 'stopped', 'outcome not recorded as stopped')
check(not c.devices.DUT.sw_enable, 'input drive still enabled after the brake')
check(c._post_test['stage'] == 'tail', 'did not hand over to the tail')
# Tail is measured from the END of the brake.
check(c._post_test['tail_ends'] > t0 + 0.06 + 1.9, 'tail not restarted from the brake end')
c._post_test_step()
check(c._post_test is not None, 'tail ended immediately')
c._post_test['tail_ends'] = time.perf_counter() - 1
c._post_test_step()
check(c._post_test is None, 'tail did not end at tail_ends')

# --- the rig's chatter case: sign(v_in) is noise once the shaft is parked ----
# Reproduces brake_checks/torque_mode_brake_check (2026-09-17): the output was
# parked (0.08 deg of movement in half a second) but the bang-bang sign flipped
# 77 times in 500 ms and rocked the driveline at +/-0.16 rad/s at the output --
# over stop_velocity -- so the settle window never closed and the brake timed
# out. The fix is the settling sub-phase: stop commanding once the output has
# been seen at rest.
print('--- chatter at standstill (the rig timeout) ---')
c = stub()
c._stop_test({'kind': 'position_window'})
t0 = c._post_test['brake']['started']
# Follow the rig's actual sequence. It slowed cleanly for 90 ms, and the output
# first read below stop_velocity at that point (-0.0296 rad/s); only AFTER that
# did the sign chatter build. So the shaft has to pass through rest first --
# starting mid-chatter would test a state the rig never entered.
c.devices.LOAD.velocity, c.devices.DUT.velocity = 4.26, 181.0
c._brake_step(c._post_test, t0 + 0.001)
check(c._post_test['brake']['phase'] == 'slowing', 'did not start in slowing')
c.devices.LOAD.velocity, c.devices.DUT.velocity = 0.0296, 0.38
c._brake_step(c._post_test, t0 + 0.090)
cmds = []
# Now the driveline rocks: output +/-0.16 rad/s (over the 0.05 stop threshold,
# which is what defeated the settle window), input sign alternating.
for k in range(400):
    c.devices.LOAD.velocity = 0.16 * (1 if k % 20 < 10 else -1)
    c.devices.DUT.velocity = 4.5 * (1 if k % 20 < 10 else -1)
    out = c._brake_step(c._post_test, t0 + 0.091 + 0.001 * k)
    if out is None:
        break
    cmds.append(out['input_command'])
flips = sum(1 for a, b in zip(cmds, cmds[1:])
            if a and b and (a > 0) != (b > 0))
print(f'  {len(cmds)} cycles commanded, {flips} sign flips, '
      f'phase={c._post_test["brake"]["phase"]}, '
      f'max |cmd|={max(abs(x) for x in cmds):.2f} Nm')
check(flips == 0,
      f'the brake still chatters: {flips} sign flips at standstill -- this is '
      f'the rig failure in brake_checks/torque_mode_brake_check')
check(c._post_test['brake']['phase'] == 'settling',
      'the brake never entered the settling sub-phase')
check(all(x == 0 for x in cmds),
      'the brake kept commanding torque after the output was seen at rest')

# ...but real motion resuming DOES push it back to slowing and brake again.
c.devices.LOAD.velocity = 0.05 * 8 + 0.01   # over _BRAKE_RESUME_FACTOR
c.devices.DUT.velocity = 60.0
out = c._brake_step(c._post_test, t0 + 0.5)
check(c._post_test['brake']['phase'] == 'slowing',
      'real motion did not push the brake back to slowing')
check(out is not None and abs(out['input_command']) == 4.0,
      f'the brake did not resume braking on real motion: {out}')
print('  chatter -> zero command; genuine motion -> braking resumes')

# and the settling time is reported separately from the settle window
c2 = stub()
c2._stop_test({'kind': 'position_window'})
t1 = c2._post_test['brake']['started']
c2.devices.LOAD.velocity = 0.01
c2.devices.DUT.velocity = 0.4
c2._brake_step(c2._post_test, t1 + 0.090)
c2._brake_step(c2._post_test, t1 + 0.145)
c2._end_brake(c2._post_test, t1 + 0.145)
rec = c2._stop_reason['brake']
print(f'  reported: slowing_s={rec.get("slowing_s")}, '
      f'stopped_after_s={rec.get("stopped_after_s")}')
check(rec.get('slowing_s') == 0.09,
      f'slowing_s not reported as the time to reach rest: {rec}')

# --- a bad settle resets ------------------------------------------------------
c = stub()
c._stop_test({'kind': 'position_window'})
t0 = c._post_test['brake']['started']
c.devices.LOAD.velocity = 0.01
c._brake_step(c._post_test, t0 + 0.001)
c.devices.LOAD.velocity = 3.0     # moving again
c._brake_step(c._post_test, t0 + 0.01)
check(c._post_test['brake']['below_since'] is None,
      'settle timer survived the shaft moving again')

# --- timeout ------------------------------------------------------------------
print('--- timeout, NaN, abort and the no-brake endings ---')
c = stub()
c._stop_test({'kind': 'position_window'})
t0 = c._post_test['brake']['started']
check(c._brake_step(c._post_test, t0 + 0.7) is None, 'brake ignored timeout_s')
check('timed out' in c._post_test['brake']['outcome'], 'timeout not recorded')

# --- NaN output velocity still brakes (a failed position source IS the trip) --
c = stub(v_out=math.nan)
c._stop_test({'kind': 'position_window'})
check(c._post_test is not None and c._post_test['stage'] == 'brake',
      'a NaN output velocity skipped the brake')

# --- NaN input feedback: skip the brake, keep the tail -----------------------
c = stub(v_in=math.nan)
c._stop_test({'kind': 'position_window'})
check(c._post_test is not None and c._post_test['stage'] == 'tail',
      'a NaN input velocity still entered the brake -- the speed-up backstop '
      'would compare against NaN and never fire')
check(not c.devices.DUT.sw_enable, 'NaN-skip left the input drive enabled')
c = stub(mode='position')
# Both, because on a real AKD they are the same reading: position_command_frame
# is position minus the mode-entry origin, so unreadable feedback makes both NaN.
c.devices.DUT.position = math.nan
c.devices.DUT.position_command_frame = math.nan
c._stop_test({'kind': 'position_window'})
check(c._post_test['stage'] == 'tail', 'a NaN hold position still braked in position mode')
print('  NaN input feedback -> brake skipped, tail kept, drives off')

# --- wrong polarity aborts rather than driving harder ------------------------
c = stub()
c._stop_test({'kind': 'position_window'})
t0 = c._post_test['brake']['started']
c.devices.DUT.velocity = 175.0 * 1.5     # sped up under braking torque
out = c._brake_step(c._post_test, t0 + 0.011)
check(out is None, 'brake kept commanding torque after the shaft sped up')
check('aborted' in c._post_test['brake']['outcome'],
      f'abort not recorded: {c._post_test["brake"]["outcome"]}')
# ...but not before verify_s has elapsed (velocity is noisy at the start).
c = stub()
c._stop_test({'kind': 'position_window'})
t0 = c._post_test['brake']['started']
c.devices.DUT.velocity = 175.0 * 1.5
check(c._brake_step(c._post_test, t0 + 0.005) is not None,
      'brake aborted before verify_s elapsed')

# --- already slow: skip the brake, go straight to the tail -------------------
c = stub(v_out=0.01)
c._stop_test({'kind': 'position_window'})
check(c._post_test is not None and c._post_test['stage'] == 'tail',
      'an already-stopped shaft still entered the brake stage')
check(not c.devices.DUT.sw_enable, 'drives left enabled for a tail-only phase')

# --- drive fault: no brake (the drive will not take a command), tail still ---
c = stub()
c._stop_test({'kind': 'drive_fault', 'drive': 'DUT'})
check(c._post_test is not None and c._post_test['stage'] == 'tail',
      'a drive fault tried to brake')
check(c._post_test['tail_s'] == 2.0, 'a drive fault lost its logging tail')
check(not c.devices.DUT.sw_enable and not c.devices.LOAD.sw_enable,
      'drive fault left a drive enabled')
print('  drive_fault -> tail only, drives off, tail kept (this is the E-Stop path)')

# --- shutdown: neither ---
c = stub()
c._stop_test({'kind': 'shutdown', 'detail': 'rig shutdown requested'})
check(c._post_test is None, 'shutdown held a wind-down open')
check(not c.devices.DUT.sw_enable, 'shutdown left the input drive enabled')

# --- shutdown mid-brake drops everything immediately -------------------------
c = stub()
c._stop_test({'kind': 'position_window'})
c.shutdown = True
c._post_test_step()
check(c._post_test is None, 'shutdown mid-brake did not end the wind-down')
check(not c.devices.DUT.sw_enable and not c.devices.LOAD.sw_enable,
      'shutdown mid-brake left a drive enabled')
check('shutdown' in (c._stop_reason['brake']['outcome'] or ''),
      'shutdown mid-brake not recorded in the reason')

# --- an idle Stop press must not open a log ----------------------------------
c = stub()
c._test_active = False
c._stop_test({'kind': 'operator', 'detail': 'stopped from the GUI by the operator'})
check(c._post_test is None,
      'Stop with no run in progress started a tail (would open a log of idle)')
print('  Stop with the rig idle -> no phase, no log')

# --- operator Stop during a run DOES brake -----------------------------------
c = stub()
c._stop_test({'kind': 'operator', 'detail': 'stopped from the GUI by the operator'})
check(c._post_test is not None and c._post_test['stage'] == 'brake',
      'the Stop button did not brake a running test')

# --- no stale test command survives into the brake --------------------------
print('--- stale command and position-mode handover ---')
c = stub()
c._safe_default_command['input_command'] = 12.0   # what the test was commanding
c._stop_test({'kind': 'position_window'})
check(c._safe_default_command['input_command'] == 0,
      'the last test command survived into the brake as the standing default')
# the fall-through path: torque mode, velocity exactly zero, no direction yet
c.devices.DUT.velocity = 0.0
cmd = c._brake_step(c._post_test, c._post_test['brake']['started'] + 0.001)
check(cmd is not None and cmd['input_command'] == 0,
      f'zero-velocity fall-through commanded {cmd and cmd["input_command"]}, not 0')

# --- position-mode brake hands over in torque mode, not "go to position 0" ---
modes = []
c = stub(mode='position')
c.devices.DUT.command_operating_mode = lambda m: modes.append(m)
c._stop_test({'kind': 'position_window'})
check(c._post_test['brake']['hold_position'] == 8.1, 'hold position not captured')
c._end_brake(c._post_test, time.perf_counter())
check('torque' in modes,
      'position-mode brake left a 0 command standing without leaving position '
      'mode -- that commands position 0 on a shaft resting at 10 rad')
check(c._safe_default_command['input_command'] == 0, 'handover did not zero the command')
print(f'  stale command cleared; position-mode handover asked for {modes}')

# --- the zero must not REACH the drive while it is still in position mode ----
# Asking for torque mode is not the same as being in it: an AKD walks back
# through Ready to Switch On to change mode, which took 18 cycles on the rig.
# Every run in logs/archimedes/gbx_1p2p0/troubleshooting_initial ended with
# dut_position_command stepping to 0 with dut_mode still 2, 21 A, and the input
# at 300 rad/s within 6 ms; in med_speed_5 the train then coasted into the
# bumper. _dut_command_for_mode is what stands between the two.
print('--- zero command during an in-flight mode switch ---')
c = stub(mode='position')
c._stop_test({'kind': 'operator', 'detail': 'stopped from the GUI by the operator'})
c._end_brake(c._post_test, time.perf_counter())
# The drive has been ASKED for torque mode but has not arrived yet.
c.devices.DUT.mode = 'position'
c.devices.DUT.switching_modes = True
sent = c._dut_command_for_mode(c._safe_default_command)
print(f'  safe default says {c._safe_default_command["input_mode"]} '
      f'{c._safe_default_command["input_command"]}, drive is in '
      f'{c.devices.DUT.mode} -> sent {sent}')
check(sent == 8.1,
      f'a command of {sent} reached a drive still in position mode; 0 there '
      'means "travel to the position-mode entry point", not "no torque"')

# Once the drive really is in torque mode, zero passes through untouched.
c.devices.DUT.mode = 'torque'
c.devices.DUT.switching_modes = False
check(c._dut_command_for_mode(c._safe_default_command) == 0,
      'torque-mode zero was rewritten; it must pass through')

# A position command written FOR position mode is never second-guessed.
c.devices.DUT.mode = 'position'
check(c._dut_command_for_mode({'input_mode': 'position', 'input_command': 42.0}) == 42.0,
      'a genuine position command was overridden')

# Unreadable feedback: fall back to the last position command, never to 0.
c.devices.DUT.position_command_frame = math.nan
c.devices.DUT.position_command = 7.25
check(c._dut_command_for_mode(c._safe_default_command) == 7.25,
      'NaN position feedback fell through to a raw 0 in position mode')

# --- brake blocks, tail yields -----------------------------------------------
print('--- refusals: the brake blocks, the tail yields ---')
c = stub()
c._stop_test({'kind': 'position_window'})
check('braking' in (c._post_test_refusal() or ''), 'no refusal while braking')
c._yield_post_test_tail()
check(c._post_test is not None and c._post_test['stage'] == 'brake',
      'an operator command cut the BRAKE short -- it must not')
c._post_test['stage'] = 'tail'
check(c._post_test_refusal() is None, 'the tail refused an operator command')
c._yield_post_test_tail()
check(c._post_test is None, 'the tail did not yield to an operator command')
check(c._stop_reason is not None,
      'yielding dropped the stop reason -- the closing log would lose it')
print('  brake -> refuses and survives; tail -> yields, reason kept')
c._post_test = None
check(c._post_test_refusal() is None, 'refusal outlived the wind-down')

# --- pinned polarity overrides the derivation --------------------------------
print('--- polarity override ---')
for pin, want in [(1, 1), (-1, -1)]:
    post = {'log_tail_s': 2.0, 'brake': dict(POST['brake'], polarity=pin)}
    # (F, T) would derive -1; the pin must win either way.
    c = stub(post=post, flip_torque=False, flip_dir=True)
    check(c._brake_polarity() == want, f'polarity: {pin} was not honoured')
print('  polarity: 1 / -1 pin; auto derives')
try:
    cc = Controller.__new__(Controller); cc.mode = 'x'; cc._load_only = False
    cc._compile_post_test({'brake': dict(POST['brake'], polarity='backwards')})
    check(False, 'a bad polarity was accepted')
except ValueError as e:
    print(f'  bad polarity refused at compile: {e}')

# --- torque clipped to the drive limit ---------------------------------------
c = stub(torque_limit=2.5)
c._stop_test({'kind': 'position_window'})
check(c._post_test['brake']['torque_nm'] == 2.5, 'brake torque not clipped to the drive limit')
check(c._post_test['brake']['clipped'], 'clipping not flagged')
c._end_brake(c._post_test, time.perf_counter())
check('clipped' in c._stop_reason['brake'].get('note', ''), 'clipping not recorded in the reason')

# --- config off / absent is the old behaviour --------------------------------
print('--- config off ---')
for post in (None, {}, {'log_tail_s': 0.0}, {'brake': {'enabled': False}}):
    c = stub(post=post)
    c._stop_test({'kind': 'position_window'})
    check(c._post_test is None, f'post_test={post!r} still created a phase')
    check(not c.devices.DUT.sw_enable and not c.devices.LOAD.sw_enable,
          f'post_test={post!r} left a drive enabled')
print('  absent / disabled config -> drives off immediately, log closes at the trip')

# --- tail with the brake off ---------------------------------------------------
c = stub(post={'log_tail_s': 2.0})
c._stop_test({'kind': 'position_window'})
check(c._post_test is not None and c._post_test['stage'] == 'tail',
      'tail-only config did not produce a tail')
check(not c.devices.DUT.sw_enable, 'tail-only config left the input drive enabled')

# --- load_only refuses a brake outright ---------------------------------------
try:
    c = Controller.__new__(Controller)
    c.mode = 'x'; c._load_only = True
    c._compile_post_test(POST)
    check(False, 'post_test.brake accepted load_only')
except ValueError as e:
    print(f'  load_only + brake -> refused at compile: {e}')

print('\nPASS' if ok else '\nFAILURES ABOVE')
sys.exit(0 if ok else 1)
