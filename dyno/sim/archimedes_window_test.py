"""Validation of the output position window for the Archimedes setup: safety
logic, declare centre, jog, touch-off, the arming gate and the window trip,
against the fake bus with rubber endstops in the plant.

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/archimedes_window_test.py [mode]
(mode defaults to inhouse_archimedes; sim endstops come from sim_params.yaml)

Two controller runs:
  A. Bumpers where the config expects them: touch-off passes, Start is gated,
     a test that leaves the window trips, and dropped EtherCAT frames do NOT
     stop a running test.
  B. No bumpers: touch-off fails at max excursion, jog stops there.
"""
import copy
import json
import math
import multiprocessing
import os
import sys
import time
import types

MODE = sys.argv[1] if len(sys.argv) > 1 else 'inhouse_archimedes'
os.environ['DYNO_SIM'] = MODE
# Frame-loss schedule for run A, step A9. Both regimes are injected -- isolated
# single-cycle drops and one dense burst -- and NEITHER may stop the test: this
# config carries no wkc_error safety, so the check here is against a nuisance
# stop, not for a trip.
WKC_DROP_FROM_S = 150       # isolated drops, 1 s apart
WKC_ISOLATED_N = 15
WKC_BURST_S = 166.0         # start of the dense burst
WKC_BURST_CYCLES = 100      # 100 ms at 1 kHz

import yaml  # noqa: E402

from deployment import dyno_paths  # noqa: E402
from dyno.src.config_utils import augment_log_keys  # noqa: E402
from dyno.src.dyno_controller import Controller  # noqa: E402

ok = True


def fail(msg):
    global ok
    ok = False
    print(f'FAIL: {msg}')


def check(cond, msg):
    if not cond:
        fail(msg)
    return cond


with open(f'{dyno_paths.dyno_config_directory}/{MODE}_dyno_config.yaml') as f:
    CFG = yaml.safe_load(f)
# deepcopy, NOT a yaml round-trip: augment_log_keys mutates what it is given, so
# it needs a copy -- but safe_dump sorts dict keys, which reorders `sensors:` and
# so reorders every auto-appended sensor column. The declared log_keys list is
# unaffected (it is a list), which is why this hid for so long: positions 0-20
# stay correct and only the sensor channels at the tail silently swap places.
LOG_KEYS = augment_log_keys(copy.deepcopy(CFG))
IDX = {k: i for i, k in enumerate(LOG_KEYS)}

# What load_position should read at power-up, derived rather than hardcoded so
# this tracks both the sim's starting angle and the config's encoder handedness.
# The sim plant starts the shaft at initial_theta_rad and its encoder behavior
# presents it negated on the LOAD channel; flip_direction_sign then negates it
# back (devices.py:1129). Getting this from the config is the point: the flag is
# a real bench decision (log finding 16), not a sim detail.
with open(f'{os.path.dirname(os.path.abspath(__file__))}/sim_params.yaml') as f:
    _SIM_PARAMS = yaml.safe_load(f)
_MODE_PARAMS = (_SIM_PARAMS.get('modes') or {}).get(MODE, {})
_LOAD_PARAMS = next(e.get('params', {}) for e in CFG['expected_slave_layout']
                    if e.get('name') == 'LOAD')
EXPECTED_POWER_UP = -float(_MODE_PARAMS.get(
    'initial_theta_rad', _SIM_PARAMS.get('initial_theta_rad', 0.0)))
if _LOAD_PARAMS.get('flip_direction_sign'):
    EXPECTED_POWER_UP = -EXPECTED_POWER_UP


# --- 1. Logic, no bus --------------------------------------------------------
print('--- 1. Safety and window logic (no bus) ---')


def stub(window_cfg=None, safeties=()):
    c = Controller.__new__(Controller)
    c.mode = MODE
    c.dyno_params = {}
    c._load_only = False
    c.time = 1.0
    c.devices = types.SimpleNamespace(
        LOAD=types.SimpleNamespace(position=0.0, velocity=0.0, fault=False),
        DUT=types.SimpleNamespace(position=0.0, fault=False))
    # Entries may be given as 4-tuples (no trip_samples); default to 1, which
    # is the instantaneous behaviour every check had before debounce existed.
    c._safety_checks = [tuple(s) if len(s) == 5 else tuple(s) + (1,)
                        for s in safeties]
    c._safety_streaks = {}
    c._window = c._compile_window(window_cfg)
    c._window_centre = None
    c._touch_off = None
    c._jog = None
    c._input_sign = None
    c._input_ratio = None
    return c


nan_value = [math.nan]
c = stub(safeties=[('t', lambda s: nan_value[0], 10.0, True)])
check(c._safety_trigger() is not None, 'nan_trips=True did not trip on NaN')
c = stub(safeties=[('t', lambda s: nan_value[0], 10.0, False)])
check(c._safety_trigger() is None, 'nan_trips=False tripped on NaN (should keep old behaviour)')
print('  NaN trips only where nan_trips is set')

# trip_samples: N consecutive breaches required, and any in-range read resets.
val = [0.0]
c = stub(safeties=[('t', lambda s: val[0], 0.0, False, 3)])
val[0] = 2.0
check(c._safety_trigger() is None and c._safety_trigger() is None,
      'trip_samples=3 tripped before 3 consecutive breaches')
check(c._safety_trigger() is not None, 'trip_samples=3 did not trip on the 3rd')
c = stub(safeties=[('t', lambda s: val[0], 0.0, False, 3)])
for _ in range(20):  # breach, clear, breach, clear ... never 3 in a row
    val[0] = 2.0
    c._safety_trigger()
    val[0] = 0.0
    check(c._safety_trigger() is None, 'isolated breaches accumulated across resets')
print('  trip_samples needs N in a row and resets on any good sample')

c = stub(CFG['position_window'])
c._window_centre = 0.2
for pos, should_trip in ((0.2 + 1.49, False), (0.2 - 1.49, False),
                         (0.2 + 1.51, True), (0.2 - 1.51, True), (math.nan, True)):
    c.devices.LOAD.position = pos
    tripped = c._safety_trigger()
    check(bool(tripped) == should_trip,
          f'window at position {pos}: tripped={bool(tripped)}, expected {should_trip}')
c._window_centre = None
c.devices.LOAD.position = 0.0
check(c._safety_trigger() is not None, 'window with no centre did not trip')
check(c._window_arming_refusal() == 'no centre declared this session',
      f'arming refusal without centre: {c._window_arming_refusal()!r}')
print('  window trips outside +/-half_window, on NaN, and with no centre')

c = stub(dict(CFG['position_window'], stop_decel_rad_s2=10.0))
c._window_centre = 0.0
for pos, vel, should_trip, why in (
        (1.2, 3.0, True, 'outward, stops at 1.2 + 9/20 = 1.65'),
        (1.2, -3.0, False, 'inward'),
        (-1.2, -3.0, True, 'outward on the - side'),
        (-1.2, 3.0, False, 'inward on the - side'),
        (1.2, 2.0, False, 'outward, stops at 1.2 + 4/20 = 1.40'),
        (1.2, math.nan, True, 'velocity NaN')):
    c.devices.LOAD.position, c.devices.LOAD.velocity = pos, vel
    tripped = c._safety_trigger()
    check(bool(tripped) == should_trip,
          f'stop-distance trip at pos {pos} vel {vel} ({why}): tripped={bool(tripped)}')
print(f'  stop-distance trip: {c._safety_trigger() and c._safety_trigger()["detail"]}')
c = stub(dict(CFG['position_window'], stop_decel_rad_s2=0.0))
c._window_centre = 0.0
c.devices.LOAD.position, c.devices.LOAD.velocity = 1.2, 100.0
check(c._safety_trigger() is None, 'stop_decel 0 should disable the early trip')
try:
    # stop_decel pinned, not inherited: the rig config now ships
    # stop_decel_rad_s2: 0 (early trip off), and at 0 there is nothing for
    # velocity_source to feed, so inheriting it made this case vacuous.
    stub(dict(CFG['position_window'], stop_decel_rad_s2=10.0, velocity_source=None))
    fail('stop_decel without velocity_source was accepted')
except ValueError:
    pass
# ...and with the early trip off it is genuinely not needed.
stub(dict(CFG['position_window'], stop_decel_rad_s2=0.0, velocity_source=None))

c = stub(CFG['position_window'])
TO = CFG['position_window']['touch_off']
half = TO['expected_half_span_rad']
for contacts, max_rc, expect_shift in (([half + 0.05, -half + 0.05], 0.2, 0.05),
                                       ([half + 0.25, -half + 0.25], 0.2, None),
                                       ([half + 0.25, -half + 0.25], 0.3, 0.25),
                                       ([half + 0.5, -half + 0.05], 0.2, None)):  # span out
    c._window['max_recentre'] = max_rc
    c._window_centre = 1.0
    j = {'contacts': contacts, 'problems': [], 'run_centre': 1.0, 'shift': None}
    shift = c._recentre(j)
    ok_shift = (j['shift'] is None if expect_shift is None
                else abs(j['shift'] - expect_shift) < 1e-9)
    check(ok_shift and abs(c._window_centre - 1.0 - shift) < 1e-9,
          f'recentre {contacts} max {max_rc}: shift {j["shift"]}, centre {c._window_centre}')
print('  recentre moves centre to the midpoint, refuses a big shift or bad span')

bad = dict(CFG['position_window'], half_window_rad=1.7)
try:
    stub(bad)
    fail('half_window beyond the bumpers was accepted')
except ValueError as e:
    print(f'  rejects a window past the bumpers: {e}')
check(stub(dict(CFG['position_window'], enabled=False))._window is None,
      'enabled: false did not turn the window off')


# --- 2. Controller on the fake bus -------------------------------------------
class Rig:
    def __init__(self, env):
        os.environ.pop('DYNO_SIM_PARAMS', None)
        os.environ.pop('DYNO_SIM_WKC_DROP', None)
        os.environ.update(env)
        self.tq = multiprocessing.Queue()
        self.cq = multiprocessing.Queue()
        self.proc = multiprocessing.Process(target=Controller,
                                            args=[self.tq, self.cq, MODE])
        self.proc.start()
        self.state = {}
        self.log = {}
        self.stop_reasons = []
        self.samples = 0
        self.last = None
        # post_test wind-down (brake, then logging tail): the stages seen in
        # order, and how many samples were still being LOGGED while it ran. The
        # tail's whole point is that those samples exist, so counting them is
        # the only check that it really happened.
        self.post_stages = []
        self.post_logged = 0
        self._last_stage = None
        # Stands in for the GUI's held-button keepalive (see gui.__jog_tick).
        self.holding = False
        self._last_alive = 0.0

    def reset_post_test(self):
        self.post_stages = []
        self.post_logged = 0
        self._last_stage = self.state.get('post_test')

    def send(self, *cmd):
        if cmd[0] == 'jog':
            self.holding = True
        elif cmd[0] == 'jog_stop':
            self.holding = False
        self.cq.put_nowait(list(cmd) if len(cmd) > 1 else [cmd[0], 0])

    def pump(self, seconds, until=None):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self.holding and time.time() - self._last_alive > 0.1:
                self.cq.put_nowait(['jog_alive', 0])
                self._last_alive = time.time()
            try:
                s = self.tq.get(timeout=0.5)
            except Exception:
                if not self.proc.is_alive():
                    fail('controller process died')
                    return False
                continue
            self.samples += 1
            self.last = s
            self.log = s[-2]
            self.state = s[-1]
            if self.log.get('stop_reason') and (
                    not self.stop_reasons or self.stop_reasons[-1] != self.log['stop_reason']):
                self.stop_reasons.append(self.log['stop_reason'])
            stage = self.state.get('post_test')
            if stage != self._last_stage:
                self.post_stages.append(stage)
                self._last_stage = stage
            if stage is not None and self.log.get('log'):
                self.post_logged += 1
            if until is not None and until(self):
                return True
        return False

    def value(self, key):
        return self.last[IDX[key]]

    @property
    def w(self):
        return self.state.get('window') or {}

    def close(self):
        self.send('shutdown')
        self.proc.join(timeout=15)
        if self.proc.is_alive():
            self.proc.terminate()


def arm(rig, test):
    rig.send('test_def', (test, MODE))
    if not rig.pump(20, until=lambda r: r.state.get('armed') == test.split('.')[0]):
        fail(f'{test} never armed (load_error={rig.state.get("load_error")})')


print('\n--- 2A. Bumpers in place ---')
# hardware_torque_sign: make the fake drives treat flip_torque_sign the way the
# real AKDs do (fake_pysoem._SimAKD.hw_torque_sign). Needed here because the
# post-test brake's torque-mode polarity depends on that sign, and the fake
# drive's default convention is the opposite one -- log finding 10.
rig = Rig({'DYNO_SIM_PARAMS': json.dumps({'hardware_torque_sign': True}),
           'DYNO_SIM_WKC_DROP': ','.join(
    [str(WKC_DROP_FROM_S + i) for i in range(WKC_ISOLATED_N)] +
    [f'{WKC_BURST_S + i * 0.001:.3f}' for i in range(WKC_BURST_CYCLES)])})
try:
    print('  bring-up...')
    rig.pump(6)
    positions = []
    rig.pump(2, until=lambda r: positions.append(r.value('load_position')) and False)
    step = max(positions) - min(positions)
    print(f'  A1 load_position over 2 s: {min(positions):+.4f}..{max(positions):+.4f} '
          f'(expected {EXPECTED_POWER_UP:+.1f})')
    check(step < 1e-3 and abs(positions[-1] - EXPECTED_POWER_UP) < 0.01,
          'load_position moved or is not the raw encoder angle: the DUT offset '
          'copy may still be running')

    arm(rig, 'sim_archimedes_window_trip.yaml')
    rig.send('start_test')
    rig.pump(2)
    print(f'  A2 start with no centre: {rig.w.get("message")}')
    check(not rig.state.get('test_active') and 'no centre' in (rig.w.get('message') or ''),
          'Start was not refused without a centre')

    rig.send('jog', 1)
    rig.pump(1)
    print(f'  A3 jog with no centre: {rig.w.get("message")}')
    check('declare centre first' in (rig.w.get('message') or ''), 'jog not refused without centre')

    rig.send('declare_centre')
    rig.pump(2, until=lambda r: r.w.get('centre') is not None)
    print(f'  A4 {rig.w.get("message")}')
    check(rig.w.get('centre') is not None
          and abs(rig.w['centre'] - EXPECTED_POWER_UP) < 0.01,
          f'centre not declared at the output angle: {rig.w.get("centre")}')

    rig.send('start_test')
    rig.pump(2)
    print(f'  A5 start with no touch-off: {rig.w.get("message")}')
    check(not rig.state.get('test_active') and 'touch-off' in (rig.w.get('message') or ''),
          'Start was not refused without touch-off')

    rig.send('jog', 1)
    rig.pump(1.5)
    rig.send('jog_stop')
    rig.pump(1.5, until=lambda r: r.w.get('jog') is None)
    rel_after_plus = rig.w.get('rel')
    print(f'  A6 hold jog + 1.5 s, release: rel {rel_after_plus:+.3f} rad; {rig.w.get("message")}')
    check(rel_after_plus is not None and rel_after_plus > 0.1, 'jog + did not move the output +')
    check(rig.w.get('jog') is None, 'jog did not stop on release')

    rel_before = rig.w.get('rel')
    rig.send('jog', -1, 'input')
    rig.pump(3.0)
    rig.send('jog_stop')
    rig.pump(1.5, until=lambda r: r.w.get('jog') is None)
    moved = (rig.w.get('rel') or 0) - (rel_before or 0)
    print(f'  A6b hold input jog - 3 s, release: output moved {moved:+.3f} rad, '
          f'input sign {rig.w.get("input_sign")}, ratio {rig.w.get("input_ratio")}; '
          f'{rig.w.get("message")}')
    check(abs(moved) > 0.1, 'input jog did not move the output')
    check(rig.w.get('jog') is None, 'input jog did not stop on release')
    check(rig.w.get('input_sign') in (1, -1)
          and abs(abs(rig.w.get('input_ratio') or 0) - 43) < 3,
          'input jog did not learn the output direction and ~43:1 ratio')
    check('contact' not in (rig.w.get('message') or ''),
          'input jog in free travel stopped on a false contact')

    # Declare centre off the bumper midpoint so recentring has something to do.
    rig.send('jog', 1)
    rig.pump(10, until=lambda r: (r.w.get('rel') or -9) > 0.08)
    rig.send('jog_stop')
    rig.pump(2, until=lambda r: r.w.get('jog') is None)
    rig.pump(1)
    rig.send('declare_centre')
    rig.pump(2)
    offset = rig.w['centre'] - EXPECTED_POWER_UP
    print(f'  A6c re-declared centre {offset:+.4f} rad off the bumper midpoint')

    print('  A7 touch-off (about a minute)...')
    rig.send('touch_off')
    rig.pump(2, until=lambda r: r.w.get('jog') == 'touch_off')
    rig.pump(150, until=lambda r: r.w.get('jog') is None)
    t = rig.w.get('touch_off') or {}
    print(f'     {rig.w.get("message")}')
    check(t.get('passed'), f'touch-off did not pass: {t}')
    check(t.get('drive') == 'input', f'touch-off ran on the {t.get("drive")} shaft, not input')
    for c in t.get('contact_torques') or []:
        # (torque - drag) x 43 is what the bumper took.
        print(f'     contact {c["torque_nm"]:.3f} Nm on {c["source"]}, drag '
              f'{c["drag_nm"]:.3f}: ~{43 * (c["torque_nm"] - c["drag_nm"]):.1f} Nm at the output')
    if t.get('plus_rad') is not None and t.get('minus_rad') is not None:
        print(f'     +{t["plus_rad"]:.4f} / {t["minus_rad"]:+.4f} rad, centre error '
              f'{t["centre_error_rad"]:+.4f}, back at rel {rig.w.get("rel"):+.3f}')
        check(abs(rig.w.get('rel') or 9) < 0.05, 'touch-off did not return to centre')
    print(f'     centre moved {t.get("centre_shift_rad")}, now {rig.w.get("centre")}')
    check(t.get('centre_shift_rad') is not None
          and abs(t['centre_shift_rad'] + offset) < 0.01
          and abs(rig.w['centre'] - EXPECTED_POWER_UP) < 0.01,
          'touch-off did not recentre on the bumper midpoint')
    check(not rig.w.get('refusal'), f'still refused after touch-off: {rig.w.get("refusal")}')

    print('  A7b repeat touch-off from the sensed centre...')
    rig.send('touch_off')
    rig.pump(2, until=lambda r: r.w.get('jog') == 'touch_off')
    rig.pump(150, until=lambda r: r.w.get('jog') is None)
    t2 = rig.w.get('touch_off') or {}
    print(f'     {rig.w.get("message")}')
    check(t2.get('passed') and abs(t2.get('centre_error_rad') or 9) < 0.005,
          f'repeat touch-off did not agree with the sensed centre: {t2}')
    hist = [h for h in rig.w.get('touch_off_history') or [] if h.get('midpoint_abs_rad') is not None]
    if len(hist) >= 2:
        print(f'     midpoint repeatability over {len(hist)} runs: '
              f'{max(h["midpoint_abs_rad"] for h in hist) - min(h["midpoint_abs_rad"] for h in hist):.5f} rad')

    rig.reset_post_test()
    # Wait for a NEW reason, not any: _stop_reason rides along on every idle
    # sample after a run (see _send_telemetry), and an earlier step in this run
    # has already tripped the window once -- matching on kind alone returned
    # instantly on that stale one, before this test had even started.
    n_before = len(rig.stop_reasons)
    rig.send('start_test')
    # The stop_reason no longer arrives on the sample after the trip: post_test
    # holds the file open through the brake and the tail, so this now waits out
    # log_tail_s as well.
    rig.pump(25, until=lambda r: len(r.stop_reasons) > n_before
             and r.stop_reasons[-1].get('kind') == 'position_window')
    reason = rig.stop_reasons[-1] if len(rig.stop_reasons) > n_before else {}
    print(f'  A8 test driving output 2 rad: stop_reason={reason.get("kind")} ({reason.get("detail")})')
    check(reason.get('kind') == 'position_window', 'the window did not trip the test')

    # A8b: the brake ran, and the tail kept the log open past the trip.
    #
    # SKIPPED rather than failed when A8 did not trip: a failing touch-off above
    # leaves `centre` somewhere the 2 rad trace can no longer reach, so there is
    # no trip to brake and asserting one would just report A7's failure twice.
    # dyno/sim/post_test_brake_test.py covers the brake and the tail on their
    # own, declaring centre at the power-up angle instead of touching off.
    rig.pump(2)
    if reason.get('kind') != 'position_window':
        print('  A8b SKIPPED: no window trip to brake (A8 above). The brake and '
              'the tail are covered on their own by '
              'dyno/sim/post_test_brake_test.py')
    else:
        brake = reason.get('brake') or {}
        tail_s = (CFG.get('post_test') or {}).get('log_tail_s') or 0.0
        # No :+d / :.3f on anything out of the reason dict -- a partial brake
        # record has None where a number would be, and a diagnostic print that
        # raises TypeError replaces the failure it was meant to describe.
        print(f'  A8b brake: {brake.get("outcome")} ({brake.get("mode")} mode, '
              f'{brake.get("torque_nm")} Nm, polarity {brake.get("polarity")}, '
              f'entry {brake.get("entry_velocity_rad_s")} rad/s'
              + (f', stopped after {brake["stopped_after_s"] * 1000:.0f} ms'
                 if brake.get('stopped_after_s') is not None else '') + ')')
        check(brake.get('outcome') == 'stopped',
              f'the post-test brake did not stop the shaft: {brake}')
        check('aborted' not in (brake.get('outcome') or ''),
              'the brake aborted on polarity -- _brake_polarity disagrees with '
              'the drive, check flip_torque_sign / flip_direction_sign')
        print(f'     stages seen: {rig.post_stages}; samples logged after the '
              f'trip: {rig.post_logged} (tail {tail_s:g} s at 1 kHz)')
        check('brake' in rig.post_stages and 'tail' in rig.post_stages,
              f'the wind-down did not pass through both stages: {rig.post_stages}')
        # Generous band: the queue is drained by a python loop and the brake
        # time rides on top of the tail. What matters is thousands, not zero.
        check(rig.post_logged > tail_s * 1000 * 0.7,
              f'only {rig.post_logged} samples logged after the trip, expected '
              f'about {tail_s * 1000:.0f} -- the tail did not hold the log open')

        # The early trip plus the brake are supposed to leave the output INSIDE
        # the window -- that is the whole design goal, and it is why this no
        # longer asserts the old "Start refused because the output is outside".
        rel_after = rig.w.get('rel')
        print(f'     after trip: rel {rel_after} (window +/-'
              f'{rig.w.get("half_window")}), start blocked: {rig.w.get("refusal")}')
        check(rel_after is not None
              and abs(rel_after) <= rig.w.get('half_window'),
              f'the braked trip still left the output outside the window at '
              f'{rel_after:+.3f} -- the stop did not arrive in time')

    rig.send('jog', -1)
    rig.pump(30, until=lambda r: (r.w.get('rel') or 9) < 0.0)
    rig.send('jog_stop')
    rig.pump(2, until=lambda r: r.w.get('jog') is None)
    print(f'  A9 jogged back to rel {rig.w.get("rel"):+.3f}; waiting for the '
          f'{WKC_DROP_FROM_S} s frame-loss schedule...')
    arm(rig, 'sim_archimedes_shuttle.yaml')
    rig.pump(WKC_DROP_FROM_S + 20, until=lambda r: r.samples >= (WKC_DROP_FROM_S - 5) * 1000)
    n_reasons = len(rig.stop_reasons)
    rig.send('start_test')

    # There is deliberately NO wkc_error safety on this rig, so the assertion
    # is the inverse of what it used to be: dropped frames must NOT stop a
    # running test. Both regimes are still injected -- isolated single-cycle
    # drops and one dense burst -- because the point is that neither is a
    # nuisance stop. Losing a slave outright still faults the drives, and the
    # drive-fault path stops the test; that is the layer being relied on.
    rig.pump(30, until=lambda r: r.samples >= (WKC_BURST_S + 4) * 1000)
    survived = len(rig.stop_reasons) == n_reasons
    stopped_by = (rig.stop_reasons[-1] if not survived else {})
    verdict = ('test still running' if survived
               else f'STOPPED by {stopped_by.get("check")}')
    print(f'     {WKC_ISOLATED_N} isolated + {WKC_BURST_CYCLES}-cycle burst of '
          f'dropped frames: {verdict}')
    check(survived,
          f'dropped frames stopped the test ({stopped_by.get("check")!r}) -- '
          f'there should be no wkc_error safety on this config')
finally:
    rig.close()


print('\n--- 2B. No bumpers ---')
rig = Rig({'DYNO_SIM_PARAMS': json.dumps({'endstop_half_span_rad': None})})
try:
    rig.pump(6)
    rig.send('declare_centre')
    rig.pump(2, until=lambda r: r.w.get('centre') is not None)
    print('  B1 touch-off with nothing to hit (about 30 s)...')
    rig.send('touch_off')
    rig.pump(2, until=lambda r: r.w.get('jog') == 'touch_off')
    rig.pump(90, until=lambda r: r.w.get('jog') is None)
    t = rig.w.get('touch_off') or {}
    print(f'     {rig.w.get("message")}')
    check(t and not t.get('passed'), 'touch-off passed with no bumpers')
    check(any('max excursion' in p for p in t.get('problems', [])),
          f'failure does not name max excursion: {t.get("problems")}')
    check(abs(rig.w.get('rel') or 9) < 0.05, 'failed touch-off did not return to centre')

    print('  B1b input jog + with no keepalive (button release lost)...')
    rig.cq.put_nowait(['jog', 1, 'input'])   # raw put: the Rig sends no keepalive
    rig.pump(2, until=lambda r: r.w.get('jog') == 'jog')
    started = time.time()
    rig.pump(5, until=lambda r: r.w.get('jog') is None)
    print(f'     after {time.time() - started:.2f} s: {rig.w.get("message")}')
    check(rig.w.get('jog') is None and 'keepalive' in (rig.w.get('message') or ''),
          'a jog with no keepalive did not stop by itself')
    rig.pump(1)
    check(abs(rig.value('dut_velocity')) < 1.0,
          f'input still turning after the keepalive stop ({rig.value("dut_velocity"):+.2f} rad/s)')
    rig.send('jog', -1, 'input')
    rig.pump(30, until=lambda r: abs(r.w.get('rel') or 9) < 0.03)
    rig.send('jog_stop')
    rig.pump(2, until=lambda r: r.w.get('jog') is None)

    print('  B2 hold jog + until it stops by itself...')
    rig.send('jog', 1)
    rig.pump(40, until=lambda r: r.w.get('jog') is None and (r.w.get('rel') or 0) > 1.0)
    print(f'     {rig.w.get("message")}')
    max_exc = CFG['position_window']['touch_off']['max_excursion_rad']
    check('max excursion' in (rig.w.get('message') or ''), 'jog did not stop at max excursion')
    check(abs((rig.w.get('rel') or 0) - max_exc) < 0.05,
          f'stopped at {rig.w.get("rel")}, not near max excursion {max_exc}')
    rig.send('jog_stop')
    rig.pump(1)
    rig.send('jog', 1)
    rig.pump(1)
    print(f'  B3 jog + again: {rig.w.get("message")}')
    check('already at max excursion' in (rig.w.get('message') or ''),
          'jog further past max excursion was not refused')
    sign = rig.w.get('input_sign')
    check(sign in (1, -1), 'input direction not learned after touch-off')
    rig.send('jog', sign or 1, 'input')
    rig.pump(1)
    print(f'  B4 input jog outward at max excursion: {rig.w.get("message")}')
    check('already at max excursion' in (rig.w.get('message') or ''),
          'input jog further past max excursion was not refused')
finally:
    rig.close()

print('\nARCHIMEDES WINDOW TEST', 'PASSED' if ok else 'FAILED')
sys.exit(0 if ok else 1)
