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
    stub(dict(CFG['position_window'], velocity_source=None))
    fail('stop_decel without velocity_source was accepted')
except ValueError:
    pass

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

    def send(self, *cmd):
        self.cq.put_nowait(list(cmd) if len(cmd) > 1 else [cmd[0], 0])

    def pump(self, seconds, until=None):
        deadline = time.time() + seconds
        while time.time() < deadline:
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
rig = Rig({'DYNO_SIM_WKC_DROP': ','.join(
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

    print('  A7 touch-off (about a minute)...')
    rig.send('touch_off')
    rig.pump(2, until=lambda r: r.w.get('jog') == 'touch_off')
    rig.pump(150, until=lambda r: r.w.get('jog') is None)
    t = rig.w.get('touch_off') or {}
    print(f'     {rig.w.get("message")}')
    check(t.get('passed'), f'touch-off did not pass: {t}')
    if t.get('plus_rad') is not None and t.get('minus_rad') is not None:
        print(f'     +{t["plus_rad"]:.4f} / {t["minus_rad"]:+.4f} rad, centre error '
              f'{t["centre_error_rad"]:+.4f}, back at rel {rig.w.get("rel"):+.3f}')
        check(abs(rig.w.get('rel') or 9) < 0.05, 'touch-off did not return to centre')
    check(not rig.w.get('refusal'), f'still refused after touch-off: {rig.w.get("refusal")}')

    rig.send('start_test')
    rig.pump(15, until=lambda r: r.stop_reasons and r.stop_reasons[-1].get('kind') == 'position_window')
    reason = rig.stop_reasons[-1] if rig.stop_reasons else {}
    print(f'  A8 test driving output 2 rad: stop_reason={reason.get("kind")} ({reason.get("detail")})')
    check(reason.get('kind') == 'position_window', 'the window did not trip the test')
    rig.pump(2)
    print(f'     after trip: rel {rig.w.get("rel"):+.3f}, start blocked: {rig.w.get("refusal")}')
    check(rig.w.get('refusal') and 'outside' in rig.w['refusal'],
          'Start not refused with the output outside the window')

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
finally:
    rig.close()

print('\nARCHIMEDES WINDOW TEST', 'PASSED' if ok else 'FAILED')
sys.exit(0 if ok else 1)
