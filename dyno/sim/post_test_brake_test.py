"""End-of-test brake and logging tail (config `post_test:`), on the fake bus.

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/post_test_brake_test.py [mode]

Deliberately narrow, and deliberately NOT part of archimedes_window_test.py:
that test's touch-off steps are mid-rework and their failures leave `centre`
somewhere the window-trip trace can no longer reach, which then hides whether
the brake and the tail worked. This declares centre at the power-up angle and
goes straight at a trip, so a failure here is about the brake or the tail.

`hardware_torque_sign` is on because the brake's torque-mode polarity depends
on the drive's torque sign, and the fake AKD's default convention is the
opposite of the real one (log finding 10 / 21).
"""
import json
import math
import multiprocessing
import os
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else 'inhouse_archimedes'
os.environ['DYNO_SIM'] = MODE
os.environ['DYNO_SIM_PARAMS'] = json.dumps({'hardware_torque_sign': True})

import yaml  # noqa: E402
from deployment import dyno_paths  # noqa: E402
from dyno.src.dyno_controller import Controller  # noqa: E402

with open(f'{dyno_paths.dyno_config_directory}/{MODE}_dyno_config.yaml') as f:
    CFG = yaml.safe_load(f)
POST = CFG.get('post_test') or {}
TAIL_S = POST.get('log_tail_s') or 0.0
BRAKE = POST.get('brake') or {}

ok = True


def fail(msg):
    global ok
    ok = False
    print(f'FAIL: {msg}')


def check(cond, msg):
    if not cond:
        fail(msg)
    return cond


class Rig:
    def __init__(self):
        self.tq = multiprocessing.Queue()
        self.cq = multiprocessing.Queue()
        self.proc = multiprocessing.Process(target=Controller,
                                            args=[self.tq, self.cq, MODE])
        self.proc.start()
        self.state, self.log = {}, {}
        self.reasons = []
        self.stages, self.tail_logged = [], 0
        self._stage = None

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
            self.log, self.state = s[-2], s[-1]
            r = self.log.get('stop_reason')
            if r and (not self.reasons or self.reasons[-1] != r):
                self.reasons.append(r)
            stage = self.state.get('post_test')
            if stage != self._stage:
                self.stages.append(stage)
                self._stage = stage
            if stage is not None and self.log.get('log'):
                self.tail_logged += 1
            if until is not None and until(self):
                return True
        return False

    @property
    def w(self):
        return self.state.get('window') or {}

    def close(self):
        self.send('shutdown')
        self.proc.join(timeout=15)
        if self.proc.is_alive():
            self.proc.terminate()


print(f'--- post_test brake + tail ({MODE}) ---')
print(f'    config: log_tail_s={TAIL_S:g}, brake={ {k: v for k, v in BRAKE.items() if k in ("enabled", "torque_nm", "polarity", "timeout_s")} }')
check(BRAKE.get('enabled'), 'post_test.brake is not enabled in this config')
check(TAIL_S > 0, 'post_test.log_tail_s is not set in this config')

rig = Rig()
try:
    print('  bring-up...')
    rig.pump(6)

    # Centre at the power-up angle, so the trace's 2 rad output command is well
    # outside the window. No touch-off: this config has require_touch_off false.
    rig.send('declare_centre')
    if not rig.pump(3, until=lambda r: r.w.get('centre') is not None):
        fail('centre was never declared')
    print(f'  centre at {rig.w.get("centre"):+.4f} rad, window +/-{rig.w.get("half_window"):g}')

    rig.send('test_def', ('sim_archimedes_window_trip.yaml', MODE))
    if not rig.pump(25, until=lambda r: r.state.get('armed') == 'sim_archimedes_window_trip'):
        fail(f'test never armed (load_error={rig.state.get("load_error")})')

    n = len(rig.reasons)
    rig.stages, rig.tail_logged, rig._stage = [], 0, rig.state.get('post_test')
    t_start = time.time()
    rig.send('start_test')
    rig.pump(2)
    # This test deliberately skips touch-off (it declares centre and goes), so
    # it cannot run against a config that requires one. Say that plainly rather
    # than reporting eight downstream assertion failures: on the fake bus the
    # input-shaft touch-off is mid-rework and does not pass, so there is no way
    # for this test to satisfy the gate.
    refusal = rig.w.get('refusal') or ''
    if 'touch-off' in refusal or 'touch-off' in (rig.w.get('message') or ''):
        print(f'\n  CANNOT RUN: {rig.w.get("message") or refusal}')
        print('  This test declares centre and goes straight at a trip, so it '
              'needs\n  position_window.require_touch_off: false. Either set '
              'that for the run,\n  or verify the brake on the rig with '
              'dyno/tests/archimedes_brake_check_*.yaml.')
        print('  The brake logic itself is covered without a bus by '
              'dyno/sim/post_test_logic_test.py.')
        rig.close()
        sys.exit(2)
    if not rig.pump(30, until=lambda r: len(r.reasons) > n):
        fail('the test never ended -- no stop_reason arrived')
    reason = rig.reasons[-1] if len(rig.reasons) > n else {}
    elapsed = time.time() - t_start
    print(f'  1. trip: {reason.get("kind")} -- {reason.get("detail")}')
    check(reason.get('kind') == 'position_window',
          f'expected a position_window trip, got {reason.get("kind")}')

    b = reason.get('brake') or {}
    print(f'  2. brake: {b.get("outcome")}  ({b.get("mode")} mode, {b.get("torque_nm")} Nm, '
          f'polarity {b.get("polarity")}, entry {b.get("entry_velocity_rad_s")} rad/s'
          + (f', stopped after {b["stopped_after_s"] * 1000:.0f} ms' if 'stopped_after_s' in b else '')
          + ')')
    check(b, 'the stop_reason carries no brake record')
    check(b.get('outcome') == 'stopped',
          f'the brake did not stop the shaft: {b.get("outcome")!r}')
    check('aborted' not in (b.get('outcome') or ''),
          'the brake aborted on polarity -- _brake_polarity disagrees with the '
          'drive; see log finding 21 before changing post_test.brake')

    print(f'  3. wind-down stages: {rig.stages}')
    check(rig.stages[:2] == ['brake', 'tail'],
          f'expected brake then tail, got {rig.stages}')
    check(bool(rig.stages) and rig.stages[-1] is None,
          f'the wind-down never ended: {rig.stages}')

    print(f'  4. samples logged after the trip: {rig.tail_logged} '
          f'(tail {TAIL_S:g} s at 1 kHz; the stop_reason took {elapsed:.1f} s to arrive)')
    check(rig.tail_logged > TAIL_S * 1000 * 0.7,
          f'only {rig.tail_logged} samples logged after the trip, expected about '
          f'{TAIL_S * 1000:.0f} -- the tail did not hold the log open')

    rel = rig.w.get('rel')
    print(f'  5. output came to rest at rel {rel:+.4f} rad '
          f'({math.degrees(rel):+.1f} deg), window +/-{rig.w.get("half_window"):g}')
    check(abs(rel) <= rig.w.get('half_window'),
          f'the braked trip left the output OUTSIDE the window at {rel:+.4f} -- '
          f'the early trip and the brake together are supposed to keep it in')

    # The wind-down is over, so nothing should be refusing Start any more.
    print(f'  6. start refusal now: {rig.w.get("refusal")}')
    check('winding down' not in (rig.w.get('refusal') or '')
          and 'braking' not in (rig.w.get('refusal') or ''),
          'the wind-down is still refusing Start after it ended')
finally:
    rig.close()

print('\nPOST-TEST BRAKE TEST PASSED' if ok else '\nPOST-TEST BRAKE TEST FAILED')
sys.exit(0 if ok else 1)
