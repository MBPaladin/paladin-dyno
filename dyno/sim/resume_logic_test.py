"""Logic-only exercise of recentre-and-resume after a position-window trip.

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/resume_logic_test.py

No bus, no drives: a stubbed Controller in the style of post_test_logic_test.
The fake bus cannot trip the window on demand (archimedes_window_test, §4 of
sim_known_failures.md), so this covers the controller half of the feature:
which stops leave a resume point, what the 'resume_test' / 'abort_resume'
commands do with it, and that the armed plan is restarted at the interrupted
segment. TestManager.reset(start_segment=) itself is checked against a real
plan at the bottom. The GUI popups are not covered here.
"""
import os
import queue
import sys
import types

os.environ.setdefault('DYNO_SIM', 'inhouse_archimedes')

from dyno.src.dyno_controller import Controller  # noqa: E402

ok = True
def check(cond, msg):
    global ok
    if not cond:
        ok = False; print('FAIL:', msg)


class FakePlan:
    """Just enough of TestManager for _stop_test / _start_test."""
    def __init__(self, name='archimedes_bench_input_spin_mid', segment=3, segments=5):
        self.name = name
        self._p = {'segment': segment, 'segments': segments, 'segment_id': 'V1200',
                   'segment_type': 'test_trace', 'repeat': 1, 'repeats': 1}
        self.resets = []
    def progress(self):
        return self._p
    def reset(self, start_segment=1):
        self.resets.append(start_segment)


def stub(plan=None, active=True):
    c = Controller.__new__(Controller)
    c.mode = 'inhouse_archimedes'
    c.dyno_params = {}
    c._load_only = False
    c.shutdown = False
    c.time = 1.0
    c._stop_reason = None
    c._resume = None
    c._test_active = active
    c._post_test = None
    c._post = {'log_tail_s': 0.0, 'brake': None}
    c.test_definition = plan
    c._pending_test_definition = None
    c._loading_test = None
    c._test_load_error = None
    c._test_load_generation = 0
    c._jog = None
    c._tare_state = None
    c._fault_clear = None
    c._window = None
    c._window_centre = None
    c._window_message = None
    c._align_load_position = False
    c.pull_cmd = False
    c._command_queue = queue.Queue()
    c._safe_default_command = {'input_mode': 'torque', 'output_mode': 'torque',
                               'input_command': 0, 'output_command': 0}
    drive = lambda: types.SimpleNamespace(position=0.0, velocity=0.0, fault=False,
                                          sw_enable=True, mode='torque',
                                          switching_modes=False, position_offset=0,
                                          command_operating_mode=lambda m: None)
    c.devices = types.SimpleNamespace(LOAD=drive(), DUT=drive())
    # Steps _stop_test runs that have nothing to do here.
    c._tare_step = lambda: None
    c._fault_clear_step = lambda: None
    c._begin_post_test = lambda reason: None
    c._yield_post_test_tail = lambda: None
    return c


TRIP = {'kind': 'position_window', 'check': 'position_window', 'value': 1.52,
        'limit': 1.5, 'at_s': 12.3, 'detail': 'position window: output +1.520 rad '
        'from centre, outside +/-1.5 rad'}

# A ratio-break (slip) trip, which reaches the same offer by a different route:
# it is an ordinary `safeties:` entry carrying `resumable: true`, not a window
# trip. See the 2026-09-17 log entry -- a slip leaves every other safety in
# range, and the customer's drive is built to slip, so the cure is the window's
# cure: put the output back on centre and restart the segment.
SLIP_TRIP = {'kind': 'safety', 'check': 'ratio_break', 'value': 0.62,
             'limit': 0.40, 'trip_samples': 50, 'resumable': True, 'at_s': 12.3,
             'detail': "safety check 'ratio_break' measured 0.62, over its "
                       'limit of 0.4 for 50 consecutive cycles'}

# --- what leaves a resume point ------------------------------------------------
c = stub(FakePlan())
c._stop_test(TRIP)
check(c._resume is not None, 'window trip should leave a resume point')
check(c._resume and c._resume['segment'] == 3 and c._resume['segments'] == 5
      and c._resume['segment_id'] == 'V1200'
      and c._resume['test'] == 'archimedes_bench_input_spin_mid',
      f'resume point should name segment 3/5 V1200: {c._resume}')
check(c.test_definition.resets == [1], 'the tripped plan is still reset() normally')
def control_state(c):
    # _control_state reads many idle-state attributes the stub does not model
    # (tare result, fault list, ...). None is what each of them is at idle.
    for _ in range(40):
        try:
            return c._control_state()
        except AttributeError as e:
            setattr(c, str(e).split("'")[-2], None)
    raise RuntimeError('stub could not satisfy _control_state')
check(control_state(c)['resume'] == c._resume, 'control_state must carry the offer')
check(not c._test_active, 'test is no longer active after the trip')

for reason in ({'kind': 'operator', 'detail': 'stopped from the GUI'},
               {'kind': 'completed', 'detail': 'test ran to completion'},
               {'kind': 'stator_temp', 'detail': 'hot'}, None):
    c = stub(FakePlan()); c._stop_test(reason)
    check(c._resume is None, f'{reason and reason["kind"]} stop must not offer a resume')

c = stub(FakePlan()); c._stop_test(SLIP_TRIP)
check(c._resume is not None, 'a resumable safety trip should leave a resume point')
check(c._resume and c._resume['label'] == 'Safety trip: ratio_break',
      f'the offer should name what tripped, for the dialog title: {c._resume}')
check(c._resume and c._resume['check'] == 'ratio_break',
      'the offer should carry the check name (the GUI keys its wording off it)')

c = stub(FakePlan())
c._stop_test(dict(SLIP_TRIP, resumable=False))
check(c._resume is None,
      'a safety WITHOUT resumable should clear the offer, as before')

c = stub(FakePlan(), active=False); c._stop_test(TRIP)
check(c._resume is None, 'a trip with no test running leaves nothing to resume')

c = stub(None); c._stop_test(TRIP)
check(c._resume is None, 'a trip with no plan armed leaves nothing to resume')

# A later stop of any kind (e.g. Stop pressed during the return-to-centre)
# withdraws the offer: Stop means abort.
c = stub(FakePlan()); c._stop_test(TRIP)
c._stop_test({'kind': 'operator', 'detail': 'stop'})
check(c._resume is None, 'operator Stop after a trip withdraws the offer')

# --- the commands ------------------------------------------------------------------
def run(c, *cmd):
    c._command_queue.put_nowait(list(cmd)); c._cmd_check()

# resume_test with an offer: restarts the same plan at the interrupted segment
c = stub(FakePlan()); c._stop_test(TRIP)
run(c, 'resume_test', 0)
check(c._test_active, 'resume_test should start the test')
check(c.test_definition.resets == [1, 3], f'resume should reset(start_segment=3): {c.test_definition.resets}')
check(c._resume is None, 'a consumed offer is cleared')
check(c._stop_reason is None, 'the trip reason must not carry into the resumed log')
check(c.devices.DUT.sw_enable and c.devices.LOAD.sw_enable, 'drives enabled on resume')
check('Resuming' in (c._window_message or ''), f'operator told about the resume: {c._window_message}')

# resume_test with no offer: refused, nothing starts
c = stub(FakePlan(), active=False); run(c, 'resume_test', 0)
check(not c._test_active and c.test_definition.resets == [],
      'resume with nothing to resume must not start a run')
check('nothing to resume' in (c._window_message or ''), f'refusal reported: {c._window_message}')

# offer for one plan, a different plan armed since: refused
c = stub(FakePlan()); c._stop_test(TRIP)
c.test_definition = FakePlan(name='archimedes_gravity_map')
run(c, 'resume_test', 0)
check(not c._test_active, 'resume must refuse once another plan is armed')
check('no longer armed' in (c._window_message or ''), f'refusal names the plan: {c._window_message}')
check(c._resume is not None, 'a refused resume keeps the offer (operator may re-arm)')

# resume while the return-to-centre jog is still running: refused
c = stub(FakePlan()); c._stop_test(TRIP); c._jog = {'kind': 'return_centre', 'legs': []}
run(c, 'resume_test', 0)
check(not c._test_active and 'return-to-centre' in (c._window_message or ''),
      f'resume must wait for the jog: {c._window_message}')

# resume during the brake / tail: refused like Start is (post_test_refusal)
c = stub(FakePlan()); c._stop_test(TRIP); c._post_test = {'stage': 'brake'}
c._post_test_refusal = lambda: 'the brake is still holding the shaft'
run(c, 'resume_test', 0)
check(not c._test_active and c._resume is not None, 'resume during the brake is refused and the offer kept')

# abort_resume clears it; harmless with nothing pending
c = stub(FakePlan()); c._stop_test(TRIP); run(c, 'abort_resume', 0)
check(c._resume is None and 'abandoned' in (c._window_message or ''), 'abort clears the offer')
c = stub(FakePlan()); run(c, 'abort_resume', 0)
check(c._resume is None and c._window_message is None, 'abort with nothing pending is silent')

# an ordinary Start after a trip drops the offer and starts from segment 1
c = stub(FakePlan()); c._stop_test(TRIP); run(c, 'start_test', 0)
check(c._test_active and c.test_definition.resets == [1, 1] and c._resume is None,
      f'plain Start after a trip starts from 1 and drops the offer: {c.test_definition.resets}')

# --- TestManager.reset(start_segment=) on a real plan -------------------------------
from dyno.src import test_preview
from dyno.src.test_manager import TestManager
lim = test_preview.limits_from_config('inhouse_archimedes')
tm = TestManager('ui_generated_tests/archimedes_bench_input_spin_mid.yaml',
                 'inhouse_archimedes', lim)
tm.reset(start_segment=3); tm.next_command(); p = tm.progress()
check(p['segment'] == 3 and p['segments'] == 5 and p['segment_id'] == 'V1200',
      f'resumed run should open on 3/5 V1200: {p}')
tm.reset(); tm.next_command()
check(tm.progress()['segment'] == 1, 'plain reset still starts at 1')
tm.reset(start_segment=99); n = 0
while tm.next_command() is not None:
    n += 1
check(n == 0, 'a start past the last segment runs nothing and completes')

print('\nRESUME LOGIC ' + ('PASS' if ok else 'FAIL'))
sys.exit(0 if ok else 1)
