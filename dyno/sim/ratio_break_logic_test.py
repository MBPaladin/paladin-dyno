"""Ratio-break (slip) safety: logic only, no bus.

Covers the two channels `inhouse_archimedes_dyno_config.yaml` watches with its
`ratio_break` / `ratio_slip` safeties, and the `resumable` flag `_safety_trigger`
stamps onto a trip. What that flag then BUYS -- the recentre-and-resume offer --
is checked in `resume_logic_test.py`, which already owns the `_stop_test` stub.

This lives apart from `archimedes_window_test.py`, where a bus-level version
would belong, because that harness crashes during bring-up before it can reach
anything -- see docs/sim_known_failures.md section 4.

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/ratio_break_logic_test.py
"""
import math
import sys
import time
import types
from operator import attrgetter

from dyno.src.dyno_controller import Controller

R = 43.88
fails = []
def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok: fails.append(msg)

def stub(ratio=R):
    c = Controller.__new__(Controller)
    c.mode = 'inhouse_archimedes'; c.dyno_params = {}; c._load_only = False; c.time = 1.0
    c._window = None
    c.devices = types.SimpleNamespace(
        LOAD=types.SimpleNamespace(position=0.0, velocity=0.0, fault=False),
        DUT=types.SimpleNamespace(position=0.0, fault=False, params={'gear_ratio': -R}))
    c._safety_checks = []; c._safety_streaks = {}; c._resumable_checks = set()
    c._input_ratio = ratio
    c._ratio_ref = None; c._ratio_rate_mark = None
    c.ratio_error_rad = math.nan; c.ratio_slip_rad_s = math.nan
    return c

print('--- ratio_error_rad ---')
c = stub()
c._update_ratio_error()
check(math.isnan(c.ratio_error_rad), 'NaN before any reference is latched')

c._latch_ratio_reference()
c._update_ratio_error()
check(c.ratio_error_rad == 0.0, 'zero at the latch point')

# No slip: shafts move together at the measured ratio.
c.devices.DUT.position = 43.88 * 0.5
c.devices.LOAD.position = 0.5
c._update_ratio_error()
check(abs(c.ratio_error_rad) < 1e-9,
      f'zero when locked at the ratio (got {c.ratio_error_rad:.2e})')

# Pure slip: input turns 10 rad, output does not follow at all.
c.devices.DUT.position = 43.88 * 0.5 + 10.0
c._update_ratio_error()
check(abs(c.ratio_error_rad - (-10.0 / R)) < 1e-9,
      f'reports -input/R of slip ({c.ratio_error_rad:+.5f}, expected {-10/R:+.5f})')

# Sign trap: the config ratio is -43.88 while the channels read +43.88. The
# measured _input_ratio must be preferred, or every normal move doubles.
c2 = stub(); c2._input_ratio = None      # force the config fallback
c2._latch_ratio_reference()
c2.devices.DUT.position = 43.88; c2.devices.LOAD.position = 1.0
c2._update_ratio_error()
c3 = stub(); c3._latch_ratio_reference()
c3.devices.DUT.position = 43.88; c3.devices.LOAD.position = 1.0
c3._update_ratio_error()
check(abs(c3.ratio_error_rad) < 1e-9 and abs(c2.ratio_error_rad - 2.0) < 1e-9,
      f'measured ratio gives 0, the config sign would give +2 rad '
      f'(measured {c3.ratio_error_rad:+.3f}, config {c2.ratio_error_rad:+.3f})')

print('--- ratio_slip_rad_s ---')
c = stub(); c._latch_ratio_reference(); c._update_ratio_error()
t0 = time.perf_counter()
while time.perf_counter() - t0 < 0.14:         # ~3 rate windows
    c.devices.DUT.position += R * 0.002        # input runs, output stays
    c._update_ratio_error()
check(c.ratio_slip_rad_s < -1.0,
      f'a runaway reads a large negative rate ({c.ratio_slip_rad_s:+.2f} rad/s)')

print('--- resumable safety trip offers recentre-and-resume ---')
c = stub()
c.ratio_error_rad = 5.0
c._safety_checks = [('ratio_break', attrgetter('ratio_error_rad'), 0.40, False, 1)]
c._resumable_checks = {'ratio_break'}
trip = c._safety_trigger()
check(trip is not None and trip.get('resumable') is True,
      f'ratio_break trip is marked resumable ({trip})')
c._safety_streaks = {}
c._safety_checks = [('output_torque', attrgetter('ratio_error_rad'), 0.40, False, 1)]
c._resumable_checks = set()
trip2 = c._safety_trigger()
check(trip2 is not None and not trip2.get('resumable'),
      'an unmarked safety is NOT resumable')

print('\n' + ('ALL OK' if not fails else f'{len(fails)} FAILED'))
print('(the resume OFFER a ratio_break trip leaves is covered in resume_logic_test.py,')
print(' which already has the _stop_test stub this would otherwise duplicate)')
sys.exit(1 if fails else 0)
