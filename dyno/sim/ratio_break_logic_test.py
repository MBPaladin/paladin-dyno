"""Ratio-break (slip) safety: logic only, no bus.

Covers the channels `inhouse_archimedes_dyno_config.yaml` watches with its
`creep_over` / `creep_under` / `ratio_slip` safeties (`ratio_error_rad`,
`ratio_slip_rad_s` and the per-leg creep ratio), and the `resumable` flag `_safety_trigger`
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

print('--- creep ratio (per leg, against the motion) ---')
def creep_stub(allow=0.0, min_travel=0.25):
    c = stub(); c._creep_lost_motion_rad = allow; c._creep_min_travel_rad = min_travel
    c._latch_ratio_reference(); c._update_ratio_error()
    return c

def roll(c, x_to, c_leg, steps=100):
    """Output rolls to x_to with the report's creep c_leg = 1 - r*d_out/d_in:
    c_leg > 0 is input over-travel, < 0 under-travel."""
    x0 = c.devices.LOAD.position
    for k in range(1, steps + 1):
        x = x0 + (x_to - x0) * k / steps
        dx = x - c.devices.LOAD.position
        c.devices.DUT.position += R * dx / (1.0 - c_leg)
        c.devices.LOAD.position = x
        c._update_ratio_error()

near = lambda a, b, tol=2e-3: abs(a - b) < tol
c = creep_stub(); roll(c, 1.0, 0.0); roll(c, -1.0, 0.0)
check(near(c.creep_over, 0) and near(c.creep_under, 0), 'locked train: no creep either way')

for sign in (+1, -1):
    c = creep_stub(); roll(c, sign * 1.0, +0.06)
    check(near(c.creep_over, 0.06) and c.creep_under == 0,
          f'input over-travel 6% reads over 0.06 (report definition) rolling {sign:+d} ({c.creep_over:.4f})')
    c = creep_stub(); roll(c, sign * 1.0, -0.06)
    check(near(c.creep_under, 0.06) and c.creep_over == 0,
          f'input under-travel 6% reads under 0.06 rolling {sign:+d} ({c.creep_under:.4f})')

# Constant torque sign: creep walks the output the SAME way in the fixed frame on
# both legs, so it is under on one leg and over on the next. Whole-run path
# length would read zero; per leg it must not.
c = creep_stub(); roll(c, 1.0, -0.04); u1 = c.creep_under
roll(c, -1.0, +0.04); o2 = c.creep_over
check(near(u1, 0.04) and near(o2, 0.04),
      f'torque-driven creep: under {u1:.4f} on the outbound leg, over {o2:.4f} on the return')

# A leg runs until the output reverses, so a slip that starts mid-leg is diluted
# by the clean rolling before it: the second rad at 10% makes the LEG
# 1 - 2/(1 + 1/0.9) = 5.3%.
c = creep_stub(); roll(c, 1.0, 0.0); roll(c, 2.0, +0.10)
check(near(c.creep_over, 1 - 2 / (1 + 1 / 0.9)), f'a slip mid-leg is a ratio of the whole leg ({c.creep_over:.4f})')
c = creep_stub(); roll(c, 1.0, 0.10); roll(c, 0.0, 0.0)
check(near(c.creep_over, 0.0), 'the next leg starts clean after the reversal')

# Lost motion at a reversal is forgiven up to the allowance, not beyond it.
c = creep_stub(allow=0.05); roll(c, 1.0, 0.0); roll(c, 0.5, +0.09)     # 0.0495 rad slip on 0.5 rolled
check(c.creep_over == 0.0, f'0.045 rad of reversal lost motion is forgiven ({c.creep_over})')
c = creep_stub(allow=0.05); roll(c, 1.0, 0.0); roll(c, 0.0, +0.15)     # 1/0.85 - 1 = 0.176 rad slip on 1.0 rolled
check(near(c.creep_over, (1 / 0.85 - 1 - 0.05) * 0.85),
      f'slip past the allowance still reads ({c.creep_over:.4f})')

# Nothing rolling: the floor turns it into an absolute limit, not a divide by zero.
c = creep_stub(allow=0.05)
c.devices.DUT.position += 1.5; c._update_ratio_error()
check(c.creep_over == 0.0, 'small input motion against a locked output is inside the allowance')
c.devices.DUT.position += 3.5; c._update_ratio_error()       # 5 rad input = 0.114 rad output
check(near(c.creep_over, (5.0 / R - 0.05) / 0.25) and c.creep_over > 0.05,
      f'input runaway on a locked output trips the 5% limit ({c.creep_over:.3f})')
c = creep_stub(allow=0.05)
c.devices.LOAD.position = 1.0; c._update_ratio_error()       # output moves, input does not
check(c.creep_under > 0.05 and c.creep_over == 0.0,
      f'output back-driving a still input reads a large under ({c.creep_under:.2f})')

# A re-latch (ratio_reset, new test) starts a fresh leg.
c = creep_stub(); roll(c, 1.0, +0.30); c._latch_ratio_reference(); c._update_ratio_error()
check(near(c.creep_over, 0) and near(c.creep_under, 0), 'latch clears the leg')

# The two safeties, wired the way the config declares them.
c = creep_stub(); roll(c, 1.0, +0.08)
c._safety_checks = [('creep_over', attrgetter('creep_over'), 0.05, False, 1),
                    ('creep_under', attrgetter('creep_under'), 0.05, False, 1)]
c._resumable_checks = {'creep_over', 'creep_under'}
trip = c._safety_trigger()
check(trip and trip['check'] == 'creep_over' and trip['resumable'], f'creep_over trips at 8% ({trip and trip["check"]})')
c = creep_stub(); roll(c, -1.0, -0.08)
c._safety_checks = [('creep_over', attrgetter('creep_over'), 0.05, False, 1),
                    ('creep_under', attrgetter('creep_under'), 0.05, False, 1)]
c._resumable_checks = {'creep_over', 'creep_under'}
trip = c._safety_trigger()
check(trip and trip['check'] == 'creep_under', f'creep_under trips at 8% ({trip and trip["check"]})')
c = creep_stub(); roll(c, 1.0, +0.04); roll(c, -1.0, -0.04)
c._safety_checks = [('creep_over', attrgetter('creep_over'), 0.05, False, 1),
                    ('creep_under', attrgetter('creep_under'), 0.05, False, 1)]
check(c._safety_trigger() is None, '4% either way stays inside the 5% limit')

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
