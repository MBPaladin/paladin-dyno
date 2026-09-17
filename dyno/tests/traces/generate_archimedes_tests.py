"""Draft plan §7 Step 4 test plans for the Archimedes drive, as builder recipes.

See dyno/docs/archimedes_gearbox_implementation_plan.md (§7 Step 4) and the
implementation log (findings 9, 13, 16). Every plan here position-commands one
shaft and runs the other in torque mode, and every shuttle ends short of the
window trip line.

Everything that is still unmeasured sits in the PENDING block below. Rig limits,
the window and `stop_decel_rad_s2` are read from the config, so after the STO
coast is measured and the config updated, re-running this script re-sizes every
traverse to match. The outputs are ordinary builder recipes, so any of them can
also be reopened and hand-tuned in the Test Builder.

Written to (gitignored, like every builder output):
    dyno/tests/ui_generated_tests/archimedes_*.yaml
    dyno/tests/traces/ui_generated/archimedes_*__<SEG>.csv

Each plan is expanded through test_preview.expand_test (the same TestManager
load and limit asserts the rig runs), then checked against the position window
including the early stopping-distance trip. The sim cannot do this check: its
single rigid shaft has no 43:1 between input and output.

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/tests/traces/generate_archimedes_tests.py [--shakedown]

--shakedown writes `*_shakedown` variants: at most 300 rpm input, 25% torque,
one cycle. Plan §7 Step 4 says run every test low first.

--bench writes only what runs today, with the output on the lockout plate and
no gearbox: an output lockout rerun and the input spin bands.
"""
import argparse
import math
import sys

import numpy as np
import yaml

from deployment import dyno_paths
from dyno.src import test_builder, test_preview

MODE = 'inhouse_archimedes'

# --- PENDING: update from gearbox day-1 and lockout results ------------------
UNIT = '1.2.0'                      # '1.2.0' | '1.2.1' | '1.2.2'
# Output-referred forward static slip torque, from archimedes_fwd_slip. Until it
# is measured, stiffness is drafted against 1.2.2's 140 Nm efficiency rating:
# a unit that meets spec cannot slip below that, so 45% / 90% of it is safe.
T_SLIP_OUT_NM = None
T_SLIP_PROVISIONAL_NM = 140.0
# Output gravity torque amplitude (m*g*r of the flange), from archimedes_gravity_map
# or CAD. Not used to shape commands yet -- recorded so the summary can warn when
# it rivals the smallest efficiency step.
GRAVITY_AMPLITUDE_NM = None
# ------------------------------------------------------------------------------

# Customer request (plan §1).
VEL_RAMP_RPM = list(range(30, 301, 30)) + list(range(600, 3601, 300))
EFF_RPM = [20, 300, 1500, 3000]
EFF_STEP_OUT_NM = 5.0
EFF_MAX_OUT_NM = {'1.2.0': 100.0, '1.2.1': 100.0, '1.2.2': 140.0}

# Throw (log finding 9): traverses end at +/-76 deg; the trip is at +/-86 deg.
OUTPUT_END_RAD = 1.33
# Unmodelled output offset the early trip must tolerate on top of the nominal
# path: centre error for no-load and back-drive (the output is commanded
# directly), plus creep drift within a cycle for loaded forward shuttles.
TRIP_ALLOWANCE_RAD = {'no_load': 0.03, 'backdrive': 0.03, 'loaded_fwd': 0.10}
# Below this much constant-speed half-traverse a speed is not worth running.
MIN_CONST_HALF_RAD = 0.15
# Accel/decel is sized to take DECEL_SHARE of the throw, capped at ACCEL_USE
# of the drive limit: gentle at low speed, the plan's ~5000 rad/s^2 at the top.
DECEL_SHARE = 0.10
ACCEL_USE = 0.9
TURN_DWELL_S = 0.25                 # stop at each end: peak lands exactly on the end
TARGET_CONST_S = 2.0                # constant-speed time per direction, per speed point
MAX_CYCLES = 12

RPM = 2 * math.pi / 60


def load_config():
    with open(f'{dyno_paths.dyno_config_directory}/{MODE}_dyno_config.yaml') as f:
        cfg = yaml.safe_load(f)
    params = {s['name']: s.get('params', {}) for s in cfg['expected_slave_layout']
              if s.get('name') in ('DUT', 'LOAD')}
    w = cfg['position_window']
    return {
        'ratio': abs(float(params['DUT']['gear_ratio'])),
        'half_window': float(w['half_window_rad']),
        'stop_decel': float(w.get('stop_decel_rad_s2', 0.0)),
        'limits': test_preview.limits_from_config(MODE),
        'output_torque_safety': float(cfg['safeties']['output_torque']['limit']),
        'input_torque_safety': float(cfg['safeties']['input_torque']['limit']),
    }


# --- traverse sizing -----------------------------------------------------------

def size_traverse(w_out, a_out, allowance, cfg):
    """Largest output turnaround that the early trip will not fire on.

    With a dwell at each end the builder's blend stops exactly on the
    amplitude, decelerating over d = w^2/2a before it. The early trip fires
    when |x| + v^2/(2*stop_decel) passes the window; along the decel that
    quantity peaks at the decel start (stop_decel < a) or at the turnaround
    (stop_decel >= a). Returns (peak, constant-speed half-length), both in
    output rad.
    """
    h = cfg['half_window'] - allowance
    d = w_out ** 2 / (2 * a_out)
    peak = min(OUTPUT_END_RAD, h)
    s = cfg['stop_decel']
    if s > 0:
        peak = min(peak, h - w_out ** 2 / (2 * s) + d)
    return peak, peak - d


def shuttle_accel(w_shaft, throw_shaft, limit):
    return min(ACCEL_USE * limit, max(w_shaft ** 2 / (2 * DECEL_SHARE * throw_shaft),
                                      0.02 * limit))


def shuttle_segment(seg_id, motor, w_shaft, amp_shaft, accel, cycles,
                    levels=(0.0,), level_rate=10.0, settle_s=1.0):
    return {
        'id': seg_id,
        'repeats': 1,
        'lead_in_s': test_builder.START_HOLD_S,
        'primary': {'motor': motor, 'control_mode': 'position',
                    'accel': round(accel, 3)},
        'secondary': {'control_mode': 'torque', 'levels': [float(v) for v in levels],
                      'rate': level_rate, 'settle_s': settle_s},
        'pattern': 'sawtooth',
        'params': {'amplitude': round(amp_shaft, 4), 'rate': round(w_shaft, 4),
                   'peak_dwell_s': TURN_DWELL_S, 'bipolar': True,
                   'start_at_peak': False, 'cycles': int(cycles),
                   'end_dwell_s': 0.0},
    }


def plan_shuttle(seg_id, motor, rpm_in, allowance, cfg, notes, cycles_cap,
                 **segment_kw):
    """A shuttle at input speed `rpm_in`, commanded on `motor`. None (with a
    note) when the window leaves too little constant-speed travel."""
    r = cfg['ratio']
    lim = cfg['limits'][motor]
    w_in = rpm_in * RPM
    w_out = w_in / r
    shaft = r if motor == 'input' else 1.0          # shaft rad per output rad
    accel = shuttle_accel(w_out * shaft, OUTPUT_END_RAD * shaft,
                          lim['acceleration'])
    peak, const_half = size_traverse(w_out, accel / shaft, allowance, cfg)
    if const_half < MIN_CONST_HALF_RAD:
        notes.append(f'{seg_id}: SKIPPED {rpm_in} rpm -- only {const_half:+.2f} rad of '
                     f'constant speed fits at stop_decel {cfg["stop_decel"]:g} '
                     f'(measure the STO coast, update the config, regenerate)')
        return None
    if peak < OUTPUT_END_RAD - 1e-9:
        notes.append(f'{seg_id}: {rpm_in} rpm turns at +/-{peak:.2f} rad, not '
                     f'{OUTPUT_END_RAD}, to clear the early trip')
    t_const = 2 * const_half / w_out
    cycles = max(1, min(cycles_cap, math.ceil(TARGET_CONST_S / t_const)))
    return shuttle_segment(seg_id, motor, w_out * shaft, peak * shaft, accel,
                           cycles, **segment_kw)


def alternating(step, top):
    """+step, -step, +2step, -2step ... Alternating the load sign between
    cycles cancels creep drift (plan §4)."""
    n = int(round(top / step))
    return [v for k in range(1, n + 1) for v in (k * step, -k * step)]


def breakaway_segment(seg_id, ramp_motor, ceiling, rate, release_s, rest_s,
                      bipolar, cycles, v_thresh, debounce):
    return {
        'id': seg_id,
        'repeats': 1,
        'lead_in_s': test_builder.START_HOLD_S,
        'primary': {'motor': ramp_motor, 'control_mode': 'torque'},
        'secondary': {'control_mode': 'position', 'levels': [0.0],
                      'rate': 0.1, 'settle_s': 1.0},
        'pattern': 'breakaway',
        'params': {'amplitude': round(ceiling, 3), 'rate': rate,
                   'release_s': release_s, 'rest_s': rest_s, 'bipolar': bipolar,
                   'cycles': cycles, 'velocity_threshold': v_thresh,
                   'stuck_s': 0.25, 'debounce_cycles': debounce,
                   'arm_fraction': 0.05},
    }


# --- the plans -------------------------------------------------------------------

def build_plans(cfg, shakedown):
    r = cfg['ratio']
    torque_scale = 0.25 if shakedown else 1.0
    cycles_cap = 1 if shakedown else MAX_CYCLES
    rpm_cap = 300 if shakedown else math.inf
    plans = []          # (recipe, notes)

    # Gravity map. Output shuttled quasi-statically, input free. Cell torque at
    # angle, averaged across the two directions, is gravity (drag flips sign
    # with direction and cancels); half their difference is drag. Two speeds
    # show whether that drag split is speed-independent. Run before touch-off
    # settings are trusted -- see the summary.
    notes = []
    segs = []
    for seg_id, w in (('SLOW', 0.05), ('FAST', 0.15)):
        segs.append(shuttle_segment(seg_id, 'output', w, OUTPUT_END_RAD, 1.0,
                                    1 if shakedown else 2))
    notes.append('output shuttle +/-1.33 rad at 0.05 and 0.15 rad/s, input 0 Nm')
    plans.append(({'name': 'archimedes_gravity_map', 'segments': segs}, notes))

    # Forward static slip (runs first: stiffness needs its result). Output held
    # by the absorber's position loop, input torque ramps until the traction
    # contact lets go. Once it slips the input has almost nothing to hold it,
    # so release fast. The ceiling stays under the output_torque safety,
    # reflected: a slip above it cannot be reached without raising that safety.
    notes = []
    ceiling_in = 0.95 * cfg['output_torque_safety'] / r * torque_scale
    rate_in = 2.0 / r                              # 2 Nm/s output-referred
    plans.append(({'name': 'archimedes_fwd_slip', 'segments': [
        breakaway_segment('FWD_SLIP', 'input', ceiling_in, round(rate_in, 4),
                          release_s=0.05, rest_s=3.0, bipolar=True,
                          cycles=1 if shakedown else 3, v_thresh=0.5, debounce=3)]},
                  [f'input ramp to {ceiling_in:.2f} Nm (= {ceiling_in * r:.0f} Nm output) '
                   f'at {rate_in * r:.0f} Nm/s output-referred; slip above that needs the '
                   'output_torque safety raised']))

    # Stiffness. Input held, output torque to 45% then 90% of slip, 5 s dwell,
    # reverse (plan §1).
    notes = []
    t_slip = T_SLIP_OUT_NM
    if t_slip is None:
        t_slip = T_SLIP_PROVISIONAL_NM
        notes.append(f'PROVISIONAL: slip torque not measured, sized on {t_slip:g} Nm')
    segs = []
    for frac in (0.45, 0.90):
        amp = min(frac * t_slip, 0.95 * cfg['output_torque_safety']) * torque_scale
        segs.append({
            'id': f'K{int(frac * 100)}',
            'repeats': 1,
            'lead_in_s': test_builder.START_HOLD_S,
            'primary': {'motor': 'output', 'control_mode': 'torque'},
            'secondary': {'control_mode': 'position', 'levels': [0.0],
                          'rate': 0.1, 'settle_s': 1.0},
            'pattern': 'sawtooth',
            'params': {'amplitude': round(amp, 2), 'rate': 10.0,
                       'peak_dwell_s': 5.0, 'bipolar': True,
                       'start_at_peak': False, 'cycles': 1, 'end_dwell_s': 2.0},
        })
    plans.append(({'name': 'archimedes_stiffness', 'segments': segs}, notes))

    # Velocity ramps, no load (plan §1: 30 rpm steps to 300, then 300 rpm steps).
    for name, motor, other, allowance in (
            ('archimedes_fwd_velocity_ramp', 'input', 'output', 'no_load'),
            ('archimedes_bwd_velocity_ramp', 'output', 'input', 'backdrive')):
        notes, segs = [], []
        rpms = [x for x in VEL_RAMP_RPM if x <= rpm_cap]
        if shakedown:
            rpms = [30, 150, 300]
        for rpm in rpms:
            seg = plan_shuttle(f'V{rpm:04d}', motor, rpm, TRIP_ALLOWANCE_RAD[allowance],
                               cfg, notes, cycles_cap)
            if seg:
                segs.append(seg)
        plans.append(({'name': name, 'segments': segs}, notes))

    # Efficiency. One plan per speed, so the output can be recentred between
    # them. Levels alternate sign; each level runs full shuttle cycles, so every
    # level carries one driving and one driven traverse.
    top_out = EFF_MAX_OUT_NM[UNIT] * torque_scale
    for rpm in EFF_RPM:
        if rpm > rpm_cap:
            continue
        for direction, motor, allowance, scale in (
                ('fwd', 'input', 'loaded_fwd', 1.0),       # output cell levels
                ('bwd', 'output', 'backdrive', 1.0 / r)):  # input levels, output-referred
            notes = []
            levels = [round(v * scale, 4) for v in alternating(EFF_STEP_OUT_NM, top_out)]
            seg = plan_shuttle(f'E{rpm:04d}', motor, rpm, TRIP_ALLOWANCE_RAD[allowance],
                               cfg, notes, cycles_cap, levels=levels,
                               level_rate=round(50.0 * scale, 4), settle_s=1.0)
            notes.append(f'{UNIT}: {len(levels)} levels to +/-{top_out:g} Nm output-referred')
            if seg:
                plans.append(({'name': f'archimedes_{direction}_efficiency_{rpm:04d}rpm',
                               'segments': [seg]}, notes))
            else:
                plans.append((None, notes))

    # Back-drive static slip, last (plan §7): input held, output torque ramps.
    # When it slips the output moves, so one ramp per direction from centre
    # leaves ~1.3 rad of travel each way before the window. RampBreak has no
    # single-direction option, so the plan's "start near the bumper" variant
    # needs a small code change if slips turn out to travel further than that.
    ceiling_out = 0.95 * cfg['output_torque_safety'] * torque_scale
    plans.append(({'name': 'archimedes_bwd_slip', 'segments': [
        breakaway_segment('BWD_SLIP', 'output', ceiling_out, 2.0, release_s=0.25,
                          rest_s=3.0, bipolar=True, cycles=1, v_thresh=0.05,
                          debounce=5)]},
                  [f'output ramp to {ceiling_out:.0f} Nm at 2 Nm/s, once each way from centre']))

    if shakedown:
        for recipe, _ in plans:
            if recipe:
                recipe['name'] += '_shakedown'
    return plans


def build_bench_plans(cfg):
    """Runnable now: output bolted to the lockout plate, no gearbox, input free.

    Neither plan can pass touch-off (the output cannot reach a bumper), so run
    them with `require_touch_off: false` and centre declared on the locked
    output. The window stays armed and trips if the plate joint lets go.
    """
    plans = []

    # Output lockout rerun, after the 2026-09-15 slip at -196 Nm. Low / high /
    # low: the two +/-50 Nm blocks bracket the high one, so a joint that shifts
    # under +/-180 Nm shows up as a changed zero crossing, and peak-anchored
    # cycles show whether a slip repeats (ratchets) or was a one-off settle.
    def lockout(seg_id, amp, cycles):
        return {
            'id': seg_id,
            'repeats': 1,
            'lead_in_s': 3.0,           # at-rest baseline to tare against
            'primary': {'motor': 'output', 'control_mode': 'torque'},
            'secondary': {'control_mode': 'torque', 'levels': [0.0],
                          'rate': 1.0, 'settle_s': 0.0},
            'pattern': 'sawtooth',
            'params': {'amplitude': amp, 'rate': 20.0, 'peak_dwell_s': 2.0,
                       'bipolar': True, 'start_at_peak': True,
                       'cycles': cycles, 'end_dwell_s': 3.0},
        }
    high = min(180.0, 0.9 * cfg['output_torque_safety'])
    plans.append(({'name': 'archimedes_bench_output_lockout', 'segments': [
        lockout('LOW_A', 50.0, 2), lockout('HIGH', high, 3), lockout('LOW_B', 50.0, 2)]},
        [f'+/-50, +/-{high:g}, +/-50 Nm at 20 Nm/s, peak-anchored; input not coupled']))

    # Input spin: the Step 4 forward velocity-ramp shuttles, unchanged, split
    # into three bands so each is a deliberate step up. With no gearbox the
    # input cell only sees what hangs on its far side, so this is a drive and
    # instrumentation shakedown, not a drag baseline: tracking at 4500 rad/s^2
    # corners, corner torque against the 15 Nm input safety, velocity overshoot
    # against 420 rad/s, cell noise vs speed, cycle timing at speed.
    for band, lo, hi in (('low', 0, 300), ('mid', 301, 1800), ('high', 1801, 3600)):
        notes, segs = [], []
        for rpm in [x for x in VEL_RAMP_RPM if lo <= x <= hi]:
            seg = plan_shuttle(f'V{rpm:04d}', 'input', rpm, TRIP_ALLOWANCE_RAD['no_load'],
                               cfg, notes, MAX_CYCLES)
            if seg:
                segs.append(seg)
        notes.append(f'input shuttles {lo}-{hi} rpm, output 0 Nm (locked)')
        plans.append(({'name': f'archimedes_bench_input_spin_{band}', 'segments': segs},
                      notes))
    return plans


# --- checks ------------------------------------------------------------------------

def window_margin(test_file, cfg):
    """Worst |output| + stopping distance over the expanded plan, output rad,
    and the allowance it was sized with. Input-commanded positions map to the
    output through the ratio magnitude; the window is symmetric, so the
    unresolved ratio sign does not matter here."""
    tl = test_preview.expand_test(test_file, MODE, cfg['limits'],
                                  max_cycles=50_000_000)
    t = tl['t']
    worst = 0.0
    for key, shaft in (('input_position', cfg['ratio']), ('output_position', 1.0)):
        x = tl.get(key)
        if x is None or not np.isfinite(x).any():
            continue
        x = np.where(np.isfinite(x), x, 0.0) / shaft
        v = np.gradient(x, t)
        outward = np.where(x >= 0, v, -v).clip(min=0)
        stop = outward ** 2 / (2 * cfg['stop_decel']) if cfg['stop_decel'] > 0 else 0
        worst = max(worst, float(np.max(np.abs(x) + stop)))
    return worst, float(t[-1]) if len(t) else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--shakedown', action='store_true')
    ap.add_argument('--bench', action='store_true',
                    help='only the plans runnable with the output locked and no gearbox')
    args = ap.parse_args()

    cfg = load_config()
    tests_dir = dyno_paths.dyno_test_directory
    print(f'{MODE}: ratio {cfg["ratio"]:g}, window +/-{cfg["half_window"]:g} rad, '
          f'stop_decel {cfg["stop_decel"]:g} rad/s^2, unit {UNIT}'
          + ('  [SHAKEDOWN]' if args.shakedown else '')
          + ('  [BENCH]' if args.bench else ''))
    if GRAVITY_AMPLITUDE_NM is not None and GRAVITY_AMPLITUDE_NM > 0.2 * EFF_STEP_OUT_NM:
        print(f'WARNING: flange gravity {GRAVITY_AMPLITUDE_NM:g} Nm is over 20% of the '
              f'{EFF_STEP_OUT_NM:g} Nm efficiency step: low levels will change '
              'driving/driven regime mid-traverse unless compensated')

    failed = False
    plans = build_bench_plans(cfg) if args.bench else build_plans(cfg, args.shakedown)
    for recipe, notes in plans:
        if recipe is None:
            for n in notes:
                print(f'  {n}')
            continue
        issues = [f'{s["id"]}: {i}' for s in recipe['segments']
                  for i in test_builder.validate_segment(s, cfg['limits'])]
        if not recipe['segments']:
            issues.append('no segments')
        if issues:
            failed = True
            print(f'\n{recipe["name"]}: NOT WRITTEN')
            for i in issues + notes:
                print(f'  {i}')
            continue
        test_file = test_builder.save_test(recipe, tests_dir)
        worst, duration = window_margin(test_file, cfg)
        flag = 'OK' if worst <= cfg['half_window'] else 'TRIPS'
        failed |= flag != 'OK'
        print(f'\n{test_file}  ({duration / 60:.1f} min, '
              f'{len(recipe["segments"])} segment(s))')
        print(f'  window: worst |x| + stop distance {worst:.3f} rad '
              f'vs {cfg["half_window"]:g}  {flag}')
        for n in notes:
            print(f'  {n}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
