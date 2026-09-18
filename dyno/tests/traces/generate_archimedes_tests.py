"""Draft plan §7 Step 4 test plans for the Archimedes drive, as builder recipes.

See dyno/docs/archimedes_gearbox_implementation_plan.md (§7 Step 4) and the
implementation log (findings 9, 13, 16). Every plan here position-commands one
shaft and runs the other in torque mode, and every shuttle ends short of the
window trip line.

FORWARD ONLY (2026-09-17). The unit will not back-drive well enough to test it
that way in this batch, so by default this script writes only plans in which the
**input is the driving shaft**: the input either commands position or ramps
torque, and the output either hangs at 0 Nm or is grounded by the absorber's
position loop. Nothing here asks the output to turn the input. Two plans the
customer's list drives from the output were rescripted rather than dropped:

  * gravity map -- the flange is now walked through its travel by the INPUT with
    the output at 0 Nm, instead of shuttling the output with the input free.
  * stiffness -- the INPUT now ramps torque (the output-referred target divided
    by the ratio) against an output held by the absorber, instead of ramping
    output torque against a held input. Same wind-up, measured from the drive
    side. The reacted torque still reads on the output cell, which is the live
    one (log finding 12).

The three genuinely back-driven plans (back-drive velocity ramp, back-drive
efficiency, back-drive static slip) are NOT written unless --include-backdrive
is passed. They are kept, not deleted, so they come back the moment the customer's
engineers settle how they want back-drive covered.

The minimal forward set is five plans:

    archimedes_fwd_gravity_map     flange gravity + drag vs angle, 2 speeds
    archimedes_fwd_slip            forward static slip torque
    archimedes_fwd_stiffness       45% / 90% of slip, 5 s dwell, both ways
    archimedes_fwd_velocity_ramp   no-load speed sweep (also feeds torque ripple)
    archimedes_fwd_efficiency      5 Nm output steps at 20 / 300 / 1500 / 3000 rpm

plus `archimedes_fwd_megabatch`, the same five concatenated into one plan, in run
order, so the whole batch is a single selection in the GUI.

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

--include-backdrive additionally writes the three output-driven plans.
--no-megabatch skips the combined plan.
"""
import argparse
import math
import os
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
# MEASURED 2026-09-17, gbx_1p2p0/fwd_passes/stiction_and_slip. The output cell
# saturates at ~53.7 Nm on the +ve ramp while the input command keeps climbing
# past 59 Nm output-referred -- the traction-limit signature -- and the -ve ramp
# agrees at ~53-55 Nm. Roughly HALF the unit's 100 Nm rating: this unit does not
# meet spec, which is consistent with the sticky points and its history.
T_SLIP_OUT_NM = 53.0
T_SLIP_PROVISIONAL_NM = 140.0
# Where measurable micro-slip (creep) starts, same run: the ratio residual leaves
# its 0.3 mrad noise band at ~20 Nm on the output cell and grows from there. The
# customer's 5 % creep limit is the thing this threatens, well below gross slip.
T_CREEP_ONSET_OUT_NM = 20.0
# Superseded by the measurement above; kept as the bound the earlier position-held
# runs established, since those logs are still referenced.
T_SLIP_HELD_OUT_NM = 11.5
# Drivetrain static breakaway (stiction) on the INPUT shaft, from the same runs:
# mean |T| over 6 ramps, 0.243 Nm on the first run and 0.161 Nm on the second,
# taken back to back. This is what those runs actually measured -- the whole
# train broke free of static friction and turned TOGETHER at the gear ratio; the
# contact never let go. Kept here because it is a real, repeatable number the
# customer's report wants, and because the SLIP segment has to arm above it.
T_BREAKAWAY_IN_NM = 0.243
T_BREAKAWAY_MAX_IN_NM = 0.359          # worst single ramp, run 1
# Output gravity torque amplitude (m*g*r of the flange), from archimedes_fwd_gravity_map
# or CAD. Not used to shape commands yet -- recorded so the summary can warn when
# it rivals the smallest efficiency step.
GRAVITY_AMPLITUDE_NM = None
# ------------------------------------------------------------------------------

# Customer-imposed hard limit on output shaft torque (IMSystems, 2026-09-17).
# EVERY output-referred ceiling below is clipped to this, so relaxing it is one
# edit. It is a limit on what this script COMMANDS -- see `output_ceiling` for
# why that is not the same as the rig enforcing it.
OUTPUT_TORQUE_CAP_NM = 100.0

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
# Slip-detector settings, sized off the 2026-09-17 breakaway runs (see
# T_BREAKAWAY_IN_NM). The input creeps at a few tenths of a rad/s while it winds
# the absorber position loop up, and stick-slip lurches on this unit sustained
# 2 rad/s for as long as 321 ms -- so the threshold clears the creep and the
# debounce outlasts the worst lurch seen. A genuine slip does not stop.
SLIP_V_THRESH_RAD_S = 2.0
SLIP_DEBOUNCE_CYCLES = 500          # 500 ms at the 1 kHz control rate
SLIP_ARM_MARGIN = 1.6               # x the worst stiction ramp / the torque held
# Efficiency levels stop at this fraction of a MEASURED slip torque. Above the
# traction limit a level does not load the drive, it grinds it.
EFF_SLIP_MARGIN = 0.8
# Once slip is measured, the slip hunt's own ceiling drops to this multiple of
# it. There is no reason to be able to command 100 Nm at a contact that lets go
# at 53, and a lower ceiling bounds a runaway.
SLIP_CEILING_MARGIN = 1.4
# Plans the megabatch leaves out, by tag. BRK ends in a deliberate slip that no
# safety on this rig stops (2026-09-17): it needs an operator watching, so it
# does not belong in a 50-minute unattended batch.
MEGABATCH_EXCLUDE = ('BRK',)
# Tag prefix marking a back-drive plan. The megabatch filters on THIS, not on a
# bare 'B' -- which silently swallowed the 'BRK' (breakaway) plan when it was
# first written.
BACKDRIVE_TAG = 'BWD_'
# Free-output stiction ramp ceiling, input Nm. Only ~3x the breakaway the rig has
# shown, deliberately: with the output in torque mode there is nothing holding
# the train, so a ramp that runs past breakaway accelerates the input instead of
# winding anything up. See the note the plan emits.
STICTION_CEILING_IN_NM = 0.8

# Gravity-map corner acceleration, output-referred. Quasi-static traverse, so
# this only has to be gentle enough that the corners add no torque of their own;
# at 0.15 rad/s it eats 0.02 rad of a 1.33 rad throw.
GRAVITY_ACCEL_OUT = 0.5
# Cap for the megabatch's load-only expansion check (it is only there to prove
# TestManager accepts the plan; the window and duration come from the parts).
MEGABATCH_CHECK_CYCLES = 200_000

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


def output_ceiling(cfg):
    """Highest output-referred torque any plan may command, full power.

    The lower of 95 % of the `output_torque` safety (margin so a tracking
    overshoot does not nuisance-trip it) and the customer's hard cap.

    Note what this does NOT do: the cap is enforced by this script, on the
    commanded value. The rig still trips at `safeties.output_torque.limit`, so a
    fault, a bad level or a hand-edited plan can put more than the cap through
    the shaft without anything stopping it. To make the cap real, bring that
    safety down to just above the cap -- see the summary this script prints.
    """
    return min(0.95 * cfg['output_torque_safety'], OUTPUT_TORQUE_CAP_NM)


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
                      bipolar, cycles, v_thresh, debounce, arm_fraction=0.05,
                      stuck_s=0.25):
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
                   'stuck_s': stuck_s, 'debounce_cycles': debounce,
                   'arm_fraction': arm_fraction},
    }


# --- the plans -------------------------------------------------------------------

def build_plans(cfg, shakedown, include_backdrive=False):
    """Returns [(tag, recipe, notes)] in run order. Forward (input-driving)
    plans first, then the back-drive plans if they were asked for."""
    r = cfg['ratio']
    torque_scale = 0.25 if shakedown else 1.0
    cycles_cap = 1 if shakedown else MAX_CYCLES
    rpm_cap = 300 if shakedown else math.inf
    plans = []          # (tag, recipe, notes)

    # 1. Gravity map, input-driven. The flange is walked quasi-statically through
    # its travel by the INPUT; the output hangs in torque mode at 0 Nm, so the
    # drive is never asked to back-drive. Output cell torque at angle, averaged
    # across the two directions, is gravity (drag flips sign with direction and
    # cancels); half their difference is drag. Two speeds show whether that drag
    # split is speed-independent -- note both now include the input's own drag
    # and the gearbox's forward loss, which the old output-driven version put on
    # the other side of the gearbox. Run before touch-off settings are trusted.
    notes = []
    segs = []
    for seg_id, w_out in (('SLOW', 0.05), ('FAST', 0.15)):
        segs.append(shuttle_segment(seg_id, 'input', w_out * r, OUTPUT_END_RAD * r,
                                    GRAVITY_ACCEL_OUT * r, 1 if shakedown else 2))
    notes.append(f'input shuttle +/-{OUTPUT_END_RAD * r:.1f} rad '
                 f'(= +/-{OUTPUT_END_RAD} rad output) at {0.05 * r:.2f} and '
                 f'{0.15 * r:.2f} rad/s input, output 0 Nm')
    notes.append('input-driven rescript of the plan §1 gravity map: reads gravity '
                 '+ forward drag, not gravity + back-drive drag')
    plans.append(('GRAV', {'name': 'archimedes_fwd_gravity_map', 'segments': segs},
                  notes))

    # 2. Forward static slip (runs before stiffness, which needs its result).
    # Output held by the absorber's position loop, input torque ramps until the
    # traction contact lets go. Once it slips the input has almost nothing to
    # hold it, so release fast. The ceiling is the customer's output cap
    # reflected to the input -- which may well be BELOW the unit's slip torque,
    # in which case this test cannot produce the number stiffness is sized on.
    # See the note it emits.
    #
    # TWO segments, because the 2026-09-17 runs showed these are two different
    # measurements sharing one fixture (log: gbx_1p2p0/fwd_passes/breakaway*).
    #
    # STICT measures INPUT BREAKAWAY STICTION, and it holds the output in TORQUE
    # mode at 0 Nm to do it -- not position. The 2026-09-17 runs used a position
    # hold and that is why their 0.243 Nm mean is not a stiction number: a
    # position-held output is not a rigid ground, it is a spring with an
    # integrator. What those runs recorded was the input winding that spring up
    # (input travelled 2.43 rad, output followed 0.055 rad, ratio 43.77 against a
    # config 43.88 -- locked, so nothing slipped) punctuated by stick-slip
    # lurches, and the detector fired on the first lurch each time. The torque at
    # a lurch is real and repeatable but it is a property of the drivetrain AND
    # the absorber's position loop together, not of the input shaft's friction.
    # With the output at 0 Nm the only thing the ramp fights is friction, which
    # is what a breakaway test is supposed to mean.
    #
    # SLIP hunts the traction contact instead, and has to be deaf to everything
    # STICT listens for. Three changes: armed only above `arm_fraction` of the
    # ceiling (well clear of the worst stiction ramp seen), a velocity threshold
    # over the creep the shaft shows while it winds the absorber loop up, and a
    # debounce long enough that a stick-slip lurch cannot pass for a slip. A real
    # slip runs as long as torque is applied; a lurch stops.
    notes = []
    ceiling_out = output_ceiling(cfg)
    if T_SLIP_OUT_NM:
        ceiling_out = min(ceiling_out, SLIP_CEILING_MARGIN * T_SLIP_OUT_NM)
    ceiling_in = ceiling_out / r * torque_scale
    rate_in = 2.0 / r                              # 2 Nm/s output-referred
    cycles = 1 if shakedown else 3
    # Arm the slip detector above everything stiction has been seen to do, with
    # margin, and above the torque already proven held. Expressed as a fraction
    # of the ceiling because that is the knob RampBreak takes.
    slip_arm_in = max(SLIP_ARM_MARGIN * T_BREAKAWAY_MAX_IN_NM,
                      SLIP_ARM_MARGIN * T_SLIP_HELD_OUT_NM / r)
    arm_fraction = min(0.9, slip_arm_in / ceiling_in) if ceiling_in > 0 else 0.05
    stict = breakaway_segment('STICT', 'input', STICTION_CEILING_IN_NM,
                              round(rate_in, 4), release_s=0.05, rest_s=3.0,
                              bipolar=True, cycles=cycles, v_thresh=0.5,
                              debounce=3, arm_fraction=0.05)
    # The one line that makes it a stiction test rather than a wind-up test.
    stict['secondary'] = {'control_mode': 'torque', 'levels': [0.0],
                          'rate': 1.0, 'settle_s': 1.0}
    segs = [
        stict,
        # ONE ramp, ONE direction. RampBreak releases on a breakaway and goes
        # straight into the next ramp; on 2026-09-17 that second ramp drove the
        # already-slipping contact to a free-spin at 155 rad/s of input, which
        # nothing on this rig stops (no safety watches the ratio) -- the operator
        # E-stopped it. Until a ratio-break safety exists, a slip hunt gets
        # exactly one attempt per run.
        breakaway_segment('SLIP', 'input', ceiling_in, round(rate_in, 4),
                          release_s=0.05, rest_s=3.0, bipolar=False,
                          cycles=1, v_thresh=SLIP_V_THRESH_RAD_S,
                          debounce=SLIP_DEBOUNCE_CYCLES,
                          arm_fraction=round(arm_fraction, 4)),
    ]
    notes.append(f'ramp rate {rate_in * r:.0f} Nm/s output-referred '
                 f'({rate_in:.4f} Nm/s input) in both segments')
    notes.append(f'STICT: input breakaway stiction, output in TORQUE mode at 0 Nm '
                 f'(not position -- see the comment). Ramps only to '
                 f'{STICTION_CEILING_IN_NM:g} Nm input')
    notes.append(f'  the 2026-09-17 position-held runs gave {T_BREAKAWAY_IN_NM:g} Nm mean; '
                 'expect this to read lower and cleaner, since it no longer includes '
                 'the torque spent winding the absorber position loop')
    notes.append('  CAUTION: with the output free, a ramp that runs past breakaway '
                 'ACCELERATES the input instead of winding up. The detector fired on '
                 '12 of 12 ramps on 2026-09-17, the ceiling is low, and the 500 rad/s '
                 'input / 12 rad/s output velocity safeties backstop it -- but watch '
                 'the first ramp')
    notes.append(f'SLIP: traction contact, ONE ramp one direction, armed above '
                 f'{arm_fraction * ceiling_in:.3f} Nm input '
                 f'(= {arm_fraction * ceiling_in * r:.0f} Nm output-referred), '
                 f'{SLIP_V_THRESH_RAD_S:g} rad/s sustained {SLIP_DEBOUNCE_CYCLES} cycles '
                 f'({SLIP_DEBOUNCE_CYCLES:g} ms)')
    notes.append(f'  slip IS measured: {T_SLIP_OUT_NM:g} Nm output (2026-09-17), about half '
                 f'the {EFF_MAX_OUT_NM[UNIT]:g} Nm rating. Re-run this only to confirm or to '
                 'track degradation -- every run costs contact')
    notes.append('  NOTHING ON THIS RIG STOPS A SLIP. The output stays put and the input '
                 'spins, so the window, both velocity safeties and both torque safeties all '
                 'stay happy. Watch it, and keep a hand on the E-stop')
    if T_SLIP_OUT_NM:
        notes.append(f'SLIP ceiling is {SLIP_CEILING_MARGIN:g} x the measured slip '
                     f'({ceiling_out:.0f} Nm output), not the '
                     f'{OUTPUT_TORQUE_CAP_NM:g} Nm customer cap: no reason to be able to '
                     'command 100 Nm at a contact that lets go at 53')
    elif OUTPUT_TORQUE_CAP_NM < 0.95 * cfg['output_torque_safety']:
        notes.append(f'CAPPED at the customer\'s {OUTPUT_TORQUE_CAP_NM:g} Nm output limit '
                     f'(the safety alone would have allowed '
                     f'{0.95 * cfg["output_torque_safety"]:.0f} Nm)')
        notes.append('A unit rated 100-140 Nm should NOT slip under this ceiling: SLIP '
                     'hitting the ceiling is the result ("no slip up to the cap"), not a '
                     'failed run. Stiffness then stays on the provisional figure')
    brk_index = len(plans)
    plans.append(('BRK', {'name': 'archimedes_fwd_slip', 'segments': segs}, notes))

    # 3. Stiffness, input-driven. Plan §1 asks for output torque to 45% / 90% of
    # slip with the input held; with the unit not back-driving, the same wind-up
    # is taken from the drive side instead: the INPUT ramps torque to the
    # output-referred target divided by the ratio, while the absorber grounds the
    # output at position 0 -- the same grounding the slip test above uses. Both
    # channels the result needs still work: the reacted torque reads on the
    # output cell (the live one, log finding 12) and the wind-up on the input
    # encoder, referred back through the ratio. What it cannot separate from the
    # gearbox is the absorber position loop's own compliance, exactly as the
    # output-driven version could not separate the input drive's.
    notes = []
    t_slip = T_SLIP_OUT_NM
    if t_slip is None:
        t_slip = T_SLIP_PROVISIONAL_NM
        notes.append(f'PROVISIONAL: slip torque not measured, sized on {t_slip:g} Nm')
    segs = []
    clipped = []
    for frac in (0.45, 0.90):
        want_out = frac * t_slip
        amp_out = min(want_out, output_ceiling(cfg))
        if amp_out < want_out - 1e-9:
            clipped.append((frac, want_out, amp_out))
        amp_out *= torque_scale
        amp_in = amp_out / r
        segs.append({
            'id': f'K{int(frac * 100)}',
            'repeats': 1,
            'lead_in_s': test_builder.START_HOLD_S,
            'primary': {'motor': 'input', 'control_mode': 'torque'},
            'secondary': {'control_mode': 'position', 'levels': [0.0],
                          'rate': 0.1, 'settle_s': 1.0},
            'pattern': 'sawtooth',
            'params': {'amplitude': round(amp_in, 4), 'rate': round(10.0 / r, 4),
                       'peak_dwell_s': 5.0, 'bipolar': True,
                       'start_at_peak': False, 'cycles': 1, 'end_dwell_s': 2.0},
        })
    notes.append(f'input torque to +/-{amp_in:.3f} Nm (= {amp_in * r:.0f} Nm output) '
                 f'at {10.0:g} Nm/s output-referred, 5 s dwell at each peak')
    for frac, want_out, got_out in clipped:
        notes.append(f'CAPPED: the {frac * 100:.0f}% point wants {want_out:.0f} Nm output '
                     f'but the customer\'s {OUTPUT_TORQUE_CAP_NM:g} Nm limit allows '
                     f'{got_out:.0f} Nm -- that peak is really {100 * got_out / t_slip:.0f}% '
                     'of slip. The two dwells are no longer 45/90 and the plan §1 '
                     'wording does not hold; say so in the report or get the limit raised')
    notes.append('input-driven rescript: absorber holds the output, wind-up off the '
                 'input encoder / r; stiffness = dT_out / d(theta_in / r)')
    plans.append(('STIF', {'name': 'archimedes_fwd_stiffness', 'segments': segs},
                  notes))

    # 4. Velocity ramp, no load, forward only (plan §1: 30 rpm steps to 300, then
    # 300 rpm steps). Also the source for the torque ripple numbers (plan §6
    # caps those at 600 rpm).
    notes, segs = [], []
    rpms = [30, 150, 300] if shakedown else [x for x in VEL_RAMP_RPM if x <= rpm_cap]
    for rpm in rpms:
        seg = plan_shuttle(f'V{rpm:04d}', 'input', rpm, TRIP_ALLOWANCE_RAD['no_load'],
                           cfg, notes, cycles_cap)
        if seg:
            segs.append(seg)
    notes.append('torque ripple is reported from the <=600 rpm points only (plan §6)')
    plans.append(('VEL', {'name': 'archimedes_fwd_velocity_ramp', 'segments': segs},
                  notes))

    # 5. Efficiency, forward only, all four speeds in one plan (one segment
    # each). Levels alternate sign so creep drift cancels within the segment
    # (plan §4). Each level runs full shuttle cycles, so the load resists the
    # input on half the traverses and assists it on the other half: the resisting
    # ones are the forward-efficiency data, the assisting ones are the return
    # stroke and should be binned out. The input commands position throughout, so
    # even on the assisting traverses the gearbox is never asked to back-drive --
    # the input simply absorbs.
    notes, segs = [], []
    want_top = EFF_MAX_OUT_NM[UNIT]
    top_out = min(want_top, output_ceiling(cfg))
    # A measured slip torque overrides both: driving an efficiency level past the
    # traction limit does not measure efficiency, it grinds the contact. Leave
    # margin, and round down to a whole number of steps.
    if T_SLIP_OUT_NM:
        slip_top = EFF_SLIP_MARGIN * T_SLIP_OUT_NM
        slip_top = EFF_STEP_OUT_NM * math.floor(slip_top / EFF_STEP_OUT_NM)
        if slip_top < top_out:
            notes.append(f'SLIP-LIMITED: measured slip is {T_SLIP_OUT_NM:g} Nm output, so '
                         f'the sweep stops at {slip_top:g} Nm '
                         f'({EFF_SLIP_MARGIN:g} x slip, rounded down to a step) instead of '
                         f'{top_out:g} Nm. Levels above the traction limit would slip, not '
                         'load')
            notes.append(f'  the customer asked for {want_top:g} Nm; this unit cannot carry '
                         f'it. Expect creep to break their 5 % limit from about '
                         f'{T_CREEP_ONSET_OUT_NM:g} Nm upward')
            top_out = slip_top
    if top_out < want_top - 1e-9 and not (T_SLIP_OUT_NM and top_out <= EFF_SLIP_MARGIN * T_SLIP_OUT_NM):
        notes.append(f'CAPPED: {UNIT} was requested to {want_top:g} Nm output but the '
                     f'customer\'s {OUTPUT_TORQUE_CAP_NM:g} Nm limit stops the sweep at '
                     f'{top_out:g} Nm -- the top of the requested efficiency range is '
                     'not covered')
    top_out *= torque_scale
    levels = [round(v, 4) for v in alternating(EFF_STEP_OUT_NM, top_out)]
    for rpm in EFF_RPM:
        if rpm > rpm_cap:
            continue
        seg = plan_shuttle(f'E{rpm:04d}', 'input', rpm, TRIP_ALLOWANCE_RAD['loaded_fwd'],
                           cfg, notes, cycles_cap, levels=levels,
                           level_rate=50.0, settle_s=1.0)
        if seg:
            segs.append(seg)
    notes.append(f'{UNIT}: {len(levels)} output-cell levels to +/-{top_out:g} Nm, '
                 f'{len(segs)} speed(s) back to back')
    notes.append('one plan per speed became one plan for all four: check output '
                 'position between segments if creep runs, the window trip is the backstop')
    plans.append(('EFF', {'name': 'archimedes_fwd_efficiency', 'segments': segs},
                  notes))

    # Slip runs LAST. Plan §7 put it first because stiffness needed its number;
    # that number is now a measured constant, and a slip hunt degrades the
    # contact, so anything measured after one is measured on a different unit.
    plans.append(plans.pop(brk_index))

    if include_backdrive:
        plans += build_backdrive_plans(cfg, shakedown)

    if shakedown:
        for _, recipe, _ in plans:
            if recipe:
                recipe['name'] += '_shakedown'
    return plans


def build_backdrive_plans(cfg, shakedown):
    """The output-driven plans, off by default (2026-09-17: the unit will not
    back-drive well enough to run them in this batch). Unchanged from the
    original draft so they come back as they were."""
    r = cfg['ratio']
    torque_scale = 0.25 if shakedown else 1.0
    cycles_cap = 1 if shakedown else MAX_CYCLES
    rpm_cap = 300 if shakedown else math.inf
    plans = []

    notes, segs = [], []
    rpms = [30, 150, 300] if shakedown else [x for x in VEL_RAMP_RPM if x <= rpm_cap]
    for rpm in rpms:
        seg = plan_shuttle(f'V{rpm:04d}', 'output', rpm, TRIP_ALLOWANCE_RAD['backdrive'],
                           cfg, notes, cycles_cap)
        if seg:
            segs.append(seg)
    plans.append((BACKDRIVE_TAG + 'VEL', {'name': 'archimedes_bwd_velocity_ramp', 'segments': segs},
                  notes))

    top_out = min(EFF_MAX_OUT_NM[UNIT], output_ceiling(cfg)) * torque_scale
    scale = 1.0 / r                                   # input levels, output-referred
    for rpm in EFF_RPM:
        if rpm > rpm_cap:
            continue
        notes = []
        levels = [round(v * scale, 4) for v in alternating(EFF_STEP_OUT_NM, top_out)]
        seg = plan_shuttle(f'E{rpm:04d}', 'output', rpm, TRIP_ALLOWANCE_RAD['backdrive'],
                           cfg, notes, cycles_cap, levels=levels,
                           level_rate=round(50.0 * scale, 4), settle_s=1.0)
        notes.append(f'{UNIT}: {len(levels)} levels to +/-{top_out:g} Nm output-referred')
        plans.append((BACKDRIVE_TAG + f'EFF{rpm}',
                      {'name': f'archimedes_bwd_efficiency_{rpm:04d}rpm',
                       'segments': [seg]} if seg else None, notes))

    # Back-drive static slip, last (plan §7): input held, output torque ramps.
    # When it slips the output moves, so one ramp per direction from centre
    # leaves ~1.3 rad of travel each way before the window. RampBreak has no
    # single-direction option, so the plan's "start near the bumper" variant
    # needs a small code change if slips turn out to travel further than that.
    ceiling_out = output_ceiling(cfg) * torque_scale
    plans.append((BACKDRIVE_TAG + 'SLIP', {'name': 'archimedes_bwd_slip', 'segments': [
        breakaway_segment('RAMP', 'output', ceiling_out, 2.0, release_s=0.25,
                          rest_s=3.0, bipolar=True, cycles=1, v_thresh=0.05,
                          debounce=5)]},
                  [f'output ramp to {ceiling_out:.0f} Nm at 2 Nm/s, once each way from centre']))
    return plans


def build_megabatch(plans, name):
    """Every forward plan's segments concatenated, in run order, as one recipe.

    Segment ids are prefixed with the plan's tag so the run log and the trace
    filenames still say which test each segment came from. Nothing else about a
    segment changes, so the megabatch runs exactly what the individual plans run
    -- which is why the window margin below is taken from the parts rather than
    re-derived over a two-hour expansion."""
    segs = []
    for tag, recipe, _ in plans:
        if not recipe:
            continue
        for seg in recipe['segments']:
            segs.append(dict(seg, id=f'{tag}_{seg["id"]}'))
    return {'name': name, 'segments': segs}


# --- checks ------------------------------------------------------------------------

def window_margin(test_file, cfg, max_cycles=50_000_000):
    """Worst |output| + stopping distance over the expanded plan, output rad,
    and the allowance it was sized with. Input-commanded positions map to the
    output through the ratio magnitude; the window is symmetric, so the
    unresolved ratio sign does not matter here."""
    tl = test_preview.expand_test(test_file, MODE, cfg['limits'],
                                  max_cycles=max_cycles)
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


def write_plan(recipe, notes, cfg, tests_dir):
    """Validate, save and report one plan. Returns (test_file, worst, duration,
    ok) or None if it was not written at all."""
    issues = [f'{s["id"]}: {i}' for s in recipe['segments']
              for i in test_builder.validate_segment(s, cfg['limits'])]
    if not recipe['segments']:
        issues.append('no segments')
    if issues:
        print(f'\n{recipe["name"]}: NOT WRITTEN')
        for i in issues + notes:
            print(f'  {i}')
        return None
    test_file = test_builder.save_test(recipe, tests_dir)
    worst, duration = window_margin(test_file, cfg)
    flag = 'OK' if worst <= cfg['half_window'] else 'TRIPS'
    print(f'\n{test_file}  ({duration / 60:.1f} min, '
          f'{len(recipe["segments"])} segment(s))')
    print(f'  window: worst |x| + stop distance {worst:.3f} rad '
          f'vs {cfg["half_window"]:g}  {flag}')
    for n in notes:
        print(f'  {n}')
    return test_file, worst, duration, flag == 'OK'


def report_stale(tests_dir, written, shakedown):
    """Generated archimedes_*.yaml left over from an earlier run with different
    plans -- back-drive files above all. Listed, never deleted: the operator
    picks tests by filename in the GUI, so a stale one is a wrong-test risk.

    Only the variant this run writes is considered: a `--shakedown` run does not
    call the full-power files stale, or the other way round, and the `--bench`
    plans (a different rig state entirely) are never in scope."""
    gen_dir = os.path.join(tests_dir, test_builder.GENERATED_TEST_DIR)
    if not os.path.isdir(gen_dir):
        return
    keep = {os.path.basename(f) for f in written}
    stale = sorted(f for f in os.listdir(gen_dir)
                   if f.startswith('archimedes_') and f.endswith('.yaml')
                   and not f.startswith('archimedes_bench_')
                   and f.endswith('_shakedown.yaml') == shakedown
                   and f not in keep)
    if stale:
        print(f'\nSTALE in {test_builder.GENERATED_TEST_DIR}/ (not written this run, '
              'still selectable in the GUI):')
        for f in stale:
            print(f'  {f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--shakedown', action='store_true')
    ap.add_argument('--bench', action='store_true',
                    help='only the plans runnable with the output locked and no gearbox')
    ap.add_argument('--include-backdrive', action='store_true',
                    help='also write the output-driven plans (off by default: the '
                         'unit is not back-driving well enough to test that way)')
    ap.add_argument('--no-megabatch', action='store_true',
                    help='skip the combined all-forward-tests plan')
    args = ap.parse_args()

    cfg = load_config()
    tests_dir = dyno_paths.dyno_test_directory
    print(f'{MODE}: ratio {cfg["ratio"]:g}, window +/-{cfg["half_window"]:g} rad, '
          f'stop_decel {cfg["stop_decel"]:g} rad/s^2, unit {UNIT}, '
          f'output cap {OUTPUT_TORQUE_CAP_NM:g} Nm'
          + ('  [SHAKEDOWN]' if args.shakedown else '')
          + ('  [BENCH]' if args.bench else '')
          + ('' if args.bench else
             '  [+BACKDRIVE]' if args.include_backdrive else '  [FORWARD ONLY]'))
    if GRAVITY_AMPLITUDE_NM is not None and GRAVITY_AMPLITUDE_NM > 0.2 * EFF_STEP_OUT_NM:
        print(f'WARNING: flange gravity {GRAVITY_AMPLITUDE_NM:g} Nm is over 20% of the '
              f'{EFF_STEP_OUT_NM:g} Nm efficiency step: low levels will change '
              'driving/driven regime mid-traverse unless compensated')

    if args.bench:
        plans = build_bench_plans(cfg)
    else:
        plans = build_plans(cfg, args.shakedown, args.include_backdrive)

    failed = False
    written = []
    forward = []
    total_s = 0.0
    worst_all = 0.0
    for tag, recipe, notes in plans:
        if recipe is None:            # a speed the window could not fit
            for n in notes:
                print(f'  {n}')
            continue
        result = write_plan(recipe, notes, cfg, tests_dir)
        if result is None:
            failed = True
            continue
        test_file, worst, duration, ok = result
        failed |= not ok
        written.append(test_file)
        if not tag.startswith(BACKDRIVE_TAG) and tag not in MEGABATCH_EXCLUDE:
            forward.append((tag, recipe, notes))
            total_s += duration
            worst_all = max(worst_all, worst)

    if not args.bench and not args.no_megabatch and forward:
        name = 'archimedes_fwd_megabatch' + ('_shakedown' if args.shakedown else '')
        mega = build_megabatch(forward, name)
        issues = [f'{s["id"]}: {i}' for s in mega['segments']
                  for i in test_builder.validate_segment(s, cfg['limits'])]
        if issues:
            failed = True
            print(f'\n{name}: NOT WRITTEN')
            for i in issues:
                print(f'  {i}')
        else:
            test_file = test_builder.save_test(mega, tests_dir)
            written.append(test_file)
            # Load-only check: TestManager parses the plan and every behavior
            # expands, capped so a two-hour timeline never has to be held in
            # memory. Window and duration are the parts' -- the megabatch is
            # their concatenation, segment for segment.
            window_margin(test_file, cfg, max_cycles=MEGABATCH_CHECK_CYCLES)
            flag = 'OK' if worst_all <= cfg['half_window'] else 'TRIPS'
            failed |= flag != 'OK'
            print(f'\n{test_file}  ({total_s / 60:.1f} min, '
                  f'{len(mega["segments"])} segment(s))')
            print(f'  window: worst |x| + stop distance {worst_all:.3f} rad '
                  f'vs {cfg["half_window"]:g}  {flag}  (max over the parts)')
            print('  ' + ' -> '.join(tag for tag, _, _ in forward))
            print('  every forward test back to back, one selection; the individual '
                  'plans above run the same segments if you need to split it')
            for tag in MEGABATCH_EXCLUDE:
                left = [r['name'] for t_, r, _ in plans if t_ == tag and r]
                for name in left:
                    print(f'  NOT in the batch: {name} -- it ends in a slip that no safety '
                          'on this rig stops. Run it on its own, watching, afterwards')

    if not args.bench:
        report_stale(tests_dir, written, args.shakedown)
    if OUTPUT_TORQUE_CAP_NM < cfg['output_torque_safety']:
        print(f'\nNOTE: the {OUTPUT_TORQUE_CAP_NM:g} Nm output cap is enforced by THIS '
              f'SCRIPT, on commanded values only.\n  The rig still trips at '
              f'safeties.output_torque.limit = {cfg["output_torque_safety"]:g} Nm, so a '
              'fault, a hand-edited plan or a\n  plan generated before the cap can put '
              'more than the cap through the output shaft with\n  nothing stopping it. To '
              f'make the limit real, set that safety to ~{OUTPUT_TORQUE_CAP_NM * 1.1:.0f} '
              'Nm in\n  '
              f'{MODE}_dyno_config.yaml and regenerate (the ceilings follow it down).')
    return 1 if failed else 0


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
    # The lockout baseline only has to cover the range the gearbox tests use, so
    # it takes the customer's cap too -- even though the unit is not fitted for it.
    high = min(180.0, 0.9 * cfg['output_torque_safety'], OUTPUT_TORQUE_CAP_NM)
    plans.append(('LOCK', {'name': 'archimedes_bench_output_lockout', 'segments': [
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
        plans.append((f'SPIN_{band.upper()}',
                      {'name': f'archimedes_bench_input_spin_{band}', 'segments': segs},
                      notes))
    return plans


if __name__ == '__main__':
    sys.exit(main())
