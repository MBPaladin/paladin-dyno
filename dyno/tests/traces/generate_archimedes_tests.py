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

The genuinely back-driven plans are NOT written unless --include-backdrive is
passed. They are kept, not deleted, so they come back the moment the customer's
engineers settle how they want back-drive covered. Rewritten 2026-09-18 to the
four tests the bench asked for -- see build_backdrive_plans:

    archimedes_bwd_velocity_ramp   output shuttles at input-referred speeds
    archimedes_bwd_efficiency_alt  5 Nm output steps, input holds torque
    archimedes_bwd_efficiency_pos  the same, one rotation sense
    archimedes_bwd_slip            input POSITION-held, output torque ramps
    archimedes_bwd_breakaway       input at 0 Nm TORQUE, output torque ramps

The forward set is six plans:

    archimedes_fwd_gravity_map     flange gravity + drag vs angle, 2 speeds
    archimedes_fwd_breakaway       input stiction, output free at 0 Nm
    archimedes_fwd_slip            forward static slip torque
    archimedes_fwd_stiffness       45% / 90% of slip, 5 s dwell, both ways
    archimedes_fwd_velocity_ramp   no-load speed sweep (also feeds torque ripple)
    archimedes_fwd_efficiency      5 Nm output steps at 20 / 300 / 1500 / 3000 rpm

SLIP AND BREAKAWAY ARE SEPARATE PLANS (split 2026-09-18, both sides). They were
never one test. They hold the far shaft in OPPOSITE modes -- position grounds the
train so the ramp winds the traction contact up until it lets go (slip), torque
at 0 Nm grounds nothing so the ramp fights only friction until the whole train
turns (breakaway) -- they want different detectors, and only the slip hunt ends
in an event that needs an operator watching. Bundled together, the safe one could
not be run without arming the dangerous one.

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
  PYTHONPATH=. .venv/bin/python dyno/tests/traces/generate_archimedes_tests.py [flags]

  --shakedown          `*_shakedown` variants: <=300 rpm, 25% torque, one cycle.
                       Plan §7 Step 4 says run every test low first.
  --bench              only what runs with the output on the lockout plate and no
                       gearbox: the lockout rerun and the input spin bands.
  --include-backdrive  also write the five output-driven plans.
  --backdrive-only     write ONLY those.
  --no-megabatch       skip the combined plan.

Every value worth changing between units or between days on the bench is also a
flag -- `--help` lists them with their current defaults. They rebind the module
constant for one run and do not edit the file, so the defaults stay the
documented ones and the notes each plan prints describe what was actually used:

  ... --slip-torque 48 --unit 1.2.1        after re-measuring slip on a new unit
  ... --eff-rpm 20,300 --eff-step 10       a shorter efficiency sweep
  ... --backdrive-only --bwd-ramps 5       five slip/breakaway ramps each way
  ... --output-cap 140                     if IMSystems raise their limit

Anything NOT a flag is a constant at the top of this file, next to the paragraph
explaining why it has the value it has. Change it there, and say why.
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

# The 2026-09-17 free-spin, at the OUTPUT: the input ran away at 155 rad/s, which
# is 3.5 rad/s of output-referred ratio error rate; the config comment beside
# `ratio_slip` records 2.1-2.7 rad/s for it. The lower figure is used, so the
# check below only complains when the limit is clearly past the whole band.
FREE_SPIN_OUT_RAD_S = 2.1
# Where the measured slip on the same day settled, in output rad of ratio error.
MEASURED_SLIP_TRAVEL_OUT_RAD = 0.356

# Customer-imposed hard limit on output shaft torque (IMSystems, 2026-09-17).
# EVERY output-referred ceiling below is clipped to this, so relaxing it is one
# edit. It is a limit on what this script COMMANDS -- see `output_ceiling` for
# why that is not the same as the rig enforcing it.
OUTPUT_TORQUE_CAP_NM = 100.0

# Customer request (plan §1).
VEL_RAMP_RPM = list(range(30, 301, 30)) + list(range(600, 3601, 300))
# A `recentre` after every velocity-ramp speed, both directions. The ramp is
# no-load, so nothing here creeps the way a loaded efficiency shuttle does --
# but a shuttle still ends wherever its own tracking error left it, 21 speeds
# accumulate that, and the fastest points are the ones with the least window
# margin AND the most tracking error. A recentre between them costs the settle
# (0.5 s) wherever the shaft is already centred, which no-load it usually is.
# It also means a window trip at speed 17 resumes at speed 17 from centre,
# instead of from wherever the trip left the output.
VEL_RECENTRE_BETWEEN_SPEEDS = True
EFF_RPM = [20, 300, 1500, 3000]
# Speeds run with only their TOP N torque magnitudes, BACK-DRIVE only (operator
# call, 2026-09-18). {rpm: how many magnitudes to keep}. Forward always runs the
# customer's full ladder.
#
# 20 rpm back-driving is the worst cell-resolution case on this rig and by far
# the most expensive, and the two problems have the same cause. The INPUT is the
# shaft holding torque here, so a 5 Nm output step is 5/43.88 = 0.114 Nm at the
# input cell -- 0.57% of its 20 Nm full scale, against a session tare whose own
# scatter was sd 0.0092 Nm (2026-09-18). Roughly 12:1 on the smallest level
# before any drag or ripple is subtracted, and efficiency is a RATIO of two such
# numbers, so that error lands on the answer twice. Forward does not have this
# problem: there the step is read on the output cell at full magnitude.
#
# Signal-to-noise scales with the level, so the top of the ladder is worth
# keeping even where the bottom is not: at 40 Nm the input cell sees 0.91 Nm,
# about 8x the smallest step. Keeping the top two magnitudes preserves the
# customer's 20 rpm line where it is most trustworthy and drops the levels that
# would have produced numbers nobody should quote.
#
# The cost side is the same arithmetic: at 0.0477 rad/s of output the 20 rpm
# levels dominate the sweep's run time, so trimming to two magnitudes takes the
# alt variant from ~41 min to ~16 and pos from ~21 to ~8.
#
# Set a speed to 0 to drop it entirely; remove it from the map to run it whole.
BWD_EFF_TOP_LEVELS = {20: 2}
EFF_STEP_OUT_NM = 5.0
EFF_MAX_OUT_NM = {'1.2.0': 100.0, '1.2.1': 100.0, '1.2.2': 140.0}

# Nominal throw: where a traverse ends when nothing forces it inward.
OUTPUT_END_RAD = 1.35
# --- commanded-vs-actual overshoot -------------------------------------------
# The shaft does not stop where it is told. It overshoots the turnaround by an
# amount PROPORTIONAL TO SPEED -- a pure time lag -- and which shaft is being
# position-commanded changes the answer by a factor of fifteen.
#
# MEASURED 2026-09-18 from the velocity ramps, peak |output| minus the 1.330
# commanded, per speed:
#
#     rpm_in    BACK-DRIVE (absorber commands)   FORWARD (input commands)
#        600            +0.064                        +0.003
#       1200            +0.145                        +0.006
#       1800            +0.175                        +0.009
#       2400            +0.258                        +0.015
#       3600             (tripped first)              +0.031
#
# Back-drive: 36 segments over 5 runs, fit 0.0463 * v_out - 0.016, residual
# sd 0.022 rad -- a 46 ms lag. Forward: 3.1 ms, and the whole 30-3600 rpm sweep
# ran to completion (fwd_passes/speed_sweeps).
#
# WHY THE TWO DIFFER SO MUCH, and why this is not a number to "tune away":
# forward, the INPUT commands position and the output sees that shaft's tracking
# error divided by 43.88, through a reduction, from a 3.6e-4 kg m^2 rotor that
# corners at 7500 rad/s^2. Back-driving, the ABSORBER commands position on the
# big-inertia output directly, flange and all, at 2000 rad/s^2 and a much slower
# loop. The back-drive figure is the absorber's own position loop, so it belongs
# to the rig, not to the gearbox, and it will not improve without retuning that
# drive.
OVERSHOOT_LAG_S = {
    'output': 0.055,        # back-drive; fit says 0.052, rounded up
    'input': 0.004,         # forward; fit says 0.0031, rounded up
}
# SECOND TERM, added 2026-09-18 after four back-drive efficiency runs aborted.
# The no-load lag above is not the whole story: a shuttle carrying a torque
# LEVEL overshoots further, and the extra is proportional to the level.
#
# The four aborts (bkw_passes/efficiency_pt_1..4) were all position trips, none
# at a high torque -- pt_3 and pt_4 both died at E1500_P15, a 15 Nm level on a
# 40 Nm ladder, commanded to the 1.271 rad the no-load model had already pulled
# them back to. They still reached 1.58.
#
# The effect is ONE-SIDED and follows the sign of the torque. Measured at
# 300 rpm, overshoot on each side of the traverse:
#
#     level    + side    - side
#      +15     -0.088    +0.118
#      -15     +0.116    -0.067
#      +25     -0.124    +0.131
#      -25     +0.192    -0.084
#
# The peak grows on the side the torque pushes and SHRINKS by about as much on
# the other (means over every loaded segment: +0.148 pushed, -0.024 other). That
# is a position loop holding station against a constant disturbance torque --
# the fitted 0.008 rad/Nm is a stiffness of ~125 Nm/rad at the absorber, in the
# same range as the 304 Nm/rad a plain least-squares over everything gives.
#
# It also FADES OUT at low speed: 40 Nm at 20 rpm overshoots 0.040 rad, against
# 0.203 for the same level at 300 rpm. Below ~300 rpm the loop has time to
# converge against the disturbance before the turnaround arrives, so the term is
# scaled by min(1, v / OVERSHOOT_TORQUE_V_REF).
OVERSHOOT_TORQUE_RAD_PER_NM = 0.008
OVERSHOOT_TORQUE_V_REF = 0.716          # rad/s output = 300 rpm input
# Added on top of lag * v, to cover the scatter around the fit (sd 0.022 rad,
# worst residual 0.044) and the fact that the two fastest back-drive points were
# themselves truncated by the trip, so their true overshoot is larger than what
# was logged. ~2.3 sigma.
OVERSHOOT_MARGIN_RAD = 0.06
# Where a traverse is allowed to actually END UP, command plus predicted
# overshoot. The config's position trip stays where it is (1.6 rad as of
# 2026-09-18) -- this is the tighter, self-imposed line the PLAN aims at, so the
# trip keeps its whole margin as insurance against the model being wrong rather
# than spending it on normal running.
PEAK_TARGET_RAD = 1.50
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
# Measured creep, as a RATIO: output rad of slip per output rad of rolling, per
# Nm of output torque. From gbx_1p2p0/fwd_passes/efficiency_to_25 and
# efficiency_to_30, five levels at 20 rpm, each one cycle of +/-1.33 rad:
#
#     T [Nm]      5       10      15      20      25
#     drift    0.0206  0.0435  0.0710  0.1067  0.1560   rad
#     c        7.7e-4  8.2e-4  8.9e-4  1.00e-3 1.17e-3  per Nm
#
# so c rises as the traction limit approaches; the worst measured is taken.
#
# THIS IS A RATIO, NOT A RATE. An earlier version of this file modelled creep in
# rad/s and concluded the fast speeds barely drift, because they spend so little
# time under load. Wrong: creep is slip per unit ROLLING DISTANCE, so it does not
# fall with speed, and a 3000 rpm level running 7 cycles drifts 7x what a 20 rpm
# level running 1 cycle does. That is what EFF_RECENTRE_PER_CYCLE_RPM is for.
# At 40 Nm this is 4.7% creep, right against the customer's 5% limit.
CREEP_RATIO_PER_NM = 1.17e-3
# Multiplier on that ratio when sizing the throw. The fit above is five points on
# one unit on one day, and the failure mode is a window trip 8 minutes in.
CREEP_SAFETY = 1.5
# At and above this input speed, each level is split into ONE SEGMENT PER CYCLE,
# each with its own recentre. Creep per cycle does not fall with speed (it is a
# ratio, see above), but the number of cycles a level needs rises sharply to
# collect TARGET_CONST_S of constant speed -- 3 at 1500 rpm, 7 at 3000. Left
# whole, a 40 Nm level at 3000 rpm drifts ~2.3 rad against 0.17 rad of headroom.
EFF_RECENTRE_PER_CYCLE_RPM = 1500
# Efficiency is emitted one segment PER LEVEL rather than one per speed. A
# window trip is resumable only from the start of the segment it happened in,
# and a 15-minute segment means losing 15 minutes.
EFF_SEGMENT_PER_LEVEL = True
# A `recentre` segment after every level: drives the output back to the declared
# centre on the INPUT shaft (test_manager.Recentre), so creep is undone between
# levels instead of being paid for out of the throw. Where the output is already
# inside `tolerance_rad` it costs only the settle, so the high-speed levels --
# which barely creep -- barely notice it.
EFF_RECENTRE_BETWEEN_LEVELS = True
RECENTRE_PARAMS = {
    'tolerance_rad': 0.02,              # ~1.1 deg of output
    'slow_within_rad': 0.05,
    'velocity_rad_s': 4.0,              # input shaft; = 0.09 rad/s at the output
    'approach_velocity_rad_s': 1.0,
    'acceleration_rad_s2': 20.0,
    'timeout_s': 20.0,
    'settle_s': 0.5,
}
# Which efficiency variant the megabatch carries. Both are always written; the
# batch takes exactly one, because running both would measure efficiency twice.
# 'alt' -- the full four quadrants -- is the one that answers the customer's
# request; 'pos' is the faster half, for when the batch is a shakedown.
MEGABATCH_EFFICIENCY = 'alt'
# Which way the OUTPUT moves for a POSITIVE input position command, in the output
# frame. +1 as of 2026-09-17: touch-off stamped `input_ratio: +43.82` on every
# run in gbx_1p2p0/fwd_passes, and the efficiency logs agree (positive output
# torque, output drifting positive, input commanded positive first).
#
# Used only to aim the FIRST leg of each efficiency shuttle. Creep walks the
# output in the direction of the applied torque, so the peak on that side is the
# one that trips the window -- and it is cheapest to visit it EARLY, while the
# drift is still small. A cycle 0 -> +A -> -A -> 0 under +T reaches +A a quarter
# of the way through (worst excursion A + D/4); the same cycle reversed reaches
# it three quarters of the way through (A + 3D/4). Aiming the first leg down-drift
# is worth half a level's drift, for free.
#
# Get this wrong and the effect simply reverses -- it costs margin, it does not
# point anything at a bumper -- but check it at the bench: the sign is the one
# thing here that is asserted rather than measured at run time.
INPUT_SIGN = +1
# Once slip is measured, the slip hunt's own ceiling drops to this multiple of
# it. There is no reason to be able to command 100 Nm at a contact that lets go
# at 53, and a lower ceiling bounds a runaway.
SLIP_CEILING_MARGIN = 1.4
# Plans the megabatch leaves out, by tag.
#   FSLIP  ends in a deliberate slip that no safety on this rig stops
#          (2026-09-17). It needs an operator watching, so it does not belong in
#          a 50-minute unattended batch. Leave this one out.
#   FBRK   is the stiction half, split out of the slip plan on 2026-09-18. It is
#          safe unattended -- low ceiling, detector fired on 12 of 12 ramps, both
#          velocity safeties backstop it -- so it is excluded only out of
#          caution over its first run with the output free. Drop it from this
#          tuple once that has been watched once.
MEGABATCH_EXCLUDE = ('FSLIP', 'FBRK')
# Why each excluded tag is out, printed next to the plan it left behind.
MEGABATCH_EXCLUDE_REASON = {
    'FSLIP': 'it ends in a deliberate slip that no safety on this rig stops',
    'FBRK': 'its ramps run with the output free, and that has not been watched once yet',
}
# Tag prefix marking a back-drive plan. The megabatch filters on THIS, not on a
# bare 'B' -- which silently swallowed the 'BRK' (breakaway) plan when it was
# first written.
BACKDRIVE_TAG = 'BWD_'
# --- BACK-DRIVE knobs --------------------------------------------------------
# All five output-driven plans are off unless --include-backdrive is passed.
# Nothing below has been MEASURED on this side: the forward numbers are the only
# evidence, and a traction drive is not symmetric, so treat these as first
# guesses that the first run replaces.
#
# Ramps each way for the slip and breakaway hunts, each its own segment with a
# recentre after it. Three is enough to see repeatability without spending the
# contact; the forward slip hunt gets exactly one because a forward slip cannot
# be stopped by anything on the rig, whereas a back-drive slip runs the OUTPUT
# away, which the position window does catch.
BWD_RAMPS_EACH_WAY = 3
# Slip hunt, output-referred. Same 2 Nm/s the forward ramp uses.
BWD_SLIP_RATE_OUT_NM_S = 2.0
# Velocity threshold on the OUTPUT. The forward 2 rad/s divided by the ratio is
# 0.046, which is inside the wind-up creep the output shows while the input
# position loop takes up; 0.20 clears it with room. Sustained for the same
# 500 ms, which is what makes a stick-slip lurch fail to qualify.
BWD_SLIP_V_THRESH_RAD_S = 0.20
# Arm above this fraction of the ceiling, so early wind-up cannot fire it.
BWD_SLIP_ARM_FRACTION = 0.25
# Breakaway hunt, output-referred. Slower than the slip ramp: the number wanted
# is the onset, and with the input free there is no wind-up to climb through.
BWD_BREAKAWAY_RATE_OUT_NM_S = 1.0
# Ceiling for the back-drive breakaway, output Nm. PENDING a measurement. The
# forward breakaway reflected through the ratio is ~10.7 Nm output; a traction
# drive resists back-driving harder, so this is set well above that and well
# below the measured slip torque -- see BWD_BREAKAWAY_SLIP_MARGIN.
BWD_BREAKAWAY_CEILING_OUT_NM = 30.0
# Hard bound on that ceiling as a fraction of measured slip. Past the traction
# limit the ramp slips the contact instead of back-driving it, and the log
# cannot tell the two apart -- both just show the output turning. Staying under
# slip is what keeps a ceiling hit reportable as "does not back-drive below X".
BWD_BREAKAWAY_SLIP_MARGIN = 0.5
# Output velocity threshold for the breakaway detector, and its debounce. Looser
# and shorter than the slip detector: with the input free there is nothing to
# creep against, so any sustained output motion IS the event.
BWD_BREAKAWAY_V_THRESH_RAD_S = 0.05
BWD_BREAKAWAY_DEBOUNCE_CYCLES = 100     # 100 ms at the 1 kHz control rate
# -----------------------------------------------------------------------------

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
        # The ratio-break (slip) safeties, added after the 2026-09-17 free-spin.
        # Read rather than assumed: the notes below tell an operator whether to
        # stand over the rig, and that answer changes with these numbers.
        'ratio_break': _safety_limit(cfg, 'ratio_break'),
        'ratio_slip': _safety_limit(cfg, 'ratio_slip'),
    }


def _safety_limit(cfg, name):
    entry = (cfg.get('safeties') or {}).get(name)
    return None if not entry else float(entry['limit'])


def ratio_break_notes(cfg):
    """What the ratio-break safeties will actually do about a slip, from the
    configured limits rather than from memory.

    This used to be a flat "nothing on this rig stops a slip", which was true
    when it was written and stopped being true when `safeties.ratio_break` and
    `ratio_slip` were added. It can go stale the same way again, so it is
    derived: the numbers below come out of the config every run.
    """
    brk, slip = cfg.get('ratio_break'), cfg.get('ratio_slip')
    if brk is None and slip is None:
        return ['  NO RATIO-BREAK SAFETY IS CONFIGURED. Nothing on this rig can see a '
                'slip. Watch it, and keep a hand on the E-stop']
    out = [f'  ratio-break safeties are armed: '
           + ', '.join(filter(None, [
               f'ratio_break {brk:g} output rad' if brk is not None else None,
               f'ratio_slip {slip:g} output rad/s' if slip is not None else None]))]
    # The free-spin these were written for ran at 2.1-2.7 rad/s at the output
    # and settled 0.356 rad out of ratio. A limit above that cannot catch it.
    if slip is not None and slip > FREE_SPIN_OUT_RAD_S:
        out.append(f'  BUT ratio_slip {slip:g} rad/s is ABOVE the '
                   f'{FREE_SPIN_OUT_RAD_S:g} rad/s the 2026-09-17 free-spin actually ran '
                   'at, so it would not have caught that event. The comment beside it in '
                   'the config reasons about 1.0 rad/s. Treat this test as unprotected '
                   'until that is resolved -- watch it, hand on the E-stop')
    if brk is not None and brk > MEASURED_SLIP_TRAVEL_OUT_RAD * 2:
        out.append(f'  ratio_break {brk:g} rad is well above the '
                   f'{MEASURED_SLIP_TRAVEL_OUT_RAD:g} rad a measured slip settled at on '
                   '2026-09-17, so it will not fire on a slip that stops on its own -- '
                   'which is the intent, but it also means it is not the fast backstop')
    return out


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


def efficiency_amplitude(levels, cfg, allowance, recentre=False, cycles=1):
    """(amplitude, predicted worst drift) for an efficiency shuttle, output rad.

    Speed does not appear: creep is a ratio of rolling distance, so what a
    segment drifts depends on its amplitude, its torque and how many cycles it
    runs -- not on how fast it runs them.

    Creep drifts the shuttle's centre while the level is applied, in the
    direction of the TORQUE -- confirmed on 2026-09-17, where the residual rose
    monotonically through outbound and return traverses alike and never once
    reversed. So the run wanders by the running sum of each level's drift, and
    what has to fit the window is that wander PLUS the amplitude.

    Drift during one level is rate * |T| * (time under it), and the time is
    4A/w_out for a full bipolar cycle -- proportional to the amplitude. So the
    whole thing scales with A and solves in closed form:

        h >= A + A * worst_running_sum_per_unit_amplitude

    With alternating signs the running sum returns to ~0 after every pair and
    the worst case is a single level's drift. Without them it is the whole
    sweep's, which is what made the +only run trip at the 25 Nm level. With
    `recentre` it is a single level's either way, because the drift is undone
    between them.
    """
    h = cfg['half_window'] - allowance
    # Drift over one cycle of amplitude A at torque T is c*T*(4A) -- four
    # quarter-traverses of rolling. Per unit amplitude that is 4*c*T*cycles.
    k = CREEP_SAFETY * CREEP_RATIO_PER_NM * 4.0 * max(1, int(cycles))
    if recentre:
        # A recentre after every segment puts the output back on centre, so
        # drift never accumulates across them: only ONE segment's worth has to
        # fit alongside the throw.
        worst = max((abs(k * float(v)) for v in levels), default=0.0)
    else:
        run = worst = 0.0
        for level in levels:
            run += k * float(level)
            worst = max(worst, abs(run))
    amp = min(OUTPUT_END_RAD, h / (1.0 + worst))
    return amp, worst * amp          # (amplitude, predicted worst drift)


def predicted_overshoot(motor, w_out, torque_out=0.0):
    """How far past its commanded turnaround the OUTPUT will actually go, rad.

    Two terms, both measured (see OVERSHOOT_LAG_S and
    OVERSHOOT_TORQUE_RAD_PER_NM):

      lag * v        the loop's tracking lag at the corner. `motor` is the shaft
                     being POSITION-COMMANDED and is what sets it -- the
                     absorber commanding the output directly is 14x worse than
                     the input commanding it through the reduction.
      k * |T| * gate a position loop's steady deflection under the constant
                     disturbance torque a loaded shuttle carries, fading out
                     below OVERSHOOT_TORQUE_V_REF where the loop has time to
                     converge before the turnaround.

    Neither scales with AMPLITUDE, which is what makes shortening the throw a
    real fix: the overshoot stays the same size and the peak comes in by
    however much the command came in.

    The torque term is one-sided -- it grows the peak on the side the torque
    pushes and shrinks the other -- so this is the WORST side, which is the only
    one that matters for a symmetric window.
    """
    w = abs(w_out)
    gate = min(1.0, w / OVERSHOOT_TORQUE_V_REF) if OVERSHOOT_TORQUE_V_REF else 1.0
    return (OVERSHOOT_LAG_S.get(motor, 0.0) * w
            + OVERSHOOT_TORQUE_RAD_PER_NM * abs(torque_out) * gate
            + OVERSHOOT_MARGIN_RAD)


def overshoot_capped_amplitude(motor, w_out, torque_out=0.0):
    """Largest command whose PREDICTED landing point is still inside
    PEAK_TARGET_RAD, and the overshoot it was sized against."""
    over = predicted_overshoot(motor, w_out, torque_out)
    return PEAK_TARGET_RAD - over, over


def shuttle_accel(w_shaft, throw_shaft, limit, drag_limit=None):
    """Corner acceleration for a shuttle, in the COMMANDED shaft's units.

    `drag_limit` is the ceiling the OTHER shaft imposes, already converted into
    the commanded shaft's units. The two shafts are geared together, so
    commanding one at `a` drags the other at `a` times the ratio between them --
    and only the commanded shaft's own limit used to be checked here.

    That asymmetry does not matter driving from the INPUT (the output is dragged
    at a/43, which no realistic input corner can trouble) and matters a great
    deal driving from the OUTPUT, where the input is dragged at 43a. At 3600 rpm
    input the back-drive shuttle wants 277 rad/s^2 at the output, which is well
    inside the absorber's 2000 -- and is 12160 rad/s^2 at the input, against a
    7500 limit. The input drive cannot follow that; what actually gives is the
    traction contact or the input's tracking, neither of which is a measurement.
    """
    a = max(w_shaft ** 2 / (2 * DECEL_SHARE * throw_shaft), 0.02 * limit)
    a = min(a, ACCEL_USE * limit)
    if drag_limit is not None:
        a = min(a, drag_limit)
    return a


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
                 amp_cap=None, amp_sign=1, torque_out=0.0, **segment_kw):
    """A shuttle at input speed `rpm_in`, commanded on `motor`. None (with a
    note) when the window leaves too little constant-speed travel."""
    r = cfg['ratio']
    lim = cfg['limits'][motor]
    w_in = rpm_in * RPM
    w_out = w_in / r
    shaft = r if motor == 'input' else 1.0          # shaft rad per output rad
    # Rad of the OTHER shaft per rad of the commanded one, and so the ceiling
    # that shaft's own acceleration limit puts on this command. Binds only on
    # output-commanded (back-drive) shuttles -- see shuttle_accel.
    other = 'output' if motor == 'input' else 'input'
    other_per_cmd = (1.0 / r) if motor == 'input' else r
    drag_limit = (ACCEL_USE * cfg['limits'][other]['acceleration'] / other_per_cmd)
    free = shuttle_accel(w_out * shaft, OUTPUT_END_RAD * shaft, lim['acceleration'])
    accel = shuttle_accel(w_out * shaft, OUTPUT_END_RAD * shaft,
                          lim['acceleration'], drag_limit)
    if accel < free - 1e-9:
        # The other shaft is what is limiting. It changes what the corners mean,
        # and at the top speeds it eats the throw -- but it is a property of the
        # SPEED, not of the segment, and efficiency calls this once per level.
        # Keyed on the speed so 16 levels at 3000 rpm say it once.
        note = (f'{rpm_in} rpm corner is limited by the {other} motor, not the '
                f'{motor} one -- {drag_limit:.0f} rad/s^2 at the {motor} drags the '
                f"{other} at its {cfg['limits'][other]['acceleration']:g} rad/s^2 "
                'ceiling')
        if note not in notes:
            notes.append(note)
    peak, const_half = size_traverse(w_out, accel / shaft, allowance, cfg)
    window_peak = peak
    # Commanded-vs-actual: pull the turnaround in until the shaft is predicted to
    # STOP inside PEAK_TARGET_RAD, not merely to be commanded there. This is the
    # cap that matters back-driving; forward its lag is 4 ms and it never bites.
    over_cap, over = overshoot_capped_amplitude(motor, w_out, torque_out)
    if over_cap < peak:
        peak = over_cap
        const_half = peak - w_out ** 2 / (2 * accel / shaft)
        # Once per SPEED AND LEVEL -- unlike the no-load case, two levels at the
        # same speed no longer size the same, so the level belongs in the key.
        load = f' at {abs(torque_out):g} Nm' if torque_out else ''
        note = (f'{rpm_in} rpm{load}: commanded +/-{peak:.3f} rad, not '
                f'{min(window_peak, OUTPUT_END_RAD):.3f} -- predicted overshoot '
                f'{over:.3f} rad puts the actual peak at '
                f'{peak + over:.3f}, against the {PEAK_TARGET_RAD:g} target')
        if note not in notes:
            notes.append(note)
    if amp_cap is not None and amp_cap < peak:
        # Creep budget is tighter than the window (see efficiency_amplitude).
        # The caller has already said so once for the whole speed, so the
        # per-segment note below stays quiet about it -- with one segment per
        # level it would otherwise repeat 64 times.
        peak = amp_cap
        const_half = peak - w_out ** 2 / (2 * accel / shaft)
    if const_half < MIN_CONST_HALF_RAD:
        notes.append(f'{seg_id}: SKIPPED {rpm_in} rpm -- only {const_half:+.2f} rad of '
                     f'constant speed fits at stop_decel {cfg["stop_decel"]:g} '
                     f'(measure the STO coast, update the config, regenerate)')
        return None
    if window_peak < OUTPUT_END_RAD - 1e-9:
        notes.append(f'{seg_id}: {rpm_in} rpm turns at +/-{window_peak:.2f} rad, not '
                     f'{OUTPUT_END_RAD}, to clear the early trip')
    t_const = 2 * const_half / w_out
    cycles = max(1, min(cycles_cap, math.ceil(TARGET_CONST_S / t_const)))
    # A negative amplitude runs the cycle 0 -> -A -> +A -> 0 (the pattern's two
    # `move` calls just swap order), which is how the first leg gets aimed.
    return shuttle_segment(seg_id, motor, w_out * shaft,
                           peak * shaft * (1 if amp_sign >= 0 else -1), accel,
                           cycles, **segment_kw)


def alternating(step, top):
    """+step, -step, +2step, -2step ... Alternating the load sign between
    cycles cancels creep drift (plan §4)."""
    n = int(round(top / step))
    return [v for k in range(1, n + 1) for v in (k * step, -k * step)]


def recentre_segment(seg_id):
    """Put the output back on centre, driving the input. Output at 0 Nm."""
    return {
        'id': seg_id,
        'repeats': 1,
        'lead_in_s': 0.0,
        'primary': {'motor': 'input', 'control_mode': 'velocity'},
        'secondary': {'control_mode': 'torque', 'levels': [0.0],
                      'rate': 1.0, 'settle_s': 0.0},
        'pattern': 'recentre',
        'params': dict(RECENTRE_PARAMS),
    }


def breakaway_segment(seg_id, ramp_motor, ceiling, rate, release_s, rest_s,
                      bipolar, cycles, v_thresh, debounce, arm_fraction=0.05,
                      stuck_s=0.25, direction=1, hold_mode='position',
                      hold_level=0.0, hold_rate=0.1):
    return {
        'id': seg_id,
        'repeats': 1,
        'lead_in_s': test_builder.START_HOLD_S,
        'primary': {'motor': ramp_motor, 'control_mode': 'torque'},
        'secondary': {'control_mode': hold_mode, 'levels': [float(hold_level)],
                      'rate': hold_rate, 'settle_s': 1.0},
        'pattern': 'breakaway',
        'params': {'amplitude': round(ceiling, 3), 'rate': rate,
                   'release_s': release_s, 'rest_s': rest_s,
                   'direction': int(direction), 'bipolar': bipolar,
                   'cycles': cycles, 'velocity_threshold': v_thresh,
                   'stuck_s': stuck_s, 'debounce_cycles': debounce,
                   'arm_fraction': arm_fraction},
    }


def ramp_and_recentre(seg_id, ramp_motor, cycles_each_way, **kw):
    """`cycles_each_way` ramps in each direction, each its OWN segment, with a
    recentre between every one of them.

    Why not one `bipolar` breakaway segment with `cycles` ramps: a bipolar
    segment runs +, -, +, - back to back with only a release and a rest between
    them, and nothing puts the shaft back where it started. That is fine for a
    stiction ramp that barely moves, and wrong for anything that BREAKS FREE and
    travels -- a back-drive slip or breakaway leaves the output metres from
    centre in output-referred terms, and the next ramp then starts from there
    and runs at the window. Splitting the segments is the only place a recentre
    can go, because a recentre is itself a segment.

    Ramps alternate direction (+ first) so the contact is not worked the same
    way six times running, and the last recentre is emitted too, so whatever
    follows starts from centre.
    """
    segs = []
    for c in range(1, int(cycles_each_way) + 1):
        for sign, tag in ((+1, 'P'), (-1, 'N')):
            sid = f'{seg_id}_{tag}{c}'
            segs.append(breakaway_segment(sid, ramp_motor, bipolar=False,
                                          cycles=1, direction=sign, **kw))
            segs.append(recentre_segment(sid + '_RC'))
    return segs


# --- the plans -------------------------------------------------------------------

def build_efficiency(cfg, shakedown, alternate, rpm_cap, cycles_cap, torque_scale,
                    bwd=False):
    """One efficiency plan, `alternate` choosing the level signs.

    `bwd` swaps which shaft drives. FORWARD: the input commands position and the
    OUTPUT holds each torque level, so levels are already output Nm. BACK-DRIVE:
    the OUTPUT commands position and the INPUT holds torque, so each level is
    divided by the ratio -- the customer asks for output torque either way, and
    on this side the input has to produce it through the gearbox.

    The speeds are input-referred in both cases (plan_shuttle takes rpm_in), so
    'E3000' is 3000 rpm at the INPUT whichever shaft is commanding.

    One thing does NOT carry across: the creep throw budget. Forward, creep
    walks the OUTPUT while the input holds position, and the window watches the
    output -- so the throw has to be cut to leave room for it. Back-driving, the
    output is the shaft being position-commanded, so it goes exactly where it is
    told and cannot creep away from the window; the slip shows up on the INPUT
    instead, where nothing trips. So the back-drive sweep keeps the full throw,
    and its recentres are cheap insurance rather than the thing holding the run
    inside the window.

    Each level is its own segment, and (with EFF_RECENTRE_BETWEEN_LEVELS) a
    `recentre` segment follows each one. Both exist for the same reason: creep.
    Per-level segments bound what a window trip costs -- a trip is resumable only
    from the start of its segment, and the old one-segment-per-speed form made
    that 15 minutes at 20 rpm. The recentres stop the drift accumulating at all,
    which is what lets the +only variant keep its throw instead of paying for the
    whole sweep's drift out of it.
    """
    notes, segs = [], []
    r = cfg['ratio']
    motor = 'output' if bwd else 'input'
    # Output Nm -> the holding motor's own Nm.
    level_scale = (1.0 / r) if bwd else 1.0
    allowance = TRIP_ALLOWANCE_RAD['backdrive' if bwd else 'loaded_fwd']
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
    if (top_out < want_top - 1e-9
            and not (T_SLIP_OUT_NM and top_out <= EFF_SLIP_MARGIN * T_SLIP_OUT_NM)):
        notes.append(f'CAPPED: {UNIT} was requested to {want_top:g} Nm output but the '
                     f"customer's {OUTPUT_TORQUE_CAP_NM:g} Nm limit stops the sweep at "
                     f'{top_out:g} Nm -- the top of the requested efficiency range is '
                     'not covered')
    top_out *= torque_scale

    if alternate:
        levels = [round(v, 4) for v in alternating(EFF_STEP_OUT_NM, top_out)]
        notes.append('ALTERNATING levels: all four quadrants, and creep cancels pair by '
                     'pair even without the recentres (plan section 4)')
    else:
        n = int(round(top_out / EFF_STEP_OUT_NM))
        levels = [round(k * EFF_STEP_OUT_NM, 4) for k in range(1, n + 1)]
        notes.append('+ONLY levels: half the time, and half the quadrants -- one rotation '
                     'sense only. Creep does not cancel here (it takes the sign of the '
                     'TORQUE, not of the traverse: measured 2026-09-17, the residual rose '
                     'through outbound and return alike and never reversed), so it is the '
                     'recentres that keep this inside the window')

    rpms = [x for x in EFF_RPM if x <= rpm_cap]
    trimmed = {}
    if bwd and BWD_EFF_TOP_LEVELS:
        keep = {x: BWD_EFF_TOP_LEVELS.get(x) for x in rpms
                if x in BWD_EFF_TOP_LEVELS}
        rpms = [x for x in rpms if keep.get(x, 1) != 0]
        trimmed = {k: v for k, v in keep.items() if v}
    for rpm in rpms:
        # Above EFF_RECENTRE_PER_CYCLE_RPM each level is split into one segment
        # per cycle, so every segment carries exactly one cycle of drift and the
        # throw solve below is the same at every speed.
        per_cycle = rpm >= EFF_RECENTRE_PER_CYCLE_RPM and not bwd
        if bwd:
            # The commanded shaft IS the one the window watches, so the throw
            # is not spent on a creep budget. See the docstring.
            amp, drift = None, 0.0
            w_out = rpm * RPM / r
            lo_cap, lo_over = overshoot_capped_amplitude('output', w_out, 0.0)
            top = max(abs(float(v)) for v in levels) if levels else 0.0
            hi_cap, hi_over = overshoot_capped_amplitude('output', w_out, top)
            notes.append(f'E{rpm:04d}: throw +/-{min(OUTPUT_END_RAD, hi_cap):.3f} rad at '
                         f'{top:g} Nm to +/-{min(OUTPUT_END_RAD, lo_cap):.3f} at 0 Nm '
                         f'(overshoot {hi_over:.3f} / {lo_over:.3f}) -- back-driving, '
                         'creep lands on the input and cannot walk the window; what '
                         'sizes the throw here is OVERSHOOT, and it grows with the '
                         'level as well as the speed')
        else:
            amp, drift = efficiency_amplitude(levels, cfg, allowance,
                                              recentre=EFF_RECENTRE_BETWEEN_LEVELS,
                                              cycles=1)
            notes.append(f'E{rpm:04d}: throw +/-{amp:.2f} rad output'
                         + ('' if amp >= OUTPUT_END_RAD - 1e-9 else
                            f' (cut from {OUTPUT_END_RAD:g} to leave room for '
                            f'{drift:.2f} rad of creep in one cycle)'))
        rpm_levels = levels
        if rpm in trimmed:
            # Keep the top N magnitudes, both signs where the variant has them.
            n = trimmed[rpm]
            mags = sorted({round(abs(v), 6) for v in levels})[-n:]
            rpm_levels = [v for v in levels if round(abs(v), 6) in mags]
            notes.append(f'  E{rpm:04d}: TRIMMED to the top {n} magnitude(s) '
                         + ', '.join(f'{m:g}' for m in mags)
                         + f' Nm output ({len(rpm_levels)} of {len(levels)} levels)')
            notes.append(f'    WHY, and what the run still has to prove: the input cell '
                         f'is where this sweep reads torque, so these levels put '
                         + ', '.join(f'{m / r:.3f}' for m in mags)
                         + f' Nm on a {cfg["input_torque_safety"]:g} Nm-limited, 20 Nm '
                         'FS cell. They are the BEST case at this speed')
            notes.append('    the cell noise floor is roughly CONSTANT in Nm (session '
                         'tare scatter was sd 0.0092 Nm, 2026-09-18) while the signal '
                         'scales with the level, so measuring the noise at these two '
                         'levels gives the signal-to-noise of every level below them by '
                         'arithmetic -- which is the evidence for dropping them, without '
                         'spending the hours to run them')
            notes.append('    so REPORT the scatter these two produce, not just their '
                         'efficiency. If it is already marginal here, the case is made; '
                         'if it is clean, put the rest of the ladder back with '
                         '--bwd-eff-top-levels')
        for level in rpm_levels:
            sid = f'E{rpm:04d}_{"P" if level >= 0 else "N"}{abs(level):02.0f}'
            # First leg aimed the way this level's creep will drift, so the peak
            # that eats the window is reached while the drift is smallest.
            # Aiming the first leg down-drift only buys anything where creep
            # eats the window, which is the forward case only.
            amp_sign = 1 if bwd else (1 if level >= 0 else -1) * INPUT_SIGN
            seg = plan_shuttle(sid, motor, rpm, allowance, cfg, notes,
                               cycles_cap, amp_cap=amp, amp_sign=amp_sign,
                               torque_out=level,
                               levels=[level * level_scale],
                               level_rate=round(50.0 * level_scale, 4),
                               settle_s=1.0)
            if not seg:
                continue
            n_cycles = max(1, int(seg['params']['cycles']))
            if per_cycle and n_cycles > 1:
                for c in range(1, n_cycles + 1):
                    sub = dict(seg, id=f'{sid}_C{c}',
                               params=dict(seg['params'], cycles=1))
                    segs.append(sub)
                    segs.append(recentre_segment(sub['id'] + '_RC'))
            else:
                if n_cycles > 1 and not bwd:
                    notes.append(f'  {sid}: {n_cycles} cycles in one segment, so its '
                                 f'drift budget is {n_cycles}x the figure above -- raise '
                                 'EFF_RECENTRE_PER_CYCLE_RPM coverage if it trips')
                segs.append(seg)
                if EFF_RECENTRE_BETWEEN_LEVELS:
                    segs.append(recentre_segment(sid + '_RC'))

    kind = 'alt' if alternate else 'pos'
    notes.append(f'{UNIT}: {len(levels)} levels to '
                 f'{"+/-" if alternate else "+"}{top_out:g} Nm across '
                 f'{len(rpms)} speed(s), '
                 f'{len(segs)} segment(s)'
                 + (f' -- {", ".join(f"{k} rpm trimmed to top {v}" for k, v in sorted(trimmed.items()))}'
                    if trimmed else ''))
    notes.append('one segment PER LEVEL: a window trip costs one level, not one speed, '
                 'and recentre-and-resume picks up at that level')
    if bwd:
        notes.append(f'BACK-DRIVE: the OUTPUT commands position at the input-referred '
                     f'speed, the INPUT holds torque. Levels are output Nm divided by '
                     f'the {r:g} ratio, so the {EFF_STEP_OUT_NM:g} Nm output step is '
                     f'{EFF_STEP_OUT_NM / r:.4f} Nm at the input cell')
        notes.append('  the input cell reads a few hundred mNm here against a 20 Nm FS '
                     'cell -- check the noise floor on the first level before trusting '
                     'an efficiency number off it')
        notes.append('  power flows output -> input on the resisting half of each '
                     'traverse; bin on those, exactly as forward does')
    else:
        notes.append(f'each shuttle starts toward its own creep direction (INPUT_SIGN '
                     f'{INPUT_SIGN:+d}), so the peak that eats the window is reached at a '
                     'quarter of the level instead of three quarters -- worth half a '
                     "level's drift. CHECK THE SIGN at the bench")
    fast = [x for x in rpms if x >= EFF_RECENTRE_PER_CYCLE_RPM]
    if fast and not bwd:
        notes.append(f'{EFF_RECENTRE_PER_CYCLE_RPM} rpm and above ({", ".join(str(x) for x in fast)}): '
                     'one segment PER CYCLE, each with its own recentre. Creep per cycle '
                     'does not fall with speed -- it is a ratio of rolling distance -- and '
                     'those levels need 3 to 7 cycles each, so left whole they drift '
                     'metres past the window')
    if EFF_RECENTRE_BETWEEN_LEVELS:
        notes.append(f'a recentre after every level: input drives the output back to '
                     f'centre within {RECENTRE_PARAMS["tolerance_rad"]:g} rad, '
                     f'{RECENTRE_PARAMS["velocity_rad_s"]:g} rad/s input, '
                     f'{RECENTRE_PARAMS["timeout_s"]:g} s timeout. Where the output is '
                     'already centred it costs only the settle')
    side = 'bwd' if bwd else 'fwd'
    tag = (BACKDRIVE_TAG if bwd else '') + f'EFF_{kind.upper()}'
    return (tag,
            {'name': f'archimedes_{side}_efficiency_{kind}', 'segments': segs}, notes)


def build_plans(cfg, shakedown, include_backdrive=False):
    """Returns [(tag, recipe, notes)] in run order. Forward (input-driving)
    plans first, then the back-drive plans if they were asked for."""
    r = cfg['ratio']
    torque_scale = 0.25 if shakedown else 1.0
    cycles_cap = 1 if shakedown else MAX_CYCLES
    rpm_cap = 300 if shakedown else math.inf
    plans = []          # (tag, recipe, notes)
    # Plans that must run LAST, in this order. Slip (and now breakaway) used to
    # be moved to the end with an index shuffle; with two of them a list is
    # clearer and survives another one being added.
    late = []

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
    # -- 2a. FORWARD BREAKAWAY (stiction), its own plan ----------------------
    # Split out of the slip plan on 2026-09-18. These were never one test: they
    # hold the output in opposite modes (torque-0 vs position), they measure
    # different things (drivetrain friction vs the traction contact), they want
    # different detectors, and only one of them ends in an event that needs an
    # operator. Bundled together the safe one could not be run without arming
    # the dangerous one.
    brk_notes = []
    stict = breakaway_segment('STICT', 'input', STICTION_CEILING_IN_NM,
                              round(rate_in, 4), release_s=0.05, rest_s=3.0,
                              bipolar=True, cycles=cycles, v_thresh=0.5,
                              debounce=3, arm_fraction=0.05,
                              hold_mode='torque', hold_level=0.0, hold_rate=1.0)
    brk_notes.append(f'ramp rate {rate_in * r:.0f} Nm/s output-referred '
                     f'({rate_in:.4f} Nm/s input)')
    brk_notes.append(f'input breakaway stiction, output in TORQUE mode at 0 Nm '
                     f'(not position -- see the comment). Ramps only to '
                     f'{STICTION_CEILING_IN_NM:g} Nm input, '
                     f'{cycles} ramp(s) each way')
    brk_notes.append(f'  the 2026-09-17 position-held runs gave {T_BREAKAWAY_IN_NM:g} Nm mean; '
                     'expect this to read lower and cleaner, since it no longer includes '
                     'the torque spent winding the absorber position loop')
    brk_notes.append('  CAUTION: with the output free, a ramp that runs past breakaway '
                     'ACCELERATES the input instead of winding up. The detector fired on '
                     '12 of 12 ramps on 2026-09-17, the ceiling is low, and the 500 rad/s '
                     'input / 12 rad/s output velocity safeties backstop it -- but watch '
                     'the first ramp')
    brk_notes.append('  split from archimedes_fwd_slip: this one is safe to run '
                     'unattended, the slip hunt is not')
    late.append(('FBRK', {'name': 'archimedes_fwd_breakaway', 'segments': [stict]},
                 brk_notes))

    # -- 2b. FORWARD STATIC SLIP, its own plan -------------------------------
    # ONE ramp, ONE direction. RampBreak releases on a breakaway and goes
    # straight into the next ramp; on 2026-09-17 that second ramp drove the
    # already-slipping contact to a free-spin at 155 rad/s of input, and nothing
    # on the rig stopped it -- the operator E-stopped it.
    #
    # A ratio-break safety DOES now exist (`safeties.ratio_break` /
    # `ratio_slip`, both resumable), so that specific hole is closed in
    # principle. This is still one ramp, because whether it is closed in
    # PRACTICE depends on the limits those two entries are set to, and they are
    # currently far wider than the comments next to them argue for -- see the
    # note this plan prints. Give it more ramps (the back-drive plans take
    # --bwd-ramps) once a slip has been watched to trip them.
    segs = [
        breakaway_segment('SLIP', 'input', ceiling_in, round(rate_in, 4),
                          release_s=0.05, rest_s=3.0, bipolar=False,
                          cycles=1, v_thresh=SLIP_V_THRESH_RAD_S,
                          debounce=SLIP_DEBOUNCE_CYCLES,
                          arm_fraction=round(arm_fraction, 4)),
    ]
    notes.append(f'ramp rate {rate_in * r:.0f} Nm/s output-referred '
                 f'({rate_in:.4f} Nm/s input)')
    notes.append(f'traction contact, ONE ramp one direction, armed above '
                 f'{arm_fraction * ceiling_in:.3f} Nm input '
                 f'(= {arm_fraction * ceiling_in * r:.0f} Nm output-referred), '
                 f'{SLIP_V_THRESH_RAD_S:g} rad/s sustained {SLIP_DEBOUNCE_CYCLES} cycles '
                 f'({SLIP_DEBOUNCE_CYCLES:g} ms)')
    if T_SLIP_OUT_NM:
        notes.append(f'  slip IS measured: {T_SLIP_OUT_NM:g} Nm output (2026-09-17), about '
                     f'half the {EFF_MAX_OUT_NM[UNIT]:g} Nm rating. Re-run this only to '
                     'confirm or to track degradation -- every run costs contact')
    else:
        notes.append(f'  slip is NOT measured on this unit: this run is the hunt for it. '
                     f'The ceiling is the customer cap reflected to the input, which may '
                     f'be below the slip torque -- a ramp that reaches it reads "no slip '
                     f'up to {OUTPUT_TORQUE_CAP_NM:g} Nm output", which is a result')
    notes.append('  the window, both velocity safeties and both torque safeties all stay '
                 'happy through a forward slip: the output stays put and the input spins. '
                 'The ratio-break safeties are the only ones that can see it')
    notes += ratio_break_notes(cfg)
    if T_SLIP_OUT_NM:
        notes.append(f'ceiling is {SLIP_CEILING_MARGIN:g} x the measured slip '
                     f'({ceiling_out:.0f} Nm output), not the '
                     f'{OUTPUT_TORQUE_CAP_NM:g} Nm customer cap: no reason to be able to '
                     'command 100 Nm at a contact that lets go at 53')
    elif OUTPUT_TORQUE_CAP_NM < 0.95 * cfg['output_torque_safety']:
        notes.append(f'CAPPED at the customer\'s {OUTPUT_TORQUE_CAP_NM:g} Nm output limit '
                     f'(the safety alone would have allowed '
                     f'{0.95 * cfg["output_torque_safety"]:.0f} Nm)')
        notes.append('A unit rated 100-140 Nm should NOT slip under this ceiling: the ramp '
                     'hitting the ceiling is the result ("no slip up to the cap"), not a '
                     'failed run. Stiffness then stays on the provisional figure')
    late.append(('FSLIP', {'name': 'archimedes_fwd_slip', 'segments': segs}, notes))

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
            if VEL_RECENTRE_BETWEEN_SPEEDS:
                segs.append(recentre_segment(seg['id'] + '_RC'))
    notes.append('torque ripple is reported from the <=600 rpm points only (plan §6)')
    if VEL_RECENTRE_BETWEEN_SPEEDS:
        notes.append(f'a recentre after every speed ({len(rpms)} of them): the output goes '
                     f'back to centre within {RECENTRE_PARAMS["tolerance_rad"]:g} rad '
                     'before the next one starts, so tracking error does not accumulate '
                     'down the sweep and a trip resumes that speed from centre')
    plans.append(('VEL', {'name': 'archimedes_fwd_velocity_ramp', 'segments': segs},
                  notes))

    # 5. Efficiency, forward only. TWO plans, written every run:
    #   ..._efficiency_pos   +5, +10 ... only
    #   ..._efficiency_alt   +5, -5, +10, -10 ...
    # They are not interchangeable. A bipolar shuttle at one torque sign covers
    # two of the four (rotation sense x power direction) quadrants; the other two
    # need the opposite sign. So `pos` is half the time and half the coverage,
    # and on a unit measurably direction-asymmetric (the breakaway ramps read
    # +0.252 / -0.233 Nm, ~7 %) the missing half is not redundant.
    for alternate in (False, True):
        plans.append(build_efficiency(cfg, shakedown, alternate, rpm_cap,
                                      cycles_cap, torque_scale))

    # Breakaway then slip run LAST. Plan §7 put slip first because stiffness
    # needed its number; that number is now a measured constant, and a slip hunt
    # degrades the contact, so anything measured after one is measured on a
    # different unit. Breakaway goes just ahead of it for the same reason --
    # its friction number is only meaningful on an unslipped contact.
    plans += late

    if include_backdrive:
        plans += build_backdrive_plans(cfg, shakedown)

    if shakedown:
        for _, recipe, _ in plans:
            if recipe:
                recipe['name'] += '_shakedown'
    return plans


def build_backdrive_plans(cfg, shakedown):
    """The four output-driven plans, off by default (`--include-backdrive`).

    Rewritten 2026-09-18 against the customer list plus the bench request. Every
    one of them commands the OUTPUT and lets the input do whatever the gearbox
    makes it do -- which is the definition of back-driving this rig can express,
    since the input has no way to resist except in torque mode.

      archimedes_bwd_velocity_ramp   no-load speed sweep, input-referred speeds
      archimedes_bwd_efficiency_alt  5 Nm output steps at 20/300/1500/3000 rpm
      archimedes_bwd_efficiency_pos  the same, one rotation sense (faster half)
      archimedes_bwd_slip            input POSITION-held, output torque ramps
      archimedes_bwd_breakaway       input at 0 Nm TORQUE, output torque ramps

    The last two are the pair the forward side splits into `_slip` and
    `_breakaway`, and they differ in exactly the same way: what the far shaft is
    doing. Holding the input in POSITION grounds the train, so the ramp winds the
    contact up and the event is the traction contact letting go -- a slip.
    Leaving the input at 0 Nm TORQUE grounds nothing, so the ramp fights only
    friction and the event is the whole train breaking free and turning -- a
    breakaway. Same fixture, same ramp, different hold, different number.
    """
    r = cfg['ratio']
    torque_scale = 0.25 if shakedown else 1.0
    cycles_cap = 1 if shakedown else MAX_CYCLES
    rpm_cap = 300 if shakedown else math.inf
    ramps = 1 if shakedown else BWD_RAMPS_EACH_WAY
    plans = []

    # 1. Back-drive velocity ramp. The output walks the window at whatever speed
    # puts the requested rpm on the INPUT shaft -- plan_shuttle's rpm argument is
    # input-referred whichever motor commands, so these are the customer's own
    # numbers (30 rpm steps to 300, then 300 rpm steps to 3600) read at the input.
    notes, segs = [], []
    rpms = [30, 150, 300] if shakedown else [x for x in VEL_RAMP_RPM if x <= rpm_cap]
    for rpm in rpms:
        seg = plan_shuttle(f'V{rpm:04d}', 'output', rpm, TRIP_ALLOWANCE_RAD['backdrive'],
                           cfg, notes, cycles_cap)
        if seg:
            segs.append(seg)
            if VEL_RECENTRE_BETWEEN_SPEEDS:
                segs.append(recentre_segment(seg['id'] + '_RC'))
    top = max(rpms) if rpms else 0
    notes.append(f'output position-commanded, input at 0 Nm. Speeds are INPUT-referred: '
                 f'{min(rpms) if rpms else 0}-{top} rpm input = '
                 f'{min(rpms) * RPM / r if rpms else 0:.3f}-{top * RPM / r:.2f} rad/s at '
                 'the output')
    notes.append(f'  top speed puts {top * RPM:.0f} rad/s on the input (limit '
                 f'{cfg["limits"]["input"]["velocity"]:g}) and {top * RPM / r:.1f} rad/s on '
                 f'the output (limit {cfg["limits"]["output"]["velocity"]:g})')
    notes.append('  the input is dragged at 43x the output corner acceleration, which is '
                 'what caps the high-speed corners here -- see any drag note above')
    notes.append('also the back-drive torque-ripple source; same <=600 rpm limit (plan '
                 'section 6)')
    if VEL_RECENTRE_BETWEEN_SPEEDS:
        notes.append(f'a recentre after every speed ({len(rpms)} of them). The recentre '
                     'drives the INPUT in velocity mode, so it also leaves the input out '
                     'of the torque mode this plan runs it in -- which is what makes the '
                     'next segment re-zero its own position frame cleanly')
    plans.append((BACKDRIVE_TAG + 'VEL',
                  {'name': 'archimedes_bwd_velocity_ramp', 'segments': segs}, notes))

    # 2. Back-drive efficiency, both level patterns, through the same builder the
    # forward sweep uses -- so it inherits the slip-limited ceiling, one segment
    # per level, and the recentres. The old version had none of those: it wrote
    # one 77-minute segment per speed and swept to the full 100 Nm on a contact
    # measured to let go at 53.
    for alternate in (False, True):
        plans.append(build_efficiency(cfg, shakedown, alternate, rpm_cap,
                                      cycles_cap, torque_scale, bwd=True))

    # 3. Back-drive static slip: input held in POSITION, output torque ramps
    # until the contact lets go. `ramps` attempts each way, each its own segment
    # with a recentre after it -- a slip moves the output and leaves it there,
    # so without the recentres attempt 2 starts wherever attempt 1 finished.
    notes = []
    ceiling_out = output_ceiling(cfg)
    if T_SLIP_OUT_NM:
        ceiling_out = min(ceiling_out, SLIP_CEILING_MARGIN * T_SLIP_OUT_NM)
    ceiling_out *= torque_scale
    # Arm above the wind-up the output shows before anything slips. Referred to
    # the output, the forward numbers are all divided by the ratio, so the floor
    # that matters here is a fraction of the ceiling rather than a measured Nm.
    arm = min(0.9, BWD_SLIP_ARM_FRACTION)
    segs = ramp_and_recentre('SLIP', 'output', ramps,
                             ceiling=ceiling_out, rate=BWD_SLIP_RATE_OUT_NM_S,
                             release_s=0.25, rest_s=3.0,
                             v_thresh=BWD_SLIP_V_THRESH_RAD_S,
                             debounce=SLIP_DEBOUNCE_CYCLES,
                             arm_fraction=round(arm, 4),
                             hold_mode='position', hold_level=0.0, hold_rate=0.1)
    notes.append(f'input POSITION-held at 0, output ramps to {ceiling_out:.0f} Nm at '
                 f'{BWD_SLIP_RATE_OUT_NM_S:g} Nm/s, {ramps} ramp(s) each way, recentre '
                 'after every ramp')
    notes.append(f'  detector: {BWD_SLIP_V_THRESH_RAD_S:g} rad/s on the OUTPUT sustained '
                 f'{SLIP_DEBOUNCE_CYCLES} cycles ({SLIP_DEBOUNCE_CYCLES:g} ms), armed '
                 f'above {arm * ceiling_out:.0f} Nm')
    notes.append(f'  the output threshold is the forward one divided by the ratio '
                 f'({SLIP_V_THRESH_RAD_S:g} rad/s input -> '
                 f'{SLIP_V_THRESH_RAD_S / r:.3f}); {BWD_SLIP_V_THRESH_RAD_S:g} is used '
                 'instead so wind-up creep at the output cannot fire it. UNMEASURED on '
                 'this side -- check it on the first ramp')
    if T_SLIP_OUT_NM:
        notes.append(f'  ceiling is {SLIP_CEILING_MARGIN:g} x the FORWARD measured slip '
                     f'({T_SLIP_OUT_NM:g} Nm). Back-drive slip torque is a different '
                     'number and may be lower; if every ramp fires well under the '
                     'ceiling, bring it down and re-run')
    notes.append('  back-driving it is the OUTPUT that runs away, so the position window '
                 'catches this one on its own -- unlike the forward slip, where the '
                 'output stays put and only the ratio-break channels can see it')
    notes += ratio_break_notes(cfg)
    plans.append((BACKDRIVE_TAG + 'SLIP',
                  {'name': 'archimedes_bwd_slip', 'segments': segs}, notes))

    # 4. Back-drive breakaway: input at 0 Nm TORQUE -- nothing grounds the train
    # -- and the output ramps until the whole thing turns. This is the number
    # that says whether the unit back-drives at all, and it is the one test here
    # that has never been run, so every figure below is a starting point.
    notes = []
    brk_out = BWD_BREAKAWAY_CEILING_OUT_NM * torque_scale
    slip_bound = None
    if T_SLIP_OUT_NM:
        slip_bound = BWD_BREAKAWAY_SLIP_MARGIN * T_SLIP_OUT_NM * torque_scale
        if slip_bound < brk_out:
            brk_out = slip_bound
    brk_out = min(brk_out, output_ceiling(cfg))
    segs = ramp_and_recentre('BRK', 'output', ramps,
                             ceiling=brk_out, rate=BWD_BREAKAWAY_RATE_OUT_NM_S,
                             release_s=0.25, rest_s=3.0,
                             v_thresh=BWD_BREAKAWAY_V_THRESH_RAD_S,
                             debounce=BWD_BREAKAWAY_DEBOUNCE_CYCLES,
                             arm_fraction=0.05,
                             hold_mode='torque', hold_level=0.0, hold_rate=1.0)
    notes.append(f'input at 0 Nm TORQUE (free), output ramps to {brk_out:.1f} Nm at '
                 f'{BWD_BREAKAWAY_RATE_OUT_NM_S:g} Nm/s, {ramps} ramp(s) each way, '
                 'recentre after every ramp')
    notes.append(f'  detector: {BWD_BREAKAWAY_V_THRESH_RAD_S:g} rad/s on the OUTPUT '
                 f'sustained {BWD_BREAKAWAY_DEBOUNCE_CYCLES} cycles, armed above 5% of '
                 'the ceiling')
    notes.append('  THE CEILING MUST STAY BELOW THE SLIP TORQUE. With the input free the '
                 'ramp meets whichever gives way first; past the traction limit it stops '
                 'being a back-drive test and quietly becomes a slip test, and the log '
                 'looks the same either way (the output turns). Keeping the ceiling under '
                 'slip means a ramp that hits it reads "does not back-drive below X Nm", '
                 'which is a real result')
    if slip_bound is not None and slip_bound <= BWD_BREAKAWAY_CEILING_OUT_NM:
        notes.append(f'  SLIP-BOUNDED: ceiling cut from '
                     f'{BWD_BREAKAWAY_CEILING_OUT_NM:g} to {brk_out:.1f} Nm '
                     f'({BWD_BREAKAWAY_SLIP_MARGIN:g} x the {T_SLIP_OUT_NM:g} Nm measured '
                     'forward slip)')
    notes.append(f'  forward breakaway was {T_BREAKAWAY_IN_NM:g} Nm at the input, which is '
                 f'{T_BREAKAWAY_IN_NM * r:.0f} Nm output-referred if the drive were '
                 'symmetric. It is not -- a traction drive resists back-driving harder -- '
                 'so expect more than that, and possibly more than the ceiling')
    notes.append('  CAUTION: past breakaway the output ACCELERATES rather than winding up, '
                 'and it carries the flange. The output velocity safety '
                 f'({cfg["limits"]["output"]["velocity"]:g} rad/s) and the position window '
                 'are the backstops. Watch the first ramp')
    if GRAVITY_AMPLITUDE_NM is None:
        notes.append('  flange gravity is UNMEASURED and biases this directly: it adds to '
                     'the ramp one way and subtracts the other, so the +/- pair will not '
                     'agree until archimedes_fwd_gravity_map has been run and subtracted')
    plans.append((BACKDRIVE_TAG + 'BRK',
                  {'name': 'archimedes_bwd_breakaway', 'segments': segs}, notes))
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


def report_stale(tests_dir, written, shakedown, prefixes=('archimedes_',)):
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
                   if f.startswith(tuple(prefixes)) and f.endswith('.yaml')
                   and not f.startswith('archimedes_bench_')
                   and f.endswith('_shakedown.yaml') == shakedown
                   and f not in keep)
    if stale:
        print(f'\nSTALE in {test_builder.GENERATED_TEST_DIR}/ (not written this run, '
              'still selectable in the GUI):')
        for f in stale:
            print(f'  {f}')


def _rpm_list(text):
    """'20,300,1500' -> [20, 300, 1500]. Also accepts 'a:b:step' ranges, so the
    customer's velocity-ramp spec stays writable as 30:300:30,600:3600:300."""
    out = []
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part:
            lo, hi, step = (int(float(x)) for x in part.split(':'))
            out += list(range(lo, hi + 1, step))
        else:
            out.append(int(float(part)))
    return sorted(set(out))


class _Unset:
    """Default for every override option.

    `None` cannot be the default: it is a MEANINGFUL value for the measured
    constants (`--slip-torque none` is how a unit goes back to the provisional
    figure), so a None default made that flag silently do nothing.
    """
    def __repr__(self):
        return '<unset>'


UNSET = _Unset()


def _top_levels(text):
    """'20=2,300=3' -> {20: 2, 300: 3}. Empty string means run every speed whole."""
    out = {}
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        rpm, _, n = part.partition('=')
        if not n:
            raise ValueError('--bwd-eff-top-levels wants <rpm>=<n>, got %r' % (part,))
        out[int(float(rpm))] = int(float(n))
    return out


def _lag_map(text):
    """'output=0.05,input=0.004' -> {'output': 0.05, 'input': 0.004}.

    Merged onto the defaults, so naming one shaft leaves the other alone.
    """
    out = dict(OVERSHOOT_LAG_S)
    for part in str(text).split(','):
        part = part.strip()
        if not part:
            continue
        motor, _, value = part.partition('=')
        motor = motor.strip()
        if motor not in ('input', 'output') or not value:
            raise ValueError(f'--overshoot-lag wants output=<s>[,input=<s>], got {part!r}')
        out[motor] = float(value)
    return out


def _opt_float(text):
    """A float, or None for 'none'/'' -- how a measured constant is un-measured
    again (e.g. --slip-torque none goes back to the provisional figure)."""
    text = str(text).strip().lower()
    return None if text in ('none', 'null', '') else float(text)


# CLI name -> module constant. Every one of these is a knob that changes between
# units or between days on the bench; everything else stays a constant at the top
# of the file, where it can carry the paragraph explaining it. Overriding here
# rebinds the module global before any plan is built, so the notes each plan
# prints describe the values actually used.
OVERRIDES = {
    'unit': 'UNIT',
    'slip_torque': 'T_SLIP_OUT_NM',
    'creep_onset': 'T_CREEP_ONSET_OUT_NM',
    'output_cap': 'OUTPUT_TORQUE_CAP_NM',
    'eff_rpm': 'EFF_RPM',
    'bwd_eff_top_levels': 'BWD_EFF_TOP_LEVELS',
    'eff_step': 'EFF_STEP_OUT_NM',
    'eff_slip_margin': 'EFF_SLIP_MARGIN',
    'vel_rpm': 'VEL_RAMP_RPM',
    'throw': 'OUTPUT_END_RAD',
    'peak_target': 'PEAK_TARGET_RAD',
    'overshoot_margin': 'OVERSHOOT_MARGIN_RAD',
    'overshoot_lag': 'OVERSHOOT_LAG_S',
    'overshoot_torque': 'OVERSHOOT_TORQUE_RAD_PER_NM',
    'input_sign': 'INPUT_SIGN',
    'bwd_ramps': 'BWD_RAMPS_EACH_WAY',
    'bwd_slip_ceiling_margin': 'SLIP_CEILING_MARGIN',
    'bwd_slip_v_thresh': 'BWD_SLIP_V_THRESH_RAD_S',
    'bwd_breakaway_ceiling': 'BWD_BREAKAWAY_CEILING_OUT_NM',
    'bwd_breakaway_v_thresh': 'BWD_BREAKAWAY_V_THRESH_RAD_S',
}


def apply_overrides(args):
    """Rebind module constants from the CLI, and report what changed."""
    changed = []
    for opt, const in OVERRIDES.items():
        value = getattr(args, opt, UNSET)
        if isinstance(value, _Unset):
            continue
        was = globals()[const]
        globals()[const] = value
        changed.append(f'{const}: {was!r} -> {value!r}')
    return changed


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split('\n\n')[0],
        epilog='Any constant not listed here lives at the top of this file, with '
               'the reasoning for its value next to it. Change it there.')
    ap.add_argument('--shakedown', action='store_true',
                    help='low-power variants: <=300 rpm, 25%% torque, one cycle')
    ap.add_argument('--bench', action='store_true',
                    help='only the plans runnable with the output locked and no gearbox')
    ap.add_argument('--include-backdrive', action='store_true',
                    help='also write the output-driven plans (off by default: the '
                         'unit is not back-driving well enough to test that way)')
    ap.add_argument('--backdrive-only', action='store_true',
                    help='write ONLY the output-driven plans (implies '
                         '--include-backdrive and --no-megabatch)')
    ap.add_argument('--no-megabatch', action='store_true',
                    help='skip the combined all-forward-tests plan')

    g = ap.add_argument_group(
        'parameter overrides',
        'Each rebinds the module constant of the same meaning for this run only; '
        'the file is not edited, so the defaults stay the documented ones.')
    g.add_argument('--unit', default=UNSET, choices=sorted(EFF_MAX_OUT_NM),
                   help=f'unit under test (default {UNIT})')
    g.add_argument('--slip-torque', default=UNSET, type=_opt_float, metavar='NM',
                   help='measured output-referred static slip torque, or "none" to '
                        f'go back to the provisional figure (default {T_SLIP_OUT_NM:g})')
    g.add_argument('--creep-onset', default=UNSET, type=_opt_float, metavar='NM',
                   help=f'output torque where measurable creep starts (default '
                        f'{T_CREEP_ONSET_OUT_NM:g})')
    g.add_argument('--output-cap', default=UNSET, type=float, metavar='NM',
                   help=f"customer's hard output torque limit (default "
                        f'{OUTPUT_TORQUE_CAP_NM:g})')
    g.add_argument('--eff-rpm', default=UNSET, type=_rpm_list, metavar='LIST',
                   help='efficiency input speeds, comma separated (default '
                        f'{",".join(str(x) for x in EFF_RPM)})')
    g.add_argument('--eff-step', default=UNSET, type=float, metavar='NM',
                   help=f'efficiency output torque step (default {EFF_STEP_OUT_NM:g})')
    g.add_argument('--eff-slip-margin', default=UNSET, type=float, metavar='F',
                   help='efficiency stops at this fraction of measured slip '
                        f'(default {EFF_SLIP_MARGIN:g})')
    g.add_argument('--vel-rpm', default=UNSET, type=_rpm_list, metavar='LIST',
                   help='velocity-ramp input speeds; accepts lo:hi:step ranges, e.g. '
                        '30:300:30,600:3600:300')
    g.add_argument('--throw', default=UNSET, type=float, metavar='RAD',
                   help=f'output half-throw each traverse ends at (default '
                        f'{OUTPUT_END_RAD:g}; the window is read from the config)')
    g.add_argument('--input-sign', default=UNSET, type=int, choices=(1, -1),
                   help='which way the output moves for a positive input command '
                        f'(default {INPUT_SIGN:+d})')
    g.add_argument('--peak-target', default=UNSET, type=float, metavar='RAD',
                   help='where a traverse may actually END UP, command plus predicted '
                        f'overshoot (default {PEAK_TARGET_RAD:g}; the config trip is '
                        'separate and stays where it is)')
    g.add_argument('--overshoot-lag', default=UNSET, type=_lag_map, metavar='SPEC',
                   help='overshoot time lag, seconds, as output=<s>[,input=<s>] '
                        f'(default output={OVERSHOOT_LAG_S["output"]:g},'
                        f'input={OVERSHOOT_LAG_S["input"]:g}). Re-fit from the velocity '
                        'ramps whenever the absorber loop is retuned')
    g.add_argument('--overshoot-torque', default=UNSET, type=float, metavar='RAD_PER_NM',
                   help='extra overshoot per Nm of output torque the shuttle carries '
                        f'(default {OVERSHOOT_TORQUE_RAD_PER_NM:g}; set 0 for the '
                        'no-load model only)')
    g.add_argument('--overshoot-margin', default=UNSET, type=float, metavar='RAD',
                   help='added to lag*v to cover scatter around the fit (default '
                        f'{OVERSHOOT_MARGIN_RAD:g})')

    b = ap.add_argument_group('back-drive overrides',
                              'Only meaningful with --include-backdrive.')
    b.add_argument('--bwd-eff-top-levels', default=UNSET, type=_top_levels,
                   metavar='SPEC',
                   help='back-drive efficiency speeds to run with only their top N '
                        'torque magnitudes, as <rpm>=<n>[,<rpm>=<n>]; n=0 drops the '
                        'speed, empty string runs every speed whole (default '
                        + ','.join(f'{k}={v}' for k, v in sorted(BWD_EFF_TOP_LEVELS.items()))
                        + ')')
    b.add_argument('--bwd-ramps', default=UNSET, type=int, metavar='N',
                   help='slip / breakaway ramps EACH WAY, one segment and one '
                        f'recentre per ramp (default {BWD_RAMPS_EACH_WAY})')
    b.add_argument('--bwd-slip-ceiling-margin', default=UNSET, type=float, metavar='F',
                   help='back-drive slip ramp ceiling, as a multiple of the measured '
                        f'slip torque (default {SLIP_CEILING_MARGIN:g})')
    b.add_argument('--bwd-slip-v-thresh', default=UNSET, type=float, metavar='RAD_S',
                   help='output speed that counts as a slip (default '
                        f'{BWD_SLIP_V_THRESH_RAD_S:g})')
    b.add_argument('--bwd-breakaway-ceiling', default=UNSET, type=float, metavar='NM',
                   help='back-drive breakaway ramp ceiling, output Nm, before the '
                        f'slip bound is applied (default '
                        f'{BWD_BREAKAWAY_CEILING_OUT_NM:g})')
    b.add_argument('--bwd-breakaway-v-thresh', default=UNSET, type=float, metavar='RAD_S',
                   help='output speed that counts as back-driving (default '
                        f'{BWD_BREAKAWAY_V_THRESH_RAD_S:g})')

    args = ap.parse_args()
    if args.backdrive_only:
        args.include_backdrive = True
        args.no_megabatch = True
    changed = apply_overrides(args)
    if changed:
        print('OVERRIDES (this run only, the file is unchanged):')
        for line in changed:
            print(f'  {line}')

    cfg = load_config()
    tests_dir = dyno_paths.dyno_test_directory
    print(f'{MODE}: ratio {cfg["ratio"]:g}, window +/-{cfg["half_window"]:g} rad, '
          f'stop_decel {cfg["stop_decel"]:g} rad/s^2, unit {UNIT}, '
          f'output cap {OUTPUT_TORQUE_CAP_NM:g} Nm'
          + ('  [SHAKEDOWN]' if args.shakedown else '')
          + ('  [BENCH]' if args.bench else '')
          + ('' if args.bench else
             '  [BACKDRIVE ONLY]' if args.backdrive_only else
             '  [+BACKDRIVE]' if args.include_backdrive else '  [FORWARD ONLY]'))
    if GRAVITY_AMPLITUDE_NM is not None and GRAVITY_AMPLITUDE_NM > 0.2 * EFF_STEP_OUT_NM:
        print(f'WARNING: flange gravity {GRAVITY_AMPLITUDE_NM:g} Nm is over 20% of the '
              f'{EFF_STEP_OUT_NM:g} Nm efficiency step: low levels will change '
              'driving/driven regime mid-traverse unless compensated')

    if args.bench:
        plans = build_bench_plans(cfg)
    elif args.backdrive_only:
        plans = build_backdrive_plans(cfg, args.shakedown)
        if args.shakedown:
            for _, recipe, _ in plans:
                if recipe:
                    recipe['name'] += '_shakedown'
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
        dropped = MEGABATCH_EXCLUDE + (
            'EFF_ALT' if MEGABATCH_EFFICIENCY == 'pos' else 'EFF_POS',)
        if not tag.startswith(BACKDRIVE_TAG) and tag not in dropped:
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
                why = MEGABATCH_EXCLUDE_REASON.get(tag, 'excluded by MEGABATCH_EXCLUDE')
                left = [r['name'] for t_, r, _ in plans if t_ == tag and r]
                for name in left:
                    print(f'  NOT in the batch: {name} -- {why}. Run it on its own, '
                          'watching, afterwards')
            other = 'EFF_ALT' if MEGABATCH_EFFICIENCY == 'pos' else 'EFF_POS'
            why = ('all four quadrants' if MEGABATCH_EFFICIENCY == 'alt'
                   else 'half the time, one rotation sense')
            for name in [r['name'] for t_, r, _ in plans if t_ == other and r]:
                print(f'  NOT in the batch: {name} -- the batch carries the '
                      f'{MEGABATCH_EFFICIENCY!r} variant ({why}). Flip '
                      'MEGABATCH_EFFICIENCY to swap them; running both would measure '
                      'efficiency twice')
            print('  duration is a LOWER bound wherever recentres are involved: the '
                  'expansion has no\n  sensor, so every recentre takes its '
                  f'"nothing to do" path and costs {RECENTRE_PARAMS["settle_s"]:g} s. On '
                  f'the rig each one that has to move adds up to\n  '
                  f'{RECENTRE_PARAMS["timeout_s"]:g} s more')

    if not args.bench:
        # A back-drive-only run has not written the forward plans and must not
        # call them stale; likewise a forward run and the back-drive files.
        prefixes = (('archimedes_bwd_',) if args.backdrive_only else
                    ('archimedes_',) if args.include_backdrive else
                    ('archimedes_fwd_',))
        report_stale(tests_dir, written, args.shakedown, prefixes)
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
