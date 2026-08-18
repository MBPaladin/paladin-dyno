"""
Parametric test builder: recipe -> keyframes -> (test yaml + trace csvs).

A *recipe* describes an experiment as a sequence of segments. Each segment
applies a pattern (sawtooth, step, ramp_release, dwell) to a primary
motor/control-mode channel while the secondary motor either holds zero or
steps through a list of levels (which is how a sawtooth becomes a grid
sweep). Segments compile to the exact same piecewise-linear keyframe CSVs
the hand-written generate_*.py scripts in tests/traces produce, and the
whole recipe compiles to a normal `behaviors:` test yaml that the runner
executes with no knowledge of the builder.

The `gridpoint` pattern is the exception: it compiles to a `grid_search`
behavior (no trace csv), preserving that behavior's per-setpoint log flags
for post-processing.

The recipe is embedded in the generated yaml under a `builder:` key (which
TestManager ignores) so a saved test is a single self-describing file that
the builder can reopen and edit. Generated artifacts:

    {tests}/ui_generated_tests/<name>.yaml
    {tests}/traces/ui_generated/<name>__<SEGID>.csv

`trace_file` entries use the "ui_generated/<file>" form because TestTrace
always reads from {tests}/traces/.

Invariants matching TestTrace validation: every trace starts at (0,0,0),
time strictly increases, and torque/velocity channels end at 0.
"""
import hashlib
import math
import os
import random
import re

import yaml

MOTORS = ('input', 'output')
MODES = ('torque', 'velocity', 'position')

# --- optional test preamble --------------------------------------------------
# A recipe-level (once per test, not per segment) measurement phase that runs
# before the first segment: a silent hold for an ambient noise floor, or that
# hold followed by a periodic multisine torque excitation at standstill for
# FRF / resonance analysis. Absent key = 'none', so existing recipes are
# untouched. `role` targets a port role the loaded bench declares (or 'all');
# per-segment `lead_in_s` is unrelated and stays exactly as it is.
PREAMBLE_MODES = ('none', 'hold', 'multisine')
PREAMBLE_LEVELS = ('gentle', 'normal', 'hard')
# One user-facing amplitude control, defined as target SNR of each excited
# line over the measured per-bin noise floor -- never as a fraction of a motor
# rating, which would be two orders of magnitude wrong across the two ports.
PREAMBLE_SNR_DB = {'gentle': 20.0, 'normal': 30.0, 'hard': 40.0}

# Fixed multisine design parameters. Deliberately not user knobs: the whole
# scheme rests on exact periodicity at the logging rate, and every derived
# quantity below is stored into the generated yaml so analysis and operators
# can see it without re-deriving.
MULTISINE_PERIOD_S = 2.0          # -> 2000 samples/period at 1 kHz, df = 0.5 Hz
MULTISINE_PERIODS = 12            # first 2 are ramp-in + settling, discarded
MULTISINE_DISCARD_PERIODS = 2
MULTISINE_BAND_HZ = (5.0, 200.0)  # low: free-shaft rigid body; high: no AA filter
MULTISINE_LINES_PER_DECADE = 24
MULTISINE_DETECTION_FRACTION = 0.25  # odd lines silenced to measure odd distortion


def default_preamble():
    return {'mode': 'none', 'quiet_s': 3.0, 'role': 'output',
            'level': 'normal', 'seed': 1234}


def design_multisine(seed, cycle_time_s=0.001):
    """Derive the multisine line set. Deterministic given `seed`.

    Log-spaced candidates across the band are snapped to the nearest ODD FFT
    bin (odd-only excitation leaves every even bin empty as a clean measure of
    even-order distortion), deduplicated, and a seeded ~25% are silenced as
    *detection lines* measuring odd-order distortion -- the dominant kind for
    friction and backlash. Excited lines get Schroeder phases (low crest
    factor without windowing, which would destroy the periodicity everything
    here depends on).
    """
    n_samples = int(round(MULTISINE_PERIOD_S / cycle_time_s))
    df = 1.0 / MULTISINE_PERIOD_S
    f_lo, f_hi = MULTISINE_BAND_HZ
    n_candidates = int(round(MULTISINE_LINES_PER_DECADE
                             * math.log10(f_hi / f_lo))) + 1
    bins = []
    for i in range(n_candidates):
        f = f_lo * (f_hi / f_lo) ** (i / (n_candidates - 1))
        k = int(round(f / df))
        k = k if k % 2 else (k + 1 if (f / df) >= k else k - 1)  # nearest odd
        if f_lo / df <= k <= f_hi / df and k not in bins:
            bins.append(k)
    bins.sort()

    rng = random.Random(int(seed))
    n_detect = max(1, int(round(MULTISINE_DETECTION_FRACTION * len(bins))))
    detection = set(rng.sample(bins, n_detect))
    excited = [k for k in bins if k not in detection]

    lines = []
    for j, k in enumerate(bins):
        if k in detection:
            lines.append({'bin': k, 'freq_hz': k * df, 'detection': True})
        else:
            # Schroeder phase over the excited lines only, indexed in order.
            m = excited.index(k)
            phase = -math.pi * m * (m + 1) / len(excited)
            lines.append({'bin': k, 'freq_hz': k * df, 'detection': False,
                          'phase': round(phase % (2 * math.pi), 6)})
    return {
        'period_s': MULTISINE_PERIOD_S,
        'samples_per_period': n_samples,
        'periods': MULTISINE_PERIODS,
        'discard_periods': MULTISINE_DISCARD_PERIODS,
        'df_hz': df,
        'band_hz': list(MULTISINE_BAND_HZ),
        'seed': int(seed),
        'n_excited': len(excited),
        'lines': lines,
    }


def validate_preamble(preamble, roles=None):
    """Human-readable issues with a recipe's preamble block (empty = clean)."""
    if not preamble or preamble.get('mode', 'none') == 'none':
        return []
    issues = []
    mode = preamble.get('mode')
    if mode not in PREAMBLE_MODES:
        issues.append(f'preamble mode must be one of {PREAMBLE_MODES}, got {mode!r}')
        return issues
    quiet_s = float(preamble.get('quiet_s', 0.0))
    if quiet_s < 1.0:
        # The quiet hold is what anchors the excitation amplitude (and the
        # ambient floor); under a second there is nothing to average.
        issues.append(f'preamble quiet_s must be >= 1.0 s, got {quiet_s:g}')
    if mode == 'multisine':
        level = preamble.get('level', 'normal')
        if level not in PREAMBLE_LEVELS:
            issues.append(f'preamble level must be one of {PREAMBLE_LEVELS}, '
                          f'got {level!r}')
        try:
            int(preamble.get('seed', 1234))
        except (TypeError, ValueError):
            issues.append('preamble seed must be an integer')
        role = preamble.get('role', 'output')
        if roles is not None and role != 'all' and role not in roles:
            issues.append(f'preamble role {role!r} is not declared by this '
                          f'bench (declared: {", ".join(roles)}, or "all")')
    return issues

# --- position command shaping ----------------------------------------------
# A piecewise-linear position trace has a velocity discontinuity at every
# keyframe corner; the drive's position loop chases it with an inertial torque
# spike (tau ~ J * dv/dt with dt of one control cycle). Position channels are
# therefore smoothed with accel-limited parabolic blends, and validation
# rejects any remaining corner whose velocity step exceeds what the rig's
# acceleration limit could produce within CORNER_DT.
DEFAULT_POSITION_ACCEL = 5.0    # [units/s^2] default blend acceleration
POSITION_BLEND_SAMPLES = 10     # subdivisions per parabolic corner blend
CORNER_DT = 0.05                # [s] window used to judge corner velocity steps
END_HOLD_S = 0.5                # [s] trailing hold so the final corner can blend

# The blend only touches interior corners, so the first keyframe keeps whatever
# discontinuity it starts with: a segment whose first secondary level is
# non-zero (a -pi..pi position sweep, say) steps the commanded velocity at t=0
# with nothing in front of it to blend into. Settle time can't fix that -- it is
# applied *after* the ramp onto each level. A lead-in hold at zero can: it makes
# that first ramp an interior corner, and gives every log a short at-rest
# baseline to tare against. Per-segment, overridable, in `lead_in_s`.
START_HOLD_S = 1.0              # [s] default lead-in hold at zero

GENERATED_TEST_DIR = 'ui_generated_tests'   # under the tests directory
GENERATED_TRACE_DIR = 'ui_generated'        # under tests/traces/

# Pattern parameter metadata: key -> (label, default, type). The UI builds its
# forms from this, and compile falls back to these defaults for missing keys.
PATTERNS = {
    'sawtooth': {
        'amplitude':    ('Amplitude', 10.0, float),
        'rate':         ('Ramp rate [units/s]', 1.0, float),
        'peak_dwell_s': ('Dwell at peak [s]', 1.0, float),
        'bipolar':      ('Bipolar (sweep to -amplitude too)', True, bool),
        'cycles':       ('Cycles', 1, int),
    },
    'step': {
        'levels':  ('Levels (comma separated)', [5.0, 10.0], list),
        'rate':    ('Ramp rate [units/s]', 1.0, float),
        'dwell_s': ('Dwell per level [s]', 2.0, float),
    },
    'ramp_release': {
        'amplitude': ('Ramp target', 5.0, float),
        'rate':      ('Ramp rate [units/s]', 0.1, float),
        'release_s': ('Release back to 0 [s]', 1.0, float),
        'rest_s':    ('Rest after release [s]', 2.0, float),
        'bipolar':   ('Repeat in negative direction', True, bool),
        'cycles':    ('Cycles', 1, int),
    },
    'dwell': {
        'duration_s': ('Duration [s]', 5.0, float),
    },
    # Compiles to a real `grid_search` behavior (not a trace): every gridpoint
    # gets its own log flag with ramps/settles untagged, so post-processing can
    # group by flag. Axes: pattern-motor levels x other-motor levels, with the
    # pattern/other control modes required to be the velocity/torque pair.
    'gridpoint': {
        'levels':               ('Pattern-motor levels (comma separated)',
                                 [15.0, 30.0, 60.0, 100.0], list),
        'duration_per_point_s': ('Hold per gridpoint [s]', 3.0, float),
        'settle_time_s':        ('Settle before logging [s]', 0.5, float),
        'transition_rate':      ('Transition rate (fraction of limit)', 0.25, float),
        'velocity_inner':       ('Velocity is inner loop (default: torque)', False, bool),
        'continuous_torque':    ('Continuous torque rating [Nm]', 110.0, float),
    },
}

# Fallbacks mirroring the GridSearch hardcodes, used when the rig config has no
# `motor_limits.continuous_torque` for the motor commanded in torque mode.
GRID_CONT_TORQUE_FALLBACK = {'input': 4.0, 'output': 110.0}


# --- level list generation --------------------------------------------------
# Level lists are always stored literally (a plain list of floats); the
# start/stop/n generator below is a convenience that writes into that list, not
# a second representation of it. The UI keeps the generator's inputs alongside
# the recipe under `levels_gen` purely so the boxes can be restored on reopen.

LEVEL_DECIMALS = 2      # 0.01 rad / Nm / (rad/s) is finer than any test needs
MAX_LEVEL_DECIMALS = 6  # widened only to keep a tight sweep's points distinct


def _round_levels(values):
    """Round to LEVEL_DECIMALS, widening only as far as needed to keep distinct
    inputs distinct -- a log sweep through small magnitudes would otherwise
    collapse several points onto the same value."""
    distinct = len(set(values))
    for decimals in range(LEVEL_DECIMALS, MAX_LEVEL_DECIMALS + 1):
        rounded = [round(v, decimals) for v in values]
        if len(set(rounded)) == distinct:
            break
    return [0.0 if r == 0 else r for r in rounded]  # normalize -0.0


def generate_levels(start, stop, n, spacing='linear', mirror=False):
    """`n` levels from `start` to `stop`, rounded to LEVEL_DECIMALS.

    `spacing` is 'linear' or 'log' (log needs both endpoints non-zero and of
    the same sign). `mirror` appends the descending sweep back to `start`
    without repeating the turnaround point, so hysteresis shows up in one run.
    """
    n = int(n)
    if n < 1:
        raise ValueError('Number of levels must be at least 1')
    if n == 1:
        levels = _round_levels([float(start)])
    elif spacing == 'linear':
        levels = _round_levels([start + (stop - start) * i / (n - 1)
                                for i in range(n)])
    elif spacing == 'log':
        if start == 0 or stop == 0 or start * stop < 0:
            raise ValueError('Log spacing needs start and stop non-zero '
                             'and of the same sign')
        sign = math.copysign(1.0, start)
        lo, hi = math.log(abs(start)), math.log(abs(stop))
        levels = _round_levels([sign * math.exp(lo + (hi - lo) * i / (n - 1))
                                for i in range(n)])
    else:
        raise ValueError(f'Unknown level spacing: {spacing}')
    if mirror and len(levels) > 1:
        levels = levels + levels[-2::-1]
    return levels


def is_gridpoint(segment):
    return segment.get('pattern') == 'gridpoint'


def gridpoint_torque_motor(segment):
    """The motor commanded in torque mode ('input'/'output'), or None if the
    segment's modes aren't the velocity/torque pair grid_search requires."""
    primary = segment['primary']
    sec_motor = 'output' if primary['motor'] == 'input' else 'input'
    modes = {primary['motor']: primary['control_mode'],
             sec_motor: segment['secondary']['control_mode']}
    if set(modes.values()) != {'velocity', 'torque'}:
        return None
    return next(m for m, mode in modes.items() if mode == 'torque')


def gridpoint_settings(segment):
    """The `grid_search` behavior settings a gridpoint segment compiles to.
    loop_order[0] is the outer axis; torque is the inner loop by default
    because torque transitions settle much faster than velocity ones."""
    primary = segment['primary']
    secondary = segment['secondary']
    sec_motor = 'output' if primary['motor'] == 'input' else 'input'
    params = segment.get('params', {})
    p = lambda key: _param(params, 'gridpoint', key)
    loop_order = (['torque', 'velocity'] if p('velocity_inner')
                  else ['velocity', 'torque'])
    return {
        primary['motor'] + '_motor': {
            'control_mode': primary['control_mode'],
            'command_list': [float(v) for v in p('levels')]},
        sec_motor + '_motor': {
            'control_mode': secondary['control_mode'],
            'command_list': [float(v) for v in secondary.get('levels', [0.0])]},
        'loop_order': loop_order,
        'duration_per_point_s': float(p('duration_per_point_s')),
        'settle_time_s': float(p('settle_time_s')),
        'transition_rate': float(p('transition_rate')),
        'continuous_torque': float(p('continuous_torque')),
    }


def gridpoint_preview_rows(segment, limits=None):
    """Approximate keyframes for the live plot, in compile_segment's
    (cols, rows) form. Ramp rates come from transition_rate x rig limits like
    GridSearch.ramp; the cooldown holds GridSearch inserts above the continuous
    torque rating are not modeled. The post-save expansion is exact."""
    if gridpoint_torque_motor(segment) is None:
        raise ValueError('gridpoint needs one motor in velocity and one in torque')
    settings = gridpoint_settings(segment)
    primary = segment['primary']
    sec_motor = 'output' if primary['motor'] == 'input' else 'input'
    prim_mode = primary['control_mode']
    sec_mode = segment['secondary']['control_mode']

    tr = settings['transition_rate']
    # Each axis ramps at a fraction of the rating of the motor driving it;
    # gridpoint_torque_motor above established the modes are the velocity/torque
    # pair, so each mode maps to exactly one motor.
    mode_limits = ({prim_mode: limits[primary['motor']], sec_mode: limits[sec_motor]}
                   if limits else None)
    rates = {'velocity': tr * (mode_limits['velocity']['acceleration'] if limits else 1.0),
             'torque': tr * (mode_limits['torque']['rotatum'] if limits else 1.0)}
    axes = {prim_mode: settings[primary['motor'] + '_motor']['command_list'],
            sec_mode: settings[sec_motor + '_motor']['command_list']}
    outer_mode, inner_mode = settings['loop_order']

    state = {'velocity': 0.0, 'torque': 0.0}
    rows = [(0.0, 0.0, 0.0)]
    t = 0.0

    def emit():
        rows.append((t, state[prim_mode], state[sec_mode]))

    def goto(setpoint):
        nonlocal t
        dt = max(abs(setpoint[m] - state[m]) / max(rates[m], 1e-12)
                 for m in setpoint)
        state.update(setpoint)
        if dt > 0:
            t += dt
            emit()

    def hold(duration):
        nonlocal t
        if duration > 0:
            t += duration
            emit()

    for outer in axes[outer_mode]:
        for inner in axes[inner_mode]:
            goto({outer_mode: float(outer), inner_mode: float(inner)})
            hold(settings['settle_time_s'])
            hold(settings['duration_per_point_s'])
    goto({'velocity': 0.0, 'torque': 0.0})
    hold(0.1)  # matches GridSearch's trailing hold at zero

    cols = ['time',
            f"{primary['motor']}_motor_{prim_mode}",
            f"{sec_motor}_motor_{sec_mode}"]
    return cols, rows


def default_segment(seg_id='SEG1'):
    return {
        'id': seg_id,
        'repeats': 1,
        'lead_in_s': START_HOLD_S,
        'primary': {'motor': 'input', 'control_mode': 'position',
                    'accel': DEFAULT_POSITION_ACCEL},
        'secondary': {'control_mode': 'torque', 'levels': [0.0],
                      'rate': 1.0, 'settle_s': 1.0},
        'pattern': 'sawtooth',
        'params': {k: v[1] for k, v in PATTERNS['sawtooth'].items()},
    }


def _param(params, pattern, key):
    return params.get(key, PATTERNS[pattern][key][1])


# --- pattern compilation ----------------------------------------------------
# A pattern compiles to a list of (t, value) keyframes for the primary
# channel, starting at (0, 0) and ending back at value 0 (except `step`,
# which appends its own return-to-zero ramp). Times are relative.

def _pattern_keys(pattern, params):
    p = lambda key: _param(params, pattern, key)
    keys = [(0.0, 0.0)]
    t = 0.0

    def move(value, rate):
        nonlocal t
        dt = abs(value - keys[-1][1]) / max(abs(rate), 1e-12)
        if dt > 0:
            t += dt
            keys.append((t, value))

    def hold(duration):
        nonlocal t
        if duration > 0:
            t += duration
            keys.append((t, keys[-1][1]))

    if pattern == 'sawtooth':
        for _ in range(max(1, int(p('cycles')))):
            move(p('amplitude'), p('rate'))
            hold(p('peak_dwell_s'))
            if p('bipolar'):
                move(-p('amplitude'), p('rate'))
                hold(p('peak_dwell_s'))
            move(0.0, p('rate'))
    elif pattern == 'step':
        for level in p('levels'):
            move(float(level), p('rate'))
            hold(p('dwell_s'))
        move(0.0, p('rate'))
    elif pattern == 'ramp_release':
        for _ in range(max(1, int(p('cycles')))):
            directions = (1.0, -1.0) if p('bipolar') else (1.0,)
            for sign in directions:
                move(sign * p('amplitude'), p('rate'))
                t += max(p('release_s'), 1e-3)
                keys.append((t, 0.0))
                hold(p('rest_s'))
    elif pattern == 'dwell':
        hold(p('duration_s'))
    else:
        raise ValueError(f'Unknown pattern: {pattern}')

    if len(keys) == 1:  # degenerate (all params zero): still emit a point
        keys.append((1.0, 0.0))
    return keys


def compile_segment(segment):
    """Compile one segment to (column_names, rows).

    rows are (time, primary_value, secondary_value) with the secondary motor
    stepping through its levels: ramp to level -> settle -> run pattern ->
    next level, ending with a ramp back to 0.
    """
    if is_gridpoint(segment):
        raise ValueError('gridpoint segments compile to a grid_search '
                         'behavior, not a trace')
    pattern = segment['pattern']
    params = segment.get('params', {})
    primary = segment['primary']
    secondary = segment['secondary']
    sec_motor = 'output' if primary['motor'] == 'input' else 'input'

    levels = [float(v) for v in secondary.get('levels', [0.0])] or [0.0]
    sec_rate = max(abs(float(secondary.get('rate', 1.0))), 1e-12)
    settle_s = float(secondary.get('settle_s', 0.0))

    body = _pattern_keys(pattern, params)

    rows = [(0.0, 0.0, 0.0)]
    t = 0.0
    sec = 0.0
    lead_in_s = max(float(segment.get('lead_in_s', START_HOLD_S)), 0.0)
    if lead_in_s > 0:
        t = lead_in_s
        rows.append((t, 0.0, 0.0))
    for level in levels:
        if level != sec:
            t += abs(level - sec) / sec_rate
            sec = level
            rows.append((t, 0.0, sec))
        if settle_s > 0:
            t += settle_s
            rows.append((t, 0.0, sec))
        for dt, value in body[1:]:  # body[0] is the (0, 0) anchor
            rows.append((t + dt, value, sec))
        t = rows[-1][0]
    if sec != 0.0:
        t += abs(sec) / sec_rate
        rows.append((t, 0.0, 0.0))

    # Accel-limited corner blending for position channels (see module
    # constants). A trailing hold gives the final corner room to blend.
    chan_modes = ((1, primary['control_mode']), (2, secondary['control_mode']))
    if any(m == 'position' for _, m in chan_modes):
        last = rows[-1]
        rows.append((last[0] + END_HOLD_S, last[1], last[2]))
    accel = abs(float(primary.get('accel', DEFAULT_POSITION_ACCEL))) or \
        DEFAULT_POSITION_ACCEL
    for idx, m in chan_modes:
        if m == 'position':
            rows = _smooth_position_corners(rows, idx, accel)

    cols = ['time',
            f"{primary['motor']}_motor_{primary['control_mode']}",
            f"{sec_motor}_motor_{secondary['control_mode']}"]
    return cols, rows


def _smooth_position_corners(rows, idx, accel):
    """Replace velocity-discontinuous corners of channel `idx` with parabolic
    blends of duration |dv|/accel (position and velocity continuous at both
    ends). Timestamps are only inserted, never shifted, so the other channel
    stays synchronized (it is linearly interpolated at the new times). A blend
    is clamped when the neighboring intervals are too short to host it; the
    residual corner step is then caught by validate_segment."""
    times = [r[0] for r in rows]
    vals = [r[idx] for r in rows]
    other_idx = 2 if idx == 1 else 1
    other = [r[other_idx] for r in rows]

    def make_row(t, v, o):
        return (t, v, o) if idx == 1 else (t, o, v)

    def interp_other(t):
        for k in range(len(times) - 1):
            if times[k] <= t <= times[k + 1]:
                dt = times[k + 1] - times[k]
                f = 0.0 if dt <= 0 else (t - times[k]) / dt
                return other[k] + f * (other[k + 1] - other[k])
        return other[-1]

    new_rows = [rows[0]]
    prev_blend_end = times[0]
    for i in range(1, len(rows) - 1):
        dt0 = times[i] - times[i - 1]
        dt1 = times[i + 1] - times[i]
        if dt0 <= 0 or dt1 <= 0:
            new_rows.append(rows[i])
            continue
        v0 = (vals[i] - vals[i - 1]) / dt0
        v1 = (vals[i + 1] - vals[i]) / dt1
        dv = v1 - v0
        if abs(dv) < 1e-9:
            new_rows.append(rows[i])
            prev_blend_end = times[i]
            continue
        # Half the blend sits in each adjacent interval; reserve half of the
        # following interval for the next corner's blend.
        room = 2.0 * min(times[i] - prev_blend_end, dt1 / 2.0)
        T = min(abs(dv) / accel, max(room, 0.0))
        if T < 1e-6:
            new_rows.append(rows[i])
            prev_blend_end = times[i]
            continue
        for k in range(POSITION_BLEND_SAMPLES + 1):
            tau = -T / 2.0 + T * k / POSITION_BLEND_SAMPLES
            t = times[i] + tau
            if t <= new_rows[-1][0] + 1e-9:
                continue
            v = vals[i] + v0 * tau + dv / (2.0 * T) * (tau + T / 2.0) ** 2
            new_rows.append(make_row(t, v, interp_other(t)))
        prev_blend_end = times[i] + T / 2.0
    new_rows.append(rows[-1])
    return new_rows


# --- validation -------------------------------------------------------------

_FORBIDDEN_MODE_PAIRS = ({'position', 'position'}, {'velocity', 'velocity'},
                         {'position', 'velocity'})


def reaction_torque_issue(output_torque_peak, limits):
    """Message if the load motor's torque overtorques the DUT through the
    shaft, else None.

    The two motors share a shaft, so torque the load applies is reacted at the
    DUT divided by the DUT's gear ratio -- the same conversion the feed-forward
    in dyno_controller.step uses. Per-motor ceilings cannot express this: a
    command well inside LOAD's own rating still overtorques a derated DUT on a
    direct-drive (1:1) rig.

    Only this direction is checked. The reverse -- DUT torque the load motor
    cannot hold -- is not an overtorque of the load motor; the shaft simply
    accelerates, which the velocity and acceleration limits govern.

    Skipped when `limits['coupled']` is False: a load_only rig has no motor or
    coupling on the DUT side (see load_only in <mode>_dyno_config.yaml), so
    there is no path for the reaction. Absent key means coupled, which is the
    safe default -- clearing load_only re-arms this check automatically.
    """
    if not limits.get('coupled', True):
        return None
    peak = abs(output_torque_peak)
    ratio = abs(limits['input'].get('gear_ratio') or 1)
    dut = limits['input']['torque']
    reacted = peak / ratio
    if reacted <= dut:
        return None
    return (f'peak {peak:g} Nm reacts {reacted:g} Nm at the DUT '
            f'(limit {dut:g}) through gear_ratio {ratio:g}')


def validate_segment(segment, limits=None):
    """Return a list of human-readable issues (empty = clean). Mirrors the
    TestTrace asserts so problems surface while editing, not at load time."""
    if is_gridpoint(segment):
        return _validate_gridpoint(segment, limits)
    issues = []
    primary = segment['primary']
    secondary = segment['secondary']
    modes = {primary['control_mode'], secondary['control_mode']}
    for pair in _FORBIDDEN_MODE_PAIRS:
        if modes == pair:
            issues.append(f"Control mode pair not allowed: {sorted(pair)}")

    cols, rows = compile_segment(segment)
    times = [r[0] for r in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        issues.append('Time is not strictly increasing (check rates > 0)')

    for idx, col in ((1, cols[1]), (2, cols[2])):
        # cols are '<motor>_motor_<mode>'; each is held to its own motor's
        # ratings, so `limits` is indexed by motor first.
        motor, mode = col.split('_', 1)[0], col.rsplit('_', 1)[-1]
        motor_limits = limits[motor] if limits else None
        values = [r[idx] for r in rows]
        rates = [abs((b - a) / (tb - ta)) if tb > ta else 0.0
                 for (ta, a), (tb, b) in zip(zip(times, values),
                                             zip(times[1:], values[1:]))]
        peak, peak_rate = max(map(abs, values)), max(rates, default=0.0)
        if mode in ('torque', 'velocity') and values[-1] != 0:
            issues.append(f'{col} must end at 0 (ends at {values[-1]:g})')
        if motor_limits:
            if mode == 'torque':
                if peak > motor_limits['torque']:
                    issues.append(f'{col}: peak {peak:g} Nm exceeds limit {motor_limits["torque"]:g}')
                if motor == 'output':
                    reaction = reaction_torque_issue(peak, limits)
                    if reaction:
                        issues.append(f'{col}: {reaction}')
                if peak_rate > motor_limits['rotatum']:
                    issues.append(f'{col}: rotatum {peak_rate:g} Nm/s exceeds limit {motor_limits["rotatum"]:g}')
            elif mode == 'velocity':
                if peak > motor_limits['velocity']:
                    issues.append(f'{col}: peak {peak:g} rad/s exceeds limit {motor_limits["velocity"]:g}')
                if peak_rate > motor_limits['acceleration']:
                    issues.append(f'{col}: accel {peak_rate:g} rad/s^2 exceeds limit {motor_limits["acceleration"]:g}')
            elif mode == 'position' and peak_rate > motor_limits['velocity']:
                issues.append(f'{col}: implied velocity {peak_rate:g} rad/s exceeds limit {motor_limits["velocity"]:g}')
        if mode == 'position':
            # Corners must be velocity-continuous (within what the accel limit
            # can absorb in CORNER_DT); a raw corner is an accel impulse that
            # spikes torque through the position loop.
            signed = [(b - a) / (tb - ta) if tb > ta else 0.0
                      for (ta, a), (tb, b) in zip(zip(times, values),
                                                  zip(times[1:], values[1:]))]
            if signed:
                if abs(signed[0]) > 1e-6:
                    issues.append(f'{col} starts moving at t=0; increase the '
                                  "segment's lead-in hold so the trace starts "
                                  'at rest')
                if abs(signed[-1]) > 1e-6:
                    issues.append(f'{col} ends while still moving')
            if motor_limits:
                step_limit = motor_limits['acceleration'] * CORNER_DT
                max_step = max((abs(b - a) for a, b in zip(signed, signed[1:])),
                               default=0.0)
                if max_step > step_limit:
                    issues.append(
                        f'{col}: corner velocity step {max_step:g} exceeds '
                        f'{step_limit:g} (accel limit x {CORNER_DT:g}s) — '
                        'blends were clamped for room; reduce the pattern '
                        'rate or increase dwell/settle times')
    return issues


def _validate_gridpoint(segment, limits=None):
    """Mirrors the GridSearch asserts so problems surface while editing."""
    if gridpoint_torque_motor(segment) is None:
        return ['gridpoint needs one motor in velocity mode and one in torque']
    issues = []
    settings = gridpoint_settings(segment)
    for motor in MOTORS:
        chan = settings[motor + '_motor']
        if not chan['command_list']:
            issues.append(f'{motor} motor has no levels')
            continue
        if limits:
            peak = max(abs(v) for v in chan['command_list'])
            limit = limits[motor].get(chan['control_mode'])
            if limit is not None and peak > limit:
                issues.append(f"{motor}_motor_{chan['control_mode']}: peak "
                              f'{peak:g} exceeds limit {limit:g}')
            if motor == 'output' and chan['control_mode'] == 'torque':
                reaction = reaction_torque_issue(peak, limits)
                if reaction:
                    issues.append(f'output_motor_torque: {reaction}')
    if settings['duration_per_point_s'] <= 0:
        issues.append('Hold per gridpoint must be > 0')
    if settings['settle_time_s'] < 0:
        issues.append('Settle time must be >= 0')
    if not 0 < settings['transition_rate'] <= 1:
        issues.append('Transition rate must be in (0, 1]')
    if settings['continuous_torque'] <= 0:
        issues.append('Continuous torque rating must be > 0')
    return issues


def validate_recipe(recipe, limits=None, roles=None):
    issues = []
    if not recipe.get('segments'):
        issues.append('Recipe has no segments')
    issues.extend(validate_preamble(recipe.get('preamble'), roles))
    seen = set()
    for seg in recipe.get('segments', []):
        if seg['id'] in seen:
            issues.append(f"Duplicate segment id: {seg['id']}")
        seen.add(seg['id'])
        issues.extend(f"[{seg['id']}] {msg}" for msg in validate_segment(seg, limits))
    return issues


# --- serialization ----------------------------------------------------------

def _csv_text(cols, rows):
    lines = [','.join(cols)]
    for row in rows:
        lines.append(','.join(str(round(v, 9)) for v in row))
    return '\n'.join(lines) + '\n'


def recipe_hash(recipe):
    return hashlib.sha256(
        yaml.safe_dump(recipe, sort_keys=True).encode()).hexdigest()[:16]


def sanitize_name(name):
    name = re.sub(r'[^A-Za-z0-9_\-]+', '_', name.strip()).strip('_')
    return name or 'untitled'


def build_yaml_dict(recipe):
    """Assemble the runner-facing yaml dict (behaviors + embedded recipe)."""
    name = sanitize_name(recipe['name'])
    behaviors = []
    preamble = recipe.get('preamble')
    if preamble and preamble.get('mode', 'none') != 'none':
        settings = {'mode': preamble['mode'],
                    'quiet_s': float(preamble.get('quiet_s', 3.0))}
        if preamble['mode'] == 'multisine':
            seed = int(preamble.get('seed', 1234))
            settings.update({
                'role': preamble.get('role', 'output'),
                'level': preamble.get('level', 'normal'),
                'seed': seed,
                # The derived line taxonomy is stored (not just the seed) so
                # analysis and operators can read the excited/detection bins
                # straight off the yaml without re-deriving them.
                'design': design_multisine(seed),
            })
        behaviors.append({'id': 'PREAMBLE', 'type': 'preamble',
                          'settings': settings})
    for seg in recipe['segments']:
        if is_gridpoint(seg):
            # Real grid_search behavior: per-setpoint log flags, no trace csv.
            # Repeats are ignored (the UI pins them to 1 for gridpoints).
            behaviors.append({'id': seg['id'], 'type': 'grid_search',
                              'settings': gridpoint_settings(seg)})
            continue
        primary = seg['primary']
        secondary = seg['secondary']
        sec_motor = 'output' if primary['motor'] == 'input' else 'input'
        settings = {
            primary['motor'] + '_motor': {'control_mode': primary['control_mode']},
            sec_motor + '_motor': {'control_mode': secondary['control_mode']},
            'trace_file': f"{GENERATED_TRACE_DIR}/{name}__{seg['id']}.csv",
        }
        behavior = {'id': seg['id'], 'type': 'test_trace', 'settings': settings}
        repeats = int(seg.get('repeats', 1))
        if repeats > 1:
            behavior = {'id': seg['id'] + '_LOOP', 'type': 'loop',
                        'settings': {'loop_count': repeats},
                        'behaviors': [behavior]}
        behaviors.append(behavior)
    return {
        'builder': {'version': 1, 'recipe_hash': recipe_hash(recipe), **recipe},
        'behaviors': behaviors,
    }


def rel_test_path(path, tests_dir):
    """Return `path` in the '/'-separated, tests-dir-relative form TestManager
    consumes, or None if it lies outside that tree.

    TestManager resolves both the plan (`{tests}/{file}`) and every `trace_file`
    a behavior references (`{tests}/traces/{file}`) relative to the tests
    directory, so a plan stored anywhere else cannot run as-is: the yaml may
    well open while its traces silently miss. Callers reject rather than
    guess."""
    tests_root = os.path.realpath(tests_dir)
    target = os.path.realpath(path)
    try:
        if os.path.commonpath([tests_root, target]) != tests_root:
            return None
    except ValueError:  # different drives (Windows) -> never inside
        return None
    return os.path.relpath(target, tests_root).replace(os.sep, '/')


def list_test_files(tests_dir):
    """Every runnable test plan under the tests directory, as '/'-separated
    relative paths. Recursive, so generated tests under ui_generated_tests/
    are listed under the same name they are armed with (a flat listing showed
    them only after the builder mirrored them in, and as a different string).
    `traces/` is skipped — it holds the CSVs behaviors reference, not plans."""
    found = []
    for root, dirs, files in os.walk(tests_dir):
        dirs[:] = [d for d in dirs if d != 'traces']
        for name in files:
            if name.endswith(('.yaml', '.yml')):
                rel = os.path.relpath(os.path.join(root, name), tests_dir)
                found.append(rel.replace(os.sep, '/'))
    return sorted(found)


def save_test(recipe, tests_dir):
    """Write the yaml + csv artifacts. Returns the test file path relative to
    the tests directory (the form TestManager / the GUI consume)."""
    name = sanitize_name(recipe['name'])
    recipe = dict(recipe, name=name)

    yaml_dir = os.path.join(tests_dir, GENERATED_TEST_DIR)
    trace_dir = os.path.join(tests_dir, 'traces', GENERATED_TRACE_DIR)
    os.makedirs(yaml_dir, exist_ok=True)
    os.makedirs(trace_dir, exist_ok=True)

    for seg in recipe['segments']:
        if is_gridpoint(seg):
            continue  # compiles to a grid_search behavior, no trace csv
        cols, rows = compile_segment(seg)
        csv_path = os.path.join(trace_dir, f"{name}__{seg['id']}.csv")
        with open(csv_path, 'w', newline='') as f:
            f.write(_csv_text(cols, rows))

    doc = build_yaml_dict(recipe)
    yaml_path = os.path.join(yaml_dir, f'{name}.yaml')
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    return f'{GENERATED_TEST_DIR}/{name}.yaml'


def load_recipe(yaml_path, tests_dir=None):
    """Load an embedded recipe from a generated test yaml.

    Returns (recipe, stale_reasons). stale_reasons is non-empty if the stored
    hash doesn't match the recipe (hand edit) or the on-disk CSVs differ from
    what the recipe would regenerate.
    """
    with open(yaml_path) as f:
        doc = yaml.safe_load(f)
    builder = doc.get('builder')
    if not builder:
        return None, ['No builder recipe embedded in this yaml (view-only)']
    stored_hash = builder.pop('recipe_hash', None)
    builder.pop('version', None)
    recipe = builder

    stale = []
    if stored_hash and stored_hash != recipe_hash(recipe):
        stale.append('recipe_hash mismatch: yaml was edited outside the builder')
    if tests_dir:
        name = sanitize_name(recipe.get('name', ''))
        trace_dir = os.path.join(tests_dir, 'traces', GENERATED_TRACE_DIR)
        for seg in recipe.get('segments', []):
            if is_gridpoint(seg):
                continue  # no trace csv to compare
            csv_path = os.path.join(trace_dir, f"{name}__{seg['id']}.csv")
            try:
                with open(csv_path, newline='') as f:
                    on_disk = f.read()
            except OSError:
                stale.append(f"missing trace csv: {os.path.basename(csv_path)}")
                continue
            cols, rows = compile_segment(seg)
            if on_disk.replace('\r\n', '\n') != _csv_text(cols, rows):
                stale.append(f"stale trace csv: {os.path.basename(csv_path)} "
                             '(regenerate by re-saving)')
    return recipe, stale
