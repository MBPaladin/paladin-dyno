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
import os
import re

import yaml

MOTORS = ('input', 'output')
MODES = ('torque', 'velocity', 'position')

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
    rates = {'velocity': tr * (limits['acceleration'] if limits else 1.0),
             'torque': tr * (limits['rotatum'] if limits else 1.0)}
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
        mode = col.rsplit('_', 1)[-1]
        values = [r[idx] for r in rows]
        rates = [abs((b - a) / (tb - ta)) if tb > ta else 0.0
                 for (ta, a), (tb, b) in zip(zip(times, values),
                                             zip(times[1:], values[1:]))]
        peak, peak_rate = max(map(abs, values)), max(rates, default=0.0)
        if mode in ('torque', 'velocity') and values[-1] != 0:
            issues.append(f'{col} must end at 0 (ends at {values[-1]:g})')
        if limits:
            if mode == 'torque':
                if peak > limits['torque']:
                    issues.append(f'{col}: peak {peak:g} Nm exceeds limit {limits["torque"]:g}')
                if peak_rate > limits['rotatum']:
                    issues.append(f'{col}: rotatum {peak_rate:g} Nm/s exceeds limit {limits["rotatum"]:g}')
            elif mode == 'velocity':
                if peak > limits['velocity']:
                    issues.append(f'{col}: peak {peak:g} rad/s exceeds limit {limits["velocity"]:g}')
                if peak_rate > limits['acceleration']:
                    issues.append(f'{col}: accel {peak_rate:g} rad/s^2 exceeds limit {limits["acceleration"]:g}')
            elif mode == 'position' and peak_rate > limits['velocity']:
                issues.append(f'{col}: implied velocity {peak_rate:g} rad/s exceeds limit {limits["velocity"]:g}')
        if mode == 'position':
            # Corners must be velocity-continuous (within what the accel limit
            # can absorb in CORNER_DT); a raw corner is an accel impulse that
            # spikes torque through the position loop.
            signed = [(b - a) / (tb - ta) if tb > ta else 0.0
                      for (ta, a), (tb, b) in zip(zip(times, values),
                                                  zip(times[1:], values[1:]))]
            if signed:
                if abs(signed[0]) > 1e-6:
                    issues.append(f'{col} starts moving at t=0; add settle '
                                  'time so the trace starts at rest')
                if abs(signed[-1]) > 1e-6:
                    issues.append(f'{col} ends while still moving')
            if limits:
                step_limit = limits['acceleration'] * CORNER_DT
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
            limit = limits.get(chan['control_mode'])
            if limit is not None and peak > limit:
                issues.append(f"{motor}_motor_{chan['control_mode']}: peak "
                              f'{peak:g} exceeds limit {limit:g}')
    if settings['duration_per_point_s'] <= 0:
        issues.append('Hold per gridpoint must be > 0')
    if settings['settle_time_s'] < 0:
        issues.append('Settle time must be >= 0')
    if not 0 < settings['transition_rate'] <= 1:
        issues.append('Transition rate must be in (0, 1]')
    if settings['continuous_torque'] <= 0:
        issues.append('Continuous torque rating must be > 0')
    return issues


def validate_recipe(recipe, limits=None):
    issues = []
    if not recipe.get('segments'):
        issues.append('Recipe has no segments')
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
