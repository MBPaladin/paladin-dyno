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
}


def default_segment(seg_id='SEG1'):
    return {
        'id': seg_id,
        'repeats': 1,
        'primary': {'motor': 'input', 'control_mode': 'position'},
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

    cols = ['time',
            f"{primary['motor']}_motor_{primary['control_mode']}",
            f"{sec_motor}_motor_{secondary['control_mode']}"]
    return cols, rows


# --- validation -------------------------------------------------------------

_FORBIDDEN_MODE_PAIRS = ({'position', 'position'}, {'velocity', 'velocity'},
                         {'position', 'velocity'})


def validate_segment(segment, limits=None):
    """Return a list of human-readable issues (empty = clean). Mirrors the
    TestTrace asserts so problems surface while editing, not at load time."""
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
