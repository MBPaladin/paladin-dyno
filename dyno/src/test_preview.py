"""
Offline expansion of a test plan into a command-vs-time timeline, for preview.

Fidelity model
--------------
The command stream a test produces is a pure function of virtual time -- the
behavior generators in `test_manager` read the clock only to decide when to
advance, and the live controller pulls `next_command()` exactly once per control
cycle. So advancing a virtual clock one cycle per consumed command reproduces the
rig's command stream exactly, while running as fast as the CPU allows.

Two speedups on top of that:

  * Per-behavior caching. `behavior_iterator` (the same iterator the runtime
    uses) flattens loops by re-yielding a behavior definition once per iteration;
    we expand each unique behavior body once and reuse it, so a `loop_count: N`
    costs one body expansion, not N.

  * Vectorized traces. A `test_trace` body is a single `np.interp` over the whole
    time axis instead of one scalar interp per cycle. `_expand_trace_body`
    reproduces `TestTrace.commands()` output exactly; `_expand_reference` +
    `cross_check` verify that equivalence so the fast path can't silently drift.

Nothing in `test_manager` is modified: the clock swap is scoped to the expansion
call, and the live controller runs in a separate process with its own module
copy, so a running test is never affected.
"""
import contextlib

import numpy as np
import yaml

from deployment import dyno_paths
from dyno.src import test_manager

MODES = ('torque', 'velocity', 'position')

# ~33 minutes at 1 kHz. Guards against an accidentally huge loop_count turning a
# preview into a multi-GB allocation. Expansions that hit this are 'truncated'.
MAX_CYCLES = 2_000_000

# Must match the constant hold in TestTrace.commands() (the `for i in range(250)`
# tail). cross_check() catches drift if that changes.
_TRACE_HOLD_CYCLES = 250


def limits_from_config(mode):
    """Recompute the controller's safety-limit dict from a rig config file.

    Mirrors `dyno_controller._get_limits` so preview can validate and apply
    relative scaling without hardware attached. For devices that read their true
    limits from the drive over SDO at runtime, this falls back to the config's
    `motor_limits` clip values -- a safe proxy for a command-shape preview.
    """
    with open(f"{dyno_paths.dyno_config_directory}/{mode}_dyno_config.yaml") as f:
        cfg = yaml.safe_load(f)

    motor_limits = {}
    for slave in cfg.get('expected_slave_layout', []):
        if slave.get('name') in ('DUT', 'LOAD'):
            motor_limits[slave['name']] = slave.get('params', {}).get('motor_limits', {})

    dut = motor_limits.get('DUT', {})
    load = motor_limits.get('LOAD', {})
    return {
        'torque': min(abs(dut['torque']), abs(load['torque'])),
        'velocity': min(abs(dut['velocity']), abs(load['velocity'])),
        'acceleration': abs(load['acceleration']),
        'rotatum': min(abs(load['rotatum']), abs(dut['torque']) * 4),
    }


def _cycle_time_s():
    with open(f"{dyno_paths.dyno_config_directory}/master_config.yaml") as f:
        return yaml.safe_load(f)['cycle_time_us'] / 1e6


class _VirtualClock:
    """Drop-in replacement for the `time` module inside `test_manager`.

    `perf_counter()` returns a value that only advances when the engine consumes
    a command (`advance()`), so every command a generator yields maps to exactly
    one control cycle. A tiny per-call epsilon keeps repeated reads within a
    single step strictly increasing; this mimics a real monotonic clock and, in
    particular, avoids a 0/0 in `GridSearch.ramp` when a setpoint doesn't change
    (transition_time == 0) and the real code would skip the ramp loop.
    """

    def __init__(self, dt):
        self.dt = dt
        self.t = 0.0
        self._eps = 0.0

    def perf_counter(self):
        self._eps += 1e-9
        return self.t + self._eps

    def time(self):
        return self.t + self._eps

    def sleep(self, _seconds):  # never real-sleep during expansion
        pass

    def advance(self):
        self.t += self.dt
        self._eps = 0.0

    def reset(self):
        self.t = 0.0
        self._eps = 0.0


# --- per-behavior body expansion -------------------------------------------
# A "body" is a 4-tuple of equal-length arrays:
#   (input_command, output_command, input_mode, output_mode)

def _run_generator_body(instance, clock):
    """Faithful path: run the behavior's real generator under the virtual clock,
    one command per cycle. Used for grid searches (already cheap) and anything
    that isn't a plain trace."""
    clock.reset()
    in_cmd, out_cmd, in_mode, out_mode = [], [], [], []
    for cmd in instance.commands():
        in_cmd.append(cmd['input_command'])
        out_cmd.append(cmd['output_command'])
        in_mode.append(cmd['input_mode'])
        out_mode.append(cmd['output_mode'])
        clock.advance()
    return (np.asarray(in_cmd, dtype=float), np.asarray(out_cmd, dtype=float),
            np.asarray(in_mode, dtype=object), np.asarray(out_mode, dtype=object))


def _expand_trace_body(instance, dt):
    """Vectorized equivalent of `TestTrace.commands()`.

    Reproduces, in order: the initial zero/mode-set command, the interpolated
    body sampled at the cycle rate, and the constant hold tail -- matching the
    generator sample-for-sample (verified by cross_check)."""
    im, om = instance.input_mode, instance.output_mode
    t_arr, in_arr, out_arr = instance._t_arr, instance._in_arr, instance._out_arr

    # Body samples: the generator yields while (perf_counter - start) < max_time,
    # interpolating at ~i*dt for i = 0, 1, 2, .... Its float clock lands the last
    # sample a hair under max_time, so it emits floor(max_time/dt)+1 samples
    # (the trailing one at ~max_time). We match that count exactly; the single
    # boundary sample's value may differ from the rig by << 1 cycle, which is why
    # cross_check tolerates one sample at the tail. See module docstring.
    n_body = int(np.floor(instance._trace_max_time / dt + 1e-9)) + 1
    args = np.arange(n_body) * dt
    body_in = np.interp(args, t_arr, in_arr)
    body_out = np.interp(args, t_arr, out_arr)

    in_cmd = np.concatenate(([0.0], body_in,
                             np.full(_TRACE_HOLD_CYCLES, instance._in_last)))
    out_cmd = np.concatenate(([0.0], body_out,
                              np.full(_TRACE_HOLD_CYCLES, instance._out_last)))
    n = len(in_cmd)
    return (in_cmd, out_cmd,
            np.full(n, im, dtype=object), np.full(n, om, dtype=object))


def _expand_behavior_body(instance, clock, dt):
    if isinstance(instance, test_manager.TestTrace):
        return _expand_trace_body(instance, dt)
    return _run_generator_body(instance, clock)


# --- top-level expansion ----------------------------------------------------

def _route_signals(in_cmd, out_cmd, in_mode, out_mode, t):
    """Split the flat command stream into per-mode signals, NaN where a motor is
    not in that control mode (so each curve draws only where it is active)."""
    result = {'t': t, 'n_cycles': len(t)}
    for side, cmds, modes in (('input', in_cmd, in_mode),
                              ('output', out_cmd, out_mode)):
        for m in MODES:
            sig = np.full(t.shape, np.nan)
            if len(modes):
                sig[modes == m] = cmds[modes == m]
            result[f'{side}_{m}'] = sig
    return result


def expand_test(test_file, mode, limits=None, max_cycles=MAX_CYCLES):
    """Expand ``test_file`` into a command timeline.

    Returns a dict with ``t`` (seconds), ``input_<mode>`` / ``output_<mode>``
    signal arrays (NaN where inactive), ``n_cycles`` and ``truncated``.

    Loading/validation is delegated to ``TestManager``; if the plan violates a
    safety limit the underlying ``AssertionError`` propagates -- that is the same
    check that runs on the rig, so surfacing it here is the point.
    """
    if limits is None:
        limits = limits_from_config(mode)
    dt = _cycle_time_s()
    clock = _VirtualClock(dt)

    with contextlib.ExitStack() as stack:
        # Swap the wall clock for the virtual one, only inside test_manager.
        orig_time = test_manager.time
        test_manager.time = clock
        stack.callback(setattr, test_manager, 'time', orig_time)

        tm = test_manager.TestManager(test_file, mode, limits)

        body_cache = {}
        seg_in, seg_out, seg_im, seg_om = [], [], [], []
        total = 0
        truncated = False
        for bdef in test_manager.behavior_iterator(tm.test_config):
            bid = bdef['id']
            if bid not in body_cache:
                body_cache[bid] = _expand_behavior_body(tm.behaviors[bid], clock, dt)
            body = body_cache[bid]
            n = len(body[0])
            if total + n > max_cycles:  # clip the final segment to the cap
                n = max_cycles - total
                body = tuple(arr[:n] for arr in body)
                truncated = True
            seg_in.append(body[0])
            seg_out.append(body[1])
            seg_im.append(body[2])
            seg_om.append(body[3])
            total += n
            if truncated:
                break

    if total == 0:
        empty = np.array([], dtype=float)
        result = _route_signals(empty, empty, np.array([], dtype=object),
                                np.array([], dtype=object), empty)
        result['truncated'] = truncated
        return result

    in_cmd = np.concatenate(seg_in)
    out_cmd = np.concatenate(seg_out)
    in_mode = np.concatenate(seg_im)
    out_mode = np.concatenate(seg_om)
    t = np.arange(total) * dt

    result = _route_signals(in_cmd, out_cmd, in_mode, out_mode, t)
    result['truncated'] = truncated
    return result


# --- equivalence check (keeps the vectorized path honest) -------------------

def _bodies_match(vec, gen, atol=1e-6):
    """Compare a vectorized body against the generator body. Tolerates a single
    trailing sample and float-noise below `atol` (the generator interpolates at a
    clock value carrying ~1e-8 of accumulated float error; a real command differs
    by ~0.1-100), which is enough to catch structural drift while ignoring that
    inherent ambiguity."""
    nv, ng = len(vec[0]), len(gen[0])
    if abs(nv - ng) > 1:
        return False, f"length {nv} vs {ng}"
    m = min(nv, ng) - 1  # ignore the boundary sample
    if m <= 0:
        return True, 'ok'
    for k in (0, 1):  # input/output command values
        if not np.allclose(vec[k][:m], gen[k][:m], rtol=0, atol=atol):
            return False, 'values'
    for k in (2, 3):  # input/output modes
        if not (vec[k][:m] == gen[k][:m]).all():
            return False, 'modes'
    return True, 'ok'


def cross_check(test_file, mode, limits=None):
    """Verify the vectorized trace path matches the real generator, per behavior.

    Done per behavior rather than on the full timeline because a 1-sample tail
    difference in one trace would otherwise shift every later behavior and break
    a pointwise comparison. Raises AssertionError on divergence; returns the
    number of trace behaviors checked."""
    if limits is None:
        limits = limits_from_config(mode)
    dt = _cycle_time_s()
    clock = _VirtualClock(dt)
    checked = 0
    with contextlib.ExitStack() as stack:
        orig_time = test_manager.time
        test_manager.time = clock
        stack.callback(setattr, test_manager, 'time', orig_time)
        tm = test_manager.TestManager(test_file, mode, limits)
        for bid, inst in tm.behaviors.items():
            if not isinstance(inst, test_manager.TestTrace):
                continue  # non-trace bodies use the generator directly, nothing to compare
            vec = _expand_trace_body(inst, dt)
            gen = _run_generator_body(inst, clock)
            ok, why = _bodies_match(vec, gen)
            assert ok, f"behavior {bid}: {why}"
            checked += 1
    return checked


if __name__ == '__main__':
    import sys
    import time as _time

    if len(sys.argv) > 1 and sys.argv[1] == '--cross-check':
        # Cross-check every clean-loading test against the reference generator.
        import io
        import os
        files = sorted(f for f in os.listdir(dyno_paths.dyno_test_directory)
                       if f.endswith('.yaml'))
        for f in files:
            passed = None
            for rig in ('gearbox', 'actuator', 'actuator_production'):
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        n = cross_check(f, rig)
                    passed = (rig, n)
                    break
                except AssertionError as e:
                    if str(e).startswith('behavior '):  # genuine vectorization drift
                        passed = ('MISMATCH', str(e))
                        break
                    continue  # validation assert -> test not valid for this rig
                except Exception:
                    continue
            if passed and passed[0] == 'MISMATCH':
                print(f"MISMATCH  {f:45s} {passed[1]}")
            elif passed:
                print(f"ok        {f:45s} {passed[0]:18s} {passed[1]} trace behavior(s)")
            else:
                print(f"skip      {f:45s} (no rig loads it cleanly)")
        sys.exit(0)

    fname = sys.argv[1] if len(sys.argv) > 1 else 'sim_demo.yaml'
    rig = sys.argv[2] if len(sys.argv) > 2 else 'gearbox'
    t0 = _time.perf_counter()
    res = expand_test(fname, rig)
    elapsed = _time.perf_counter() - t0
    dur = res['t'][-1] if res['n_cycles'] else 0.0
    print(f"{fname} [{rig}]: {res['n_cycles']:,} cycles, {dur:.2f}s duration, "
          f"expanded in {elapsed:.3f}s{' (truncated)' if res['truncated'] else ''}")
