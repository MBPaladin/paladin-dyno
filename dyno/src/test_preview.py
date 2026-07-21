"""
Offline expansion of a test plan into a command-vs-time timeline, for preview.

This runs the *real* behavior generators from `test_manager` (single source of
truth for what a test does) but drives them with a virtual clock instead of the
wall clock. Because the generators only read the clock to decide when to advance
-- and the live controller pulls `next_command()` exactly once per control cycle
-- advancing the virtual clock by one cycle per consumed command reproduces the
rig's command stream exactly, while running as fast as the CPU allows. A test
that would take 20 minutes on the dyno expands here in a couple of seconds.

Nothing in `test_manager` is modified: the clock swap is scoped to this module's
expansion call, and the live controller runs in a separate process with its own
module copy, so a running test is never affected.
"""
import contextlib

import numpy as np
import yaml

from deployment import dyno_paths
from dyno.src import test_manager

MODES = ('torque', 'velocity', 'position')

# ~33 minutes at 1 kHz. Guards against an accidentally huge loop_count turning a
# preview into a multi-GB allocation. Expansions that hit this are flagged
# 'truncated' so the UI can say so.
MAX_CYCLES = 2_000_000


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
    one control cycle -- identical to the rig, where the controller pulls
    `next_command()` once per cycle. A tiny per-call epsilon keeps repeated reads
    within a single step strictly increasing; this mimics a real monotonic clock
    and, in particular, avoids a 0/0 in `GridSearch.ramp` when a setpoint doesn't
    change (transition_time == 0) and the real code would skip the ramp loop.
    """

    def __init__(self, dt):
        self.t = 0.0
        self.dt = dt
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


def expand_test(test_file, mode, limits=None, max_cycles=MAX_CYCLES):
    """Expand ``test_file`` into a command timeline.

    Returns a dict with:
      - ``t``: time in seconds (np.ndarray)
      - ``input_<mode>`` / ``output_<mode>`` for mode in torque/velocity/position:
        the commanded value where that motor is in that control mode, NaN
        elsewhere (so each curve draws only where it is active).
      - ``n_cycles``: total command count
      - ``truncated``: True if expansion hit ``max_cycles``

    Loading/validation is delegated to ``TestManager``; if the plan violates a
    safety limit, the underlying ``AssertionError`` propagates -- that is the
    same check that runs on the rig, so surfacing it here is the point.
    """
    if limits is None:
        limits = limits_from_config(mode)

    clock = _VirtualClock(_cycle_time_s())

    t_list, in_cmd, out_cmd, in_mode, out_mode = [], [], [], [], []

    with contextlib.ExitStack() as stack:
        # Swap the wall clock for the virtual one, only inside test_manager.
        orig_time = test_manager.time
        test_manager.time = clock
        stack.callback(setattr, test_manager, 'time', orig_time)

        tm = test_manager.TestManager(test_file, mode, limits)
        tm.reset()

        truncated = False
        while True:
            cmd = tm.next_command()
            if cmd is None:
                break
            t_list.append(clock.t)
            in_cmd.append(cmd['input_command'])
            out_cmd.append(cmd['output_command'])
            in_mode.append(cmd['input_mode'])
            out_mode.append(cmd['output_mode'])
            clock.advance()
            if len(t_list) >= max_cycles:
                truncated = True
                break

    t = np.asarray(t_list, dtype=float)
    result = {'t': t, 'n_cycles': len(t_list), 'truncated': truncated}

    for side, cmds, modes in (('input', in_cmd, in_mode),
                              ('output', out_cmd, out_mode)):
        cmds = np.asarray(cmds, dtype=float)
        modes = np.asarray(modes, dtype=object)
        for m in MODES:
            sig = np.full(t.shape, np.nan)
            if len(modes):
                mask = modes == m
                sig[mask] = cmds[mask]
            result[f'{side}_{m}'] = sig

    return result


if __name__ == '__main__':
    import sys
    import time as _time

    fname = sys.argv[1] if len(sys.argv) > 1 else 'sim_demo.yaml'
    rig = sys.argv[2] if len(sys.argv) > 2 else 'gearbox'

    t0 = _time.perf_counter()
    res = expand_test(fname, rig)
    elapsed = _time.perf_counter() - t0

    dur = res['t'][-1] if res['n_cycles'] else 0.0
    print(f"{fname} [{rig}]: {res['n_cycles']:,} cycles, "
          f"{dur:.2f}s test duration, expanded in {elapsed:.3f}s"
          f"{' (truncated)' if res['truncated'] else ''}")
    for side in ('input', 'output'):
        for m in MODES:
            sig = res[f'{side}_{m}']
            active = np.count_nonzero(~np.isnan(sig))
            if active:
                print(f"  {side}_{m}: {active:,} active samples, "
                      f"range [{np.nanmin(sig):.3f}, {np.nanmax(sig):.3f}]")
