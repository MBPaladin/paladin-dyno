"""Run every Skip analysis on one cogging log, loading it once.

    .venv/bin/python dyno/src/post_processors/skip/run_all.py        # v1, 2026-10-19
    .venv/bin/python dyno/src/post_processors/skip/run_all.py v2     # v2, 2026-10-21

Runs are listed in skip_common.RUNS. Figures and CSVs land in that run's
results folder (<results>/<NN_topic>/), headline numbers in its
skip_key_numbers.json. v2 takes a few minutes and ~5 GB of RAM. Existing
files are overwritten in place; nothing is deleted.

The angle-domain stages (folds, spectra, decomposition, windup/TE) only see
the quasi-static speeds (skip_common.COGGING_MAX_SPEED); the rest see the
whole grid.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import skip_common as c  # noqa: E402
import s01_overview, s02_torque_current, s03_stick_slip, s04_folds, s05_spectra  # noqa: E402,E401
import s06_decomposition, s07_windup, s08_load_cell, s09_time_response  # noqa: E402,E401
import s10_speed_and_feedforward, s11_efficiency  # noqa: E402,E401

STAGES = ((s01_overview, False), (s02_torque_current, False), (s03_stick_slip, False),
          (s04_folds, True), (s05_spectra, True), (s06_decomposition, True), (s07_windup, True),
          (s08_load_cell, False), (s09_time_response, False),
          (s10_speed_and_feedforward, False), (s11_efficiency, False))


def main():
    c.use_run_from_argv()
    t0 = time.time()
    d = c.load()
    dq = c.cogging_view(d)
    print(f'run {c.RUN}: {len(d["dwells"])} dwells, {len(dq["dwells"])} at <= {c.COGGING_MAX_SPEED} rad/s')
    for mod, quasi_static in STAGES:
        t = time.time()
        print(f'[{mod.__name__}]')
        mod.run(dq if quasi_static else d)
        print(f'  ({time.time() - t:.1f} s)')
    print(f'done in {time.time() - t0:.0f} s -> {c.RESULTS_DIR}')


if __name__ == '__main__':
    main()
