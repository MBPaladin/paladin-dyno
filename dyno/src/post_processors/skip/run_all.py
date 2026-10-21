"""Run every Skip analysis on the cogging log, loading it once.

    "C:/Users/Nathan Justus/anaconda3/python.exe" dyno/src/post_processors/skip/run_all.py

Figures and CSVs land in dyno/logs/skip/results/<NN_topic>/, headline numbers
in dyno/logs/skip/results/skip_key_numbers.json. Takes about a minute and
~3 GB of RAM. Existing files are overwritten in place; nothing is deleted.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import skip_common as c  # noqa: E402
import s01_overview, s02_torque_current, s03_stick_slip, s04_folds, s05_spectra  # noqa: E402,E401
import s06_decomposition, s07_windup, s08_load_cell, s09_time_response  # noqa: E402,E401

STAGES = (s01_overview, s02_torque_current, s03_stick_slip, s04_folds, s05_spectra,
          s06_decomposition, s07_windup, s08_load_cell, s09_time_response)


def main():
    t0 = time.time()
    d = c.load()
    for mod in STAGES:
        t = time.time()
        print(f'[{mod.__name__}]')
        mod.run(d)
        print(f'  ({time.time() - t:.1f} s)')
    print(f'done in {time.time() - t0:.0f} s -> {c.RESULTS_DIR}')


if __name__ == '__main__':
    main()
