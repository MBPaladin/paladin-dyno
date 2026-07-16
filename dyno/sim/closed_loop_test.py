"""Closed-loop validation of the sim sandbox: run a real test definition
through the real Controller + TestManager against the fake bus and plant,
with a real Logger process attached, then check that the physics moved and
the log recorded it.

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/closed_loop_test.py [mode] [test.yaml]
Defaults: actuator_production, sim_demo.yaml
"""
import glob
import json
import multiprocessing
import os
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else 'actuator_production'
TEST = sys.argv[2] if len(sys.argv) > 2 else 'sim_demo.yaml'
os.environ['DYNO_SIM'] = MODE

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from deployment import dyno_paths  # noqa: E402
from dyno.src.dyno_controller import Controller  # noqa: E402
from dyno.src.logger import Logger  # noqa: E402
from dyno.src.config_utils import augment_log_keys  # noqa: E402


def main():
    print(f'--- closed-loop sim test: mode={MODE} test={TEST} ---')
    with open(f'{dyno_paths.dyno_config_directory}/{MODE}_dyno_config.yaml') as f:
        dyno_params = yaml.safe_load(f)
    log_keys = augment_log_keys(dyno_params)
    idx = {k: i for i, k in enumerate(log_keys)}

    telemetry_q = multiprocessing.Queue()
    command_q = multiprocessing.Queue()
    logging_q = multiprocessing.Queue()
    ctrl = multiprocessing.Process(target=Controller,
                                   args=[telemetry_q, command_q, MODE])
    logger = multiprocessing.Process(target=Logger, args=[logging_q, MODE])
    ctrl.start()
    logger.start()

    samples = []

    def pump(seconds):
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                s = telemetry_q.get(timeout=1.0)
                samples.append(s)
                logging_q.put_nowait(s)
            except Exception:
                if not ctrl.is_alive():
                    print('FAIL: controller died')
                    sys.exit(1)

    pump(4)  # bring-up + ADC settling
    command_q.put_nowait(['test_def', (TEST, MODE)])
    pump(3)  # TestManager load
    command_q.put_nowait(['start_test'])
    n_before_test = len(samples)
    pump(14)  # test runs (~10 s of grid points) plus margin
    command_q.put_nowait(['stop_test'])
    pump(2)
    command_q.put_nowait(['shutdown', 0])
    ctrl.join(timeout=10)
    if ctrl.is_alive():
        ctrl.terminate()
    time.sleep(1)  # let logger drain and close the file
    logger.terminate()
    logger.join()

    arr = np.array([s[:-2] for s in samples[n_before_test:]], dtype=float)
    ok = True

    def series(*candidates):
        for key in candidates:
            if key in idx:
                return arr[:, idx[key]]
        return None

    dut_v = series('dut_velocity', 'input_velocity')
    load_t_cmd = series('load_torque_command', 'output_torque_command')
    load_torque = series('load_torque', 'output_torque')
    print(f'\ntest-phase samples: {len(arr)}')
    print(f'dut_velocity:  max={np.nanmax(np.abs(dut_v)):.2f} rad/s (expect ~2-5)')
    print(f'load_torque_command: max|.|={np.nanmax(np.abs(load_t_cmd)):.1f} Nm (expect 5)')
    if load_torque is not None:
        print(f'load_torque sensor: max|.|={np.nanmax(np.abs(load_torque)):.1f} Nm')

    if np.nanmax(np.abs(dut_v)) < 1.0:
        print('FAIL: DUT never moved — plant/velocity loop not working')
        ok = False
    if np.nanmax(np.abs(dut_v)) > 20.0:
        print('FAIL: DUT velocity implausibly large — sim unstable?')
        ok = False
    if load_torque is not None and np.nanmax(np.abs(load_torque)) < 1.0:
        print('WARN: load torque sensor saw nothing — check plant sensor mapping')

    # Newest log file should carry the resolved-config attribute and data
    logs = sorted(glob.glob(f'{dyno_paths.dyno_logs_directory}/*/log.hdf5'))
    if logs:
        import h5py
        with h5py.File(logs[-1], 'r') as f:
            has_attr = 'resolved_config' in f.attrs
            n_rows = f['time'].shape[0] if 'time' in f else 0
            print(f'log {logs[-1]}: rows={n_rows} resolved_config attr={has_attr}')
            if has_attr:
                cfg = json.loads(f.attrs['resolved_config'])
                print('  resolved devices:', sorted(cfg['devices'].keys()))
            else:
                print('FAIL: resolved_config attribute missing from log')
                ok = False
            if n_rows == 0:
                print('FAIL: no data rows written to log')
                ok = False
    else:
        print('FAIL: no log file produced')
        ok = False

    print('\nCLOSED-LOOP TEST', 'PASSED' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
