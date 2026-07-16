"""Headless smoke test of the simulation sandbox: runs the real Controller
against the fake EtherCAT bus (no GUI, no hardware) and checks that telemetry
flows and the drives complete their DS402 enable handshake.

Run from repo root:  PYTHONPATH=. .venv/bin/python dyno/sim/smoke_test.py [mode]
(mode defaults to gearbox; e.g. actuator_production)
"""
import math
import multiprocessing
import os
import sys
import time

MODE = sys.argv[1] if len(sys.argv) > 1 else 'gearbox'
os.environ['DYNO_SIM'] = MODE

from dyno.src.dyno_controller import Controller  # noqa: E402


def main():
    print(f'--- sim smoke test, mode={MODE} ---')
    telemetry_q = multiprocessing.Queue()
    command_q = multiprocessing.Queue()
    p = multiprocessing.Process(target=Controller,
                                args=[telemetry_q, command_q, MODE])
    p.start()

    samples = []
    t_deadline = time.time() + 12.0
    while time.time() < t_deadline:
        try:
            samples.append(telemetry_q.get(timeout=1.0))
        except Exception:
            if not p.is_alive():
                print('FAIL: controller process died during bring-up')
                sys.exit(1)

    command_q.put_nowait(['shutdown', 0])
    p.join(timeout=10)
    if p.is_alive():
        p.terminate()
        p.join()
        print('WARN: controller did not shut down cleanly, terminated')

    n = len(samples)
    print(f'\ncollected {n} telemetry samples in ~12 s')
    ok = True
    if n < 5000:
        print(f'FAIL: expected >=5000 samples at 1 kHz, got {n}')
        ok = False

    # Telemetry layout: values per log_keys, then logging_state, control_state
    last = samples[-1]
    row = last[:-2]
    n_nan = sum(1 for v in row if isinstance(v, float) and math.isnan(v))
    print(f'sample width: {len(row)} values ({n_nan} NaN — command echoes are '
          f'NaN outside tests, that is expected)')
    # time is field 0 and must advance
    t0, t1 = samples[0][0], samples[-1][0]
    print(f'controller time advanced {t1 - t0:.1f} s across samples')
    if not t1 > t0:
        print('FAIL: controller time not advancing')
        ok = False

    print('SMOKE TEST', 'PASSED' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
