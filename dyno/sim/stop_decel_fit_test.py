"""The stop-decel experiment loop: Throw behavior -> log -> fit, no bus.

Runs the real Throw behavior against a stand-in plant with a KNOWN decel at a
ladder of speeds for each drive motor, writes the channels the fit reads into a
scratch hdf5, and checks fit_stop_decel recovers that decel.

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/stop_decel_fit_test.py
"""
import os
import sys
import tempfile

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from throw_sim_plant import Plant, RATIO, make  # noqa: E402
from dyno.src.archimedes import fit_stop_decel  # noqa: E402

TRUE_DECEL = 62.0
fails = []


def check(ok, msg):
    print(('  ok   ' if ok else '  FAIL ') + msg)
    if not ok:
        fails.append(msg)


def log_for(drive, speeds, hold=0.0):
    """One scratch log: a throw series per speed, back to back on one plant."""
    plant = Plant(drive=drive, true_decel=TRUE_DECEL, dt=0.001)
    cols = {k: [] for k in ('time', 'load_position', 'load_velocity',
                            'dut_velocity_command', 'load_velocity_command',
                            'dut_torque_command', 'load_torque_command')}
    t = 0.0
    for v in speeds:
        k = RATIO if drive == 'input' else 1.0
        beh = make(plant, drive_motor=drive, speed_rad_s=v * k, ramp_accel=60.0 * k,
                   target_rad=1.0, n_throws=4, stop_decel_rad_s2=40.0,
                   hold_level=hold, hold_follows_dir=True)
        for c in beh.commands():
            plant.step(c)
            t += 0.001
            cols['time'].append(t)
            cols['load_position'].append(plant.pos)
            cols['load_velocity'].append(plant.v)
            vin = c['input_command'] if drive == 'input' else float('nan')
            vout = c['output_command'] if drive == 'output' else float('nan')
            cols['dut_velocity_command'].append(vin)
            cols['load_velocity_command'].append(vout)
            cols['dut_torque_command'].append(c['input_command'] if drive == 'output'
                                              else float('nan'))
            cols['load_torque_command'].append(c['output_command'] if drive == 'input'
                                               else float('nan'))
    path = os.path.join(tempfile.mkdtemp(), f'{drive}.hdf5')
    with h5py.File(path, 'w') as f:
        for k, v in cols.items():
            f[k] = np.array(v, dtype='float32')
    return path


for drive in ('output', 'input'):
    print(f'--- {drive} drive ---')
    path = log_for(drive, (1.0, 2.0, 3.0, 5.0), hold=0.0)
    rows = fit_stop_decel.stops_in(path)
    check(len(rows) >= 14, f'found the stops ({len(rows)} for 4 speeds x 4 throws)')
    r = fit_stop_decel.fit(rows)
    check(abs(r['a'] - TRUE_DECEL) / TRUE_DECEL < 0.05,
          f"fitted a {r['a']:.1f} recovers the plant's {TRUE_DECEL:g}")
    check(r['a_safe'] <= TRUE_DECEL * 1.02 and r['a_safe'] > TRUE_DECEL * 0.85,
          f"a_safe {r['a_safe']:.1f} is at or just under the true decel")
    check({round(x['v0'], 0) for x in rows} >= {1.0, 2.0, 3.0, 5.0},
          'stops found at every speed of the ladder')

print('\n' + ('ALL OK' if not fails else f'{len(fails)} FAILED'))
sys.exit(1 if fails else 0)
