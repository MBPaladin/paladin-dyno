"""Synthetic multi-park-angle characterization log.

Mimics the experiment we designed for but have not yet run: DUT torque sawtooth,
LOAD holding position at four park angles, with slews in between. Exercises the
level detection, park-angle grouping, and the no-sign-flip branch -- none of
which the real blocked-rotor log touches.

Ground truth baked in:
    kt         = 1.08 Nm/A  (configured 1.0 -> +8%, must trip the kt warning)
    cogging    = +-1% of kt, varying by park angle
    hysteresis = 0.10 Nm loop width
    flip       = False, so current and torque correlate positively (no flip)
"""
import json
import os
import sys

import h5py
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else 'dyno/logs/sim_multilevel'
DT = 0.001
KT_TRUE = 1.08
COGGING = [+0.01, -0.008, +0.004, -0.006]     # fractional kt error per level
HYST = 0.05                                    # +- this, so 0.10 Nm loop
LEVELS = [0.0, 0.10, 0.20, 0.30]               # rad
PEAK_NM = 8.0
RAMP_S = 20.0                                  # per quarter-sawtooth
SLEW_S = 2.0

rng = np.random.default_rng(7)
t, dut_tcmd, load_pcmd, load_pos, load_vel, dut_vel, torque, current = ([] for _ in range(8))
clock = 0.0


def emit(n, tcmd, pcmd, pos, vel, kt):
    """Append n samples; torque follows current through kt with hysteresis."""
    global clock
    tt = clock + np.arange(n) * DT
    clock = tt[-1] + DT
    cur = np.asarray(tcmd) / kt
    d = np.gradient(np.asarray(tcmd)) if len(tcmd) > 1 else np.zeros(n)
    hyst = np.where(d > 0, HYST, np.where(d < 0, -HYST, 0.0))
    tq = kt * cur + hyst + rng.normal(0, 0.004, n)
    t.append(tt); dut_tcmd.append(np.asarray(tcmd)); load_pcmd.append(np.full(n, pcmd))
    load_pos.append(pos); load_vel.append(vel); dut_vel.append(vel)
    torque.append(tq); current.append(cur + rng.normal(0, 0.0008, n))


for i, park in enumerate(LEVELS):
    kt = KT_TRUE * (1 + COGGING[i])

    if i > 0:                                   # slew to the new park angle
        n = int(SLEW_S / DT)
        prev = LEVELS[i - 1]
        pos = np.linspace(prev, park, n)
        vel = np.gradient(pos) / DT             # ~0.05 rad/s, must be excluded
        emit(n, np.zeros(n), park, pos, vel, kt)

    n_ramp = int(RAMP_S / DT)
    saw = np.concatenate([
        np.linspace(0, PEAK_NM, n_ramp),
        np.linspace(PEAK_NM, -PEAK_NM, 2 * n_ramp),
        np.linspace(-PEAK_NM, 0, n_ramp),
    ])
    n = len(saw)
    # Windup under load: the holding servo sags slightly with applied torque.
    pos = park + saw * 2e-5
    emit(n, saw, park, pos, np.gradient(pos) / DT, kt)

time = np.concatenate(t)
N = len(time)
cat = lambda x: np.concatenate(x).astype(np.float32)
nan = np.full(N, np.nan, dtype=np.float32)

config = {
    'mode': 'inhouse',
    'devices': {
        'DUT': {'class': 'AKD', 'params': {
            'flip_torque_sign': False, 'gear_ratio': 1,
            'motor_params': {'kt': 1.0, 'k_tanh': 0.01},
            'motor_limits': {'torque': 100, 'continuous_torque': 10,
                             'velocity': 10, 'acceleration': 5, 'rotatum': 100},
            'drive_params': {'i_cont': 1}, 'type': 'rotary'}},
        'LOAD': {'class': 'AKD', 'params': {
            'flip_torque_sign': True, 'gear_ratio': 1,
            'motor_params': {'kt': 7.5, 'k_tanh': 0.01},
            'motor_limits': {'torque': 100, 'continuous_torque': 4,
                             'velocity': 50, 'acceleration': 20, 'rotatum': 400},
            'drive_params': {'i_cont': 15}, 'type': 'rotary'}},
    },
}
tare = {'input_torque': {'bias': 0.02, 'stddev': 0.0087, 'full_scale': 20.0,
                         'samples': 3001},
        'load_torque': {'bias': -1.01, 'stddev': 0.136, 'full_scale': 500.0,
                        'samples': 3001}}

os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, 'dut_kt_multilevel.hdf5')
with h5py.File(path, 'w') as f:
    f.create_dataset('time', data=time.astype(np.float32))
    f.create_dataset('dut_current', data=cat(current))
    f.create_dataset('dut_torque_command', data=cat(dut_tcmd))
    f.create_dataset('dut_velocity', data=cat(dut_vel))
    f.create_dataset('dut_output_position', data=cat(load_pos))
    f.create_dataset('input_torque', data=cat(torque))
    f.create_dataset('load_position_command', data=cat(load_pcmd))
    f.create_dataset('load_position', data=cat(load_pos))
    f.create_dataset('load_velocity', data=cat(load_vel))
    f.create_dataset('load_current', data=np.zeros(N, dtype=np.float32))
    f.create_dataset('load_torque', data=(cat(torque) * 0 + rng.normal(0, 0.14, N)).astype(np.float32))
    for k in ('dut_position_command', 'dut_velocity_command',
              'load_torque_command', 'load_velocity_command', 'load_stator_temp'):
        f.create_dataset(k, data=nan)
    f.create_dataset('behavior_ids', data=np.array([b'SEG1-RUN0'],
                                                   dtype=h5py.string_dtype()))
    f.create_dataset('behavior_indices', data=np.array([[0, N - 1]], dtype=np.int32))
    f.attrs['resolved_config'] = json.dumps(config)
    f.attrs['tare'] = json.dumps(tare)

print(f'Wrote {path}: {N} samples, {time[-1]:.0f}s, {len(LEVELS)} park angles')
print(f'Ground truth: kt={KT_TRUE} (+{100*(KT_TRUE/1.0-1):.0f}% vs config), '
      f'cogging spread={KT_TRUE*(max(COGGING)-min(COGGING)):.4f} Nm/A, '
      f'hysteresis={2*HYST:.2f} Nm')
