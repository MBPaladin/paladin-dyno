"""Synthetic gearbox backlash log with a known planted gap.

The point of this file is to answer a question the real log cannot: when the
`backlash` processor reports 11 arc-min, how much of that is the geometric tooth
gap and how much is the estimator's own bias? Here the gap is an input, so the
answer is a subtraction.

The plant is a dead-zone spring with the two effects that are visible in
2026-08-18_161709 and that break the textbook parallelogram loop:

    deflection = sign(Te) * gap/2  +  flank(Te),   Te = T_in - sign(dT/dt) * T_fric

  * `gap` is the geometric backlash, traversed instantly at Te = 0.
  * `T_fric` displaces the loop horizontally by +-T_fric, which is what opens
    the loop into a lens instead of a closed line. Friction is why the two
    branches do not meet, and it biases *both* estimators -- neither of them
    can see through it from a quasi-static sweep.
  * `flank` stiffens with load: d(deflection)/dT falls as |T| rises, so a
    straight line fitted far out on the flank and extrapolated back to zero
    lands above the true gap. This is what makes 'edges' depend on `edge_frac`,
    and it is the specific failure the processor's band-sensitivity check
    exists to catch.

Run it, then analyze the result, and compare the reported numbers against the
GROUND TRUTH block this script prints:

    .venv/bin/python dyno/sim/synthetic_backlash_log.py
    ./dyno/utilities/analyze.sh dyno/logs/sim_backlash --only backlash

Tune GAP_ARCMIN / STIFFNESS / ALPHA / T_FRIC to bracket a real gearbox and see
how each estimator's bias moves. That is a cheaper experiment than another
2 hours on the rig.
"""
import json
import os
import sys

import h5py
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else 'dyno/logs/sim_backlash'
DT = 0.001
RAD_TO_ARCMIN = 60.0 * 180.0 / np.pi

# -- ground truth -------------------------------------------------------------
RATIO = 26.0            # input:output reduction
GAP_ARCMIN = 8.0        # geometric backlash, output frame
STIFFNESS = 9000.0      # small-signal output torsional stiffness [Nm/rad]
ALPHA = 0.35            # flank stiffening per Nm of input torque
T_FRIC = 0.15           # friction torque that opens the loop [Nm]
COGGING = 0.06          # fractional gap variation with mesh position

# -- schedule (mirrors backlash_inputDriven_1p5Nm_upAndDownPosition) ----------
PEAK_NM = 1.5
RAMP_S = 3.0            # per quarter-sawtooth
REST_S = 2.65
POSITIONS = list(np.arange(0.0, 6.3, 0.63))          # rad, output frame
POSITIONS = POSITIONS + POSITIONS[-2::-1]            # up then back down

rng = np.random.default_rng(11)


def flank(torque):
    """Elastic deflection [arc-min] under an input torque, stiffening with load."""
    compliance = RATIO / STIFFNESS * RAD_TO_ARCMIN   # arc-min per Nm at zero load
    return compliance * torque / (1.0 + ALPHA * np.abs(torque))


def deflection(torque, gap):
    """Dead-zone spring with friction. `torque` is the commanded input torque."""
    rate = np.gradient(torque) if torque.size > 1 else np.zeros_like(torque)
    effective = torque - np.sign(rate) * T_FRIC
    return np.sign(effective) * (gap / 2.0) + flank(effective)


time, tcmd, in_pos, out_pos, out_cmd, in_tq, out_tq = ([] for _ in range(7))
clock = 0.0


def emit(n, torque_cmd, position, gap):
    """Append n samples holding `position` while the input applies torque_cmd."""
    global clock
    time.append(clock + np.arange(n) * DT)
    clock = time[-1][-1] + DT

    defl_arcmin = deflection(torque_cmd, gap)
    held = position + rng.normal(0, 2e-6, n)          # servo holds, with noise
    # theta_in = ratio * (theta_out + deflection); the encoder reads motor frame.
    motor = RATIO * (held + defl_arcmin / RAD_TO_ARCMIN)

    tcmd.append(torque_cmd)
    out_pos.append(held)
    out_cmd.append(np.full(n, position))
    in_pos.append(motor + rng.normal(0, 1e-5, n))
    in_tq.append(torque_cmd + rng.normal(0, 0.006, n))
    out_tq.append(RATIO * torque_cmd + rng.normal(0, 0.6, n))


n_ramp = int(RAMP_S / DT)
sawtooth = np.concatenate([
    np.linspace(0, PEAK_NM, n_ramp),
    np.linspace(PEAK_NM, -PEAK_NM, 2 * n_ramp),
    np.linspace(-PEAK_NM, 0, n_ramp),
])

for i, park in enumerate(POSITIONS):
    # Mesh-position dependence: the gap varies smoothly around the output turn.
    gap = GAP_ARCMIN * (1 + COGGING * np.sin(2 * np.pi * park / 6.28))
    n_rest = int(REST_S / DT)
    emit(n_rest, np.zeros(n_rest), park, gap)         # rest / slew window
    emit(len(sawtooth), sawtooth.copy(), park, gap)

emit(int(REST_S / DT), np.zeros(int(REST_S / DT)), POSITIONS[-1], GAP_ARCMIN)

t = np.concatenate(time)
N = t.size
cat = lambda parts: np.concatenate(parts).astype(np.float32)
nan = np.full(N, np.nan, dtype=np.float32)

config = {
    'mode': 'inhouse',
    'ports': [
        {'role': 'input', 'prefix': 'dut', 'device': 'DUT',
         'attached': 'synthetic gearbox', 'cell': 'input_torque'},
        {'role': 'output', 'prefix': 'load', 'device': 'LOAD',
         'attached': 'none', 'cell': 'load_torque'},
    ],
    'devices': {
        'DUT': {'class': 'AKD', 'params': {
            'flip_torque_sign': False, 'gear_ratio': RATIO,
            'motor_params': {'kt': 0.983, 'k_tanh': 0.0001},
            'motor_limits': {'torque': 12, 'continuous_torque': 7,
                             'velocity': 600, 'acceleration': 50, 'rotatum': 500},
            'drive_params': {'i_cont': 7}, 'type': 'rotary'}},
        'LOAD': {'class': 'AKD', 'params': {
            'flip_torque_sign': True, 'gear_ratio': 1,
            'motor_params': {'kt': 7.79, 'k_tanh': 0.0164},
            'motor_limits': {'torque': 220, 'continuous_torque': 100,
                             'velocity': 50, 'acceleration': 20, 'rotatum': 1000},
            'drive_params': {'i_cont': 15}, 'type': 'rotary'}},
    },
}

os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, 'synthetic_backlash.hdf5')
with h5py.File(path, 'w') as f:
    f.create_dataset('time', data=t.astype(np.float32))
    f.create_dataset('dut_torque_command', data=cat(tcmd))
    f.create_dataset('dut_output_position', data=cat(in_pos))
    f.create_dataset('input_torque', data=cat(in_tq))
    f.create_dataset('load_position', data=cat(out_pos))
    f.create_dataset('load_position_command', data=cat(out_cmd))
    f.create_dataset('load_torque', data=cat(out_tq))
    f.create_dataset('dut_current', data=(cat(tcmd) / 0.983).astype(np.float32))
    f.create_dataset('dut_velocity', data=np.zeros(N, dtype=np.float32))
    f.create_dataset('load_velocity', data=np.zeros(N, dtype=np.float32))
    f.create_dataset('load_current', data=np.zeros(N, dtype=np.float32))
    for k in ('dut_position_command', 'dut_velocity_command',
              'load_torque_command', 'load_velocity_command', 'load_stator_temp'):
        f.create_dataset(k, data=nan)
    f.create_dataset('behavior_ids',
                     data=np.array([b'SEG1-RUN0'], dtype=h5py.string_dtype()))
    f.create_dataset('behavior_indices', data=np.array([[0, N - 1]], dtype=np.int32))
    f.attrs['resolved_config'] = json.dumps(config)

# What the two estimators should read off this plant, derived from the model
# rather than measured -- so a disagreement with the processor is a bug in the
# processor, not a property of the data.
zero_width_truth = GAP_ARCMIN + 2 * flank(np.array(T_FRIC))

# 'edges' fits a straight line to the flank over |T| > edge_frac * peak, so the
# compliance it reports is a chord across that band, not the small-signal slope
# at zero load. On a stiffening flank those differ a lot, and the processor is
# right to report the chord -- it is what was measured. Deriving it here keeps
# the comparison honest instead of flagging correct behaviour as a failure.
EDGE_FRAC = 0.25
band = np.linspace(EDGE_FRAC * PEAK_NM, PEAK_NM, 2000)
chord_compliance = float(np.polyfit(band, flank(band), 1)[0])   # arc-min per Nm
chord_stiffness = RATIO / (chord_compliance / RAD_TO_ARCMIN)

print(f'Wrote {path}: {N} samples, {t[-1]:.0f}s, {len(POSITIONS)} mesh positions')
print()
print('GROUND TRUTH')
print(f'  gear ratio                    {RATIO:g}:1')
print(f'  geometric backlash            {GAP_ARCMIN:.2f} arc-min '
      f'(+-{100 * COGGING:.0f}% with mesh position)')
print(f'  output torsional stiffness    {STIFFNESS:.0f} Nm/rad (small-signal)')
print(f'  flank stiffening              {ALPHA:.2f} /Nm')
print(f'  friction torque               {T_FRIC:.2f} Nm')
print()
print('EXPECTED ESTIMATOR BIAS')
print(f"  'zero_width' should read      {float(zero_width_truth):.2f} arc-min "
      f'= gap + 2*flank(T_fric);')
print(f'                                friction alone inflates it by '
      f'{float(zero_width_truth) - GAP_ARCMIN:.2f} arc-min.')
print(f"  'edges' should read more still -- it also extrapolates the stiffening")
print(f'                                flank back to zero, and by construction')
print(f'                                depends on edge_frac.')
print(f'  reported stiffness            {chord_stiffness:.0f} Nm/rad, NOT the '
      f'{STIFFNESS:.0f} small-signal')
print(f'                                value: the flank fit measures a chord '
      f'over |T| > {EDGE_FRAC:g}*peak.')
