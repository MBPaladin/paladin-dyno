"""Combined post-processing of Brian's gearbox dyno test logs.

Opens each relevant .hdf5 file in dyno/logs/brian_tests/gearbox_tests/, runs the
analysis selected by filename, and writes report materials (figures, CSVs,
summary text/JSON) into a per-test subfolder of
dyno/logs/brian_tests/gearbox_tests/results/. The results folder is wiped at the
start of every run so stale outputs never accumulate. (Motor logs live in a
sibling dyno/logs/brian_tests/motor_tests/ folder, handled by
mac_motor_post_processing.py.)

Run with the anaconda base interpreter:
    "C:/Users/Nathan Justus/anaconda3/python.exe" dyno/src/mac_gearbox_post_processing.py
"""

import os
import glob
import json
import shutil
from collections import defaultdict

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless: save figures, never block on display
import matplotlib.pyplot as plt

# --- Paths ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GEARBOX_TESTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'logs', 'brian_tests', 'gearbox_tests'))
RESULTS_DIR = os.path.join(GEARBOX_TESTS_DIR, 'results')

# --- Tunables ---------------------------------------------------------------
GEAR_RATIO = 26                  # input:output reduction ratio
BACKLASH_FILE_STEM = 'backlash2'  # which gearbox backlash file to run
TARE_TORQUE_CMD_THRESH = 0.005   # |dut_torque_command| below this = "no torque" (tare)
TARE_MIN_DUR_S = 0.5             # min duration of a sustained zero-torque window
RAD_TO_ARCMIN = 60.0 * 180.0 / np.pi  # 3437.75 arc-min per radian
BACKLASH_FIT_TORQUE_FRAC = 0.25  # fit flanks above this fraction of peak |input torque|
BACKLASH_DROP_FIRST_SAMPLE = True  # exclude the first sweep (first-cycle settling artifact)
DUPLICATE_STEP_FRAC = 0.5        # consecutive samples closer than this*median step = a double
# NOTE: gearbox_backlash2 also has output-torque-cell spikes (and real dynamic
# torque transients when the input breaks loose). These live in load_torque,
# which the deflection-vs-input-torque backlash analysis does not use, so they
# are intentionally left unfiltered for now.

EFFICIENCY_FILE_GLOB = 'efficiency*.hdf5'  # files containing the efficiency sweep
EFFICIENCY_SETTLE_FRAC = 0.30    # trim this leading fraction of each setpoint dwell
EFFICIENCY_MIN_DWELL = 200       # min samples for a setpoint to count
EFFICIENCY_LIGHT_LOAD_NM = 10.0  # 'loaded' map drops torque columns with |setpoint| below this
EFFICIENCY_MIN_OUTPUT_TORQUE_NM = 5.0  # forward/backdrive maps: drop dwells with |measured output torque| below this (low-torque efficiency is noise)

# --- Loaded torque ripple (reuses the efficiency dwells) --------------------
# Each efficiency dwell holds a constant input speed and output torque for ~1 s.
# The AC RMS of the (linearly-detrended) output torque over a dwell is the torque
# ripple at that operating point. Standstill dwells (zero input speed -> no
# rotation) give the load-side noise floor; rotating dwells add the gearbox's
# transmission ripple on top. Quantified only on dwells transmitting a real load.
RIPPLE_MIN_OUTPUT_TORQUE_NM = 40.0  # only quantify ripple on dwells above this transmitted torque

RUNNING_FILE_STEM = 'running_input'   # no-load input running-torque file
RUNNING_VEL_BIN_RAD_S_INPUT = 10.0      # velocity bin width for the drag/ripple curves
RUNNING_VEL_BIN_RAD_S_OUTPUT = 1.0
RUNNING_FIT_MIN_VEL = 10.0        # ignore |v| below this for the Coulomb+viscous fit
RUNNING_DETREND_WIN = 101        # moving-average window (samples) to detrend torque -> ripple
RUNNING_STFT_WIN = 512           # STFT window (samples) for the Campbell spectrogram
RUNNING_SPEED_BIN_RAD_S = 4.0    # |speed| bin width for the Campbell spectrogram
RUNNING_VCMD_BIN_RAD_S = 0.025     # commanded-velocity bin width for the vs-command spectrogram
RUNNING_FFT_MAX_HZ = 250.0       # max frequency shown on the running-torque FFT spectrum & spectrogram (tunable)

TAP_FILE_STEM = 'taptest'        # impulse (rubber-mallet) ring-down test
TAP_SIGNAL = 'input_torque'      # signal to FFT; torque rings much cleaner than velocity here
TAP_FFT_MAX_HZ = 500.0           # upper frequency limit for the plotted/CSV spectrum (~Nyquist at 1 kHz sampling)
TAP_N_PEAKS = 5                  # number of resonance peaks to label and report
TAP_PEAK_MIN_HZ = 3.0            # ignore peaks below this (drift / DC skirt)
TAP_PEAK_MIN_SEP_HZ = 3.0        # minimum spacing between reported peaks
TAP_STFT_WIN = 1024              # STFT window (samples) for the ring-down spectrogram

# --- Torque-ramp stiction (breakaway) ---------------------------------------
# These tests slowly ramp the DUT torque command (a triangular sawtooth) until
# the shaft breaks free; the breakaway torque is the static friction. Three
# files share one analyzer, distinguished by 'role':
#   motor  - free-hanging DUT motor, gearbox removed  -> motor's own stiction
#   input  - DUT drives the gearbox INPUT shaft        -> motor + gearbox-input
#   output - gearbox flipped, DUT drives the OUTPUT    -> motor + gearbox-output
# (role, find_file needles, human label)
TSTIC_FILES = [
    ('motor',  'unattached_motor_stiction', 'Free motor (no gearbox)'),
    ('input',  'torque_stiction_input',     'Gearbox input-driven'),
    ('output', 'output_driven_stiction',    'Gearbox output-driven'),
]
TSTIC_KT_NM_PER_A = 3.03         # motor torque constant (Nm/Arms): dut_current -> motor torque
TSTIC_VEL_THRESH_RAD_S = 0.5     # |dut_velocity| above this = the shaft has broken free
TSTIC_STUCK_S = 0.25             # require the shaft to have been stuck this long before a breakaway
TSTIC_PRE_S = 0.02               # window just before motion onset to read breakaway torque/current
TSTIC_FFT_MAX_HZ = 200.0         # upper frequency for the stiction torque FFT/spectrogram
TSTIC_STFT_WIN = 512             # STFT window (samples) for the stiction spectrogram

# --- Output-driven running torque -------------------------------------------
# Gearbox flipped so the DUT drives the OUTPUT shaft directly; dut_velocity is
# therefore already the output velocity (no /ratio) and input_torque (inline
# sensor) is the output-referred drag torque.
OUT_RUNNING_FILE_STEM = 'output_driven_running_torque_10rps'
OUT_RUNNING_FIT_MIN_VEL = 1.0    # ignore |output vel| below this for the Coulomb+viscous fit


def _linfit(x, y):
    """Linear fit y = slope*x + intercept; returns (slope, intercept, r2)."""
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(slope), float(intercept), r2


def reset_results_dir():
    """Delete and recreate the results folder so each run starts clean."""
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR)
    print(f'Results dir reset: {RESULTS_DIR}')


def test_dir(name):
    """Create and return a per-test subfolder inside the results directory."""
    d = os.path.join(RESULTS_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def behavior_span(f):
    """Return (start, end) index of the first logged behavior in the file."""
    start, end = f['behavior_indices'][0]
    return int(start), int(end)


def write_summary(name, summary_dict, out_dir):
    """Write a per-test summary as both human-readable .txt and .json."""
    with open(os.path.join(out_dir, f'{name}_summary.json'), 'w') as jf:
        json.dump(summary_dict, jf, indent=2, default=str)
    with open(os.path.join(out_dir, f'{name}_summary.txt'), 'w') as tf:
        tf.write(f'{name} summary\n')
        tf.write('=' * (len(name) + 8) + '\n\n')
        for k, v in summary_dict.items():
            if isinstance(v, float):
                tf.write(f'{k:32s}: {v:.6g}\n')
            else:
                tf.write(f'{k:32s}: {v}\n')
    print(f'  Saved: {name}_summary.json / .txt')
    return summary_dict


def detect_tare_windows(tcmd, t):
    """Return [(start, end), ...] for sustained zero-torque-command windows."""
    zero = np.abs(tcmd) < TARE_TORQUE_CMD_THRESH
    chg = np.where(np.diff(zero.astype(int)) != 0)[0] + 1
    bounds = np.concatenate(([0], chg, [len(zero)]))
    wins = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        a, b = int(a), int(b)
        if zero[a] and (t[b - 1] - t[a]) >= TARE_MIN_DUR_S:
            wins.append((a, b))
    return wins


# ---------------------------------------------------------------------------
# BACKLASH / TORSIONAL DEFLECTION  (gearbox_backlash1.hdf5 -> BACKLASH-RUN1)
# ---------------------------------------------------------------------------
# The output is held at a fixed position while a torque sawtooth is applied at
# the input; the output is then shifted slightly and the sweep repeats, sampling
# backlash at many mesh positions. Each sample begins with a stationary, zero-
# torque window used to tare the input and output torque sensors.
#
# We refer the input angle to the output side (divide by GEAR_RATIO) and take
# the deflection = output_angle - input_angle/ratio, signed so positive input
# torque gives positive deflection, tared per-sample on its zero-torque window.
# Plotting deflection vs (tared) input torque yields a hysteresis loop. Backlash
# is quantified by fitting the two loaded flanks (|torque| above a threshold) and
# taking the deflection gap between them extrapolated to zero torque; the flank
# slope gives torsional compliance/stiffness. The first sweep is dropped as a
# first-cycle settling artifact.
# ---------------------------------------------------------------------------

def _fit_flanks(it, defl):
    """Fit positive/negative loaded flanks. Returns dict of fits + backlash."""
    thr = BACKLASH_FIT_TORQUE_FRAC * np.max(np.abs(it))
    pos, neg = it > thr, it < -thr
    sp, ip, r2p = _linfit(it[pos], defl[pos])
    sn, in_, r2n = _linfit(it[neg], defl[neg])
    backlash = ip - in_                       # deflection gap at zero torque (arc-min)
    compliance = 0.5 * (sp + sn)              # arc-min / Nm(input)
    # Output-side torsional stiffness: K = ratio / (compliance in rad/Nm_in)
    stiffness = (GEAR_RATIO / (compliance / RAD_TO_ARCMIN)) if compliance != 0 else float('nan')
    return {'sp': sp, 'ip': ip, 'r2p': r2p, 'sn': sn, 'in': in_, 'r2n': r2n,
            'backlash_arcmin': backlash, 'compliance_arcmin_per_Nm': compliance,
            'stiffness_Nm_per_rad': stiffness, 'thr': thr}


def _plot_sample(out_dir, name, idx, out_pos, it, defl, fit):
    """Per-sample plot showing the loop and how backlash was quantified."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(it, defl, '-', color='0.6', linewidth=0.7, alpha=0.7, label='Loop (deflection vs torque)')
    thr = fit['thr']
    ax.scatter(it[it > thr], defl[it > thr], s=6, color='limegreen', alpha=0.5, label='+ flank (fit pts)')
    ax.scatter(it[it < -thr], defl[it < -thr], s=6, color='red', alpha=0.5, label='− flank (fit pts)')

    xr = np.array([0, np.max(it)])
    xl = np.array([np.min(it), 0])
    ax.plot(xr, fit['sp'] * xr + fit['ip'], color='green', linewidth=2)
    ax.plot(xl, fit['sn'] * xl + fit['in'], color='darkred', linewidth=2)

    # Backlash = gap between the two flank intercepts at zero torque.
    ax.plot([0, 0], [fit['in'], fit['ip']], 'k-', linewidth=1)
    ax.plot(0, fit['ip'], 'o', color='green', markeredgecolor='k')
    ax.plot(0, fit['in'], 'o', color='darkred', markeredgecolor='k')
    ax.annotate(f'backlash = {fit["backlash_arcmin"]:.2f} arc-min',
                (0, 0.5 * (fit['ip'] + fit['in'])),
                textcoords='offset points', xytext=(10, 0), fontsize=10, fontweight='bold')

    ax.set_xlabel('Input Torque (Nm, tared)')
    ax.set_ylabel('Output-referred Deflection (arc-min)')
    ax.set_title(f'{name} sample {idx} (output {out_pos:.3f} rad)\n'
                 f'stiffness = {fit["stiffness_Nm_per_rad"]:.0f} Nm/rad, '
                 f'R²(+)={fit["r2p"]:.3f} R²(−)={fit["r2n"]:.3f}')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=8)
    fig.tight_layout()
    per_dir = os.path.join(out_dir, 'per_sample')
    os.makedirs(per_dir, exist_ok=True)
    fig.savefig(os.path.join(per_dir, f'{name}_sample_{idx:02d}.png'), dpi=150)
    plt.close(fig)


def analyze_backlash(filepath):
    name = 'backlash'
    out_dir = test_dir(name)
    print(f'Processing GEARBOX BACKLASH: {os.path.basename(filepath)}')

    with h5py.File(filepath, 'r') as f:
        start, end = behavior_span(f)
        t = np.asarray(f['time'][start:end], dtype=float)
        tcmd = np.asarray(f['dut_torque_command'][start:end], dtype=float)
        input_torque = np.asarray(f['input_torque'][start:end], dtype=float)   # Nm (input sensor)
        output_torque = np.asarray(f['load_torque'][start:end], dtype=float)   # Nm (output sensor)
        input_pos = np.asarray(f['dut_output_position'][start:end], dtype=float)  # rad (input encoder)
        output_pos = np.asarray(f['load_position'][start:end], dtype=float)       # rad (output encoder)

    tares = detect_tare_windows(tcmd, t)
    if len(tares) < 2:
        raise RuntimeError(f'Expected multiple tare windows, found {len(tares)}')
    n_samples = len(tares) - 1

    # Deflection referred to output, signed so +input torque -> +deflection.
    deflection_rad = output_pos - input_pos / GEAR_RATIO

    # Torque sensor zero: the post-shift stationary windows carry real wound-up
    # holding torque (not sensor bias), so they are invalid zero references. Only
    # the initial rest (first window) is genuinely unloaded -> use it as the
    # sensor-zero tare. (The sensors turn out to be essentially unbiased.)
    it0 = float(np.mean(input_torque[tares[0][0]:tares[0][1]]))
    ot0 = float(np.mean(output_torque[tares[0][0]:tares[0][1]]))
    rest_it, rest_ot = [], []  # residual holding torque per used sample (diagnostic)

    cmap = plt.cm.viridis
    out_positions = [float(np.mean(output_pos[a:b])) for a, b in tares[:-1]]

    # Determine which samples to drop:
    #  - the first sweep (first-cycle settling artifact), and
    #  - the first of any accidental "double" (two consecutive sweeps abnormally
    #    close in output position; the tester doubled a run-through).
    drop = set()
    if BACKLASH_DROP_FIRST_SAMPLE:
        drop.add(0)
    steps = np.abs(np.diff(out_positions))
    med_step = np.median(steps)
    dup_firsts = [int(j) for j in np.where(steps < DUPLICATE_STEP_FRAC * med_step)[0]]
    drop.update(dup_firsts)  # j = first sample of the close pair (j, j+1)
    used = [i for i in range(n_samples) if i not in drop]
    print(f'  {len(tares)} tare windows -> {n_samples} samples; '
          f'dropped {sorted(drop)} (first-cycle + double); using {len(used)}')
    pos_min, pos_max = min(out_positions[i] for i in used), max(out_positions[i] for i in used)
    norm = plt.Normalize(pos_min, pos_max)

    fig, ax = plt.subplots(figsize=(12, 8))
    csv_rows = []
    metric_rows = []
    backlashes, stiffnesses = [], []
    for i in range(n_samples):
        sweep_a, sweep_b = tares[i][1], tares[i + 1][0]
        tare_a, tare_b = tares[i]
        it = input_torque[sweep_a:sweep_b] - it0
        ot = output_torque[sweep_a:sweep_b] - ot0
        defl_abs = deflection_rad[sweep_a:sweep_b] * RAD_TO_ARCMIN  # absolute (carries encoder offset)

        if i in drop:
            continue

        fit = _fit_flanks(it, defl_abs)
        # Center the loop on its elastic neutral axis (midpoint of the two flank
        # intercepts) -> removes the encoder-alignment offset; loop straddles 0.
        center = 0.5 * (fit['ip'] + fit['in'])
        defl = defl_abs - center
        fit_plot = dict(fit, ip=fit['ip'] - center, **{'in': fit['in'] - center})

        backlashes.append(fit['backlash_arcmin'])
        stiffnesses.append(fit['stiffness_Nm_per_rad'])
        rest_it.append(float(np.mean(input_torque[tare_a:tare_b])) - it0)
        rest_ot.append(float(np.mean(output_torque[tare_a:tare_b])) - ot0)

        color = cmap(norm(out_positions[i]))
        ax.plot(it, defl, '-', color=color, linewidth=0.8, alpha=0.8)
        _plot_sample(out_dir, name, i, out_positions[i], it, defl, fit_plot)

        metric_rows.append((i, out_positions[i], fit['backlash_arcmin'],
                            fit['compliance_arcmin_per_Nm'], fit['stiffness_Nm_per_rad'],
                            fit['r2p'], fit['r2n']))
        for k in range(len(it)):
            csv_rows.append((i, out_positions[i], it[k], ot[k], defl[k]))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label='Output hold position (rad)')
    ax.set_xlabel('Input Torque (Nm, tared)')
    ax.set_ylabel('Output-referred Deflection (arc-min)')
    ax.set_title(f'Gearbox Backlash: Deflection vs Input Torque '
                 f'({len(used)} mesh positions, ratio {GEAR_RATIO}:1)')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_deflection_vs_torque.png'), dpi=200)
    plt.close(fig)
    print(f'  Saved: {name}_deflection_vs_torque.png + {len(used)} per-sample plots')

    # CSVs: tidy plotted points, and per-sample metrics.
    with open(os.path.join(out_dir, f'{name}_deflection_vs_torque.csv'), 'w') as cf:
        cf.write('sample,output_pos_rad,input_torque_Nm,output_torque_Nm,deflection_arcmin\n')
        for s, op, it, ot, d in csv_rows:
            cf.write(f'{s},{op:.5f},{it:.5f},{ot:.5f},{d:.5f}\n')
    with open(os.path.join(out_dir, f'{name}_metrics.csv'), 'w') as cf:
        cf.write('sample,output_pos_rad,backlash_arcmin,compliance_arcmin_per_Nm,stiffness_Nm_per_rad,r2_pos,r2_neg\n')
        for row in metric_rows:
            cf.write(','.join(f'{v:.5f}' if isinstance(v, float) else str(v) for v in row) + '\n')
    print(f'  Saved: {name}_deflection_vs_torque.csv, {name}_metrics.csv')

    backlashes = np.array(backlashes)
    stiffnesses = np.array(stiffnesses)
    return write_summary(name, {
        'source_file': os.path.basename(filepath),
        'analysis': 'Gearbox backlash + torsional stiffness via loaded-flank fits; per-sample tared, first sweep + doubled sweep dropped',
        'gear_ratio': GEAR_RATIO,
        'n_samples_total': n_samples,
        'n_samples_used': len(used),
        'dropped_samples': sorted(drop),
        'duplicate_first_index': dup_firsts,
        'output_pos_min_rad': pos_min,
        'output_pos_max_rad': pos_max,
        'sensor_zero_input_Nm': it0,
        'sensor_zero_output_Nm': ot0,
        'rest_holding_torque_input_mean_Nm': float(np.mean(rest_it)),
        'rest_holding_torque_output_mean_Nm': float(np.mean(rest_ot)),
        'backlash_mean_arcmin': float(np.mean(backlashes)),
        'backlash_std_arcmin': float(np.std(backlashes, ddof=1)),
        'backlash_min_arcmin': float(np.min(backlashes)),
        'backlash_max_arcmin': float(np.max(backlashes)),
        'stiffness_mean_Nm_per_rad': float(np.mean(stiffnesses)),
        'stiffness_std_Nm_per_rad': float(np.std(stiffnesses, ddof=1)),
    }, out_dir)


# ---------------------------------------------------------------------------
# File routing (by filename) + driver
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# EFFICIENCY  (efficiency*.hdf5 -> EFFICIENCY-RUN0-SETPOINTn)
# ---------------------------------------------------------------------------
# DUT is velocity-controlled (input speed setpoint), load applies an output
# torque setpoint; input_torque is measured. Each behavior is a ~1 s dwell at
# one (input velocity, output torque) operating point. The sweep is split across
# several files (restarts from load-cell safety trips), with some overlap.
#
# Since the gearbox speed ratio is fixed (26:1, no slip), power efficiency equals
# torque efficiency: with T_out the measured output torque and T_in the measured
# input torque referred to the output as 26*T_in, the transmitted-to-supplied
# ratio is min(26|T_in|, |T_out|) / max(26|T_in|, |T_out|). The min/max form
# automatically yields <=1 and handles both forward driving (output torque
# opposes motion) and back-driving (output overhauls), which is set by the sign
# of input_velocity * output_torque.
# ---------------------------------------------------------------------------

def _eff_ratio(t_in, t_out):
    """Transmission efficiency from torques (kinematic ratio fixed): the driven
    shaft's torque is the one reduced by friction, so min/max gives <=1 and
    mode = backdrive iff |t_out| > ratio*|t_in| (output torque is the larger)."""
    a1, a2 = GEAR_RATIO * abs(t_in), abs(t_out)
    if max(a1, a2) == 0:
        return np.nan, 'fwd'
    return min(a1, a2) / max(a1, a2), ('back' if a2 > a1 else 'fwd')


def _efficiency_setpoints(filepath, input_torque_zero=0.0):
    """Extract per-setpoint operating points (mean- and median-based) from one file.
    input_torque_zero is subtracted from the input torque to remove the measured
    sensor zero offset (which otherwise shows up as a CW/CCW efficiency asymmetry)."""
    pts = []
    with h5py.File(filepath, 'r') as f:
        bidx = f['behavior_indices'][:]
        for i in range(len(bidx)):
            a, b = int(bidx[i, 0]), int(bidx[i, 1])
            if b - a < EFFICIENCY_MIN_DWELL:
                continue
            sl = slice(a + int((b - a) * EFFICIENCY_SETTLE_FRAC), b)
            vcmd = round(float(np.nanmean(f['dut_velocity_command'][sl])))
            tcmd = round(float(np.nanmean(f['load_torque_command'][sl])))
            itq = np.asarray(f['input_torque'][sl], dtype=float) - input_torque_zero
            ti_mean = float(np.nanmean(itq))
            ti_med = float(np.nanmedian(itq))
            to_mean = float(np.nanmean(f['load_torque'][sl]))
            to_med = float(np.nanmedian(f['load_torque'][sl]))
            eff_mean, mode = _eff_ratio(ti_mean, to_mean)
            eff_med, _ = _eff_ratio(ti_med, to_med)
            pts.append({'v_cmd': vcmd, 't_cmd': tcmd, 'eff_mean': eff_mean,
                        'eff_median': eff_med, 'mode': mode, 'ti': ti_mean, 'to': to_mean})
    return pts


def _implied_input_zero(pts):
    """Estimate the input-torque sensor zero from CW/CCW symmetry of the measured
    input torque: delta = median over matched (+v, -v) load pairs of
    (Ti(+v) + Ti(-v))/2 — which is zero only if the sensor is unbiased."""
    ti_cell = defaultdict(list)
    for p in pts:
        ti_cell[(p['v_cmd'], p['t_cmd'])].append(p['ti'])
    ti_cell = {k: float(np.mean(v)) for k, v in ti_cell.items()}
    deltas = []
    for av in sorted({abs(v) for v, t in ti_cell if v != 0}):
        for at in sorted({abs(t) for v, t in ti_cell if t != 0}):
            for sgn in (-1, 1):
                cw, ccw = ti_cell.get((av, sgn * at)), ti_cell.get((-av, -sgn * at))
                if cw is not None and ccw is not None:
                    deltas.append((cw + ccw) / 2.0)
    return float(np.median(deltas)) if deltas else 0.0


def _plot_eff_map(out_dir, suffix, vels, tqs, grid, title,
                  xlabel='Output Torque setpoint (Nm)', ylabel='Input Velocity setpoint (rad/s)'):
    """Render one efficiency map (velocity rows x torque cols)."""
    fig, ax = plt.subplots(figsize=(max(10, len(tqs) * 0.9), 8))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad('lightgray')  # cells with no data
    vmin = max(0.0, np.floor(np.nanmin(grid) * 100 / 5) * 5)
    im = ax.imshow(np.ma.masked_invalid(grid * 100), origin='lower', aspect='auto',
                   cmap=cmap, vmin=vmin, vmax=100)
    ax.set_xticks(range(len(tqs)))
    ax.set_xticklabels(tqs)
    ax.set_yticks(range(len(vels)))
    ax.set_yticklabels([f'{v/GEAR_RATIO:.2f}' for v in vels])  # output velocity = input / ratio
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for iv in range(len(vels)):
        for it in range(len(tqs)):
            if not np.isnan(grid[iv, it]):
                ax.text(it, iv, f'{grid[iv, it]*100:.0f}', ha='center', va='center',
                        fontsize=12, color='white')
    fig.colorbar(im, ax=ax, label='Efficiency (%)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'efficiency_map_{suffix}.png'), dpi=200)
    plt.close(fig)
    print(f'  Saved: efficiency_map_{suffix}.png')


def _plot_asymmetry(out_dir, eff_cell, tq_cell, vels_pos, tqs_pos, cap=20.0, highload_nm=50.0):
    """CCW-minus-CW efficiency difference (percentage points) per mode, as a
    diverging heatmap. Also returns the high-load asymmetry and the equivalent
    directional loss-torque offset (the load-invariant quantity).

    CW = +input velocity, CCW = -input velocity. For a given |v|,|T|:
      forward  : CW=(+v,-T), CCW=(-v,+T)   backdrive: CW=(+v,+T), CCW=(-v,-T)
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad('lightgray')
    stats = {}
    for ax, (mlabel, sgn) in zip(axes, [('forward', -1), ('backdrive', +1)]):
        diff = np.full((len(vels_pos), len(tqs_pos)), np.nan)
        dloss, hl = [], []
        for iv, av in enumerate(vels_pos):
            for it, at in enumerate(tqs_pos):
                cw = eff_cell.get((av, sgn * at))
                ccw = eff_cell.get((-av, -sgn * at))
                if cw is None or ccw is None or np.isnan(cw) or np.isnan(ccw):
                    continue
                diff[iv, it] = (ccw - cw) * 100.0
                if at >= highload_nm:
                    hl.append(diff[iv, it])
                    def loss(key):
                        ti, to = tq_cell[key]
                        return abs(abs(to) - GEAR_RATIO * abs(ti))  # friction-eaten torque (output Nm)
                    dloss.append(loss((av, sgn * at)) - loss((-av, -sgn * at)))
        im = ax.imshow(np.ma.masked_invalid(diff), origin='lower', aspect='auto',
                       cmap=cmap, vmin=-cap, vmax=cap)
        ax.set_xticks(range(len(tqs_pos)))
        ax.set_xticklabels(tqs_pos)
        ax.set_yticks(range(len(vels_pos)))
        ax.set_yticklabels(vels_pos)
        ax.set_xlabel('|Output Torque| (Nm)')
        ax.set_ylabel('|Input Velocity| (rad/s)')
        ax.set_title(f'{mlabel.upper()}: CCW − CW efficiency (pp)')
        for iv in range(len(vels_pos)):
            for it in range(len(tqs_pos)):
                if not np.isnan(diff[iv, it]):
                    ax.text(it, iv, f'{diff[iv, it]:+.0f}', ha='center', va='center', fontsize=6)
        fig.colorbar(im, ax=ax, label='CCW − CW (percentage points)')
        stats[f'asym_highload_{mlabel}_pp'] = float(np.mean(hl)) if hl else float('nan')
        stats[f'loss_torque_offset_{mlabel}_Nm'] = float(np.mean(dloss)) if dloss else float('nan')
    fig.suptitle('Direction asymmetry — positive = CCW more efficient (concentrated at light load)', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'efficiency_asymmetry_CW_vs_CCW.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: efficiency_asymmetry_CW_vs_CCW.png')
    return stats


def analyze_efficiency(filepaths, input_torque_zero=0.0, auto_zero=False):
    name = 'efficiency'
    out_dir = test_dir(name)

    # auto_zero: estimate the input-torque sensor zero from the data's own CW/CCW
    # symmetry (a raw first pass), then re-extract with that correction applied.
    if auto_zero:
        raw = []
        for fp in filepaths:
            raw.extend(_efficiency_setpoints(fp, 0.0))
        input_torque_zero = _implied_input_zero(raw)
    src = 'efficiency-implied' if auto_zero else 'fixed'
    print(f'Processing GEARBOX EFFICIENCY: {len(filepaths)} files '
          f'(input-torque zero correction: {input_torque_zero:+.4f} Nm [{src}])')

    # Pool all setpoints across files; dedupe overlapping (v_cmd, t_cmd) cells by
    # averaging (restarts produce slightly duplicated operating points).
    pts = []
    for fp in filepaths:
        pts.extend(_efficiency_setpoints(fp, input_torque_zero))
    cells = defaultdict(list)
    for p in pts:
        cells[(p['v_cmd'], p['t_cmd'])].append(p)

    vels = sorted({k[0] for k in cells})
    tqs = sorted({k[1] for k in cells})
    keep = [j for j, tq in enumerate(tqs) if abs(tq) >= EFFICIENCY_LIGHT_LOAD_NM]
    tqs_loaded = [tqs[j] for j in keep]

    # Build a grid for the chosen per-dwell statistic, plot full + loaded versions.
    def build_grid(eff_key):
        grid = np.full((len(vels), len(tqs)), np.nan)  # nan only where no data
        for (v, tq), plist in cells.items():
            grid[vels.index(v), tqs.index(tq)] = float(np.nanmean([p[eff_key] for p in plist]))
        return grid

    summary_stats = {}
    for statname, eff_key in (('mean', 'eff_mean'), ('median', 'eff_median')):
        grid = build_grid(eff_key)
        _plot_eff_map(out_dir, f'full_{statname}', vels, tqs, grid,
                      f'Gearbox Efficiency Map (26:1) — all setpoints [{statname}]')
        grid_loaded = grid[:, keep]
        _plot_eff_map(out_dir, f'loaded_{statname}', vels, tqs_loaded, grid_loaded,
                      f'Gearbox Efficiency Map (26:1) — light load (|T|<{EFFICIENCY_LIGHT_LOAD_NM:g} Nm) excluded [{statname}]')
        valid_loaded = grid_loaded[~np.isnan(grid_loaded)]
        summary_stats[f'peak_efficiency_loaded_{statname}'] = float(np.nanmax(valid_loaded))
        summary_stats[f'mean_efficiency_loaded_{statname}'] = float(np.nanmean(valid_loaded))

    # --- Split forward / backdrive maps -------------------------------------
    # Re-index by magnitude (|input velocity| x |output torque|), separating the
    # two power-flow modes. The two rotations (CW/CCW) of a given (|v|,|T|,mode)
    # are averaged together. Uses the mean per-dwell statistic.
    split_stats = {}
    for mlabel, mkey in (('forward', 'fwd'), ('backdrive', 'back')):
        mcells = defaultdict(list)
        for p in pts:
            if p['mode'] != mkey or p['v_cmd'] == 0 or p['t_cmd'] == 0:
                continue
            if abs(p['to']) < EFFICIENCY_MIN_OUTPUT_TORQUE_NM:
                continue  # measured output torque too small -> efficiency is noise
            mcells[(abs(p['v_cmd']), abs(p['t_cmd']))].append(p['eff_mean'])
        avs = sorted({k[0] for k in mcells})
        ats = sorted({k[1] for k in mcells})
        mgrid = np.full((len(avs), len(ats)), np.nan)
        for (av, at), vals in mcells.items():
            mgrid[avs.index(av), ats.index(at)] = float(np.nanmean(vals))
        _plot_eff_map(out_dir, mlabel, avs, ats, mgrid,
                      f'Gearbox Efficiency — {mlabel.upper()} driving (26:1, CW/CCW averaged)',
                      xlabel='|Output Torque| (Nm)', ylabel='|Output Velocity| (rad/s)')
        loaded = mgrid[:, [j for j, at in enumerate(ats) if at >= EFFICIENCY_LIGHT_LOAD_NM]]
        split_stats[f'peak_efficiency_{mlabel}'] = float(np.nanmax(loaded[~np.isnan(loaded)]))
        split_stats[f'mean_efficiency_{mlabel}_loaded'] = float(np.nanmean(loaded[~np.isnan(loaded)]))

    # --- Direction asymmetry (CCW - CW) figure + loss-torque offset ---------
    eff_cell, tq_cell = {}, {}
    for (v, tq), plist in cells.items():
        eff_cell[(v, tq)] = float(np.nanmean([p['eff_mean'] for p in plist]))
        tq_cell[(v, tq)] = (float(np.nanmean([p['ti'] for p in plist])),
                            float(np.nanmean([p['to'] for p in plist])))
    vels_pos = sorted({abs(v) for v, tq in cells if v != 0})
    tqs_pos = sorted({abs(tq) for v, tq in cells if tq != 0})
    asym_stats = _plot_asymmetry(out_dir, eff_cell, tq_cell, vels_pos, tqs_pos)

    # --- CSV: tidy long-format grid (both statistics) -----------------------
    with open(os.path.join(out_dir, f'{name}_grid.csv'), 'w') as cf:
        cf.write('input_velocity_rad_s,output_torque_Nm,efficiency_mean,efficiency_median,n_duplicates,mode\n')
        for (v, tq), plist in sorted(cells.items()):
            em = float(np.nanmean([p['eff_mean'] for p in plist]))
            ed = float(np.nanmean([p['eff_median'] for p in plist]))
            cf.write(f'{v},{tq},{em:.4f},{ed:.4f},{len(plist)},{plist[0]["mode"]}\n')
    print(f'  Saved: {name}_grid.csv')

    return write_summary(name, {
        'analysis': 'Gearbox efficiency map from pooled velocity x torque sweep; mean & median per-dwell',
        'gear_ratio': GEAR_RATIO,
        'n_files': len(filepaths),
        'n_setpoints_total': len(pts),
        'n_unique_cells': len(cells),
        'n_cells_with_duplicates': sum(1 for k in cells if len(cells[k]) > 1),
        'velocity_setpoints_rad_s': vels,
        'torque_setpoints_Nm': tqs,
        'light_load_excluded_Nm': EFFICIENCY_LIGHT_LOAD_NM,
        'split_map_min_output_torque_Nm': EFFICIENCY_MIN_OUTPUT_TORQUE_NM,
        'input_torque_zero_applied_Nm': input_torque_zero,
        **summary_stats,
        **split_stats,
        **asym_stats,
    }, out_dir)


# ---------------------------------------------------------------------------
# RUNNING TORQUE  (running_input_*.hdf5)
# ---------------------------------------------------------------------------
# Output shaft disconnected; the input is driven through a +/-velocity triangle.
# input_torque is then the no-load drag of the gearbox referred to the input.
# Binning by velocity averages the accel/decel passes (cancels inertia) and the
# angular ripple, leaving the drag curve (Coulomb + viscous). The per-bin spread
# is the torque ripple, which we characterize vs speed and in a Campbell
# spectrogram (frequency vs speed) since it is dominated by a fixed resonance
# excited as the mesh orders sweep through it.
# ---------------------------------------------------------------------------

def analyze_running_torque(filepath):
    name = 'running_torque'
    out_dir = test_dir(name)
    print(f'Processing GEARBOX RUNNING TORQUE: {os.path.basename(filepath)}')

    with h5py.File(filepath, 'r') as f:
        start, end = behavior_span(f)
        t = np.asarray(f['time'][start:end], dtype=float)
        v = np.asarray(f['dut_velocity'][start:end], dtype=float)       # input velocity (rad/s)
        tq = np.asarray(f['input_torque'][start:end], dtype=float)      # input torque (Nm)
    fs = 1.0 / np.median(np.diff(t))

    # --- Drag curve: mean torque per velocity bin (ripple + inertia averaged) ---
    bw = RUNNING_VEL_BIN_RAD_S_INPUT
    edges = np.arange(np.floor(v.min() / bw) * bw, np.ceil(v.max() / bw) * bw + bw, bw)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(v, edges) - 1, 0, len(centers) - 1)
    # Detrend torque (remove the slow drag/inertia trend) to isolate ripple.
    trend = np.convolve(tq, np.ones(RUNNING_DETREND_WIN) / RUNNING_DETREND_WIN, mode='same')
    ripple = tq - trend
    centers/= GEAR_RATIO  # convert to output velocity (rad/s)

    raw_v = v / GEAR_RATIO
    
    bmean = np.full(len(centers), np.nan)
    bripple = np.full(len(centers), np.nan)
    for b in range(len(centers)):
        sel = idx == b
        if np.count_nonzero(sel) >= 20:
            bmean[b] = np.mean(tq[sel])
            bripple[b] = np.std(ripple[sel])
    bmean += 0.014  # sensor bias correction (from the running-torque zero test)
    #Drop first and last bins (too few samples, too much ripple)
    bmean[0] = np.nan
    bmean[-1] = np.nan
    # bmean[-2] = np.nan
    centers[0] = np.nan
    centers[-1] = np.nan
    # centers[-2] = np.nan

    # Coulomb + viscous fit on the drag curve (per side, above the stiction floor)
    # centers is now output-referenced (see above), so the input-referenced
    # threshold must be converted to match.
    fmin = RUNNING_FIT_MIN_VEL / GEAR_RATIO
    fit = {}
    for tag, mask in (('pos', centers >= fmin), ('neg', centers <= -fmin)):
        m = mask & ~np.isnan(bmean)
        if np.count_nonzero(m) >= 2:
            slope, intercept, _ = _linfit(centers[m], bmean[m])
            fit[tag] = (slope, intercept)
    # avg_slope = 0.5 * (fit['pos'][0] - fit['neg'][0]) if 'pos' in fit and 'neg' in fit else float('nan')
    # avg_intercept = 0.5 * (abs(fit['pos'][1]) + abs(fit['neg'][1])) if 'pos' in fit and 'neg' in fit else float('nan')
    # #overwrite per side values with the average
    # if 'pos' in fit:
    #     fit['pos'] = (-avg_slope, -avg_intercept)
    # if 'neg' in fit:
    #     fit['neg'] = (-avg_slope, avg_intercept)
    coulomb = abs(0.5 * (fit['pos'][1] - fit['neg'][1])) if 'pos' in fit and 'neg' in fit else float('nan')
    viscous = 0.5 * (abs(fit['pos'][0]) + abs(fit['neg'][0])) if 'pos' in fit and 'neg' in fit else float('nan')
    vmax = float(np.max(np.abs(v)))  # input-referenced; kept for the Campbell spectrogram below
    vmax_out = vmax / GEAR_RATIO     # output-referenced, to match coulomb/viscous
    drag_at_vmax = coulomb + viscous * vmax_out

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(centers, bmean, 'b.-', markersize=5, label='Mean drag torque (bin)')
    #ax.plot(raw_v, tq, 'k.', markersize=1, alpha=0.2, label='Raw torque samples')
    for tag, c in (('pos', 'red'), ('neg', 'green')):
        if tag in fit:
            xs = np.linspace(0, np.nanmax(centers) if tag == 'pos' else np.nanmin(centers), 50)
            ax.plot(xs, fit[tag][0] * xs + fit[tag][1], '--', color=c, linewidth=1.5)
    ax.plot([], [], ' ', label=f'Coulomb={coulomb:.3f} Nm, viscous={viscous:.4f} Nm/(rad/s)')
    ax.set_xlabel('Output Velocity (rad/s)')
    ax.set_ylabel('Running (drag) Torque (Nm)')
    ax.set_title('No-load Input Running Torque')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_drag_vs_velocity.png'), dpi=200)
    plt.close(fig)
    print(f'  Saved: {name}_drag_vs_velocity.png')

    # --- Campbell spectrogram: torque spectrum vs |speed| -------------------
    w = RUNNING_STFT_WIN
    hop = w // 2
    win = np.hanning(w)
    fr = np.fft.rfftfreq(w, 1.0 / fs)
    sp_edges = np.arange(0, vmax + RUNNING_SPEED_BIN_RAD_S, RUNNING_SPEED_BIN_RAD_S)
    sp_centers = 0.5 * (sp_edges[:-1] + sp_edges[1:])
    accum = [np.zeros(len(fr)) for _ in sp_centers]
    count = np.zeros(len(sp_centers), dtype=int)
    seg_spectra, seg_times = [], []  # per-segment FFTs, reused below for the time spectrogram + averaged spectrum
    for i0 in range(0, len(tq) - w, hop):
        seg = tq[i0:i0 + w]
        amp = np.abs(np.fft.rfft((seg - seg.mean()) * win)) * 2 / np.sum(win)
        sbin = int(np.clip(np.digitize(np.mean(np.abs(v[i0:i0 + w])), sp_edges) - 1, 0, len(sp_centers) - 1))
        accum[sbin] += amp
        count[sbin] += 1
        seg_spectra.append(amp)
        seg_times.append(t[i0 + w // 2])
    camp = np.array([accum[k] / count[k] if count[k] else np.full(len(fr), np.nan)
                     for k in range(len(sp_centers))])
    seg_spectra = np.array(seg_spectra)  # [n_segments, n_freq]
    seg_times = np.array(seg_times)
    colsum = np.nansum(camp, axis=0)
    res_hz = float(fr[30:][np.argmax(colsum[30:])]) if len(fr) > 30 else float('nan')

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7))
    # left: ripple RMS vs speed
    axL.plot(centers, bripple, 'm.-')
    axL.set_xlabel('Output Velocity (rad/s)')
    axL.set_ylabel('Torque Ripple RMS (Nm)')
    axL.set_title('Ripple grows with speed')
    axL.grid(True, alpha=0.4)
    # right: Campbell
    sm = np.ma.masked_invalid(camp)
    pcm = axR.pcolormesh(np.append(fr, fr[-1] + (fr[1] - fr[0])), sp_edges,
                         20 * np.log10(sm / np.nanmax(sm) + 1e-9), cmap='magma', vmin=-40, vmax=0)
    axR.axvline(res_hz, color='cyan', linestyle=':', linewidth=1.3, label=f'resonance ~{res_hz:.0f} Hz')
    axR.set_xlabel('Frequency (Hz)')
    axR.set_ylabel('|Input Velocity| (rad/s)')
    axR.set_title('Campbell: torque spectrum vs speed')
    axR.legend(loc='upper right', facecolor='gray')
    fig.colorbar(pcm, ax=axR, label='Torque amplitude (dB re max)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_ripple.png'), dpi=200)
    plt.close(fig)
    print(f'  Saved: {name}_ripple.png  (resonance ~{res_hz:.0f} Hz)')

    # --- FFT spectrum + time spectrogram of input torque (noise/ripple content) ---
    fft_max_hz = min(RUNNING_FFT_MAX_HZ, fr.max())
    fkeep = fr <= fft_max_hz
    mean_spectrum = np.nanmean(seg_spectra, axis=0)  # Welch-style average across all segments

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(12, 10))
    axT.semilogy(fr[fkeep], mean_spectrum[fkeep] + 1e-12, color='tab:blue', linewidth=1.0)
    axT.set_xlim(0, fft_max_hz)
    axT.set_xlabel('Frequency (Hz)')
    axT.set_ylabel('Torque Amplitude (Nm)')
    axT.set_title(f'Running Torque Noise Spectrum (segment-averaged, 0-{fft_max_hz:g} Hz)')
    axT.grid(True, which='both', alpha=0.4)

    spec_db = 20 * np.log10(seg_spectra[:, fkeep] / np.nanmax(seg_spectra[:, fkeep]) + 1e-9)
    pcm = axB.pcolormesh(seg_times, fr[fkeep], spec_db.T, cmap='magma', vmin=-40, vmax=0, shading='nearest')
    axB.set_xlabel('Time (s)')
    axB.set_ylabel('Frequency (Hz)')
    axB.set_title('Running Torque Spectrogram')
    fig.colorbar(pcm, ax=axB, label='Torque amplitude (dB re max)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_fft.png'), dpi=200)
    plt.close(fig)
    print(f'  Saved: {name}_fft.png  (fmax={fft_max_hz:g} Hz)')

    # --- CSV: drag + ripple vs velocity -------------------------------------
    with open(os.path.join(out_dir, f'{name}_curve.csv'), 'w') as cf:
        cf.write('velocity_rad_s,mean_drag_torque_Nm,ripple_rms_Nm\n')
        for c, m, r in zip(centers, bmean, bripple):
            cf.write(f'{c:.3f},{"" if np.isnan(m) else f"{m:.5f}"},{"" if np.isnan(r) else f"{r:.5f}"}\n')
    print(f'  Saved: {name}_curve.csv')

    return write_summary(name, {
        'source_file': os.path.basename(filepath),
        'analysis': 'No-load input running torque: drag (Coulomb+viscous) + ripple vs speed + Campbell',
        'max_speed_rad_s': vmax_out,
        'coulomb_drag_Nm': coulomb,
        'viscous_drag_Nm_per_rad_s': viscous,
        'drag_at_max_speed_Nm': drag_at_vmax,
        'ripple_rms_at_max_speed_Nm': float(np.nanmax(bripple)),
        'ripple_resonance_hz': res_hz,
        'fft_max_hz': fft_max_hz,
    }, out_dir)


# ---------------------------------------------------------------------------
# TAP TEST  (tapTest.hdf5 -> TAP-TEST)
# ---------------------------------------------------------------------------
# The (stationary, unpowered) dynamometer is struck with a rubber mallet and the
# structure's free ring-down is recorded. An FFT of the response reveals the
# natural frequencies of the drivetrain/fixture. Torque rings far more cleanly
# than velocity here (SNR ~120 vs ~15 at the dominant mode), so we FFT torque by
# default (TAP_SIGNAL). We show the time trace, a full-record amplitude spectrum
# with the resonant peaks labeled, and a spectrogram of the ring-down decay.
# ---------------------------------------------------------------------------

def _spectral_peaks(fr, amp, n, min_hz, min_sep_hz):
    """Return indices of the n strongest local maxima above min_hz, enforcing a
    minimum frequency separation. Sorted by ascending frequency."""
    cand = [i for i in range(1, len(amp) - 1)
            if fr[i] >= min_hz and amp[i] > amp[i - 1] and amp[i] >= amp[i + 1]]
    cand.sort(key=lambda i: amp[i], reverse=True)
    chosen = []
    for i in cand:
        if all(abs(fr[i] - fr[j]) >= min_sep_hz for j in chosen):
            chosen.append(i)
        if len(chosen) >= n:
            break
    return sorted(chosen, key=lambda i: fr[i])


def analyze_tap_test(filepath):
    name = 'tap_test'
    out_dir = test_dir(name)
    print(f'Processing GEARBOX TAP TEST: {os.path.basename(filepath)}')

    with h5py.File(filepath, 'r') as f:
        start, end = behavior_span(f)
        t = np.asarray(f['time'][start:end], dtype=float)
        sig = np.asarray(f[TAP_SIGNAL][start:end], dtype=float)
    fs = 1.0 / np.median(np.diff(t))
    sig_d = sig - np.mean(sig)

    # --- Full-record amplitude spectrum (Hann window) -----------------------
    win = np.hanning(len(sig_d))
    amp = np.abs(np.fft.rfft(sig_d * win)) * 2 / np.sum(win)
    fr = np.fft.rfftfreq(len(sig_d), 1.0 / fs)
    band = fr <= TAP_FFT_MAX_HZ
    fr_b, amp_b = fr[band], amp[band]
    peaks = _spectral_peaks(fr_b, amp_b, TAP_N_PEAKS, TAP_PEAK_MIN_HZ, TAP_PEAK_MIN_SEP_HZ)
    peak_freqs = [float(fr_b[i]) for i in peaks]
    dominant_hz = peak_freqs[int(np.argmax([amp_b[i] for i in peaks]))] if peaks else float('nan')

    # --- Ring-down spectrogram ----------------------------------------------
    w = min(TAP_STFT_WIN, len(sig_d))
    hop = w // 2
    swin = np.hanning(w)
    sfr = np.fft.rfftfreq(w, 1.0 / fs)
    starts = range(0, len(sig_d) - w + 1, hop)
    spec = np.array([np.abs(np.fft.rfft((sig_d[i0:i0 + w] - sig_d[i0:i0 + w].mean()) * swin))
                     * 2 / np.sum(swin) for i0 in starts])
    spec_t = np.array([t[i0 + w // 2] - t[0] for i0 in starts])
    sband = sfr <= TAP_FFT_MAX_HZ

    fig, (axT, axS, axG) = plt.subplots(3, 1, figsize=(12, 13))
    label = TAP_SIGNAL.replace('_', ' ')
    axT.plot(t - t[0], sig_d, 'b-', linewidth=0.5)
    axT.set_xlabel('Time (s)')
    axT.set_ylabel(f'{label} (mean-removed)')
    axT.set_title(f'{os.path.basename(filepath)}: tap response ({label})')
    axT.grid(True, alpha=0.4)

    axS.plot(fr_b, amp_b, 'b-', linewidth=0.8)
    for i in peaks:
        axS.plot(fr_b[i], amp_b[i], 'ro', markersize=5)
        axS.annotate(f'{fr_b[i]:.1f} Hz', (fr_b[i], amp_b[i]),
                     textcoords='offset points', xytext=(4, 4), fontsize=8)
    axS.set_xlabel('Frequency (Hz)')
    axS.set_ylabel('Amplitude')
    axS.set_title(f'Response spectrum — dominant resonance ~{dominant_hz:.1f} Hz')
    axS.set_xlim(0, TAP_FFT_MAX_HZ)
    axS.grid(True, alpha=0.4)

    sm = np.ma.masked_invalid(spec[:, sband].T)
    pcm = axG.pcolormesh(spec_t, sfr[sband], 20 * np.log10(sm / np.nanmax(sm) + 1e-9),
                         cmap='magma', vmin=-60, vmax=0)
    axG.set_xlabel('Time (s)')
    axG.set_ylabel('Frequency (Hz)')
    axG.set_title('Ring-down spectrogram')
    fig.colorbar(pcm, ax=axG, label='Amplitude (dB re max)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_fft.png'), dpi=200)
    plt.close(fig)
    print(f'  Saved: {name}_fft.png  (dominant ~{dominant_hz:.1f} Hz; peaks {[round(p,1) for p in peak_freqs]})')

    # --- CSV: amplitude spectrum --------------------------------------------
    with open(os.path.join(out_dir, f'{name}_spectrum.csv'), 'w') as cf:
        cf.write('frequency_hz,amplitude\n')
        for fq, a in zip(fr_b, amp_b):
            cf.write(f'{fq:.4f},{a:.8f}\n')
    print(f'  Saved: {name}_spectrum.csv')

    summary = {
        'source_file': os.path.basename(filepath),
        'analysis': f'Impulse (rubber-mallet) ring-down FFT of {TAP_SIGNAL}; structural natural frequencies',
        'signal': TAP_SIGNAL,
        'sample_rate_hz': float(fs),
        'duration_s': float(t[-1] - t[0]),
        'dominant_resonance_hz': dominant_hz,
        'resonance_peaks_hz': peak_freqs,
    }
    for j, pf in enumerate(peak_freqs):
        summary[f'peak_{j + 1}_hz'] = pf
    return write_summary(name, summary, out_dir)


def _fft_spectrogram(out_dir, name, t, sig, fs, fmax, win_n, title, ylabel='Torque Amplitude (Nm)'):
    """Segment-averaged amplitude spectrum + time spectrogram of `sig`.

    Saves `{name}_fft.png` (Welch-style mean spectrum over a time spectrogram)
    and `{name}_spectrum.csv`. Returns the dominant frequency (Hz, above DC)."""
    w = min(win_n, len(sig))
    hop = w // 2
    win = np.hanning(w)
    fr = np.fft.rfftfreq(w, 1.0 / fs)
    sigd = sig - np.mean(sig)
    seg_spectra, seg_times = [], []
    for i0 in range(0, len(sigd) - w, hop):
        seg = sigd[i0:i0 + w]
        amp = np.abs(np.fft.rfft((seg - seg.mean()) * win)) * 2 / np.sum(win)
        seg_spectra.append(amp)
        seg_times.append(t[i0 + w // 2] - t[0])
    seg_spectra = np.array(seg_spectra)
    seg_times = np.array(seg_times)
    fmax = min(fmax, fr.max())
    fk = fr <= fmax
    mean_spectrum = np.nanmean(seg_spectra, axis=0)
    # Dominant line, ignoring the lowest few bins (DC / ramp skirt).
    lo = np.searchsorted(fr, 1.0)
    dom = float(fr[lo:][np.argmax(mean_spectrum[lo:])]) if len(fr) > lo else float('nan')

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(12, 10))
    axT.semilogy(fr[fk], mean_spectrum[fk] + 1e-12, color='tab:blue', linewidth=1.0)
    #axT.axvline(dom, color='tab:red', linestyle=':', linewidth=1.2, label=f'dominant ~{dom:.1f} Hz')
    axT.set_xlim(0, fmax)
    axT.set_xlabel('Frequency (Hz)')
    axT.set_ylabel(ylabel)
    axT.set_title(f'{title} — segment-averaged spectrum (0-{fmax:g} Hz)')
    axT.legend(loc='upper right', fontsize=8)
    axT.grid(True, which='both', alpha=0.4)

    spec_db = 20 * np.log10(seg_spectra[:, fk] / np.nanmax(seg_spectra[:, fk]) + 1e-9)
    pcm = axB.pcolormesh(seg_times, fr[fk], spec_db.T, cmap='magma', vmin=-40, vmax=0, shading='nearest')
    axB.set_xlabel('Time (s)')
    axB.set_ylabel('Frequency (Hz)')
    axB.set_title(f'{title} — spectrogram')
    fig.colorbar(pcm, ax=axB, label='Amplitude (dB re max)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_fft.png'), dpi=200)
    plt.close(fig)
    print(f'  Saved: {name}_fft.png  (dominant ~{dom:.1f} Hz)')

    with open(os.path.join(out_dir, f'{name}_spectrum.csv'), 'w') as cf:
        cf.write('frequency_hz,amplitude\n')
        for fq, a in zip(fr[fk], mean_spectrum[fk]):
            cf.write(f'{fq:.4f},{a:.8f}\n')
    print(f'  Saved: {name}_spectrum.csv')
    return dom


def _fft_spectrogram_vs_command(out_dir, name, t, sig, vcmd, fs, fmax, win_n, bin_w, title,
                                xlabel='|Commanded Output Velocity| (rad/s)'):
    """Torque spectrogram with the x-axis mapped to |COMMANDED velocity| instead of time.

    Identical per-segment STFT to `_fft_spectrogram`, but each segment is placed at
    the |commanded velocity| at its center time (looked up from `vcmd`) and the
    segments are binned by |commanded velocity| and averaged. Binning is what makes
    this well-defined: the command is a triangular sweep, so any given speed is
    visited several times (up/down, both signs) -- a literal time->velocity remap
    would be non-monotonic and un-plottable, so we fold the sign and collapse it onto
    one freq-vs-speed map. Saves `{name}_fft_vs_command.png` and
    `{name}_spectrum_vs_command.csv`. Returns the resonance frequency (Hz)."""
    w = min(win_n, len(sig))
    hop = w // 2
    win = np.hanning(w)
    fr = np.fft.rfftfreq(w, 1.0 / fs)
    sigd = sig - np.mean(sig)

    # |commanded velocity| bins spanning the swept speed range (0 -> max).
    vfin = np.abs(vcmd[np.isfinite(vcmd)])
    vmax = float(np.max(vfin))
    edges = np.arange(0.0, np.ceil(vmax / bin_w) * bin_w + bin_w, bin_w)
    centers = 0.5 * (edges[:-1] + edges[1:])
    accum = np.zeros((len(centers), len(fr)))
    count = np.zeros(len(centers), dtype=int)
    for i0 in range(0, len(sigd) - w, hop):
        vc = np.nanmean(np.abs(vcmd[i0:i0 + w]))   # |commanded velocity| at this segment's center
        if not np.isfinite(vc):
            continue
        seg = sigd[i0:i0 + w]
        amp = np.abs(np.fft.rfft((seg - seg.mean()) * win)) * 2 / np.sum(win)
        b = int(np.clip(np.digitize(vc, edges) - 1, 0, len(centers) - 1))
        accum[b] += amp
        count[b] += 1
    spec = np.array([accum[k] / count[k] if count[k] else np.full(len(fr), np.nan)
                     for k in range(len(centers))])  # (n_vbins, n_freq)

    fmax = min(fmax, fr.max())
    fk = fr <= fmax
    specf = spec[:, fk]
    colsum = np.nansum(specf, axis=0)
    lo = np.searchsorted(fr[fk], 1.0)               # ignore DC / ramp skirt
    res_hz = float(fr[fk][lo:][np.argmax(colsum[lo:])]) if np.count_nonzero(fk) > lo else float('nan')

    # Segment-averaged spectrum (over every segment, identical to `_fft_spectrogram`'s
    # top panel): total amplitude summed over all bins / total segment count.
    mean_spectrum = accum.sum(axis=0) / max(int(count.sum()), 1)

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(12, 10))
    axT.semilogy(fr[fk], mean_spectrum[fk] + 1e-12, color='tab:blue', linewidth=1.0)
    axT.set_xlim(0, fmax)
    axT.set_xlabel('Frequency (Hz)')
    axT.set_ylabel('Torque Amplitude (Nm)')
    axT.set_title(f'{title} — segment-averaged spectrum (0-{fmax:g} Hz)')
    axT.grid(True, which='both', alpha=0.4)

    sm = np.ma.masked_invalid(specf)
    spec_db = 20 * np.log10(sm / np.nanmax(sm) + 1e-9)
    f_edges = np.append(fr[fk], fr[fk][-1] + (fr[1] - fr[0]))
    pcm = axB.pcolormesh(edges, f_edges, spec_db.T, cmap='magma', vmin=-40, vmax=0)
    #axB.axhline(res_hz, color='cyan', linestyle=':', linewidth=1.3, label=f'resonance ~{res_hz:.0f} Hz')
    axB.set_xlabel(xlabel)
    axB.set_ylabel('Frequency (Hz)')
    axB.set_xlim(edges[0], edges[-1])
    axB.set_ylim(0, fmax)
    axB.set_title(f'{title} — spectrogram vs commanded velocity')
    #axB.legend(loc='upper right', facecolor='gray')
    fig.colorbar(pcm, ax=axB, label='Amplitude (dB re max)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_fft_vs_command.png'), dpi=200)
    plt.close(fig)
    print(f'  Saved: {name}_fft_vs_command.png  (resonance ~{res_hz:.0f} Hz)')

    with open(os.path.join(out_dir, f'{name}_spectrum_vs_command.csv'), 'w') as cf:
        cf.write('command_velocity_rad_s,' + ','.join(f'{fq:.2f}Hz' for fq in fr[fk]) + '\n')
        for c, row in zip(centers, specf):
            cf.write(f'{c:.3f},' + ','.join('' if np.isnan(a) else f'{a:.8f}' for a in row) + '\n')
    print(f'  Saved: {name}_spectrum_vs_command.csv')
    return res_hz


# ---------------------------------------------------------------------------
# TORQUE-RAMP STICTION  (breakaway tests -> stiction/<role>)
# ---------------------------------------------------------------------------
# The DUT is torque-controlled with a slow triangular sawtooth: |torque command|
# ramps up from 0, the shaft stays stuck (static friction holds it) until the
# applied torque overcomes breakaway, at which point it lurches and spins until
# the command ramps back through 0; then the opposite sign repeats. The torque at
# the instant of first motion is the breakaway (static-friction) torque.
#
# Preferred stiction metric is the inline input_torque CELL (tared on the at-rest
# windows), read just before motion: it sits at the gearbox shaft, so it reads the
# torque actually delivered there at breakaway -- the gearbox's own stiction, with
# motor friction already excluded (no subtraction needed). The free-motor test had
# the input_torque sensor DISCONNECTED (motor free-hanging, cell out of the load
# path -> reads ~0), so it uses Kt*current instead. The commanded motor torque
# (~= Kt*current) is kept as the actuator-side cross-check. Three roles:
#   motor  -> motor's own internal stiction (Kt*current; input_torque disconnected)
#   input  -> gearbox-input stiction at the INPUT shaft (cell)
#   output -> gearbox-output stiction at the OUTPUT shaft (cell; gearbox flipped)
# Caveat: the output cell reads higher than the commanded motor torque (impossible
# for a clean series measurement), so its absolute value is suspect -- cycloid
# dynamic loading / calibration; treat the output number as an upper bound.
# ---------------------------------------------------------------------------

def _breakaway_events(vel, fs):
    """Return indices where the shaft first breaks free: |vel| crosses the motion
    threshold after having been stuck (below threshold) for TSTIC_STUCK_S."""
    moving = np.abs(vel) > TSTIC_VEL_THRESH_RAD_S
    rises = np.where((~moving[:-1]) & (moving[1:]))[0] + 1
    stuck_n = int(TSTIC_STUCK_S * fs)
    out = []
    for i in rises:
        if i < stuck_n:
            continue
        if np.any(moving[i - stuck_n:i]):  # was already moving recently -> not a fresh breakaway
            continue
        out.append(int(i))
    return out


def analyze_torque_stiction(filepath, role, label):
    name = f'stiction_{role}'
    out_dir = test_dir(os.path.join('stiction', role))
    print(f'Processing GEARBOX STICTION [{role}]: {os.path.basename(filepath)}')

    with h5py.File(filepath, 'r') as f:
        start, end = behavior_span(f)
        t = np.asarray(f['time'][start:end], dtype=float)
        tcmd = np.asarray(f['dut_torque_command'][start:end], dtype=float)  # commanded motor torque (Nm)
        cur = np.asarray(f['dut_current'][start:end], dtype=float)
        vel = np.asarray(f['dut_velocity'][start:end], dtype=float)
        itq = np.asarray(f['input_torque'][start:end], dtype=float)         # inline torque cell (Nm)
    fs = 1.0 / np.median(np.diff(t))
    tl = t - t[0]
    kt = TSTIC_KT_NM_PER_A
    motor_tq = cur * kt  # actual delivered motor torque from current (cross-checks tcmd)
    pre = max(1, int(TSTIC_PRE_S * fs))

    # Tare the inline torque cell on the at-rest windows (|command|~0): the cell
    # carries a DC offset (~-0.03 Nm) that would otherwise bias the breakaway read.
    rest = np.abs(tcmd) < TARE_TORQUE_CMD_THRESH
    cell_offset = float(np.mean(itq[rest])) if np.any(rest) else 0.0
    itq_t = itq - cell_offset  # tared signed inline torque

    events = []
    for i in _breakaway_events(vel, fs):
        a = max(0, i - pre)
        sgn = float(np.sign(vel[i]) or np.sign(tcmd[i - 1]) or 1.0)
        # Breakaway torque = level held just before the shaft moves. tcmd is the
        # clean monotonic ramp (peak at onset); current/inline are pre-onset means.
        # For the inline cell, average the SIGNED tared signal then take magnitude
        # (avoids rectifying its ripple, which would inflate the mean).
        b_cmd = float(abs(tcmd[i - 1]))
        b_motor = float(np.mean(np.abs(motor_tq[a:i])))
        b_inline = float(abs(np.mean(itq_t[a:i])))
        events.append({'i': i, 't': float(tl[i]), 'sgn': sgn,
                       'cmd': b_cmd, 'motor': b_motor, 'inline': b_inline})

    pos = [e for e in events if e['sgn'] > 0]
    neg = [e for e in events if e['sgn'] < 0]
    print(f'  {len(events)} breakaways ({len(pos)} +, {len(neg)} -)')

    # Stiction metric = the inline input_torque cell (tared) for the two gearbox
    # tests: it sits at the gearbox shaft, reading the torque actually delivered
    # there at breakaway (the gearbox's own stiction, free of motor friction).
    # The free-motor test had the input_torque sensor DISCONNECTED (motor was
    # free-hanging, cell out of the load path -> reads ~0), so it must use the
    # Kt*current estimate of the motor's internal friction instead.
    if role == 'motor':
        prim_key = 'motor'
        prim_label = '|breakaway torque| (Nm, Kt·current)'
        prim_src = 'Kt*current (input_torque sensor disconnected on free motor)'
    else:
        prim_key = 'inline'
        prim_label = '|breakaway torque| (Nm, input_torque cell, tared)'
        prim_src = 'input_torque cell (tared)'

    def arr(lst, k):
        return np.array([e[k] for e in lst]) if lst else np.array([])

    def stat(lst, k):
        a = arr(lst, k)
        return (float(np.mean(a)) if len(a) else float('nan'),
                float(np.std(a, ddof=1)) if len(a) > 1 else float('nan'))

    # --- Figure 1: full-record overview (torque command + velocity, breakaways) ---
    fig, (axT, axV) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    axT.plot(tl, tcmd, color='tab:blue', linewidth=0.6, label='torque command (Nm)')
    axT.plot(tl, motor_tq, color='0.6', linewidth=0.4, alpha=0.7, label='Kt·current (Nm)')
    if role != 'motor':
        axT.plot(tl, itq, color='tab:green', linewidth=0.4, alpha=0.6, label='inline input_torque (Nm)')
    for e in events:
        axT.plot(e['t'], e['sgn'] * e['cmd'], 'rv' if e['sgn'] > 0 else 'r^', markersize=5)
    axT.plot([], [], 'rv', label='breakaway')
    axT.set_ylabel('torque (Nm)')
    axT.set_title(f'{os.path.basename(filepath)}: torque-ramp stiction — {label}')
    axT.legend(loc='upper right', fontsize=8)
    axT.grid(True, alpha=0.3)

    axV.plot(tl, vel, color='0.4', linewidth=0.4)
    axV.axhline(TSTIC_VEL_THRESH_RAD_S, color='tab:red', linestyle=':', linewidth=0.8)
    axV.axhline(-TSTIC_VEL_THRESH_RAD_S, color='tab:red', linestyle=':', linewidth=0.8)
    for e in events:
        axV.axvline(e['t'], color='tab:red', alpha=0.25, linewidth=0.6)
    axV.set_ylabel('velocity (rad/s)')
    axV.set_xlabel('Time (s)')
    axV.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_overview.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved: {name}_overview.png')

    # --- Figure 2: breakaway torque per cycle + histogram -------------------
    # Plots the primary metric: inline cell (tared) for the gearbox tests, motor
    # current for the free motor.
    prim_mean, prim_std = stat(events, prim_key)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
    if pos:
        axL.plot(arr(pos, 't'), arr(pos, prim_key), 'o-', color='tab:red', label='+ direction')
    if neg:
        axL.plot(arr(neg, 't'), arr(neg, prim_key), 'o-', color='tab:blue', label='− direction')
    axL.axhline(prim_mean, color='k', linestyle='--', linewidth=1.0,
                label=f'mean {prim_mean:.3f} ± {prim_std:.3f} Nm')
    axL.set_xlabel('Time (s)')
    axL.set_ylabel(prim_label)
    axL.set_title(f'Breakaway torque per ramp cycle  [{prim_src}]')
    axL.legend(fontsize=8)
    axL.grid(True, alpha=0.3)

    axR.hist(arr(pos, prim_key), bins=10, color='tab:red', alpha=0.6, label='+ direction')
    axR.hist(arr(neg, prim_key), bins=10, color='tab:blue', alpha=0.6, label='− direction')
    axR.axvline(prim_mean, color='k', linestyle='--', linewidth=1.2)
    axR.set_xlabel(prim_label)
    axR.set_ylabel('count')
    axR.set_title('Breakaway torque distribution')
    axR.legend(fontsize=8)
    axR.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_breakaway.png'), dpi=150)
    plt.close(fig)
    print(f'  Saved: {name}_breakaway.png')

    # --- Figure 3: FFT + spectrogram of the inline torque (cycloid ripple) ---
    # The inline cell carries the mesh/cycloid ripple during the spin phases; the
    # free-motor test has no inline load, so FFT its current instead.
    if role == 'motor':
        _fft_spectrogram(out_dir, name, t, motor_tq, fs, TSTIC_FFT_MAX_HZ, TSTIC_STFT_WIN,
                         f'Stiction {label}: motor torque (Kt·current)')
    else:
        _fft_spectrogram(out_dir, name, t, itq, fs, TSTIC_FFT_MAX_HZ, TSTIC_STFT_WIN,
                         f'Stiction {label}: inline torque')

    # --- CSV: per-breakaway table -------------------------------------------
    with open(os.path.join(out_dir, f'{name}_events.csv'), 'w') as cf:
        cf.write('time_s,direction,breakaway_cmd_Nm,breakaway_KtCurrent_Nm,breakaway_inline_Nm\n')
        for e in events:
            cf.write(f'{e["t"]:.3f},{"+" if e["sgn"] > 0 else "-"},'
                     f'{e["cmd"]:.5f},{e["motor"]:.5f},{e["inline"]:.5f}\n')
    print(f'  Saved: {name}_events.csv')

    cmd_mean, cmd_std = stat(events, 'cmd')
    motor_mean, motor_std = stat(events, 'motor')
    inline_mean, inline_std = stat(events, 'inline')
    pos_mean, _ = stat(pos, 'cmd')
    neg_mean, _ = stat(neg, 'cmd')
    return write_summary(name, {
        'source_file': os.path.basename(filepath),
        'role': role,
        'description': label,
        'analysis': 'Torque-ramp breakaway: static-friction torque at first shaft motion',
        'duration_s': float(tl[-1]),
        'ramp_peak_cmd_Nm': float(np.max(np.abs(tcmd))),
        'n_breakaways': len(events),
        'n_pos': len(pos),
        'n_neg': len(neg),
        # Headline stiction estimate: the inline torque cell (tared) for the
        # gearbox tests, Kt*current for the free motor (which has no inline load).
        'stiction_estimate_Nm': prim_mean,
        'stiction_estimate_std_Nm': prim_std,
        'stiction_source': prim_src,
        'inline_cell_offset_Nm': cell_offset,
        'breakaway_inline_mean_Nm': inline_mean,
        'breakaway_inline_std_Nm': inline_std,
        'breakaway_cmd_mean_Nm': cmd_mean,
        'breakaway_cmd_std_Nm': cmd_std,
        'breakaway_cmd_pos_mean_Nm': pos_mean,
        'breakaway_cmd_neg_mean_Nm': neg_mean,
        'breakaway_KtCurrent_mean_Nm': motor_mean,
        'breakaway_KtCurrent_std_Nm': motor_std,
        'kt_Nm_per_A': kt,
        'note': 'stiction_estimate = inline input_torque cell (tared on rest windows) read just '
                'before first motion = torque delivered to the gearbox shaft at breakaway, already '
                'free of motor friction (preferred). The free-motor test had the input_torque sensor '
                'disconnected, so its estimate uses Kt*current instead. breakaway_cmd = |torque '
                'command| at breakaway (the actuator-side total, for cross-checking). High std '
                'reflects real cycle-to-cycle variation (e.g. cycloid ripple).',
    }, out_dir)


def compare_stiction(results):
    """Combine the per-role breakaway results into a motor-vs-gearbox picture.

    Gearbox stiction is read DIRECTLY from the inline torque cell (the torque
    delivered to the gearbox shaft at breakaway), which already excludes motor
    friction -- no subtraction needed. The motor's own stiction comes from the
    free-motor Kt*current (its input_torque sensor was disconnected). The
    actuator-side command total and the motor-subtraction estimate are cross-checks."""
    out_dir = test_dir('stiction')
    motor = results.get('motor')
    if motor is None:
        print('  compare_stiction: no motor baseline, skipping comparison')
        return None
    m = motor['stiction_estimate_Nm']            # motor's own stiction (Kt*current, Nm)
    m_std = motor['stiction_estimate_std_Nm']

    # rows: (role, label, stiction[primary], std, cmd_total, gearbox_via_subtraction)
    rows = []
    for role in ('input', 'output'):
        r = results.get(role)
        if r is None:
            continue
        stic = r['stiction_estimate_Nm']                   # cell for gearbox, current for motor
        cmd_total = r['breakaway_cmd_mean_Nm']             # actuator-side breakaway
        gb_sub = (cmd_total - m) if role != 'motor' else 0.0
        rows.append((role, r['description'], stic, r['stiction_estimate_std_Nm'], cmd_total, gb_sub))
    by = {r[0]: r for r in rows}

    gb_input_at_input = by['input'][2] if 'input' in by else float('nan')      # cell, Nm at input
    gb_output_at_output = by['output'][2] if 'output' in by else float('nan')  # cell, Nm at output
    gb_output_at_input = gb_output_at_output / GEAR_RATIO if gb_output_at_output == gb_output_at_output else float('nan')

    # --- Bar chart: direct (torque-cell) gearbox stiction per test ----------
    # Bars are the torque cell read at the gearbox shaft (input / output), each
    # annotated with its measured mean +/- std. The output bar is flagged where the
    # cell reads MORE than the motor even commanded (a physically impossible series
    # reading -> suspect absolute calibration / cycloid dynamic loading).
    fig, ax = plt.subplots(figsize=(10, 6.5))
    labels = [r[1] for r in rows]
    stics = [r[2] for r in rows]
    errs = [r[3] for r in rows]
    x = np.arange(len(rows))
    ax.bar(x, stics, yerr=errs, capsize=5, color=['tab:blue', 'tab:orange'][:len(rows)], alpha=0.85)
    for xi, r in zip(x, rows):
        note = f'{r[2]:.3f} ± {r[3]:.3f} Nm'
        ax.annotate(note, (xi, r[2] + r[3]), textcoords='offset points', xytext=(0, 8),
                    ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Stiction breakaway torque (Nm)')
    ax.set_title('Gearbox stiction breakaway (torque-cell estimate)')
    if stics:
        ax.set_ylim(0, (max(s + e for s, e in zip(stics, errs))) * 1.18)  # headroom for the annotations
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'stiction_comparison.png'), dpi=200)
    plt.close(fig)
    print('  Saved: stiction_comparison.png')

    # --- CSV ----------------------------------------------------------------
    with open(os.path.join(out_dir, 'stiction_comparison.csv'), 'w') as cf:
        cf.write('role,description,stiction_estimate_Nm,stiction_std_Nm,source,'
                 'cmd_total_Nm,gearbox_via_subtraction_Nm\n')
        for r in rows:
            src = 'Kt*current' if r[0] == 'motor' else 'inline cell (tared)'
            cf.write(f'{r[0]},{r[1]},{r[2]:.5f},{r[3]:.5f},{src},{r[4]:.5f},{r[5]:.5f}\n')
    print('  Saved: stiction_comparison.csv')

    return write_summary('stiction_comparison', {
        'analysis': 'Motor-vs-gearbox stiction; gearbox = inline torque cell at the shaft (direct, '
                    'motor friction already excluded). Motor = free-motor Kt*current.',
        'gear_ratio': GEAR_RATIO,
        'motor_stiction_Nm': m,
        'motor_stiction_std_Nm': m_std,
        'gearbox_input_stiction_at_input_Nm': gb_input_at_input,
        'gearbox_output_stiction_at_output_Nm': gb_output_at_output,
        'gearbox_output_stiction_referred_to_input_Nm': gb_output_at_input,
        'input_cmd_total_breakaway_Nm': by['input'][4] if 'input' in by else float('nan'),
        'output_cmd_total_breakaway_Nm': by['output'][4] if 'output' in by else float('nan'),
        'input_gearbox_via_subtraction_Nm': by['input'][5] if 'input' in by else float('nan'),
        'output_gearbox_via_subtraction_Nm': by['output'][5] if 'output' in by else float('nan'),
        'finding': 'Gearbox-input stiction at the input (torque cell) ~ {:.3f} Nm vs motor stiction '
                   '~ {:.3f} Nm: the gearbox input adds far less than the motor\'s own friction, so '
                   'the input-driven command reading is motor-dominated. The output torque cell reads '
                   'higher than the commanded motor torque (physically impossible in series), so its '
                   'absolute value is suspect -- treat the output number as an upper bound with large '
                   'cycloid-driven scatter.'.format(gb_input_at_input, m),
    }, out_dir)


# ---------------------------------------------------------------------------
# OUTPUT-DRIVEN RUNNING TORQUE  (output_driven_running_torque_*.hdf5)
# ---------------------------------------------------------------------------
# Gearbox flipped: the DUT drives the OUTPUT shaft through a +/- velocity
# triangle, so dut_velocity is already the output velocity (no /ratio) and the
# inline input_torque is the output-referred back-drive drag. Same drag (Coulomb
# + viscous) + ripple + Campbell + FFT/spectrogram treatment as the input run.
# ---------------------------------------------------------------------------

def analyze_output_running_torque(filepath):
    name = 'output_running_torque'
    out_dir = test_dir(name)
    print(f'Processing GEARBOX OUTPUT RUNNING TORQUE: {os.path.basename(filepath)}')

    with h5py.File(filepath, 'r') as f:
        start, end = behavior_span(f)
        t = np.asarray(f['time'][start:end], dtype=float)
        v = np.asarray(f['dut_velocity'][start:end], dtype=float)     # output velocity (rad/s)
        vcmd = np.asarray(f['dut_velocity_command'][start:end], dtype=float)  # commanded velocity (rad/s)
        tq = np.asarray(f['input_torque'][start:end], dtype=float)    # output-referred drag (Nm)
    fs = 1.0 / np.median(np.diff(t))

    # --- Drag curve: mean torque per velocity bin (ripple + inertia averaged) ---
    bw = RUNNING_VEL_BIN_RAD_S_OUTPUT
    edges = np.arange(np.floor(v.min() / bw) * bw, np.ceil(v.max() / bw) * bw + bw, bw)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(v, edges) - 1, 0, len(centers) - 1)
    trend = np.convolve(tq, np.ones(RUNNING_DETREND_WIN) / RUNNING_DETREND_WIN, mode='same')
    ripple = tq - trend

    bmean = np.full(len(centers), np.nan)
    bripple = np.full(len(centers), np.nan)
    for b in range(len(centers)):
        sel = idx == b
        if np.count_nonzero(sel) >= 20:
            bmean[b] = np.mean(tq[sel])
            bripple[b] = np.std(ripple[sel])
    # Drop the extreme bins (turn-around points: few samples, high ripple).
    bmean[0] = bmean[-1] = np.nan

    # Coulomb + viscous fit on the (output-referred) drag curve, per side.
    fmin = OUT_RUNNING_FIT_MIN_VEL
    fit = {}
    for tag, mask in (('pos', centers >= fmin), ('neg', centers <= -fmin)):
        m = mask & ~np.isnan(bmean)
        if np.count_nonzero(m) >= 2:
            slope, intercept, _ = _linfit(centers[m], bmean[m])
            fit[tag] = (slope, intercept)
    coulomb = abs(0.5 * (fit['pos'][1] - fit['neg'][1])) if 'pos' in fit and 'neg' in fit else float('nan')
    viscous = 0.5 * (abs(fit['pos'][0]) + abs(fit['neg'][0])) if 'pos' in fit and 'neg' in fit else float('nan')
    vmax = float(np.max(np.abs(v)))  # output velocity
    drag_at_vmax = coulomb + viscous * vmax

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(centers, bmean, 'b.-', markersize=5, label='Mean drag torque (bin)')
    for tag, c in (('pos', 'red'), ('neg', 'green')):
        if tag in fit:
            xs = np.linspace(0, np.nanmax(centers) if tag == 'pos' else np.nanmin(centers), 50)
            ax.plot(xs, fit[tag][0] * xs + fit[tag][1], '--', color=c, linewidth=1.5)
    ax.plot([], [], ' ', label=f'Coulomb={coulomb:.3f} Nm, viscous={viscous:.4f} Nm/(rad/s)')
    ax.set_xlabel('Output Velocity (rad/s)')
    ax.set_ylabel('Running (drag) Torque (Nm, output-referred)')
    ax.set_title('Output-driven (back-drive) Running Torque')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_drag_vs_velocity.png'), dpi=200)
    plt.close(fig)
    print(f'  Saved: {name}_drag_vs_velocity.png')

    # --- Campbell spectrogram: torque spectrum vs |speed| -------------------
    w = RUNNING_STFT_WIN
    hop = w // 2
    win = np.hanning(w)
    fr = np.fft.rfftfreq(w, 1.0 / fs)
    sp_edges = np.arange(0, vmax + RUNNING_SPEED_BIN_RAD_S, RUNNING_SPEED_BIN_RAD_S)
    sp_centers = 0.5 * (sp_edges[:-1] + sp_edges[1:])
    accum = [np.zeros(len(fr)) for _ in sp_centers]
    count = np.zeros(len(sp_centers), dtype=int)
    for i0 in range(0, len(tq) - w, hop):
        seg = tq[i0:i0 + w]
        amp = np.abs(np.fft.rfft((seg - seg.mean()) * win)) * 2 / np.sum(win)
        sbin = int(np.clip(np.digitize(np.mean(np.abs(v[i0:i0 + w])), sp_edges) - 1, 0, len(sp_centers) - 1))
        accum[sbin] += amp
        count[sbin] += 1
    camp = np.array([accum[k] / count[k] if count[k] else np.full(len(fr), np.nan)
                     for k in range(len(sp_centers))])
    colsum = np.nansum(camp, axis=0)
    res_hz = float(fr[30:][np.argmax(colsum[30:])]) if len(fr) > 30 else float('nan')

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(18, 7))
    axL.plot(centers, bripple, 'm.-')
    axL.set_xlabel('Output Velocity (rad/s)')
    axL.set_ylabel('Torque Ripple RMS (Nm)')
    axL.set_title('Ripple vs speed')
    axL.grid(True, alpha=0.4)
    sm = np.ma.masked_invalid(camp)
    pcm = axR.pcolormesh(np.append(fr, fr[-1] + (fr[1] - fr[0])), sp_edges,
                         20 * np.log10(sm / np.nanmax(sm) + 1e-9), cmap='magma', vmin=-40, vmax=0)
    axR.axvline(res_hz, color='cyan', linestyle=':', linewidth=1.3, label=f'resonance ~{res_hz:.0f} Hz')
    axR.set_xlabel('Frequency (Hz)')
    axR.set_ylabel('|Output Velocity| (rad/s)')
    axR.set_title('Campbell: torque spectrum vs speed')
    axR.legend(loc='upper right', facecolor='gray')
    fig.colorbar(pcm, ax=axR, label='Torque amplitude (dB re max)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_ripple.png'), dpi=200)
    plt.close(fig)
    print(f'  Saved: {name}_ripple.png  (resonance ~{res_hz:.0f} Hz)')

    # --- FFT spectrum + time spectrogram of the output drag torque ----------
    fft_max_hz = _fft_spectrogram(out_dir, name, t, tq, fs, RUNNING_FFT_MAX_HZ, RUNNING_STFT_WIN,
                                  'Output-driven Running Torque')

    # --- Same spectrogram, x-axis remapped to commanded velocity ------------
    _fft_spectrogram_vs_command(out_dir, name, t, tq, vcmd, fs, RUNNING_FFT_MAX_HZ,
                                RUNNING_STFT_WIN, RUNNING_VCMD_BIN_RAD_S,
                                'Output-driven Running Torque')

    # --- CSV: drag + ripple vs velocity -------------------------------------
    with open(os.path.join(out_dir, f'{name}_curve.csv'), 'w') as cf:
        cf.write('output_velocity_rad_s,mean_drag_torque_Nm,ripple_rms_Nm\n')
        for c, mn, r in zip(centers, bmean, bripple):
            cf.write(f'{c:.3f},{"" if np.isnan(mn) else f"{mn:.5f}"},{"" if np.isnan(r) else f"{r:.5f}"}\n')
    print(f'  Saved: {name}_curve.csv')

    return write_summary(name, {
        'source_file': os.path.basename(filepath),
        'analysis': 'Output-driven (back-drive) running torque: drag (Coulomb+viscous) + ripple + Campbell, output-referred',
        'max_output_speed_rad_s': vmax,
        'coulomb_drag_Nm': coulomb,
        'viscous_drag_Nm_per_rad_s': viscous,
        'drag_at_max_speed_Nm': drag_at_vmax,
        'ripple_rms_at_max_speed_Nm': float(np.nanmax(bripple)) if np.any(~np.isnan(bripple)) else float('nan'),
        'ripple_resonance_hz': res_hz,
        'fft_dominant_hz': fft_max_hz,
    }, out_dir)


# ---------------------------------------------------------------------------
# LOADED TORQUE RIPPLE  (efficiency*.hdf5 dwells -> torque_ripple)
# ---------------------------------------------------------------------------
# The cleanest ripple magnitude comes from the loaded, constant-speed efficiency
# dwells (not the no-load running-torque sweeps, whose high-speed ripple is
# dominated by a structural fixture resonance and whose input-side torques sit
# near the cell noise floor). Per loaded dwell we detrend the output torque and
# take its AC RMS. Standstill dwells (no rotation) isolate the load-side noise
# floor; rotating dwells reveal the gearbox transmission ripple above it.
# ---------------------------------------------------------------------------

def _ac_rms(x):
    """RMS of x after removing a linear trend (DC + slow settling drift)."""
    n = len(x)
    tt = np.arange(n)
    c = np.polyfit(tt, x, 1)
    r = x - (c[0] * tt + c[1])
    return float(np.sqrt(np.mean(r ** 2)))


def analyze_loaded_ripple(filepaths):
    name = 'torque_ripple'
    out_dir = test_dir(name)
    print(f'Processing GEARBOX LOADED TORQUE RIPPLE: {len(filepaths)} efficiency files')

    moving, still = [], []  # each row: (|mean output torque| Nm, ripple RMS Nm, ripple %)
    for fp in filepaths:
        with h5py.File(fp, 'r') as f:
            bidx = f['behavior_indices'][:]
            for i in range(len(bidx)):
                a, b = int(bidx[i, 0]), int(bidx[i, 1])
                if b - a < EFFICIENCY_MIN_DWELL:
                    continue
                sl = slice(a + int((b - a) * EFFICIENCY_SETTLE_FRAC), b)
                vcmd = round(float(np.nanmean(f['dut_velocity_command'][sl])))
                lt = np.asarray(f['load_torque'][sl], dtype=float)  # output torque (Nm)
                m = float(np.mean(lt))
                if abs(m) < RIPPLE_MIN_OUTPUT_TORQUE_NM:
                    continue
                rms = _ac_rms(lt)
                (still if vcmd == 0 else moving).append((abs(m), rms, 100.0 * rms / abs(m)))
    moving = np.array(moving) if moving else np.empty((0, 3))
    still = np.array(still) if still else np.empty((0, 3))

    floor_Nm = float(np.mean(still[:, 1])) if len(still) else float('nan')
    ripple_Nm = float(np.mean(moving[:, 1])) if len(moving) else float('nan')
    ripple_pct = float(np.mean(moving[:, 2])) if len(moving) else float('nan')
    # Remove the static load-side floor (in quadrature) to isolate the gearbox.
    gearbox_Nm = float(np.sqrt(max(ripple_Nm ** 2 - floor_Nm ** 2, 0.0))) \
        if ripple_Nm == ripple_Nm and floor_Nm == floor_Nm else ripple_Nm
    print(f'  loaded ripple: rotating {ripple_Nm:.3f} Nm RMS ({ripple_pct:.2f}% of load), '
          f'static floor {floor_Nm:.3f} Nm -> gearbox {gearbox_Nm:.3f} Nm')

    # --- Figure: ripple RMS vs transmitted torque (rotating vs standstill) ---
    fig, ax = plt.subplots(figsize=(10, 6.5))
    if len(still):
        ax.scatter(still[:, 0], still[:, 1], s=30, color='0.55', alpha=0.8,
                   label='static hold (no rotation) — load-side noise floor')
        ax.axhline(floor_Nm, color='0.55', linestyle='--', linewidth=1.2)
    if len(moving):
        ax.scatter(moving[:, 0], moving[:, 1], s=30, color='tab:blue', alpha=0.8,
                   label='rotating (loaded dwells)')
        ax.axhline(ripple_Nm, color='tab:blue', linestyle='--', linewidth=1.4,
                   label=f'rotating mean {ripple_Nm:.2f} Nm RMS')
    ax.set_xlabel('Transmitted output torque (Nm)')
    ax.set_ylabel('Output torque ripple, RMS (Nm)')
    ax.set_title('Output torque ripple from loaded constant-speed dwells')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}.png'), dpi=200)
    plt.close(fig)
    print(f'  Saved: {name}.png')

    # --- CSV: per-dwell ripple ----------------------------------------------
    with open(os.path.join(out_dir, f'{name}.csv'), 'w') as cf:
        cf.write('state,transmitted_torque_Nm,ripple_rms_Nm,ripple_pct\n')
        for state, arr in (('static', still), ('rotating', moving)):
            for t_nm, rms, pct in arr:
                cf.write(f'{state},{t_nm:.3f},{rms:.5f},{pct:.4f}\n')
    print(f'  Saved: {name}.csv')

    return write_summary(name, {
        'analysis': 'Output torque ripple from loaded constant-speed efficiency dwells; AC RMS of '
                    'detrended output torque, rotating vs standstill',
        'source': 'efficiency dwells',
        'min_transmitted_torque_Nm': RIPPLE_MIN_OUTPUT_TORQUE_NM,
        'n_rotating_dwells': int(len(moving)),
        'n_standstill_dwells': int(len(still)),
        'ripple_rms_rotating_Nm': ripple_Nm,
        'ripple_rms_rotating_pct_of_load': ripple_pct,
        'noise_floor_static_Nm': floor_Nm,
        'gearbox_ripple_floor_removed_Nm': gearbox_Nm,
        'note': 'Rotating ripple is ~load-independent in absolute terms, so the percentage falls as '
                'load rises. The static-hold dwells (no rotation) set the load-side noise floor; the '
                'gearbox figure removes it in quadrature.',
    }, out_dir)


def find_file(*needles):
    """Return the path of the first .hdf5 whose lowercased name contains all needles."""
    for fn in sorted(os.listdir(GEARBOX_TESTS_DIR)):
        if not fn.endswith('.hdf5'):
            continue
        if all(n in fn.lower() for n in needles):
            return os.path.join(GEARBOX_TESTS_DIR, fn)
    return None


def main():
    reset_results_dir()
    combined = {}

    bl_path = find_file(BACKLASH_FILE_STEM)
    if bl_path:
        combined['backlash'] = analyze_backlash(bl_path)
    else:
        print(f'No gearbox backlash file found for stem {BACKLASH_FILE_STEM!r}.')

    eff_paths = sorted(glob.glob(os.path.join(GEARBOX_TESTS_DIR, EFFICIENCY_FILE_GLOB)))
    if eff_paths:
        # Correct the input-torque sensor zero using the value implied by the
        # efficiency data's own CW/CCW symmetry (the backlash2 zero drifted and
        # over-corrected; this self-derived value matches backlash1's -0.010 Nm).
        combined['efficiency'] = analyze_efficiency(eff_paths, auto_zero=True)
        combined['torque_ripple'] = analyze_loaded_ripple(eff_paths)
    else:
        print('No gearbox efficiency files found.')

    rt_path = find_file(RUNNING_FILE_STEM)
    if rt_path:
        combined['running_torque'] = analyze_running_torque(rt_path)
    else:
        print(f'No gearbox running-torque file found for stem {RUNNING_FILE_STEM!r}.')

    ort_path = find_file(OUT_RUNNING_FILE_STEM)
    if ort_path:
        combined['output_running_torque'] = analyze_output_running_torque(ort_path)
    else:
        print(f'No output running-torque file found for stem {OUT_RUNNING_FILE_STEM!r}.')

    tap_path = find_file(TAP_FILE_STEM)
    if tap_path:
        combined['tap_test'] = analyze_tap_test(tap_path)
    else:
        print(f'No gearbox tap-test file found for stem {TAP_FILE_STEM!r}.')

    # Torque-ramp stiction: motor baseline + the two gearbox-driven tests, then a
    # combined motor-vs-gearbox comparison.
    stic_results = {}
    for role, needle, label in TSTIC_FILES:
        stic_path = find_file(needle)
        if stic_path:
            summary = analyze_torque_stiction(stic_path, role, label)
            stic_results[role] = summary
            combined[f'stiction_{role}'] = summary
        else:
            print(f'No stiction file found for needle {needle!r}.')
    if stic_results:
        cmp_summary = compare_stiction(stic_results)
        if cmp_summary:
            combined['stiction_comparison'] = cmp_summary

    if combined:
        with open(os.path.join(RESULTS_DIR, 'gearbox_tests_report.json'), 'w') as jf:
            json.dump(combined, jf, indent=2, default=str)
        with open(os.path.join(RESULTS_DIR, 'gearbox_tests_report.txt'), 'w') as tf:
            tf.write('Gearbox Tests — Combined Report\n')
            tf.write('=' * 40 + '\n\n')
            for test, summary in combined.items():
                tf.write(f'[{test}]\n')
                for k, v in summary.items():
                    if isinstance(v, float):
                        tf.write(f'  {k:30s}: {v:.6g}\n')
                    else:
                        tf.write(f'  {k:30s}: {v}\n')
                tf.write('\n')
        print('Saved combined report: gearbox_tests_report.txt / .json')

    print('Done.')


if __name__ == '__main__':
    main()
