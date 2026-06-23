"""Combined post-processing of Brian's motor dyno test logs.

Opens each relevant .hdf5 file in dyno/logs/brian_tests/motor_tests/, runs the
analysis selected by filename, and writes all report materials (figures, CSVs,
summary text/JSON) into dyno/logs/brian_tests/motor_tests/results/. The results
folder is wiped at the start of every run so stale outputs never accumulate.
(Gearbox logs live in a sibling dyno/logs/brian_tests/gearbox_tests/ folder and
are handled by a separate script.)

Run with the anaconda base interpreter:
    "C:/Users/Nathan Justus/anaconda3/python.exe" dyno/src/mac_post_processing.py
"""

import os
import json
import shutil

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless: save figures, never block on display
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# --- Paths ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MOTOR_TESTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'logs', 'brian_tests', 'motor_tests'))
RESULTS_DIR = os.path.join(MOTOR_TESTS_DIR, 'results')

# --- Tunables ---------------------------------------------------------------
TORQUE_RESPONSE_SAMPLE_STEP_ARMS = 0.25  # density of the fit-derived current/torque CSV

RUNNING_TORQUE_VEL_BIN_RAD_S = 0.5     # velocity bin width for the smoothed drag curve
RUNNING_TORQUE_MIN_BIN_COUNT = 20      # min samples for a bin to be kept
RUNNING_TORQUE_FIT_MIN_ABS_VEL = 2.0   # ignore |w| below this for the Coulomb+viscous fit
                                       # (excludes the stiction / zero-crossing nonlinearity)

COGGING_FILES = ('cogging_5s', 'cogging_10s')  # filename stems to analyze
COGGING_POLE_PAIRS = 30                # electrical periods per mechanical revolution
COGGING_GLOBAL_BINS = 360             # angle bins over one full mechanical revolution
COGGING_LOCAL_BINS = 100              # angle bins over one pole-pair (phase) period
COGGING_VEL_TOL_FRAC = 0.10           # steady-state filter: |v - target| < frac*|target|
COGGING_MIN_DWELL_S = 1.0             # min segment duration to count as a setpoint dwell
COGGING_DIRECTION = 1                 # use +velocity (1) or -velocity (-1) segments
COGGING_SPECTRUM_FMAX_HZ = 480        # max frequency on the Campbell diagram (~just below Nyquist)
COGGING_SPECTRUM_NBINS = 480          # frequency-grid bins (~1 Hz); max-pooled to preserve peaks
COGGING_SPECTRUM_ORDERS = 6           # number of pole-pass harmonic order lines to overlay
COGGING_SPECTRUM_N_RESONANCES = 3     # number of fixed-resonance lines to detect and mark


def reset_results_dir():
    """Delete and recreate the results folder so each run starts clean."""
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR)
    print(f'Results dir reset: {RESULTS_DIR}')


def behavior_span(f):
    """Return (start, end) index of the first logged behavior in the file."""
    start, end = f['behavior_indices'][0]
    return int(start), int(end)


def test_dir(name):
    """Create and return a per-test subfolder inside the results directory."""
    d = os.path.join(RESULTS_DIR, name)
    os.makedirs(d, exist_ok=True)
    return d


def write_summary(name, summary_dict, out_dir):
    """Write a per-test summary as both human-readable .txt and .json."""
    json_path = os.path.join(out_dir, f'{name}_summary.json')
    with open(json_path, 'w') as jf:
        json.dump(summary_dict, jf, indent=2, default=str)

    txt_path = os.path.join(out_dir, f'{name}_summary.txt')
    with open(txt_path, 'w') as tf:
        tf.write(f'{name} summary\n')
        tf.write('=' * (len(name) + 8) + '\n\n')
        for k, v in summary_dict.items():
            if isinstance(v, float):
                tf.write(f'{k:32s}: {v:.6g}\n')
            else:
                tf.write(f'{k:32s}: {v}\n')
    print(f'  Saved: {os.path.basename(json_path)}, {os.path.basename(txt_path)}')
    return summary_dict


# ---------------------------------------------------------------------------
# TORQUE RESPONSE  (torque response final.hdf5 -> WARMUP-RUN0)
# ---------------------------------------------------------------------------
# Blocked-rotor torque-vs-current sweep. The DUT is held stationary while its
# torque command is swept; we measure phase current (Arms) against load-cell
# torque (Nm). We fit T = A*tanh(B*I) + C: the tanh captures magnetic
# saturation while C absorbs the measured zero-current torque bias (torque-cell
# zero offset, cogging at the parked angle, gravity, etc.). C is reported as a
# diagnostic, then subtracted so the plotted data, fit, and exported CSV are all
# bias-corrected (odd, through origin). The datasheet curve is overlaid and the
# de-biased fit is exported as a CSV sampled at a fixed current density.
# ---------------------------------------------------------------------------

def tanh_model(current, A, B):
    """De-biased motor torque model (odd, through origin)."""
    return A * np.tanh(B * current)


def tanh_model_offset(current, A, B, C):
    """Measurement model including the zero-current torque bias C."""
    return A * np.tanh(B * current) + C


def load_spec_sheet():
    """Read motor_current_specs.csv (Phase Current Arms, Torque Nm) if present."""
    spec_path = os.path.join(MOTOR_TESTS_DIR, 'motor_current_specs.csv')
    if not os.path.isfile(spec_path):
        print(f'  WARNING: spec sheet not found at {spec_path}')
        return None, None
    rows = np.genfromtxt(spec_path, delimiter=',', skip_header=1)
    rows = rows[~np.isnan(rows).any(axis=1)]  # drop blank trailing lines
    return rows[:, 0], rows[:, 1]


def analyze_torque_response(filepath):
    name = 'torque_response'
    out_dir = test_dir(name)
    print(f'Processing TORQUE RESPONSE: {os.path.basename(filepath)}')

    with h5py.File(filepath, 'r') as f:
        start, end = behavior_span(f)
        current = np.asarray(f['dut_current'][start:end], dtype=float)   # Arms
        torque = np.asarray(f['load_torque'][start:end], dtype=float)    # Nm

    # Drop any NaNs before fitting
    valid = ~(np.isnan(current) | np.isnan(torque))
    current, torque = current[valid], torque[valid]

    # --- Fit T = A*tanh(B*I) + C --------------------------------------------
    # Seed A near the torque magnitude span, B from the small-signal slope, C=0.
    A0 = float(np.max(np.abs(torque)))
    B0 = 1.0 / max(float(np.max(np.abs(current))), 1e-6)
    popt, _ = curve_fit(tanh_model_offset, current, torque, p0=[A0, B0, 0.0], maxfev=10000)
    A, B, C = float(popt[0]), float(popt[1]), float(popt[2])

    fit_torque = tanh_model_offset(current, A, B, C)
    residuals = torque - fit_torque
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((torque - np.mean(torque)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    kt_small_signal = A * B  # Nm/Arm slope at I=0

    spec_current, spec_torque = load_spec_sheet()

    # --- Plot: bias-corrected scatter + spec sheet + de-biased fit ----------
    # Subtract the fitted bias C from the measured data so it shares the odd,
    # through-origin frame of the de-biased fit and the datasheet.
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(current, torque - C, s=4, alpha=0.25, color='steelblue',
               label=f'Measured, bias-corrected (−C = {-C:+.3f} Nm)')

    cur_min, cur_max = float(np.min(current)), float(np.max(current))
    fit_x = np.linspace(cur_min, cur_max, 500)
    ax.plot(fit_x, tanh_model(fit_x, A, B), 'r-', linewidth=2,
            label=f'Fit: T = {A:.2f}·tanh({B:.4f}·I)\nR²={r_squared:.4f}, Kt₀={kt_small_signal:.3f} Nm/Arm')

    if spec_current is not None:
        ax.plot(spec_current, spec_torque, 'k^--', markersize=7,
                linewidth=1.2, label='Spec sheet')

    ax.set_xlabel('Phase Current (Arms)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Torque Response: Torque vs Current')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(out_dir, f'{name}_torque_vs_current.png')
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f'  Saved: {os.path.basename(fig_path)}')

    # --- CSV: fit-derived current/torque at fixed sampling density ----------
    step = TORQUE_RESPONSE_SAMPLE_STEP_ARMS
    sample_lo = np.floor(cur_min / step) * step
    sample_hi = np.ceil(cur_max / step) * step
    sample_current = np.arange(sample_lo, sample_hi + step / 2, step)
    sample_torque = tanh_model(sample_current, A, B)
    csv_path = os.path.join(out_dir, f'{name}_fit_current_torque.csv')
    with open(csv_path, 'w') as cf:
        cf.write('Phase Current (Arms),Torque (Nm)\n')
        for i, t in zip(sample_current, sample_torque):
            cf.write(f'{i:.4f},{t:.4f}\n')
    print(f'  Saved: {os.path.basename(csv_path)} ({len(sample_current)} rows @ {step} Arms)')

    return write_summary(name, {
        'source_file': os.path.basename(filepath),
        'analysis': 'Blocked-rotor torque vs phase current; tanh+offset fit, bias-corrected',
        'n_samples': int(current.size),
        'current_min_Arms': cur_min,
        'current_max_Arms': cur_max,
        'fit_model': 'T = A*tanh(B*I) + C',
        'fit_A_Nm': A,
        'fit_B_perArms': B,
        'zero_current_bias_C_Nm': C,
        'small_signal_Kt_Nm_per_Arms': kt_small_signal,
        'saturation_torque_Nm': A,
        'fit_r_squared': r_squared,
        'fit_rmse_Nm': rmse,
        'csv_is_bias_corrected': True,
        'csv_sample_step_Arms': step,
    }, out_dir)


# ---------------------------------------------------------------------------
# RUNNING TORQUE  (rt8.hdf5 -> CSV-ENCODER-RUN0)
# ---------------------------------------------------------------------------
# No-load drag sweep. The LOAD motor spins the unpowered DUT (dut_current ~ 0)
# through a triangular velocity profile (0 -> +30 -> 0 -> -30 -> 0 rad/s) while
# the load cell records the torque needed to turn it. That torque is pure
# parasitic loss: Coulomb friction + viscous drag (+ cogging/noise ripple).
#
# The triangle ramps have constant acceleration, so the raw torque also carries
# an inertia term J*alpha that is +on the accel pass and -on the decel pass.
# Binning by velocity over the whole run averages the accel and decel passes
# together, which cancels J*alpha and leaves the true friction-vs-speed curve.
# We then fit Coulomb+viscous (T = Tc*sign(w) + b*w) per direction and report.
# ---------------------------------------------------------------------------

def _linfit(x, y):
    """Linear fit y = slope*x + intercept; returns (slope, intercept, r2)."""
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(slope), float(intercept), r2


def analyze_running_torque(filepath):
    name = 'running_torque'
    out_dir = test_dir(name)
    print(f'Processing RUNNING TORQUE: {os.path.basename(filepath)}')

    with h5py.File(filepath, 'r') as f:
        start, end = behavior_span(f)
        velocity = np.asarray(f['load_velocity'][start:end], dtype=float)  # rad/s
        # Negate the raw load-cell reading so torque is expressed as the
        # conventional "torque required to drive" (same sign as velocity:
        # positive for +w, negative for -w). This puts the drag curve in
        # quadrants 1 & 3 and yields naturally positive Coulomb/viscous coeffs.
        torque = -np.asarray(f['load_torque'][start:end], dtype=float)     # Nm (required-to-drive)

    valid = ~(np.isnan(velocity) | np.isnan(torque))
    velocity, torque = velocity[valid], torque[valid]

    # --- Bin by velocity; mean per bin averages accel+decel passes ----------
    bw = RUNNING_TORQUE_VEL_BIN_RAD_S
    edges = np.arange(np.floor(velocity.min() / bw) * bw,
                      np.ceil(velocity.max() / bw) * bw + bw, bw)
    centers = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(velocity, edges) - 1
    idx = np.clip(idx, 0, len(centers) - 1)

    bin_center, bin_mean, bin_median, bin_std, bin_count = [], [], [], [], []
    for b in range(len(centers)):
        sel = idx == b
        n = int(np.count_nonzero(sel))
        if n < RUNNING_TORQUE_MIN_BIN_COUNT:
            continue
        bin_center.append(float(centers[b]))
        bin_mean.append(float(np.mean(torque[sel])))
        bin_median.append(float(np.median(torque[sel])))
        bin_std.append(float(np.std(torque[sel])))
        bin_count.append(n)
    bin_center = np.array(bin_center)
    bin_mean = np.array(bin_mean)
    bin_median = np.array(bin_median)
    bin_std = np.array(bin_std)

    # --- Coulomb + viscous fit, per direction -------------------------------
    # Measured load_torque opposes the motion (it's drag), so a per-side linear
    # fit T = slope*w + intercept yields negative coefficients; the physical
    # quantities are the magnitudes. We fit each side above the stiction floor
    # and report:
    #   - Coulomb friction = half the torque discontinuity at w=0 (magnitude)
    #   - Viscous coeff per direction = |slope| (Nm per rad/s)
    #   - Zero-velocity offset = midpoint of the two intercepts. This is either a
    #     torque-cell bias or a genuine Coulomb direction-asymmetry; the two are
    #     degenerate from this test alone, so it is reported as a single number.
    fmin = RUNNING_TORQUE_FIT_MIN_ABS_VEL
    pos = bin_center >= fmin
    neg = bin_center <= -fmin
    summary_fit = {}
    fit_lines = {}
    raw = {}
    for tag, mask in (('pos', pos), ('neg', neg)):
        if np.count_nonzero(mask) >= 2:
            slope, intercept, r2 = _linfit(bin_center[mask], bin_mean[mask])
            raw[tag] = (slope, intercept)
            fit_lines[tag] = (slope, intercept)
            # With the required-to-drive sign convention the viscous slope is
            # naturally positive; report it signed so noise sign-flips are visible.
            summary_fit[f'viscous_{tag}_Nm_per_rad_s'] = slope
            summary_fit[f'fit_r2_{tag}'] = r2
            summary_fit[f'_intercept_{tag}_Nm'] = intercept  # raw, leading-underscore = diagnostic

    if 'pos' in raw and 'neg' in raw:
        ip, in_ = raw['pos'][1], raw['neg'][1]
        summary_fit['coulomb_friction_Nm'] = abs(0.5 * (ip - in_))
        summary_fit['zero_velocity_offset_Nm'] = 0.5 * (ip + in_)
        summary_fit['viscous_mean_Nm_per_rad_s'] = 0.5 * (raw['pos'][0] + raw['neg'][0])
        summary_fit['viscous_asymmetry_Nm_per_rad_s'] = raw['pos'][0] - raw['neg'][0]

    # --- Remove the zero-velocity offset -----------------------------------
    # Subtract the fitted offset so the drag curve is centered on zero and the
    # Coulomb step is symmetric (+/-). Reported as a diagnostic; note it could be
    # a genuine Coulomb direction-asymmetry rather than pure cell bias (the two
    # are unresolvable from this test alone).
    offset = summary_fit.get('zero_velocity_offset_Nm', 0.0)
    torque_c = torque - offset
    bin_mean_c = bin_mean - offset
    bin_median_c = bin_median - offset

    # --- Plot: raw cloud + smoothed curve + per-direction fits --------------
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(velocity, torque_c, s=2, alpha=0.10, color='gray',
               label=f'Raw, offset-corrected (−{offset:+.3f} Nm; incl. inertia/ripple)')
    ax.plot(bin_center, bin_mean_c, 'b.-', markersize=6, linewidth=1.2,
            label=f'Smoothed mean ({bw:g} rad/s bins, accel+decel averaged)')

    for tag, color in (('pos', 'limegreen'), ('neg', 'red')):
        if tag in fit_lines:
            slope, intercept = fit_lines[tag]
            mask = pos if tag == 'pos' else neg
            xs = np.linspace(0, bin_center[mask].max() if tag == 'pos' else bin_center[mask].min(), 50)
            ax.plot(xs, slope * xs + (intercept - offset), color=color, linestyle='--', linewidth=1.8,
                    label=(f'{tag} fit: b={summary_fit[f"viscous_{tag}_Nm_per_rad_s"]:.4f} Nm/(rad/s)'))

    if 'coulomb_friction_Nm' in summary_fit:
        ax.plot([], [], ' ',
                label=(f'Coulomb={summary_fit["coulomb_friction_Nm"]:.3f} Nm '
                       f'(removed offset={offset:+.3f} Nm)'))

    ax.set_xlabel('Driven Velocity (rad/s)')
    ax.set_ylabel('Running Torque (Nm, required to drive), offset-corrected')
    ax.set_title('Running Torque vs Driven Velocity (no-load drag)')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig_path = os.path.join(out_dir, f'{name}_torque_vs_velocity.png')
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f'  Saved: {os.path.basename(fig_path)}')

    # --- CSV: processed (binned) running-torque curve, offset-corrected -----
    csv_path = os.path.join(out_dir, f'{name}_curve.csv')
    with open(csv_path, 'w') as cf:
        cf.write('Velocity (rad/s),Mean Torque (Nm),Median Torque (Nm),Std Torque (Nm),Sample Count\n')
        for v, m, med, sd, n in zip(bin_center, bin_mean_c, bin_median_c, bin_std, bin_count):
            cf.write(f'{v:.4f},{m:.4f},{med:.4f},{sd:.4f},{n}\n')
    print(f'  Saved: {os.path.basename(csv_path)} ({len(bin_center)} bins @ {bw:g} rad/s, offset-corrected)')

    return write_summary(name, {
        'source_file': os.path.basename(filepath),
        'analysis': 'No-load running torque vs velocity; accel+decel averaged, Coulomb+viscous fit, offset-corrected',
        'n_samples': int(velocity.size),
        'velocity_min_rad_s': float(velocity.min()),
        'velocity_max_rad_s': float(velocity.max()),
        'velocity_bin_rad_s': bw,
        'n_bins_kept': int(len(bin_center)),
        'fit_model': 'T(w) = Tc*sign(w) + b*w',
        'fit_min_abs_vel_rad_s': fmin,
        'outputs_offset_corrected': True,
        **summary_fit,
    }, out_dir)


# ---------------------------------------------------------------------------
# COGGING  (cogging_5s.hdf5, cogging_10s.hdf5 -> COGGING-TORQUE-VC-RUN0)
# ---------------------------------------------------------------------------
# Velocity-controlled cogging sweep. The LOAD motor drives the DUT through a
# grid of speed plateaus (+/-4..30 rad/s) x torque levels (0,10,20,25 Nm); the
# DUT holds each torque while we record load_torque vs motor angle. Because each
# dwell spans many revolutions, folding torque over angle reveals the cogging
# pattern. Three views are produced (each with a per-trace CSV):
#   1. Global: torque vs full mechanical angle (pos % 2pi) at the slowest speed,
#      one shaded trace per torque level.
#   2. Local : torque vs pole-pair phase angle (pos % (2pi/pole_pairs)), folded
#      and mean-subtracted, phase-shifted so peak cogging sits at 0/100%.
#   3. Speed dependency: the local view for the lowest and highest torque level,
#      overlaying every speed that level was run at.
# ---------------------------------------------------------------------------

def _cogging_segments(vc, tc, t):
    """Split the run into constant (speed-plateau, torque) dwell segments."""
    vr = np.round(vc / 2) * 2   # snap velocity command to nearest plateau
    tr = np.round(tc)
    key = vr * 1000 + tr
    chg = np.where(np.diff(key) != 0)[0] + 1
    bounds = np.concatenate(([0], chg, [len(key)]))
    segs = []
    for i in range(len(bounds) - 1):
        a, b = int(bounds[i]), int(bounds[i + 1])
        if t[b - 1] - t[a] < COGGING_MIN_DWELL_S:
            continue
        segs.append({'v': float(vr[a]), 'torque': float(tr[a]), 'start': a, 'end': b})
    return segs


def _cogging_trace(pos, vel, tq, segs, v_signed, torque):
    """Return (position, torque) steady-state samples for one (speed, torque)."""
    for sg in segs:
        if abs(sg['v'] - v_signed) < 1.0 and abs(sg['torque'] - torque) < 0.5:
            a, b = sg['start'], sg['end']
            mask = np.abs(vel[a:b] - v_signed) < COGGING_VEL_TOL_FRAC * abs(v_signed)
            return pos[a:b][mask], tq[a:b][mask]
    return None, None


def _fold(frac, val, nbins):
    """Bin val by frac in [0,1); return bin centers, mean, std, count."""
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(frac, edges) - 1, 0, nbins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(nbins, np.nan)
    std = np.full(nbins, np.nan)
    cnt = np.zeros(nbins, dtype=int)
    for b in range(nbins):
        sel = val[idx == b]
        cnt[b] = sel.size
        if sel.size:
            mean[b] = sel.mean()
            std[b] = sel.std()
    return centers, mean, std, cnt


def _write_wide_csv(path, index_name, index, columns):
    """Write a wide CSV: index column + one column per (name, array) in columns."""
    with open(path, 'w') as cf:
        cf.write(','.join([index_name] + [c[0] for c in columns]) + '\n')
        for i in range(len(index)):
            cells = [f'{index[i]:.4f}']
            for _, arr in columns:
                cells.append('' if np.isnan(arr[i]) else f'{arr[i]:.5f}')
            cf.write(','.join(cells) + '\n')


def _cogging_spectrum(q, fs, fedges):
    """Amplitude spectrum of q, max-pooled into the bins defined by fedges.
    Max-pooling (vs interpolation) preserves narrow high-Q resonance peaks."""
    nb = len(fedges) - 1
    q = q - np.mean(q)
    n = q.size
    if n < 256:
        return np.full(nb, np.nan)
    win = np.hanning(n)
    amp = np.abs(np.fft.rfft(q * win)) * 2.0 / np.sum(win)  # amplitude-scaled
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    idx = np.digitize(fr, fedges) - 1
    out = np.full(nb, np.nan)
    for b in range(nb):
        sel = amp[idx == b]
        if sel.size:
            out[b] = sel.max()
    return out


def cogging_harmonics(name, stem, out_dir, segs, pos, vel, tq, speeds, sign, pp, fs, torque):
    """Campbell diagram: spectral amplitude vs (frequency, speed), with cogging
    order rays overlaid and fixed resonances marked. Returns dominant resonance Hz."""
    fmax = min(COGGING_SPECTRUM_FMAX_HZ, 0.97 * fs / 2)
    fedges = np.linspace(0.0, fmax, COGGING_SPECTRUM_NBINS + 1)
    centers = 0.5 * (fedges[:-1] + fedges[1:])
    M = []
    for spd in speeds:
        _, q = _cogging_trace(pos, vel, tq, segs, sign * spd, torque)
        M.append(_cogging_spectrum(q, fs, fedges) if q is not None and q.size else np.full(len(centers), np.nan))
    M = np.array(M)  # [n_speeds, n_freq]

    # Detect fixed resonances: peaks in the cross-speed energy sum (a resonance
    # is reinforced wherever a cogging order sweeps through it).
    colsum = np.nansum(M, axis=0)
    valid = centers >= 30  # ignore very low freq
    pk_idx, props = find_peaks(np.where(valid, colsum, 0),
                               height=np.nanmax(colsum) * 0.15, distance=15)
    order = np.argsort(props['peak_heights'])[::-1][:COGGING_SPECTRUM_N_RESONANCES]
    res_freqs = sorted(float(centers[pk_idx[o]]) for o in order)
    dominant = float(centers[pk_idx[order[0]]]) if len(order) else float('nan')

    sp = np.array(speeds, dtype=float)
    Mdb = 20.0 * np.log10(M / np.nanmax(M) + 1e-9)
    sedges = np.concatenate(([sp[0] - (sp[1] - sp[0]) / 2],
                             (sp[:-1] + sp[1:]) / 2,
                             [sp[-1] + (sp[-1] - sp[-2]) / 2]))

    fig, ax = plt.subplots(figsize=(14, 8))
    pcm = ax.pcolormesh(fedges, sedges, Mdb, cmap='magma', vmin=-40, vmax=0, shading='flat')
    fig.colorbar(pcm, ax=ax, label='Torque amplitude (dB re max)')

    # Cogging order rays: freq = n * (w*pp/2pi); diagonal lines from origin.
    ss = np.linspace(sp.min(), sp.max(), 50)
    for n in range(1, COGGING_SPECTRUM_ORDERS + 1):
        ax.plot(n * ss * pp / (2 * np.pi), ss, '--', color='cyan', linewidth=0.9, alpha=0.7)
        # label each ray near the top of the plot if it's still on-axis there
        ftop = n * sp.max() * pp / (2 * np.pi)
        if ftop <= fmax:
            ax.annotate(f'{n}×', (ftop, sp.max()), color='cyan', fontsize=8,
                        ha='center', va='bottom')

    # Fixed resonance lines (vertical) with frequency labels.
    for rf in res_freqs:
        ax.axvline(rf, color='white', linestyle=':', linewidth=1.3, alpha=0.9)
        ax.annotate(f'{rf:.0f} Hz', (rf, sp.min()), color='white', fontsize=9,
                    rotation=90, va='bottom', ha='right')

    ax.plot([], [], '--', color='cyan', label=f'cogging orders (n=1..{COGGING_SPECTRUM_ORDERS})')
    ax.plot([], [], ':', color='white', label='fixed resonances')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Driven Speed (rad/s)')
    ax.set_title(f'{stem}: Torque Harmonics vs Speed (Campbell) @ {torque:g} Nm')
    ax.set_xlim(0, fmax)
    ax.legend(loc='upper right', facecolor='gray')
    fig.tight_layout()
    tag = f'{torque:g}Nm'
    fig.savefig(os.path.join(out_dir, f'{name}_harmonics_{tag}.png'), dpi=200)
    plt.close(fig)
    _write_wide_csv(os.path.join(out_dir, f'{name}_harmonics_{tag}.csv'),
                    'frequency_hz', centers, [(f'{spd:g}rad_s', M[i]) for i, spd in enumerate(speeds)])
    print(f'  Saved: {name}_harmonics_{tag}.png / .csv  (resonances: {[round(r) for r in res_freqs]} Hz)')
    return dominant


def analyze_cogging(filepath):
    stem = os.path.splitext(os.path.basename(filepath))[0]
    name = stem  # output prefix, e.g. 'cogging_5s'
    out_dir = test_dir(name)
    pp = COGGING_POLE_PAIRS
    sign = COGGING_DIRECTION
    print(f'Processing COGGING: {os.path.basename(filepath)}')

    with h5py.File(filepath, 'r') as f:
        start, end = behavior_span(f)
        pos = np.asarray(f['dut_output_position'][start:end], dtype=float)  # rad, continuous
        vel = np.asarray(f['load_velocity'][start:end], dtype=float)        # rad/s
        tq = np.asarray(f['load_torque'][start:end], dtype=float)           # Nm (output)
        vc = np.asarray(f['load_velocity_command'][start:end], dtype=float)
        tc = np.asarray(f['dut_torque_command'][start:end], dtype=float)
        t = np.asarray(f['time'][start:end], dtype=float)

    fs = 1.0 / np.median(np.diff(t))  # sample rate for spectra
    segs = _cogging_segments(vc, tc, t)
    torque_levels = sorted({sg['torque'] for sg in segs})
    speeds = sorted({abs(sg['v']) for sg in segs if abs(sg['v']) > 1e-6})
    slowest = speeds[0]
    cmap = plt.cm.viridis

    # --- Phase-alignment offset from the purest cogging (slowest, 0 Nm) -----
    p0, t0 = _cogging_trace(pos, vel, tq, segs, sign * slowest, torque_levels[0])
    phase_raw = (p0 / (2 * np.pi / pp)) % 1.0
    c0, m0, _, _ = _fold(phase_raw, t0, COGGING_LOCAL_BINS)
    m0 -= np.nanmean(m0)
    offset_frac = float(c0[np.nanargmax(m0)])  # this phase -> 0 (peak at 0/100%)

    summary = {
        'source_file': os.path.basename(filepath),
        'analysis': 'Velocity-controlled cogging: global, local (phase-folded), and speed dependency',
        'pole_pairs': pp,
        'direction': '+v' if sign > 0 else '-v',
        'torque_levels_Nm': torque_levels,
        'speeds_rad_s': speeds,
        'slowest_speed_rad_s': slowest,
        'phase_peak_offset_frac': offset_frac,
    }

    # ---- 1) GLOBAL: torque vs mechanical angle at the slowest speed ---------
    fig, ax = plt.subplots(figsize=(14, 8))
    cols = []
    for ti, torque in enumerate(torque_levels):
        p, q = _cogging_trace(pos, vel, tq, segs, sign * slowest, torque)
        frac = (p % (2 * np.pi)) / (2 * np.pi)
        c, m, sd, _ = _fold(frac, q, COGGING_GLOBAL_BINS)
        color = cmap(ti / max(len(torque_levels) - 1, 1))
        ax.plot(c * 360, m, color=color, linewidth=1.3, label=f'{torque:g} Nm')
        ax.fill_between(c * 360, m - sd, m + sd, color=color, alpha=0.2)
        cols += [(f'{torque:g}Nm_mean', m), (f'{torque:g}Nm_std', sd)]
    ax.set_xlabel('Mechanical Angle (deg, position % 2π)')
    ax.set_ylabel('Output Torque (Nm)')
    ax.set_title(f'{stem}: Cogging over Full Rotation @ {slowest:g} rad/s (slowest)')
    ax.grid(True, alpha=0.4)
    ax.legend(title='Torque level')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_global.png'), dpi=200)
    plt.close(fig)
    _write_wide_csv(os.path.join(out_dir, f'{name}_global.csv'),
                    'angle_deg', c * 360, cols)
    print(f'  Saved: {name}_global.png / .csv')

    # ---- 2) LOCAL: phase-folded, mean-subtracted cogging @ slowest speed ----
    fig, ax = plt.subplots(figsize=(12, 8))
    cols = []
    pkpk = {}
    for ti, torque in enumerate(torque_levels):
        p, q = _cogging_trace(pos, vel, tq, segs, sign * slowest, torque)
        phase = ((p / (2 * np.pi / pp)) - offset_frac) % 1.0
        c, m, sd, _ = _fold(phase, q, COGGING_LOCAL_BINS)
        m = m - np.nanmean(m)  # isolate cogging ripple
        color = cmap(ti / max(len(torque_levels) - 1, 1))
        ax.plot(c * 100, m, color=color, linewidth=1.6, label=f'{torque:g} Nm')
        ax.fill_between(c * 100, m - sd, m + sd, color=color, alpha=0.15)
        cols += [(f'{torque:g}Nm_mean', m), (f'{torque:g}Nm_std', sd)]
        pkpk[f'cogging_pkpk_{torque:g}Nm'] = float(np.nanmax(m) - np.nanmin(m))
    ax.set_xlabel('Phase Angle (%, one pole-pair period, peak @ 0/100%)')
    ax.set_ylabel('Cogging Torque (Nm, mean-subtracted)')
    ax.set_title(f'{stem}: Averaged Cogging vs Phase @ {slowest:g} rad/s')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.grid(True, alpha=0.4)
    ax.legend(title='Torque level')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_local.png'), dpi=200)
    plt.close(fig)
    _write_wide_csv(os.path.join(out_dir, f'{name}_local.csv'),
                    'phase_percent', c * 100, cols)
    summary.update(pkpk)
    print(f'  Saved: {name}_local.png / .csv')

    # ---- 3) SPEED DEPENDENCY: local view per speed, low & high torque ------
    for torque in (torque_levels[0], torque_levels[-1]):
        fig, ax = plt.subplots(figsize=(12, 8))
        cols = []
        for si, spd in enumerate(speeds):
            p, q = _cogging_trace(pos, vel, tq, segs, sign * spd, torque)
            if p is None or p.size == 0:
                continue
            phase = ((p / (2 * np.pi / pp)) - offset_frac) % 1.0
            c, m, sd, _ = _fold(phase, q, COGGING_LOCAL_BINS)
            m = m - np.nanmean(m)
            color = cmap(si / max(len(speeds) - 1, 1))
            ax.plot(c * 100, m, color=color, linewidth=1.4, label=f'{spd:g} rad/s')
            cols += [(f'{spd:g}rad_s_mean', m)]
        ax.set_xlabel('Phase Angle (%, one pole-pair period, peak @ 0/100%)')
        ax.set_ylabel('Cogging Torque (Nm, mean-subtracted)')
        ax.set_title(f'{stem}: Cogging vs Speed @ {torque:g} Nm')
        ax.axhline(0, color='k', linewidth=0.5)
        ax.grid(True, alpha=0.4)
        ax.legend(title='Speed', ncol=2)
        fig.tight_layout()
        tag = f'{torque:g}Nm'
        fig.savefig(os.path.join(out_dir, f'{name}_speed_dep_{tag}.png'), dpi=200)
        plt.close(fig)
        _write_wide_csv(os.path.join(out_dir, f'{name}_speed_dep_{tag}.csv'),
                        'phase_percent', c * 100, cols)
        print(f'  Saved: {name}_speed_dep_{tag}.png / .csv')

    # ---- 4) HARMONICS: Campbell diagram for lowest & highest torque ---------
    res_hz = None
    for torque in (torque_levels[0], torque_levels[-1]):
        rh = cogging_harmonics(name, stem, out_dir, segs, pos, vel, tq, speeds, sign, pp, fs, torque)
        if torque == torque_levels[0]:
            res_hz = rh
    summary['resonance_hz'] = res_hz

    return write_summary(name, summary, out_dir)


# ---------------------------------------------------------------------------
# File routing (by filename) + driver
# ---------------------------------------------------------------------------

def _norm(filename):
    """Normalize a filename for matching: lowercase, spaces->underscores."""
    return filename.lower().replace(' ', '_')


def find_file(*needles):
    """Return the path of the first .hdf5 whose normalized name contains all needles."""
    for fn in sorted(os.listdir(MOTOR_TESTS_DIR)):
        if not fn.endswith('.hdf5'):
            continue
        norm = _norm(fn)
        if all(n in norm for n in needles):
            return os.path.join(MOTOR_TESTS_DIR, fn)
    return None


def main():
    reset_results_dir()
    combined = {}

    tr_path = find_file('torque', 'response')
    if tr_path:
        combined['torque_response'] = analyze_torque_response(tr_path)
    else:
        print('No torque-response file found.')

    # Running torque: using rt8.hdf5 for now ('running torque final.hdf5' is the
    # same CSV-ENCODER test type if you want to switch the needle later).
    rt_path = find_file('rt8')
    if rt_path:
        combined['running_torque'] = analyze_running_torque(rt_path)
    else:
        print('No running-torque file found.')

    # Cogging: run on each configured file (cogging_5s, cogging_10s).
    for stem in COGGING_FILES:
        cog_path = find_file(stem)
        if cog_path:
            combined[stem] = analyze_cogging(cog_path)
        else:
            print(f'No cogging file found for {stem}.')

    # Combined roll-up across all analyses run this pass.
    if combined:
        with open(os.path.join(RESULTS_DIR, 'motor_tests_report.json'), 'w') as jf:
            json.dump(combined, jf, indent=2, default=str)
        with open(os.path.join(RESULTS_DIR, 'motor_tests_report.txt'), 'w') as tf:
            tf.write('Motor Tests — Combined Report\n')
            tf.write('=' * 40 + '\n\n')
            for test, summary in combined.items():
                tf.write(f'[{test}]\n')
                for k, v in summary.items():
                    if isinstance(v, float):
                        tf.write(f'  {k:30s}: {v:.6g}\n')
                    else:
                        tf.write(f'  {k:30s}: {v}\n')
                tf.write('\n')
        print('Saved combined report: motor_tests_report.txt / .json')

    print('Done.')


if __name__ == '__main__':
    main()
