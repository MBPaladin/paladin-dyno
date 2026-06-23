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
import json
import shutil

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
