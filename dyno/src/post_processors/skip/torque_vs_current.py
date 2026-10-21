"""Torque-vs-current characterization for the Elmo-driven, geared, backlashy actuator.

Written for the held-torque test (skip_torque_holds_samePositionDiffRamps): the
DUT motor is in torque mode and steps +/- through zero between held setpoints,
while LOAD holds position with its position loop. The generic current_torque
analyzer assumes a stationary rotor turns current into torque through one
constant, and this actuator breaks that in four ways, so this keeps its own
analysis rather than bending that one:

  1. the cell sits behind a 26:1 gear train with backlash and a torsional
     spring, so it reads the gear train's DEFLECTION, not the motor's torque;
  2. LOAD's position loop yields a few milliradians against that spring and then
     integrates back over seconds, so torque keeps moving at constant current;
  3. the input shaft stops where static friction catches it, so the same current
     holds different torques depending on how it was approached;
  4. the cell and the drive current have opposite signs on this rig.

What it produces (into <log dir>/torque_vs_current/ unless --out-dir):

  01_torque_vs_current.png      every sample, coloured ramp-up / hold / ramp-down
  02_ascending_curves.png       torque vs current ONLY while |I| is rising from
                                zero, one panel per kind of ramp
  03_slope_vs_rate.png          slope and secant of those curves per ramp
  04_hold_relaxation.png        what torque does after the current stops moving,
                                and how much of it is LOAD's position loop
  05_deflection.png             torque against gear-train deflection
  06_kt_by_definition.png       effective kt (output Nm per input A / gear ratio)
                                under each way of defining it
  07_gap_impact.png             what happens when a reversal closes the backlash
                                gap: LOAD kick, input speed, torque peak
  per_ramp.csv, per_hold.csv, per_reversal_impact.csv, results.json, report.txt

Conventions. The cell reads opposite to the drive current on this rig, so the
plots show the RESPONSE: torque is sign-flipped (auto-detected from the data)
so that more current in either direction gives a positive number. Current is
always shown as magnitude. 'kt_eff' is output torque per input amp divided by
the gear ratio read from the log's resolved config; it folds in gearbox
efficiency, so it is a lower bound on the motor's own kt.

Usage:
    .venv/bin/python dyno/src/post_processors/torque_vs_current.py <log dir|hdf5>
        [--out-dir DIR] [--gear-ratio N] [--torque-channel load_torque]
"""

import argparse
import csv
import glob
import json
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless: save figures, never block on display
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.optimize import curve_fit

# --- Tunables ---------------------------------------------------------------
COMMAND = 'dut_torque_command'
IN_VEL = 'dut_velocity'          # input shaft speed, rad/s
CURRENT = 'dut_current'
TORQUE = 'load_torque'
OUT_POS = 'load_position'        # output shaft, held by LOAD
IN_POS = 'dut_output_position'   # input shaft, via the Elmo encoder
RAMP_EPS = 1e-7                  # |d(command)| above this counts as ramping
RAMP_MERGE_GAP = 5               # merge ramp runs closer than this many samples
MIN_RAMP_SAMPLES = 3
SMOOTH_SAMPLES = 41              # moving average on torque/current (41 ms at 1 kHz)
FIT_LO_A, FIT_HI_A = 2.0, 10.0   # current window for the ascending slope fit
LEVELS_A = (1, 2, 4, 6, 8, 10, 12)
HOLD_SKIP_S = 0.25               # settle after a ramp ends before a hold is analysed
HOLD_END_GUARD_S = 0.05          # stop this long before the next ramp starts
MIN_HOLD_S = 5.0
PLOT_DECIMATE = 20

INK = '#0b0b0b'
INK_MUTED = '#52514e'
GRID = '#e4e3df'
C_UP, C_HOLD, C_DOWN = '#2f86d0', '#8a8a86', '#d9622b'
KIND_COLOR = {'virgin': '#1fa39a', 'reversal+': '#5c6fd6', 'reversal-': '#d9622b'}
RATE_CMAP = 'viridis'


def find_log(path):
    if os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, '*.hdf5')))
        if not hits:
            raise SystemExit(f'no .hdf5 in {path}')
        return hits[0]
    return path


def moving_average(x, n=SMOOTH_SAMPLES):
    return np.convolve(x, np.ones(n) / n, mode='same')


def find_ramps(cmd):
    """Contiguous runs where the command is changing -> [(start, end), ...]."""
    moving = np.abs(np.diff(cmd)) > RAMP_EPS
    edges = np.diff(np.r_[0, moving.astype(int), 0])
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    runs = []
    for s, e in zip(starts, ends):
        if runs and s - runs[-1][1] < RAMP_MERGE_GAP:
            runs[-1] = (runs[-1][0], e)
        else:
            runs.append((s, e))
    return [(s, e) for s, e in runs if e - s >= MIN_RAMP_SAMPLES]


def load(log, torque_channel):
    with h5py.File(log, 'r') as f:
        d = {k: f[k][:].astype(float) for k in (COMMAND, CURRENT, torque_channel, OUT_POS, IN_POS, IN_VEL)}
        d['t'] = f['time'][:].astype(float)
        d['t'] -= d['t'][0]
        gear = None
        try:
            cfg = json.loads(f.attrs['resolved_config'])
            gear = float(cfg['devices']['DUT']['params']['gear_ratio'])
        except (KeyError, TypeError, ValueError):
            pass
    d['torque'] = d.pop(torque_channel)
    d['gear'] = gear
    return d


def response_sign(d):
    """+1 if the cell rises with current, -1 if it falls (it does on this rig)."""
    m = np.abs(d[CURRENT]) > 1.0
    return -1.0 if np.corrcoef(d[CURRENT][m], d['torque'][m])[0, 1] < 0 else 1.0


def classify_ramps(d):
    """One dict per ramp with its kind and the index range of its ascending-|I| part.

    virgin     starts from zero command: the first loading after the actuator
               has sat unloaded
    reversal+  swings through zero and ends at +setpoint
    reversal-  swings through zero and ends at -setpoint
    return     ends at zero: only unloads, so it has no ascending part

    The ascending part is the stretch where current magnitude is climbing away
    from zero: from the instant the measured current takes the sign of the
    ramp's destination (reversals) or from the ramp start (virgin) to its end.
    """
    cmd, I = d[COMMAND], d['I_s']
    ramps = []
    for s, e in find_ramps(cmd):
        c0, c1 = cmd[s], cmd[e]
        rate = abs(c1 - c0) / (d['t'][e] - d['t'][s])
        if abs(c1) < 1e-6:
            ramps.append(dict(s=s, e=e, kind='return', rate=rate, sgn=0.0))
            continue
        sgn = float(np.sign(c1))
        if abs(c0) < 1e-6:
            kind, a = 'virgin', s
        else:
            kind = 'reversal+' if sgn > 0 else 'reversal-'
            seg = np.arange(s, e + 1)
            a = seg[np.flatnonzero(np.sign(I[seg]) == sgn)[0]]
        ramps.append(dict(s=s, e=e, a=a, kind=kind, rate=round(rate, 4), sgn=sgn))
    return ramps


def ascending_curve(d, r):
    """(|I|, response torque) smoothed over a ramp's ascending part."""
    sl = slice(r['a'], r['e'] + 1)
    x = r['sgn'] * d['I_s'][sl]
    # response torque: positive when more current gives more torque
    y = r['sgn'] * d['rsign'] * d['T_s'][sl]
    return x, y


def ramp_metrics(d, r, gear):
    x, y = ascending_curve(d, r)
    m = (x >= FIT_LO_A) & (x <= FIT_HI_A)
    slope, icpt = np.polyfit(x[m], y[m], 1)
    mono = np.maximum.accumulate(x)
    at = {L: float(np.interp(L, mono, y)) if mono[-1] >= L else np.nan for L in LEVELS_A}
    out = dict(kind=r['kind'], rate=r['rate'], t_start=float(d['t'][r['s']]),
               i_end=float(x[-1]), t_end=float(y[-1]),
               slope_nm_per_a=float(slope), intercept_nm=float(icpt),
               i_at_zero_torque=float(-icpt / slope) if slope else np.nan,
               secant_nm_per_a=float(y[-1] / x[-1]),
               kt_eff_slope=float(slope / gear), kt_eff_secant=float(y[-1] / x[-1] / gear))
    out.update({f't_at_{L}A': v for L, v in at.items()})
    return out


def hold_windows(d, ramps):
    """[(ramp, a, b)] sample ranges of each hold that follows a loading ramp."""
    out = []
    starts = [r['s'] for r in ramps]
    dt = float(np.median(np.diff(d['t'])))
    for k, r in enumerate(ramps):
        if r['kind'] == 'return' or k + 1 >= len(ramps):
            continue
        a = r['e'] + int(HOLD_SKIP_S / dt)
        b = starts[k + 1] - int(HOLD_END_GUARD_S / dt)
        if (b - a) * dt >= MIN_HOLD_S:
            out.append((r, a, b))
    return out


def expo(t, tinf, amp, tau):
    return tinf + amp * np.exp(-t / tau)


def hold_metrics(d, r, a, b, lp0, gear):
    t = d['t'][a:b] - d['t'][a]
    y = r['sgn'] * d['rsign'] * d['torque'][a:b]
    i_hold = float(np.mean(d[CURRENT][a:b]))   # signed: the up/down pairing needs the sign
    o = dict(kind=r['kind'], rate=r['rate'], t_start=float(d['t'][a]), i_hold=i_hold,
             t_after_ramp_0p25=float(np.mean(y[:250])))
    for tag, sec in (('1s', 1.0), ('5s', 5.0), ('19s', 19.0)):
        j = min(int(sec * 1000), len(y) - 1)
        o[f't_after_ramp_{tag}'] = float(np.mean(y[max(j - 50, 0):j + 50]))
    ds = slice(None, None, 20)
    try:
        (tinf, amp, tau), _ = curve_fit(expo, t[ds], y[ds], p0=(y[-1], y[0] - y[-1], 4.0),
                                        bounds=([-np.inf, -np.inf, 0.2], [np.inf, np.inf, 40.0]))
    except (RuntimeError, ValueError):
        tinf, amp, tau = np.nan, np.nan, np.nan
    o.update(t_inf=float(tinf), amp=float(amp), tau_s=float(tau))
    # How much of the drift is LOAD's position loop pulling the output back
    # against the gear train's spring: raw torque vs output displacement.
    x = d[OUT_POS][a:b] - lp0
    if np.ptp(x) > 2e-4:
        k, c = np.polyfit(x, d['torque'][a:b], 1)
        o.update(k_out_nm_per_rad=float(k), k_out_r=float(np.corrcoef(x, d['torque'][a:b])[0, 1]),
                 lp_excursion_rad=float(np.ptp(x)))
    else:
        o.update(k_out_nm_per_rad=np.nan, k_out_r=np.nan, lp_excursion_rad=float(np.ptp(x)))
    o['in_pos_rad'] = float(np.mean(d[IN_POS][b - 500:b]))
    o['out_pos_rad'] = float(np.mean(d[OUT_POS][b - 500:b]) - lp0)
    o['t_raw_end'] = float(np.mean(d['torque'][b - 500:b]))
    o['kt_eff_plateau'] = float(abs(np.mean(d['torque'][b - 500:b])) / abs(np.mean(d[CURRENT][b - 500:b])) / gear)
    return o


# --- plotting ---------------------------------------------------------------

def style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID, lw=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    if title:
        ax.set_title(title, color=INK, loc='left', fontsize=11)


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def fig_overview(d, ramps, out_dir, gear):
    dec = slice(None, None, PLOT_DECIMATE)
    I = np.abs(d[CURRENT])
    y = d['rsign'] * np.sign(d[CURRENT]) * d['torque']
    # Phase comes from the command ramps, not from the noisy measured current:
    # rising = the ascending part of a loading ramp, falling = the unloading part
    # of a ramp (before the sign change on a reversal), hold = everything else.
    phase = np.zeros(len(I), dtype=int)
    for r in ramps:
        if r['kind'] == 'return':
            phase[r['s']:r['e'] + 1] = -1
        else:
            phase[r['s']:r['a']] = -1
            phase[r['a']:r['e'] + 1] = 1
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    for ph, col, lab in ((0, C_HOLD, '|I| steady (holds)'), (-1, C_DOWN, '|I| falling'), (1, C_UP, '|I| rising')):
        m = np.flatnonzero(phase == ph)[::PLOT_DECIMATE]
        ax.scatter(I[m], y[m], s=3, color=col, alpha=0.35, label=lab, linewidths=0)
    ax.set_xlim(-0.3, 14)
    style(ax, '|DUT current| [A]', 'response torque at the cell [Nm]\n(sign-flipped: more current = more torque)',
          'Torque is not a function of current: every sample, by what the current is doing')
    ax.legend(markerscale=4, frameon=False, loc='upper left')
    return save(fig, out_dir, '01_torque_vs_current.png')


def fig_ascending(d, ramps, metrics, out_dir):
    kinds = [('virgin', 'First loading from rest (0 -> +setpoint)'),
             ('reversal+', 'After a reversal, current rising toward +'),
             ('reversal-', 'After a reversal, current rising toward -')]
    rates = sorted({r['rate'] for r in ramps})
    norm = Normalize(min(rates), max(rates))
    cmap = plt.get_cmap(RATE_CMAP)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    for ax, (kind, title) in zip(axes, kinds):
        for r in ramps:
            if r['kind'] != kind:
                continue
            x, y = ascending_curve(d, r)
            ax.plot(x[::10], y[::10], color=cmap(norm(r['rate'])), lw=1.1, alpha=0.9)
        ms = [m for m in metrics if m['kind'] == kind]
        sl = np.nanmean([m['slope_nm_per_a'] for m in ms])
        ic = np.nanmean([m['intercept_nm'] for m in ms])
        xs = np.array([FIT_LO_A, FIT_HI_A])
        ax.plot(xs, sl * xs + ic, color=INK, lw=1.6, ls='--',
                label=f'mean fit {FIT_LO_A:.0f}-{FIT_HI_A:.0f} A: {sl:.2f} Nm/A')
        ax.legend(frameon=False, loc='upper left', fontsize=9)
        style(ax, '|DUT current| [A]', 'response torque [Nm]' if ax is axes[0] else '', title)
    cb = fig.colorbar(ScalarMappable(norm, cmap), ax=axes, pad=0.01, fraction=0.025)
    cb.set_label('command ramp rate [Nm/s]', color=INK)
    fig.suptitle('Torque vs current, only while |I| is climbing away from zero', x=0.06, ha='left', color=INK)
    return save(fig, out_dir, '02_ascending_curves.png')


def fig_slope_rate(metrics, out_dir, gear):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(wspace=0.4))
    for kind, col in KIND_COLOR.items():
        ms = sorted((m for m in metrics if m['kind'] == kind), key=lambda m: m['rate'])
        if not ms:
            continue
        rt = [m['rate'] for m in ms]
        axes[0].plot(rt, [m['slope_nm_per_a'] for m in ms], 'o-', color=col, ms=4, lw=1, label=kind)
        axes[1].plot(rt, [m['secant_nm_per_a'] for m in ms], 'o-', color=col, ms=4, lw=1, label=kind)
    style(axes[0], 'command ramp rate [Nm/s]', 'slope dT/dI [Nm/A]', f'Slope over {FIT_LO_A:.0f}-{FIT_HI_A:.0f} A')
    style(axes[1], 'command ramp rate [Nm/s]', 'T / I at end of ramp [Nm/A]', 'Secant at the end of the ramp (~12.5 A)')
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        sec = ax.secondary_yaxis('right', functions=(lambda v: v / gear, lambda v: v * gear))
        sec.set_ylabel(f'kt_eff = / {gear:g}  [Nm/A]', color=INK_MUTED)
        sec.tick_params(colors=INK_MUTED)
    return save(fig, out_dir, '03_slope_vs_rate.png')


def fig_hold(d, ramps, holds, hold_rows, out_dir):
    rates = sorted({r['rate'] for r, _, _ in holds})
    norm = Normalize(min(rates), max(rates))
    cmap = plt.get_cmap(RATE_CMAP)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw=dict(wspace=0.38))
    # (a) response torque after the ramp ends, every positive-going first hold
    for (r, a, b), row in zip(holds, hold_rows):
        if r['kind'] != 'virgin':
            continue
        tt = d['t'][a:b] - d['t'][a]
        y = r['sgn'] * d['rsign'] * d['torque'][a:b]
        axes[0].plot(tt[::20], y[::20], color=cmap(norm(r['rate'])), lw=1.1)
    style(axes[0], 'time since the current stopped ramping [s]', 'response torque [Nm]',
          'Constant current, torque keeps moving\n(first hold of each rate)')
    # (b) one representative hold, torque against LOAD's output displacement
    rep = next(((r, a, b) for r, a, b in holds if r['kind'] == 'virgin' and abs(r['rate'] - 0.05) < 1e-6), holds[0])
    r, a, b = rep
    tt = d['t'][a:b] - d['t'][a]
    ax = axes[1]
    ax.plot(tt[::20], r['sgn'] * d['rsign'] * d['torque'][a:b][::20], color=INK, lw=1.4, label='cell torque (response)')
    style(ax, 'time since the current stopped ramping [s]', 'response torque [Nm]',
          f'Same hold, with LOAD\'s output position (rate {r["rate"]})')
    ax2 = ax.twinx()
    ax2.plot(tt[::20], (d[OUT_POS][a:b] - np.median(d[OUT_POS][:1500]))[::20] * 1e3, color=C_DOWN, lw=1.4,
             label='LOAD position error')
    ax2.set_ylabel('LOAD position error [mrad]', color=C_DOWN)
    ax2.tick_params(colors=C_DOWN)
    ax2.spines['top'].set_visible(False)
    # (c) all holds: torque vs output displacement
    allx, ally = [], []
    for (r, a, b) in holds:
        lp0 = np.median(d[OUT_POS][:1500])
        x = (d[OUT_POS][a:b] - lp0)[::20] * 1e3
        y = (d['torque'][a:b])[::20]
        axes[2].scatter(x, y, s=2, alpha=0.25, color=KIND_COLOR.get(r['kind'], INK), linewidths=0)
    ks = [h['k_out_nm_per_rad'] for h in hold_rows if np.isfinite(h['k_out_nm_per_rad']) and abs(h['k_out_r']) > 0.9]
    style(axes[2], 'LOAD output displacement from its hold target [mrad]', 'raw cell torque [Nm]',
          f'Torque vs output displacement within holds\nmedian slope {np.median(ks) / 1e3:.2f} Nm/mrad  '
          f'({len(ks)} of {len(hold_rows)} holds with |r|>0.9)')
    return save(fig, out_dir, '04_hold_relaxation.png'), (float(np.median(ks)) if ks else np.nan, len(ks))


def deflection_fit(d, holds, gear):
    """Linear gear-train spring on each side of the backlash gap.

    Deflection (rad at the output) = input angle / N - output angle, both zeroed
    at the start of the run. Torque is fitted against it separately for the two
    loading directions; where each line reaches zero torque marks one edge of the
    backlash gap, so the distance between the two zero crossings is the gap.
    """
    lp0 = np.median(d[OUT_POS][:1500])
    ip0 = np.median(d[IN_POS][:1500])
    xs, ys = [], []
    for r, a, b in holds:
        xs.append(((d[IN_POS][a:b] - ip0) / gear - (d[OUT_POS][a:b] - lp0))[::10])
        ys.append(d['torque'][a:b][::10])
    x, y = np.concatenate(xs), np.concatenate(ys)
    out = {}
    for name, m in (('positive_cell_torque', y > 0), ('negative_cell_torque', y < 0)):
        k, c = np.polyfit(x[m], y[m], 1)
        out[name] = dict(stiffness_nm_per_rad=float(k), zero_torque_deflection_rad=float(-c / k),
                         r=float(np.corrcoef(x[m], y[m])[0, 1]), n=int(m.sum()))
    out['backlash_gap_rad_at_output'] = float(abs(out['positive_cell_torque']['zero_torque_deflection_rad'] -
                                                  out['negative_cell_torque']['zero_torque_deflection_rad']))
    return out


def fig_deflection(d, holds, hold_rows, gear, out_dir, fit):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    rates = sorted({h['rate'] for h in hold_rows})
    norm = Normalize(min(rates), max(rates))
    cmap = plt.get_cmap(RATE_CMAP)
    lp0 = np.median(d[OUT_POS][:1500])
    ip0 = np.median(d[IN_POS][:1500])
    for (r, a, b), h in zip(holds, hold_rows):
        defl = ((d[IN_POS][a:b] - ip0) / gear - (d[OUT_POS][a:b] - lp0))[::50] * 1e3   # mrad at the output
        axes[0].scatter(defl, d['torque'][a:b][::50], s=2, alpha=0.5, color=cmap(norm(r['rate'])), linewidths=0)
        axes[1].scatter(np.abs(h['i_hold']), abs(h['t_raw_end']), s=28, color=cmap(norm(r['rate'])),
                        marker='o' if r['sgn'] > 0 else '^', edgecolor='none')
    for key, col in (('positive_cell_torque', INK), ('negative_cell_torque', INK)):
        f_ = fit[key]
        xz = f_['zero_torque_deflection_rad'] * 1e3
        xx = np.array([xz, xz + (18 if key.startswith('pos') else -18) / (f_['stiffness_nm_per_rad'] / 1e3)])
        axes[0].plot(xx, f_['stiffness_nm_per_rad'] / 1e3 * (xx - xz), color=col, lw=1.2, ls='--')
    axes[0].axhline(0, color=INK_MUTED, lw=0.6)
    axes[0].annotate(f"backlash gap ~ {fit['backlash_gap_rad_at_output'] * 1e3:.1f} mrad at the output\n"
                     f"({np.degrees(fit['backlash_gap_rad_at_output']):.2f} deg, {np.degrees(fit['backlash_gap_rad_at_output']) * gear:.0f} deg at the input)\n"
                     f"spring {fit['positive_cell_torque']['stiffness_nm_per_rad'] / 1e3:.2f} / "
                     f"{fit['negative_cell_torque']['stiffness_nm_per_rad'] / 1e3:.2f} Nm/mrad (+/- side)",
                     (0.03, 0.97), xycoords='axes fraction', va='top', fontsize=9, color=INK)
    style(axes[0], 'gear-train deflection  = input angle / N - output angle  [mrad at the output]',
          'raw cell torque [Nm]', 'The cell reads the gear train\'s deflection\n(each dot cloud is one hold)')
    style(axes[1], '|current| during the hold [A]', 'settled |torque| at the end of the hold [Nm]',
          'Same current, different settled torque\n(circle = + hold, triangle = - hold; colour = ramp rate)')
    cb = fig.colorbar(ScalarMappable(norm, cmap), ax=axes, pad=0.01, fraction=0.02)
    cb.set_label('command ramp rate [Nm/s]', color=INK)
    return save(fig, out_dir, '05_deflection.png')


def kt_definitions(metrics, hold_rows, gear):
    """Effective kt (Nm/A at the output / gear ratio) under each definition."""
    def agg(vals):
        v = np.array([x for x in vals if np.isfinite(x)])
        return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    cv_pct=float(100 * v.std(ddof=1) / v.mean()) if len(v) > 1 else 0.0,
                    lo=float(v.min()), hi=float(v.max()), n=int(len(v)))
    rows = {}
    for kind in ('virgin', 'reversal+', 'reversal-'):
        ms = [m for m in metrics if m['kind'] == kind]
        rows[f'ascending slope, {kind}'] = agg([m['kt_eff_slope'] for m in ms])
        rows[f'ascending secant at ramp end, {kind}'] = agg([m['kt_eff_secant'] for m in ms])
    rows['settled hold plateau, all holds'] = agg([h['kt_eff_plateau'] for h in hold_rows])
    # Up/down midline: pair each + hold with the - hold that follows it.
    pairs = []
    for k in range(len(hold_rows) - 1):
        a_, b_ = hold_rows[k], hold_rows[k + 1]
        if a_['i_hold'] * b_['i_hold'] < 0 and abs(a_['rate'] - b_['rate']) < 1e-6:
            pairs.append(abs(abs(a_['t_raw_end']) + abs(b_['t_raw_end'])) /
                         (abs(a_['i_hold']) + abs(b_['i_hold'])) / gear)
    rows['settled midline (+ and - holds paired)'] = agg(pairs)
    return rows


def fig_kt(rows, out_dir):
    names = list(rows)
    fig, ax = plt.subplots(1, 1, figsize=(10, 0.55 * len(names) + 1.6))
    ys = np.arange(len(names))[::-1]
    for y, n in zip(ys, names):
        r = rows[n]
        ax.plot([r['lo'], r['hi']], [y, y], color=INK_MUTED, lw=2, alpha=0.5)
        ax.plot(r['mean'], y, 'o', color=INK, ms=6)
        ax.text(r['hi'] + 0.0007, y, f"{r['mean']:.4f}   CV {r['cv_pct']:.0f}%  (n={r['n']})", va='center',
                fontsize=8.5, color=INK_MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, color=INK)
    ax.set_xlim(left=0)
    xmax = max(r['hi'] for r in rows.values())
    ax.set_xlim(0, xmax * 1.6)
    style(ax, 'kt_eff = output Nm per input A / gear ratio  [Nm/A]', '',
          'One actuator, several different "kt" values depending on the definition - bars span min..max')
    ax.grid(False, axis='y')
    return save(fig, out_dir, '06_kt_by_definition.png')


IMPACT_WINDOW_S = 6.0            # after a ramp ends, how long to look for the torque peak


def impact_metrics(d, ramps):
    """Gap-crossing transient of each reversal: LOAD kick, input speed, torque peak.

    All times are relative to the end of the command ramp. 'kick' is LOAD's peak
    position error (mrad) between the start of the ascending part and
    IMPACT_WINDOW_S after the ramp: what the input shaft's arrival does to the
    held output.
    """
    lp0 = np.median(d[OUT_POS][:1500])
    t = d['t']
    rows = []
    for k, r in enumerate(ramps):
        if not r['kind'].startswith('reversal'):
            continue
        nxt = ramps[k + 1]['s'] if k + 1 < len(ramps) else len(t) - 1
        w = slice(r['a'], min(r['e'] + int(IMPACT_WINDOW_S * 1000), nxt))
        err = (d[OUT_POS][w] - lp0) * 1e3
        v = d[IN_VEL][w]
        resp = r['sgn'] * d['rsign'] * d['torque'][w]
        i_err, i_v, i_t = np.argmax(np.abs(err)), np.argmax(np.abs(v)), np.argmax(resp)
        te = t[r['e']]
        rows.append(dict(kind=r['kind'], rate=r['rate'],
                         kick_mrad=float(abs(err[i_err])), kick_t_rel_ramp_end_s=float(t[w][i_err] - te),
                         peak_input_speed_rad_s=float(abs(v[i_v])), peak_speed_t_rel_ramp_end_s=float(t[w][i_v] - te),
                         peak_torque_nm=float(resp[i_t]), peak_torque_t_rel_ramp_end_s=float(t[w][i_t] - te),
                         load_error_at_torque_peak_mrad=float(err[i_t])))
    return rows


def fig_impact(d, ramps, rows, out_dir):
    pick = [0.05, 0.15, 0.30]
    cmap = plt.get_cmap(RATE_CMAP)
    norm = Normalize(0.05, 0.30)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), gridspec_kw=dict(wspace=0.32, hspace=0.3))
    lp0 = np.median(d[OUT_POS][:1500])
    # (0,0) current  (0,1) input speed  (0,2) summary: kick and speed vs rate
    # (1,0) LOAD position error  (1,1) cell torque  (1,2) deflection
    ip0 = np.median(d[IN_POS][:1500])
    chosen = []
    for rate in pick:
        r = next((q for q in ramps if q['kind'] == 'reversal-' and abs(q['rate'] - rate) < 1e-6), None)
        if r is not None:
            chosen.append(r)
    for r in chosen:
        a, b = r['s'] - 500, r['e'] + 4000
        tt = d['t'][a:b] - d['t'][r['e']]
        col = cmap(norm(r['rate']))
        lab = f"{r['rate']:g} Nm/s"
        axes[0, 0].plot(tt, np.abs(d['dut_current'][a:b]) * np.sign(r['sgn']) * r['sgn'] * np.sign(d['dut_current'][a:b]), color=col, lw=1.2, label=lab)
        axes[0, 1].plot(tt, d[IN_VEL][a:b], color=col, lw=1.0, label=lab)
        axes[1, 0].plot(tt, (d[OUT_POS][a:b] - lp0) * 1e3, color=col, lw=1.2, label=lab)
        axes[1, 1].plot(tt, r['sgn'] * d['rsign'] * d['torque'][a:b], color=col, lw=1.0, label=lab)
        axes[1, 2].plot(tt, ((d[IN_POS][a:b] - ip0) / 26 - (d[OUT_POS][a:b] - lp0)) * 1e3, color=col, lw=1.2, label=lab)
    style(axes[0, 0], 'time relative to the end of the command ramp [s]', 'signed DUT current [A]', 'Current ramp (reversal, toward -)')
    style(axes[0, 1], 'time relative to the end of the command ramp [s]', 'input shaft speed [rad/s]', 'Input speed: the rotor crossing the gap')
    style(axes[1, 0], 'time relative to the end of the command ramp [s]', 'LOAD position error [mrad]', 'LOAD is kicked, then recovers')
    style(axes[1, 1], 'time relative to the end of the command ramp [s]', 'response torque [Nm]', 'Cell torque peaks after the kick')
    style(axes[1, 2], 'time relative to the end of the command ramp [s]', 'gear-train deflection [mrad at output]', 'Deflection')
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2]):
        ax.legend(frameon=False, fontsize=8)
        ax.axvline(0, color=INK_MUTED, lw=0.6, ls=':')
    ax = axes[0, 2]
    for kind, mk in (('reversal+', '^'), ('reversal-', 'o')):
        rr = sorted((x for x in rows if x['kind'] == kind), key=lambda x: x['rate'])
        ax.plot([x['rate'] for x in rr], [x['kick_mrad'] for x in rr], mk + '-', color=C_DOWN, ms=4, lw=0.8,
                label=f'LOAD kick, {kind}')
        ax.plot([x['rate'] for x in rr], [x['peak_input_speed_rad_s'] for x in rr], mk + '-', color=C_UP, ms=4, lw=0.8,
                label=f'peak input speed [rad/s], {kind}')
    style(ax, 'command ramp rate [Nm/s]', 'LOAD kick [mrad]  /  input speed [rad/s]', 'Kick grows with ramp rate; speed does not')
    ax.legend(frameon=False, fontsize=7)
    return save(fig, out_dir, '07_gap_impact.png')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('log', help='log folder or .hdf5 file')
    p.add_argument('--out-dir', help='output folder (default: <log dir>/torque_vs_current)')
    p.add_argument('--gear-ratio', type=float, help='override the gear ratio read from the log')
    p.add_argument('--torque-channel', default=TORQUE)
    args = p.parse_args()

    log = find_log(args.log)
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(log)), 'torque_vs_current')
    os.makedirs(out_dir, exist_ok=True)

    d = load(log, args.torque_channel)
    gear = args.gear_ratio or d['gear']
    if gear is None:
        raise SystemExit('gear ratio not in the log config; pass --gear-ratio')
    d['rsign'] = response_sign(d)
    d['I_s'] = moving_average(d[CURRENT])
    d['T_s'] = moving_average(d['torque'])

    ramps = classify_ramps(d)
    loading = [r for r in ramps if r['kind'] != 'return']
    metrics = [ramp_metrics(d, r, gear) for r in loading]
    lp0 = float(np.median(d[OUT_POS][:1500]))
    holds = hold_windows(d, ramps)
    hold_rows = [hold_metrics(d, r, a, b, lp0, gear) for r, a, b in holds]

    figs = [fig_overview(d, ramps, out_dir, gear),
            fig_ascending(d, loading, metrics, out_dir),
            fig_slope_rate(metrics, out_dir, gear)]
    hold_fig, (k_med, k_n) = fig_hold(d, ramps, holds, hold_rows, out_dir)
    figs.append(hold_fig)
    dfit = deflection_fit(d, holds, gear)
    figs.append(fig_deflection(d, holds, hold_rows, gear, out_dir, dfit))
    rows = kt_definitions(metrics, hold_rows, gear)
    impact = impact_metrics(d, ramps)
    figs.append(fig_impact(d, ramps, impact, out_dir))
    write_csv(os.path.join(out_dir, 'per_reversal_impact.csv'), impact)
    figs.append(fig_kt(rows, out_dir))

    write_csv(os.path.join(out_dir, 'per_ramp.csv'), metrics)
    write_csv(os.path.join(out_dir, 'per_hold.csv'), hold_rows)
    taus = [h['tau_s'] for h in hold_rows if np.isfinite(h['tau_s']) and h['tau_s'] < 39]
    results = dict(log=os.path.abspath(log), gear_ratio=gear, response_sign=d['rsign'],
                   n_ramps=len(loading), n_holds=len(hold_rows),
                   kt_eff_by_definition=rows,
                   output_stiffness_nm_per_rad=dict(median=k_med, n_holds=k_n),
                   gear_train_deflection_fit=dfit,
                   hold_time_constant_s=dict(median=float(np.median(taus)), p10=float(np.percentile(taus, 10)),
                                             p90=float(np.percentile(taus, 90)), n=len(taus)))
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    lines = [f'Torque vs current: {os.path.basename(log)}',
             f'gear ratio {gear:g}, cell sign relative to current {d["rsign"]:+.0f}, '
             f'{len(loading)} loading ramps, {len(hold_rows)} holds', '',
             'kt_eff = output Nm per input A / gear ratio  (folds in gearbox efficiency)']
    for n, r in rows.items():
        lines.append(f'  {n:48s} {r["mean"]:.4f}  CV {r["cv_pct"]:4.1f}%  range {r["lo"]:.4f}..{r["hi"]:.4f}  n={r["n"]}')
    lines += ['', f'Hold time constant (exponential fit): median {results["hold_time_constant_s"]["median"]:.1f} s '
                  f'(10-90%: {results["hold_time_constant_s"]["p10"]:.1f}-{results["hold_time_constant_s"]["p90"]:.1f} s)',
              f'Torque change per output displacement while LOAD relaxes in a hold: {abs(k_med):.0f} Nm/rad ({k_n} holds, |r|>0.9)',
              f'Gear-train spring (torque vs deflection): '
              f'{dfit["positive_cell_torque"]["stiffness_nm_per_rad"]:.0f} Nm/rad (+ cell side), '
              f'{dfit["negative_cell_torque"]["stiffness_nm_per_rad"]:.0f} Nm/rad (- cell side)',
              f'Backlash gap from the two zero-torque deflections: {dfit["backlash_gap_rad_at_output"] * 1e3:.1f} mrad at the output '
              f'({np.degrees(dfit["backlash_gap_rad_at_output"]):.2f} deg)']
    report = '\n'.join(lines)
    with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
        f.write(report + '\n')
    print(report)
    print('\nwrote:')
    for fpath in figs + [os.path.join(out_dir, n) for n in ('per_ramp.csv', 'per_hold.csv', 'results.json', 'report.txt')]:
        print('  ' + fpath)


if __name__ == '__main__':
    main()
