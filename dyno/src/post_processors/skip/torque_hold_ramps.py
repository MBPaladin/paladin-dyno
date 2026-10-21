"""Overlay plot for torque-hold tests run at several ramp rates (rotatums).

Each segment in the log (SEG1-RUN0, SEG2-RUN0, ...) ramps the DUT torque command
between the same setpoints with the same hold durations, only at a different
ramp rate. This overlays every segment on one time axis, one colour per ramp
rate, so the output-torque peak and fall-off during each hold can be compared
directly.

Alignment ('--align'):
    zero   (default) one section per hold. The first section is plotted from
           segment start; every later section is shifted so the instant the
           command passes through zero (on the ramp into that hold) lands at
           the same x. Each section is cut just before the next ramp starts,
           with a fixed gap so every trace visibly breaks. The final return-
           to-zero ramp stays attached to the last section.
    torque_zero
           as zero, but anchored on the MEASURED torque's zero crossing
           (first crossing after each ramp starts, in the ramp's direction,
           on a 25 ms moving average). Written to *_torquezero.png.
    hold   piecewise: every hold is shifted so that the instant the
           command reaches its setpoint lands at the same x. Peaks line up; a
           fast segment shows a short gap before each ramp because its ramp is
           shorter than the slowest one's.
    start  raw time since segment start. Holds drift apart as the slower
           ramps accumulate.

Usage:
    .venv/bin/python dyno/src/post_processors/torque_hold_ramps.py <log dir|hdf5>
        [--current] [--align zero|torque_zero|hold|start] [--torque-channel load_torque]
        [--out file.png]

The figure is written next to the log unless --out is given.
"""

import argparse
import glob
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless: save figures, never block on display
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec

# --- Tunables ---------------------------------------------------------------
COMMAND_CHANNEL = 'dut_torque_command'
CURRENT_CHANNEL = 'dut_current'
PLOT_DECIMATE = 10          # block-mean this many samples per plotted point (1 kHz -> 100 Hz)
RAMP_EPS = 1e-7             # |d(command)| above this counts as ramping
RAMP_MERGE_GAP = 5          # merge ramp runs separated by fewer samples than this
MIN_RAMP_SAMPLES = 3
SECTION_GAP_S = 2.0         # blank space between sections in --align zero
TORQUE_SMOOTH_S = 0.025     # moving-average window for the measured-torque zero crossing

# Ramp rate is a magnitude, so lines are coloured on one ordered ramp:
# purple (slow) -> blue -> teal -> green (fast). The hue sweep separates
# neighbouring rates better than a single-hue ramp; lightness still rises
# gently (OKLab L 0.58 -> 0.70) so the order survives greyscale, and every
# stop clears 2.4:1 against the white surface.
RATE_RAMP = ['#8a5cd0', '#5c6fd6', '#2f86d0', '#1fa39a', '#3dbb6e']
LOG_SPAN = 10.0             # colour on a log scale when max/min rate exceeds this
MAX_CBAR_TICKS = 12         # beyond this, label every other tested rate
INK = '#0b0b0b'
INK_MUTED = '#52514e'
GRID = '#e4e3df'


def find_log(path):
    if os.path.isdir(path):
        hits = sorted(glob.glob(os.path.join(path, '*.hdf5')))
        if not hits:
            raise SystemExit(f'no .hdf5 in {path}')
        return hits[0]
    return path


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


def zero_crossing(t, cmd, s, e):
    """Time the command passes through (or lands on) zero inside ramp [s, e].

    Linear interpolation between the bracketing samples. A ramp that starts at
    zero has no crossing after its first sample and falls back to its end.
    """
    c = cmd[s:e + 1]
    hit = np.flatnonzero((c[:-1] * c[1:] <= 0) & (np.abs(c[1:]) < np.abs(c[:-1]) + 1e-12))
    if not len(hit):
        return t[e]
    i = hit[0]
    c0, c1 = c[i], c[i + 1]
    frac = 0.0 if c0 == c1 else c0 / (c0 - c1)
    return t[s + i] + frac * (t[s + i + 1] - t[s + i])


def measured_zero_crossing(t, torque, cmd, s, stop):
    """Time the measured torque crosses zero after ramp start s, before index stop.

    The direction is taken from the ramp (the cell's sign is opposite to the
    command's, but only the direction of travel matters). Falls back to None
    when the smoothed torque never crosses, e.g. the final ramp landing near 0.
    """
    n = max(int(round(TORQUE_SMOOTH_S / np.median(np.diff(t)))), 1)
    y = np.convolve(torque[s:stop], np.ones(n) / n, mode='same')
    # +1 if the torque has to rise through zero, -1 if fall: opposite of where it starts
    rising = y[:n].mean() < 0
    hit = np.flatnonzero((y[:-1] < 0) & (y[1:] >= 0) if rising else (y[:-1] > 0) & (y[1:] <= 0))
    if not len(hit):
        return None
    i = hit[0]
    frac = y[i] / (y[i] - y[i + 1])
    return t[s + i] + frac * (t[s + i + 1] - t[s + i])


def decimate(x, n=PLOT_DECIMATE):
    m = len(x) // n * n
    return x[:m].reshape(-1, n).mean(axis=1)


def load_segments(log, torque_channel):
    segs = []
    with h5py.File(log, 'r') as f:
        ids = [b.decode() if isinstance(b, bytes) else str(b) for b in f['behavior_ids'][:]]
        for raw_id, (a, b) in zip(ids, f['behavior_indices'][:]):
            t = f['time'][a:b].astype(float)
            cmd = f[COMMAND_CHANNEL][a:b].astype(float)
            ramps = find_ramps(cmd)
            if not ramps:
                print(f'{raw_id}: no ramps found, skipped')
                continue
            s, e = ramps[0]
            # snap to 3 significant figures: the sampled endpoints make 0.05 read 0.0499963
            rate = float(f'{abs(cmd[e] - cmd[s]) / (t[e] - t[s]):.3g}')
            segs.append(dict(
                id=raw_id, t=t - t[0], cmd=cmd, rate=rate,
                torque=f[torque_channel][a:b].astype(float),
                current=f[CURRENT_CHANNEL][a:b].astype(float),
                ramp_start=[t[s_] - t[0] for s_, _ in ramps],
                ramp_end=[t[e_] - t[0] for _, e_ in ramps],
                ramp_zero=[zero_crossing(t, cmd, s_, e_) - t[0] for s_, e_ in ramps],
                torque_zero=_torque_zeros(t, f[torque_channel][a:b].astype(float), cmd, ramps, raw_id)))
    return sorted(segs, key=lambda s: s['rate'])


def _torque_zeros(t, torque, cmd, ramps, raw_id):
    """Measured-torque zero crossing per ramp; command crossing where there is none."""
    out = []
    for j, (s_, e_) in enumerate(ramps):
        stop = ramps[j + 1][0] if j + 1 < len(ramps) else len(t) - 1
        z = measured_zero_crossing(t, torque, cmd, s_, stop)
        if z is None:
            z = zero_crossing(t, cmd, s_, e_)
            if 0 < j < len(ramps) - 1:   # only ramps 1..n-2 are used as anchors
                print(f'{raw_id}: ramp {j} torque never crosses zero; using the command crossing')
        out.append(z - t[0])
    return out


def aligned_time(segs, mode):
    """Per-segment x array (NaN where a gap must break the line) + hold-start xs."""
    n_ramps = {len(s['ramp_end']) for s in segs}
    if mode in ('hold', 'zero', 'torque_zero') and len(n_ramps) != 1:
        print(f'segments disagree on ramp count {sorted(n_ramps)}; falling back to --align start')
        mode = 'start'
    if mode == 'start':
        for s in segs:
            s['x'] = s['t']
        return [], mode
    if mode in ('zero', 'torque_zero'):
        key = 'ramp_zero' if mode == 'zero' else 'torque_zero'
        return _align_zero(segs, n_ramps.pop(), key), mode

    # Canonical hold-start for each ramp j: far enough after hold j-1 started
    # that the slowest segment's hold j-1 and ramp j both fit.
    n = n_ramps.pop()
    canon = [max(s['ramp_end'][0] for s in segs)]
    for j in range(1, n):
        canon.append(canon[-1] + max(s['ramp_end'][j] - s['ramp_end'][j - 1] for s in segs))

    for s in segs:
        x = np.empty_like(s['t'])
        # piece j spans [ramp_start_j, ramp_start_{j+1}); piece 0 also owns the lead-in
        bounds = [0.0] + s['ramp_start'][1:] + [np.inf]
        for j in range(n):
            m = (s['t'] >= bounds[j]) & (s['t'] < bounds[j + 1])
            x[m] = s['t'][m] - s['ramp_end'][j] + canon[j]
        s['x'] = x
    return canon, mode


def _align_zero(segs, n, key='ramp_zero'):
    """Section 0 raw; section k>=1 shifted so ramp k's zero crossing hits a shared x.

    Section k spans [ramp_start_k, ramp_start_{k+1}); the last ramp is the
    return to zero, so the last section runs to the end of the data instead.
    """
    n_sec = max(n - 1, 1)
    end = lambda s, k: s['ramp_start'][k + 1] if k + 1 < n_sec else s['t'][-1]
    canon = []
    prev_end = max(end(s, 0) for s in segs)          # section 0 is unshifted
    for k in range(1, n_sec):
        z = prev_end + SECTION_GAP_S + max(s[key][k] - s['ramp_start'][k] for s in segs)
        canon.append(z)
        prev_end = max(z + end(s, k) - s[key][k] for s in segs)

    for s in segs:
        x = np.full_like(s['t'], np.nan)
        m = s['t'] < s['ramp_start'][1] if n_sec > 1 else np.ones_like(s['t'], bool)
        x[m] = s['t'][m]
        for k in range(1, n_sec):
            m = (s['t'] >= s['ramp_start'][k]) & (s['t'] < end(s, k) if k + 1 < n_sec else True)
            x[m] = s['t'][m] - s[key][k] + canon[k - 1]
        s['x'] = x
    return canon


def rate_colors(segs):
    """Colormap + norm over the tested rates, and one colour per segment."""
    rates = np.array([s['rate'] for s in segs])
    cmap = LinearSegmentedColormap.from_list('rate', RATE_RAMP)
    lo, hi = rates.min(), rates.max()
    if hi == lo:
        norm = Normalize(lo - 0.5 * lo, hi + 0.5 * hi)
    elif hi / lo > LOG_SPAN:
        norm = LogNorm(lo, hi)
    else:
        norm = Normalize(lo, hi)
    return cmap, norm, [cmap(norm(r)) for r in rates]


def plot_series(ax, segs, key, colors):
    for s, c in zip(segs, colors):
        # decimate each contiguous piece on its own so a block mean never
        # straddles an alignment cut, then join pieces with NaN to break the line
        dt = np.median(np.diff(s['t']))
        cuts = np.flatnonzero(np.abs(np.diff(s['x']) - dt) > 3 * dt) + 1
        xs, ys = [], []
        for xp, yp in zip(np.split(s['x'], cuts), np.split(s[key], cuts)):
            keep = np.isfinite(xp)
            xs += [decimate(xp[keep]), [np.nan]]
            ys += [decimate(yp[keep]), [np.nan]]
        ax.plot(np.concatenate(xs), np.concatenate(ys), color=c, lw=1.2,
                label=f'{s["rate"]:g} Nm/s')


def style(ax, ylabel, title=None):
    ax.set_ylabel(ylabel, color=INK)
    ax.set_xlabel('time [s]' if not title else ax.get_xlabel(), color=INK)
    ax.grid(True, color=GRID, lw=0.6)
    ax.axhline(0, color=INK_MUTED, lw=0.6)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED)
    if title:
        ax.set_title(title, color=INK, loc='left', fontsize=11)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('log', help='log folder or .hdf5 file')
    p.add_argument('--current', action='store_true',
                   help='add a DUT current vs time panel on the right (2:1 width)')
    p.add_argument('--align', choices=('zero', 'torque_zero', 'hold', 'start'), default='zero')
    p.add_argument('--torque-channel', default='load_torque')
    p.add_argument('--out', help='output .png (default: next to the log)')
    args = p.parse_args()

    log = find_log(args.log)
    segs = load_segments(log, args.torque_channel)
    if not segs:
        raise SystemExit('no segments with ramps found')
    cmap, norm, colors = rate_colors(segs)
    canon, mode = aligned_time(segs, args.align)

    if args.current:
        fig = plt.figure(figsize=(17, 6.5))
        gs = GridSpec(1, 3, figure=fig, wspace=0.28)
        ax_t, ax_i = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2])
    else:
        fig, ax_t = plt.subplots(figsize=(13, 6.5))
        ax_i = None

    plot_series(ax_t, segs, 'torque', colors)
    xlabel = {'hold': 'time [s], holds aligned at command-reaches-setpoint',
              'zero': 'time [s], sections 2+ aligned at command zero crossing',
              'torque_zero': 'time [s], sections 2+ aligned at measured torque zero crossing',
              'start': 'time since segment start [s]'}[mode]
    for x in canon:
        ax_t.axvline(x, color=INK_MUTED, lw=0.6, ls=':')
    style(ax_t, f'{args.torque_channel} [Nm]',
          'Output Torque per Commanded Ramp Rate')
    ax_t.set_xlabel(xlabel, color=INK)

    if ax_i is not None:
        plot_series(ax_i, segs, 'current', colors)
        for x in canon:
            ax_i.axvline(x, color=INK_MUTED, lw=0.6, ls=':')
        style(ax_i, f'{CURRENT_CHANNEL} [A]', 'DUT current (drive-reported)')
        ax_i.set_xlabel(xlabel.split(',')[0], color=INK)

    rates = sorted({s['rate'] for s in segs})
    ticks = rates if len(rates) <= MAX_CBAR_TICKS else rates[::2]
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                        ax=[a for a in (ax_t, ax_i) if a is not None],
                        pad=0.015, fraction=0.025, aspect=35)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f'{r:g}' for r in ticks])
    cbar.minorticks_off()
    cbar.set_label('DUT cmd ramp rate [Nm/s]', color=INK)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(colors=INK_MUTED)

    out = args.out or os.path.join(
        os.path.dirname(log), 'torque_ramp_overlay' + ('_current' if args.current else '')
        + {'zero': '', 'torque_zero': '_torquezero', 'hold': '_hold', 'start': '_start'}[mode]
        + '.png')
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'wrote {out}')
    for s in segs:
        print(f'  {s["id"]}: {s["rate"]:g} Nm/s, {len(s["ramp_end"])} ramps')


if __name__ == '__main__':
    main()
