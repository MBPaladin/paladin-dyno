"""Load-cell 0 V split checkout: is the hole there, and how big is the split?

Written for tests/ui_generated_tests/skip_cell_split_check.yaml: LOAD turns at a
steady +/-0.05 rad/s while DUT runs a slow +/-0.1 Nm torque triangle, so the
cell sweeps through 0 V many times. LOAD turns rather than holds because LOAD
current is only a clean torque reference while LOAD is sliding: at standstill
its own static friction soaks up a history-dependent share of the torque, which
shows as two hysteresis branches against the cell (torque_test_17) and biases
the split. Sliding, that friction is a constant c*sign(v). Works on any log
that crosses 0 V, including the 2026-10-19 cogging run.

On a log the logger already corrected (sensors.load_torque.split_mv, from
2026-10-21), load_torque should come out CLEAN here: that is the check that the
correction works. --raw runs the same measures on load_torque_raw instead.

Three measures, the first two being the ones the cogging writeup used (s08):

  * HOLE DENSITY. Sample density of the raw (untared) cell voltage inside the
    historical gap (-1.5..+2 mV on the wire) against its flanks. Clean channel
    ~1; the load cell has read 0.01-0.15 in every log since 2026-08-27. This is
    the A/B/C verdict.
  * GAP WIDTH. Width of the low-density band around 0 V in the histogram,
    which is the split itself (no reference needed).
  * SPLIT vs LOAD CURRENT. Bin medians of raw cell against LOAD current x kt,
    one line per motion direction, separate intercepts either side of 0 V; g is
    the step. Only steady-velocity samples, and only bins clear of 0 V, so a
    sample's own noise cannot pick its side (that would bias g up on a perfect
    channel).

The channel's port / fs_v / fs_pos come from the log's resolved_config, so runs
A (as wired), B (ch1 at +/-10 V) and C (cells swapped) label themselves.

    .venv/bin/python dyno/src/post_processors/skip/check_cell_split.py LOG.hdf5 [LOG2 ...]
"""

import argparse
import json
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS_HZ = 1000.0
KT_LOAD = 7.79                 # Nm/A
HOLE_MV = (-1.5, 2.0)          # gap on the wire, from every Skip log since 2026-08-27
FLANK_MV = ((-7.0, -3.0), (3.5, 7.5))
GAP_SEARCH_MV = 4.0            # look for the gap edges within this of 0 V; an edge
                               # that reaches it is reported as a lower bound
GAP_LEVEL = 0.25               # gap = where density drops below this x flank density
CLEAR_NM = 0.8                 # fit bins only where the raw cell is this far from 0 V
FIT_SPAN_NM = 6.0              # ... and within this, to keep the fit local to zero
REF_BIN_NM = 0.1               # bin width on the LOAD-current reference
STEADY_AFTER_S = 3.0           # skip this long after any LOAD velocity-command change
SMOOTH = 25                    # samples; boxcar on both signals before binning
JUMP_NM = 0.6                  # single-sample step that counts as a jump (noise is ~0.2)


def boxcar(x, n=SMOOTH):
    return np.convolve(x, np.ones(n) / n, mode='same')


def channel_info(f, name='load_torque'):
    rc = json.loads(f.attrs['resolved_config'])
    for dev, spec in rc['devices'].items():
        for ch, p in (spec.get('params') or {}).items():
            if isinstance(p, dict) and p.get('name') == name:
                return {'device': dev, 'channel': ch, 'port': p.get('port'),
                        'fs_v': float(p['fs_v']), 'fs_pos': float(p['fs_pos']),
                        'split_mv': p.get('split_mv'),
                        'split_center_mv': p.get('split_center_mv', 0.0)}
    raise KeyError(f'{name} not found in resolved_config')


def tare_bias(f, name='load_torque'):
    try:
        return float(json.loads(f.attrs['tare'])[name]['bias'])
    except (KeyError, ValueError, TypeError):
        return 0.0


def hole_density(mv):
    in_hole = np.count_nonzero((mv > HOLE_MV[0]) & (mv < HOLE_MV[1])) / (HOLE_MV[1] - HOLE_MV[0])
    flank_n = sum(np.count_nonzero((mv > lo) & (mv < hi)) for lo, hi in FLANK_MV)
    flank = flank_n / sum(hi - lo for lo, hi in FLANK_MV)
    return (in_hole / flank if flank > 0 else np.nan), int(flank_n)


def gap_width(mv, bin_mv=0.1):
    """Edges of the contiguous low-density band nearest 0 V, in mV, or None.

    Density is judged against the mean of the two flanks, so a signal that
    simply spends less time near 0 V does not read as a gap, but one that
    spends little time anywhere near 0 V cannot be judged at all.
    """
    edges = np.arange(-GAP_SEARCH_MV - 4, GAP_SEARCH_MV + 4 + 1e-9, bin_mv)
    h, _ = np.histogram(mv, bins=edges)
    h = np.convolve(h, np.ones(3) / 3, mode='same')
    c = 0.5 * (edges[1:] + edges[:-1])
    flank = np.mean([h[(c > lo) & (c < hi)].mean() for lo, hi in FLANK_MV])
    if not flank > 0:
        return None
    low = h < GAP_LEVEL * flank
    i0 = int(np.argmin(np.abs(c - 0.25)))      # start from the middle of the historical hole
    if not low[i0]:
        return None
    lo = i0
    while lo > 0 and low[lo - 1] and c[lo - 1] > -GAP_SEARCH_MV:
        lo -= 1
    hi = i0
    while hi < len(c) - 1 and low[hi + 1] and c[hi + 1] < GAP_SEARCH_MV:
        hi += 1
    capped = c[lo] <= -GAP_SEARCH_MV + bin_mv or c[hi] >= GAP_SEARCH_MV - bin_mv
    return float(edges[lo]), float(edges[hi + 1]), bool(capped)


def steady_mask(v_cmd):
    """Samples at a constant, nonzero LOAD velocity command, STEADY_AFTER_S past
    the last change."""
    v = np.nan_to_num(v_cmd)
    change = np.flatnonzero(np.abs(np.diff(v)) > 1e-9) + 1
    since = np.full(len(v), np.inf)
    last = -np.inf
    j = 0
    for i in range(len(v)):
        while j < len(change) and change[j] <= i:
            last = change[j]
            j += 1
        since[i] = i - last
    return (np.abs(v) > 1e-6) & (since > STEADY_AFTER_S * FS_HZ)


def split_vs_load(raw, i_load, v_cmd, ok):
    """g from binned medians; None (with a reason) if the log cannot give it."""
    steady = steady_mask(v_cmd) & ok
    if steady.sum() < 20 * FS_HZ:
        return None, ('under 20 s at a steady nonzero LOAD velocity; LOAD current is '
                      'not a clean reference at standstill')
    rs = boxcar(np.where(ok, raw, 0.0))
    ref = boxcar(np.where(ok, KT_LOAD * i_load, 0.0))
    sv = np.sign(np.nan_to_num(v_cmd))
    rows = []   # (ref median, raw median, side, direction, count)
    for direction in (-1.0, 1.0):
        sel = steady & (sv == direction)
        if not sel.any():
            continue
        b = np.floor(ref[sel] / REF_BIN_NM).astype(int)
        r_sel, x_sel = rs[sel], ref[sel]
        for k in np.unique(b):
            m = b == k
            if m.sum() < 50:
                continue
            y = float(np.median(r_sel[m]))
            if not (CLEAR_NM < abs(y) < FIT_SPAN_NM):
                continue
            rows.append((float(np.median(x_sel[m])), y, np.sign(y), direction, int(m.sum())))
    rows = np.array(rows)
    if len(rows) == 0 or min(np.count_nonzero(rows[:, 2] < 0), np.count_nonzero(rows[:, 2] > 0)) < 3:
        return None, ('the steady samples do not reach +/-%.1f Nm raw on both sides of 0 V'
                      % CLEAR_NM)
    dirs = np.unique(rows[:, 3])
    # raw = a*ref + b + (g/2)*side + c*direction  (c only if both directions are present)
    cols = [rows[:, 0], np.ones(len(rows)), 0.5 * rows[:, 2]]
    if len(dirs) > 1:
        cols.append(rows[:, 3])
    A = np.column_stack(cols)
    w = np.sqrt(rows[:, 4])
    coef, *_ = np.linalg.lstsq(A * w[:, None], rows[:, 1] * w, rcond=None)
    resid = rows[:, 1] - A @ coef
    fit = {'slope': float(coef[0]), 'split_nm': float(coef[2]),
           'drag_nm': float(coef[3]) if len(dirs) > 1 else np.nan,
           'bin_rms_nm': float(np.sqrt(np.average(resid ** 2, weights=rows[:, 4]))),
           'bins': len(rows), 'rows': rows, 'coef': coef}
    return fit, None


def analyse(path, use_raw=False):
    with h5py.File(path, 'r') as f:
        ch = channel_info(f)
        split = ch.get('split_mv')
        if use_raw:
            if 'load_torque_raw' not in f:
                raise KeyError(f'{path}: --raw asked for, but the log has no load_torque_raw')
            cell = f['load_torque_raw'][:].astype(float)
            bias = 0.0                                   # logged untared
        else:
            cell = f['load_torque'][:].astype(float)
            bias = tare_bias(f)
        i_load = f['load_current'][:].astype(float)
        v_cmd = f['load_velocity_command'][:].astype(float)
    ok = np.isfinite(cell) & np.isfinite(i_load)
    raw = cell - bias                                # untared: the split sits at 0 V raw
    mv_per_nm = 1000.0 * ch['fs_v'] / ch['fs_pos']   # +/-fs_v on the wire at +/-fs_pos
    mv = raw[ok] * mv_per_nm
    out = {'log': os.path.basename(path), **ch, 'tare_bias_nm': bias,
           'mv_per_nm': mv_per_nm, '_mv': mv,
           'signal': ('load_torque_raw (uncorrected)' if use_raw else
                      f"load_torque, split-corrected at acquisition ({split:g} mV)" if split
                      else 'load_torque (no correction at acquisition)')}
    out['hole_density'], out['flank_samples'] = hole_density(mv)
    out['gap_mv'] = gap_width(mv)
    out['fit'], out['fit_note'] = split_vs_load(raw, i_load, v_cmd, ok)
    if out['fit']:
        # On a log whose reference sits in a few discrete clusters (a dwell
        # grid) the slope and the two intercepts trade off and g moves with
        # the window; on a continuous sweep it does not. Say which this is.
        global FIT_SPAN_NM
        keep, alt = FIT_SPAN_NM, []
        for span in (0.67 * keep, 1.67 * keep):
            FIT_SPAN_NM = span
            f, _ = split_vs_load(raw, i_load, v_cmd, ok)
            alt.append(f['split_nm'] if f else np.nan)
        FIT_SPAN_NM = keep
        out['fit']['split_alt_nm'] = alt

    d = np.abs(np.diff(raw))
    both = ok[:-1] & ok[1:]
    near = both & (np.abs(raw[:-1]) < 1.0)
    far = both & (np.abs(raw[:-1]) > 2.0)
    out['jumps_near0'] = 1000.0 * np.count_nonzero(d[near] > JUMP_NM) / max(near.sum(), 1)
    out['jumps_far'] = 1000.0 * np.count_nonzero(d[far] > JUMP_NM) / max(far.sum(), 1)
    return out


def verdict(r):
    h = r['hole_density']
    if r['flank_samples'] < 5000:
        return 'INCONCLUSIVE: too little time near 0 V to judge'
    if h < 0.3:
        return 'SPLIT PRESENT'
    if h > 0.7:
        return 'CLEAN'
    return 'PARTIAL: between clean and the historical hole'


def report(r):
    print(f"\n== {r['log']}")
    print(f"   channel   {r['device']}.{r['channel']} ({r['port']}), ±{r['fs_v']:g} V, "
          f"fs_pos {r['fs_pos']:g} Nm -> {r['mv_per_nm']:.1f} mV/Nm; "
          f"tare bias {r['tare_bias_nm']:+.3f} Nm")
    print(f"   signal    {r['signal']}")
    print(f"   hole      density {r['hole_density']:.2f} of flanks "
          f"({r['flank_samples']:,} flank samples) -> {verdict(r)}")
    if r['gap_mv']:
        lo, hi, capped = r['gap_mv']
        note = (' -- AT LEAST: an edge ran into the search limit, the signal is just '
                'sparse there' if capped else '')
        print(f"   gap       {lo:+.1f} .. {hi:+.1f} mV, width {hi - lo:.1f} mV "
              f"= {(hi - lo) / r['mv_per_nm']:.2f} Nm{note}")
    else:
        print('   gap       none found at 0 V')
    fit = r['fit']
    if fit:
        drag = '' if np.isnan(fit['drag_nm']) else f", LOAD-side drag {fit['drag_nm']:+.2f} Nm x sign(v)"
        print(f"   split     g = {fit['split_nm']:.3f} Nm = {fit['split_nm'] * r['mv_per_nm']:.2f} mV "
              f"vs LOAD current (slope {fit['slope']:.3f}{drag}; {fit['bins']} bins, "
              f"rms {fit['bin_rms_nm']:.3f} Nm)")
        alt = fit['split_alt_nm']
        spread = np.nanmax(np.abs(np.array(alt) - fit['split_nm'])) if np.any(np.isfinite(alt)) else np.inf
        flag = ('stable' if spread < 0.05 else
                'UNSTABLE: the reference does not sweep continuously, trust the gap width instead')
        print(f"             same fit over narrower / wider windows: "
              f"{alt[0]:.3f} / {alt[1]:.3f} Nm -> {flag}")
    else:
        print(f"   split     not fitted: {r['fit_note']}")
    print(f"   jumps     {r['jumps_near0']:.1f} per 1000 samples within 1 Nm of 0 V, "
          f"{r['jumps_far']:.1f} elsewhere (> {JUMP_NM} Nm in one sample)")


def plot(results, out_png):
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(5.5 * n, 7.5), squeeze=False)
    for k, r in enumerate(results):
        ax = axes[0, k]
        m = r['_mv']
        ax.hist(m[np.abs(m) < 12], bins=np.arange(-12, 12.01, 0.25), color='0.35')
        ax.axvspan(*HOLE_MV, color='tab:red', alpha=0.15, label='historical hole')
        if r['gap_mv']:
            for e in r['gap_mv'][:2]:
                ax.axvline(e, color='tab:red', lw=1)
        ax.set_xlabel('raw cell on the wire [mV]')
        ax.set_ylabel('samples')
        ax.set_title(f"{r['log']}\n{r['channel']} ({r['port']}), ±{r['fs_v']:g} V, fs {r['fs_pos']:g} Nm"
                     f"\nhole density {r['hole_density']:.2f}: {verdict(r)}", fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax = axes[1, k]
        fit = r['fit']
        if fit:
            rows, coef = fit['rows'], fit['coef']
            base = coef[0] * rows[:, 0] + coef[1]
            if len(coef) > 3:
                base = base + coef[3] * rows[:, 3]
            for direction, mk in ((-1, 'v'), (1, '^')):
                s = rows[:, 3] == direction
                ax.scatter(rows[s, 0], rows[s, 1] - base[s], c=np.where(rows[s, 2] > 0, 'tab:blue', 'tab:orange'),
                           marker=mk, s=12, label=f'LOAD moving {"+" if direction > 0 else "-"}')
            ax.axhline(+coef[2] / 2, color='tab:red', lw=1)
            ax.axhline(-coef[2] / 2, color='tab:red', lw=1)
            ax.set_title(f"split vs LOAD current: g = {fit['split_nm']:.3f} Nm\n"
                         'blue: raw > 0 V, orange: raw < 0 V', fontsize=9)
            ax.legend(fontsize=8)
        else:
            ax.set_title(r['fit_note'], fontsize=8, wrap=True)
        ax.set_xlabel('LOAD current x kt, bin median [Nm]')
        ax.set_ylabel('raw cell minus line and drag [Nm]')
        ax.set_ylim(-0.6, 0.6)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--png', help='figure path (default: next to the first log)')
    ap.add_argument('--raw', action='store_true',
                    help='analyse load_torque_raw (logs with acquisition-side correction) '
                         'instead of load_torque')
    args = ap.parse_args()
    results = [analyse(p, args.raw) for p in args.logs]
    for r in results:
        report(r)
    png = args.png or os.path.splitext(args.logs[0])[0] + '_cell_split_check.png'
    plot(results, png)
    print(f'\nfigure: {png}')


if __name__ == '__main__':
    main()
