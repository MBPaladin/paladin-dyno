"""Score the DUT compensation profiles from an R2 / R3 log (dut_compensation_profiles grids).

Every hold is a logged span flagged <id>-RUN<r>-SETPOINT<n>-<profile>. For each,
the torque error e = cell - target is cut to a whole number of motor revs and
reduced to:

  bias     mean(e): what base and speed are for
  ripple   rms(e - bias): what the harmonic terms are for
  total    rms(e)
  p1_p99   spread of e
  A1/A2/A6 order amplitudes of e per motor rev (angle-resampled fit): each
           harmonic term should remove its own line

The cell is the logged, tared load_torque (split-corrected at acquisition).
The target is dut_comp_target as the controller used it.

Run:
    .venv/bin/python dyno/src/post_processors/skip/s16_comp_ab.py LOG.hdf5

Outputs (results/comp_ab/<log>/):
  comp_ab_holds.csv       one row per hold
  comp_ab_summary.csv     per speed x profile, means over the holds
  comp_ab.png             bias and ripple vs target, per speed, by profile
"""

import argparse
import csv
import os
import re

import h5py
import numpy as np
import matplotlib.pyplot as plt

import skip_common as c

FLAG = re.compile(r'^(?P<seg>.+)-RUN(?P<run>\d+)-SETPOINT(?P<pt>\d+)-(?P<profile>[A-Za-z0-9_]+)$')
ORDERS = (1, 2, 6)
STEP_M = c.TWO_PI / 512
# Fixed order and colours; the drift-check repeat shares C1's colour, dashed.
PROFILE_ORDER = ('C0_naive', 'C1_base', 'C2_speed', 'C3_h1', 'C4_h6', 'C5_hx', 'C6_all',
                 'C7_qdrag', 'C8_all_q', 'C1_repeat')
PROFILE_COLOR = {'C0_naive': '#8a8984', 'C1_base': '#0b0b0b', 'C2_speed': '#eb6834',
                 'C3_h1': '#2a78d6', 'C4_h6': '#1baf7a', 'C5_hx': '#a35ad6',
                 'C6_all': '#e34948', 'C7_qdrag': '#d6a51b', 'C8_all_q': '#7a1f1e',
                 'C1_repeat': '#0b0b0b'}


def holds(path):
    with h5py.File(path, 'r') as f:
        ids = [b.decode() if isinstance(b, bytes) else str(b) for b in f['behavior_ids'][:]]
        idx = f['behavior_indices'][:]
        cell = f['load_torque'][:].astype(float)
        tgt = f['dut_comp_target'][:].astype(float)
        th = f['dut_output_position'][:].astype(float)
        vc = f['load_velocity_command'][:].astype(float)
        cur = f['dut_comp_current'][:].astype(float)
    rows = []
    for bid, (i0, i1) in zip(ids, idx):
        m = FLAG.match(bid)
        if not m or i1 <= i0:
            continue
        sl = slice(int(i0), int(i1) + 1)
        t, y, a = tgt[sl], cell[sl], th[sl]
        ok = np.isfinite(t) & np.isfinite(y)
        if ok.sum() < 500:
            continue
        # whole motor revs from the start of the span
        revs = np.abs(a - a[0]) / c.TWO_PI
        n_rev = int(np.floor(revs[ok].max()))
        keep = ok & (revs <= max(n_rev, 1))
        e = y[keep] - t[keep]
        target = float(np.median(t[keep]))
        speed = float(np.median(vc[sl]))
        row = {'flag': bid, 'segment': m['seg'], 'point': int(m['pt']), 'profile': m['profile'],
               'speed': speed, 'target': target,
               'quadrant': 'forward' if target * speed > 0 else 'backdrive',
               'motor_revs': n_rev, 'current_mean': float(np.nanmean(cur[sl][keep])),
               'bias': float(e.mean()), 'ripple': float(e.std()),
               'total': float(np.sqrt(np.mean(e ** 2))),
               'p1_p99': float(np.percentile(e, 99) - np.percentile(e, 1))}
        if n_rev >= 2:
            yr, _ = c.angle_resample(a[keep], e, STEP_M)
            n = len(yr)
            tg = a[keep][0] + np.sign(speed) * STEP_M * (np.arange(n) + 0.5)
            amps, _, _, _ = c.lsq_orders([(k, tg, k) for k in ORDERS], yr,
                                         c.legendre_nuisance(n, min(4, n_rev - 1)))
            for k in ORDERS:
                row[f'A{k}'] = amps[k][0]
        rows.append(row)
    return rows


def profile_order(rows):
    """Known profiles in their fixed order, then any others by name."""
    seen = {r['profile'] for r in rows}
    return [p for p in PROFILE_ORDER if p in seen] + sorted(seen - set(PROFILE_ORDER))


def summarise(rows):
    out = []
    for v in sorted({abs(r['speed']) for r in rows}):
        for p in profile_order(rows):
            rs = [r for r in rows if abs(r['speed']) == v and r['profile'] == p]
            if not rs:
                continue
            s = {'speed_abs': v, 'profile': p, 'holds': len(rs),
                 'mean_abs_bias': np.mean([abs(r['bias']) for r in rs]),
                 'mean_ripple': np.mean([r['ripple'] for r in rs]),
                 'mean_total': np.mean([r['total'] for r in rs])}
            for k in ORDERS:
                vals = [r[f'A{k}'] for r in rs if f'A{k}' in r]
                s[f'mean_A{k}'] = np.mean(vals) if vals else np.nan
            for q in ('forward', 'backdrive'):
                qs = [r for r in rs if r['quadrant'] == q]
                s[f'total_{q}'] = np.mean([r['total'] for r in qs]) if qs else np.nan
            out.append(s)
    return out


def figure(rows, path, title):
    speeds = sorted({abs(r['speed']) for r in rows})
    fig, ax = plt.subplots(len(speeds), 2, figsize=(12, 3.6 * len(speeds)), squeeze=False)
    for i, v in enumerate(speeds):
        for j, (key, lab) in enumerate((('bias', 'bias = mean(cell - target) [Nm]'),
                                        ('ripple', 'ripple = rms about the mean [Nm]'))):
            a = ax[i, j]
            for p in profile_order(rows):
                for sgn, mk in ((1, '^'), (-1, 'v')):
                    rs = sorted([r for r in rows if abs(r['speed']) == v and r['profile'] == p
                                 and np.sign(r['speed']) == sgn], key=lambda r: r['target'])
                    if not rs:
                        continue
                    a.plot([r['target'] for r in rs], [r[key] for r in rs], marker=mk, ms=5,
                           lw=1.2, color=PROFILE_COLOR.get(p, c.INK_3), ls='--' if p == 'C1_repeat' else '-',
                           label=p if sgn == 1 else None)
            a.axhline(0, color=c.INK_3, lw=0.6)
            a.axvline(0, color=c.INK_3, lw=0.6)
            a.set_title(f'{v:g} rad/s: {key} (^ output +, v output -)', fontsize=9)
            a.set_xlabel('target output torque [Nm]')
            a.set_ylabel(lab)
            a.grid(color=c.GRID, lw=0.6)
            if i == 0 and j == 1:
                a.legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('log')
    args = ap.parse_args()
    stem = os.path.splitext(os.path.basename(args.log))[0]
    out_dir = os.path.join(c.SKIP_LOGS, 'results', 'comp_ab', stem)
    os.makedirs(out_dir, exist_ok=True)

    rows = holds(args.log)
    if not rows:
        raise SystemExit('no profile-flagged holds in this log')
    summ = summarise(rows)
    for name, data in (('comp_ab_holds.csv', rows), ('comp_ab_summary.csv', summ)):
        keys = list(dict.fromkeys(k for r in data for k in r))
        with open(os.path.join(out_dir, name), 'w', newline='') as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(data)

    ref_profile = 'C1_base' if any(r['profile'] == 'C1_base' for r in rows) else 'C2_speed'
    print(f'{len(rows)} holds. Per speed, mean over holds (Nm); ratio = total / {ref_profile} total:')
    print(f'{"speed":>6s} {"profile":10s} {"|bias|":>7s} {"ripple":>7s} {"total":>7s} {"ratio":>6s} '
          f'{"fwd":>6s} {"bd":>6s} {"A1":>6s} {"A2":>6s} {"A6":>6s}')
    for v in sorted({s['speed_abs'] for s in summ}):
        ref = next((s['mean_total'] for s in summ if s['speed_abs'] == v and s['profile'] == ref_profile),
                   np.nan)
        for s in [s for s in summ if s['speed_abs'] == v]:
            print(f'{v:6.2f} {s["profile"]:10s} {s["mean_abs_bias"]:7.3f} {s["mean_ripple"]:7.3f} '
                  f'{s["mean_total"]:7.3f} {s["mean_total"] / ref:6.2f} {s["total_forward"]:6.3f} '
                  f'{s["total_backdrive"]:6.3f} {s["mean_A1"]:6.3f} {s["mean_A2"]:6.3f} {s["mean_A6"]:6.3f}')
    figure(rows, os.path.join(out_dir, 'comp_ab.png'), f'{stem}: open-loop torque error by profile')
    print(f'wrote {out_dir}/')


if __name__ == '__main__':
    main()
