"""Motor-angle phase calibration: where today's encoder zero sits against v2's.

The Elmo's position counter (0x6064) restarts when the drive powers up, and the
within-rev phase of `dut_output_position` is not reliably kept. Across the two
resets in the Skip logs, v1 -> v2 came back 0.1 deg from where it was, and
v3 -> efficiency came back 11 deg off (DUT-current folds, every sliding point
agreeing within ~1 deg). 11 deg at the motor is 66 deg on the 6/rev, enough
for that compensation term to add ripple instead of removing it. A GUI restart
without a power cycle keeps it: v2 -> v3 carried over. The angle-locked ripple,
though, is fixed to the gearbox. So a short
dwell at known conditions, folded over motor angle and lined up against the v2
fold at the same conditions, gives the one offset that maps today's angle onto
v2's frame:

    theta_v2 = theta_today + phase_offset_rad

Every ripple phase fitted from v2 then applies today as cos(k*(theta + offset) + phi).

Method. Each dwell is resampled onto a uniform motor-angle grid and fitted with
motor orders 1-20 (plus the 21 and 42 electrical lines, and a slow drift).
Today's order-k phase is v2's plus k*offset, so the offset maximises

    J(offset) = sum over cells and orders of A_today*A_ref*cos(phi_today - phi_ref - k*offset),

which is the circular cross-correlation of the two folds over those orders. It
is solved jointly over every cell and per cell. The per-cell spread, and the
1/rev against the 6/rev estimate, are the checks: a pure encoder shift moves
every cell and every order together, and anything else means the gearbox moved.

Cells are matched to the reference by direction, quadrant, speed and nearest
|current along the motion|. The test plans are
`tests/ui_generated_tests/skip_phase_cal_R1a.yaml` (4 loaded cells + 0 A) and
`skip_phase_cal_R1b.yaml` (backdrive cells only, for after a GUI restart).

Run:
    .venv/bin/python dyno/src/post_processors/skip/s14_phase_cal.py LOG.hdf5 [LOG2.hdf5 ...]
        [--ref v2] [--speed 0.1]

With several logs, it also reports the offset of each against the first. That
is the R1a -> GUI restart -> R1b check, which should read ~0.

Outputs (results/phase_cal/):
  <log>_phase_cal.json   phase_offset_rad and every check below
  <log>_phase_cal.png    folds lined up, J(offset), convergence vs revs
"""

import argparse
import json
import os
import warnings

import numpy as np
import matplotlib.pyplot as plt

import skip_common as c

OUT_DIR = os.path.join(c.SKIP_LOGS, 'results', 'phase_cal')
STEP_M = c.TWO_PI / 1024                  # motor-angle grid, as skip_orders
MECH = tuple(range(1, 21))                # orders used for the offset
ELEC = (c.POLE_PAIRS, 2 * c.POLE_PAIRS)   # fitted so they cannot leak, not used
SLOW_DEGREE = 8
FOLD_BINS = 180
OFFSET_GRID = np.linspace(0.0, c.TWO_PI, 36000, endpoint=False)   # 0.01 deg
CONVERGENCE_REVS = (1, 2, 3, 5, 10, 20)
# Warn when the cells disagree by more than this. For the 6/rev, 5 deg at the
# motor is 30 deg of phase, i.e. cos(30) = 87% of the ripple cancelled.
CELL_SPREAD_WARN_DEG = 5.0


# --- loading ------------------------------------------------------------------------
def load_log(path):
    """skip_common.load, plus a cell zero for logs with no 0 A dwells (R1b):
    fall back to the session tare, i.e. the cell as logged."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # mean of no zero dwells
        d = c.load(path)
    if not np.isfinite(d['cell_zero']):
        d['cell_zero'] = -d['tare_bias']
        d['T_cell'] = d['cell_corr'] - d['cell_zero']
        for w in d['dwells']:
            w['T_cell'] = float(np.mean(d['T_cell'][c.dwell_slice(w)]))
        print(f'  no 0 A dwells: cell zero taken from the session tare '
              f'({d["tare_bias"]:+.3f} Nm bias)')
    return d


def label(w):
    return f'{"+" if w["dir"] > 0 else "-"}v {w["quadrant"]} {abs(w["i_along_nom"]):g} A'


# --- fitting --------------------------------------------------------------------------
def fit_orders(d, w, revs=None):
    """Order amplitudes/phases of the cell torque over one dwell, optionally
    only its first `revs` motor revs. Returns ({k: (A, phi)}, grid dict)."""
    sl = c.dwell_slice(w)
    th, y = d['th_m'][sl], d['T_cell'][sl]
    if revs is not None:
        keep = np.abs(th - th[0]) <= revs * c.TWO_PI
        th, y = th[keep], y[keep]
    yr, _ = c.angle_resample(th, y, STEP_M)
    n = len(yr)
    th_grid = th[0] + w['dir'] * STEP_M * (np.arange(n) + 0.5)
    total_revs = n * STEP_M / c.TWO_PI
    # A degree-8 drift over one rev would eat the 1/rev itself.
    degree = SLOW_DEGREE if total_revs >= 8 else max(1, int(total_revs) - 1)
    cols = [(k, th_grid, k) for k in MECH + ELEC]
    amps, _, slow, _ = c.lsq_orders(cols, yr, c.legendre_nuisance(n, degree))
    return amps, {'th': th_grid, 'y': yr - slow, 'revs': total_revs}


def objective(pairs, mirror=False):
    """J(offset) over OFFSET_GRID, normalised to a correlation in [-1, 1].
    pairs: list of (amps_today, amps_ref)."""
    J = np.zeros_like(OFFSET_GRID)
    nt = nr = 0.0
    for at, ar in pairs:
        for k in MECH:
            A_t, p_t = at[k]
            A_r, p_r = ar[k]
            if mirror:
                p_t = -p_t
            J += A_t * A_r * np.cos(p_t - p_r - k * OFFSET_GRID)
            nt += A_t ** 2
            nr += A_r ** 2
    return J / np.sqrt(nt * nr)


def best_offset(J):
    """Peak of J with a parabolic refinement between grid points."""
    i = int(np.argmax(J))
    a, b, cc = J[i - 1], J[i], J[(i + 1) % len(J)]
    den = a - 2 * b + cc
    frac = 0.5 * (a - cc) / den if den != 0 else 0.0
    step = OFFSET_GRID[1] - OFFSET_GRID[0]
    return float(np.mod(OFFSET_GRID[i] + frac * step, c.TWO_PI)), float(b)


def order_offset(at, ar, k, near):
    """Offset implied by order k alone, on the branch nearest `near`."""
    raw = (at[k][1] - ar[k][1]) / k
    branches = raw + np.arange(k) * c.TWO_PI / k
    dist = np.abs(np.angle(np.exp(1j * (branches - near))))
    return float(np.mod(branches[np.argmin(dist)], c.TWO_PI))


def wrap_deg(x):
    """Radians -> degrees in (-180, 180]."""
    return float(np.degrees(np.angle(np.exp(1j * x))))


# --- matching ---------------------------------------------------------------------------
def match_reference(ref, w, ref_speed):
    cands = [r for r in ref['dwells']
             if r['dir'] == w['dir'] and r['quadrant'] == w['quadrant']
             and abs(r['speed_abs'] - ref_speed) < 1e-6]
    if not cands:
        return None
    return min(cands, key=lambda r: abs(abs(r['i_along_nom']) - abs(w['i_along_nom'])))


def calibrate(d, ref, speed, ref_speed=None):
    """Offset of log `d` against `ref`. Returns a result dict."""
    cells = [w for w in d['dwells']
             if w['quadrant'] != 'zero' and abs(w['speed_abs'] - speed) < 1e-6]
    if not cells:
        raise SystemExit(f'no loaded dwells at {speed} rad/s in {d["path"]}')
    pairs, rows = [], []
    for w in cells:
        r = match_reference(ref, w, speed if ref_speed is None else ref_speed)
        if r is None:
            print(f'  {label(w)}: no reference cell, skipped')
            continue
        at, grid = fit_orders(d, w)
        ar, _ = fit_orders(ref, r)
        pairs.append((at, ar))
        rows.append({'w': w, 'r': r, 'at': at, 'ar': ar, 'grid': grid})

    J = objective(pairs)
    offset, corr = best_offset(J)
    _, corr_mirror = best_offset(objective(pairs, mirror=True))

    cell_results = []
    for row in rows:
        w, r, at, ar = row['w'], row['r'], row['at'], row['ar']
        off_c, corr_c = best_offset(objective([(at, ar)]))
        conv = {}
        for n in CONVERGENCE_REVS:
            if n > row['grid']['revs'] * 0.98:
                break
            at_n, _ = fit_orders(d, w, revs=n)
            conv[n] = wrap_deg(best_offset(objective([(at_n, ar)]))[0] - off_c)
        cell_results.append({
            'cell': label(w),
            'ref_cell': label(r) + f' @ {r["speed_abs"]:g} rad/s',
            'motor_revs': row['grid']['revs'],
            'offset_deg': float(np.degrees(off_c)),
            'vs_joint_deg': wrap_deg(off_c - offset),
            'correlation': corr_c,
            'offset_from_1rev_deg': float(np.degrees(order_offset(at, ar, 1, offset))),
            'offset_from_6rev_deg': float(np.degrees(order_offset(at, ar, 6, offset))),
            'A1_today_ref_Nm': [at[1][0], ar[1][0]],
            'A6_today_ref_Nm': [at[6][0], ar[6][0]],
            'T_mean_today_ref_Nm': [w['T_cell'], r['T_cell']],
            'current_today_ref_A': [w['current'], r['current']],
            'convergence_deg_vs_full': conv,
        })
        row['offset_cell'] = off_c

    zero = [{'cell': f'{"+" if w["dir"] > 0 else "-"}v 0 A', 'T_mean_Nm': w['T_cell']}
            for w in d['dwells'] if w['quadrant'] == 'zero'
            and abs(w['speed_abs'] - speed) < 1e-6]
    spread = max(abs(cr['vs_joint_deg']) for cr in cell_results)
    return {'log': d['path'], 'speed_rad_s': speed,
            'phase_offset_rad': offset, 'phase_offset_deg': float(np.degrees(offset)),
            'correlation': corr, 'correlation_if_mirrored': corr_mirror,
            'max_cell_spread_deg': spread,
            'cells': cell_results, 'zero_current': zero,
            '_J': J, '_rows': rows}


# --- reporting ------------------------------------------------------------------------------
def report(res, ref_name):
    print(f'\n=== {os.path.basename(res["log"])} vs {ref_name} '
          f'at {res["speed_rad_s"]:g} rad/s ===')
    print(f'phase_offset_rad: {res["phase_offset_rad"]:.4f}   '
          f'({res["phase_offset_deg"]:.2f} deg motor; theta_{ref_name} = theta + offset)')
    print(f'fold correlation at that offset: {res["correlation"]:.3f}   '
          f'(mirrored angle would give {res["correlation_if_mirrored"]:.3f})')
    print(f'{"cell":26s} {"revs":>5s} {"off-joint":>9s} {"corr":>5s} {"1/rev":>7s} '
          f'{"6/rev":>7s} {"A1 now/ref":>11s} {"A6 now/ref":>11s} {"T now/ref":>13s}')
    for cr in res['cells']:
        a1, a6, tm = cr['A1_today_ref_Nm'], cr['A6_today_ref_Nm'], cr['T_mean_today_ref_Nm']
        print(f'{cr["cell"]:26s} {cr["motor_revs"]:5.1f} {cr["vs_joint_deg"]:+8.2f}° '
              f'{cr["correlation"]:5.2f} {cr["offset_from_1rev_deg"]:6.1f}° '
              f'{cr["offset_from_6rev_deg"]:6.1f}° {a1[0]:5.2f}/{a1[1]:4.2f} '
              f'{a6[0]:5.2f}/{a6[1]:4.2f} {tm[0]:+6.2f}/{tm[1]:+6.2f}')
    for z in res['zero_current']:
        print(f'{z["cell"]:26s} mean {z["T_mean_Nm"]:+.3f} Nm')
    print('convergence (deg from the full-dwell estimate, first N motor revs):')
    for cr in res['cells']:
        conv = '  '.join(f'{n}:{v:+.1f}' for n, v in cr['convergence_deg_vs_full'].items())
        print(f'  {cr["cell"]:24s} {conv}')
    if res['correlation_if_mirrored'] > res['correlation']:
        print('WARNING: the mirrored angle fits better. A direction flag '
              '(flip_direction_sign) differs from the reference run.')
    if res['correlation'] < 0.8:
        print('WARNING: correlation under 0.8. The fold shape differs from the '
              'reference; check the gearbox state (warm-up) before trusting the offset.')
    if res['max_cell_spread_deg'] > CELL_SPREAD_WARN_DEG:
        print(f'WARNING: cells disagree by up to {res["max_cell_spread_deg"]:.1f} deg. '
              f'A pure encoder shift moves every cell together.')


# Fixed colours: skip_common's current colours are global and follow whichever
# log loaded last, which paints some of today's cells the reference's gray.
QUADRANT_COLOR = {'backdrive': '#e34948', 'forward': '#3987e5'}
REF_COLOR = '#c9c8c3'
NCOLS = 4


def figure(res, ref_name, path):
    rows = res['_rows']
    n = len(rows)
    fold_rows = -(-n // NCOLS)
    fig = plt.figure(figsize=(4.2 * NCOLS, 3.3 * (fold_rows + 1)))
    gs = fig.add_gridspec(fold_rows + 1, NCOLS, hspace=0.55, wspace=0.28)
    off = res['phase_offset_rad']
    for j, row in enumerate(rows):
        ax = fig.add_subplot(gs[j // NCOLS, j % NCOLS])
        w = row['w']
        g = row['grid']
        x, m, _ = c.fold(g['th'] + off, g['y'], FOLD_BINS)
        _, gref = fit_orders_ref_cache(row)
        xr, mr, _ = c.fold(gref['th'], gref['y'], FOLD_BINS)
        ax.plot(xr, mr - np.nanmean(mr), color=REF_COLOR, lw=3, label=ref_name)
        ax.plot(x, m - np.nanmean(m), color=QUADRANT_COLOR[w['quadrant']], lw=1.2,
                label='this log, shifted')
        cr = res['cells'][j]
        ax.set_title(f'{label(w)}\ncorr {cr["correlation"]:.2f}, '
                     f'{cr["vs_joint_deg"]:+.1f}° from joint', fontsize=9)
        ax.set_xlim(0, 360)
        ax.set_xticks(range(0, 361, 90))
        if j // NCOLS == fold_rows - 1 or j + NCOLS >= n:
            ax.set_xlabel(f'motor angle, {ref_name} frame [deg]')
        if j % NCOLS == 0:
            ax.set_ylabel('cell torque, mean removed [Nm]')
        if j == 0:
            ax.legend(frameon=False, fontsize=8)
        ax.grid(color=c.GRID, lw=0.6)

    ax = fig.add_subplot(gs[fold_rows, 0])
    ax.plot(np.degrees(OFFSET_GRID), res['_J'], color=c.INK, lw=1.5)
    ax.axvline(res['phase_offset_deg'], color=c.INK_3, lw=1, ls='--')
    ax.set_xlim(0, 360)
    ax.set_xticks(range(0, 361, 90))
    ax.set_xlabel('candidate offset [deg motor]')
    ax.set_ylabel('fold correlation, all cells')
    ax.set_title(f'offset {res["phase_offset_deg"]:.2f}° '
                 f'({res["phase_offset_rad"]:.4f} rad)', fontsize=9)
    ax.grid(color=c.GRID, lw=0.6)

    ax = fig.add_subplot(gs[fold_rows, 1:3])
    for cr, row in zip(res['cells'], rows):
        conv = cr['convergence_deg_vs_full']
        if conv:
            ax.plot(list(conv), list(conv.values()), marker='o', ms=4, lw=1.2, alpha=0.8,
                    color=QUADRANT_COLOR[row['w']['quadrant']], ls=c.DIR_LS[row['w']['dir']])
    ax.axhspan(-CELL_SPREAD_WARN_DEG, CELL_SPREAD_WARN_DEG, color=c.GRID, alpha=0.6, lw=0)
    ax.axhline(0, color=c.INK_3, lw=0.8)
    ax.set_xscale('log')
    ax.set_xticks(CONVERGENCE_REVS)
    ax.set_xticklabels([str(r) for r in CONVERGENCE_REVS])
    ax.minorticks_off()
    ax.set_xlabel('motor revs used, one cell alone')
    ax.set_ylabel('offset error vs full dwell [deg]')
    ax.set_title(f'how short a re-phasing motion can be (band ±{CELL_SPREAD_WARN_DEG:g}°; '
                 f'±60° jumps are the 6/rev picking the wrong peak)', fontsize=9)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=QUADRANT_COLOR['backdrive'], label='backdrive'),
                       Line2D([], [], color=QUADRANT_COLOR['forward'], label='forward'),
                       Line2D([], [], color=c.INK_3, ls=c.DIR_LS[1], label='output turning +'),
                       Line2D([], [], color=c.INK_3, ls=c.DIR_LS[-1], label='output turning -')],
              frameon=False, fontsize=8, ncol=2)
    ax.grid(color=c.GRID, lw=0.6)

    fig.suptitle(f'{os.path.basename(res["log"])}: motor-angle offset against {ref_name}, '
                 f'{res["speed_rad_s"]:g} rad/s', fontsize=11)
    fig.savefig(path, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def fit_orders_ref_cache(row):
    if '_ref_grid' not in row:
        row['_ref_grid'] = fit_orders(row['ref_d'], row['r'])
    return row['_ref_grid']


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--ref', default='v2', choices=sorted(c.RUNS))
    ap.add_argument('--speed', type=float, default=0.1, help='output rad/s of the cal dwells')
    ap.add_argument('--ref-speed', type=float, default=None,
                    help='reference speed, if not the same (a self-test: v2 0.05 vs v2 0.1)')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    ref = load_log(c.RUNS[args.ref]['log'])
    results = []
    for path in args.logs:
        d = load_log(path)
        res = calibrate(d, ref, args.speed, args.ref_speed)
        for row in res['_rows']:
            row['ref_d'] = ref
        report(res, args.ref)
        stem = os.path.splitext(os.path.basename(path))[0]
        figure(res, args.ref, os.path.join(OUT_DIR, f'{stem}_phase_cal.png'))
        out = {k: v for k, v in res.items() if not k.startswith('_')}
        out['reference'] = args.ref
        with open(os.path.join(OUT_DIR, f'{stem}_phase_cal.json'), 'w') as f:
            json.dump(out, f, indent=2, default=float)
        print(f'wrote {OUT_DIR}/{stem}_phase_cal.{{json,png}}')
        results.append(res)

    if len(results) > 1:
        print(f'\n=== consistency against {os.path.basename(results[0]["log"])} ===')
        for res in results[1:]:
            print(f'{os.path.basename(res["log"])}: '
                  f'{wrap_deg(res["phase_offset_rad"] - results[0]["phase_offset_rad"]):+.2f} deg')
        print('A GUI restart without an Elmo power cycle should read within a few degrees of 0.')


if __name__ == '__main__':
    main()
