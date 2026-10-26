"""Fit the `dut_compensation:` config block (dyno/src/dut_compensation.py) from v2.

What it fits, all in the DUT command frame (I = dut_torque_command / kt 0.02,
positive tau = what the cell reads for a positive command):

  base         per direction, T = K_f*I (forward) or K_b*I (backdrive) - D*s,
               least squares over v2's dwells at <= 0.25 rad/s, 0 A included.
  speed_table  the same fit at each v2 speed (0.05 ... 4 rad/s).
  harmonics    per (direction, quadrant) cell and order k, the complex order
               coefficient of the cell torque at each current (<= 0.25 rad/s,
               as the angle stages), fitted as (a0 + a1*|T|) * exp(i*phi) with
               one phase per cell. Phases are in v2's motor-angle frame; today's
               phase_offset_rad (s14_phase_cal.py) maps the live angle onto it.
               The 0 A dwell belongs to both quadrants of its direction.

Checks it prints:
  - fit rms of the base and speed lines (Nm);
  - how much of each cell's order content the shared-phase model keeps;
  - the base lines against the 2026-10-22 efficiency run, which reaches
    +/-20 Nm (swapped modes, ~3 C colder, own cell zero): the residual
    current vs torque says whether the forward line can be extrapolated to
    the 12 Nm A/B points, where v2's forward data stops at ~6 Nm.

Run:
    .venv/bin/python dyno/src/post_processors/skip/s15_comp_fit.py [--phase-offset RAD]

Outputs (results/comp_fit/):
  dut_compensation_fit.yaml   paste under the rig config (phase_offset_rad from s14)
  Skip_comp_fit.png           base lines, speed table, harmonic amplitude vs load
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
import yaml

import skip_common as c
import s14_phase_cal as pc

OUT_DIR = os.path.join(c.SKIP_LOGS, 'results', 'comp_fit')
EFFICIENCY_LOG = os.path.join(c.SKIP_LOGS, 'efficiency', 'efficiency', 'skip_efficiency.hdf5')
BASE_MAX_SPEED = 0.25
# 36 and 42 (2x electrical) were tried: one phase per cell keeps only 1-31% of
# their content (vs 85-100% for 1, 2, 6), so injecting them adds error.
GROUPS = {'h1': (1,), 'h6': (6,), 'hx': (2,)}
FIT_ORDERS = tuple(range(1, 21)) + (21, 36, 42)
PROFILES = {
    'C0_naive': [],
    'C1_base': ['base'],
    'C2_speed': ['base', 'speed'],
    'C3_h1': ['base', 'h1'],
    'C4_h6': ['base', 'h6'],
    'C5_hx': ['base', 'hx'],
    'C6_all': ['base', 'speed', 'h1', 'h6', 'hx'],
    # C1 again under its own name, run last at each point: a drift check that
    # keeps its log flag distinct from the first C1 hold.
    'C1_repeat': ['base'],
}
QUADRANT_COLOR = {'backdrive': '#e34948', 'forward': '#3987e5'}
DIR_NAME = {1: 'pos', -1: 'neg'}


def command_current(w):
    return w['tcmd'] / c.KT_CONFIGURED


def fit_branch(dwells):
    """Least squares T = K_f*I*[fwd] + K_b*I*[bd] - D*s for one direction.
    Returns (K_f, K_b, D, rms)."""
    X, y = [], []
    for w in dwells:
        I = command_current(w)
        fwd = w['quadrant'] == 'forward'
        bd = w['quadrant'] == 'backdrive'
        X.append([I * fwd, I * bd, -w['dir']])
        y.append(w['T_cell'])
    X, y = np.array(X, float), np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    rms = float(np.sqrt(np.mean((y - X @ coef) ** 2)))
    return float(coef[0]), float(coef[1]), float(coef[2]), rms


def fit_base(d, speeds):
    out = {}
    for s in (1, -1):
        ws = [w for w in d['dwells'] if w['dir'] == s and w['speed_abs'] in speeds]
        out[s] = fit_branch(ws)
    return out


def order_coefs(d, w):
    """Complex order coefficients C_k = A*exp(i*phi), x ~ A cos(k*theta + phi)."""
    sl = c.dwell_slice(w)
    th, y = d['th_m'][sl], d['T_cell'][sl]
    yr, _ = c.angle_resample(th, y, pc.STEP_M)
    n = len(yr)
    tg = th[0] + w['dir'] * pc.STEP_M * (np.arange(n) + 0.5)
    amps, _, _, _ = c.lsq_orders([(k, tg, k) for k in FIT_ORDERS], yr,
                                 c.legendre_nuisance(n, pc.SLOW_DEGREE))
    return {k: amps[k][0] * np.exp(1j * amps[k][1]) for k in FIT_ORDERS}


def fit_harmonics(d):
    """{(dir, quadrant): {k: (a0, a1, phi, kept, points)}} over the slow dwells."""
    slow = [w for w in d['dwells'] if w['speed_abs'] <= BASE_MAX_SPEED + 1e-9]
    coefs = {w['n']: order_coefs(d, w) for w in slow}
    out = {}
    for s in (1, -1):
        for q in ('forward', 'backdrive'):
            ws = [w for w in slow if w['dir'] == s and w['quadrant'] in (q, 'zero')]
            load = np.array([abs(w['T_cell']) for w in ws])
            out[(s, q)] = {}
            for k in sorted({k for ks in GROUPS.values() for k in ks}):
                C = np.array([coefs[w['n']][k] for w in ws])
                X = np.column_stack([np.ones_like(load), load])
                uv, *_ = np.linalg.lstsq(X.astype(complex), C, rcond=None)
                phi = float(np.angle(uv[0] + uv[1] * load.max()))
                rot = np.exp(-1j * phi)
                a0, a1 = float((uv[0] * rot).real), float((uv[1] * rot).real)
                model = (a0 + a1 * load) * np.exp(1j * phi)
                kept = 1.0 - float(np.sum(np.abs(C - model) ** 2) / np.sum(np.abs(C) ** 2))
                out[(s, q)][k] = (a0, a1, phi, kept, (load, np.abs(C)))
    return out


def efficiency_check(base):
    """Residual current of the base lines against the efficiency run's plateaus
    at <= 0.25 rad/s output. Returns rows (dir, quadrant, T, I_meas, I_pred)."""
    rows = []
    with h5py.File(EFFICIENCY_LOG, 'r') as f:
        vc = f['dut_velocity_command'][:].astype(float)
        tq = f['load_torque_command'][:].astype(float)
        T = f['load_torque'][:].astype(float)
        I = -f['dut_current'][:].astype(float)   # drive frame -> command frame (flip_torque_sign)
    key = np.round(np.nan_to_num(vc), 3) * 1e4 + np.round(np.nan_to_num(tq), 2)
    edges = np.flatnonzero(np.diff(key) != 0) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(key)]):
        if b - a < 4000 or not np.isfinite(vc[a]) or abs(vc[a]) < 1e-3:
            continue
        v_out = vc[a] / c.GEAR_RATIO
        if abs(v_out) > BASE_MAX_SPEED + 1e-6 or not np.isclose(abs(v_out), [0.05, 0.1, 0.25], atol=1e-3).any():
            continue
        sl = slice(a + 1500, b)
        s = int(np.sign(v_out))
        Tm, Im = float(T[sl].mean()), float(I[sl].mean())
        K_f, K_b, D, _ = base[s]
        num = Tm + D * s
        q = 'forward' if s * num > 0 else 'backdrive'
        rows.append((s, q, Tm, Im, num / (K_f if q == 'forward' else K_b)))
    return rows


def to_config(base, table, harm, speeds, phase_offset):
    def pair(vals):
        return [round(vals[1], 4), round(vals[-1], 4)]
    cfg = {
        'enabled': True,
        'phase_offset_rad': round(float(phase_offset), 4),
        'base': {'K_f': pair({s: base[s][0] for s in base}),
                 'K_b': pair({s: base[s][1] for s in base}),
                 'drag_nm': pair({s: base[s][2] for s in base})},
        'speed_table': {
            'speed_rad_s': [float(v) for v in speeds],
            'K_f': [pair({s: table[v][s][0] for s in (1, -1)}) for v in speeds],
            'K_b': [pair({s: table[v][s][1] for s in (1, -1)}) for v in speeds],
            'drag_nm': [pair({s: table[v][s][2] for s in (1, -1)}) for v in speeds],
        },
        'harmonics': [],
        'profiles': PROFILES,
    }
    for group, orders in GROUPS.items():
        for k in orders:
            cells = {}
            for (s, q), byk in harm.items():
                a0, a1, phi, _, _ = byk[k]
                cells[f'{DIR_NAME[s]}_{q}'] = {'a0': round(a0, 4), 'a1': round(a1, 5),
                                              'phase': round(phi, 4)}
            cfg['harmonics'].append({'group': group, 'order': k, 'cells': cells})
    return cfg


def figure(d, base, table, speeds, harm, eff_rows, path):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))
    # base lines
    for j, s in enumerate((1, -1)):
        a = ax[0, j]
        K_f, K_b, D, rms = base[s]
        for w in d['dwells']:
            if w['dir'] != s or w['speed_abs'] > BASE_MAX_SPEED + 1e-9:
                continue
            col = QUADRANT_COLOR.get(w['quadrant'], c.INK_3)
            a.plot(command_current(w), w['T_cell'], 'o', ms=6, color=col, mec='white', mew=1)
        for (es, q, Tm, Im, _) in eff_rows:
            if es == s:
                a.plot(Im, Tm, 'x', ms=6, color=QUADRANT_COLOR[q], alpha=0.7)
        I = np.linspace(-45, 45, 200)
        fwd = s * I > 0
        a.plot(I, np.where(fwd, K_f * I, K_b * I) - D * s, color=c.INK, lw=1.2)
        a.axhline(0, color=c.INK_3, lw=0.6)
        a.axvline(0, color=c.INK_3, lw=0.6)
        a.set_title(f'output turning {"+" if s > 0 else "-"}: K_f {K_f:.3f}, K_b {K_b:.3f} Nm/A, '
                    f'D {D:.2f} Nm\nrms {rms:.2f} Nm. o v2 <= 0.25 rad/s, x efficiency run', fontsize=9)
        a.set_xlabel('DUT current, command frame [A]')
        a.set_ylabel('cell torque [Nm]')
        a.grid(color=c.GRID, lw=0.6)
    # efficiency residual
    a = ax[0, 2]
    for (s, q, Tm, Im, Ip) in eff_rows:
        a.plot(abs(Tm), abs(Im) - abs(Ip), c.DIR_MARKER[s], ms=6, color=QUADRANT_COLOR[q], alpha=0.8)
    a.axhline(0, color=c.INK_3, lw=0.8)
    a.set_xlabel('|cell torque| [Nm]')
    a.set_ylabel('|I measured| - |I v2 base line| [A]')
    a.set_title('efficiency run against the v2 base lines\n(^ +v, v -v; red backdrive, blue forward)',
                fontsize=9)
    a.grid(color=c.GRID, lw=0.6)
    # speed table
    a = ax[1, 0]
    for key, col, lab in (('K_f', QUADRANT_COLOR['forward'], 'K_f'),
                          ('K_b', QUADRANT_COLOR['backdrive'], 'K_b')):
        idx = {'K_f': 0, 'K_b': 1}[key]
        for s in (1, -1):
            a.plot(speeds, [table[v][s][idx] for v in speeds], marker=c.DIR_MARKER[s],
                   ls=c.DIR_LS[s], color=col, label=f'{lab} {"+" if s > 0 else "-"}v')
    a.set_xscale('log')
    a.set_xlabel('output speed [rad/s]')
    a.set_ylabel('output Nm per DUT amp')
    a.set_title('speed table: branch slopes', fontsize=9)
    a.legend(frameon=False, fontsize=8, ncol=2)
    a.grid(color=c.GRID, lw=0.6)
    a = ax[1, 1]
    for s in (1, -1):
        a.plot(speeds, [table[v][s][2] for v in speeds], marker=c.DIR_MARKER[s], ls=c.DIR_LS[s],
               color=c.INK, label=f'{"+" if s > 0 else "-"}v')
    a.set_xscale('log')
    a.set_xlabel('output speed [rad/s]')
    a.set_ylabel('drag D [Nm]')
    a.set_title('speed table: drag', fontsize=9)
    a.legend(frameon=False, fontsize=8)
    a.grid(color=c.GRID, lw=0.6)
    # harmonic amplitude vs load, 1/rev and 6/rev
    a = ax[1, 2]
    for (s, q), byk in harm.items():
        for k, mk in ((1, 'o'), (6, 's')):
            a0, a1, _, _, (load, amp) = byk[k]
            a.plot(load, amp, mk, ms=5, color=QUADRANT_COLOR[q], alpha=0.75,
                   mfc='white' if s < 0 else QUADRANT_COLOR[q])
            L = np.linspace(0, load.max(), 20)
            a.plot(L, a0 + a1 * L, ls=c.DIR_LS[s], color=QUADRANT_COLOR[q], lw=1, alpha=0.75)
    a.set_xlabel('|cell torque| [Nm]')
    a.set_ylabel('order amplitude [Nm]')
    a.set_title('ripple vs load: o 1/rev, s 6/rev; filled +v, open -v', fontsize=9)
    a.grid(color=c.GRID, lw=0.6)
    fig.suptitle('Skip DUT compensation fit from v2', fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor='white')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--phase-offset', type=float, default=0.0,
                    help='phase_offset_rad from s14_phase_cal.py for today')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    d = c.load(c.RUNS['v2']['log'])
    all_speeds = c.speeds(d)
    slow = [v for v in all_speeds if v <= BASE_MAX_SPEED + 1e-9]
    base = fit_base(d, slow)
    table = {v: fit_base(d, [v]) for v in all_speeds}
    harm = fit_harmonics(d)
    eff = efficiency_check(base)

    print('\nbase (<= 0.25 rad/s):')
    for s in (1, -1):
        K_f, K_b, D, rms = base[s]
        print(f'  {"+" if s > 0 else "-"}v  K_f {K_f:.4f}  K_b {K_b:.4f}  D {D:+.3f} Nm  rms {rms:.3f} Nm')
    print('speed table (+v / -v):')
    for v in all_speeds:
        t = table[v]
        print(f'  {v:5.2f} rad/s  K_f {t[1][0]:.3f}/{t[-1][0]:.3f}  K_b {t[1][1]:.3f}/{t[-1][1]:.3f}  '
              f'D {t[1][2]:+.2f}/{t[-1][2]:+.2f}  rms {t[1][3]:.2f}/{t[-1][3]:.2f}')
    print('harmonics (a0 + a1*|T|, shared phase; kept = order content the model keeps):')
    for (s, q), byk in harm.items():
        txt = '  '.join(f'k{k}: {a0:+.2f}{a1:+.3f}|T| {np.degrees(phi):6.1f}° {kept:4.0%}'
                        for k, (a0, a1, phi, kept, _) in byk.items())
        print(f'  {DIR_NAME[s]}_{q:9s} {txt}')
    # Magnitudes: a signed mean cancels between the two directions.
    print('efficiency run against the base lines, |I measured| - |I predicted| (A):')
    for q in ('forward', 'backdrive'):
        for lo, hi in ((0, 5), (5, 10), (10, 15), (15, 21)):
            r = [abs(Im) - abs(Ip) for (s, qq, Tm, Im, Ip) in eff if qq == q and lo <= abs(Tm) < hi]
            if r:
                print(f'  {q:9s} |T| {lo:2d}-{hi:2d} Nm: mean {np.mean(r):+.2f} A '
                      f'({np.mean(r) / max(np.mean([abs(Ip) for (s, qq, Tm, Im, Ip) in eff if qq == q and lo <= abs(Tm) < hi]), 1e-9):+.0%}), n {len(r)}')

    cfg = to_config(base, table, harm, all_speeds, args.phase_offset)
    path = os.path.join(OUT_DIR, 'dut_compensation_fit.yaml')
    with open(path, 'w') as f:
        f.write('# Generated by dyno/src/post_processors/skip/s15_comp_fit.py from v2.\n'
                '# phase_offset_rad comes from s14_phase_cal.py for the session.\n')
        yaml.safe_dump({'dut_compensation': cfg}, f, sort_keys=False, default_flow_style=None, width=100)
    figure(d, base, table, all_speeds, harm, eff, os.path.join(OUT_DIR, 'Skip_comp_fit.png'))
    print(f'wrote {path}\nwrote {OUT_DIR}/Skip_comp_fit.png')


if __name__ == '__main__':
    main()
