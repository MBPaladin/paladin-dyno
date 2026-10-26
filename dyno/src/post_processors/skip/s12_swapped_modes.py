"""The swapped-mode cogging run (v3, 2026-10-22): LOAD in torque mode, DUT in
velocity mode, same speeds as v2 and LOAD torques of 0, +-6.67, +-13.33, +-20 Nm.

What the swap changes. In v2 the LOAD held speed and the DUT held current, so
every angle-locked torque ripple had to appear in the load cell. Here the LOAD
holds torque (a soft output side) and the DUT's velocity loop holds the motor
speed (a stiff input side), so the ripple is supplied by the DUT current
instead, and the cell reads an almost constant torque. That makes this run:

  * a CROSS-CHECK of v2's transmission lines from the other side: the output
    torque is set and the current is measured, instead of the reverse;
  * a direct CURRENT CAPABILITY map: the current the DUT needs for a given
    output torque, and where 30 A runs out;
  * a kt-free EFFICIENCY on a torque grid: at the same |T| and speed,
    eta_f * eta_b = I_backdrive / I_forward;
  * a MOTOR-SIDE COGGING map: DUT current folded over motor angle is the
    current a feedforward table would have to supply, measured by the drive's
    own sensor, and comparable (via the local slope dT_out/dI) to v2's cell
    folds.

The log has no tare (the operator forgot); the cell zero is the 30 s idle at
the start, cross-checked against the LOAD current in its torque-mode dwells.
Only the + direction ran (the operator stopped it at the first - dwell).

    .venv/bin/python dyno/src/post_processors/skip/s12_swapped_modes.py

Outputs (results/cogging_v3/):
  Skip_v3_transmission_crosscheck.png
  Skip_v3_current_capability.png
  Skip_v3_efficiency_torque_grid.png
  Skip_v3_cogging_from_current.png
  Skip_v3_current_orders.png
  Skip_v3_dwell_table.csv
"""

import csv
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import skip_common as c
import skip_orders as so

LOG = os.path.join(c.SKIP_LOGS, 'cogging', 'cogging_vs_torque_and_speed_03', 'skip_cogging_v2_swapped.hdf5')
OUT_DIR = os.path.join(c.SKIP_LOGS, 'results', 'cogging_v3')
IDLE = (2.0, 29.0)            # s, the opening idle hold used as the cell zero
STALL_FRAC = 0.5              # dwells stuck more than this are stalls, not slides
SAT_OK = 0.15                 # forward dwells clipped at the current limit more than this are left out
I_LIMIT = 30.0                # DUT i_cont in this run
FOLD_SPEEDS = (0.05, 0.1)     # velocity loop easily follows order 126 here (<= 52 Hz)
M_BINS = 180
ORDERS = (1, 2, 6, 21, 36, 42, 126)
_CH = ('time', 'load_torque_command', 'dut_velocity_command', 'dut_current', 'load_torque',
       'load_current', 'dut_output_position', 'load_position', 'pt100_rtd_top')


def load():
    with h5py.File(LOG, 'r') as f:
        d = {k: f[k][:].astype(np.float64) for k in _CH}
    d['t'] = d['time'] - d['time'][0]
    d['th_m'] = d['dut_output_position']
    d['th_o'] = d['load_position']
    d['w_m'] = c.speed_from_position(d['th_m'])
    idle = slice(int(IDLE[0] * c.FS_HZ), int(IDLE[1] * c.FS_HZ))
    d['zero'] = float(np.mean(d['load_torque'][idle]))
    d['T_cell'] = d['load_torque'] - d['zero']
    tc, vc, t = d['load_torque_command'], d['dut_velocity_command'], d['t']
    key = np.round(np.nan_to_num(tc), 3) * 1e4 + np.round(np.nan_to_num(vc), 3)
    edges = np.flatnonzero(np.diff(key) != 0) + 1
    dw = []
    for s, e in zip(np.r_[0, edges], np.r_[edges, len(t)]):
        if t[e - 1] - t[s] < c.DWELL_MIN_S or not np.isfinite(vc[s]) or abs(vc[s]) < 1e-6:
            continue
        i0, i1 = s + int(c.DWELL_SETTLE_S * c.FS_HZ), e - int(c.DWELL_TAIL_S * c.FS_HZ)
        sl = slice(i0, i1)
        direction = int(np.sign(vc[s]))
        v_out = round(abs(vc[s]) / c.GEAR_RATIO, 4)
        r = d['w_m'][sl] * direction / abs(vc[s])
        T = float(np.mean(d['T_cell'][sl]))
        P = -direction * T                       # torque LOAD pushes along the motion (+ = backdrive)
        cur = float(np.mean(d['dut_current'][sl]))
        dw.append({'n': len(dw), 'i0': i0, 'i1': i1, 'speed_abs': v_out, 'dir': direction,
                   'speed': direction * v_out, 'motor_speed_nom': abs(vc[s]), 'tcmd': float(tc[s]),
                   'T_cell': T, 'P': P, 'T_lm': float(c.KT_LOAD * np.mean(d['load_current'][sl])),
                   'current': cur, 'i_along': -direction * cur, 'I_std': float(np.std(d['dut_current'][sl])),
                   'sat_frac': float(np.mean(np.abs(d['dut_current'][sl]) > I_LIMIT - 0.5)),
                   'stuck': float(np.mean(r < c.STUCK_FRAC)),
                   'quadrant': 'zero' if tc[s] == 0 else ('backdrive' if P > 0 else 'forward'),
                   'T_nom': float(round(abs(tc[s]), 2)), 'temp_C': float(np.nanmean(d['pt100_rtd_top'][sl]))})
    d['dwells'] = dw
    d['_rs'], d['_fit'] = {}, {}
    print(f'loaded {os.path.basename(LOG)}: {len(dw)} dwells, cell zero {d["zero"]:+.3f} Nm (idle)')
    return d


def v2_lines():
    """v2 per-speed branch fits: P = c - K * i_along per branch, keyed by signed speed."""
    path = os.path.join(c.RUNS['v2']['results'], '02_torque_vs_current', 'Skip_torque_current_fits.csv')
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r['output_speed_rad_s'] != 'pooled':
                out[float(r['output_speed_rad_s'])] = {k: float(r[k]) for k in
                                                         ('K_forward_Nm_per_A', 'K_backdrive_Nm_per_A',
                                                          'c_forward_Nm', 'c_backdrive_Nm')}
    return out


def predicted_current(line, P):
    if P > 0:
        return -(P - line['c_backdrive_Nm']) / line['K_backdrive_Nm_per_A']
    return (line['c_forward_Nm'] - P) / line['K_forward_Nm_per_A']


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {os.path.relpath(os.path.join(OUT_DIR, name), c.DYNO_DIR)}')


def current_fold(d, w):
    """DUT current minus its mean, folded over one motor rev (angle domain)."""
    sl = slice(w['i0'], w['i1'])
    y, _ = c.angle_resample(d['th_m'][sl], d['dut_current'][sl], so.STEP_M)
    th = d['th_m'][w['i0']] + w['dir'] * so.STEP_M * (np.arange(len(y)) + 0.5)
    x = np.arange(len(y))
    y = y - np.polyval(np.polyfit(x, y, 3), x)
    xb, m, _ = c.fold(th, y, M_BINS)
    return xb, m - np.nanmean(m), th, y


def run():
    c.apply_style()
    d = load()
    dw = d['dwells']
    lines = v2_lines()
    sp = sorted({w['speed_abs'] for w in dw})
    cols = dict(zip(sp, [c._SPEED_RAMP[k] for k in np.round(np.linspace(0, len(c._SPEED_RAMP) - 1, len(sp))).astype(int)]))
    for w in dw:
        w['stall'] = w['stuck'] > STALL_FRAC
        w['i_pred_v2'] = predicted_current(lines[w['speed']], w['P']) if w['speed'] in lines else np.nan

    # cell zero cross-check: in torque mode the LOAD current is the output torque less LOAD-side drag
    moving = [w for w in dw if not w['stall']]
    A = np.column_stack([np.ones(len(moving)), [-w['T_lm'] for w in moving]])
    (z_off, k_ratio), *_ = np.linalg.lstsq(A, [w['T_cell'] for w in moving], rcond=None)

    # -- Figure 1: transmission cross-check against v2's lines ------------------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))
    a = ax[0]
    for w in dw:
        a.plot(w['i_along'], w['P'], 'x' if w['stall'] else 'o', color=cols[w['speed_abs']], ms=8,
               mec=cols[w['speed_abs']] if w['stall'] else 'white', mew=1.6 if w['stall'] else 0.8)
    ii = np.linspace(-20, 40, 2)
    for v in (sp[0], sp[-1]):
        L = lines[v]
        ib, iff = np.linspace(-20, 0, 2), np.linspace(0, 40, 2)
        a.plot(ib, L['c_backdrive_Nm'] - L['K_backdrive_Nm_per_A'] * ib, color=cols[v], lw=1.2, ls='--')
        a.plot(iff, L['c_forward_Nm'] - L['K_forward_Nm_per_A'] * iff, color=cols[v], lw=1.2, ls='--')
    a.axvline(I_LIMIT, color='#a8302f', lw=1.2, ls=':')
    a.text(I_LIMIT + 0.3, 18, f'{I_LIMIT:g} A limit', color='#a8302f', fontsize=9)
    a.axhline(0, color=c.INK_3, lw=0.8)
    a.axvline(0, color=c.INK_3, lw=0.8)
    a.set_xlabel('DUT current along the motion (A)  [+ = DUT drives]')
    a.set_ylabel('Torque LOAD applies along the motion (Nm)')
    a.set_title('v3 operating points on v2 lines (dashed: slowest and fastest speed)')
    handles = [Line2D([], [], color=cols[v], marker='o', ls='', label=f'{v:g} rad/s') for v in sp]
    handles += [Line2D([], [], color=c.INK_2, marker='x', ls='', label='stalled at the current limit')]
    a.legend(handles=handles, loc='upper right')
    b = ax[1]
    for w in dw:
        if not w['stall']:
            b.plot(w['P'], w['i_along'] - w['i_pred_v2'], 'o', color=cols[w['speed_abs']], ms=8, mec='white')
    b.axhline(0, color=c.INK_3, lw=0.8)
    b.axvline(0, color=c.INK_3, lw=0.8)
    b.set_xlabel('Torque LOAD applies along the motion (Nm)  [- = DUT drives]')
    b.set_ylabel('v3 current minus v2 line prediction (A)')
    b.set_title('Where the two runs disagree')
    t3 = (np.nanmin(d['pt100_rtd_top']), np.nanmax(d['pt100_rtd_top']))
    fig.suptitle('Skip v3 (output torque set, current measured) against v2 (current set, torque measured)')
    c.note(fig, f'Cell zero = 30 s idle ({d["zero"]:+.2f} Nm untared); against LOAD current x kt in its torque-mode dwells the '
                f'cell reads {k_ratio:.3f} x with a {z_off:+.2f} Nm offset. RTD top {t3[0]:.1f}-{t3[1]:.1f} C (v2: 23.1-26.7 C). '
                'Positive residual = DUT needed more current than v2 predicts.', y=0.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    save(fig, 'Skip_v3_transmission_crosscheck.png')

    # -- Figure 2: current capability --------------------------------------------------
    fig, a = plt.subplots(figsize=(10, 6))
    for v in sp:
        pts = sorted([w for w in dw if w['speed_abs'] == v and w['tcmd'] != 0], key=lambda w: w['P'])
        for q, mk in (('forward', 'o'), ('backdrive', 's')):
            qq = [w for w in pts if w['quadrant'] == q]
            a.plot([abs(w['P']) for w in qq], [abs(w['current']) for w in qq], '-' + mk, color=cols[v], ms=7,
                   mfc=cols[v] if q == 'forward' else 'white')
    for v in (sp[0], sp[-1]):
        L = lines[v]
        TT = np.linspace(0, 22, 20)
        a.plot(TT, (TT + L['c_forward_Nm']) / L['K_forward_Nm_per_A'], color=cols[v], ls=':', lw=1.2)
    a.axhline(I_LIMIT, color='#a8302f', ls='--', lw=1.2)
    a.set_xlabel('Output torque (Nm)')
    a.set_ylabel('|DUT current| (A)')
    a.set_title('Current the DUT needs for a given output torque')
    a.legend(handles=[Line2D([], [], color=c.INK_2, marker='o', label='forward: DUT drives the load'),
                      Line2D([], [], color=c.INK_2, marker='s', mfc='white', label='backdrive: DUT holds back the load'),
                      Line2D([], [], color=c.INK_2, ls=':', label='v2 forward line, extrapolated'),
                      Line2D([], [], color='#a8302f', ls='--', label=f'{I_LIMIT:g} A i_cont')]
              + [Line2D([], [], color=cols[v], lw=2, label=f'{v:g} rad/s') for v in sp], loc='upper left', fontsize=8)
    a.set_ylim(0, 45)
    c.note(fig, 'Forward needs ~3x the current of backdrive for the same output torque (efficiency enters once each way). '
                'The +20 Nm forward points sit on the limit because they stalled.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, 'Skip_v3_current_capability.png')

    # -- Figure 3: kt-free efficiency on the torque grid --------------------------------
    Ts = sorted({w['T_nom'] for w in dw if w['T_nom'] > 0})
    G = np.full((len(Ts), len(sp)), np.nan)
    for i, T in enumerate(Ts):
        for j, v in enumerate(sp):
            f = [w for w in dw if w['speed_abs'] == v and w['T_nom'] == T and w['quadrant'] == 'forward' and not w['stall']]
            bk = [w for w in dw if w['speed_abs'] == v and w['T_nom'] == T and w['quadrant'] == 'backdrive']
            if f and bk and f[0]['sat_frac'] < SAT_OK:
                # torque-normalised so small differences in the achieved |T| do not leak in
                G[i, j] = (abs(bk[0]['current']) / abs(bk[0]['P'])) / (abs(f[0]['current']) / abs(f[0]['P']))
    fig, a = plt.subplots(figsize=(10, 4.2))
    E = np.sqrt(G)
    im = a.imshow(np.ma.masked_invalid(E), origin='lower', aspect='auto', cmap='Blues', vmin=0.3, vmax=1.0)
    for i in range(len(Ts)):
        for j in range(len(sp)):
            txt = f'{100 * E[i, j]:.0f}%' if np.isfinite(E[i, j]) else 'current\nlimit'
            a.text(j, i, txt, ha='center', va='center', fontsize=9 if np.isfinite(E[i, j]) else 7.5,
                   fontweight='bold' if np.isfinite(E[i, j]) else 'normal', color=c.INK if not np.isfinite(E[i, j]) or E[i, j] < 0.7 else 'white')
    a.set_xticks(range(len(sp)))
    a.set_xticklabels([f'{v:g}' for v in sp])
    a.set_yticks(range(len(Ts)))
    a.set_yticklabels([f'{T:g} Nm' for T in Ts])
    a.grid(False)
    a.set_xlabel('Output speed (rad/s)')
    a.set_ylabel('Output torque')
    a.set_title('sqrt(eta_f x eta_b) = sqrt(I_backdrive / I_forward) at equal output torque (kt-free)')
    fig.colorbar(im, ax=a, fraction=0.03, pad=0.01)
    c.note(fig, f'Includes the no-load drag. Forward 13.3 Nm rode the 30 A limit on its ripple peaks (left out when clipped '
                f'> {100 * SAT_OK:.0f}% of the time; the kept ones are biased slightly high by the clipping); 20 Nm forward '
                'stalled at every speed.', y=-0.02)
    save(fig, 'Skip_v3_efficiency_torque_grid.png')

    # -- Figure 4: cogging from DUT current vs v2's cell -------------------------------
    c.use_run('v2')
    d2 = c.load(verbose=False)
    c.use_run('v1')
    pairs = (('backdrive', 20.0, -12.5), ('backdrive', 13.33, -10), ('backdrive', 6.67, -5),
             ('zero', 0.0, 0), ('forward', 6.67, 12.5))
    curves = {}
    for v in FOLD_SPEEDS:
        line = lines[v]
        for q, T, i2 in pairs:
            w3 = next(w for w in dw if w['speed_abs'] == v and w['T_nom'] == T and w['quadrant'] == q)
            w2 = next(w for w in d2['dwells'] if w['speed'] == v and w['i_along_nom'] == i2)
            x, m3, _, _ = current_fold(d, w3)
            # local slope of output torque per DUT amp in this quadrant converts amps to output Nm
            K = line['K_backdrive_Nm_per_A'] if q == 'backdrive' else line['K_forward_Nm_per_A']
            sl = slice(w2['i0'], w2['i1'])
            y2, _ = c.angle_resample(d2['th_m'][sl], d2['T_cell'][sl], so.STEP_M)
            th2 = d2['th_m'][w2['i0']] + w2['dir'] * so.STEP_M * (np.arange(len(y2)) + 0.5)
            xx = np.arange(len(y2))
            y2 = y2 - np.polyval(np.polyfit(xx, y2, 3), xx)
            _, m2, _ = c.fold(th2, y2, M_BINS)
            curves[(v, q, T)] = (x, m2 - np.nanmean(m2), m3 * K, K, i2)

    # The motor encoder zero is not kept between power-ups, so v2 and v3 differ by
    # one unknown angle -- the same for every dwell. Fit that one shift (and the
    # sign relating cell torque to DUT current) jointly over all pairs, then judge
    # each pair at it: a per-pair shift could make anything look alike.
    def total(shift, sgn):
        return sum(np.dot(np.nan_to_num(m2), np.nan_to_num(np.roll(sgn * m3, shift)))
                   / np.sqrt(np.nansum(m2 ** 2) * np.nansum(m3 ** 2)) for _, m2, m3, _, _ in curves.values())
    shift, sgn = max(((k, s_) for k in range(M_BINS) for s_ in (-1, 1)), key=lambda ks: total(*ks))

    qcol = {'backdrive': '#a8302f', 'forward': '#1c5cab', 'zero': c.INK}
    fig, ax = plt.subplots(len(FOLD_SPEEDS), len(pairs), figsize=(4 * len(pairs), 3.6 * len(FOLD_SPEEDS)),
                           sharex=True, squeeze=False)
    rec_pairs = {}
    for r_, v in enumerate(FOLD_SPEEDS):
        for k, (q, T, i2) in enumerate(pairs):
            a = ax[r_, k]
            x, m2, m3n, K, i2 = curves[(v, q, T)]
            m3a = np.roll(sgn * m3n, shift)
            ok = np.isfinite(m2) & np.isfinite(m3a)
            corr = float(np.corrcoef(m2[ok], m3a[ok])[0, 1])
            a.plot(x, m2, color=c.INK_3, lw=1.4, label=f'v2 cell, {abs(i2):g} A')
            a.plot(x, m3a, color=qcol[q], lw=1.4, label=f'v3 current x {K:.2f} Nm/A')
            a.set_title(f'{v:g} rad/s, {q} {T:g} Nm: r = {corr:.2f}', fontsize=9.5)
            a.axhline(0, color=c.INK_3, lw=0.6)
            a.set_xlim(0, 360)
            a.set_xticks(range(0, 361, 90))
            if r_ == len(FOLD_SPEEDS) - 1:
                a.set_xlabel('Motor angle (deg)')
            if k == 0:
                a.set_ylabel('Ripple at the output (Nm)')
            a.legend(loc='upper right', fontsize=7)
            rec_pairs[f'{v:g}_{q}_{T:g}'] = {'v3_current_pkpk_A': float((np.nanmax(m3n) - np.nanmin(m3n)) / K),
                                             'v3_as_output_pkpk_Nm': float(np.nanmax(m3n) - np.nanmin(m3n)),
                                             'v2_cell_pkpk_Nm': float(np.nanmax(m2) - np.nanmin(m2)), 'corr': corr}
    rec_pairs['global_shift_deg'] = float(shift * 360 / M_BINS)
    rec_pairs['sign'] = int(sgn)
    fig.suptitle('Skip cogging seen two ways: v2 load cell (DUT current held) vs v3 DUT current (output torque held)')
    c.note(fig, f'v3 = DUT current ripple x the local slope dT_out/dI from v2. One shift ({shift * 360 / M_BINS:.0f} deg) and '
                f'sign ({sgn:+d}) fitted jointly over all {len(curves)} pairs in case the motor encoder zero moved between '
                'runs; it did not (a ~0 shift). r = correlation at that alignment. The 0 Nm pair is not like for like: '
                'v2 at 0 A is backdriven by the LOAD, v3 at 0 Nm is driven lightly by the DUT.', y=-0.01)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, 'Skip_v3_cogging_from_current.png')

    # -- Figure 5: current ripple orders vs output torque -------------------------------
    amps = {}
    for w in dw:
        if w['speed_abs'] not in FOLD_SPEEDS or w['stall'] or w['sat_frac'] > 0.05:
            continue
        sl = slice(w['i0'], w['i1'])
        y, _ = c.angle_resample(d['th_m'][sl], d['dut_current'][sl], so.STEP_M)
        th = d['th_m'][w['i0']] + w['dir'] * so.STEP_M * (np.arange(len(y)) + 0.5)
        out, _, _, _ = c.lsq_orders([(f'o{k}', th, k) for k in ORDERS], y, c.legendre_nuisance(len(y), 8))
        amps[w['n']] = {k: out[f'o{k}'][0] for k in ORDERS}
    fig, ax = plt.subplots(1, len(ORDERS), figsize=(3.1 * len(ORDERS), 3.8), sharex=True)
    for a, k in zip(ax, ORDERS):
        for v, mk in zip(FOLD_SPEEDS, ('o', 's')):
            pts = sorted([w for w in dw if w['n'] in amps and w['speed_abs'] == v], key=lambda w: w['P'])
            a.plot([-w['P'] for w in pts], [amps[w['n']][k] for w in pts], '-' + mk, color=cols[v], ms=5,
                   label=f'{v:g} rad/s')
        a.set_title(f'order {k}' + (' (1x el)' if k == 21 else ' (2x el)' if k == 42 else ' (6x el)' if k == 126 else ''),
                    fontsize=10)
        a.set_ylim(bottom=0)
        a.axvline(0, color=c.INK_3, lw=0.8)
        a.set_xlabel('Output torque (Nm)\n[- backdrive, + forward]')
    ax[0].set_ylabel('DUT current ripple amplitude (A)')
    ax[0].legend(loc='upper left', fontsize=8)
    fig.suptitle('Skip v3: angle-locked DUT current ripple per motor-rev order (what an anti-ripple table would inject)')
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, 'Skip_v3_current_orders.png')

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'Skip_v3_dwell_table.csv'), 'w') as fh:
        cols_ = ['speed_abs', 'tcmd', 'quadrant', 'T_cell', 'P', 'T_lm', 'current', 'i_along', 'i_pred_v2',
                 'I_std', 'sat_frac', 'stuck', 'stall', 'temp_C']
        fh.write('n,' + ','.join(cols_) + '\n')
        for w in dw:
            fh.write(f'{w["n"]},' + ','.join(f'{w[k]:.5g}' if isinstance(w[k], float) else str(w[k]) for k in cols_) + '\n')

    c.RESULTS_DIR = OUT_DIR
    c.KEY_NUMBERS_PATH = os.path.join(OUT_DIR, 'skip_key_numbers.json')
    resid = [w['i_along'] - w['i_pred_v2'] for w in dw if not w['stall']]
    c.record('swapped_modes', {
        'cell_zero_idle_Nm': d['zero'], 'cell_vs_load_ratio': float(k_ratio), 'cell_vs_load_offset_Nm': float(z_off),
        'rtd_top_C': [float(t3[0]), float(t3[1])],
        'residual_vs_v2_A': {q: float(np.mean([w['i_along'] - w['i_pred_v2'] for w in dw if w['quadrant'] == q and not w['stall']]))
                             for q in ('forward', 'backdrive', 'zero')},
        'residual_vs_v2_A_by_speed_forward': {f'{v:g}': float(np.mean([w['i_along'] - w['i_pred_v2'] for w in dw if w['speed_abs'] == v and w['quadrant'] == 'forward' and not w['stall']])) for v in sp},
        'no_load_current_by_speed_A': {f'{v:g}': float(np.mean([w['i_along'] for w in dw if w['speed_abs'] == v and w['quadrant'] == 'zero'])) for v in sp},
        'stalled_dwells': [(w['speed_abs'], w['tcmd']) for w in dw if w['stall']],
        'eta_sym_torque_grid': {f'{T:g}Nm': {f'{v:g}': float(E[i, j]) for j, v in enumerate(sp)} for i, T in enumerate(Ts)},
        'cogging_pairs': rec_pairs,
        'current_order_amps_A': {f'{w["speed_abs"]:g}_{w["tcmd"]:+g}': {str(k): float(a_) for k, a_ in amps[w['n']].items()}
                                 for w in dw if w['n'] in amps},
        'n_resid': len(resid),
    })


if __name__ == '__main__':
    run()
