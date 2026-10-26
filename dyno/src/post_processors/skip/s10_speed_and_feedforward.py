"""What changes with speed, what the LOAD feedforward did, and how this run
compares with the 2026-10-19 one at the grid points both ran.

v2 (2026-10-21) added 0.25-4 rad/s to the grid. Above ~0.25 rad/s the angle-
domain machinery of s04-s07 cannot resolve the electrical orders at 1 kHz, so
everything here uses dwell means, percentiles and LOW mechanical orders fitted
in time against the motor encoder (only orders whose frequency stays under
ORDER_MAX_HZ, below the ~283-420 Hz rig and cell lines):

  TRANSMISSION  per-speed branch slopes (forward / backdrive), symmetric-model
                efficiency and kt, no-load drag; drag fitted as Coulomb + viscous
  RIPPLE        p1-p99 swing, std, and the 1, 2, 6 per motor rev amplitudes
  FEEDFORWARD   share of the LOAD torque the logged feedforward supplied, what
                the velocity loop had to add, and the LOAD speed error
  V1 vs V2      same speed and current: mean torque, std, stuck time, ripple

Outputs (results/10_speed_and_feedforward/):
  Skip_transmission_vs_speed.png
  Skip_ripple_vs_speed.png
  Skip_feedforward_vs_speed.png
  Skip_run_comparison.png        (needs the v1 CSVs in results/)
  Skip_speed_per_dwell.csv
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import skip_common as c
import s02_torque_current as s02

SUB = '10_speed_and_feedforward'
ORDER_MAX_HZ = 150.0
LOW_ORDERS = (1, 2, 6)


def low_order_fit(d, w):
    """Amplitudes of LOW_ORDERS per motor rev, least squares in time with a
    cubic drift, skipping any order above ORDER_MAX_HZ at this speed."""
    sl = c.dwell_slice(w)
    y = d['T_cell'][sl]
    th = d['th_m'][sl]
    f_rev = w['motor_speed_nom'] / c.TWO_PI
    ks = [k for k in LOW_ORDERS if k * f_rev < ORDER_MAX_HZ]
    cols = [(f'm{k}', th, k) for k in ks]
    amps, _, _, _ = c.lsq_orders(cols, y, c.legendre_nuisance(len(y), 3))
    return {k: (amps[f'm{k}'][0] if k in ks else np.nan) for k in LOW_ORDERS}


def per_dwell(d):
    rows = []
    for w in d['dwells']:
        sl = c.dwell_slice(w)
        P = -w['dir'] * d['T_cell'][sl]
        p1, p99 = np.percentile(P, [1, 99])
        r = d['w_m'][sl] * w['dir'] / w['motor_speed_nom']
        verr = d['load_velocity'][sl] - d['load_velocity_command'][sl]
        amps = low_order_fit(d, w)
        lm = w['T_lm_cmd']
        rows.append({
            'w': w, 'P': w['P'], 'std': float(np.std(P)), 'swing': float(p99 - p1),
            'stuck': float(np.mean(r < c.STUCK_FRAC)), 'speed_ripple': float(np.std(r)),
            'm1': amps[1], 'm2': amps[2], 'm6': amps[6],
            'ff': w['ff'], 'lm': lm, 'loop': lm - w['ff'],
            'verr_rms': float(np.sqrt(np.mean(verr ** 2))),
            'verr_rel': float(np.sqrt(np.mean(verr ** 2)) / w['speed_abs']),
        })
    return rows


def drag_fit(groups):
    """No-load drag vs speed, both directions: P0 = Fc + b*|v|."""
    v = np.array([vs[0] for vs in groups])
    p0 = np.array([g['P0'] for g in groups.values()])
    A = np.column_stack([np.ones_like(v), v])
    (fc, b), *_ = np.linalg.lstsq(A, p0, rcond=None)
    return float(fc), float(b), float(np.sqrt(np.mean((p0 - A @ [fc, b]) ** 2)))


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return list(csv.DictReader(fh))


def run(d):
    c.apply_style()
    dw = d['dwells']
    sp = c.speeds(d)
    cur = c.currents_along(d)
    R = per_dwell(d)
    groups = {(v, s): s02.branch_fit([w for w in dw if w['speed_abs'] == v and w['dir'] == s])
              for v in sp for s in (-1, 1)}
    fc, b_visc, drag_rms = drag_fit(groups)

    def by_speed(key):
        """Mean of a branch_fit quantity over the two directions, per speed."""
        out = []
        for v in sp:
            vals = []
            for s in (-1, 1):
                g = groups[(v, s)]
                vals.append(g[key[0]][key[1]] if isinstance(key, tuple) else g[key])
            out.append(np.mean(vals))
        return np.array(out)

    # -- Figure 1: transmission vs speed ------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0))
    a = ax[0]
    for ia in cur:
        ys = [np.mean([r['P'] for r in R if r['w']['speed_abs'] == v and r['w']['i_along_nom'] == ia]) for v in sp]
        a.plot(sp, ys, '-o', color=c.current_color(ia), ms=6, label=c.current_label(ia))
    a.set_xscale('log')
    a.set_xlabel('Output speed (rad/s, log)')
    a.set_ylabel('Torque LOAD applies along the motion (Nm)')
    a.set_title('Transmitted torque at constant current')
    a.axhline(0, color=c.INK_3, lw=0.8)
    a.legend(loc='center left', fontsize=8)
    a = ax[1]
    eta = by_speed('eta_symmetric')
    a.plot(sp, eta, '-o', color=c.INK, label='symmetric efficiency (forward = backdrive)')
    a.plot(sp, by_speed('eta_f_worm'), '--^', color='#1c5cab', label='worm-type, forward')
    a.plot(sp, by_speed('eta_b_worm'), '--v', color='#a8302f', label='worm-type, backdrive')
    a.set_xscale('log')
    a.set_ylim(0.4, 1.0)
    a.set_xlabel('Output speed (rad/s, log)')
    a.set_ylabel('Gearbox efficiency (load-dependent part)')
    a.set_title('Efficiency from the two branch slopes')
    kt = by_speed('kt_symmetric')
    a.text(0.03, 0.97, f'implied kt (symmetric) {kt.min():.4f}-{kt.max():.4f} Nm/A\nacross all speeds: kt is constant, '
                       'efficiency is not', transform=a.transAxes, va='top', fontsize=8.5, color=c.INK_2)
    a.legend(loc='lower right')
    a = ax[2]
    for s in (-1, 1):
        a.plot(sp, [groups[(v, s)]['P0'] for v in sp], c.DIR_MARKER[s], color=c.INK_2, ms=8, ls='',
               label=c.DIR_LABEL[s])
    vv = np.linspace(0, sp[-1], 50)
    a.plot(vv, fc + b_visc * vv, color=c.INK, lw=1.4, label=f'{fc:.2f} Nm + {b_visc:.2f} Nm/(rad/s) x speed')
    a.set_xlabel('Output speed (rad/s, linear)')
    a.set_ylabel('No-load drag at the output (Nm)')
    a.set_title('Zero-current drag: Coulomb + viscous')
    a.set_ylim(bottom=0)
    a.legend(loc='upper left')
    fig.suptitle('Skip transmission vs speed: the gearbox gets more efficient as it speeds up, drag grows')
    c.note(fig, 'Efficiency here is the load-dependent part only (from the slopes K_f = eta*26*kt and K_b = 26*kt/eta); '
                'the no-load drag is a separate, additive loss. Both directions averaged.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    c.save(fig, SUB, 'Skip_transmission_vs_speed.png')

    # -- Figure 2: ripple vs speed ---------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    panels = (('swing', 'p1-p99 torque swing within the dwell'), ('std', 'Torque std within the dwell'),
              ('m1', 'Once per motor rev amplitude'), ('m6', '6 per motor rev amplitude'))
    for a, (key, title) in zip(ax.flat, panels):
        for ia in cur:
            ys = [np.nanmean([r[key] for r in R if r['w']['speed_abs'] == v and r['w']['i_along_nom'] == ia]) for v in sp]
            a.plot(sp, ys, '-o', color=c.current_color(ia), ms=5, lw=1.4)
        a.set_xscale('log')
        a.set_title(title)
        a.set_ylabel('Nm (output)')
        a.set_ylim(bottom=0)
    for a in ax[1]:
        a.set_xlabel('Output speed (rad/s, log)')
    fm = c.GEAR_RATIO * np.array(sp) / c.TWO_PI
    ax[1, 1].text(0.02, 0.97, '6/rev frequency: ' + ', '.join(f'{6 * f:.0f}' for f in fm) + ' Hz',
                  transform=ax[1, 1].transAxes, va='top', fontsize=8, color=c.INK_2)
    handles = [Line2D([], [], color=c.current_color(i), lw=2, label=c.current_label(i)) for i in cur]
    fig.legend(handles=handles, loc='lower center', ncol=len(cur), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Skip torque ripple vs speed and current (both directions averaged)')
    c.note(fig, f'Orders fitted in time against the motor encoder with a cubic drift; an order is skipped where it '
                f'would exceed {ORDER_MAX_HZ:.0f} Hz. The swing includes noise and every order.', y=-0.01)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    c.save(fig, SUB, 'Skip_ripple_vs_speed.png')

    # -- Figure 3: feedforward and LOAD loop -------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0))
    qcol = {'forward': '#1c5cab', 'backdrive': '#a8302f'}
    for q in ('forward', 'backdrive'):
        share = [np.mean([r['ff'] / r['lm'] for r in R if r['w']['speed_abs'] == v and r['w']['quadrant'] == q
                          and abs(r['w']['i_along_nom']) >= 5]) for v in sp]
        ax[0].plot(sp, share, '-o', color=qcol[q], label=q)
        loop = [np.mean([abs(r['loop']) for r in R if r['w']['speed_abs'] == v and r['w']['quadrant'] == q
                         and abs(r['w']['i_along_nom']) == c.i_max(d)]) for v in sp]
        ax[1].plot(sp, loop, '-o', color=qcol[q], label=f'{q}, {c.i_max(d):g} A')
    loop0 = [np.mean([abs(r['loop']) for r in R if r['w']['speed_abs'] == v and r['w']['quadrant'] == 'zero']) for v in sp]
    ax[1].plot(sp, loop0, '-o', color=c.INK_3, label='0 A (no FF sent)')
    for ia in cur:
        ys = [np.mean([r['verr_rel'] for r in R if r['w']['speed_abs'] == v and r['w']['i_along_nom'] == ia]) for v in sp]
        ax[2].plot(sp, 100 * np.array(ys), '-o', color=c.current_color(ia), ms=5, lw=1.2)
    ax[0].axhline(1, color=c.INK_3, ls=':', lw=1)
    ax[0].set_ylabel('Feedforward sent / LOAD torque produced')
    ax[0].set_title(f'Share supplied by FF ({c.feedforward_ratio(d):g} x 26 x command)')
    ax[1].set_ylabel('|LOAD torque - feedforward| (Nm)')
    ax[1].set_title('What the velocity loop still had to find')
    ax[2].set_ylabel('LOAD speed error rms / commanded speed (%)')
    ax[2].set_title('LOAD speed tracking')
    for a in ax:
        a.set_xscale('log')
        a.set_xlabel('Output speed (rad/s, log)')
        a.set_ylim(bottom=0)
    ax[0].set_ylim(0, max(1.2, 1.05 * ax[0].get_ylim()[1]))
    ax[0].legend(loc='center right')
    ax[1].legend(loc='center right')
    ax[2].legend(handles=[Line2D([], [], color=c.current_color(i), lw=2, label=c.current_label(i)) for i in cur],
                 loc='upper right', fontsize=8)
    fig.suptitle('Skip LOAD feedforward: about right forward, about a third of what backdrive needs')
    c.note(fig, "LOAD torque is the LOAD current x kt in LOAD's command frame (the frame load_torque_ff is logged in), "
                'dwell means. The loop part includes LOAD-side drag, which no DUT-torque feedforward supplies.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    c.save(fig, SUB, 'Skip_feedforward_vs_speed.png')

    # needed FF per speed: LOAD torque = 26*tc*r_q + c*sign(v)
    ff_by_speed = {f'{v:g}': s02.feedforward_fit([w for w in dw if w['speed_abs'] == v]) for v in sp}

    # -- Figure 4: v1 vs v2 at shared grid points ------------------------------------------------
    cmp_rec = None
    v1_dir = c.RUNS['v1']['results']
    v1_tab = read_csv(os.path.join(v1_dir, '01_overview', 'Skip_dwell_table.csv'))
    v1_rip = read_csv(os.path.join(v1_dir, '04_cogging_folds', 'Skip_ripple_per_dwell.csv'))
    v2_rip = read_csv(os.path.join(c.RESULTS_DIR, '04_cogging_folds', 'Skip_ripple_per_dwell.csv'))
    if c.RUN != 'v1' and v1_tab and v1_rip and v2_rip:
        def key(speed, i):
            # the dwell table logs the measured current, the ripple table the nominal one
            return (round(float(speed), 4), round(2 * float(i)) / 2)
        v1 = {key(r['output_speed_cmd_rad_s'], r['current_along_motion_A']): r for r in v1_tab}
        v1r = {key(r['output_speed_cmd_rad_s'], r['current_along_motion_A']): r for r in v1_rip}
        v2r = {key(r['output_speed_cmd_rad_s'], r['current_along_motion_A']): r for r in v2_rip}
        pairs = []
        for r in R:
            w = r['w']
            k = key(w['speed'], w['i_along_nom'])
            if k in v1 and k in v1r and k in v2r:
                pairs.append((w, r, v1[k], v1r[k], v2r[k]))
        if pairs:
            fig, ax = plt.subplots(1, 4, figsize=(18, 4.8))
            specs = (('Mean torque along motion (Nm)', lambda w, r, a, ar, br: (float(a['load_push_along_motion_Nm']), r['P'])),
                     ('Torque std within dwell (Nm)', lambda w, r, a, ar, br: (float(a['cell_torque_std_Nm']), r['std'])),
                     ('Time stuck (%)', lambda w, r, a, ar, br: (100 * float(a['stuck_fraction']), 100 * r['stuck'])),
                     ('Angle-locked ripple pk-pk (Nm)', lambda w, r, a, ar, br: (float(ar['angle_locked_pkpk_Nm']),
                                                                                  float(br['angle_locked_pkpk_Nm']))))
            cmp_rec = {}
            for a_, (title, fn) in zip(ax, specs):
                xs, ys = [], []
                for w, r, a, ar, br in pairs:
                    x, y = fn(w, r, a, ar, br)
                    xs.append(x)
                    ys.append(y)
                    a_.plot(x, y, c.DIR_MARKER[w['dir']], color=c.current_color(w['i_along_nom']), ms=8,
                            mec='white', mew=0.8)
                lo = min(min(xs), min(ys), 0)
                hi = max(max(xs), max(ys)) * 1.05
                a_.plot([lo, hi], [lo, hi], color=c.INK_3, ls='--', lw=1)
                a_.set_xlabel('v1 (2026-10-19)')
                a_.set_ylabel('v2 (2026-10-21)')
                a_.set_title(title, fontsize=10)
                cmp_rec[title] = {'v1_mean': float(np.mean(xs)), 'v2_mean': float(np.mean(ys))}
            handles = [Line2D([], [], color=c.current_color(i), marker='o', ls='', label=c.current_label(i))
                       for i in sorted({p[0]['i_along_nom'] for p in pairs})]
            fig.legend(handles=handles, loc='lower center', ncol=len(handles), bbox_to_anchor=(0.5, 0.0))
            fig.suptitle(f'Skip v1 vs v2 at the {len(pairs)} dwells both runs share (0.05 and 0.1 rad/s; 0, 5, 10 A)')
            c.note(fig, 'Dashed = no change. v2 has LOAD feedforward reaching the drive, stiffer LOAD velocity gains and a '
                        'warm gearbox; v1 had none of the three, so the change cannot be split between them.', y=-0.02)
            fig.tight_layout(rect=(0, 0.07, 1, 0.94))
            c.save(fig, SUB, 'Skip_run_comparison.png')
            # stuck time in backdrive at 10 A, the condition v1 was worst at among the shared ones
            bd10 = [(float(a['stuck_fraction']), r['stuck']) for w, r, a, ar, br in pairs if w['i_along_nom'] == -10]
            cmp_rec['stuck_backdrive_10A'] = {'v1': float(np.mean([x for x, _ in bd10])), 'v2': float(np.mean([y for _, y in bd10]))}

    c.save_csv(SUB, 'Skip_speed_per_dwell.csv',
               ['dwell', 'output_speed_cmd_rad_s', 'current_along_motion_A', 'quadrant', 'P_Nm', 'std_Nm',
                'swing_p1_p99_Nm', 'stuck_fraction', 'motor_speed_ripple_rel', 'm1_Nm', 'm2_Nm', 'm6_Nm',
                'load_torque_cmd_frame_Nm', 'ff_sent_Nm', 'loop_part_Nm', 'load_speed_err_rms_rad_s'],
               [[r['w']['n'], r['w']['speed'], r['w']['i_along_nom'], r['w']['quadrant'], r['P'], r['std'],
                 r['swing'], r['stuck'], r['speed_ripple'], r['m1'], r['m2'], r['m6'], r['lm'], r['ff'],
                 r['loop'], r['verr_rms']] for r in R])

    def mean_where(key, cond):
        v = [r[key] for r in R if cond(r['w'])]
        return float(np.nanmean(v)) if v else None
    im = c.i_max(d)
    c.record('speed_and_feedforward', {
        'speeds': sp,
        'eta_symmetric_by_speed': dict(zip([f'{v:g}' for v in sp], map(float, eta))),
        'eta_forward_worm_by_speed': dict(zip([f'{v:g}' for v in sp], map(float, by_speed('eta_f_worm')))),
        'eta_backdrive_worm_by_speed': dict(zip([f'{v:g}' for v in sp], map(float, by_speed('eta_b_worm')))),
        'kt_symmetric_by_speed': dict(zip([f'{v:g}' for v in sp], map(float, kt))),
        'K_forward_by_speed': dict(zip([f'{v:g}' for v in sp], map(float, by_speed(('forward', 'K'))))),
        'K_backdrive_by_speed': dict(zip([f'{v:g}' for v in sp], map(float, by_speed(('backdrive', 'K'))))),
        'drag_coulomb_Nm': fc, 'drag_viscous_Nm_per_rad_s': b_visc, 'drag_fit_rms_Nm': drag_rms,
        'swing_backdrive_imax_by_speed': {f'{v:g}': mean_where('swing', lambda w, v=v: w['speed_abs'] == v and w['i_along_nom'] == -im) for v in sp},
        'swing_forward_imax_by_speed': {f'{v:g}': mean_where('swing', lambda w, v=v: w['speed_abs'] == v and w['i_along_nom'] == im) for v in sp},
        'swing_zero_by_speed': {f'{v:g}': mean_where('swing', lambda w, v=v: w['speed_abs'] == v and w['i_along_nom'] == 0) for v in sp},
        'm6_backdrive_imax_by_speed': {f'{v:g}': mean_where('m6', lambda w, v=v: w['speed_abs'] == v and w['i_along_nom'] == -im) for v in sp},
        'm1_backdrive_imax_by_speed': {f'{v:g}': mean_where('m1', lambda w, v=v: w['speed_abs'] == v and w['i_along_nom'] == -im) for v in sp},
        'stuck_max': max(r['stuck'] for r in R),
        'ff_share_forward': float(np.mean([r['ff'] / r['lm'] for r in R if r['w']['quadrant'] == 'forward' and abs(r['w']['i_along_nom']) >= 5])),
        'ff_share_backdrive': float(np.mean([r['ff'] / r['lm'] for r in R if r['w']['quadrant'] == 'backdrive' and abs(r['w']['i_along_nom']) >= 5])),
        'needed_ff_by_speed': ff_by_speed,
        'load_speed_err_rel_by_speed': {f'{v:g}': mean_where('verr_rel', lambda w, v=v: w['speed_abs'] == v) for v in sp},
        'v1_vs_v2': cmp_rec,
    })
    return R


if __name__ == '__main__':
    c.use_run_from_argv()
    run(c.load())
