"""Load cell torque folded over electrical angle, motor angle and output angle,
for every speed and current, and ripple magnitude across the grid.

All folds use the angle-domain series from skip_orders (equal weight per angle,
so stuck time does not pile up at the stick angles), with the dwell's slow
polynomial removed so only angle-periodic content and noise remain.

  electrical fold  slow and motor-locked MECH orders removed first, so what
                   is left is everything periodic in one electrical period
  motor fold       slow removed only: the eccentric/gearbox pattern with the
                   electrical ripple riding on it (21 bumps per rev)
  output fold      lives in s06 (needs a joint fit across dwells)

Outputs (results/04_cogging_folds/):
  Skip_fold_electrical_angle.png
  Skip_fold_motor_angle.png
  Skip_fold_zero_current_by_speed.png   the cleanest condition, all six speeds overlaid
  Skip_fold_conservative_vs_dissipative.png
  Skip_ripple_vs_speed_and_current.png
  Skip_ripple_per_dwell.csv
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import skip_common as c
import skip_orders as so

SUB = '04_cogging_folds'
E_BINS = 48      # 7.5 electrical deg
M_BINS = 180     # 2 motor deg


def elec_fold(d, w):
    f = so.fit_dwell(d, w)
    rs = so.resample_dwell(d, w)
    y = rs['y'] - f['slow'] - f['mech']
    x, m, _ = c.fold(rs['th_e'], y, E_BINS)
    return x, m - np.nanmean(m)


def motor_fold(d, w):
    f = so.fit_dwell(d, w)
    rs = so.resample_dwell(d, w)
    y = rs['y'] - f['slow']
    x, m, _ = c.fold(rs['th_m'], y, M_BINS)
    return x, m - np.nanmean(m)


def grid_figure(d, fold_fn, xlabel, title, fname, ylim, xticks):
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for i, s in enumerate((-1, 1)):
        for j, v in enumerate(c.speeds(d)):
            a = ax[i, j]
            for w in sorted([w for w in d['dwells'] if w['dir'] == s and w['speed_abs'] == v],
                            key=lambda w: abs(w['i_along_nom'])):
                x, m = fold_fn(d, w)
                a.plot(x, m, color=c.current_color(w['i_along_nom']),
                       lw=1.9 if w['i_along_nom'] == 0 else 1.3,
                       zorder=5 if w['i_along_nom'] == 0 else 3)
            a.axhline(0, color=c.INK_3, lw=0.8)
            a.set_title(f'{s * v:+g} rad/s output')
            a.set_xlim(0, 360)
            a.set_xticks(xticks)
            a.set_ylim(*ylim)
            if i == 1:
                a.set_xlabel(xlabel)
        ax[i, 0].set_ylabel(f'{c.DIR_LABEL[s]}\nCell torque ripple (Nm)')
    handles = [Line2D([], [], color=c.current_color(i), lw=2, label=c.current_label(i))
               for i in c.currents_along(d)]
    fig.legend(handles=handles, loc='lower center', ncol=7, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig


def run(d):
    c.apply_style()
    dw = d['dwells']

    fig = grid_figure(d, elec_fold, 'Electrical angle (deg)',
                      'Skip load torque folded over one electrical period (motor-locked mechanical orders removed)',
                      'Skip_fold_electrical_angle.png', (-0.45, 0.45), range(0, 361, 90))
    c.note(fig, 'Electrical angle is 21 x motor encoder angle with an arbitrary zero (consistent across the run). '
                'Red = LOAD backdrives against DUT current, blue = DUT drives, gray = 0 A.', y=0.0)
    c.save(fig, SUB, 'Skip_fold_electrical_angle.png')

    fig = grid_figure(d, motor_fold, 'Motor angle (deg, one motor rev = 1/26 output rev)',
                      'Skip load torque folded over one motor revolution (eccentric / input phase)',
                      'Skip_fold_motor_angle.png', (-3.0, 3.0), range(0, 361, 60))
    c.note(fig, 'Raw angle-domain fold with only the slow drift removed. The fine 21-per-rev ripple is electrical; '
                'the broad shape is the gearbox input phase. Backdrive folds include angle-locked stick-slip.', y=0.0)
    c.save(fig, SUB, 'Skip_fold_motor_angle.png')

    # -- Zero current, all six speeds overlaid: is it the same pattern? --------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for w in [w for w in dw if w['quadrant'] == 'zero']:
        col = c.SPEED_COLORS[w['speed_abs']]
        x, m = elec_fold(d, w)
        ax[0].plot(x, m, color=col, ls=c.DIR_LS[w['dir']], lw=1.6)
        x, m = motor_fold(d, w)
        ax[1].plot(x, m, color=col, ls=c.DIR_LS[w['dir']], lw=1.2)
    ax[0].set_xlabel('Electrical angle (deg)')
    ax[0].set_xticks(range(0, 361, 90))
    ax[0].set_title('Electrical period')
    ax[1].set_xlabel('Motor angle (deg)')
    ax[1].set_xticks(range(0, 361, 60))
    ax[1].set_title('Motor revolution')
    for a in ax:
        a.set_xlim(0, 360)
        a.axhline(0, color=c.INK_3, lw=0.8)
        a.set_ylabel('Cell torque ripple (Nm)')
    handles = [Line2D([], [], color=col, lw=2, label=f'{v:g} rad/s') for v, col in c.SPEED_COLORS.items()]
    handles += [Line2D([], [], color=c.INK_2, ls=c.DIR_LS[s], lw=1.6, label=c.DIR_LABEL[s]) for s in (-1, 1)]
    ax[1].legend(handles=handles, loc='upper right', ncol=2)
    fig.suptitle('Skip zero-current backdrive ripple: same angle-locked pattern at every speed and direction')
    c.note(fig, 'Cell frame (not flipped by direction): a pattern that keeps its sign when the direction reverses '
                'is a position-dependent (conservative) torque such as magnetic cogging.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    c.save(fig, SUB, 'Skip_fold_zero_current_by_speed.png')

    # -- Conservative vs dissipative split at zero current --------------------------
    th_e = np.linspace(0, c.TWO_PI, 361)
    th_m = np.linspace(0, c.TWO_PI, 721)
    fig, ax = plt.subplots(2, 2, figsize=(15, 8), sharex='col')
    split_pp = {}
    for v in c.speeds(d):
        pair = {w['dir']: so.fit_dwell(d, w)['amps'] for w in dw
                if w['quadrant'] == 'zero' and w['speed_abs'] == v}
        for j, (orders, pre, th, scale) in enumerate(((so.ELEC_ORDERS, 'e', th_e / c.POLE_PAIRS, c.POLE_PAIRS),
                                                      (so.MECH_ORDERS, 'm', th_m, 1))):
            Fp = so.recon(pair[1], orders, pre, th)
            Fm = so.recon(pair[-1], orders, pre, th)
            C, D = 0.5 * (Fp + Fm), 0.5 * (Fp - Fm)
            x = th * scale * 180 / np.pi
            col = c.SPEED_COLORS[v]
            ax[0, j].plot(x, C, color=col, lw=1.4, label=f'{v:g} rad/s')
            ax[1, j].plot(x, D, color=col, lw=1.4, label=f'{v:g} rad/s')
            split_pp[f'{"elec" if pre == "e" else "mech"}_{v:g}_conservative_pkpk'] = so.pkpk(C)
            split_pp[f'{"elec" if pre == "e" else "mech"}_{v:g}_dissipative_pkpk'] = so.pkpk(D)
    ax[0, 0].set_title('Electrical orders - conservative (same sign both directions)')
    ax[1, 0].set_title('Electrical orders - dissipative (flips with direction)')
    ax[0, 1].set_title('Motor-locked mechanical orders - conservative')
    ax[1, 1].set_title('Motor-locked mechanical orders - dissipative')
    ax[1, 0].set_xlabel('Electrical angle (deg)')
    ax[1, 1].set_xlabel('Motor angle (deg)')
    for a in ax.flat:
        a.set_xlim(0, 360)
        a.axhline(0, color=c.INK_3, lw=0.8)
        a.set_ylabel('Torque (Nm, cell frame)')
    for a in ax[:, 0]:
        a.set_ylim(-0.12, 0.12)
        a.set_xticks(range(0, 361, 90))
    for a in ax[:, 1]:
        a.set_ylim(-0.4, 0.4)
        a.set_xticks(range(0, 361, 60))
    ax[0, 0].legend(loc='upper left', ncol=3)
    fig.suptitle('Skip zero-current ripple split into position-dependent and friction-like parts')
    c.note(fig, 'From the fitted orders of the + and - direction zero-current dwells at each speed: conservative = '
                '(F+ + F-)/2, dissipative = (F+ - F-)/2. Magnetic cogging is conservative; friction that varies '
                'with angle, or energy lost snapping through detents, is dissipative.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    c.save(fig, SUB, 'Skip_fold_conservative_vs_dissipative.png')

    # -- Ripple magnitude across the grid --------------------------------------------
    rows, R = [], []
    th1 = np.linspace(0, c.TWO_PI, 2049)
    for w in dw:
        f = so.fit_dwell(d, w)
        a = f['amps']
        e = so.recon(a, so.ELEC_ORDERS, 'e', th1)
        m = so.recon(a, so.MECH_ORDERS, 'm', th1)
        r = {'n': w['n'], 'speed': w['speed'], 'v': w['speed_abs'], 'dir': w['dir'],
             'I': w['i_along_nom'], 'elec_pp': so.pkpk(e), 'mech_pp': so.pkpk(m),
             'total_pp': so.pkpk(e + m), 'resid_rms': float(np.std(f['resid'])),
             'slow_pp': so.pkpk(f['slow'])}
        R.append(r)
        rows.append([w['n'], w['speed'], w['i_along_nom'], w['quadrant'], r['total_pp'], r['elec_pp'],
                     r['mech_pp'], r['resid_rms'], r['slow_pp']])
    c.save_csv(SUB, 'Skip_ripple_per_dwell.csv',
               ['dwell', 'output_speed_cmd_rad_s', 'current_along_motion_A', 'quadrant',
                'angle_locked_pkpk_Nm', 'electrical_pkpk_Nm', 'motor_locked_mech_pkpk_Nm',
                'non_repeatable_rms_Nm', 'slow_wander_pkpk_Nm'], rows)

    fig, ax = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    panels = (('total_pp', 'All angle-locked ripple over a motor rev (pk-pk)'),
              ('elec_pp', 'Electrical part (21k per motor rev, pk-pk)'),
              ('mech_pp', 'Motor-locked mechanical part (pk-pk)'),
              ('resid_rms', 'Not locked to motor angle (rms): stick-slip, noise, output-phase'))
    for a, (key, title) in zip(ax.flat, panels):
        for v in c.speeds(d):
            for s in (-1, 1):
                pts = sorted([r for r in R if r['v'] == v and r['dir'] == s], key=lambda r: r['I'])
                a.plot([r['I'] for r in pts], [r[key] for r in pts], color=c.SPEED_COLORS[v],
                       marker=c.DIR_MARKER[s], ls=c.DIR_LS[s], lw=1.2, ms=6)
        a.set_title(title)
        a.set_ylabel('Nm (output)')
        a.set_ylim(bottom=0)
        a.axvline(0, color=c.INK_3, lw=0.8)
    for a in ax[1]:
        a.set_xlabel('DUT current along motion (A)  [- = backdrive, + = forward]')
        a.set_xticks(c.currents_along(d))
    handles = [Line2D([], [], color=col, lw=2, label=f'{v:g} rad/s') for v, col in c.SPEED_COLORS.items()]
    handles += [Line2D([], [], color=c.INK_2, marker=c.DIR_MARKER[s], ls=c.DIR_LS[s], label=c.DIR_LABEL[s])
                for s in (-1, 1)]
    ax[0, 0].legend(handles=handles, loc='upper center', ncol=2)
    fig.suptitle('Skip torque ripple magnitude vs speed and current')
    c.note(fig, 'Pk-pk from the fitted order series (noise-free); rms of what the angle fits leave behind. '
                'Ripple grows with transmitted load and is largest in backdrive; speed matters little at 0.025-0.1 rad/s.',
           y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    c.save(fig, SUB, 'Skip_ripple_vs_speed_and_current.png')

    def mean_of(key, cond):
        return float(np.mean([r[key] for r in R if cond(r)]))
    c.record('ripple', {
        'zero_current': {k: mean_of(k, lambda r: r['I'] == 0) for k in ('total_pp', 'elec_pp', 'mech_pp', 'resid_rms')},
        'forward_15A': {k: mean_of(k, lambda r: r['I'] == 15) for k in ('total_pp', 'elec_pp', 'mech_pp', 'resid_rms')},
        'backdrive_15A': {k: mean_of(k, lambda r: r['I'] == -15) for k in ('total_pp', 'elec_pp', 'mech_pp', 'resid_rms')},
        'zero_current_conservative_vs_dissipative_pkpk': split_pp,
    })
    return R


if __name__ == '__main__':
    run(c.load())
