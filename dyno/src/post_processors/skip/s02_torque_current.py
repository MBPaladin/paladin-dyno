"""Output torque vs DUT current in all four quadrants, and what it implies for
kt, gearbox efficiency, torque control and the LOAD feedforward.

Every dwell is a steady slide at constant current, so the cell reads the torque
the gear train transmits while sliding -- the kinetic-friction value, free of
the arrest-position ambiguity that makes static holds scatter. Plotted as the
torque LOAD must push along the motion against the DUT current along the
motion, all 42 dwells collapse onto one kinked line:

  * DUT pushing with the motion (forward power flow): slope K_f
  * DUT resisting the motion (LOAD backdrives it):    slope K_b, about 3x K_f
  * zero current: the no-load backdrive drag

K_f = eta_f * 26 * kt and K_b = 26 * kt / eta_b. One slope pair cannot separate
kt from efficiency, so kt is reported as a bracket and under two standard
friction models (symmetric efficiency, and the worm-type eta_b = 2 - 1/eta_f).

Outputs (results/02_torque_vs_current/):
  Skip_load_torque_vs_current.png     the kinked transmission curve + standstill band
  Skip_efficiency_and_kt_by_speed.png branch slopes, efficiency, drag per speed
  Skip_feedforward_map.png            LOAD torque needed vs DUT command; FF model
  Skip_torque_current_fits.csv        per-speed-group fits
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import skip_common as c

SUB = '02_torque_vs_current'


def branch_fit(dw):
    """Linear fit per branch: P = c - K * I_along. Returns dict."""
    out = {}
    for name, sel in (('forward', lambda w: w['i_along_nom'] > 0),
                      ('backdrive', lambda w: w['i_along_nom'] < 0)):
        pts = [w for w in dw if sel(w)]
        x = np.array([w['i_along'] for w in pts])
        y = np.array([w['P'] for w in pts])
        A = np.column_stack([np.ones_like(x), -x])
        (c0, k), *_ = np.linalg.lstsq(A, y, rcond=None)
        out[name] = {'K': float(k), 'c': float(c0),
                     'rms_resid': float(np.sqrt(np.mean((y - A @ [c0, k]) ** 2)))}
    z = [w['P'] for w in dw if w['i_along_nom'] == 0]
    out['P0'] = float(np.mean(z))
    kf, kb = out['forward']['K'], out['backdrive']['K']
    out['eta_symmetric'] = float(np.sqrt(kf / kb))
    out['kt_symmetric'] = float(np.sqrt(kf * kb) / c.GEAR_RATIO)
    eta_f = 0.5 * (1.0 + kf / kb)                 # worm-type: eta_f*eta_b = 2 eta_f - 1
    out['eta_f_worm'] = float(eta_f)
    out['eta_b_worm'] = float(2.0 - 1.0 / eta_f)
    out['kt_worm'] = float(kf / (c.GEAR_RATIO * eta_f))
    out['kt_lower'] = float(kf / c.GEAR_RATIO)    # forward efficiency 100%
    out['kt_upper'] = float(kb / c.GEAR_RATIO)    # backdrive efficiency 100%
    return out


def load_torque_test_17_holds():
    path = os.path.join(c.SKIP_LOGS, 'torque_tests', 'torque_test_17', 'torque_vs_current',
                        'per_hold.csv')
    if not os.path.exists(path):
        return None
    i, T = [], []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            ih = abs(float(r['i_hold']))
            i.append(ih)
            T.append(float(r['kt_eff_plateau']) * c.GEAR_RATIO * ih)
    return np.array(i), np.array(T)


def feedforward_fit(dw):
    """LOAD motor torque (LOAD command frame) = 26*tcmd*r_q + c_v*sign(speed) + c0."""
    y = np.array([w['T_lm_cmd'] for w in dw])
    g = np.array([c.GEAR_RATIO * w['tcmd'] for w in dw])
    fwd = np.array([w['quadrant'] == 'forward' for w in dw], float)
    bd = np.array([w['quadrant'] == 'backdrive' for w in dw], float)
    sv = np.array([w['dir'] for w in dw], float)
    A = np.column_stack([g * fwd, g * bd, sv, np.ones_like(sv)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return {'r_forward': float(coef[0]), 'r_backdrive': float(coef[1]),
            'c_direction_Nm': float(coef[2]), 'c0_Nm': float(coef[3]),
            'rms_resid_Nm': float(np.sqrt(np.mean(resid ** 2))),
            'max_abs_resid_Nm': float(np.max(np.abs(resid)))}


def run(d):
    c.apply_style()
    dw = d['dwells']
    # The headline line is the quasi-static one (all of v1; <= 0.25 rad/s in v2);
    # the per-speed groups below show how it changes with speed.
    fit = branch_fit([w for w in dw if w['speed_abs'] <= c.COGGING_MAX_SPEED])
    groups = {}
    for v in c.speeds(d):
        for s in (-1, 1):
            groups[(v, s)] = branch_fit([w for w in dw if w['speed_abs'] == v and w['dir'] == s])
    ff = feedforward_fit([w for w in dw if w['speed_abs'] <= c.COGGING_MAX_SPEED])

    # -- Figure 1: the kinked curve + standstill band -----------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.2))
    a = ax[0]
    im = c.i_max(d) + 1.0
    for w in dw:
        a.plot(w['i_along'], w['P'], c.DIR_MARKER[w['dir']], color=c.SPEED_COLORS[w['speed_abs']],
               ms=8, mec='white', mew=0.8, zorder=3)
    xb = np.linspace(-im, 0, 20)
    xf = np.linspace(0, im, 20)
    a.plot(xb, fit['backdrive']['c'] - fit['backdrive']['K'] * xb, color=c.INK, lw=1.4, zorder=2)
    a.plot(xf, fit['forward']['c'] - fit['forward']['K'] * xf, color=c.INK, lw=1.4, zorder=2)
    xx = np.linspace(-im, im, 2)
    a.plot(xx, -c.GEAR_RATIO * fit['kt_symmetric'] * xx, color=c.INK_3, ls='--', lw=1.2, zorder=1)
    a.plot(xx, -c.GEAR_RATIO * c.KT_CONFIGURED * xx, color=c.INK_3, ls=':', lw=1.2, zorder=1)
    a.text(-im + 0.5, -c.GEAR_RATIO * fit['kt_symmetric'] * (-im + 0.5) - 2.6,
           f'lossless, kt = {fit["kt_symmetric"]:.3f}', color=c.INK_2, fontsize=8.5)
    a.text(-im + 0.5, -c.GEAR_RATIO * c.KT_CONFIGURED * (-im + 0.5) - 2.6,
           f'lossless, kt = {c.KT_CONFIGURED} (configured)', color=c.INK_2, fontsize=8.5)
    a.axhline(0, color=c.INK_3, lw=0.8)
    a.axvline(0, color=c.INK_3, lw=0.8)
    a.annotate(f'backdrive slope {fit["backdrive"]["K"]:.2f} Nm/A', xy=(-10, fit['backdrive']['c'] + 10 * fit['backdrive']['K']),
               xytext=(-8.5, 19.0), fontsize=9, color=c.INK, arrowprops=dict(arrowstyle='-', color=c.INK_3))
    a.annotate(f'forward slope {fit["forward"]["K"]:.2f} Nm/A', xy=(10, fit['forward']['c'] - 10 * fit['forward']['K']),
               xytext=(3.0, -13.5), fontsize=9, color=c.INK, arrowprops=dict(arrowstyle='-', color=c.INK_3))
    a.annotate(f'no-load drag {fit["P0"]:.2f} Nm', xy=(0, fit['P0']), xytext=(1.2, 6.5), fontsize=9,
               color=c.INK, arrowprops=dict(arrowstyle='-', color=c.INK_3))
    a.text(-im + 0.5, -7.0, 'DUT resists the motion\n(LOAD backdrives the gearbox)', fontsize=9, color='#a8302f', va='top')
    a.text(im - 0.5, 21.5, 'DUT pushes with the motion\n(DUT drives LOAD)', fontsize=9, color='#1c5cab',
           va='top', ha='right')
    a.set_xlabel('DUT current along the motion (A)')
    a.set_ylabel('Torque LOAD applies along the motion (Nm, output)')
    a.set_title('Transmitted torque while sliding: one kinked line')
    a.set_xlim(-im - 0.5, im + 0.5)
    a.set_ylim(-17, 24)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=col, marker='o', ls='', ms=8, label=f'{v:g} rad/s')
               for v, col in c.SPEED_COLORS.items() if v in c.speeds(d)]
    handles += [Line2D([], [], color=c.INK_2, marker=c.DIR_MARKER[s], ls='', ms=8, label=c.DIR_LABEL[s])
                for s in (-1, 1)]
    a.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.0, 0.86))

    b = ax[1]
    ii = np.linspace(0, im, 50)
    lo = np.maximum(fit['forward']['K'] * ii - fit['forward']['c'], 0)
    hi = fit['backdrive']['K'] * ii + fit['backdrive']['c']
    b.fill_between(ii, lo, hi, color='#e8e7e2', lw=0, label='possible output torque at standstill')
    b.plot(ii, hi, color='#a8302f', lw=1.6, label='upper edge: LOAD about to backdrive')
    b.plot(ii, lo, color='#1c5cab', lw=1.6, label='lower edge: DUT about to drive')
    b.plot(ii, c.GEAR_RATIO * fit['kt_symmetric'] * ii, color=c.INK_3, ls='--', lw=1.2,
           label=f'lossless 26 x {fit["kt_symmetric"]:.3f} x I')
    tt17 = load_torque_test_17_holds()
    if tt17 is not None:
        b.plot(tt17[0], tt17[1], 'o', color=c.INK, ms=5, alpha=0.7,
               label='torque_test_17 settled holds (static)')
    b.set_xlabel('|DUT current| (A)')
    b.set_ylabel('|Output torque| (Nm)')
    b.set_title('Why static torque-vs-current scatters')
    b.set_xlim(0, im)
    b.set_ylim(0, 25)
    b.legend(loc='upper left')
    span12 = (fit['backdrive']['K'] * 12.5 + fit['backdrive']['c'],
              fit['forward']['K'] * 12.5 - fit['forward']['c'])
    sp = c.speeds(d)
    fig.suptitle(f'Skip output torque vs motor current ({c.RUN}: all {len(dw)} dwells, {sp[0]:g}-{sp[-1]:g} rad/s)')
    c.note(fig, 'Left: every dwell mean; torque is the load cell (split-corrected, re-zeroed) signed so that + means '
                'LOAD pushes along the motion. Right: with no motion, friction can hold the output anywhere '
                f'between the two sliding lines - at 12.5 A that is {span12[1]:.1f} to {span12[0]:.1f} Nm, '
                'which brackets the static holds of torque_test_17.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    c.save(fig, SUB, 'Skip_load_torque_vs_current.png')

    # -- Figure 2: per speed-group slopes, efficiency, drag ---------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    order = sorted(groups, key=lambda vs: vs[1] * vs[0])
    xpos = {vs: k for k, vs in enumerate(order)}
    for (v, s), g in sorted(groups.items(), key=lambda kv: kv[0][1] * kv[0][0]):
        x = xpos[(v, s)]
        ax[0].plot(x, g['forward']['K'], 'o', color='#1c5cab', ms=8)
        ax[0].plot(x, g['backdrive']['K'], 's', color='#a8302f', ms=8)
        ax[1].plot(x, g['eta_symmetric'], 'o', color=c.INK, ms=8)
        ax[1].plot(x, g['eta_f_worm'], '^', color='#1c5cab', ms=7)
        ax[1].plot(x, g['eta_b_worm'], 'v', color='#a8302f', ms=7)
        ax[2].plot(x, g['P0'], 'o', color=c.INK, ms=8)
    ax[0].set_title('Branch slopes')
    ax[0].set_ylabel('Output Nm per DUT A')
    ax[0].legend(handles=[Line2D([], [], color='#1c5cab', marker='o', ls='', label='forward (DUT drives)'),
                          Line2D([], [], color='#a8302f', marker='s', ls='', label='backdrive')], loc='center right')
    ax[0].set_ylim(0, 1.7)
    ax[1].set_title('Implied gearbox efficiency')
    ax[1].set_ylim(0.3, 1.0)
    ax[1].legend(handles=[Line2D([], [], color=c.INK, marker='o', ls='', label='symmetric model'),
                          Line2D([], [], color='#1c5cab', marker='^', ls='', label='worm-type, forward'),
                          Line2D([], [], color='#a8302f', marker='v', ls='', label='worm-type, backdrive')],
                 loc='lower right')
    ax[2].set_title('No-load backdrive drag (0 A)')
    ax[2].set_ylabel('Nm at output')
    ax[2].set_ylim(0, max(1.2, 1.15 * max(g['P0'] for g in groups.values())))
    for a in ax:
        a.set_xlabel('Output speed (rad/s)')
        a.set_xticks(range(len(order)))
        a.set_xticklabels([f'{s * v:+g}' for v, s in order])
        a.tick_params(axis='x', labelrotation=45)
    fig.suptitle('Skip gearbox transmission by speed and direction')
    c.note(fig, f'Pooled: forward {fit["forward"]["K"]:.3f}, backdrive {fit["backdrive"]["K"]:.3f} Nm/A; '
                f'kt bracket {fit["kt_lower"]:.4f}..{fit["kt_upper"]:.4f} Nm/A; '
                f'kt = {fit["kt_symmetric"]:.4f} (symmetric) or {fit["kt_worm"]:.4f} (worm-type).', y=0.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    c.save(fig, SUB, 'Skip_efficiency_and_kt_by_speed.png')

    # -- Figure 3: feedforward map ---------------------------------------------------
    # LOAD torque in its COMMAND frame (the frame a feedforward is sent in):
    # the logged current is in the drive's frame, commands go through
    # flip_torque_sign. Quasi-static speeds only; s10 has the speed dependence.
    qs = [w for w in dw if w['speed_abs'] <= c.COGGING_MAX_SPEED]
    fig, a = plt.subplots(figsize=(9.5, 6.5))
    dir_col = {-1: '#2a78d6', 1: '#eb6834'}
    # With more than three speeds the spread at each command is speed: colour points by speed.
    by_speed = len({w['speed_abs'] for w in qs}) > 3
    for w in qs:
        a.plot(w['tcmd'], w['T_lm_cmd'], c.DIR_MARKER[w['dir']],
               color=c.SPEED_COLORS[w['speed_abs']] if by_speed else dir_col[w['dir']], ms=8,
               mec='white', mew=0.8, zorder=3)
    tmax = max(abs(w['tcmd']) for w in dw)
    tq = np.linspace(-tmax, tmax, 61)
    for s in (-1, 1):
        # which sign of the command is forward in this direction (depends on the DUT's flip)
        fsign = np.sign(np.mean([np.sign(w['tcmd']) for w in qs if w['dir'] == s and w['quadrant'] == 'forward']))
        fwd = np.sign(tq) == fsign
        y = c.GEAR_RATIO * tq * np.where(fwd, ff['r_forward'], ff['r_backdrive']) \
            + ff['c_direction_Nm'] * s + ff['c0_Nm']
        y[tq == 0] = np.nan
        a.plot(tq, y, color=dir_col[s], lw=1.6, label=f'fit to what LOAD needed, {c.DIR_LABEL[s]}')
    ratio = c.feedforward_ratio(d)
    a.plot(tq, ratio * c.GEAR_RATIO * tq, color=c.INK_3, ls='--', lw=1.4,
           label=f'FF {"sent" if c.RUN != "v1" else "configured"}: {ratio:g} x 26 x command')
    a.axhline(0, color=c.INK_3, lw=0.8)
    a.axvline(0, color=c.INK_3, lw=0.8)
    a.set_xlabel(f'DUT torque command (Nm, motor side, as logged; kt = {c.KT_CONFIGURED} -> {1 / c.KT_CONFIGURED:g} A/Nm)')
    a.set_ylabel("LOAD motor torque, kt x current (Nm, LOAD's command frame)")
    a.set_title('Skip: torque the LOAD needed vs what feedforward supplies')
    if by_speed:
        from matplotlib.lines import Line2D as _L
        a.legend(handles=a.get_legend_handles_labels()[0]
                 + [_L([], [], color=c.SPEED_COLORS[v], marker='o', ls='', label=f'{v:g} rad/s')
                    for v in sorted({w['speed_abs'] for w in qs})], loc='upper left', fontsize=8)
    else:
        a.legend(loc='upper left')
    a.text(0.98, 0.02, f'r_forward = {ff["r_forward"]:.2f}, r_backdrive = {ff["r_backdrive"]:.2f}\n'
                       f'direction term {ff["c_direction_Nm"]:+.2f} Nm x sign(speed cmd), offset {ff["c0_Nm"]:+.2f} Nm\n'
                       f'fit residual {ff["rms_resid_Nm"]:.2f} Nm rms over {len(qs)} dwells',
           transform=a.transAxes, ha='right', fontsize=9, color=c.INK_2, va='bottom')
    c.note(fig, f'Points are dwell means of the LOAD current x {c.KT_LOAD} Nm/A at <= {c.COGGING_MAX_SPEED:g} rad/s, '
                'i.e. the feedforward plus whatever the velocity loop added. A perfect feedforward lies on the points.',
           y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    c.save(fig, SUB, 'Skip_feedforward_map.png')

    rows = []
    for (v, s), g in sorted(groups.items()):
        rows.append([v * s, g['forward']['K'], g['backdrive']['K'], g['forward']['c'], g['backdrive']['c'],
                     g['P0'], g['eta_symmetric'], g['kt_symmetric'], g['eta_f_worm'], g['eta_b_worm'],
                     g['kt_worm']])
    rows.append(['pooled', fit['forward']['K'], fit['backdrive']['K'], fit['forward']['c'],
                 fit['backdrive']['c'], fit['P0'], fit['eta_symmetric'], fit['kt_symmetric'],
                 fit['eta_f_worm'], fit['eta_b_worm'], fit['kt_worm']])
    c.save_csv(SUB, 'Skip_torque_current_fits.csv',
               ['output_speed_rad_s', 'K_forward_Nm_per_A', 'K_backdrive_Nm_per_A', 'c_forward_Nm',
                'c_backdrive_Nm', 'P0_no_load_drag_Nm', 'eta_symmetric', 'kt_symmetric_Nm_per_A',
                'eta_forward_worm', 'eta_backdrive_worm', 'kt_worm_Nm_per_A'], rows)

    c.record('torque_vs_current', {
        'pooled': fit,
        'per_group_K_forward_range': [min(g['forward']['K'] for g in groups.values()),
                                      max(g['forward']['K'] for g in groups.values())],
        'per_group_K_backdrive_range': [min(g['backdrive']['K'] for g in groups.values()),
                                        max(g['backdrive']['K'] for g in groups.values())],
        'per_group_P0_range': [min(g['P0'] for g in groups.values()), max(g['P0'] for g in groups.values())],
        'standstill_band_at_12p5A_Nm': [float(span12[1]), float(span12[0])],
        'feedforward_fit': ff,
        'existing_ff_ratio': c.feedforward_ratio(d),
    })
    return fit, ff


if __name__ == '__main__':
    c.use_run_from_argv()
    run(c.load())
