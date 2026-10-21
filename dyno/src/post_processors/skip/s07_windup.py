"""Gear-train windup and transmission error while moving.

    deflection = motor angle / 26 - output angle      [output rad]

measured by the two encoders. With the DUT in torque mode the motor side is a
force source, so the deflection holds three things: the spring windup under
the transmitted torque, which flank of the backlash is loaded, and the
kinematic transmission error (TE) of the gear geometry. Dwell means give the
first two; folding the deflection over motor and output angle gives the third.

The load encoder is stored as float32 near 2.4e4 rad (2 mrad steps); dwell
means and folds average that away.

Outputs (results/07_windup_and_TE/):
  Skip_windup_vs_torque.png
  Skip_transmission_error.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import skip_common as c

SUB = '07_windup_and_TE'


def run(d):
    c.apply_style()
    dw = d['dwells']
    defl = d['th_m'] / c.GEAR_RATIO - d['th_o']
    z = {s: np.mean([np.mean(defl[c.dwell_slice(w)]) for w in dw if w['quadrant'] == 'zero' and w['dir'] == s])
         for s in (-1, 1)}
    defl = defl - 0.5 * (z[-1] + z[1])
    for w in dw:
        w['defl_mrad'] = 1e3 * float(np.mean(defl[c.dwell_slice(w)]))

    T = np.array([w['T_cell'] for w in dw])
    D = np.array([w['defl_mrad'] for w in dw])
    A = np.column_stack([np.ones_like(T), np.sign(T), T])
    coef, *_ = np.linalg.lstsq(A, D, rcond=None)
    gap = 2 * coef[1]
    k = 1.0 / coef[2]                      # Nm per mrad
    fits = {}
    for s in (1, -1):
        m = np.sign(T) == s
        p = np.polyfit(T[m], D[m], 1)
        fits[s] = {'k_Nm_per_mrad': float(1.0 / p[0]), 'zero_torque_defl_mrad': float(p[1])}

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.8))
    a = ax[0]
    for w in dw:
        a.plot(w['T_cell'], w['defl_mrad'], c.DIR_MARKER[w['dir']], color=c.SPEED_COLORS[w['speed_abs']],
               ms=8, mec='white', mew=0.8)
    for s in (1, -1):
        tt = np.linspace(0, 23, 10) * s
        a.plot(tt, fits[s]['zero_torque_defl_mrad'] + tt / fits[s]['k_Nm_per_mrad'], color=c.INK, lw=1.2)
    a.axhline(0, color=c.INK_3, lw=0.8)
    a.axvline(0, color=c.INK_3, lw=0.8)
    a.set_xlabel('Cell torque (Nm, dwell mean)')
    a.set_ylabel('Deflection, motor/26 - output (mrad at output)')
    a.set_title('Windup while sliding (dwell means)')
    a.text(0.02, 0.97, f'+ flank: {fits[1]["k_Nm_per_mrad"]:.2f} Nm/mrad\n'
                       f'- flank: {fits[-1]["k_Nm_per_mrad"]:.2f} Nm/mrad\n'
                       f'gap between flanks at 0 Nm: {fits[1]["zero_torque_defl_mrad"] - fits[-1]["zero_torque_defl_mrad"]:.1f} mrad\n'
                       f'(torque_test_17 static: 3.66 / 2.94 Nm/mrad, 9.7 mrad)',
           transform=a.transAxes, va='top', fontsize=9, color=c.INK_2)
    handles = [Line2D([], [], color=col, marker='o', ls='', ms=8, label=f'{v:g} rad/s') for v, col in c.SPEED_COLORS.items()]
    handles += [Line2D([], [], color=c.INK_2, marker=c.DIR_MARKER[s], ls='', ms=8, label=c.DIR_LABEL[s]) for s in (-1, 1)]
    a.legend(handles=handles, loc='lower right')

    # Within-dwell clouds for the extreme dwells: the deflection excursion during stick-slip.
    b = ax[1]
    for w in dw:
        if abs(w['i_along_nom']) != 15 or w['speed_abs'] != 0.05:
            continue
        sl = c.dwell_slice(w)
        col = '#a8302f' if w['quadrant'] == 'backdrive' else '#1c5cab'
        b.plot(d['T_cell'][sl][::20], 1e3 * defl[sl][::20], '.', color=col, ms=1.5, alpha=0.35)
    b.set_xlabel('Cell torque (Nm, every 20th sample)')
    b.set_ylabel('Deflection (mrad at output)')
    b.set_title('Inside the 15 A dwells at 0.05 rad/s')
    b.legend(handles=[Line2D([], [], color='#a8302f', marker='o', ls='', label='backdrive 15 A'),
                      Line2D([], [], color='#1c5cab', marker='o', ls='', label='forward 15 A')], loc='lower right')
    fig.suptitle('Skip gear-train windup and backlash measured while moving')
    c.note(fig, 'Deflection zero = mean of the two zero-current dwell groups. Load encoder quantisation is 2 mrad '
                'per sample, so the clouds are striped; the dwell means are not.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    c.save(fig, SUB, 'Skip_windup_vs_torque.png')

    # -- Transmission error folds ------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    te_pp = {}
    for s in (-1, 1):
        clean = [w for w in dw if w['dir'] == s and w['quadrant'] in ('zero', 'forward')]
        xs_m, ys_m, xs_o, ys_o = [], [], [], []
        for w in clean:
            sl = c.dwell_slice(w)
            dd = defl[sl]
            t = np.arange(len(dd))
            dd = dd - np.polyval(np.polyfit(t, dd, 2), t)
            xs_m.append(d['th_m'][sl])
            ys_m.append(dd)
            xs_o.append(d['th_o'][sl])
            ys_o.append(defl[sl] - np.mean(defl[sl]))
        col = '#2a78d6' if s < 0 else '#eb6834'
        thm = np.concatenate(xs_m)
        x, m, _ = c.fold(thm, np.concatenate(ys_m), 180, min_count=200)
        m = m - np.nanmean(m)
        ax[0].plot(x, 1e3 * m, color=col, lw=1.6, label=c.DIR_LABEL[s])
        te_pp[f'motor_dir{s:+d}_pkpk_mrad'] = float(1e3 * (np.nanmax(m) - np.nanmin(m)))
        # The motor-rev TE is exactly order 26 per output rev, so it must come
        # out before folding over the output angle or it aliases straight in.
        motor_te = np.interp(np.mod(thm, c.TWO_PI) * 180 / np.pi, x, m, period=360.0)
        x, m, _ = c.fold(np.concatenate(xs_o), np.concatenate(ys_o) - motor_te, 72, min_count=200)
        m = m - np.nanmean(m)
        ax[1].plot(x, 1e3 * m, color=col, lw=1.6, label=c.DIR_LABEL[s])
        te_pp[f'output_dir{s:+d}_pkpk_mrad'] = float(1e3 * (np.nanmax(m) - np.nanmin(m)))
    ax[0].set_xlabel('Motor angle (deg)')
    ax[0].set_title('Over one motor rev (eccentric phase)')
    ax[1].set_xlabel('Output angle (deg)')
    ax[1].set_title('Over one output rev (dwell means and motor-rev pattern removed)')
    for a in ax:
        a.set_xlim(0, 360)
        a.set_xticks(range(0, 361, 60))
        a.axhline(0, color=c.INK_3, lw=0.8)
        a.set_ylabel('Deflection about mean (mrad at output)')
        a.legend(loc='upper right')
    fig.suptitle('Skip transmission error: zero-current and forward dwells, pooled per direction')
    c.note(fig, 'Folding the encoder-to-encoder deflection. Left: a clean once-per-motor-rev error, the same in both '
                'directions (kinematic: eccentric/pin-circle error, or a once-per-rev motor encoder error). Right: what '
                'is left over the output rev is mostly a ~25-per-output-rev zigzag, the same feature as the torque '
                f'sideband at order 25. 1 mrad at the output is ~{1e-3 * 180 / np.pi * 60:.1f} arcmin.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    c.save(fig, SUB, 'Skip_transmission_error.png')

    c.record('windup', {
        'flank_fits': {('+' if s > 0 else '-'): v for s, v in fits.items()},
        'gap_between_flanks_mrad': float(fits[1]['zero_torque_defl_mrad'] - fits[-1]['zero_torque_defl_mrad']),
        'joint_fit_k_Nm_per_mrad': float(k), 'joint_fit_gap_mrad': float(gap),
        'te_pkpk_mrad': te_pp,
    })


if __name__ == '__main__':
    run(c.load())
