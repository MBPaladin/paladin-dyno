"""Time dependence while moving: how fast the transmitted torque follows a
step in DUT current, and how much it wanders within a dwell.

The static holds (torque_test_17) relaxed with a ~3 s time constant. Here the
gearbox is sliding when each 5 A current step arrives, so if that slow
relaxation belonged to the gearbox (a viscoelastic or thermal effect) it
would show up again; if it belonged to the stuck, arrested gear train and the
LOAD position loop, it would not.

Response metric per step: torque LOAD pushes along the motion, 50 ms centred
mean, normalised from the previous dwell's mean (0) to the new dwell's mean (1).
Steps split into fast ones (gear train keeps sliding; the cell follows the
current within ~0.2 s) and slow ramps whose length scales with 1/speed: the
gear train sticks and LOAD's velocity loop, with no feedforward, has to
integrate the new torque out of its speed error.

Outputs (results/09_time_response/):
  Skip_torque_step_response.png
  Skip_within_dwell_wander.png
  Skip_step_response.csv
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import skip_common as c

SUB = '09_time_response'
PRE_S, POST_S = 1.0, 8.0
SLOW_T90_S = 0.4


def smooth(x, n=50):
    return np.convolve(x, np.ones(n) / n, mode='same')


def crossing_time(t, y, level):
    idx = np.flatnonzero(y >= level)
    return float(t[idx[0]]) if len(idx) else np.nan


def run(d):
    c.apply_style()
    dw = d['dwells']
    steps = []
    for prev, w in zip(dw[:-1], dw[1:]):
        if prev['speed'] != w['speed']:
            continue
        i_step = w['i0'] - int(c.DWELL_SETTLE_S * c.FS_HZ)       # command plateau start
        sl = slice(i_step - int(PRE_S * c.FS_HZ), i_step + int(POST_S * c.FS_HZ))
        P = smooth(-w['dir'] * d['T_cell'][sl])
        t = (np.arange(sl.stop - sl.start) - int(PRE_S * c.FS_HZ)) / c.FS_HZ
        y = (P - prev['P']) / (w['P'] - prev['P'])
        kind = ('crosses 0 A' if prev['i_along_nom'] * w['i_along_nom'] <= 0
                else 'backdrive' if w['i_along_nom'] < 0 else 'forward')
        steps.append({'speed': w['speed'], 'v': w['speed_abs'], 'dir': w['dir'], 'from': prev['i_along_nom'],
                      'to': w['i_along_nom'], 'kind': kind, 't': t, 'y': y,
                      't63': crossing_time(t[t > 0], y[t > 0], 0.63), 't90': crossing_time(t[t > 0], y[t > 0], 0.9),
                      'dP': w['P'] - prev['P']})

    kind_col = {'backdrive': '#a8302f', 'crosses 0 A': c.INK_3, 'forward': '#1c5cab'}
    sp = c.speeds(d)
    nr = 1 if len(sp) <= 3 else 2
    nc = int(np.ceil(len(sp) / nr))
    fig, ax = plt.subplots(nr, nc, figsize=(5.3 * nc, 5.2 * nr), sharey=True, squeeze=False)
    ax = ax.ravel()
    for a in ax[len(sp):]:
        a.set_visible(False)
    for a, v in zip(ax, sp):
        for st in [s for s in steps if s['v'] == v]:
            a.plot(st['t'], st['y'], color=kind_col[st['kind']], lw=1.0, alpha=0.85)
        a.axhline(1, color=c.INK_3, ls=':', lw=0.8)
        a.axhline(0, color=c.INK_3, lw=0.8)
        a.axvline(0, color=c.INK_3, lw=0.8)
        a.set_xlim(-PRE_S, POST_S)
        a.set_ylim(-0.6, 2.0)
        a.set_title(f'{v:g} rad/s output ({c.GEAR_RATIO * v / c.TWO_PI:.2f} motor rev/s)')
        a.set_xlabel('Time from current step (s)')
    ax[0].set_ylabel('Normalised torque change (previous dwell = 0, new dwell = 1)')
    handles = [Line2D([], [], color=col, lw=2, label=f'step within {k}' if k != 'crosses 0 A' else 'step to/from 0 A')
               for k, col in kind_col.items()]
    ax[0].legend(handles=handles, loc='upper right')
    t63 = np.nanmedian([s['t63'] for s in steps])
    t90 = np.nanmedian([s['t90'] for s in steps])
    slow = [s for s in steps if s['t90'] > SLOW_T90_S]
    # A slow step is the gear train stuck while LOAD's velocity loop integrates
    # the new torque from its speed error (~ the commanded speed): ramp time ~
    # dT / (K v). K is then LOAD's effective integral stiffness at the output.
    k_int = [abs(s['dP']) / (s['v'] * s['t90']) for s in slow]
    t90_by_speed = {f'{v:g}': float(np.nanmedian([s['t90'] for s in slow if s['v'] == v])) if any(s['v'] == v for s in slow) else None
                    for v in c.speeds(d)}
    fig.suptitle(f'Skip torque response to DUT current steps while sliding ({c.RUN})')
    msg = f'{len(steps) - len(slow)} of {len(steps)} steps reach 90% within {1e3 * SLOW_T90_S:.0f} ms. '
    if slow:
        msg += ('The rest ramp for ' + ', '.join(f'{v} rad/s: {t:.1f} s' for v, t in t90_by_speed.items() if t) +
                f' (implied LOAD integral stiffness ~{np.median(k_int):.0f} Nm/rad at the output). ')
    c.note(fig, msg + 'Later wiggles are angle-locked ripple, not relaxation.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    c.save(fig, SUB, 'Skip_torque_step_response.png')

    c.save_csv(SUB, 'Skip_step_response.csv',
               ['output_speed_cmd_rad_s', 'from_A_along', 'to_A_along', 'kind', 'delta_P_Nm', 't63_s', 't90_s'],
               [[s['speed'], s['from'], s['to'], s['kind'], s['dP'], s['t63'], s['t90']] for s in steps])

    # -- Wander within a dwell: 5 s means vs time, with the output angle travelled -------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    wander = {}
    for a, q in zip(ax, ('forward', 'backdrive')):
        for w in [w for w in dw if w['quadrant'] == q and abs(w['i_along_nom']) == c.i_max(d)]:
            sl = c.dwell_slice(w)
            P = -w['dir'] * d['T_cell'][sl]
            n = int(5 * c.FS_HZ)
            m = P[:len(P) // n * n].reshape(-1, n).mean(1)
            tt = (np.arange(len(m)) + 0.5) * 5.0
            a.plot(tt, m - m.mean(), color=c.SPEED_COLORS[w['speed_abs']], ls=c.DIR_LS[w['dir']], lw=1.4,
                   marker='o', ms=3)
            wander[f'{q}_{w["speed"]:+g}'] = float(np.ptp(m))
        a.axhline(0, color=c.INK_3, lw=0.8)
        a.set_xlabel('Time in dwell (s)')
        a.set_title(f'{"DUT drives" if q == "forward" else "LOAD backdrives"}, {c.i_max(d):g} A: 5 s means')
    ax[0].set_ylabel('Torque along motion minus dwell mean (Nm)')
    handles = [Line2D([], [], color=col, lw=2, label=f'{v:g} rad/s') for v, col in c.SPEED_COLORS.items() if v in c.speeds(d)]
    handles += [Line2D([], [], color=c.INK_2, ls=c.DIR_LS[s], lw=1.6, label=c.DIR_LABEL[s]) for s in (-1, 1)]
    ax[0].legend(handles=handles, loc='upper right', ncol=2)
    fig.suptitle('Skip torque wander within a dwell at constant current')
    c.note(fig, f'A 5 s mean still contains the once-per-motor-rev ripple at the slowest speed '
                f'({c.TWO_PI / (c.GEAR_RATIO * sp[0]):.1f} s per motor rev); at faster speeds it averages out.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    c.save(fig, SUB, 'Skip_within_dwell_wander.png')

    c.record('time_response', {
        'median_t63_s': float(t63), 'median_t90_s': float(t90),
        'n_steps': len(steps), 'n_slow_steps': len(slow),
        'slow_step_t90_by_speed_s': t90_by_speed,
        'slow_step_kinds': {k: sum(1 for s in slow if s['kind'] == k) for k in kind_col},
        'implied_load_integral_stiffness_Nm_per_rad': float(np.median(k_int)) if k_int else None,
        'n_steps_by_speed': {f'{v:g}': sum(1 for s in steps if s['v'] == v) for v in sp},
        'median_t90_by_speed_s': {f'{v:g}': float(np.nanmedian([s['t90'] for s in steps if s['v'] == v])) for v in sp},
        't63_by_kind_s': {k: float(np.nanmedian([s['t63'] for s in steps if s['kind'] == k])) for k in kind_col},
        't90_by_kind_s': {k: float(np.nanmedian([s['t90'] for s in steps if s['kind'] == k])) for k in kind_col},
        'wander_5s_means_pkpk_Nm': wander,
    })


if __name__ == '__main__':
    c.use_run_from_argv()
    run(c.load())
