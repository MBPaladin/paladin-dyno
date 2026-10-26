"""Gearbox efficiency maps for Skip from one torque cell, a la
src/analysis/processors/efficiency.py.

That processor has a torque cell on each side of the gearbox. Here there is only
the output (load) cell; the input side is known only through the DUT current,
so input torque = kt x current, and kt itself is not measured. What can still
be said:

  ROUND TRIP (kt-free). At the same speed and |current| the grid runs both power
  flows. Forward the output gives T_f = eta_f*26*kt*I; backdrive the LOAD must
  push T_b = 26*kt*I/eta_b. So

        eta_f * eta_b = T_f / T_b

  per (speed, |current|) cell, with kt cancelling. Its square root is the
  efficiency if both directions are equally lossy -- the symmetric model, now
  per operating point and including the no-load drag, rather than per slope.

  ONE-WAY (needs kt). With a kt, input torque is known and

        forward    eta_f = |T_out| / (26 kt I)       (DUT drives)
        backdrive  eta_b = 26 kt I / |T_out|         (LOAD drives)

  kt defaults to the symmetric-model value from the quasi-static slopes
  (s02, ~0.033 Nm/A in the drive's current units). The maps scale as 1/kt
  (forward) and kt (backdrive): a 10% kt error moves them 10% in opposite
  directions, and the round-trip map is the check on any kt.

  MOTOR + GEARBOX (needs R and kt, optional). With a phase resistance the
  copper loss can be added: P_elec ~ kt*I*w_m + 1.5*R*I^2 (I as the q-axis
  current amplitude). Iron, drive and switching losses are not modelled, and
  the logs carry no DUT voltage, so this is an estimate until the Elmo's DC-link
  voltage and current (or an external power meter) are logged. Set R_PHASE_OHM
  to enable it.

Outputs (results/11_efficiency/):
  Skip_efficiency_maps.png       round-trip, forward and backdrive maps
  Skip_efficiency_vs_load.png    efficiency vs output torque, per speed
  Skip_efficiency_grid.csv
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import skip_common as c
import s02_torque_current as s02

SUB = '11_efficiency'
KT_OVERRIDE = None        # Nm/A; None = symmetric-model kt from the quasi-static slopes
R_PHASE_OHM = None        # line-to-neutral phase resistance; None = no motor map
CMAP = 'Blues'


def kt_estimate(d):
    if KT_OVERRIDE:
        return float(KT_OVERRIDE), 'set by hand'
    fit = s02.branch_fit([w for w in d['dwells'] if w['speed_abs'] <= c.COGGING_MAX_SPEED])
    return fit['kt_symmetric'], 'symmetric model, quasi-static slopes'


def grid(d, kt):
    """Per (speed, |I|): both directions averaged forward and backdrive output
    torques, and the efficiencies built from them."""
    sp = c.speeds(d)
    levels = sorted({abs(i) for i in c.currents_along(d) if i != 0})
    G = {k: np.full((len(levels), len(sp)), np.nan)
         for k in ('T_f', 'T_b', 'I', 'round_trip', 'eta_sym', 'eta_f', 'eta_b', 'P_out_f', 'eta_motor_gb')}
    for j, v in enumerate(sp):
        for i, lv in enumerate(levels):
            fw = [w for w in d['dwells'] if w['speed_abs'] == v and w['i_along_nom'] == lv]
            bd = [w for w in d['dwells'] if w['speed_abs'] == v and w['i_along_nom'] == -lv]
            if not fw or not bd:
                continue
            # P = torque LOAD pushes along the motion: negative forward (LOAD brakes), positive backdrive
            T_f = -np.mean([w['P'] for w in fw])
            T_b = np.mean([w['P'] for w in bd])
            I_f = np.mean([abs(w['current']) for w in fw])
            I_b = np.mean([abs(w['current']) for w in bd])
            G['T_f'][i, j], G['T_b'][i, j], G['I'][i, j] = T_f, T_b, 0.5 * (I_f + I_b)
            # current-normalised so a small mismatch between the two |I| does not leak in
            G['round_trip'][i, j] = (T_f / I_f) / (T_b / I_b)
            G['eta_sym'][i, j] = np.sqrt(max(G['round_trip'][i, j], 0.0))
            G['eta_f'][i, j] = T_f / (c.GEAR_RATIO * kt * I_f)
            G['eta_b'][i, j] = c.GEAR_RATIO * kt * I_b / T_b
            G['P_out_f'][i, j] = T_f * v
            if R_PHASE_OHM:
                w_m = c.GEAR_RATIO * v
                p_in = kt * I_f * w_m + 1.5 * R_PHASE_OHM * I_f ** 2
                G['eta_motor_gb'][i, j] = T_f * v / p_in
    return sp, levels, G


def heatmap(ax, M, T, sp, levels, title, vmin=0.3, vmax=1.0):
    im = ax.imshow(np.ma.masked_invalid(M), origin='lower', aspect='auto', cmap=CMAP, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(sp)))
    ax.set_xticklabels([f'{v:g}' for v in sp])
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels([f'{lv:g} A' for lv in levels])
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel('Output speed (rad/s)')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                dark = (M[i, j] - vmin) / (vmax - vmin) > 0.6
                ax.text(j, i + 0.08, f'{100 * M[i, j]:.0f}%', ha='center', va='center', fontsize=10,
                        fontweight='bold', color='white' if dark else c.INK)
                if T is not None:
                    ax.text(j, i - 0.22, f'{T[i, j]:.1f} Nm', ha='center', va='center', fontsize=7.5,
                            color='white' if dark else c.INK_2)
    return im


def run(d):
    c.apply_style()
    kt, kt_src = kt_estimate(d)
    sp, levels, G = grid(d, kt)

    # -- Figure 1: maps ---------------------------------------------------------------------
    n = 4 if R_PHASE_OHM else 3
    fig, ax = plt.subplots(1, n, figsize=(6 * n, 5.2), sharey=True)
    heatmap(ax[0], G['eta_sym'], None, sp, levels, 'sqrt(eta_f x eta_b): kt-free')
    heatmap(ax[1], G['eta_f'], G['T_f'], sp, levels, f'Forward (DUT drives), kt = {kt:.4f}')
    im = heatmap(ax[2], G['eta_b'], G['T_b'], sp, levels, f'Backdrive (LOAD drives), kt = {kt:.4f}')
    if R_PHASE_OHM:
        im = heatmap(ax[3], G['eta_motor_gb'], G['T_f'], sp, levels,
                     f'Motor + gearbox, forward (R = {R_PHASE_OHM:g} ohm, copper only)', vmin=0.0)
    ax[0].set_ylabel('|DUT current|')
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label='Efficiency')
    fig.suptitle(f'Skip gearbox efficiency map ({c.RUN}), including no-load drag')
    c.note(fig, f'Left needs no kt: forward and backdrive output torque at the same |current| and speed give '
                f'eta_f x eta_b directly. Middle/right use kt = {kt:.4f} Nm/A ({kt_src}); they scale as 1/kt and kt '
                'respectively. Small text = output torque (forward: delivered; backdrive: what LOAD pushed). '
                'Both directions of rotation averaged.', y=-0.02)
    c.save(fig, SUB, 'Skip_efficiency_maps.png')

    # -- Figure 2: efficiency vs output torque, per speed ------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for j, v in enumerate(sp):
        col = c.SPEED_COLORS[v]
        ax[0].plot(G['T_f'][:, j], G['eta_f'][:, j], '-o', color=col, ms=6)
        ax[1].plot(G['T_b'][:, j], G['eta_b'][:, j], '-o', color=col, ms=6)
    # what the quasi-static line predicts: T_f = K_f I - c_f, so eta_f = (K_f I - c_f) / (26 kt I)
    fit = s02.branch_fit([w for w in d['dwells'] if w['speed_abs'] <= c.COGGING_MAX_SPEED])
    ii = np.linspace(1, max(levels), 50)
    ax[0].plot(fit['forward']['K'] * ii - fit['forward']['c'],
               (fit['forward']['K'] * ii - fit['forward']['c']) / (c.GEAR_RATIO * kt * ii),
               color=c.INK_3, ls='--', lw=1.2, label='quasi-static line fit')
    ax[1].plot(fit['backdrive']['K'] * ii + fit['backdrive']['c'],
               c.GEAR_RATIO * kt * ii / (fit['backdrive']['K'] * ii + fit['backdrive']['c']),
               color=c.INK_3, ls='--', lw=1.2, label='quasi-static line fit')
    ax[0].set_title('Forward: DUT drives the load')
    ax[1].set_title('Backdrive: LOAD drives the DUT')
    ax[0].set_xlabel('Output torque delivered (Nm)')
    ax[1].set_xlabel('Output torque LOAD pushed (Nm)')
    ax[0].set_ylabel(f'Efficiency (kt = {kt:.4f})')
    for a in ax:
        a.set_ylim(0, 1)
        a.set_xlim(left=0)
    handles = [Line2D([], [], color=c.SPEED_COLORS[v], marker='o', lw=1.6, label=f'{v:g} rad/s') for v in sp]
    handles.append(Line2D([], [], color=c.INK_3, ls='--', label='quasi-static line fit'))
    ax[1].legend(handles=handles, loc='lower right', ncol=2)
    fig.suptitle('Skip efficiency vs load: drag dominates at light load, speed lifts the ceiling')
    c.note(fig, 'Efficiency falls toward light load because the ~0.7 Nm + 0.34 Nm/(rad/s) no-load drag is a fixed '
                'loss. The load-dependent part improves with speed (s10).', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    c.save(fig, SUB, 'Skip_efficiency_vs_load.png')

    rows = []
    for i, lv in enumerate(levels):
        for j, v in enumerate(sp):
            rows.append([v, lv, G['I'][i, j], G['T_f'][i, j], G['T_b'][i, j], G['round_trip'][i, j],
                         G['eta_sym'][i, j], G['eta_f'][i, j], G['eta_b'][i, j], G['P_out_f'][i, j]])
    c.save_csv(SUB, 'Skip_efficiency_grid.csv',
               ['output_speed_rad_s', 'current_level_A', 'mean_abs_current_A', 'T_forward_out_Nm',
                'T_backdrive_in_Nm', 'eta_f_x_eta_b', 'eta_symmetric', 'eta_forward_kt', 'eta_backdrive_kt',
                'P_out_forward_W'], rows)

    def tab(M):
        return {f'{lv:g}A': {f'{v:g}': float(M[i, j]) for j, v in enumerate(sp)} for i, lv in enumerate(levels)}
    c.record('efficiency', {
        'kt_used_Nm_per_A': kt, 'kt_source': kt_src,
        'eta_symmetric': tab(G['eta_sym']), 'eta_forward': tab(G['eta_f']), 'eta_backdrive': tab(G['eta_b']),
    })
    return G


if __name__ == '__main__':
    c.use_run_from_argv()
    run(c.load())
