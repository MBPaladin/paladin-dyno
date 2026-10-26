"""How much of the torque variation in each dwell is locked to which angle, and
how much is not locked to anything (stick-slip, noise, drift).

Per dwell, evaluated back on the time samples (what the cell actually showed):

  ELEC    electrical orders (21k per motor rev), from the per-dwell angle fit
  MECH    motor-locked non-electrical orders
  OUTPUT  output-angle-locked orders 1-4 plus the 25/27 sidebands, from a fit
          shared by the dwells at the same current and direction (three speeds,
          ~3.3 output revs between them; one dwell alone covers too little)
  REST    whatever is left about the mean: stick-slip, noise, slow drift

The angle-locked part is what a position-indexed feedforward table could
cancel; REST is what only feedback (or mechanics) can deal with.

Outputs (results/06_decomposition/):
  Skip_torque_variance_breakdown.png
  Skip_output_phase_profile.png
  Skip_variance_breakdown.csv
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import skip_common as c
import skip_orders as so

SUB = '06_decomposition'
OUT_ORDERS = (1, 2, 3, 4, 25, 27)
COMPONENT_COLORS = {'ELEC': '#6250d6', 'MECH': '#2a78d6', 'OUTPUT': '#1baf7a', 'REST': '#c3c2b7'}
COMPONENT_LABELS = {'ELEC': 'electrical (21k per motor rev)', 'MECH': 'motor-rev locked (gearbox input side)',
                    'OUTPUT': 'output-rev locked (gearbox output side)', 'REST': 'not angle-locked (stick-slip, noise)'}


def time_components(d, w, out_eval):
    """Angle-fit components evaluated at every time sample of dwell w."""
    sl = c.dwell_slice(w)
    th = d['th_m'][sl]
    travel = np.abs(th - th[0])
    rs = so.resample_dwell(d, w)
    f = so.fit_dwell(d, w)
    g = (np.arange(rs['n']) + 0.5) * so.STEP_M
    y = d['T_cell'][sl]
    elec = np.interp(travel, g, f['elec'])
    mech = np.interp(travel, g, f['mech'])
    out = out_eval(d['th_o'][sl])
    out = out - out.mean()
    rest = y - y.mean() - elec - mech - out
    return y, {'ELEC': elec, 'MECH': mech, 'OUTPUT': out, 'REST': rest}


def run(d):
    c.apply_style()
    dw = d['dwells']

    groups = {}
    for s in (-1, 1):
        for ia in c.currents_along(d):
            members = [w for w in dw if w['dir'] == s and w['i_along_nom'] == ia]
            groups[(s, ia)] = so.fit_output_group(d, members, orders=OUT_ORDERS)

    shares, rows = {}, []
    for w in dw:
        ab, _, ev = groups[(w['dir'], w['i_along_nom'])]
        y, comp = time_components(d, w, ev)
        vt = float(np.var(y))
        sh = {k: float(np.var(v)) / vt for k, v in comp.items()}
        shares[w['n']] = (sh, float(np.std(y)))
        rows.append([w['n'], w['speed'], w['i_along_nom'], w['quadrant'], float(np.std(y))]
                    + [sh[k] for k in ('ELEC', 'MECH', 'OUTPUT', 'REST')]
                    + [float(np.std(comp[k])) for k in ('ELEC', 'MECH', 'OUTPUT', 'REST')])
    c.save_csv(SUB, 'Skip_variance_breakdown.csv',
               ['dwell', 'output_speed_cmd_rad_s', 'current_along_motion_A', 'quadrant', 'torque_std_Nm',
                'share_ELEC', 'share_MECH', 'share_OUTPUT', 'share_REST',
                'rms_ELEC_Nm', 'rms_MECH_Nm', 'rms_OUTPUT_Nm', 'rms_REST_Nm'], rows)

    # -- Stacked shares, one panel per speed, both directions averaged ------------------
    cur = c.currents_along(d)
    ns = len(c.speeds(d))
    fig, ax = plt.subplots(1, ns, figsize=(5.3 * ns, 5.6), sharey=True, squeeze=False)
    ax = ax[0]
    for a, v in zip(ax, c.speeds(d)):
        bottoms = np.zeros(len(cur))
        stds = []
        for comp in ('ELEC', 'MECH', 'OUTPUT', 'REST'):
            vals = []
            for ia in cur:
                ws = [w for w in dw if w['speed_abs'] == v and w['i_along_nom'] == ia]
                vals.append(100 * np.mean([shares[w['n']][0][comp] for w in ws]))
            a.bar(range(len(cur)), vals, bottom=bottoms, color=COMPONENT_COLORS[comp], width=0.72,
                  edgecolor=c.SURFACE, linewidth=1.5)
            bottoms += np.array(vals)
        for j, ia in enumerate(cur):
            ws = [w for w in dw if w['speed_abs'] == v and w['i_along_nom'] == ia]
            sd = np.mean([shares[w['n']][1] for w in ws])
            a.text(j, min(bottoms[j], 112) + 2, f'{sd:.2f}', ha='center', va='bottom', fontsize=8.5, color=c.INK_2)
        a.set_xticks(range(len(cur)))
        a.set_xticklabels([f'{i:+g}' for i in cur])
        a.set_xlabel('DUT current along motion (A)')
        a.set_title(f'{v:g} rad/s output')
        a.set_ylim(0, 125)
        a.grid(axis='x', visible=False)
    ax[0].set_ylabel('Share of torque variance within the dwell (%)')
    handles = [Line2D([], [], color=COMPONENT_COLORS[k], lw=8, label=COMPONENT_LABELS[k])
               for k in ('ELEC', 'MECH', 'OUTPUT', 'REST')]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Skip: where the torque variation comes from (numbers on top = total torque std, Nm)')
    c.note(fig, 'Shares are var(component)/var(torque), both directions averaged; they sum to ~100% because the '
                'components are nearly orthogonal. Angle-locked shares are what a position-indexed feedforward '
                'could remove; the gray share needs feedback or mechanical change.', y=-0.02)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    c.save(fig, SUB, 'Skip_torque_variance_breakdown.png')

    # -- Output-phase profile (orders 1-4 only, sidebands excluded) ----------------------
    th = np.linspace(0, c.TWO_PI, 721)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), sharey=True)
    prof_pp = {}
    for a, s in zip(ax, (-1, 1)):
        for ia in cur:
            ab, _, _ = groups[(s, ia)]
            prof = np.zeros_like(th)
            for k in (1, 2, 3, 4):
                A, B = ab[k]
                prof += A * np.cos(k * th) + B * np.sin(k * th)
            a.plot(th * 180 / np.pi, prof, color=c.current_color(ia), lw=1.8 if ia == 0 else 1.4)
            prof_pp[f'dir{s:+d}_I{ia:+g}'] = so.pkpk(prof)
        a.axhline(0, color=c.INK_3, lw=0.8)
        a.set_xlim(0, 360)
        a.set_xticks(range(0, 361, 60))
        a.set_xlabel('Output angle (deg, load encoder)')
        a.set_title(c.DIR_LABEL[s])
    ax[0].set_ylabel('Output-locked torque, orders 1-4 (Nm, cell frame)')
    handles = [Line2D([], [], color=c.current_color(i), lw=2, label=c.current_label(i)) for i in cur]
    fig.legend(handles=handles, loc='lower center', ncol=7, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Skip torque locked to the output (gearbox) angle, low orders')
    revs = sum(abs(d['th_o'][w['i1'] - 1] - d['th_o'][w['i0']]) for w in dw) / c.TWO_PI / (2 * len(cur))
    c.note(fig, f'Joint fit across the {ns} speeds at each current and direction (~{revs:.1f} output revs). Orders 25/27 '
                '(cycloid sidebands) are fitted too but left out of this plot; see the output-order spectrum.',
           y=-0.02)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    c.save(fig, SUB, 'Skip_output_phase_profile.png')

    def mean_share(cond, comp):
        return float(np.mean([shares[w['n']][0][comp] for w in dw if cond(w)]))
    rec = {}
    im = c.i_max(d)
    for name, cond in (('zero', lambda w: w['i_along_nom'] == 0),
                       ('forward_imax', lambda w: w['i_along_nom'] == im),
                       ('backdrive_imax', lambda w: w['i_along_nom'] == -im),
                       ('forward_all', lambda w: w['quadrant'] == 'forward'),
                       ('backdrive_all', lambda w: w['quadrant'] == 'backdrive')):
        rec[name] = {k: mean_share(cond, k) for k in ('ELEC', 'MECH', 'OUTPUT', 'REST')}
        rec[name]['torque_std_Nm'] = float(np.mean([shares[w['n']][1] for w in dw if cond(w)]))
    rec['output_profile_pkpk_orders_1_4'] = prof_pp
    rec['sideband_amplitudes'] = {f'dir{s:+d}_I{ia:+g}': {str(k): float(np.hypot(*groups[(s, ia)][0][k])) for k in (25, 27)}
                                  for s in (-1, 1) for ia in cur}
    c.record('decomposition', rec)
    return shares


if __name__ == '__main__':
    c.use_run_from_argv()
    run(c.cogging_view(c.load()))
