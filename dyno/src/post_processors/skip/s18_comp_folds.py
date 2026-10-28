"""Before/after views of the motor-rev torque ripple from the R4 compensation log.

Before = C7_qdrag (best base law: branches, speed, split drag; no ripple terms),
labelled "current Skip feedforward". After = C8_all_q (the same plus the
h1/h6/hx ripple terms), labelled "proposed feedforward". C0_naive, labelled
"typical feedforward (one torque constant)", is added to the intensity figure.
In R4, C7 is our refit base law standing in for the client's law; a run with the
client's exact law can be plotted by pointing BEFORE/AFTER/NAIVE at its profiles.

Per hold the torque error e = cell - target is put on a uniform motor-angle
grid (skip_orders.STEP_M) and fitted like s04/s06: MECH + ELEC orders per motor
rev with a Legendre drift as nuisance (degree capped by the revs available).

  fold       e minus the drift, folded over one motor rev in the frame the
             compensation uses (motor angle + phase_offset_rad), coloured by
             target torque along motion (red backdrive, blue forward)
  intensity  pk-pk of the fitted MECH orders (motor-locked, not multiples of
             21) against target torque and against mean DUT current

Run:
    python s18_comp_folds.py [LOG.hdf5]

Outputs (results/feedforward_compensation_2026-10-26/figures/):
  07_fold_before_after_{0.1,0.5,2}rad_s.png
  08_mech_ripple_vs_torque_before_after.png
  ../data/mech_ripple_per_hold.csv
"""

import csv
import os
import re
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import skip_common as c
import skip_orders as so

LOG = os.path.join(c.SKIP_LOGS, 'feedforward_compensation_tests', '06_R4_compensation_test',
                   'skip_comp_R4_qdrag.hdf5')
OUT = os.path.join(c.SKIP_LOGS, 'results', 'feedforward_compensation_2026-10-26')
PHASE_OFFSET = 0.1879        # phase_offset_rad as R4 ran it (data/dut_compensation_block_as_run_R4.yaml)
FLAG = re.compile(r'^V(?P<v>\d+)-RUN\d+-SETPOINT(?P<pt>\d+)-(?P<profile>\w+)$')
# Which profiles play which role. Point these at the new profile names for a
# later run (e.g. the typical / client / proposed comparison).
BEFORE, AFTER, NAIVE = 'C7_qdrag', 'C8_all_q', 'C0_naive'
LABEL = {NAIVE: 'typical feedforward (one torque constant)',
         BEFORE: 'current Skip feedforward',
         AFTER: 'proposed feedforward'}
PROFILE_STYLE = {NAIVE: ('#8a8984', LABEL[NAIVE]),
                 BEFORE: ('#0b0b0b', LABEL[BEFORE]),
                 AFTER: ('#e34948', LABEL[AFTER])}
M_BINS = 180
TORQUE_SHADE = {4: 0, 8: 1, 12: 2}
FADE = (0.25, 1.0)


def torque_color(t_along):
    return c._SHADES[1 if t_along > 0 else -1][TORQUE_SHADE[int(round(abs(t_along)))]]


def torque_label(t_along):
    return f'{"forward" if t_along > 0 else "backdrive"} {abs(t_along):g} Nm'


def ripple_gain(v):
    return float(np.clip((FADE[1] - v) / (FADE[1] - FADE[0]), 0, 1))


def load_holds(path):
    with h5py.File(path, 'r') as f:
        ids = [b.decode() if isinstance(b, bytes) else str(b) for b in f['behavior_ids'][:]]
        idx = f['behavior_indices'][:]
        cell = f['load_torque'][:].astype(float)
        tgt = f['dut_comp_target'][:].astype(float)
        th = f['dut_output_position'][:].astype(float)     # DUT motor angle (rad)
        vc = f['load_velocity_command'][:].astype(float)
        cur = f['dut_comp_current'][:].astype(float)
    holds = []
    for bid, (i0, i1) in zip(ids, idx):
        m = FLAG.match(bid)
        if not m or m['profile'] not in PROFILE_STYLE:
            continue
        sl = slice(int(i0), int(i1) + 1)
        ok = np.isfinite(tgt[sl]) & np.isfinite(cell[sl])
        speed = float(np.median(vc[sl]))
        target = float(np.median(tgt[sl][ok]))
        e = (cell[sl] - tgt[sl])[ok]
        a = th[sl][ok]
        y, _ = c.angle_resample(a, e, so.STEP_M)
        n = len(y)
        sgn = np.sign(a[-1] - a[0])
        th_grid = a[0] + sgn * so.STEP_M * (np.arange(n) + 0.5) + PHASE_OFFSET
        revs = n * so.STEP_M / c.TWO_PI
        cols = [(f'm{k}', th_grid, k) for k in so.MECH_ORDERS]
        cols += [(f'e{k}', th_grid, k) for k in so.ELEC_ORDERS]
        deg = int(min(so.SLOW_DEGREE, max(1, np.floor(revs) - 1)))
        amps, comps, slow, resid = c.lsq_orders(cols, y, c.legendre_nuisance(n, deg))
        th1 = np.linspace(0, c.TWO_PI, 2049)
        mech = so.recon(amps, so.MECH_ORDERS, 'm', th1)
        elec = so.recon(amps, so.ELEC_ORDERS, 'e', th1)
        holds.append({'flag': bid, 'profile': m['profile'], 'speed': speed,
                      'v': round(abs(speed), 2), 'dir': int(np.sign(speed)),
                      'target': target, 't_along': target * np.sign(speed),
                      'i_along': float(np.mean(cur[sl][ok])) * np.sign(speed),
                      'revs': revs, 'th': th_grid, 'y': y - slow,
                      'mech_pp': so.pkpk(mech), 'elec_pp': so.pkpk(elec),
                      'A1': amps['m1'][0], 'A2': amps['m2'][0], 'A6': amps['m6'][0],
                      'resid_rms': float(np.std(resid))})
    return holds


def fold_figure(holds, v):
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5), sharex=True, sharey=True)
    lim = 0
    for i, s in enumerate((-1, 1)):
        for j, p in enumerate((BEFORE, AFTER)):
            a = ax[i, j]
            hs = sorted([h for h in holds if h['v'] == v and h['dir'] == s and h['profile'] == p],
                        key=lambda h: -abs(h['t_along']))
            for h in hs:
                x, m, _ = c.fold(h['th'], h['y'], M_BINS, min_count=1)
                m = m - np.nanmean(m)
                lim = max(lim, np.nanmax(np.abs(m)))
                a.plot(x, m, color=torque_color(h['t_along']), lw=1.4)
            a.axhline(0, color=c.INK_3, lw=0.8)
            pp = np.mean([h['mech_pp'] for h in hs])
            a.set_title(f'{s * v:+g} rad/s output, {LABEL[p]}\nmean mechanical ripple {pp:.2f} Nm pk-pk')
            a.set_xlim(0, 360)
            a.set_xticks(range(0, 361, 60))
            if i == 1:
                a.set_xlabel('Motor angle + phase offset (deg, one motor rev = 1/26 output rev)')
        ax[i, 0].set_ylabel(f'{c.DIR_LABEL[s]}\nCell torque ripple (Nm)')
    lim = np.ceil(lim * 4) / 4
    for a in ax.flat:
        a.set_ylim(-lim, lim)
    handles = [Line2D([], [], color=torque_color(t), lw=2, label=torque_label(t))
               for t in (-12, -8, -4, 4, 8, 12)]
    fig.legend(handles=handles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0.02))
    g = ripple_gain(v)
    fig.suptitle(f'Skip torque ripple over one motor revolution at {v:g} rad/s: '
                 f'{LABEL[BEFORE]} vs {LABEL[AFTER]}')
    c.note(fig, f'R4 log (2026-10-26), open loop, target output torque held per line. Cell - target, slow drift '
                f'removed, folded over the motor rev. Ripple terms (1/rev, 2/rev, 6/rev) act at '
                f'{100 * g:.0f}% at this speed (fade 0.25 -> 1 rad/s)'
                + ('; at 2 rad/s they are off, so the two columns differ only by run-to-run scatter.'
                   if g == 0 else '.'), y=0.0)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    path = os.path.join(OUT, 'figures', f'07_fold_before_after_{v:g}rad_s.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', os.path.relpath(path, c.DYNO_DIR))


def intensity_figure(holds):
    speeds = sorted({h['v'] for h in holds})
    fig, ax = plt.subplots(2, len(speeds), figsize=(5.2 * len(speeds), 9), sharey=True, squeeze=False)
    for j, v in enumerate(speeds):
        for i, (key, xlabel) in enumerate((('t_along', 'Target output torque along motion (Nm)\n'
                                                       '[- = backdrive, + = forward]'),
                                           ('i_along', 'Mean DUT current along motion (A)\n'
                                                       '[- = backdrive, + = forward]'))):
            a = ax[i, j]
            for p, (col, _) in PROFILE_STYLE.items():
                for s in (-1, 1):
                    hs = sorted([h for h in holds if h['v'] == v and h['dir'] == s and h['profile'] == p],
                                key=lambda h: h[key])
                    a.plot([h[key] for h in hs], [h['mech_pp'] for h in hs], color=col,
                           marker=c.DIR_MARKER[s], ls=c.DIR_LS[s], lw=1.3, ms=6)
            a.axvline(0, color=c.INK_3, lw=0.8)
            a.set_ylim(bottom=0)
            a.set_xlabel(xlabel)
            if i == 0:
                a.set_title(f'{v:g} rad/s output (ripple terms at {100 * ripple_gain(v):.0f}%)')
        ax[0, j].set_ylabel('Motor-locked mechanical ripple, pk-pk (Nm)')
        ax[1, j].set_ylabel('Motor-locked mechanical ripple, pk-pk (Nm)')
    handles = [Line2D([], [], color=col, lw=2, label=lab) for col, lab in PROFILE_STYLE.values()]
    handles += [Line2D([], [], color=c.INK_2, marker=c.DIR_MARKER[s], ls=c.DIR_LS[s], label=c.DIR_LABEL[s])
                for s in (-1, 1)]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Skip motor-locked mechanical ripple vs torque and current, by feedforward')
    c.note(fig, 'Pk-pk of the fitted motor-locked orders that are not multiples of 21 (1-20, 27, 35-37, 57, 72 per '
                'motor rev) in cell - target, per R4 hold. Naive holds land at a different actual torque (e.g. '
                'about -21 Nm for a -12 Nm backdrive target), so they sit at the target but carry the ripple of '
                'the torque actually reached.', y=-0.02)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    path = os.path.join(OUT, 'figures', '08_mech_ripple_vs_torque_before_after.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', os.path.relpath(path, c.DYNO_DIR))


def main():
    c.apply_style()
    holds = load_holds(sys.argv[1] if len(sys.argv) > 1 else LOG)
    for v in sorted({h['v'] for h in holds}):
        fold_figure(holds, v)
    intensity_figure(holds)
    keys = ('flag', 'profile', 'speed', 'target', 't_along', 'i_along', 'revs',
            'mech_pp', 'elec_pp', 'A1', 'A2', 'A6', 'resid_rms')
    path = os.path.join(OUT, 'data', 'mech_ripple_per_hold.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(keys)
        for h in holds:
            w.writerow([h[k] if isinstance(h[k], str) else f'{h[k]:.5g}' for k in keys])
    print('  wrote', os.path.relpath(path, c.DYNO_DIR))
    print(f'{"v":>5s} {"profile":10s} {"mech_pp":>8s} {"A1":>6s} {"A2":>6s} {"A6":>6s}')
    for v in sorted({h['v'] for h in holds}):
        for p in PROFILE_STYLE:
            hs = [h for h in holds if h['v'] == v and h['profile'] == p]
            print(f'{v:5.2f} {p:10s} ' + ' '.join(f'{np.mean([h[k] for h in hs]):{8 if k == "mech_pp" else 6}.3f}'
                                                  for k in ('mech_pp', 'A1', 'A2', 'A6')))


if __name__ == '__main__':
    main()
