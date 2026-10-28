"""Why the low-speed ripple table fails at 2 rad/s, from the R4 compensation log.

Four checks, all on R4 (2026-10-26) holds:

  rig     cell torque vs LOAD motor torque (KT_LOAD x load_current) from the
          broadband noise in every hold (Welch cross-spectrum), per speed.
          Flat gain with linear phase = a pure delay between the two signals,
          not a structural mode. The slope gives the delay.
  phasor  1/rev and 6/rev of cell - target per hold (motor angle + phase
          offset frame, the frame the law injects in), against the low-speed
          table the law ran. At 2 rad/s, C7 and C8 run with ripple terms off
          (fade), so the pair measures run-to-run repeatability.
  model   at 2 rad/s, per quadrant cell, a complex c0 + c1|tau| fit across the
          three torques, scored leave-one-out: can a per-speed table of the
          law's form capture the pattern?
  path    at 0.5 rad/s, (cell_C8 - cell_C7) / (-injected ripple) per order:
          how much of an injected motor-angle harmonic reaches the cell, and
          how late.

Run:
    python s19_ripple_speed_dynamics.py [LOG.hdf5]

Outputs (results/feedforward_compensation_2026-10-26/):
  figures/09_rig_transfer_cell_vs_load.png
  figures/10_ripple_phasors_by_speed.png
  data/ripple_phasors_per_hold.csv
"""

import csv
import os
import re
import sys

import h5py
import numpy as np
import yaml
from scipy.signal import csd, welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import skip_common as c
import skip_orders as so

OUT = os.path.join(c.SKIP_LOGS, 'results', 'feedforward_compensation_2026-10-26')
LOG = os.path.join(c.SKIP_LOGS, 'feedforward_compensation_tests', '06_R4_compensation_test',
                   'skip_comp_R4_qdrag.hdf5')
BLOCK = os.path.join(OUT, 'data', 'dut_compensation_block_as_run_R4.yaml')
FLAG = re.compile(r'^V(?P<v>\d+)-RUN\d+-SETPOINT(?P<pt>\d+)-(?P<profile>\w+)$')
ORDERS = (1, 2, 6)
FS = 1000.0
SPEED_COL = {0.1: '#86b6ef', 0.5: '#2a78d6', 2.0: '#104281'}
DELAY_BAND = (3.0, 55.0)      # Hz, where cell/LOAD coherence is high at every speed


def detrend2(x):
    n = np.arange(len(x))
    return x - np.polyval(np.polyfit(n, x, 2), n)


def load(path):
    with h5py.File(path, 'r') as f:
        ids = [b.decode() if isinstance(b, bytes) else str(b) for b in f['behavior_ids'][:]]
        idx = f['behavior_indices'][:]
        sig = {k: f[k][:].astype(float) for k in
               ('load_torque', 'load_current', 'dut_comp_target', 'dut_output_position',
                'load_velocity_command', 'dut_comp_ripple')}
    holds = []
    for bid, (i0, i1) in zip(ids, idx):
        m = FLAG.match(bid)
        if not m:
            continue
        sl = slice(int(i0) + 300, int(i1))      # skip the profile switch transient
        holds.append({'flag': bid, 'v': int(m['v']) / 100, 'profile': m['profile'], 'sl': sl,
                      'dir': int(np.sign(np.median(sig['load_velocity_command'][sl]))),
                      'target': float(np.median(sig['dut_comp_target'][sl]))})
    return sig, holds


def phasors(sig, h, offset, channel):
    """Complex amplitude per order of `channel` (cell - target, or the injected
    ripple) on the motor-angle grid, in the frame the law injects in."""
    sl = h['sl']
    y = sig[channel][sl] - (sig['dut_comp_target'][sl] if channel == 'load_torque' else 0.0)
    a = sig['dut_output_position'][sl]                       # DUT motor angle (rad)
    yr, _ = c.angle_resample(a, y, so.STEP_M)
    n = len(yr)
    g = a[0] + np.sign(a[-1] - a[0]) * so.STEP_M * (np.arange(n) + 0.5) + offset
    revs = n * so.STEP_M / c.TWO_PI
    amps, *_ = c.lsq_orders([(f'm{k}', g, k) for k in ORDERS], yr,
                            c.legendre_nuisance(n, int(min(8, max(1, revs - 1)))))
    return {k: amps[f'm{k}'][0] * np.exp(1j * amps[f'm{k}'][1]) for k in ORDERS}


def cell_name(h):
    quad = 'forward' if h['target'] * h['dir'] > 0 else 'backdrive'
    return ('pos_' if h['dir'] > 0 else 'neg_') + quad


def table_phasor(block, k, cell, tau):
    for hm in block['harmonics']:
        if hm['order'] == k and cell in hm['cells']:
            p = hm['cells'][cell]
            return (p['a0'] + p['a1'] * abs(tau)) * np.exp(1j * p['phase'])
    return np.nan


# --- rig: cell vs LOAD motor torque -------------------------------------------------
def rig_transfer(sig, holds):
    out = {}
    for v in sorted({h['v'] for h in holds}):
        Pxy = Pxx = Pyy = 0.0
        for h in [h for h in holds if h['v'] == v]:
            x = detrend2(sig['load_current'][h['sl']]) * c.KT_LOAD
            y = detrend2(sig['load_torque'][h['sl']])
            fr, a = csd(x, y, FS, nperseg=1024)
            Pxy = Pxy + a
            Pxx = Pxx + welch(x, FS, nperseg=1024)[1]
            Pyy = Pyy + welch(y, FS, nperseg=1024)[1]
        H = Pxy / Pxx
        coh = np.abs(Pxy) ** 2 / (Pxx * Pyy)
        band = (fr >= DELAY_BAND[0]) & (fr <= DELAY_BAND[1]) & (coh > 0.5)
        phase = np.unwrap(np.angle(H[band]))
        slope, icpt = np.polyfit(fr[band], phase, 1, w=coh[band])
        out[v] = {'f': fr, 'H': H, 'coh': coh, 'delay_ms': -slope / c.TWO_PI * 1e3,
                  'dc_phase_deg': np.degrees(icpt), 'gain_band': float(np.median(np.abs(H[band])))}
    return out


def rig_figure(rig):
    fig, ax = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)
    for v, r in rig.items():
        lab = f'{v:g} rad/s: delay {r["delay_ms"]:.1f} ms, gain {r["gain_band"]:.2f}'
        ax[0].plot(r['f'], np.abs(r['H']), color=SPEED_COL[v], lw=1.3, label=lab)
        ax[1].plot(r['f'], np.degrees(np.angle(r['H'])), color=SPEED_COL[v], lw=1.3)
        ax[2].plot(r['f'], r['coh'], color=SPEED_COL[v], lw=1.3)
    ax[0].set_yscale('log')
    ax[0].set_ylim(1e-2, 3)
    ax[0].set_ylabel('|cell / LOAD motor torque|')
    ax[1].set_ylabel('Phase (deg)')
    ax[2].set_ylabel('Coherence')
    ax[2].set_xlabel('Frequency (Hz)')
    for a in ax:
        a.set_xlim(0, 120)
        for F, lab in ((8.3, '1/rev @ 2 rad/s'), (49.7, '6/rev @ 2 rad/s')):
            a.axvline(F, color=c.INK_3, ls=':', lw=1)
    ax[0].text(49.7, 2.2, ' 6/rev @ 2 rad/s', fontsize=8, color=c.INK_2)
    ax[0].text(8.3, 2.2, ' 1/rev', fontsize=8, color=c.INK_2)
    ax[0].legend(loc='lower left')
    fig.suptitle('Skip dyno: load cell against LOAD motor torque, from broadband noise in the R4 holds')
    c.note(fig, 'Flat gain and a straight phase line through 55 Hz: the two signals differ by a pure delay, '
                'the same at every speed. No structural mode between the LOAD motor and the cell near 50 Hz. '
                f'Delay from a coherence-weighted phase-slope fit, {DELAY_BAND[0]:g}-{DELAY_BAND[1]:g} Hz.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    path = os.path.join(OUT, 'figures', '09_rig_transfer_cell_vs_load.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', os.path.relpath(path, c.DYNO_DIR))


# --- phasors -----------------------------------------------------------------------
def phasor_figure(P, block):
    speeds = sorted({p['v'] for p in P})
    fig = plt.figure(figsize=(15, 10))
    for i, k in enumerate((1, 6)):
        lim = max(abs(p['cell_ph'][k]) for p in P if p['profile'] in ('C7_qdrag', 'C8_all_q', 'C2_speed')) * 1.1
        for j, v in enumerate(speeds):
            a = fig.add_subplot(2, len(speeds), i * len(speeds) + j + 1, projection='polar')
            for p in P:
                if p['v'] != v:
                    continue
                if p['profile'] == 'C7_qdrag' or (v == 2.0 and p['profile'] == 'C8_all_q'):
                    z = p['cell_ph'][k]
                    t_along = p['target'] * p['dir']
                    col = c._SHADES[1 if t_along > 0 else -1][{4: 0, 8: 1, 12: 2}[int(round(abs(t_along)))]]
                    hollow = p['profile'] == 'C8_all_q'
                    a.plot([np.angle(z)], [abs(z)], marker='o' if p['dir'] > 0 else 's', ms=7,
                           mfc='none' if hollow else col, mec=col, ls='none')
                    if p['profile'] == 'C7_qdrag':
                        T = table_phasor(block, k, p['cell'], p['target'])
                        a.plot([np.angle(z), np.angle(T)], [abs(z), abs(T)], color=col, lw=0.6, alpha=0.5)
                        a.plot([np.angle(T)], [abs(T)], marker='x', color=col, ms=6, ls='none')
            a.set_rlim(0, lim)
            a.set_title(f'{k}/rev at {v:g} rad/s', pad=14)
            a.tick_params(labelsize=7)
    handles = [Line2D([], [], color=c._SHADES[s][q], marker='o', ls='none',
                      label=f'{"forward" if s > 0 else "backdrive"} {(4, 8, 12)[q]} Nm')
               for s in (-1, 1) for q in range(3)]
    handles += [Line2D([], [], color=c.INK_2, marker='o', ls='none', label='output +'),
                Line2D([], [], color=c.INK_2, marker='s', ls='none', label='output -'),
                Line2D([], [], color=c.INK_2, marker='o', mfc='none', ls='none', label='2 rad/s repeat (C8, terms faded off)'),
                Line2D([], [], color=c.INK_2, marker='x', ls='none', label='low-speed table (thin line joins)')]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Skip 1/rev and 6/rev ripple phasors (cell - target) per hold, against the low-speed table')
    c.note(fig, 'C7 holds (no ripple terms). Frame: motor angle + phase_offset, the frame the law injects in; the '
                'cell lag (fig 09) is NOT removed. A good table puts each x on its dot. At 2 rad/s the filled C7 '
                'and hollow C8 dots sit on top of each other (repeatable) but far from the x (wrong table).', y=0.06)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    path = os.path.join(OUT, 'figures', '10_ripple_phasors_by_speed.png')
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('  wrote', os.path.relpath(path, c.DYNO_DIR))


def main():
    c.apply_style()
    path = sys.argv[1] if len(sys.argv) > 1 else LOG
    block = yaml.safe_load(open(BLOCK))['dut_compensation']
    offset = block['phase_offset_rad']
    sig, holds = load(path)

    rig = rig_transfer(sig, holds)
    rig_figure(rig)
    print('\nrig: cell vs KT_LOAD x load_current')
    for v, r in rig.items():
        print(f'  {v:4.1f} rad/s  delay {r["delay_ms"]:.2f} ms  DC phase {r["dc_phase_deg"]:+.0f} deg  '
              f'median gain {r["gain_band"]:.2f}')
    delay = float(np.median([r['delay_ms'] for r in rig.values()]))

    P = []
    for h in holds:
        P.append({**h, 'cell': cell_name(h), 'cell_ph': phasors(sig, h, offset, 'load_torque'),
                  'inj_ph': phasors(sig, h, offset, 'dut_comp_ripple') if h['profile'] == 'C8_all_q' else None})
    by = {(p['v'], p['dir'], round(p['target']), p['profile']): p for p in P}

    print('\nphasor: residual a perfect injection would leave, as a fraction of the ripple (median over holds)')
    print('   speed ord  |A| Nm   table   repeat-hold   phase vs table (median, spread)')
    for v in sorted({p['v'] for p in P}):
        for k in (1, 6):
            amp, rt, rr, ph = [], [], [], []
            for p in [p for p in P if p['v'] == v and p['profile'] == 'C7_qdrag']:
                A = p['cell_ph'][k]
                T = table_phasor(block, k, p['cell'], p['target'])
                amp.append(abs(A))
                rt.append(abs(A - T) / abs(A))
                ph.append(np.degrees(np.angle(A / T)))
                rep = {0.1: 'C2_speed', 2.0: 'C8_all_q'}.get(v)
                q = by.get((v, p['dir'], round(p['target']), rep)) if rep else None
                if q is not None:
                    rr.append(abs(A - q['cell_ph'][k]) / abs(A))
            print(f'   {v:4.1f} {k:3d}  {np.mean(amp):6.3f}  {np.median(rt):6.2f}   '
                  f'{np.median(rr) if rr else float("nan"):8.2f}       {np.median(ph):+5.0f}, {np.std(ph):3.0f}')

    print('\nmodel: 2 rad/s, complex c0 + c1|tau| per cell, leave-one-torque-out residual / |A|')
    for k in (1, 6):
        res = []
        for cell in ('pos_forward', 'pos_backdrive', 'neg_forward', 'neg_backdrive'):
            pts = []
            for p in [p for p in P if p['v'] == 2.0 and p['profile'] == 'C7_qdrag' and p['cell'] == cell]:
                q = by[(2.0, p['dir'], round(p['target']), 'C8_all_q')]
                pts.append((abs(p['target']), 0.5 * (p['cell_ph'][k] + q['cell_ph'][k])))
            pts.sort()
            for i in range(len(pts)):
                tr = [x for j, x in enumerate(pts) if j != i]
                cf = np.linalg.lstsq(np.array([[1, x[0]] for x in tr], dtype=complex),
                                     np.array([x[1] for x in tr]), rcond=None)[0]
                res.append(abs(pts[i][1] - (cf[0] + cf[1] * pts[i][0])) / abs(pts[i][1]))
            print(f'   {k}/rev {cell:14s} ' + '  '.join(f'{abs(z):.2f}@{np.degrees(np.angle(z)):+4.0f}deg'
                                                     for _, z in pts) + '   (4 / 8 / 12 Nm)')
        print(f'   {k}/rev median LOO residual {np.median(res):.2f}')

    print('\npath: 0.5 rad/s, (cell_C8 - cell_C7) / (-injected), median over holds')
    for k in ORDERS:
        G = []
        for p in [p for p in P if p['v'] == 0.5 and p['profile'] == 'C8_all_q']:
            R = p['inj_ph'][k]
            q = by.get((0.5, p['dir'], round(p['target']), 'C7_qdrag'))
            if q is not None and abs(R) > 0.03:
                G.append((p['cell_ph'][k] - q['cell_ph'][k]) / -R)
        G = np.array(G)
        md = np.median(G.real) + 1j * np.median(G.imag)
        f_hz = 26 * 0.5 * k / c.TWO_PI
        print(f'   order {k} ({f_hz:4.1f} Hz): gain {abs(md):.2f}, phase {np.degrees(np.angle(md)):+.0f} deg '
              f'(a {delay:.1f} ms cell lag alone would give {-360 * f_hz * delay / 1e3:+.0f} deg), n={len(G)}')

    phasor_figure(P, block)
    path = os.path.join(OUT, 'data', 'ripple_phasors_per_hold.csv')
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['flag', 'speed_abs', 'dir', 'profile', 'cell', 'target'] +
                   [f'{q}{k}' for k in ORDERS for q in ('amp', 'phase_rad_')])
        for p in P:
            w.writerow([p['flag'], p['v'], p['dir'], p['profile'], p['cell'], f'{p["target"]:.3f}'] +
                       [f'{x:.5g}' for k in ORDERS for x in (abs(p['cell_ph'][k]), np.angle(p['cell_ph'][k]))])
    print('  wrote', os.path.relpath(path, c.DYNO_DIR))


if __name__ == '__main__':
    main()
