"""Feedforward checkout: does LOAD's torque feedforward reach the drive, with
the right sign and size? Also confirms the DUT/cell sign frames.

Written for tests/ui_generated_tests/skip_ff_check.yaml (LOAD at +/-0.05 rad/s,
DUT torque steps of 0.1 Nm at 5 Nm/s), run once at feedforward_ratio 0 and once
at 0.8. Works on any log with DUT torque ramps under a steady LOAD velocity,
including the 2026-10-19 cogging run.

  * FF SLOPE. The velocity loop needs a speed error before it reacts, so the
    LOAD current that tracks the DUT command right after a ramp starts is
    mostly the feedforward. It shows up FF_LAG_MS late (drive latency), and the
    loop joins in within a few ms more, so this is a strong / weak / absent
    test, not a precise gain: no-FF runs read about 0, working runs 0.8-1.5 of
    what was sent. Expected slope: ratio x DUT gear_ratio / LOAD kt through
    LOAD's flip_torque_sign (send_command negates the offset; load_current is
    logged in the drive's own frame, unflipped), or from load_torque_ff.
  * WHAT LOAD NEEDS. The settled LOAD current change per Nm of DUT command,
    whichever loop supplied it. The feedforward cancels if it has the same sign.
  * SIGN FRAMES. dut_current per Nm commanded, cell per Nm commanded, and which
    sign of tc*v is forward (DUT drives: the smaller cell change per Nm).
  * STICK-SLIP. Share of each steady plateau the motor is stuck, and the cell
    torque swing (p1-p99) and peak, by quadrant. Judge gains on the swing and
    peak: stuck time can fall while the swing grows.
  * RINGING. LOAD current rms in 150-460 Hz over the steady dwells: the
    warning sign when raising LOAD's velocity gains.
  * DISTURBANCE. Median peak LOAD velocity error after a step. Compare the
    ratio-0 and ratio-0.8 runs: a working, correctly signed feedforward makes
    it smaller, a wrongly signed one roughly doubles it.

From 2026-10-21 the log records the effective feedforward (resolved_config
'feedforward') and what LOAD was sent (load_torque_ff), so no --ratio is needed
and quadrant mode is judged against what was actually sent. For older logs pass
--ratio per log; without either the script assumes the code default 0.8 and
says so.

    .venv/bin/python dyno/src/post_processors/skip/check_ff.py LOG0.hdf5 LOG08.hdf5 --ratio 0 0.8
"""

import argparse
import json
import os

import h5py
import numpy as np
from scipy.signal import welch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FS_HZ = 1000.0
CODE_DEFAULT_RATIO = 0.8       # dyno_controller: dyno_params.get('feedforward_ratio', 0.8)
EARLY_MS = 10                  # window length for the feedforward slope
FF_LAG_MS = 6                  # LOAD current trails the sent feedforward by ~6 ms (CSV, KBUSFF 1);
                               # measured 2026-10-21: at this lag the no-feedforward run reads
                               # 0.05 of 'configured' and every working run 0.8-1.5. Without the
                               # shift the window sat in the lag and read 0.1-0.7 on runs that
                               # were fine.
PRE_MS = (60, 3)               # baseline window before a ramp starts
GROUP_GAP = 100                # samples: command changes closer than this are one ramp
DISTURB_MS = 500
SETTLED_S = 2.0                # last part of each plateau used for settled values
PLOT_MS = (-20, 300)
STUCK_FRAC = 0.2               # motor under this x the commanded speed counts as stuck
STICK_SKIP_S = 1.0             # ignore the start of each plateau (step transient)
RING_BAND_HZ = (150, 460)      # LOAD-side line ~283-310 Hz and cell/flange mode ~400-440 Hz


def flags(f):
    rc = json.loads(f.attrs['resolved_config'])
    dev = rc['devices']
    g = lambda name, key, default=None: dev[name]['params'].get(key, default)
    return {
        'dut_flip_torque': bool(g('DUT', 'flip_torque_sign', False)),
        'dut_flip_dir': bool(g('DUT', 'flip_direction_sign', False)),
        'load_flip_torque': bool(g('LOAD', 'flip_torque_sign', False)),
        'load_flip_dir': bool(g('LOAD', 'flip_direction_sign', False)),
        'gear_ratio': float(g('DUT', 'gear_ratio')),
        'kt_load': float(dev['LOAD']['params']['motor_params']['kt']),
        'kt_dut': float(dev['DUT']['params']['motor_params']['kt']),
        'timestamp': rc.get('timestamp'),
        # Logs from 2026-10-21 on record the effective feedforward (Master).
        'feedforward': rc.get('feedforward'),
    }


def ramps(tc, vc):
    """(start, end) sample index of each DUT command ramp under a steady,
    nonzero LOAD velocity command."""
    ch = np.flatnonzero(np.diff(tc) != 0) + 1
    if len(ch) == 0:
        return []
    groups, s, prev = [], ch[0], ch[0]
    for i in ch[1:]:
        if i - prev > GROUP_GAP:
            groups.append((s, prev))
            s = i
        prev = i
    groups.append((s, prev))
    out = []
    for s, e in groups:
        lo, hi = s - PRE_MS[0], e + DISTURB_MS
        if lo < 0 or hi >= len(tc):
            continue
        if np.ptp(vc[lo:hi]) > 1e-9 or abs(vc[s]) < 1e-6:
            continue        # LOAD speed changed nearby, or LOAD stopped
        out.append((int(s), int(e)))
    return out


def plateaus(tc, vc, t_starts):
    """Settled (tc, v) plateaus between ramps: (i0, i1, tc, v)."""
    bounds = np.r_[0, np.flatnonzero((np.diff(tc) != 0) | (np.diff(vc) != 0)) + 1, len(tc)]
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < (SETTLED_S + 0.5) * FS_HZ or abs(vc[a]) < 1e-6:
            continue
        out.append((int(b - SETTLED_S * FS_HZ), int(b), float(tc[a]), float(vc[a])))
    return out


def r_forward_sign(fl):
    """+1 if forward (DUT drives) is tc*v > 0 in this log's frames, else -1.

    Physically the DUT pushes with the motion when its command, through its
    own torque and direction flags, has the sign of the output velocity. With
    DUT flip_torque_sign == flip_direction_sign (2026-10-21 on) that is
    tc*v > 0; the 2026-10-19 cogging log (False, True) had tc*v < 0. LOAD
    reports velocity unflipped on this rig.
    """
    s = 1 if fl['dut_flip_torque'] == fl['dut_flip_dir'] else -1
    return -s if fl['load_flip_dir'] else s


def analyse(path, ratio):
    with h5py.File(path, 'r') as f:
        fl = flags(f)
        tc = np.nan_to_num(f['dut_torque_command'][:].astype(float))
        vc = np.nan_to_num(f['load_velocity_command'][:].astype(float))
        il = f['load_current'][:].astype(float)
        id_ = f['dut_current'][:].astype(float)
        lv = f['load_velocity'][:].astype(float)
        cell = f['load_torque'][:].astype(float)
        th_m = f['dut_output_position'][:].astype(float)   # motor side, despite the name
        ff_log = f['load_torque_ff'][:].astype(float) if 'load_torque_ff' in f else None
    r = {'log': os.path.basename(path), **fl}
    cfg = fl.get('feedforward')
    if ratio is not None:                         # --ratio wins: older logs
        r.update(ff_mode='fixed', ratio=ratio, ratio_source='--ratio')
    elif cfg:
        r.update(ff_mode=cfg['mode'], ratio=(0.0 if cfg['mode'] == 'off' else cfg.get('ratio')),
                 ratio_source='log')
    else:
        r.update(ff_mode='fixed', ratio=CODE_DEFAULT_RATIO, ratio_source='assumed')
    r['ratio_assumed'] = r['ratio_source'] == 'assumed'
    r['ff_on'] = r['ff_mode'] == 'quadrant' or bool(r['ratio'])
    r['ff_label'] = ('quadrant ' + '/'.join(f"{cfg[k]:g}" for k in ('forward_ratio', 'backdrive_ratio', 'drag_nm'))
                     if r['ff_mode'] == 'quadrant' else f"ratio {r['ratio']:g}")

    # Configured feedforward, as LOAD's logged current: ratio * tc * G, negated
    # by send_command when LOAD has flip_torque_sign. With load_torque_ff in the
    # log (2026-10-21 on) the expected slope comes from what was actually sent,
    # which is the only way to judge quadrant mode.
    load_sign = -1.0 if fl['load_flip_torque'] else 1.0
    r['ff_expected_A_per_Nm'] = (load_sign * r['ratio'] * fl['gear_ratio'] / fl['kt_load']
                                 if r['ff_mode'] == 'fixed' else np.nan)
    ff_A = None if ff_log is None else load_sign * np.nan_to_num(ff_log) / fl['kt_load']

    rr = ramps(tc, vc)
    r['n_ramps'] = len(rr)
    if not rr:
        r['error'] = 'no DUT torque ramps under a steady LOAD velocity'
        return r

    # --- early-window feedforward slope -------------------------------------------
    num = den = num_ff = 0.0
    curves, curves_tc, curves_ff, disturb, early_verr = [], [], [], [], []
    w = np.arange(PLOT_MS[0], PLOT_MS[1])
    for s, e in rr:
        base_i = np.median(il[s - PRE_MS[0]:s - PRE_MS[1]])
        base_tc = tc[s - 1]
        k = np.arange(s, s + EARLY_MS)          # command samples
        km = k + FF_LAG_MS                       # the LOAD current they show up in
        dI, dT = il[km] - base_i, tc[k] - base_tc
        num += np.sum(dI * dT)
        den += np.sum(dT * dT)
        if ff_A is not None:
            num_ff += np.sum((ff_A[k] - ff_A[s - 1]) * dT)
        step = tc[e] - base_tc
        if abs(step) > 1e-6 and s + PLOT_MS[1] < len(il):
            curves.append((il[s + w] - base_i) / step)
            curves_tc.append((tc[s + w] - base_tc) / step)
            if ff_A is not None:
                curves_ff.append((ff_A[s + w] - ff_A[s - 1]) / step)
        verr = lv - vc
        disturb.append(np.nanmax(np.abs(verr[s:s + DISTURB_MS])))
        early_verr.append(np.nanmax(np.abs(verr[s:s + EARLY_MS])))
    r['ff_measured_A_per_Nm'] = num / den if den > 0 else np.nan
    if ff_A is not None and den > 0:
        r['ff_expected_A_per_Nm'] = num_ff / den
        r['ff_expected_from'] = 'load_torque_ff'
    r['ff_fraction'] = (r['ff_measured_A_per_Nm'] / r['ff_expected_A_per_Nm']
                        if r['ff_expected_A_per_Nm'] != 0 else np.nan)
    r['early_dtc'] = float(np.sqrt(den / (len(rr) * EARLY_MS)))
    r['disturb_rad_s'] = float(np.median(disturb))
    r['early_verr_rad_s'] = float(np.median(early_verr))
    r['curve'] = ((w, np.mean(curves, 0), np.mean(curves_tc, 0),
                   np.mean(curves_ff, 0) if curves_ff else None) if curves else None)

    # --- settled values: what LOAD needs, and the sign frames ---------------------
    pl = plateaus(tc, vc, [s for s, _ in rr])
    rows = np.array([(tc_, v_, np.median(il[a:b]), np.median(id_[a:b]), np.median(cell[a:b]))
                     for a, b, tc_, v_ in pl]) if pl else np.empty((0, 5))
    r['plateaus'] = len(rows)

    # --- stick-slip: share of each plateau the motor is stuck, by quadrant ----------
    # Speed from a 25 ms centred difference of the motor position (dut_velocity
    # is coarse). Stuck = under STUCK_FRAC of the speed the output command
    # implies. First STICK_SKIP_S of each plateau dropped as the step transient.
    h = 12
    w_m = np.full_like(th_m, np.nan)
    w_m[h:-h] = (th_m[2 * h:] - th_m[:-2 * h]) * FS_HZ / (2 * h)
    stuck, swing, peak, ring = {}, {}, {}, []
    bounds = np.r_[0, np.flatnonzero((np.diff(tc) != 0) | (np.diff(vc) != 0)) + 1, len(tc)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        a2 = a + int(STICK_SKIP_S * FS_HZ)
        if b - a2 < 1000 or abs(vc[a]) < 1e-6:
            continue
        nom = abs(fl['gear_ratio'] * vc[a])
        q = 'zero' if abs(tc[a]) < 1e-6 else ('forward' if r_forward_sign(fl) * tc[a] * vc[a] > 0 else 'backdrive')
        stuck.setdefault(q, []).append(float(np.nanmean(np.abs(w_m[a2:b]) < STUCK_FRAC * nom)))
        seg = il[a2:b]
        if np.all(np.isfinite(seg)) and len(seg) >= 1024:
            fr, p = welch(seg - seg.mean(), fs=FS_HZ, nperseg=1024)
            band = (fr >= RING_BAND_HZ[0]) & (fr < RING_BAND_HZ[1])
            ring.append(float(np.sqrt(np.sum(p[band]) * (fr[1] - fr[0]))))
        c = cell[a2:b]
        c = c[np.isfinite(c)]
        if len(c):
            swing.setdefault(q, []).append(float(np.percentile(c, 99) - np.percentile(c, 1)))
            peak.setdefault(q, []).append(float(np.max(np.abs(c))))
    r['stuck'] = {q: float(np.mean(v)) for q, v in stuck.items()}
    # Stuck time alone misleads: a stiffer LOAD loop can cut it at low torque
    # and still double the torque swing at high torque (2026-10-21, 15 A).
    r['swing'] = {q: float(np.median(v)) for q, v in swing.items()}
    r['peak'] = {q: float(np.max(v)) for q, v in peak.items()}
    r['ring_A'] = float(np.median(ring)) if ring else np.nan
    nz = np.abs(rows[:, 0]) > 1e-6 if len(rows) else np.array([], bool)
    if nz.sum() >= 2:
        r['dut_A_per_Nm'] = float(np.median(rows[nz, 3] / rows[nz, 0]))
    # Per speed, change from the 0 Nm plateau, split by the sign of tc*v.
    need, dcell = [], {+1: [], -1: []}
    for v in np.unique(rows[:, 1]) if len(rows) else []:
        at_v = rows[rows[:, 1] == v]
        zero = at_v[np.abs(at_v[:, 0]) < 1e-6]
        if not len(zero):
            continue
        z_il, z_cell = np.median(zero[:, 2]), np.median(zero[:, 4])
        for row in at_v[np.abs(at_v[:, 0]) > 1e-6]:
            need.append((row[2] - z_il) / row[0])
            dcell[int(np.sign(row[0] * v))].append((row[4] - z_cell) / row[0])
    if need:
        r['need_A_per_Nm'] = float(np.median(need))
    if dcell[+1] and dcell[-1]:
        r['cell_Nm_per_Nm'] = {k: float(np.median(v)) for k, v in dcell.items()}
        # Forward (DUT drives) transmits less torque per Nm than backdrive.
        r['forward_is'] = ('tc*v > 0' if abs(r['cell_Nm_per_Nm'][+1]) < abs(r['cell_Nm_per_Nm'][-1])
                           else 'tc*v < 0')
    return r


def report(r):
    print(f"\n== {r['log']}  (config {r['timestamp']})")
    print(f"   flags     DUT flip_torque {r['dut_flip_torque']}, flip_dir {r['dut_flip_dir']}, "
          f"G {r['gear_ratio']:g}; LOAD flip_torque {r['load_flip_torque']}, flip_dir {r['load_flip_dir']}")
    if 'error' in r:
        print(f"   ERROR     {r['error']}")
        return
    note = {'assumed': '  (ASSUMED: code default, not recorded in the log -- pass --ratio)',
            '--ratio': '  (from --ratio)', 'log': '  (from the log)'}[r['ratio_source']]
    print(f"   ff        {r['ff_mode']}, {r['ff_label']}{note}")
    print(f"   ramps     {r['n_ramps']} DUT ramps, rms command change in the first {EARLY_MS} ms "
          f"{r['early_dtc']:.4f} Nm; velocity error there {r['early_verr_rad_s']:.4f} rad/s "
          f"(ripple only: compare with the disturbance below)")
    exp, meas = r['ff_expected_A_per_Nm'], r['ff_measured_A_per_Nm']
    src = 'sent (load_torque_ff)' if r.get('ff_expected_from') else 'configured'
    print(f"   FF slope  measured {meas:+.3f} A/Nm vs {src} {exp:+.3f} A/Nm in LOAD's logged current")
    if not r['ff_on']:
        print(f"             ratio 0: this is the no-FF baseline; anything beyond noise here is not feedforward")
    else:
        frac = r['ff_fraction']
        if abs(frac) < 0.25:
            v = 'NOT REACHING THE DRIVE'
        elif frac < 0:
            v = 'REACHING, OPPOSITE SIGN TO CONFIGURED (a drive-side sign convention we do not model)'
        elif 0.5 <= frac <= 2.0:
            # The velocity loop joins in a few ms after the feedforward lands,
            # so this window cannot resolve the gain better than about 2x.
            v = 'REACHING (this window resolves presence and sign, not the gain to better than ~2x)'
        else:
            v = f'REACHING, but {frac:.2f}x what was sent: outside the 0.5-2x a working run reads'
        print(f"             {frac:+.2f} of configured -> {v}")
    if 'need_A_per_Nm' in r:
        need = r['need_A_per_Nm']
        print(f"   LOAD needs {need:+.3f} A/Nm settled (whichever loop supplied it)")
        if r['ff_on'] and np.isfinite(exp):
            ok = np.sign(need) == np.sign(exp)
            print(f"             configured FF {'CANCELS the DUT torque' if ok else 'ADDS to the DUT torque: WRONG SIGN'}")
    if 'dut_A_per_Nm' in r:
        print(f"   DUT       {r['dut_A_per_Nm']:+.1f} A per Nm commanded (1/kt = {1 / r['kt_dut']:.0f}; "
              f"negative is expected with flip_torque_sign True)")
    if 'cell_Nm_per_Nm' in r:
        c = r['cell_Nm_per_Nm']
        print(f"   cell      {c[+1]:+.1f} Nm/Nm at tc*v>0, {c[-1]:+.1f} Nm/Nm at tc*v<0 "
              f"-> forward (DUT drives) is {r['forward_is']}; cell sign vs command "
              f"{'ALIGNED' if np.median([c[+1], c[-1]]) > 0 else 'OPPOSITE'}")
    print(f"   disturb   median peak |LOAD velocity error| after a step {r['disturb_rad_s']:.4f} rad/s")
    if r.get('stuck'):
        print('   stuck     ' + ', '.join(f"{q} {100 * v:.1f}%" for q, v in sorted(r['stuck'].items()))
              + '  (share of time the motor is stuck)')
    if r.get('swing'):
        print('   swing     ' + ', '.join(f"{q} {v:.2f} Nm" for q, v in sorted(r['swing'].items()))
              + '  (median p1-p99 cell torque per dwell: the stick-slip size)')
        print('   peak      ' + ', '.join(f"{q} {v:.1f} Nm" for q, v in sorted(r['peak'].items()))
              + '  (largest |cell| in any steady dwell: headroom to the torque cap)')
    if np.isfinite(r.get('ring_A', np.nan)):
        print(f"   ringing   LOAD current {RING_BAND_HZ[0]}-{RING_BAND_HZ[1]} Hz rms {1000 * r['ring_A']:.1f} mA "
              f"(median over steady dwells; 11-12 mA at KP 1-2 on skip_ff_check_12A. Rising = the loop is finding a mode)")


def plot(results, out_png):
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, r in enumerate(results):
        if not r.get('curve'):
            continue
        w, il, tcn, ffc = r['curve']
        col = f'C{k}'
        ax.plot(w, il, color=col, label=f"{r['log']}: LOAD current ({r['ff_label']})")
        if ffc is not None and r['ff_on']:
            ax.plot(w, ffc, '--', color=col, lw=1, label=f"FF sent ({r['ff_label']})")
        elif r['ff_on']:
            ax.plot(w, r['ff_expected_A_per_Nm'] * tcn, '--', color=col, lw=1,
                    label=f"configured FF alone ({r['ff_label']})")
    ax.axvspan(FF_LAG_MS, FF_LAG_MS + EARLY_MS, color='0.85',
               label=f'{FF_LAG_MS}-{FF_LAG_MS + EARLY_MS} ms: FF slope window')
    ax.axhline(0, color='0.5', lw=0.5)
    ax.set_xlabel('ms from start of DUT ramp')
    ax.set_ylabel('LOAD current change per Nm of DUT step [A/Nm, logged frame]')
    ax.set_title('LOAD current through a DUT torque step (mean over steps)', fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--ratio', type=float, nargs='+',
                    help='feedforward_ratio each log ran with, in order (only for logs '
                         'from before 2026-10-21, which do not record it)')
    ap.add_argument('--png', help='figure path (default: next to the first log)')
    args = ap.parse_args()
    ratios = args.ratio or []
    if ratios and len(ratios) != len(args.logs):
        ap.error('give one --ratio per log')
    results = [analyse(p, ratios[i] if ratios else None) for i, p in enumerate(args.logs)]
    for r in results:
        report(r)
    png = args.png or os.path.splitext(args.logs[0])[0] + '_ff_check.png'
    plot(results, png)
    print(f'\nfigure: {png}')


if __name__ == '__main__':
    main()
