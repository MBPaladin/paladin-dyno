"""The load cell's bimodal readings near zero: what they are, what they do to
every other reading, and how to correct them.

Findings this script documents:

  1. GAP. In every Skip log the raw (untared) cell almost never reads between
     about -0.2 and +0.15 Nm, i.e. within ~2 mV of 0 V at the ELM3004. The
     edges are sharp, so the dead zone sits late in the analog chain.
  2. SPLIT. It is not a local dead band but a split of the whole transfer
     function: y = x + (g/2) sign(x). Against the LOAD motor current (an
     independent torque measurement through the same shaft) every dwell with
     a positive raw reading sits g/2 high and every negative one g/2 low;
     g ~ 0.39 Nm. Closing the histogram gap gives the same g (~0.42 Nm).
  3. "BIMODAL" READINGS. A signal whose true value sits within ~0.5 Nm of 0 V
     has its noise excursions thrown across the gap, so single samples jump by
     ~0.4-0.8 Nm. At idle the raw signal sits right there, which is why the
     reading looks bimodal at rest and why tares taken at rest are unstable.
     In motion it only happens in the dwells whose raw level is near 0 V
     (here: zero current, output turning -), where the cogging ripple sweeps
     the signal in and out of the band -- which is why the jumps looked
     locked to motor angle.
  4. NOT SOFTWARE. The acquisition path (devices.ELM3004.process_txpdo ->
     fs_pos * raw / 7812500 + offset -> + tare -> float32 log) is linear with
     no sign-dependent step. Channel 2 of the same terminal, through the same
     code, crosses 0 V with no gap in older rig logs, while channel 1 (the load
     cell) shows the same hole back to 2026-08-27 on other gearboxes. The fault
     follows the load cell's chain: cell/amplifier/cable, or (less likely)
     channel 1's +/-5 V range.

Outputs (results/08_load_cell_signal/):
  Skip_load_cell_dead_zone.png
  Skip_load_cell_split_vs_load_motor.png
  Skip_load_cell_correction_example.png
  Skip_load_cell_channel_comparison.png  (needs the older rig logs in dyno/logs)
"""

import json
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import median_filter

import skip_common as c

SUB = '08_load_cell_signal'
SPIKE_NM = 0.4

# Older logs from the same rig and ELM3004 (adc_1), used to show the gap
# belongs to channel 1's signal chain rather than the terminal or the code.
# Skipped quietly when not present on this machine.
RIG_LOGS = {
    '2026-08-27 good gearbox, backlash': os.path.join(c.DYNO_DIR, 'logs', 'good_gearbox', 'backlash',
                                                      'backlash_gearbox_1Nm.hdf5'),
    '2026-09-18 Archimedes, velocity ramp': os.path.join(c.DYNO_DIR, 'logs', 'archimedes_bwd_velocity_ramp.hdf5'),
}


def raw_cell(path):
    with h5py.File(path, 'r') as f:
        return f['load_torque'][:].astype(float) - c.cell_tare_bias(f)


def channel_millivolts(path):
    """{sensor: untared signal in mV at the ELM input} for the ELM channels
    of one log, from its own resolved config (fs_pos/fs_v can differ by rig)."""
    out = {}
    with h5py.File(path, 'r') as f:
        cfg = json.loads(f.attrs['resolved_config'])
        tare = json.loads(f.attrs['tare']) if 'tare' in f.attrs else {}
        for dev in cfg.get('devices', {}).values():
            if not str(dev.get('class', '')).startswith('ELM'):
                continue
            for ch, cp in dev.get('params', {}).items():
                name = cp.get('name') if isinstance(cp, dict) else None
                if not name or name not in f:
                    continue
                bias = (tare.get(name) or {}).get('bias', 0.0)
                x = f[name][:].astype(float) - bias
                out[name] = (ch, 1e3 * x * float(cp['fs_v']) / float(cp['fs_pos']))
    return out


def spike_rate_by_level(raw, bins):
    med = median_filter(raw, 11, mode='nearest')
    r = raw - med
    k = np.digitize(med, bins)
    neg, pos, n = [], [], []
    for b in range(1, len(bins)):
        m = k == b
        n.append(int(m.sum()))
        neg.append(1e3 * np.mean(r[m] < -SPIKE_NM) if m.sum() > 1500 else np.nan)
        pos.append(1e3 * np.mean(r[m] > SPIKE_NM) if m.sum() > 1500 else np.nan)
    return np.array(neg), np.array(pos), np.array(n)


def split_from_load_motor(d):
    """Dwell means: cell_raw + LOAD torque = a + b*cell + c*sign(speed) [+ h*sign(cell)]."""
    L = np.array([np.mean(d['cell_raw'][c.dwell_slice(w)]) for w in d['dwells']])
    Tl = np.array([w['T_lm'] for w in d['dwells']])
    sv = np.array([w['dir'] for w in d['dwells']], float)
    D = L + Tl
    X0 = np.column_stack([np.ones_like(L), L, sv])
    X1 = np.column_stack([X0, np.sign(L)])
    c0, *_ = np.linalg.lstsq(X0, D, rcond=None)
    c1, *_ = np.linalg.lstsq(X1, D, rcond=None)
    return L, D, X0, X1, c0, c1


def split_from_histogram(raw):
    x = raw[np.abs(raw) < 1.5]
    best = (0.0, np.inf)
    for g in np.arange(0.0, 0.8, 0.01):
        h, _ = np.histogram(x - np.sign(x) * g / 2, bins=np.arange(-0.6, 0.601, 0.04))
        rough = np.sum(np.diff(np.log(h + 1.0), 2) ** 2)
        if rough < best[1]:
            best = (float(g), float(rough))
    return best[0]


def run(d):
    c.apply_style()
    logs = {'Skip cogging run': d['cell_raw']}
    for name, p in c.TORQUE_TEST_LOGS.items():
        if os.path.exists(p):
            logs[name] = raw_cell(p)

    L, D, X0, X1, c0, c1 = split_from_load_motor(d)
    g_lm = 2 * c1[3]
    g_hist = split_from_histogram(d['cell_raw'])

    # -- Figure 1: the gap, its closure, and spike rate vs level ---------------------
    bins = np.arange(-1.5, 1.5001, 0.025)
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    cols = ['#2a78d6', '#eb6834', '#1baf7a']
    for col, (name, raw) in zip(cols, logs.items()):
        h, e = np.histogram(raw, bins=bins, density=True)
        ax[0].plot(0.5 * (e[1:] + e[:-1]), h, color=col, lw=1.4, label=name)
    ax[0].axvspan(-0.2, 0.15, color='#fcebea', lw=0)
    ax[0].set_title('Raw cell output near 0 V: a gap in every log')
    ax[0].set_xlabel('Raw cell reading before tare (Nm; 10 mV per Nm at the ADC)')
    ax[0].set_ylabel('Density')
    ax[0].legend(loc='upper left')

    raw = d['cell_raw']
    h0, e = np.histogram(raw, bins=bins)
    h1, _ = np.histogram(c.split_correct(raw, g_lm), bins=bins)
    xc = 0.5 * (e[1:] + e[:-1])
    ax[1].plot(xc, h0, color=c.INK_3, lw=1.4, label='as logged')
    ax[1].plot(xc, h1, color='#1c5cab', lw=1.6, label=f'split-corrected (g = {g_lm:.2f} Nm)')
    ax[1].set_yscale('log')
    ax[1].set_title('Cogging run: correction closes the gap')
    ax[1].set_xlabel('Cell reading before tare (Nm)')
    ax[1].set_ylabel('Samples per 0.025 Nm bin')
    ax[1].legend(loc='upper left')

    lv = np.arange(-3.0, 3.0001, 0.25)
    neg, pos, n = spike_rate_by_level(raw, lv)
    xl = 0.5 * (lv[1:] + lv[:-1])
    ax[2].bar(xl - 0.06, neg, width=0.11, color='#a8302f', label='downward jumps')
    ax[2].bar(xl + 0.06, pos, width=0.11, color='#1c5cab', label='upward jumps')
    ax[2].set_yscale('symlog', linthresh=1)
    ax[2].set_title(f'Single-sample jumps > {SPIKE_NM} Nm vs local level')
    ax[2].set_xlabel('Local (11-sample median) raw level (Nm)')
    ax[2].set_ylabel('Jumps per 1000 samples')
    ax[2].legend(loc='upper right')
    fig.suptitle('Skip load cell: a split in the signal chain at 0 V')
    c.note(fig, 'Readings within ~2 mV of 0 V are missing; signals near there are thrown across the gap, which is the '
                'bimodal reading seen at rest. Away from 0 V the jump rate is the ordinary noise tail (~1-4 per 1000).',
           y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    c.save(fig, SUB, 'Skip_load_cell_dead_zone.png')

    # -- Figure 2: split seen against the LOAD motor current -------------------------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    r0 = D - X0 @ c0
    for a, r, title in ((ax[0], r0, 'Fit without a split term'),
                        (ax[1], D - X1 @ c1, f'Fit with a split term (g = {g_lm:.2f} Nm)')):
        for w, x, y in zip(d['dwells'], L, r):
            a.plot(x, y, c.DIR_MARKER[w['dir']], color=c.SPEED_COLORS[w['speed_abs']], ms=8, mec='white', mew=0.8)
        a.axhline(0, color=c.INK_3, lw=0.8)
        a.axvline(0, color=c.INK_3, lw=0.8)
        a.set_title(f'{title}: rms {np.sqrt(np.mean(r ** 2)):.3f} Nm')
        a.set_xlabel('Raw cell reading, dwell mean (Nm)')
    ax[0].set_ylabel('Cell + LOAD motor torque, fit residual (Nm)')
    handles = [Line2D([], [], color=col, marker='o', ls='', ms=8, label=f'{v:g} rad/s') for v, col in c.SPEED_COLORS.items()]
    handles += [Line2D([], [], color=c.INK_2, marker=c.DIR_MARKER[s], ls='', ms=8, label=c.DIR_LABEL[s]) for s in (-1, 1)]
    ax[0].legend(handles=handles, loc='lower right')
    fig.suptitle('Skip load cell vs LOAD motor current: every reading carries a sign-dependent offset')
    c.note(fig, 'Both measure the same shaft torque. Modelled per dwell as a + b*cell + c*sign(speed) (LOAD friction). '
                'Without a sign(cell) term the residual steps by ~0.4 Nm where the raw reading changes sign; with it, the '
                'residual falls 5x. b and c are the LOAD motor scale and drag.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    c.save(fig, SUB, 'Skip_load_cell_split_vs_load_motor.png')

    # -- Figure 3: correction and filtering on the worst dwell -------------------------------
    w = next(w for w in d['dwells'] if w['quadrant'] == 'zero' and w['dir'] < 0 and w['speed_abs'] == 0.05)
    sl = c.dwell_slice(w)
    seg = slice(sl.start + 30000, sl.start + 32000)
    t = np.arange(seg.stop - seg.start) / c.FS_HZ
    raw_w = d['cell_raw'][sl]
    cor_w = c.split_correct(raw_w, g_lm)
    med_w = median_filter(raw_w, 5, mode='nearest')
    ma = np.convolve(cor_w, np.ones(25) / 25, mode='same')
    j = slice(30000, 32000)
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={'width_ratios': [2.2, 1]})
    ax[0].plot(t, raw_w[j], color=c.INK_3, lw=0.7, label='as logged (raw, untared)')
    ax[0].plot(t, cor_w[j], color='#3987e5', lw=0.7, label='split-corrected')
    ax[0].plot(t, med_w[j], color='#eb6834', lw=1.0, label='5-sample median of raw (hides jumps, keeps bias)')
    ax[0].plot(t, ma[j], color=c.INK, lw=1.6, label='split-corrected, 25 ms mean')
    ax[0].axhspan(-0.2, 0.15, color='#fcebea', lw=0)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Cell reading before tare (Nm)')
    ax[0].set_title('Zero current, 0.05 rad/s, output turning - (raw level ~+0.6 Nm)')
    ax[0].legend(loc='upper right', ncol=2)
    hb = np.arange(-1.0, 2.0, 0.025)
    ax[1].hist(raw_w, bins=hb, color=c.INK_3, alpha=0.7, label='as logged')
    ax[1].hist(cor_w, bins=hb, histtype='step', color='#1c5cab', lw=1.6, label='split-corrected')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Cell reading before tare (Nm)')
    ax[1].set_title('Whole dwell: bimodal -> unimodal')
    ax[1].legend(loc='upper right')
    fig.suptitle('Skip load cell: correcting the split, then filtering')
    c.note(fig, f'Dwell means (raw Nm): as logged {raw_w.mean():+.3f}, 5-sample median {med_w.mean():+.3f}, '
                f'split-corrected {cor_w.mean():+.3f}. Correct first, then use a mean-type low-pass; a median '
                'filter alone removes the visible jumps but leaves the 0.2 Nm offset.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    c.save(fig, SUB, 'Skip_load_cell_correction_example.png')

    # -- Figure 4: same terminal, other channel, older logs -------------------------
    present = {k: p for k, p in RIG_LOGS.items() if os.path.exists(p)}
    gap_depth = {}
    if present:
        mv_bins = np.arange(-12.0, 12.0001, 0.25)
        xc = 0.5 * (mv_bins[1:] + mv_bins[:-1])
        fig, ax = plt.subplots(1, len(present) + 1, figsize=(5.4 * (len(present) + 1), 4.8), sharex=True)
        panels = [('2026-10-19 Skip cogging run', c.COGGING_LOG)] + list(present.items())
        for a, (title, path) in zip(np.atleast_1d(ax), panels):
            chans = channel_millivolts(path)
            for name, col in (('load_torque', '#a8302f'), ('input_torque', '#1c5cab')):
                if name not in chans:
                    continue
                ch, mv = chans[name]
                near = mv[np.abs(mv) < 12.0]
                both_sides = min(np.sum((mv > -8) & (mv < -4)), np.sum((mv > 4) & (mv < 8)))
                if both_sides < 500:
                    a.text(0.5, 0.9 if name == 'load_torque' else 0.82,
                           f'{name} ({ch}) never crosses 0 V here', transform=a.transAxes, ha='center',
                           fontsize=8.5, color=col)
                    continue
                h, _ = np.histogram(near, bins=mv_bins)
                h = h / h.mean()
                a.plot(xc, h, color=col, lw=1.6, label=f'{name} ({ch})')
                inside = np.abs(xc - 0.25) < 1.0
                outside = (np.abs(xc) > 4) & (np.abs(xc) < 8)
                gap_depth[f'{title} | {name}'] = float(h[inside].mean() / h[outside].mean())
            a.axvspan(-1.5, 2.0, color='#fcebea', lw=0, zorder=0)
            a.set_title(title, fontsize=10)
            a.set_xlabel('Signal at the ELM3004 input (mV, untared)')
            a.set_ylim(bottom=0)
            a.legend(loc='lower right')
        np.atleast_1d(ax)[0].set_ylabel('Density, normalised to the +-12 mV mean')
        fig.suptitle('Same ELM3004, same code: only the load cell channel has a hole at 0 V')
        c.note(fig, 'adc_1 ch1 = load cell (+-5 V range), ch2 = input cell (+-10 V range); both go through '
                    'ELM3004.process_txpdo -> fs_pos*raw/7812500 + offset -> tare. The hole sits at the same '
                    'millivolts on every rig and date, regardless of tare or DUT.', y=0.0)
        fig.tight_layout(rect=(0, 0.05, 1, 0.93))
        c.save(fig, SUB, 'Skip_load_cell_channel_comparison.png')

    med_c = median_filter(cor_w, 11, mode='nearest')
    c.record('load_cell', {
        'split_from_load_motor_Nm': float(g_lm), 'split_from_histogram_Nm': float(g_hist),
        'split_mV_at_adc': float(g_lm * c.CELL_V_PER_NM * 1e3),
        'residual_rms_without_split_Nm': float(np.sqrt(np.mean((D - X0 @ c0) ** 2))),
        'residual_rms_with_split_Nm': float(np.sqrt(np.mean((D - X1 @ c1) ** 2))),
        'load_motor_scale_b': float(c1[1]), 'load_side_drag_c_Nm': float(c1[2]),
        'tare_bias_this_run_Nm': float(d['tare_bias']),
        'gap_depth_density_ratio_by_log_and_channel': gap_depth,
        'example_dwell_downward_jumps_per_s_raw': float(np.sum((raw_w - median_filter(raw_w, 11, mode='nearest')) < -SPIKE_NM) / (len(raw_w) / c.FS_HZ)),
        'example_dwell_downward_jumps_per_s_corrected': float(np.sum((cor_w - med_c) < -SPIKE_NM) / (len(cor_w) / c.FS_HZ)),
    })


if __name__ == '__main__':
    run(c.load())
