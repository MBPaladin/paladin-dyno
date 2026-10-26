"""Stick-slip: where in the grid it happens, how big it is, and whether the
places the motor sticks are locked to an angle.

Motor speed comes from the motor encoder (25 ms centred difference) and is
normalised by the commanded speed (26 x output speed command), signed along the
motion, so 1 = tracking, 0 = stuck, >2 = a slip burst.

  stuck fraction  share of samples with normalised speed < 0.2
  slip event      normalised speed rising through 2.0 (re-armed below 1.0)
  breakaway drop  torque LOAD was pushing just before a slip minus the lowest
                  value just after it -- the torque released by one slip

Stick locations: each stuck interval (>= 30 ms) contributes its mid angle.
If sticking were indifferent to angle, those angles would be uniform in every
phase; the phase-locking value |mean(exp(i k theta))| per order tells which
order (if any) decides where the motor sticks.

Outputs (results/03_stick_slip/):
  Skip_stick_slip_examples.png     10 s of forward, zero and backdrive at one speed
  Skip_stick_slip_map.png          stuck %, slip rate, breakaway drop over the grid
  Skip_stick_location_vs_angle.png stick-angle histograms and phase locking by order
  Skip_stick_slip_per_dwell.csv
"""

import numpy as np
import matplotlib.pyplot as plt

import skip_common as c

SUB = '03_stick_slip'
SLIP_ON, SLIP_REARM = 2.0, 1.0
MIN_STICK_S = 0.03
MOTOR_ORDERS = (1, 2, 3, 6, 21, 27, 36, 42)
OUTPUT_ORDERS = (1, 2, 3, 4, 6, 25, 26, 27)


def runs(mask):
    """(start, end) index pairs of True runs."""
    dm = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(dm == 1), np.flatnonzero(dm == -1)


def dwell_stick_slip(d, w):
    sl = c.dwell_slice(w)
    r = d['w_m'][sl] * w['dir'] / w['motor_speed_nom']
    P = -w['dir'] * d['T_cell'][sl]
    dur = (w['i1'] - w['i0']) / c.FS_HZ
    # Slip events: rising through SLIP_ON, counted again only once the speed
    # has fallen back below SLIP_REARM (so one burst with a ragged top counts once).
    n = len(r)
    above = r >= SLIP_ON
    on_idx = np.flatnonzero(above[1:] & ~above[:-1]) + 1
    rearm_idx = np.flatnonzero(r < SLIP_REARM)
    events = []
    for k in on_idx:
        if events:
            j = np.searchsorted(rearm_idx, events[-1])
            if not (j < len(rearm_idx) and rearm_idx[j] < k):
                continue
        events.append(k)
    drops, angles = [], []
    th = d['th_m'][sl]
    for k in events:
        j0 = max(0, k - 300)
        back = np.flatnonzero(r[j0:k] < SLIP_REARM)
        start = j0 + back[-1] if len(back) else j0
        fwd = np.flatnonzero(r[k:min(n, k + 2000)] < SLIP_REARM)
        end = k + fwd[0] if len(fwd) else min(n - 1, k + 2000)
        pre = P[max(0, start - 100):k + 1]
        post = P[k:min(n, end + 100)]
        if len(pre) and len(post):
            drops.append(float(np.max(pre) - np.min(post)))
        angles.append(float(abs(th[end] - th[start])))
    stuck = r < c.STUCK_FRAC
    s0, s1 = runs(stuck)
    keep = (s1 - s0) >= int(MIN_STICK_S * c.FS_HZ)
    stick_mid = ((s0[keep] + s1[keep]) // 2) + w['i0']
    return {
        'stuck_frac': float(stuck.mean()),
        'slip_rate_hz': len(events) / dur,
        'n_slips': len(events),
        'drop_median_Nm': float(np.median(drops)) if drops else np.nan,
        'slip_angle_median_rad': float(np.median(angles)) if angles else np.nan,
        'P_std_Nm': float(np.std(P)),
        'stick_mid_idx': stick_mid,
        'stick_dur_s': (s1[keep] - s0[keep]) / c.FS_HZ,
    }


def heat(ax, grid, row_labels, col_labels, title, fmt, cmap='Blues', vmax=None):
    im = ax.imshow(grid, cmap=cmap, aspect='auto', vmin=0, vmax=vmax if vmax else np.nanmax(grid))
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.grid(False)
    ax.set_title(title)
    hi = np.nanmax(grid) if vmax is None else vmax
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha='center', va='center', fontsize=8.5,
                        color='white' if v > 0.55 * hi else c.INK)
    return im


def run(d):
    c.apply_style()
    dw = d['dwells']
    stats = {w['n']: dwell_stick_slip(d, w) for w in dw}

    # -- Example traces -------------------------------------------------------------
    v_ex = 0.05
    im = c.i_max(d)
    picks = [(f'DUT drives, {im:g} A (forward)', im), ('Zero current', 0), (f'LOAD backdrives DUT, {im:g} A', -im)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 6.4), sharex=True)
    for j, (title, ia) in enumerate(picks):
        w = next(w for w in dw if w['speed_abs'] == v_ex and w['dir'] < 0 and w['i_along_nom'] == ia)
        i0 = w['i0'] + 40000
        sl = slice(i0, i0 + 10000)
        tt = (np.arange(sl.stop - sl.start)) / c.FS_HZ
        P = -w['dir'] * d['T_cell'][sl]
        r = d['w_m'][sl] * w['dir'] / w['motor_speed_nom']
        ax[0, j].plot(tt, P, color='#1c5cab', lw=0.6)
        ax[0, j].set_title(title)
        ax[1, j].plot(tt, r, color='#c14d1f', lw=0.6)
        ax[1, j].axhline(1, color=c.INK_3, ls=':', lw=0.8)
        ax[1, j].axhline(0, color=c.INK_3, lw=0.8)
        ax[1, j].set_ylim(-1, 8)
        ax[1, j].set_xlabel('Time (s)')
        st = stats[w['n']]
        ax[1, j].text(0.98, 0.95, f'stuck {100 * st["stuck_frac"]:.0f}%  |  {st["slip_rate_hz"]:.1f} slips/s',
                      transform=ax[1, j].transAxes, ha='right', va='top', fontsize=9, color=c.INK_2)
        lo, hi = np.percentile(P, [0.5, 99.5])
        mid = 0.5 * (lo + hi)
        ax[0, j].set_ylim(mid - 4.5, mid + 4.5)
    ax[0, 0].set_ylabel('LOAD torque along motion (Nm)')
    ax[1, 0].set_ylabel('Motor speed / commanded')
    fig.suptitle(f'Skip stick-slip at {v_ex} rad/s output (10 s windows, same 9 Nm torque span per panel)')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    c.save(fig, SUB, 'Skip_stick_slip_examples.png')

    # -- Grid maps ------------------------------------------------------------------
    sp_rows = sorted({w['speed'] for w in dw})
    cols = c.currents_along(d)
    G = {k: np.full((len(sp_rows), len(cols)), np.nan) for k in ('stuck', 'rate', 'drop', 'std')}
    for w in dw:
        i = sp_rows.index(w['speed'])
        j = cols.index(w['i_along_nom'])
        st = stats[w['n']]
        G['stuck'][i, j] = 100 * st['stuck_frac']
        G['rate'][i, j] = st['slip_rate_hz']
        G['drop'][i, j] = st['drop_median_Nm'] if st['n_slips'] >= 20 else np.nan
        G['std'][i, j] = st['P_std_Nm']
    rl = [f'{v:+g}' for v in sp_rows]
    cl = [f'{i:+g}' for i in cols]
    fig, ax = plt.subplots(1, 3, figsize=(16, 2.6 + 0.37 * len(sp_rows)))
    heat(ax[0], G['stuck'], rl, cl, 'Time stuck (%)', '{:.0f}')
    heat(ax[1], G['rate'], rl, cl, 'Slip bursts per second', '{:.1f}')
    heat(ax[2], G['std'], rl, cl, 'Torque std within dwell (Nm)', '{:.2f}', cmap='Oranges')
    for a in ax:
        a.set_xlabel('DUT current along motion (A)  [- = backdrive, + = forward]')
    ax[0].set_ylabel('Output speed command (rad/s)')
    fig.suptitle('Skip stick-slip across the grid')
    c.note(fig, 'Stuck = motor below 0.2 x commanded speed (25 ms window); slip burst = rising through 2 x. '
                'The std includes all ripple, not only stick-slip.', y=0.0)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    c.save(fig, SUB, 'Skip_stick_slip_map.png')

    # -- Stick location vs angle --------------------------------------------------------
    heavy = [w for w in dw if w['i_along_nom'] <= -10]
    idx = np.concatenate([stats[w['n']]['stick_mid_idx'] for w in heavy]).astype(int)
    rec = _record(d, dw, stats)
    if len(idx) < 20:
        print(f'  only {len(idx)} stuck intervals in heavy backdrive: no stick-location figure')
        c.record('stick_slip', rec)
        _per_dwell_csv(dw, stats)
        return stats
    th_m = d['th_m'][idx]
    th_o = d['th_o'][idx]
    n = len(idx)
    thr = np.sqrt(-np.log(0.01) / n)    # Rayleigh 1% level for phase-locking value
    plv_m = {k: float(np.abs(np.mean(np.exp(1j * k * th_m)))) for k in MOTOR_ORDERS}
    plv_o = {k: float(np.abs(np.mean(np.exp(1j * k * th_o)))) for k in OUTPUT_ORDERS}

    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))
    for a, ph, nb, lab in ((ax[0, 0], th_m, 36, 'Motor angle (deg, one motor rev)'),
                           (ax[0, 1], c.POLE_PAIRS * th_m, 24, 'Electrical angle (deg)')):
        h, e = np.histogram(np.mod(ph, c.TWO_PI) * 180 / np.pi, bins=nb, range=(0, 360))
        a.bar(0.5 * (e[1:] + e[:-1]), h / h.mean(), width=0.9 * 360 / nb, color='#3987e5')
        a.axhline(1, color=c.INK_3, ls=':', lw=1)
        a.set_xlabel(lab)
        a.set_ylabel('Stick events / uniform expectation')
        a.set_xlim(0, 360)
        a.set_xticks(range(0, 361, 60))
    ax[0, 0].set_title('Where the motor sticks: motor angle')
    ax[0, 1].set_title('Where the motor sticks: electrical angle')
    for a, plv, lab in ((ax[1, 0], plv_m, 'Order per MOTOR rev'), (ax[1, 1], plv_o, 'Order per OUTPUT rev')):
        ks = list(plv)
        a.bar(range(len(ks)), [plv[k] for k in ks], color=[('#1c5cab' if plv[k] > thr else '#b7d3f6') for k in ks])
        a.axhline(thr, color='#a8302f', ls='--', lw=1.2, label=f'1% chance level (n={n})')
        a.set_xticks(range(len(ks)))
        a.set_xticklabels([str(k) for k in ks])
        a.set_xlabel(lab)
        a.set_ylabel('Phase locking |mean e^(ik theta)|')
        a.legend(loc='upper right')
    ax[1, 0].set_title('Which motor-side order decides where it sticks')
    ax[1, 1].set_title('Which output-side order decides where it sticks')
    fig.suptitle(f'Skip stick locations, backdrive dwells at 10-{c.i_max(d):g} A ({n} stuck intervals >= {int(MIN_STICK_S * 1e3)} ms)')
    c.note(fig, 'Order 21 per motor rev = electrical fundamental; 26 per output rev = once per motor rev; '
                '25/27 per output rev = cycloid lobe/pin candidates. Bars above the dashed line are '
                'statistically locked.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    c.save(fig, SUB, 'Skip_stick_location_vs_angle.png')

    _per_dwell_csv(dw, stats)
    rec.update({'stick_plv_motor_orders': plv_m, 'stick_plv_output_orders': plv_o,
                'stick_plv_threshold_1pct': thr, 'n_stick_intervals': n})
    c.record('stick_slip', rec)
    return stats


def _per_dwell_csv(dw, stats):
    rows = []
    for w in dw:
        st = stats[w['n']]
        rows.append([w['n'], w['speed'], w['i_along_nom'], w['quadrant'], st['stuck_frac'],
                     st['slip_rate_hz'], st['n_slips'], st['drop_median_Nm'],
                     st['slip_angle_median_rad'] * 180 / np.pi, st['P_std_Nm']])
    c.save_csv(SUB, 'Skip_stick_slip_per_dwell.csv',
               ['dwell', 'output_speed_cmd_rad_s', 'current_along_motion_A', 'quadrant', 'stuck_fraction',
                'slip_bursts_per_s', 'n_slip_bursts', 'median_breakaway_drop_Nm',
                'median_slip_travel_motor_deg', 'torque_std_Nm'], rows)


def _record(d, dw, stats):
    im = c.i_max(d)

    def mean_where(key, cond):
        v = [stats[w['n']][key] for w in dw if cond(w)]
        return float(np.nanmean(v)) if v else None
    return {
        'i_max_A': im,
        'stuck_frac_backdrive_imax_by_speed': {f'{v:g}': mean_where('stuck_frac', lambda w, v=v: w['speed_abs'] == v and w['i_along_nom'] == -im) for v in c.speeds(d)},
        'stuck_frac_forward_mean': mean_where('stuck_frac', lambda w: w['quadrant'] == 'forward'),
        'stuck_frac_zero_mean': mean_where('stuck_frac', lambda w: w['quadrant'] == 'zero'),
        'stuck_frac_backdrive_max': max(stats[w['n']]['stuck_frac'] for w in dw if w['quadrant'] == 'backdrive'),
        'drop_median_backdrive_imax_Nm': mean_where('drop_median_Nm', lambda w: w['i_along_nom'] == -im),
        'slip_angle_median_backdrive_imax_motor_deg': 180 / np.pi * (mean_where('slip_angle_median_rad', lambda w: w['i_along_nom'] == -im) or np.nan),
    }


if __name__ == '__main__':
    c.use_run_from_argv()
    run(c.load())
