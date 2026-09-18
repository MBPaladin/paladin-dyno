"""Torsional stiffness from the wind-up ramps.

Customer ask (1.2.2 Paladin Test Request, 'Stiffness test'):

    Gradually increase torque on the output up to specified torque and hold
    for 5 seconds before reversing direction
    Static torque limit = 90% and 45% of measured static slip torque
    Stiffness: end to end, or as a function of torque

HOW THE WIND-UP IS MEASURED, and why it needs saying: this bench has an encoder
on each side of the drive, both of which turn a long way during the test. The
input turns R times as far as the output by construction, so the wind-up is the
SMALL RESIDUAL left after taking the ratio out:

    windup = (output angle) - (input angle / R)

On the 2026-09-17 K45 ramp those two terms are 0.2426 and 0.2429 rad and the
residual is 2.6 mrad -- a 1% difference of two much larger numbers. That is
resolvable on these encoders, but it means the answer is only as good as R, so
R is taken from the NO-LOAD sweep (where there is no wind-up to corrupt it)
rather than from the nameplate or from the loaded data itself.

It also means the absorber's own position loop is in series with the drive. The
absorber is holding position, not grounded, so what deflects is the gearbox AND
the servo stiffness behind it. The number here is therefore a lower bound on
the drive's own stiffness, and the report says so rather than quietly calling
it the drive's.
"""

import numpy as np

from .. import naming, physics, plotting
from ..result import Result

# Fit the loop over the middle of the torque range. The ends are the dwell and
# the reversal, where the loop is at its widest and the slope is backlash, not
# stiffness.
FIT_BAND = (0.2, 0.9)

# The output cell chatters tens of Nm sample to sample while the absorber holds
# the dwell -- see the raw K45 loop, which is a 20 Nm curve inside a 90 Nm
# spray. The slope is fitted to the filtered trace; the raw one is still what
# the pk-pk columns report, so the chatter is not hidden, only kept out of the
# regression.
SMOOTH_S = 0.1


def analyze(ds, cfg):
    res = Result('stiffness', 'Torsional stiffness')
    spans = ds.select(kind=naming.STIFFNESS)
    if not spans:
        res.add('warn', 'no_data', 'no stiffness ramps in this campaign')
        return res

    ratio = cfg['ratio']
    fits = []
    for span in sorted(spans, key=lambda s: (s.direction, s.point.pct or 0)):
        fits.append(_fit(span, ratio))

    res.tables.append(('stiffness__fits', plotting.csv(
        [[f.get(k) for k in _COLS] for f in fits], _COLS)))
    for f in fits:
        if f['curve'] is not None:
            res.tables.append((f'stiffness__curve_{f["segment"]}', f['curve']))

    res.figures.append(('hysteresis', _fig_hysteresis(spans, fits, ratio, cfg)))
    res.figures.append(('stiffness_vs_torque', _fig_vs_torque(fits, cfg)))
    res.figures.append(('tracking', _fig_tracking(spans, fits, ratio, cfg)))
    _findings(res, fits, cfg)
    res.metrics['fits'] = {f['segment']: {
        'k_nm_per_rad': f['k_nm_per_rad'],
        'windup_pk_pk_rad': f['windup_pk_pk_rad'],
        'torque_pk_pk_nm': f['torque_pk_pk_nm']} for f in fits}
    return res


_COLS = ['segment', 'direction', 'target_pct', 'source', 'n',
         'torque_pk_pk_nm', 'torque_pk_pk_raw_nm', 'torque_peak_nm',
         'windup_pk_pk_rad',
         'k_nm_per_rad', 'k_r2', 'hysteresis_rad', 'sign_used',
         'ratio_tracking_r', 'residual_frac_of_travel']


def _torque(seg):
    """Output torque, filtered, as every stiffness number here reads it."""
    return physics.median_filter(seg['load_torque'],
                                 physics.samples_for(seg, SMOOTH_S))


def _windup(seg, ratio):
    """Output-referred relative deflection across the drive, zeroed at start.

    Sign is resolved by the data, not assumed: the bench's LOAD channel already
    carries a direction flip chosen for a different test, so which of the two
    differences is 'wind-up' is not knowable from the config. The one whose
    deflection rises WITH the applied torque is the physical one; the other is
    its negative.
    """
    dp = seg['dut_output_position'] / ratio
    lp = seg['load_position']
    to = _torque(seg)
    a = (lp - lp[0]) - (dp - dp[0])
    ok = np.isfinite(a) & np.isfinite(to)
    if ok.sum() < 10:
        return a, +1
    corr = np.corrcoef(a[ok], to[ok])[0, 1]
    sign = -1 if corr < 0 else +1
    return a * sign, sign


def _fit(span, ratio):
    seg = span.seg
    w, sign = _windup(seg, ratio)
    to = _torque(seg)
    raw = seg['load_torque']
    dp = seg['dut_output_position'] / ratio
    lp = seg['load_position']

    row = {
        'segment': span.point.raw, 'direction': span.direction,
        'target_pct': span.point.pct, 'source': span.source, 'n': len(seg),
        'torque_pk_pk_nm': float(np.ptp(to)),
        'torque_peak_nm': float(np.nanmax(np.abs(to))),
        # The unfiltered swing, so the chatter the filter removed stays visible
        # in the table rather than being quietly dropped.
        'torque_pk_pk_raw_nm': float(np.ptp(raw)),
        'windup_pk_pk_rad': float(np.ptp(w)),
        'sign_used': sign,
        # How faithfully the two shafts tracked the ratio. Near 1 means the
        # residual really is wind-up; well below means something slipped and
        # the 'wind-up' is contaminated by gross motion.
        'ratio_tracking_r': float(np.corrcoef(dp, lp)[0, 1]),
        'residual_frac_of_travel': float(np.ptp(w) / max(np.ptp(lp), 1e-12)),
        'k_nm_per_rad': None, 'k_r2': None, 'hysteresis_rad': None,
        'curve': None,
    }

    peak = np.nanmax(np.abs(to))
    band = (np.abs(to) > FIT_BAND[0] * peak) & (np.abs(to) < FIT_BAND[1] * peak)
    m = band & np.isfinite(w) & np.isfinite(to)
    if m.sum() > 50:
        k, c = np.polyfit(w[m], to[m], 1)
        pred = k * w[m] + c
        ss = np.sum((to[m] - np.mean(to[m])) ** 2)
        row['k_nm_per_rad'] = float(k)
        row['k_r2'] = float(1 - np.sum((to[m] - pred) ** 2) / ss) if ss else None
        # Loop width at zero torque: the backlash-plus-hysteresis the customer
        # sees as lost motion, quoted separately from the slope.
        near0 = np.abs(to) < 0.05 * peak
        if near0.sum() > 20:
            row['hysteresis_rad'] = float(np.ptp(w[near0]))
        row['curve'] = _curve_csv(w, to)
    return row


def _curve_csv(w, to, bins=60):
    """Stiffness as a function of torque: local slope in torque bins.

    The customer asked for 'end to end OR as a function of torque'. This is the
    second form, and it is the one that shows a traction drive softening as the
    contact approaches slip.
    """
    edges = np.linspace(np.nanmin(to), np.nanmax(to), bins + 1)
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (to >= a) & (to < b) & np.isfinite(w)
        if m.sum() < 30:
            continue
        k = np.polyfit(w[m], to[m], 1)[0] if np.ptp(w[m]) > 1e-9 else float('nan')
        rows.append([float((a + b) / 2), float(np.mean(w[m])), float(k),
                     int(m.sum())])
    if not rows:
        return None
    return plotting.csv(rows, ['torque_nm', 'windup_rad', 'k_nm_per_rad', 'n'])


def _fig_hysteresis(spans, fits, ratio, cfg):
    fig, axes = plotting.grid_figure(
        1, len(spans), f'{cfg["unit_label"]}  --  stiffness hysteresis loops',
        'output torque against output-referred wind-up; the fitted slope is '
        'taken over the middle of the range, away from the reversal')
    for ax, span, f in zip(axes.ravel(), spans, fits):
        w, _ = _windup(span.seg, ratio)
        to = _torque(span.seg)
        ax.plot(w * 1e3, to, lw=0.7, color=plotting.FORWARD_C, alpha=0.8)
        if f['k_nm_per_rad']:
            xs = np.linspace(np.nanmin(w), np.nanmax(w), 10)
            ax.plot(xs * 1e3, f['k_nm_per_rad'] * xs
                    + (np.nanmean(to) - f['k_nm_per_rad'] * np.nanmean(w)),
                    ls='--', lw=1.4, color=plotting.BACKDRIVE_C,
                    label=f'K = {f["k_nm_per_rad"]:.0f} Nm/rad')
            # Pinned upper-left so it cannot land on the caveat stamped in the
            # bottom-right corner, which is the thing a reader most needs on
            # the ramp where the fit is meaningless.
            ax.legend(fontsize=8, loc='upper left')
        ax.set_title(f'{span.point.raw}  ({f["target_pct"]}% of slip)',
                     fontsize=10)
        ax.set_xlabel('wind-up (mrad, output-referred)')
        ax.set_ylabel('output torque (Nm)')
        plotting.stamp(ax, f'residual is {f["residual_frac_of_travel"] * 100:.1f}% '
                           'of the output travel')
    return plotting.finish_grid(fig)


def _fig_vs_torque(fits, cfg):
    fig, ax = plotting.figure(
        f'{cfg["unit_label"]}  --  stiffness as a function of torque',
        'local slope of the wind-up curve in torque bins')
    import io
    for f in fits:
        if not f['curve']:
            continue
        rows = np.genfromtxt(io.StringIO(f['curve']), delimiter=',',
                             names=True)
        ax.plot(np.atleast_1d(rows['torque_nm']),
                np.atleast_1d(rows['k_nm_per_rad']),
                marker='.', ms=4, lw=1.1,
                label=f'{f["segment"]} ({f["target_pct"]}% of slip)')
    ax.set_xlabel('output torque (Nm)')
    ax.set_ylabel('local stiffness (Nm/rad)')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    return plotting.finish(fig)


def _fig_tracking(spans, fits, ratio, cfg):
    """The two shaft angles on one axis, so the reader can see the residual.

    This figure exists to keep the stiffness number honest. The wind-up is a
    1% difference between these two traces; a reader who cannot see how close
    they are cannot judge how much to trust the slope fitted from their gap.
    """
    fig, axes = plotting.grid_figure(
        1, len(spans), f'{cfg["unit_label"]}  --  shaft angles during the '
        'stiffness ramps',
        'input referred to the output through the no-load ratio; wind-up is '
        'the gap between them, magnified below')
    for ax, span, f in zip(axes.ravel(), spans, fits):
        seg = span.seg
        t = seg['time'] - seg['time'][0]
        dp = seg['dut_output_position'] / ratio
        lp = seg['load_position']
        ax.plot(t, dp - dp[0], lw=1.0, color='#7f7f7f',
                label='input / ratio')
        ax.plot(t, lp - lp[0], lw=1.0, color=plotting.FORWARD_C,
                label='output')
        w, _ = _windup(seg, ratio)
        twin = ax.twinx()
        twin.plot(t, w * 1e3, lw=0.9, color=plotting.BACKDRIVE_C, alpha=0.8)
        twin.set_ylabel('wind-up (mrad)', color=plotting.BACKDRIVE_C)
        ax.set_title(f'{span.point.raw}  (tracking r = '
                     f'{f["ratio_tracking_r"]:.4f})', fontsize=10)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('angle from start (rad, output-referred)')
        ax.legend(loc='upper left', fontsize=8)
    return plotting.finish_grid(fig)


def _findings(res, fits, cfg):
    for f in fits:
        if f['k_nm_per_rad'] is None:
            res.add('warn', 'no_fit',
                    f'{f["segment"]}: not enough samples in the fit band to '
                    'fit a stiffness')
            continue
        res.add('info', 'stiffness',
                f'{f["segment"]} ({f["target_pct"]}% of slip): '
                f'K = {f["k_nm_per_rad"]:.0f} Nm/rad over '
                f'{f["torque_pk_pk_nm"]:.0f} Nm and '
                f'{f["windup_pk_pk_rad"] * 1e3:.1f} mrad'
                + (f', R2 = {f["k_r2"]:.3f}' if f['k_r2'] is not None else ''))
        if f['ratio_tracking_r'] < 0.99:
            res.add('warn', 'tracking_lost',
                    f'{f["segment"]}: the shafts tracked the ratio at only '
                    f'r = {f["ratio_tracking_r"]:.4f} -- the drive did not '
                    'stay coupled through this ramp, so its "wind-up" contains '
                    'gross motion and its stiffness is not trustworthy')
        if f['residual_frac_of_travel'] < 0.02:
            res.add('warn', 'small_residual',
                    f'{f["segment"]}: wind-up is only '
                    f'{f["residual_frac_of_travel"] * 100:.1f}% of the output '
                    'travel, so the stiffness is a small difference of two '
                    'large angles and is sensitive to the ratio calibration')
        if f['hysteresis_rad']:
            res.add('info', 'lost_motion',
                    f'{f["segment"]}: loop width at zero torque is '
                    f'{f["hysteresis_rad"] * 1e3:.2f} mrad (lost motion, '
                    'quoted apart from the slope)')
    res.add('warn', 'series_compliance',
            'the absorber holds the output with a position loop rather than a '
            'ground, so its servo stiffness is in series with the drive: every '
            'K here is a LOWER BOUND on the drive\'s own stiffness')
    good = [f for f in fits if f['k_nm_per_rad'] and f['ratio_tracking_r'] >= 0.99]
    res.summary = (
        f'{len(fits)} stiffness ramp(s)'
        + (f'; K = {np.median([f["k_nm_per_rad"] for f in good]):.0f} Nm/rad '
           f'from {len(good)} trustworthy ramp(s)' if good
           else '; no ramp stayed coupled well enough to trust its slope'))
