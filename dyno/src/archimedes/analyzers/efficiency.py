"""Efficiency, and the creep ratio the customer wants watched alongside it.

Customer ask (1.2.2 Paladin Test Request, 'Efficiency'):

    Torque step = 5 Nm, to 100 Nm (1.2.0/1.2.1) or 140 Nm (1.2.2)
    w_in = 20, 300, 1500, 3000 rpm
    Additional monitoring: creep ratio should stay below 5%

Each level is a bipolar shuttle against a held output torque, so -- and this is
the thing to understand before reading any number out of here -- ONE SEGMENT
CONTAINS BOTH DIRECTIONS OF POWER FLOW. On the half-traverse where the held
torque opposes the motion the input drives the output; on the other half it
assists, and the output drives the input. Averaging the segment whole mixes a
forward efficiency with a back-driven one and lands between them, which is a
number that describes nothing.

So legs are measured separately and binned by where the power actually flowed
(physics.classify_leg). The forward sweep therefore yields a back-driven
efficiency too, at no extra bench time. It is reported under its own heading
and never merged with a dedicated back-drive run: here the INPUT is still the
shaft holding the traverse, so it is the drive's back-driven loss measured
under a position-controlled input, not under the customer's back-drive plan.

Two corrections matter at the low end and are applied explicitly:

  * The output cell's zero. Measured from the no-load traverses, ~1.4 Nm on the
    2026-09-17 campaign -- 28% of the smallest 5 Nm step. Uncorrected, the
    bottom of every speed curve is wrong by that much and some points come out
    over 100% efficient.
  * Nothing is done about train drag, which is real and NOT part of the
    gearbox. It is reported next to the curves so a reader can see how much of
    the low-torque loss is the bench rather than the unit.
"""

import numpy as np

from .. import naming, physics, plotting
from ..result import Result

# Legs whose two shaft powers disagree in sign are coasting or mis-zeroed; an
# efficiency off them is meaningless. Counted and reported, never plotted.
# Legs above this are not believable and point at a cell fault, not a drive.
ETA_SANE_MAX = 1.0

CREEP_LIMIT_PCT = 5.0

# Creep is a RATIO of two speeds, so it is far more sensitive to an unsettled
# leg than a power measurement is: a 50 ms leg caught while the output is still
# reversing divides two numbers that are not yet describing the same motion and
# reports thousands of percent. It is therefore measured only on legs that held
# for this long, and from the leg's MEAN speeds rather than sample by sample.
CREEP_MIN_LEG_S = 0.25

# Creep only means anything while the two shafts are still coupled. Past this
# deviation from the kinematic ratio they are not creeping, they have come
# apart -- the output turns and the input does not follow -- and calling that a
# creep percentage produces the six-figure numbers the 2026-09-18 back-drive
# run first reported. Such legs are counted and reported as not tracking,
# which is the fact, rather than averaged into a creep figure.
TRACKING_MAX_PCT = 100.0

# The output cell can carry a bias whose sign flips with rotation direction --
# Coulomb friction parasitic to the measurement path rather than torque:
#
#     T_measured = T_true + z + h * sign(w_out)
#
# `z` is the constant zero physics.cell_offsets recovers from the no-load
# traverses. `h` matters here far more than it looks, because a forward-driving
# leg and a back-driven leg of the SAME traverse have opposite w_out: an
# unremoved h is subtracted from one and added to the other, and so manufactures
# a forward-versus-back-drive gap out of nothing.
#
# The estimator is the one dyno/src/analysis/processors/efficiency.py uses (see
# its _hysteresis): within a group of legs sharing the sign of the TRUE output
# torque, two members differ by exactly 2h -- the torque, the zero, the loss and
# any loss asymmetry are common to both and cancel. Two such groups give two
# independent estimates, and they are kept apart on purpose: if they disagree
# the sign model itself is wrong.
#
# It is a DIAGNOSTIC here, not a correction, unless it passes both gates below.
# That processor reads steady dwells at a commanded operating point; this bench
# runs a bipolar shuttle that holds its commanded speed for as little as 9% of a
# span, so the premise of a settled operating point is much weaker.
HYST_AGREE_TOL_NM = 0.3      # the two independent halves must agree to this
HYST_SPREAD_TOL_NM = 0.5     # and h must be consistent across the grid

# Below this fraction of a segment spent at the commanded speed, the operating
# point is not a steady traverse and the values taken from it rest on a small
# part of the record. Reported, not silently dropped.
LOW_COVERAGE_FRAC = 0.20


def analyze(ds, cfg):
    res = Result('efficiency', 'Efficiency and creep ratio')
    spans = ds.select(kind=naming.EFFICIENCY)
    if not spans:
        res.add('warn', 'no_data', 'no efficiency segments in this campaign')
        return res

    ratio = cfg['ratio']
    offsets, detail = (cfg.get('cell_offsets') or ({}, {}))
    t_out_off = offsets.get('load_torque', 0.0) if cfg.get('correct_zero', True) else 0.0
    t_in_off = offsets.get('input_torque', 0.0) if cfg.get('correct_zero', True) else 0.0

    legs = []
    for span in spans:
        legs.extend(_span_legs(span, ratio, t_out_off, t_in_off))
    if not legs:
        res.add('warn', 'no_plateaus', 'efficiency segments hold no measurable '
                                       'constant-speed leg')
        return res

    hyst = _hysteresis(legs)
    if hyst.get('rows'):
        res.tables.append(('efficiency__torque_asymmetry', plotting.csv(
            [[f'{a:g}', f'{b:g}', c, d, e] for a, b, c, d, e in hyst['rows']],
            ['rpm_cmd', 't_cmd_nm', 'h_pos_nm', 'h_neg_nm', 'h_nm'])))
    if hyst.get('applied') and cfg.get('correct_asymmetry', True):
        h = hyst['h']
        for l in legs:
            adj = h * (1 if l['w_out'] > 0 else -1)
            l['t_out'] -= adj
            l['p_out'] = l['t_out'] * l['w_out']
            c = physics.classify_leg(l, ratio)
            l.update(eta=c['eta'], loss_w=c['loss_w'],
                     coherent=c['coherent'], flow=c['direction'])

    res.tables.append(('efficiency__per_leg', plotting.csv(
        [[l.get(k) for k in _LEG_COLS] for l in legs], _LEG_COLS)))

    usable = [l for l in legs if l['coherent'] and np.isfinite(l['eta'])]
    points = _pool(usable)
    res.tables.append(('efficiency__per_point', plotting.csv(
        [[p.get(k) for k in _PT_COLS] for p in points], _PT_COLS)))

    for direction in ('forward', 'backdrive'):
        sel = [p for p in points if p['flow'] == direction]
        if not sel:
            continue
        res.figures.append((f'efficiency_vs_torque_{direction}',
                            _fig_efficiency(sel, direction, cfg)))
        res.figures.append((f'loss_vs_torque_{direction}',
                            _fig_loss(sel, direction, cfg)))
    if len({p['flow'] for p in points}) > 1:
        res.figures.append(('efficiency_both_directions',
                            _fig_both(points, cfg)))

    creep = _creep(spans, ratio, cfg)
    if creep:
        res.tables.append(('efficiency__creep', plotting.csv(
            [[c.get(k) for k in _CREEP_COLS] for c in creep], _CREEP_COLS)))
        res.figures.append(('creep_vs_torque', _fig_creep(creep, cfg)))

    cov = _coverage(spans, ratio)
    if cov:
        res.tables.append(('efficiency__speed_coverage', plotting.csv(
            [[c.get(k) for k in _COV_COLS] for c in cov], _COV_COLS)))
        cfg['_coverage'] = cov

    if detail:
        res.figures.append(('cell_zero_and_drag', _fig_zero(detail, cfg)))

    res.figures.append(('coverage', _fig_coverage(points, cfg)))
    _findings(res, legs, usable, points, creep, detail, cfg, cov, hyst)
    res.metrics['points'] = {
        f'{p["flow"]}_{p["rpm"]:.0f}rpm_{p["t_out_nm"]:+.0f}Nm': {
            'eta': p['eta'], 'n_legs': p['n_legs'], 'spread': p['eta_spread'],
        } for p in points}
    return res


_LEG_COLS = ['segment', 'source', 'flow', 'span_direction', 'rpm_cmd',
             't_cmd_nm', 'cycle',
             'leg', 'sign', 'n', 'w_in', 'w_out', 't_in', 't_out',
             'p_in', 'p_out', 'eta', 'loss_w', 'coherent']

_PT_COLS = ['flow', 'span_direction', 'rpm', 't_out_nm', 't_cmd_nm',
            'n_legs', 'eta',
            'eta_spread', 'eta_lo', 'eta_hi', 'loss_w', 't_in_nm',
            'p_in_w', 'p_out_w']

_CREEP_COLS = ['segment', 'direction', 'rpm_cmd', 't_cmd_nm', 'n_legs',
               'n_tracking', 'creep_pct', 'creep_pct_max', 'source_channel']

_COV_COLS = ['segment', 'direction', 'rpm_cmd', 't_cmd_nm', 'n_samples',
             'n_at_speed', 'coverage_frac', 'frac_at_rest',
             'rpm_in_during_legs', 'rpm_in_span_average']


def _span_legs(span, ratio, t_out_off, t_in_off):
    seg = span.seg
    drive = ('dut_velocity' if span.direction == 'forward' else 'load_velocity')
    target = physics.drive_target(span.point.rpm, span.direction, ratio)
    out = []
    for i, (sl, sign) in enumerate(physics.legs(seg, drive, target=target)):
        p = physics.leg_power(seg, sl, t_out_offset=t_out_off,
                              t_in_offset=t_in_off)
        c = physics.classify_leg(p, ratio)
        out.append({
            'segment': span.point.raw, 'source': span.source,
            'flow': c['direction'],
            # Which sweep this leg came from, as opposed to which way power
            # flowed on it. A back-driven leg of the FORWARD sweep and a leg
            # of the dedicated back-drive sweep are different measurements --
            # the shaft holding the traverse is not the same one -- so they
            # are kept apart rather than averaged together.
            'span_direction': span.direction,
            'rpm_cmd': span.point.rpm,
            't_cmd_nm': span.point.torque_nm, 'cycle': span.point.cycle,
            'leg': i, 'sign': sign, 'n': c['n'],
            'w_in': c['w_in'], 'w_out': c['w_out'],
            't_in': c['t_in'], 't_out': c['t_out'],
            'p_in': c['p_in'], 'p_out': c['p_out'],
            'eta': c['eta'], 'loss_w': c['loss_w'],
            'coherent': c['coherent'],
        })
    return out


def _pool(legs):
    """One point per (flow, speed, |commanded torque|).

    Keyed on the MAGNITUDE of the commanded level, because the plan alternates
    the sign of every level to cancel creep between cycles -- +20 Nm and -20 Nm
    are the same operating point driven the two ways round, and the customer's
    curve has one point at 20 Nm, not two.
    """
    buckets = {}
    for l in legs:
        key = (l['flow'], l['span_direction'], l['rpm_cmd'],
               abs(l['t_cmd_nm']))
        buckets.setdefault(key, []).append(l)
    out = []
    for (flow, span_direction, rpm, t_cmd), ls in sorted(buckets.items()):
        etas = [l['eta'] for l in ls]
        out.append({
            'flow': flow, 'span_direction': span_direction,
            'rpm': rpm, 't_cmd_nm': t_cmd, 'n_legs': len(ls),
            't_out_nm': float(np.median([abs(l['t_out']) for l in ls])),
            't_in_nm': float(np.median([abs(l['t_in']) for l in ls])),
            'eta': float(np.median(etas)),
            'eta_spread': float(np.ptp(etas)) if len(etas) > 1 else 0.0,
            # The extremes themselves, so the plotted bar is the range the
            # legs actually covered rather than a symmetric width that can
            # reach past 100% or below zero.
            'eta_lo': float(np.min(etas)), 'eta_hi': float(np.max(etas)),
            'loss_w': float(np.median([l['loss_w'] for l in ls])),
            'p_in_w': float(np.median([abs(l['p_in']) for l in ls])),
            'p_out_w': float(np.median([abs(l['p_out']) for l in ls])),
        })
    return out


def _speeds(points):
    return sorted({p['rpm'] for p in points})


def _speed_colors(speeds):
    cmap = plotting.plt.get_cmap('viridis')
    n = max(len(speeds) - 1, 1)
    return {s: cmap(i / n * 0.85) for i, s in enumerate(speeds)}


def _fig_efficiency(points, direction, cfg):
    fig, ax = plotting.figure(
        f'{cfg["unit_label"]}  --  efficiency vs output torque '
        f'({plotting.DIR_LABEL[direction]})',
        'median over repeat legs; bars span the legs at that point')
    colors = _speed_colors(_speeds(points))
    for rpm in _speeds(points):
        sel = sorted((p for p in points if p['rpm'] == rpm),
                     key=lambda p: p['t_out_nm'])
        ax.errorbar([p['t_out_nm'] for p in sel],
                    [p['eta'] * 100 for p in sel],
                    yerr=[[max((p['eta'] - p['eta_lo']) * 100, 0) for p in sel],
                          [max((p['eta_hi'] - p['eta']) * 100, 0) for p in sel]],
                    marker='o', ms=4, capsize=2, lw=1.4, color=colors[rpm],
                    label=f'{rpm:g} rpm in')
    ax.set_xlabel('output torque (Nm, measured)')
    ax.set_ylabel('efficiency (%)')
    ax.set_ylim(0, 105)
    ax.legend(title='input speed', fontsize=8)
    plotting.stamp(ax, _zero_note(cfg))
    return plotting.finish(fig)


def _fig_loss(points, direction, cfg):
    fig, ax = plotting.figure(
        f'{cfg["unit_label"]}  --  power loss vs output torque '
        f'({plotting.DIR_LABEL[direction]})',
        'shaft power in minus shaft power out')
    colors = _speed_colors(_speeds(points))
    for rpm in _speeds(points):
        sel = sorted((p for p in points if p['rpm'] == rpm),
                     key=lambda p: p['t_out_nm'])
        ax.plot([p['t_out_nm'] for p in sel], [p['loss_w'] for p in sel],
                marker='o', ms=4, lw=1.4, color=colors[rpm],
                label=f'{rpm:g} rpm in')
    ax.set_xlabel('output torque (Nm, measured)')
    ax.set_ylabel('power loss (W)')
    ax.legend(title='input speed', fontsize=8)
    return plotting.finish(fig)


def _fig_both(points, cfg):
    speeds = _speeds(points)
    fig, axes = plotting.grid_figure(
        1, len(speeds),
        f'{cfg["unit_label"]}  --  forward vs back-driven efficiency',
        'both halves of the same shuttle: the held torque opposes the traverse '
        'on one leg and assists on the other')
    for ax, rpm in zip(axes.ravel(), speeds):
        for flow in ('forward', 'backdrive'):
            sel = sorted((p for p in points
                          if p['rpm'] == rpm and p['flow'] == flow),
                         key=lambda p: p['t_out_nm'])
            if not sel:
                continue
            ax.plot([p['t_out_nm'] for p in sel], [p['eta'] * 100 for p in sel],
                    marker='o', ms=3.5, lw=1.3,
                    color=plotting.DIR_COLOR[flow],
                    label=plotting.DIR_LABEL[flow])
        ax.set_title(f'{rpm:g} rpm input', fontsize=10)
        ax.set_xlabel('output torque (Nm)')
        ax.set_ylim(0, 105)
    axes.ravel()[0].set_ylabel('efficiency (%)')
    axes.ravel()[0].legend(fontsize=8)
    return plotting.finish_grid(fig)


def _creep(spans, ratio, cfg):
    """Creep ratio per level, as a percentage of the ideal output speed.

    creep = 1 - (ratio x w_out) / w_in, with `ratio` taken from the NO-LOAD
    sweep so it is the kinematic ratio and not one already containing the creep
    being measured.

    Evaluated from the MEAN speeds of each settled constant-speed leg, not
    sample by sample. Sample-wise, the quotient is unbounded wherever the input
    speed passes near zero -- which it does at every reversal and throughout the
    low-speed points, where this drive does not hold speed at all -- and a
    single such sample sets the maximum for the whole level. The 2026-09-17
    campaign reported creep maxima of 2.3e4 and 1.6e5 percent that way.
    """
    out = []
    for span in spans:
        seg = span.seg
        drive = ('dut_velocity' if span.direction == 'forward'
                 else 'load_velocity')
        target = physics.drive_target(span.point.rpm, span.direction, ratio)
        vals = []
        for sl, _sign in physics.legs(seg, drive, target=target,
                                      min_s=CREEP_MIN_LEG_S):
            wi = float(np.nanmean(seg['dut_velocity'][sl]))
            wo = float(np.nanmean(seg['load_velocity'][sl]))
            if not np.isfinite(wi) or not np.isfinite(wo) or abs(wi) < 1e-9:
                continue
            vals.append(abs(1.0 - (ratio * wo) / wi) * 100.0)
        if not vals:
            continue
        tracking = [v for v in vals if v <= TRACKING_MAX_PCT]
        out.append({
            'segment': span.point.raw, 'direction': span.direction,
            'rpm_cmd': span.point.rpm,
            't_cmd_nm': span.point.torque_nm,
            'n_legs': len(vals), 'n_tracking': len(tracking),
            'creep_pct': float(np.median(tracking)) if tracking else None,
            'creep_pct_max': float(np.max(tracking)) if tracking else None,
            'source_channel': 'leg-mean velocities',
        })
    return out


def _hysteresis(legs, tol=HYST_AGREE_TOL_NM, spread_tol=HYST_SPREAD_TOL_NM):
    """Recover the output cell's direction-dependent bias h, if it exists.

    Legs are grouped by (|commanded speed|, |commanded torque|) and then by
    (power-flow direction, sign of w_out). Those four combinations carry:

        flow          sign(w_out)   true torque sign   the cell reads
        forward           +1              +1            T + z + h
        backdrive         -1              +1            T + z - h
        backdrive         +1              -1           -T + z + h
        forward           -1              -1           -T + z - h

    so the first pair differs by 2h and the second pair differs by 2h, with
    everything else common and cancelling. Estimated from RAW torque: a
    constant z cancels in every difference, so this does not care whether the
    zero has been removed yet.
    """
    groups = {}
    for l in legs:
        if l['t_cmd_nm'] is None or l['rpm_cmd'] is None:
            continue
        key = (abs(l['rpm_cmd']), abs(l['t_cmd_nm']))
        combo = (l['flow'], 1 if l['w_out'] > 0 else -1)
        groups.setdefault(key, {}).setdefault(combo, []).append(l['t_out'])

    need = (('forward', +1), ('backdrive', -1),
            ('backdrive', +1), ('forward', -1))
    per_key, pos, neg, rows = [], [], [], []
    for key in sorted(groups):
        g = groups[key]
        if not all(c in g for c in need):
            continue
        m = {c: float(np.mean(g[c])) for c in need}
        h_pos = (m[need[0]] - m[need[1]]) / 2
        h_neg = (m[need[2]] - m[need[3]]) / 2
        per_key.append(0.5 * (h_pos + h_neg))
        pos.append(h_pos)
        neg.append(h_neg)
        rows.append([key[0], key[1], h_pos, h_neg, 0.5 * (h_pos + h_neg)])

    if not per_key:
        return {'n_keys': 0, 'applied': False, 'rows': rows}
    h_pos, h_neg = float(np.median(pos)), float(np.median(neg))
    spread = float(np.std(per_key))
    disagree = abs(h_pos - h_neg)
    return {
        'n_keys': len(per_key), 'h': float(np.median(per_key)),
        'h_pos': h_pos, 'h_neg': h_neg,
        'disagree': disagree, 'spread': spread,
        'range': (float(np.min(per_key)), float(np.max(per_key))),
        'halves_agree': disagree <= tol,
        'consistent': spread <= spread_tol,
        'applied': disagree <= tol and spread <= spread_tol,
        'rows': rows,
    }


def _coverage(spans, ratio):
    """How much of each segment was actually spent at the commanded speed.

    A traverse that holds its speed spends most of its span in band. A point
    where the drive judders -- which this one does at the lowest speed under
    load -- spends very little, and every value taken from it rests on that
    small part of the record. This is the number that says which is which.
    """
    out = []
    for span in spans:
        seg = span.seg
        drive = ('dut_velocity' if span.direction == 'forward'
                 else 'load_velocity')
        target = physics.drive_target(span.point.rpm, span.direction, ratio)
        legs = physics.legs(seg, drive, target=target)
        held = sum(sl.stop - sl.start for sl, _ in legs)
        w = seg['dut_velocity']
        # Two different speeds, and the difference between them is the point.
        # `in_leg` is how fast the shaft moved while it was moving -- the speed
        # the efficiency numbers were actually measured at. `mean_abs` is the
        # average over the whole span including everything it spent stationary,
        # taken from the position travel so no velocity-estimator artefact can
        # reach it. At the low-speed points the drive does not hold speed: it
        # sits still and catches up in bursts, and these two numbers are what
        # say so.
        in_leg = float(np.nanmean(np.concatenate(
            [np.abs(w[sl]) for sl, _ in legs]))) if legs else float('nan')
        dur = seg.duration
        travel = float(np.ptp(seg['dut_output_position']))
        # Split what is NOT on speed into standing still versus moving at the
        # wrong speed. They are different facts: a segment that is mostly at
        # rest is spending its time in the plan's own lead-in, settle and
        # turnaround dwells, which is a property of the test; one that is
        # moving off-speed is a property of the drive.
        drv = seg[drive]
        ref = abs(target) if target else float('nan')
        at_rest = (float(np.mean(np.abs(drv) < 0.05 * ref))
                   if np.isfinite(ref) and ref else float('nan'))
        out.append({
            'segment': span.point.raw, 'direction': span.direction,
            'rpm_cmd': span.point.rpm,
            't_cmd_nm': span.point.torque_nm,
            'n_samples': len(seg), 'n_at_speed': held,
            'coverage_frac': held / len(seg) if len(seg) else 0.0,
            'frac_at_rest': at_rest,
            'rpm_in_during_legs': physics.rpm_of(in_leg),
            'rpm_in_span_average': physics.rpm_of(travel / dur) if dur else
                                   float('nan'),
        })
    return out


def _fig_creep(creep, cfg):
    fig, ax = plotting.figure(
        f'{cfg["unit_label"]}  --  creep ratio vs output torque',
        'creep = 1 - (ratio x output speed) / input speed, on the '
        'constant-speed legs; ratio from the no-load sweep')
    speeds = sorted({c['rpm_cmd'] for c in creep})
    colors = _speed_colors(speeds)
    for rpm in speeds:
        sel = sorted((c for c in creep if c['rpm_cmd'] == rpm
                      and c['creep_pct'] is not None),
                     key=lambda c: abs(c['t_cmd_nm']))
        if not sel:
            continue
        ax.plot([abs(c['t_cmd_nm']) for c in sel],
                [c['creep_pct'] for c in sel],
                marker='o', ms=4, lw=1.4, color=colors[rpm],
                label=f'{rpm:g} rpm in')
    ax.axhline(CREEP_LIMIT_PCT, ls='--', lw=1.4, color=plotting.BACKDRIVE_C)
    ax.text(0.01, CREEP_LIMIT_PCT, f'  customer limit {CREEP_LIMIT_PCT:g}%',
            transform=ax.get_yaxis_transform(), va='bottom', fontsize=8,
            color=plotting.BACKDRIVE_C)
    ax.set_xlabel('commanded output torque (Nm)')
    ax.set_ylabel('creep ratio (%)')
    ax.legend(title='input speed', fontsize=8)
    return plotting.finish(fig)


def _fig_zero(detail, cfg):
    fig, ax = plotting.figure(
        f'{cfg["unit_label"]}  --  torque cell zero and train drag',
        'from the no-load traverses: a zero offset survives a direction flip, '
        'Coulomb drag reverses with it')
    chans = list(detail)
    x = np.arange(len(chans))
    ax.bar(x - 0.18, [detail[c]['offset_nm'] for c in chans], width=0.34,
           label='zero offset (does not reverse)', color=plotting.FORWARD_C)
    ax.bar(x + 0.18, [detail[c]['drag_nm'] for c in chans], width=0.34,
           label='Coulomb drag (reverses)', color='#7f7f7f')
    ax.set_xticks(x)
    ax.set_xticklabels(chans)
    ax.set_ylabel('Nm (in that cell\'s own frame)')
    ax.legend(fontsize=8)
    plotting.stamp(ax, _zero_note(cfg))
    return plotting.finish(fig)


def _fig_coverage(points, cfg):
    """Which of the customer's grid points this campaign actually reached."""
    fig, ax = plotting.figure(
        f'{cfg["unit_label"]}  --  efficiency grid coverage',
        'one mark per measured operating point; the cap is what the sweep '
        'stopped at')
    for flow, marker in (('forward', 'o'), ('backdrive', 'x')):
        sel = [p for p in points if p['flow'] == flow]
        if not sel:
            continue
        ax.scatter([p['t_cmd_nm'] for p in sel], [p['rpm'] for p in sel],
                   marker=marker, s=28, color=plotting.DIR_COLOR[flow],
                   label=plotting.DIR_LABEL[flow])
    cap = cfg.get('torque_cap_nm')
    if cap:
        ax.axvline(cap, ls='--', lw=1.2, color='#444444')
        ax.text(cap, 0.02, f' customer limit {cap:g} Nm', rotation=90,
                transform=ax.get_xaxis_transform(), fontsize=8, va='bottom')
    ax.set_yscale('log')
    ax.set_yticks(sorted({p['rpm'] for p in points}))
    ax.get_yaxis().set_major_formatter(plotting.matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel('commanded output torque (Nm)')
    ax.set_ylabel('input speed (rpm)')
    ax.legend(fontsize=8)
    return plotting.finish(fig)


def _zero_note(cfg):
    off = (cfg.get('cell_offsets') or ({}, {}))[0].get('load_torque')
    if not cfg.get('correct_zero', True):
        return 'output cell zero NOT corrected'
    if off is None:
        return 'no no-load traverses: output cell zero could not be corrected'
    return f'output cell zero corrected by {off:+.3f} Nm'


def _findings(res, legs, usable, points, creep, detail, cfg, cov=(), hyst=None):
    n_drop = len(legs) - len(usable)
    if n_drop:
        res.add('info', 'legs_dropped',
                f'{n_drop} of {len(legs)} legs dropped: the two shaft powers '
                'disagreed in sign, which is a coasting or mis-zeroed leg '
                'rather than a transmitting one')
    hot = [p for p in points if p['eta'] > ETA_SANE_MAX]
    if hot:
        worst = max(hot, key=lambda p: p['eta'])
        res.add('warn', 'eta_over_unity',
                f'{len(hot)} operating point(s) came out over 100% efficient '
                f'(worst {worst["eta"] * 100:.0f}% at {worst["rpm"]:g} rpm, '
                f'{worst["t_out_nm"]:.0f} Nm) -- a torque-cell zero or scale '
                'error, not a drive; treat the low-torque end as unproven')
    if hyst and hyst.get('n_keys'):
        lo, hi = hyst['range']
        if hyst['applied']:
            res.add('info', 'torque_asymmetry_applied',
                    f'output cell direction-dependent bias h = '
                    f'{hyst["h"]:+.3f} Nm from {hyst["n_keys"]} grid cells '
                    f'(halves {hyst["h_pos"]:+.3f} / {hyst["h_neg"]:+.3f}, '
                    f'spread {hyst["spread"]:.3f} Nm); removed as '
                    'h*sign(w_out)')
        else:
            why = []
            if not hyst['halves_agree']:
                why.append(f'the two independent halves differ by '
                           f'{hyst["disagree"]:.3f} Nm')
            if not hyst['consistent']:
                why.append(f'it is not constant across the grid '
                           f'(sd {hyst["spread"]:.3f} Nm, {lo:+.2f} to '
                           f'{hi:+.2f} Nm)')
            res.add('warn', 'torque_asymmetry_not_applied',
                    f'a direction-dependent output-cell bias was estimated at '
                    f'{hyst["h"]:+.3f} Nm over {hyst["n_keys"]} grid cells but '
                    'NOT removed, because ' + ' and '.join(why)
                    + '. The forward and back-driven curves therefore still '
                      'carry whatever part of their difference is cell bias '
                      'rather than drive behaviour; see '
                      'efficiency__torque_asymmetry.csv')
    off = (cfg.get('cell_offsets') or ({}, {}))[0].get('load_torque')
    step = cfg.get('torque_step_nm', 5.0)
    if off and abs(off) > 0.1 * step:
        res.add('warn', 'cell_zero',
                f'output cell zero is {off:+.3f} Nm, '
                f'{abs(off) / step * 100:.0f}% of the {step:g} Nm torque step '
                + ('(corrected here)' if cfg.get('correct_zero', True)
                   else '(NOT corrected -- correct_zero is off)'))
    if detail.get('load_torque'):
        drag = detail['load_torque']['drag_nm']
        res.add('info', 'train_drag',
                f'output-referred Coulomb drag of the whole train is '
                f'{drag:.2f} Nm; it is charged to the drive in every number '
                'here, and it is '
                f'{drag / step * 100:.0f}% of the {step:g} Nm step')
    poor = [c for c in cov if c['coverage_frac'] < LOW_COVERAGE_FRAC]
    if poor:
        # Grouped by direction as well as speed. Back-driven spans mask on the
        # OUTPUT shaft, so their input speed during a leg says whether the
        # input followed -- a different quantity from the forward case, and
        # pooling the two produces a '20 rpm commanded, 3 rpm measured' line
        # that describes neither.
        by_key = {}
        for c in poor:
            by_key.setdefault((c['direction'], c['rpm_cmd']), []).append(c)
        parts = []
        for (d, r), cs in sorted(by_key.items()):
            parts.append(
                f'{d} {r:g} rpm: on speed '
                f'{np.median([c["coverage_frac"] for c in cs]) * 100:.0f}%, '
                f'at rest '
                f'{np.median([c["frac_at_rest"] for c in cs]) * 100:.0f}% of '
                f'the span; input at '
                f'{np.median([c["rpm_in_during_legs"] for c in cs]):.0f} rpm '
                f'while on speed, '
                f'{np.median([c["rpm_in_span_average"] for c in cs]):.0f} rpm '
                'averaged over the span')
        res.add('warn', 'low_speed_coverage',
                f'{len(poor)} of {len(cov)} operating points held the '
                f'commanded speed for less than '
                f'{LOW_COVERAGE_FRAC * 100:.0f}% of their segment -- '
                + '; '.join(parts)
                + '. Where the at-rest fraction is large the segment is '
                  'dominated by the plan\'s lead-in, settle and turnaround '
                  'dwells; where it is small the shaft was moving but not at '
                  'the commanded speed')
    if creep:
        lost = [c for c in creep if c['n_tracking'] < c['n_legs']]
        if lost:
            by_dir = {}
            for c in lost:
                by_dir.setdefault(c['direction'], 0)
                by_dir[c['direction']] += 1
            res.add('warn', 'ratio_not_tracked',
                    'the shafts stopped tracking the gear ratio on at least '
                    'one leg of ' + ', '.join(
                        f'{n} {d} point(s)' for d, n in sorted(by_dir.items()))
                    + ' -- on those legs one shaft turned and the other did '
                      'not follow, so no creep figure is quoted for them')
        rated = [c for c in creep if c['creep_pct'] is not None]
        if rated:
            over = [c for c in rated if c['creep_pct'] > CREEP_LIMIT_PCT]
            worst = max(rated, key=lambda c: c['creep_pct'])
            res.add('warn' if over else 'info', 'creep',
                    f'creep ratio peaks at {worst["creep_pct"]:.2f}% '
                    f'({worst["direction"]}, {worst["rpm_cmd"]:g} rpm, '
                    f'{worst["t_cmd_nm"]:+g} Nm); {len(over)} of {len(rated)} '
                    f'levels exceed the customer\'s {CREEP_LIMIT_PCT:g}% '
                    'limit')
    cap = cfg.get('torque_cap_nm')
    if cap and points:
        top = max(p['t_cmd_nm'] for p in points)
        if top < cap:
            res.add('warn', 'range_short',
                    f'sweep reached {top:g} Nm of the customer\'s {cap:g} Nm '
                    'range; the top of the requested band is not covered')
    flows = {p['flow'] for p in points}
    best = max((p for p in points if p['flow'] == 'forward'),
               key=lambda p: p['eta'], default=None)
    res.summary = (
        f'{len(points)} operating point(s) from {len(legs)} legs across '
        f'{len(flows)} power-flow direction(s)'
        + (f'; peak forward efficiency {best["eta"] * 100:.1f}% at '
           f'{best["rpm"]:g} rpm, {best["t_out_nm"]:.0f} Nm' if best else ''))
