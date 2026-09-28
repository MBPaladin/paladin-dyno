"""Velocity ramp up, and the torque ripple the customer reads off it.

Customer ask (1.2.2 Paladin Test Request, 'Velocity ramp up'):

    0 < w_in < 300 rpm, dw = 30;  300 < w_in < 3600 rpm, dw = 300
    Calculate robust pk-pk torque ripple for all steady state no-load
    velocity cycles
    Back-driving: same steps based on INPUT-side speed, and the output
    position monitored because of the physical endstop

Both halves come off the same spans, so they are one analyzer. The steps are
one segment each (`V0300`), and each segment is a bipolar sawtooth, so a step
holds its speed on several separate legs -- one per half-traverse. Ripple is
measured per leg and then pooled, which is what makes the spread across legs
available as an honest error bar rather than a single number with no context.

'No-load' is the plan's word, not a measurement: the far shaft is commanded to
0 Nm, but the train's own drag still reads on both cells. The drag curve is
plotted beside the ripple for exactly that reason -- it is the offset the
ripple sits on, and at these speeds it is not small.
"""

import numpy as np

from .. import naming, physics, plotting
from ..result import Result

# Ripple is a property of the INPUT shaft -- it is the drive's own torque
# variation, and the 43:1 reduction smears it at the output. The output cell is
# still reported, because a customer reading an output-referred number wants to
# see it rather than be told it was divided.
RIPPLE_CHANNEL = 'input_torque'


def analyze(ds, cfg):
    res = Result('velocity_ramp', 'Velocity ramp up and no-load torque ripple')
    spans = ds.select(kind=naming.VELOCITY)
    if not spans:
        res.add('warn', 'no_data', 'no velocity-ramp segments in this campaign')
        return res

    ratio, _ = ds.ratio(cfg.get('ratio', 43.88))
    rows = []
    for span in sorted(spans, key=lambda s: (s.direction, s.point.rpm)):
        rows.extend(_step_rows(span, ratio))
    if not rows:
        res.add('warn', 'no_plateaus',
                'velocity-ramp segments hold no measurable constant-speed leg')
        return res

    res.tables.append(('velocity_ramp__per_leg', plotting.csv(
        [[r[k] for k in _COLS] for r in rows], _COLS)))

    by_step = _pool(rows)
    res.tables.append(('velocity_ramp__per_step', plotting.csv(
        [[s[k] for k in _STEP_COLS] for s in by_step], _STEP_COLS)))

    res.figures.append(('ripple_vs_speed', _fig_ripple(by_step, cfg)))
    res.figures.append(('speed_tracking', _fig_tracking(by_step, cfg)))

    # One drag figure per direction, each on the cell of the shaft that was
    # driven and against that shaft's own speed. Two figures rather than one
    # because the two are read in frames 43:1 apart and nothing here is
    # referred through the ratio.
    fits = {}
    for direction in ('forward', 'backdrive'):
        fig, fit = _fig_drag(by_step, cfg, direction)
        if fig is None:
            continue
        res.figures.append((f'no_load_drag_{direction}', fig))
        fits[direction] = fit
    if fits:
        res.tables.append(('velocity_ramp__drag_fit',
                           plotting.csv(_drag_fit_rows(fits),
                                        _DRAG_FIT_COLS)))
        res.tables.append(('velocity_ramp__drag_vs_speed',
                           plotting.csv(_drag_speed_rows(by_step),
                                        _DRAG_SPEED_COLS)))

    endstop = _fig_output_travel(spans, cfg)
    if endstop is not None:
        res.figures.append(('output_travel', endstop))

    _findings(res, by_step, spans, cfg)
    res.metrics['drag'] = {
        direction: {k: fit[k] for k in
                    ('coulomb_nm', 'viscous_nm_per_krpm', 'zero_nm',
                     'rpm_max', 'drag_at_rpm_max_nm', 'n_points', 'r2')
                    if k in fit}
        for direction, fit in fits.items()}
    res.metrics['steps'] = {
        f'{s["direction"]}_{s["rpm_cmd"]:.0f}rpm': {
            'ripple_pk_pk_nm': s['ripple_nm'],
            'drag_in_nm': s['t_in_nm'],
            'speed_error_pct': s['speed_err_pct'],
            'n_legs': s['n_legs'],
        } for s in by_step}
    return res


_COLS = ['direction', 'rpm_cmd', 'segment', 'source', 'leg', 'sign',
         'n', 'duration_s', 'rpm_in_meas', 'rpm_out_meas',
         't_in_mean_nm', 't_out_mean_nm', 'ripple_in_pk_pk_nm',
         'ripple_out_pk_pk_nm', 'ripple_in_rms_nm']

_STEP_COLS = ['direction', 'rpm_cmd', 'n_legs', 'rpm_in_meas', 'speed_err_pct',
              'ripple_nm', 'ripple_spread_nm', 'ripple_lo_nm', 'ripple_hi_nm',
              'ripple_out_nm', 't_in_nm', 't_out_nm']

# The drag split gets its own two CSVs rather than more columns on the per-step
# table: per-step is the customer's ripple table in CSV form, and the signed
# means and the fit are a different question asked of the same legs.
_DRAG_FIT_COLS = ['direction', 'cell', 'shaft', 'n_points', 'coulomb_nm',
                  'viscous_nm_per_krpm', 'cell_zero_nm', 'rpm_max',
                  'drag_at_rpm_max_nm', 'r2']

_DRAG_SPEED_COLS = ['direction', 'rpm_cmd', 'rpm_in_meas', 'rpm_out_meas',
                    't_in_pos_nm', 't_in_neg_nm', 't_out_pos_nm',
                    't_out_neg_nm', 't_in_pos_sd_nm', 't_in_neg_sd_nm',
                    't_out_pos_sd_nm', 't_out_neg_sd_nm']


def _drag_fit_rows(fits):
    rows = []
    for direction in ('forward', 'backdrive'):
        fit = fits.get(direction)
        if not fit:
            continue
        view = _DRAG_VIEW[direction]
        rows.append([direction, view['cell'], view['shaft'],
                     fit.get('n_points'), fit.get('coulomb_nm'),
                     fit.get('viscous_nm_per_krpm'), fit.get('zero_nm'),
                     fit.get('rpm_max'), fit.get('drag_at_rpm_max_nm'),
                     fit.get('r2')])
    return rows


def _drag_speed_rows(steps):
    return [[s[k] for k in _DRAG_SPEED_COLS] for s in steps]


def _step_rows(span, ratio):
    """One row per constant-speed leg of one velocity step."""
    seg = span.seg
    drive = ('dut_velocity' if span.direction == 'forward' else 'load_velocity')
    target = physics.drive_target(span.point.rpm, span.direction, ratio)
    out = []
    for i, (sl, sign) in enumerate(physics.legs(seg, drive, target=target)):
        wi = seg['dut_velocity'][sl]
        wo = seg['load_velocity'][sl]
        ti = seg['input_torque'][sl] if seg.has('input_torque') else np.array([])
        to = seg['load_torque'][sl] if seg.has('load_torque') else np.array([])
        n = sl.stop - sl.start
        out.append({
            'direction': span.direction,
            'rpm_cmd': span.point.rpm,
            'segment': span.point.raw,
            'source': span.source,
            'leg': i, 'sign': sign, 'n': n,
            'duration_s': n * seg.dt,
            'rpm_in_meas': physics.rpm_of(float(np.nanmean(np.abs(wi)))),
            'rpm_out_meas': physics.rpm_of(float(np.nanmean(np.abs(wo)))),
            't_in_mean_nm': float(np.nanmean(ti)) if ti.size else float('nan'),
            't_out_mean_nm': float(np.nanmean(to)) if to.size else float('nan'),
            'ripple_in_pk_pk_nm': physics.robust_pk_pk(ti),
            'ripple_out_pk_pk_nm': physics.robust_pk_pk(to),
            'ripple_in_rms_nm': (float(np.nanstd(ti)) if ti.size
                                 else float('nan')),
        })
    return out


def _pool(rows):
    """Collapse legs into one entry per (direction, commanded speed).

    The median across legs is the quoted ripple and the leg-to-leg spread is
    its uncertainty: a step whose two travel directions disagree is telling you
    the ripple is direction-dependent, which a single pooled pk-pk would bury.
    """
    buckets = {}
    for r in rows:
        buckets.setdefault((r['direction'], r['rpm_cmd']), []).append(r)
    out = []
    for (direction, rpm), rs in sorted(buckets.items(),
                                       key=lambda kv: (kv[0][0], kv[0][1])):
        rip = [r['ripple_in_pk_pk_nm'] for r in rs
               if np.isfinite(r['ripple_in_pk_pk_nm'])]
        meas = float(np.median([r['rpm_in_meas'] for r in rs]))
        out.append({
            'direction': direction, 'rpm_cmd': rpm, 'n_legs': len(rs),
            'rpm_in_meas': meas,
            'speed_err_pct': (meas - rpm) / rpm * 100.0 if rpm else float('nan'),
            'ripple_nm': float(np.median(rip)) if rip else float('nan'),
            'ripple_spread_nm': (float(np.ptp(rip)) if len(rip) > 1 else 0.0),
            # Kept as the actual extremes, not as a half-width about the
            # median: ripple is a magnitude, and a symmetric bar drawn around
            # a skewed set of legs reaches below zero, which is not a value a
            # pk-pk can take.
            'ripple_lo_nm': float(np.min(rip)) if rip else float('nan'),
            'ripple_hi_nm': float(np.max(rip)) if rip else float('nan'),
            'ripple_out_nm': float(np.median(
                [r['ripple_out_pk_pk_nm'] for r in rs])),
            # Drag is signed against travel, so pooling the raw mean would
            # cancel it. The magnitude is what the drag curve wants.
            't_in_nm': float(np.median([abs(r['t_in_mean_nm']) for r in rs])),
            't_out_nm': float(np.median([abs(r['t_out_mean_nm']) for r in rs])),
            'rpm_out_meas': float(np.median([r['rpm_out_meas'] for r in rs])),
            # The same means kept SIGNED and split by travel direction, which
            # is what separates the cell's zero from the train's friction: the
            # zero survives a direction flip and the friction reverses with it.
            # Pooled to a magnitude above, these are unrecoverable.
            't_in_pos_nm': _signed(rs, 't_in_mean_nm', +1),
            't_in_neg_nm': _signed(rs, 't_in_mean_nm', -1),
            't_out_pos_nm': _signed(rs, 't_out_mean_nm', +1),
            't_out_neg_nm': _signed(rs, 't_out_mean_nm', -1),
            't_in_pos_sd_nm': _signed_sd(rs, 't_in_mean_nm', +1),
            't_in_neg_sd_nm': _signed_sd(rs, 't_in_mean_nm', -1),
            't_out_pos_sd_nm': _signed_sd(rs, 't_out_mean_nm', +1),
            't_out_neg_sd_nm': _signed_sd(rs, 't_out_mean_nm', -1),
        })
    return out


def _signed(rows, key, sign):
    """Median of `key` over the legs travelling in `sign`, signed."""
    vals = [r[key] for r in rows
            if r['sign'] * sign > 0 and np.isfinite(r[key])]
    return float(np.median(vals)) if vals else float('nan')


def _signed_sd(rows, key, sign):
    """Spread of `key` across the legs travelling in `sign`.

    The standard deviation of the PER-LEG MEANS, not of the samples inside a
    leg: within-leg scatter is torque ripple, which the section above already
    reports, and quoting it here would draw a confidence band an order of
    magnitude wider than the repeatability it is meant to show. What this
    measures is whether the step gave the same answer each time it was held.

    Zero legs or one leg gives no spread -- 0.0, not nan, so the band simply
    collapses to the line rather than putting a hole in it.
    """
    vals = [r[key] for r in rows
            if r['sign'] * sign > 0 and np.isfinite(r[key])]
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def _by_direction(steps):
    for direction in ('forward', 'backdrive'):
        sel = [s for s in steps if s['direction'] == direction]
        if sel:
            yield direction, sel


def _fig_ripple(steps, cfg):
    fig, ax = plotting.figure(
        f'{cfg["unit_label"]}  --  no-load torque ripple vs speed',
        'robust pk-pk (0.5-99.5 percentile) on the input cell; '
        'the band spans the legs of each step')
    for direction, sel in _by_direction(steps):
        x = [s['rpm_cmd'] for s in sel]
        y = [s['ripple_nm'] for s in sel]
        # The band is the actual leg extremes, not a symmetric spread about the
        # median. Ripple is a magnitude, and a +/- half-width drawn around a
        # skewed set of legs reaches below zero, which is not a value a pk-pk
        # can take.
        plotting.band(ax, x, [s['ripple_lo_nm'] for s in sel],
                      [s['ripple_hi_nm'] for s in sel],
                      plotting.DIR_COLOR[direction])
        ax.plot(x, y, marker='o', ms=4, lw=1.4,
                color=plotting.DIR_COLOR[direction],
                label=plotting.DIR_LABEL[direction])
    ax.set_xlabel('commanded input speed (rpm)')
    ax.set_ylabel('input torque ripple, robust pk-pk (Nm)')
    ax.set_ylim(bottom=0)
    ax.legend()
    plotting.stamp(ax, 'far shaft commanded to 0 Nm; train drag not removed')
    return plotting.finish(fig)


def _fig_tracking(steps, cfg):
    fig, ax = plotting.figure(
        f'{cfg["unit_label"]}  --  speed tracking',
        'measured input speed against the commanded step')
    lo = min(s['rpm_cmd'] for s in steps)
    hi = max(s['rpm_cmd'] for s in steps)
    ax.plot([lo, hi], [lo, hi], ls='--', lw=1, color='#888888', label='ideal')
    for direction, sel in _by_direction(steps):
        ax.plot([s['rpm_cmd'] for s in sel], [s['rpm_in_meas'] for s in sel],
                marker='o', ms=4, lw=1.4, color=plotting.DIR_COLOR[direction],
                label=plotting.DIR_LABEL[direction])
    ax.set_xlabel('commanded input speed (rpm)')
    ax.set_ylabel('measured input speed (rpm)')
    ax.legend()
    return plotting.finish(fig)


# Which cell and which shaft speed each direction's drag is read on. The shaft
# that was DRIVEN is the one whose cell carries the drag as a torque the motor
# actually had to supply, so forward driving is read on the input and
# back-driving on the output. Neither is referred through the gear ratio: a
# ratio-referred output number is a derived quantity, and the point of these
# two figures is to show the measurement each cell made.
_DRAG_VIEW = {
    'forward':   {'cell': 'input_torque', 'shaft': 'input (high-speed)',
                  'pos': 't_in_pos_nm', 'neg': 't_in_neg_nm',
                  'pos_sd': 't_in_pos_sd_nm', 'neg_sd': 't_in_neg_sd_nm',
                  'rpm': 'rpm_in_meas'},
    'backdrive': {'cell': 'load_torque', 'shaft': 'output (low-speed)',
                  'pos': 't_out_pos_nm', 'neg': 't_out_neg_nm',
                  'pos_sd': 't_out_pos_sd_nm', 'neg_sd': 't_out_neg_sd_nm',
                  'rpm': 'rpm_out_meas'},
}

_KRPM = 1000.0


def _drag_split(sel, view):
    """Separate the cell zero from Coulomb and viscous friction against speed.

    A no-load constant-speed leg reads

        T(w) = z + Tc*sgn(w) + b*w

    -- the cell's own zero `z`, which does not care which way the shaft is
    turning; a Coulomb term `Tc` that reverses with travel; and a viscous term
    `b*w` that grows with speed. Fitting the positive-travel and
    negative-travel points as two straight lines against SIGNED speed recovers
    all three without a tare and without a model of the flange: the two
    intercepts are z+Tc and z-Tc, so their half-difference is Coulomb and their
    mean is the zero, and the slopes are the viscous term.

    This is the same decomposition `physics.cell_offsets` makes, except that it
    is made against speed instead of pooled over every leg in the campaign, so
    a viscous term is no longer folded into the Coulomb number.
    """
    pts = [(s[view['rpm']], s[view['pos']], s[view['neg']]) for s in sel
           if np.isfinite(s[view['rpm']])]
    fit = {'n_points': 0}
    for tag, col in (('pos', 1), ('neg', 2)):
        sgn = 1.0 if tag == 'pos' else -1.0
        xy = [(sgn * p[0], p[col]) for p in pts if np.isfinite(p[col])]
        if len(xy) >= 2:
            x = np.array([a for a, _ in xy])
            y = np.array([b for _, b in xy])
            slope, icept = np.polyfit(x, y, 1)
            fit[tag] = {'slope': float(slope), 'intercept': float(icept),
                        'n': len(xy)}
            fit['n_points'] += len(xy)
    if 'pos' not in fit or 'neg' not in fit:
        # One travel direction only. Coulomb and the cell zero are the same
        # number in that case and cannot be told apart, so neither is reported.
        return fit

    # Writing the flanks out, with `k` = +1 on a cell that reads drag positive
    # when the shaft turns positive and -1 on one that reads it negative:
    #
    #   x > 0:  T = z + k*Tc + k*b*x      intercept z + k*Tc, slope  k*b
    #   x < 0:  T = z - k*Tc + k*b*x      intercept z - k*Tc, slope  k*b
    #
    # so Tc is half the size of the intercept gap, z is the intercept mean,
    # and the cell's sense `k` is the SIGN of the intercept gap. That sign is
    # what recovers b from the flank slopes -- the input and output cells here
    # do not agree on it, so taking |slope| (which is what the running-torque
    # processor does, its cells all reading one way) would report a drag that
    # rises with speed on a shaft whose drag measurably falls.
    gap = fit['pos']['intercept'] - fit['neg']['intercept']
    sense = 1.0 if gap >= 0 else -1.0
    fit['cell_sense'] = sense
    fit['coulomb_nm'] = abs(gap) / 2.0
    fit['zero_nm'] = (fit['pos']['intercept'] + fit['neg']['intercept']) / 2.0
    fit['viscous_nm_per_rpm'] = sense * (fit['pos']['slope']
                                         + fit['neg']['slope']) / 2.0
    fit['viscous_nm_per_krpm'] = fit['viscous_nm_per_rpm'] * _KRPM

    # How well a straight line actually describes the drag, scored against the
    # per-speed reversing component rather than against the raw flank points:
    # a flank's own scatter is dominated by the cell zero, which this model
    # does not claim to predict. Back-driving is the case that needs the
    # number -- its speed holding is poor and its drag is far from linear.
    w = np.array([p[0] for p in pts])
    drag = np.array([sense * (p[1] - p[2]) / 2.0 for p in pts])
    keep = np.isfinite(w) & np.isfinite(drag)
    if keep.sum() >= 3:
        model = fit['coulomb_nm'] + fit['viscous_nm_per_rpm'] * w[keep]
        ss_res = float(np.sum((drag[keep] - model) ** 2))
        ss_tot = float(np.sum((drag[keep] - np.mean(drag[keep])) ** 2))
        fit['r2'] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # The non-extrapolated anchor for the viscous term. A Nm/krpm slope read
    # off an output shaft that never passed 90 rpm is a unit, not a measurement
    # at 1000 rpm, and the total at the top speed actually reached is the
    # number that says how much drag was really seen.
    fit['rpm_max'] = max(p[0] for p in pts)
    fit['drag_at_rpm_max_nm'] = (fit['coulomb_nm']
                                 + fit['viscous_nm_per_rpm'] * fit['rpm_max'])
    return fit


def _fig_drag(steps, cfg, direction):
    """Signed no-load cell reading against signed speed, with the two flanks.

    Shaped after the running-torque processor's drag figure
    (`analysis.processors.running_torque._fig_drag`): the same Coulomb-plus-
    viscous split, read here off the velocity ramp's constant-speed legs rather
    than off a triangular sweep.
    """
    view = _DRAG_VIEW[direction]
    sel = [s for s in steps if s['direction'] == direction]
    if not sel:
        return None, {}
    fit = _drag_split(sel, view)

    fig, ax = plotting.figure(
        f'{cfg["unit_label"]}  --  no-load drag vs speed, '
        f'{plotting.DIR_LABEL[direction]}',
        f'mean {view["cell"]} reading per constant-speed leg, against the '
        f'{view["shaft"]} shaft it was measured on')
    c = plotting.DIR_COLOR[direction]
    for tag, marker, lbl in (('pos', 'o', 'travel positive'),
                             ('neg', 's', 'travel negative')):
        sgn = 1.0 if tag == 'pos' else -1.0
        use = [s for s in sel if np.isfinite(s[view[tag]])]
        xs = [sgn * s[view['rpm']] for s in use]
        ys = [s[view[tag]] for s in use]
        sd = [s.get(view[tag + '_sd'], 0.0) or 0.0 for s in use]
        if not xs:
            continue
        # +/-1 SD of the per-leg means, as a band behind the line. Labelled
        # once: both flanks draw the same kind of band, and two legend entries
        # saying so push the fit lines off the bottom.
        ya, sa = np.asarray(ys), np.asarray(sd)
        plotting.band(ax, xs, ya - sa, ya + sa, c,
                      label=('$\\pm$1 SD of the per-leg means'
                             if tag == 'pos' else None))
        ax.plot(xs, ys, marker=marker, ms=4, lw=1.2, ls='-', color=c,
                alpha=1.0 if tag == 'pos' else 0.55, label=lbl)
        if tag in fit:
            line = np.linspace(0.0, max(xs) if tag == 'pos' else min(xs), 50)
            ax.plot(line, fit[tag]['slope'] * line + fit[tag]['intercept'],
                    ls='--', lw=1.4, color='#444444',
                    label=(f'{tag} fit: {fit[tag]["slope"] * _KRPM:+.3f} '
                           f'Nm/krpm, {fit[tag]["intercept"]:+.3f} Nm'))
    if 'coulomb_nm' in fit:
        ax.plot([], [], ' ',
                label=(f'Coulomb {fit["coulomb_nm"]:.3f} Nm, viscous '
                       f'{fit["viscous_nm_per_krpm"]:+.3f} Nm/krpm'
                       + (f' (R2 {fit["r2"]:.2f})' if 'r2' in fit else '')))
        ax.plot([], [], ' ',
                label=(f'drag at {fit["rpm_max"]:.0f} rpm '
                       f'{fit["drag_at_rpm_max_nm"]:.3f} Nm; cell zero '
                       f'{fit["zero_nm"]:+.3f} Nm (does not reverse)'))
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel(f'measured {view["shaft"]} shaft speed, signed by travel '
                  f'direction (rpm)')
    ax.set_ylabel(f'{view["cell"]} (Nm, as read)')
    ax.legend(fontsize=8)
    plotting.stamp(ax, 'far shaft commanded to 0 Nm; cell reading as logged, '
                       'no tare or zero correction applied')
    return plotting.finish(fig), fit


def _fig_output_travel(spans, cfg):
    """Output angle against time for every step, with the endstop window.

    The customer asked for the output position to be monitored on the
    back-driven ramp because the Archimedes drive has an internal endstop. This
    is that monitor: how close each step ran to the window, and which ones the
    window stopped.
    """
    sel = [s for s in spans if s.seg.has('load_position')]
    if not sel:
        return None
    half = cfg.get('position_half_window_rad')
    fig, ax = plotting.figure(
        f'{cfg["unit_label"]}  --  output travel during the velocity ramp',
        'each step, centred; the endstop window is what stops a step early')
    for span in sorted(sel, key=lambda s: (s.direction, s.point.rpm)):
        seg = span.seg
        pos = seg['load_position']
        t = seg['time'] - seg['time'][0]
        ax.plot(t, pos - np.nanmedian(pos[:200]), lw=0.8,
                color=plotting.DIR_COLOR[span.direction], alpha=0.55)
    if half:
        for sgn in (1, -1):
            ax.axhline(sgn * half, ls='--', lw=1.1, color='#444444')
        ax.text(0.01, 0.97, f'position window +/-{half:g} rad',
                transform=ax.transAxes, va='top', fontsize=8)
    ax.set_xlabel('time into step (s)')
    ax.set_ylabel('output angle from step start (rad)')
    for direction in ('forward', 'backdrive'):
        if any(s.direction == direction for s in sel):
            ax.plot([], [], color=plotting.DIR_COLOR[direction],
                    label=plotting.DIR_LABEL[direction])
    ax.legend()
    return plotting.finish(fig)


def _findings(res, steps, spans, cfg):
    want = cfg.get('velocity_steps_rpm')
    for direction, sel in _by_direction(steps):
        got = {s['rpm_cmd'] for s in sel}
        if want:
            missing = sorted(set(want) - got)
            if missing:
                res.add('warn', 'steps_missing',
                        f'{direction}: no usable data at '
                        f'{", ".join(f"{m:g}" for m in missing)} rpm '
                        f'({len(got)} of {len(want)} steps covered)')
        bad = [s for s in sel if abs(s['speed_err_pct']) > 5]
        if bad:
            res.add('warn', 'speed_tracking',
                    f'{direction}: {len(bad)} step(s) missed the commanded '
                    f'speed by more than 5% -- worst '
                    f'{max(bad, key=lambda s: abs(s["speed_err_pct"]))["speed_err_pct"]:+.1f}% '
                    'at '
                    f'{max(bad, key=lambda s: abs(s["speed_err_pct"]))["rpm_cmd"]:g} rpm')
        lonely = [s for s in sel if s['n_legs'] < 2]
        if lonely:
            res.add('info', 'single_leg',
                    f'{direction}: {len(lonely)} step(s) hold only one '
                    'constant-speed leg, so their ripple has no spread')
    top = max(steps, key=lambda s: s['ripple_nm'] if np.isfinite(s['ripple_nm'])
              else -1)
    res.summary = (
        f'{len(steps)} velocity step(s) across '
        f'{len({s["direction"] for s in steps})} direction(s); ripple peaks at '
        f'{top["ripple_nm"]:.3f} Nm pk-pk at {top["rpm_cmd"]:g} rpm '
        f'({plotting.DIR_LABEL[top["direction"]]})')
