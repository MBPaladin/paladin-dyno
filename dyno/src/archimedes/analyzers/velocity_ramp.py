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
    res.figures.append(('no_load_drag', _fig_drag(by_step, cfg)))
    endstop = _fig_output_travel(spans, cfg)
    if endstop is not None:
        res.figures.append(('output_travel', endstop))

    _findings(res, by_step, spans, cfg)
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
        })
    return out


def _by_direction(steps):
    for direction in ('forward', 'backdrive'):
        sel = [s for s in steps if s['direction'] == direction]
        if sel:
            yield direction, sel


def _fig_ripple(steps, cfg):
    fig, ax = plotting.figure(
        f'{cfg["unit_label"]}  --  no-load torque ripple vs speed',
        'robust pk-pk (0.5-99.5 percentile) on the input cell, per steady '
        'constant-speed leg; bars span the legs')
    for direction, sel in _by_direction(steps):
        x = [s['rpm_cmd'] for s in sel]
        y = [s['ripple_nm'] for s in sel]
        e = [[max(s['ripple_nm'] - s['ripple_lo_nm'], 0) for s in sel],
             [max(s['ripple_hi_nm'] - s['ripple_nm'], 0) for s in sel]]
        ax.errorbar(x, y, yerr=e, marker='o', ms=4, capsize=3, lw=1.4,
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


def _fig_drag(steps, cfg):
    fig, ax = plotting.figure(
        f'{cfg["unit_label"]}  --  no-load running torque vs speed',
        'magnitude of the mean cell reading on a constant-speed leg')
    for direction, sel in _by_direction(steps):
        c = plotting.DIR_COLOR[direction]
        ax.plot([s['rpm_cmd'] for s in sel], [s['t_in_nm'] for s in sel],
                marker='o', ms=4, lw=1.4, color=c,
                label=f'{plotting.DIR_LABEL[direction]}: input cell')
        ax.plot([s['rpm_cmd'] for s in sel],
                [s['t_out_nm'] / max(cfg['ratio'], 1e-9) for s in sel],
                marker='s', ms=4, lw=1.2, ls='--', color=c,
                label=f'{plotting.DIR_LABEL[direction]}: output cell / ratio')
    ax.set_xlabel('commanded input speed (rpm)')
    ax.set_ylabel('input-referred running torque (Nm)')
    ax.legend(fontsize=8)
    plotting.stamp(ax, 'output cell carries its own zero offset -- see the '
                       'cell-zero figure')
    return plotting.finish(fig)


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
