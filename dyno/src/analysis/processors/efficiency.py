"""Gearbox efficiency from a velocity x output-torque grid sweep.

The experiment: the input motor holds a velocity setpoint while the output motor
holds a torque setpoint, and the grid walks both. Every logged setpoint span is
one steady dwell at one operating point, with an input torque cell on the motor
shaft and an output torque cell on the load shaft.

Efficiency is a power ratio, computed per dwell from the two cells and the two
encoders:

    P_in  = T_in  * w_in          P_out = T_out * w_out
    forward   (P_in > 0):  eta = P_out / P_in
    backdrive (P_in < 0):  eta = P_in  / P_out

Both signs of output torque appear at every speed, so every operating point is
measured in both power-flow directions: the load either opposes the motion
(forward, the motor drives the load through the reduction) or assists it
(backdrive, the load overhauls the gearbox). They are different numbers and are
mapped separately -- a reduction that is 90% efficient forward can be far worse
backdriven, or refuse to backdrive at all.

The load-invariant form of the same measurement is the output-referred loss
torque, which is what the loss-model fit and most of the diagnostics use:

    P_loss = P_in - P_out = w_out * (ratio * T_in - T_out)   [>= 0, always]
    L      = P_loss / |w_out|                                [Nm at the output]

CELL ZEROS ARE THE WHOLE BALLGAME AT LIGHT LOAD. L is a difference of two large
numbers -- at 5 Nm of output torque on a 500 Nm cell it is the difference
between two readings that are each ~1% of full scale -- so a fraction of a Nm of
zero offset on either cell moves light-load efficiency by tens of points. This
log has no unloaded rest window to tare against, so the zeros are recovered from
the grid's own mirror symmetry instead: the physical torques at (+w, +T) and
(-w, -T) are equal and opposite, so whatever survives averaging that pair is
sensor offset and not torque. That assumes the machine itself is
direction-symmetric, which is exactly what the CW/CCW asymmetry figure is for.

A constant zero is not the whole story on the output cell. That cell also
carries a bias whose sign flips with rotation direction -- Coulomb friction
parasitic to the measurement path, confirmed at the bench:

    T_out_measured = T_true + z + h * sign(w_out)

The mirror tare above averages (+w, +T) against (-w, -T), reading (T + z + h)
and (-T + z - h). Their mean is z, so h cancels IN THE ZERO ESTIMATE while
surviving untouched in every individual dwell. Removing it needs the other
symmetry of the grid: group dwells by the sign of the true output torque, which
mode and rotation fix between them, and the two members of a group differ only
by 2h -- the torque, the zero, the loss and any loss asymmetry are common to
both and cancel. h is therefore measured from the (mode, rotation) sign
structure alone, never from the loss or the forward/backdrive split, which is
what keeps correcting it from being circular.

Two independent consistency checks decide whether the resulting numbers can be
quoted, and both are reported as findings rather than folded away:

  * the gear ratio implied by the two encoders, against the configured one; and
  * the feasible-offset interval. P_loss >= 0 must hold at every dwell, which
    for a constant combined cell offset c = ratio*z_in - z_out is one linear
    constraint per dwell. Their intersection is an interval that any honest
    calibration must lie inside. When it comes out empty, no pair of cell zeros
    can reconcile the two sensors, and its width is a lower bound on the
    cross-calibration error in output Nm.
"""

import numpy as np

from ..processor import Applicability, Processor, Result
from ..registry import register

_MODES = ('forward', 'backdrive')


def _linfit(x, y):
    """Linear fit y = slope*x + intercept; returns (slope, intercept, r2)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 2 or np.ptp(x) == 0:
        return float('nan'), float('nan'), float('nan')
    slope, intercept = np.polyfit(x, y, 1)
    ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return (float(slope), float(intercept),
            1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'))


def _sd(v):
    v = np.asarray(v, float)
    return float(np.nanstd(v, ddof=1)) if np.sum(np.isfinite(v)) > 1 else float('nan')


@register
class Efficiency(Processor):
    name = 'efficiency'
    title = 'Gearbox Efficiency Map'
    description = (
        'Maps gearbox efficiency over a grid of input speed and output torque. '
        'Computes a power-ratio efficiency per dwell, separates forward driving '
        'from backdriving, recovers both torque-cell zeros from the grid\'s '
        'mirror symmetry, fits an output-referred loss-torque model, and bounds '
        'the cross-calibration error between the two cells.'
    )
    granularity = 'run'

    params = {
        # -- roles and channels (auto-detected; override to force) ------------
        'input_device':          (None, str,   'Device holding the speed setpoint (auto-detected)'),
        'output_device':         (None, str,   'Device holding the torque setpoint (auto-detected)'),
        'input_torque_channel':  (None, str,   'Input torque cell (default: the input port\'s cell)'),
        'output_torque_channel': (None, str,   'Output torque cell (default: the output port\'s cell)'),
        'gear_ratio':            (None, float, 'Input:output reduction (default: from the log config)'),

        # -- dwell conditioning ----------------------------------------------
        # The runner's spans are already trimmed to the settled dwell, so the
        # default trims nothing; the knob exists for logs that are not.
        'settle_frac':      (0.0,   float, 'Trim this leading fraction of each dwell before averaging'),
        'min_dwell_samples': (200,  int,   'Ignore spans shorter than this'),
        'min_output_speed': (1e-3,  float, 'Drop dwells below this |output speed|; no power flows [rad/s]'),
        'grid_decimals':    (2,     int,   'Rounding used to collapse repeated setpoints onto one grid cell'),

        # -- cell zeros -------------------------------------------------------
        # 'mirror' is the only estimator that works without an unloaded rest
        # window. Set 'none' to see the raw cells, or force either zero directly.
        'torque_zero':          ('mirror', str,   "Cell zero estimate: 'mirror' (grid symmetry) or 'none'"),
        'input_torque_zero':    (None,     float, 'Force the input cell zero [Nm]'),
        'output_torque_zero':   (None,     float, 'Force the output cell zero [Nm]'),
        'zero_drift_frac':      (0.5,      float, 'Warn when the mirror estimate drifts across the load range '
                                                  'by more than this fraction of itself (then it is not an offset)'),

        # -- direction-dependent output-cell bias ------------------------------
        # Survives the mirror tare (it cancels in the zero estimate but not in
        # any single dwell), so it needs its own estimator. 'none' reaches the
        # uncorrected numbers.
        'torque_hysteresis':    ('auto',   str,   "Direction-dependent output-cell bias: 'auto' "
                                                  "(sign structure) or 'none'"),
        'output_torque_hyst':   (None,     float, 'Force the hysteresis term h [Nm]'),
        'hyst_agree_tol':       (0.3,      float, 'Warn when the two independent h estimates differ '
                                                  'by more than this [Nm]'),
        'hyst_loss_frac':       (0.25,     float, 'Warn when h exceeds this fraction of the median '
                                                  'loss torque (then it is not a cosmetic correction)'),

        # -- reporting thresholds ---------------------------------------------
        'headline_min_load_frac': (0.5,  float, 'Headline efficiency stats use dwells above this fraction '
                                                'of peak commanded output torque'),
        'loss_fit_min_load_frac': (0.1,  float, 'Fit the loss model above this fraction of peak output torque'),
        'min_output_revs':        (1.0,  float, 'Warn about speed rows whose dwells turn the output less '
                                                'than this many revolutions (mesh position is not averaged)'),
        'ratio_tol_pct':          (2.0,  float, 'Config-vs-measured gear ratio disagreement that warns [%]'),
        'per_dwell_csv':          (True, bool,  'Export the per-dwell table as well as the per-cell one'),
    }

    # -- discovery ---------------------------------------------------------

    def _roles(self, segs):
        """Identify the speed-holding port and the torque-holding port.

        Read from the command channels, not from `ports`: a command channel is
        NaN whenever its mode is inactive, so 'which motor was in velocity mode'
        is a fact in the data rather than a naming convention. `ports` is
        consulted only for channel prefixes, cells, and labels.
        """
        seg = segs[0]
        driver = loader = None
        for dev in seg.devices():
            prefix = seg.prefix(dev)
            if any(s.is_active(f'{prefix}_velocity_command') for s in segs[:8]):
                driver = driver or (dev, prefix)
            if any(s.is_active(f'{prefix}_torque_command') for s in segs[:8]):
                loader = loader or (dev, prefix)
        return driver, loader

    def _config_ratio(self, seg, in_dev, out_dev):
        """Reduction from the input motor shaft to the output shaft."""
        r_in = seg.device_params(in_dev).get('gear_ratio')
        r_out = seg.device_params(out_dev).get('gear_ratio') or 1.0
        if r_in is None:
            return None
        try:
            return float(r_in) / float(r_out or 1.0)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _cells(self, seg, p, in_prefix, out_prefix):
        """(input torque channel, output torque channel), from params or ports."""
        in_cell = (p['input_torque_channel']
                   or (seg.port_by_role('input') or {}).get('cell')
                   or f'{in_prefix}_torque')
        out_cell = (p['output_torque_channel']
                    or (seg.port_by_role('output') or {}).get('cell')
                    or f'{out_prefix}_torque')
        return in_cell, out_cell

    def applies_to(self, segs):
        d = self.defaults()
        seg = segs[0]
        driver, loader = self._roles(segs)

        if driver is None:
            return Applicability('no', 'no motor is holding a velocity command')
        if loader is None:
            return Applicability('no', 'no motor is holding a torque command; '
                                       'nothing is loading the output')
        if driver[0] == loader[0]:
            return Applicability('no', f'{driver[0]} carries both a velocity and '
                                       'a torque command; cannot assign roles')

        in_dev, in_prefix = driver
        out_dev, out_prefix = loader
        in_cell, out_cell = self._cells(seg, d, in_prefix, out_prefix)
        missing = [c for c in (in_cell, out_cell, f'{in_prefix}_velocity',
                               f'{out_prefix}_velocity') if not seg.has(c)]
        if missing:
            return Applicability('no', 'efficiency needs a torque cell and an '
                                       'encoder on both shafts; missing '
                                       + ', '.join(missing))

        ratio = self._config_ratio(seg, in_dev, out_dev)
        if ratio is None or not np.isfinite(ratio) or ratio == 0:
            return Applicability('no', f'no usable gear_ratio for {in_dev}')

        # One span per grid setpoint is the shape this analysis needs: a dwell
        # is the unit of measurement. A log whose spans are whole sweeps would
        # average unrelated operating points together.
        dwells = [s for s in segs if s.setpoint is not None
                  and len(s) >= d['min_dwell_samples']]
        if len(dwells) < 8:
            return Applicability(
                'no', f'need at least 8 setpoint spans of '
                      f'{d["min_dwell_samples"]}+ samples; found {len(dwells)}')

        vs = {round(float(np.nanmean(s[f'{in_prefix}_velocity_command'])), 2)
              for s in dwells}
        ts = {round(float(np.nanmean(s[f'{out_prefix}_torque_command'])), 2)
              for s in dwells}
        shape = (f'{len(dwells)} dwells over {len(vs)} speed x {len(ts)} torque '
                 f'setpoints at {ratio:g}:1')

        # Without both torque signs there is only one power-flow direction and
        # no mirror pair to recover the cell zeros from. Still worth running --
        # the map is real -- but the light-load numbers are then untared.
        if not (max(ts) > 0 > min(ts)):
            return Applicability(
                'maybe', f'{shape}, but the output torque never changes sign: '
                         'only one power-flow direction, and no mirror pairs to '
                         'tare the torque cells with',
                {'input_device': in_dev, 'output_device': out_dev})

        return Applicability(
            'yes', f'{in_dev} sweeps speed while {out_dev} sweeps output torque '
                   f'in both directions; {shape}',
            {'input_device': in_dev, 'output_device': out_dev})

    # -- dwell extraction --------------------------------------------------

    def _dwells(self, segs, p, chans):
        """One record per setpoint span: the steady operating point it holds.

        Torques stay RAW here. The zero estimate is built from these records and
        applied afterwards, so that switching `torque_zero` never re-reads the
        log and the raw readings remain available for the diagnostics.
        """
        out = []
        for s in segs:
            if s.setpoint is None or len(s) < p['min_dwell_samples']:
                continue
            a = int(len(s) * max(0.0, min(p['settle_frac'], 0.9)))
            m = lambda ch: float(np.nanmean(s[ch][a:]))
            w_out = m(chans['w_out'])
            out.append({
                'raw_id': s.raw_id,
                'setpoint': s.setpoint,
                'n': len(s) - a,
                'duration': s.duration * (1 - a / max(len(s), 1)),
                'v_cmd': round(m(chans['v_cmd']), p['grid_decimals']),
                't_cmd': round(m(chans['t_cmd']), p['grid_decimals']),
                'w_in': m(chans['w_in']),
                'w_out': w_out,
                't_in_raw': m(chans['t_in']),
                't_out_raw': m(chans['t_out']),
                't_in_sd': float(np.nanstd(s[chans['t_in']][a:])),
                't_out_sd': float(np.nanstd(s[chans['t_out']][a:])),
            })
        for d in out:
            d['rev_out'] = abs(d['w_out']) * d['duration'] / (2 * np.pi)
        return out

    # -- cell zeros --------------------------------------------------------

    def _mirror_zero(self, dwells, key, decimals):
        """Recover one cell's zero from the grid's mirror symmetry.

        (+v, +T) and (-v, -T) are the same operating point with every physical
        torque reversed, so their average is what the sensor reads with no
        torque on it. Returns the median over all available mirror pairs, plus
        the pair-mean's drift across the load range -- a genuine zero offset is
        load-independent, so drift is the signature of this estimator having
        absorbed something that is not an offset.
        """
        cells = {}
        for d in dwells:
            cells.setdefault((d['v_cmd'], d['t_cmd']), []).append(d[key])
        cells = {k: float(np.mean(v)) for k, v in cells.items()}

        loads, means = [], []
        for (v, t), val in cells.items():
            if v <= 0:
                continue
            mirror = cells.get((round(-v, decimals), round(-t, decimals)))
            if mirror is None:
                continue
            loads.append(abs(t))
            means.append(0.5 * (val + mirror))
        if not means:
            return {'zero': 0.0, 'n_pairs': 0, 'drift': float('nan'),
                    'spread': float('nan')}

        slope, _, _ = _linfit(loads, means)
        span = (max(loads) - min(loads)) if len(set(loads)) > 1 else 0.0
        return {'zero': float(np.median(means)),
                'n_pairs': len(means),
                'drift': abs(slope) * span if np.isfinite(slope) else float('nan'),
                'spread': _sd(means)}

    def _hysteresis(self, dwells, tol):
        """Recover the output cell's direction-dependent bias h.

        The cell reads T_true + z + h*sign(w_out). The mirror tare removes z
        and leaves h in every dwell (see the module docstring), so h has to
        come from the other symmetry: the sign of the TRUE output torque,
        s = sign(T_out_true), which mode and rotation fix between them.

            mode        sign(w_out)   s        the cell reads
            forward         +1        +1        T + z + h
            backdrive       -1        +1        T + z - h
            backdrive       +1        -1       -T + z + h
            forward         -1        -1       -T + z - h

        Within one s group the two members differ by exactly 2h: the torque,
        the zero, the loss torque and any forward/backdrive loss asymmetry are
        common to both and cancel. Nothing about efficiency is assumed.

        The two groups give two INDEPENDENT estimates. They are reported
        separately on purpose -- if they disagree, the sign model itself is
        wrong and the correction should not be applied, which no single pooled
        number would reveal.

        Estimated from RAW torque: a constant z cancels in every difference
        above, so this does not care whether the tare has happened yet.
        """
        need = (('forward', +1), ('backdrive', -1),   # s = +1
                ('backdrive', +1), ('forward', -1))   # s = -1
        groups = {}
        for d in dwells:
            if d['mode'] not in _MODES:
                continue
            combo = (d['mode'], 1 if d['w_out'] > 0 else -1)
            key = (abs(d['v_cmd']), abs(d['t_cmd']))
            groups.setdefault(key, {}).setdefault(combo, []).append(d['t_out_raw'])

        per_key, pos, neg, speeds = [], [], [], {}
        for (v, _t), g in groups.items():
            # All four combos, or the differences below are not like-for-like.
            if not all(c in g for c in need):
                continue
            m = {c: float(np.mean(g[c])) for c in need}
            h_pos = (m[need[0]] - m[need[1]]) / 2
            h_neg = (m[need[2]] - m[need[3]]) / 2
            per_key.append(0.5 * (h_pos + h_neg))
            pos.append(h_pos)
            neg.append(h_neg)
            speeds.setdefault(v, []).append(0.5 * (h_pos + h_neg))

        if not per_key:
            return {'h': 0.0, 'h_pos': float('nan'), 'h_neg': float('nan'),
                    'n_keys': 0, 'spread': float('nan'), 'disagree': float('nan'),
                    'agrees': False, 'by_speed': {}}

        # Median over keys: h is nominally load-independent, and a median stops
        # a handful of ill-conditioned cells (near-zero torque command, where
        # the mode label is noise) from dragging the estimate.
        h_pos, h_neg = float(np.median(pos)), float(np.median(neg))
        return {'h': float(np.median(per_key)),
                'h_pos': h_pos, 'h_neg': h_neg,
                'n_keys': len(per_key),
                'spread': _sd(per_key),
                'disagree': abs(h_pos - h_neg),
                'agrees': abs(h_pos - h_neg) <= tol,
                'by_speed': {v: float(np.median(vals))
                             for v, vals in sorted(speeds.items())}}

    # -- physics -----------------------------------------------------------

    def _derive(self, dwells, ratio, z_in, z_out, min_speed, h_out=0.0):
        """Attach powers, mode, efficiency and loss torque to each dwell.

        `mode` is decided by the sign of the two mechanical powers, which is the
        measurement rather than an assumption about which way the test intended
        the power to flow:

            both positive -> forward     (input drives the load)
            both negative -> backdrive   (load overhauls the gearbox)
            opposite      -> dissipative if both shafts feed power in (the
                             gearbox is eating all of it and delivering none),
                             non-physical if both extract it, which cannot
                             happen and means a cell zero or sign is wrong.
        """
        kept, dropped = [], []
        for d in dwells:
            if abs(d['w_out']) < min_speed:
                dropped.append(d)
                continue
            t_in = d['t_in_raw'] - z_in
            t_out = d['t_out_raw'] - z_out - h_out * np.sign(d['w_out'])
            p_in = t_in * d['w_in']
            p_out = t_out * d['w_out']

            if p_in > 0 and p_out > 0:
                mode, eff = 'forward', p_out / p_in
            elif p_in < 0 and p_out < 0:
                mode, eff = 'backdrive', p_in / p_out
            elif p_in > 0 or p_out < 0:
                mode, eff = 'dissipative', 0.0
            else:
                mode, eff = 'non_physical', float('nan')

            d.update({
                't_in': t_in, 't_out': t_out, 'p_in': p_in, 'p_out': p_out,
                'mode': mode, 'eff': eff,
                'rotation': 'cw' if d['w_in'] > 0 else 'ccw',
                # Output-referred loss torque. Positive is the physical sign;
                # negative means the two cells disagree by more than the loss.
                'loss_Nm': (p_in - p_out) / abs(d['w_out']),
                'loss_W': abs(p_in - p_out),
                'load_Nm': abs(t_out),
            })
            kept.append(d)
        return kept, dropped

    def _feasible_offset(self, dwells, ratio, h_out=0.0):
        """Interval that a constant combined cell offset must lie in.

        P_loss = w_out * (ratio*T_in_raw - T_out_raw - c) >= 0 at every dwell,
        with c = ratio*z_in - z_out the only combination of the two zeros that
        efficiency depends on. Each dwell is therefore one bound on c, and the
        interval below is their intersection. An empty interval means no pair of
        cell zeros makes this log self-consistent, and |width| is a lower bound
        on the cross-calibration error between the cells in output Nm.

        `h_out` is removed from the output reading first because it is not part
        of c: it flips sign with rotation, so it cannot be absorbed by any
        constant offset. That is precisely why it shows up here as an EMPTY
        interval -- an uncorrected h opens the interval by 2h in the infeasible
        direction, which makes the width before and after the correction the
        natural acceptance test for the whole hysteresis model.
        """
        lo, hi = -np.inf, np.inf
        lo_at = hi_at = None
        for d in dwells:
            m = (ratio * d['t_in_raw']
                 - (d['t_out_raw'] - h_out * np.sign(d['w_out'])))
            if d['w_out'] > 0 and m < hi:
                hi, hi_at = m, d
            elif d['w_out'] < 0 and m > lo:
                lo, lo_at = m, d
        return {'lo': lo, 'hi': hi, 'width': hi - lo,
                'lo_at': lo_at, 'hi_at': hi_at}

    # -- aggregation -------------------------------------------------------

    @staticmethod
    def _pool(dwells, keyfn, valfn):
        """{key: mean of valfn} over the dwells sharing each key."""
        buckets = {}
        for d in dwells:
            buckets.setdefault(keyfn(d), []).append(valfn(d))
        return {k: float(np.nanmean(v)) for k, v in buckets.items()}

    def _mode_grid(self, dwells, mode, valfn):
        """(rows, cols, grid) over |input speed| x |commanded output torque|."""
        pooled = self._pool([d for d in dwells if d['mode'] == mode],
                            lambda d: (abs(d['v_cmd']), abs(d['t_cmd'])), valfn)
        rows = sorted({k[0] for k in pooled})
        cols = sorted({k[1] for k in pooled})
        grid = np.full((len(rows), len(cols)), np.nan)
        for (v, t), val in pooled.items():
            grid[rows.index(v), cols.index(t)] = val
        return rows, cols, grid

    # -- plotting ----------------------------------------------------------

    def _heatmap(self, ax, rows, cols, grid, ratio, cmap, vmin, vmax, fmt='{:.0f}'):
        import matplotlib.pyplot as plt

        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad('lightgray')
        im = ax.imshow(np.ma.masked_invalid(grid), origin='lower', aspect='auto',
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels([f'{c:g}' for c in cols], rotation=45, ha='right')
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f'{r / ratio:.2f}' for r in rows])
        # Ink colour from the cell's own luminance rather than from the value:
        # a diverging map is dark at BOTH ends, so a threshold on the value gets
        # half of them wrong.
        for i in range(len(rows)):
            for j in range(len(cols)):
                if not np.isfinite(grid[i, j]):
                    continue
                r, g, b, _ = cmap((grid[i, j] - vmin) / (vmax - vmin or 1))
                ax.text(j, i, fmt.format(grid[i, j]), ha='center', va='center',
                        fontsize=7,
                        color='white' if 0.299 * r + 0.587 * g + 0.114 * b < 0.55
                        else 'black')
        return im

    def _plot_maps(self, dwells, ratio, labels):
        import matplotlib.pyplot as plt

        panels = [(m, *self._mode_grid(dwells, m, lambda d: 100 * d['eff']),
                   f'{m.upper()} driving (both cells)') for m in _MODES]

        # One colour scale across both panels: the point of the figure is
        # comparing them, and per-panel scaling would hide exactly that.
        finite = np.concatenate([g[np.isfinite(g)].ravel() for _, _, _, g, _
                                 in panels if np.any(np.isfinite(g))] or [np.array([0., 100.])])
        vmin = max(0.0, np.floor(finite.min() / 5) * 5)
        vmax = max(100.0, np.ceil(finite.max()))

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        for ax, (mode, rows, cols, grid, title) in zip(axes, panels):
            ax.set_title(title)
            ax.set_xlabel('|Output torque| setpoint (Nm)')
            ax.set_ylabel('|Output speed| (rad/s)')
            if not rows or not np.any(np.isfinite(grid)):
                ax.text(0.5, 0.5, f'no {mode} data', ha='center', va='center',
                        transform=ax.transAxes)
                continue
            im = self._heatmap(ax, rows, cols, grid, ratio, 'viridis', vmin, vmax)
            fig.colorbar(im, ax=ax, label='Efficiency (%)')
        fig.suptitle(f'{labels["title"]} -- efficiency map at {ratio:g}:1 '
                     '(rotations averaged; anything over 100% is a '
                     'measurement error, not a machine)')
        fig.tight_layout()
        return fig

    def _plot_curves(self, dwells, ratio, labels):
        """Efficiency against output torque, one line per speed."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
        speeds = sorted({abs(d['v_cmd']) for d in dwells})
        colors = plt.cm.viridis(np.linspace(0, 0.92, max(len(speeds), 1)))
        for ax, mode in zip(axes, _MODES):
            for c, v in zip(colors, speeds):
                row = [d for d in dwells
                       if d['mode'] == mode and abs(d['v_cmd']) == v]
                load = self._pool(row, lambda d: abs(d['t_cmd']),
                                  lambda d: d['load_Nm'])
                eff = self._pool(row, lambda d: abs(d['t_cmd']),
                                 lambda d: 100 * d['eff'])
                pts = sorted((load[t], eff[t]) for t in load)
                if pts:
                    ax.plot([x for x, _ in pts], [y for _, y in pts], 'o-',
                            color=c, markersize=4, linewidth=1.4,
                            label=f'{v / ratio:.2f} rad/s')
            ax.axhline(100, color='k', linewidth=0.8, linestyle=':')
            ax.set_xscale('log')
            ax.set_xlabel('|Output torque| (Nm)')
            ax.set_title(f'{mode.upper()} driving')
            ax.grid(True, which='both', alpha=0.3)
        axes[0].set_ylabel('Efficiency (%)')
        axes[0].legend(fontsize=7, title='output speed', ncol=2)
        fig.suptitle(f'{labels["title"]} -- efficiency vs load '
                     '(the 100% line is the sanity check)')
        fig.tight_layout()
        return fig

    def _plot_loss(self, dwells, fits, ratio, labels):
        """The load-invariant view: output-referred loss torque vs load."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
        speeds = sorted({abs(d['v_cmd']) for d in dwells})
        colors = plt.cm.plasma(np.linspace(0, 0.85, max(len(speeds), 1)))
        top = max(d['load_Nm'] for d in dwells)
        for ax, mode in zip(axes, _MODES):
            for c, v in zip(colors, speeds):
                # Pool the two rotations at each grid cell: a residual cell
                # offset displaces them in opposite directions, so the pair
                # mean is the comparable quantity.
                row = [d for d in dwells
                       if d['mode'] == mode and abs(d['v_cmd']) == v]
                load = self._pool(row, lambda d: abs(d['t_cmd']),
                                  lambda d: d['load_Nm'])
                loss = self._pool(row, lambda d: abs(d['t_cmd']),
                                  lambda d: d['loss_Nm'])
                pts = sorted((load[t], loss[t]) for t in load)
                if pts:
                    ax.plot([x for x, _ in pts], [y for _, y in pts], 'o-',
                            color=c, markersize=4, linewidth=1.0,
                            label=f'{v / ratio:.2f} rad/s out')
            fit = fits.get(mode)
            if fit and np.isfinite(fit['slope']):
                xs = np.array([0.0, top])
                ax.plot(xs, fit['slope'] * xs + fit['intercept'], 'k--',
                        linewidth=1.4,
                        label=f'L = {fit["intercept"]:.2f} Nm '
                              f'+ {100 * fit["slope"]:.1f}% T')
            ax.axhline(0, color='k', linewidth=0.8)
            ax.set_xlabel('|Output torque| (Nm)')
            ax.set_title(f'{mode.upper()} driving')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        axes[0].set_ylabel('Output-referred loss torque (Nm)')
        fig.suptitle(f'{labels["title"]} -- loss torque; negative is impossible '
                     'and means the two cells disagree by more than the loss')
        fig.tight_layout()
        return fig

    def _plot_asymmetry(self, dwells, ratio, labels, cap=20.0):
        """CCW minus CW efficiency, per mode. A direction-symmetric machine
        reads zero everywhere; structure here is either a real asymmetry or a
        cell zero the mirror estimate has not removed."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(17, 7))
        for ax, mode in zip(axes, ('forward', 'backdrive')):
            sel = [d for d in dwells if d['mode'] == mode]
            pooled = self._pool(
                sel, lambda d: (abs(d['v_cmd']), abs(d['t_cmd']), d['rotation']),
                lambda d: 100 * d['eff'])
            rows = sorted({k[0] for k in pooled})
            cols = sorted({k[1] for k in pooled})
            grid = np.full((len(rows), len(cols)), np.nan)
            for i, v in enumerate(rows):
                for j, t in enumerate(cols):
                    cw, ccw = pooled.get((v, t, 'cw')), pooled.get((v, t, 'ccw'))
                    if cw is not None and ccw is not None:
                        grid[i, j] = ccw - cw
            ax.set_title(f'{mode.upper()} driving')
            ax.set_xlabel('|Output torque| setpoint (Nm)')
            ax.set_ylabel('|Output speed| (rad/s)')
            if not rows or not np.any(np.isfinite(grid)):
                ax.text(0.5, 0.5, 'no matched CW/CCW pairs', ha='center',
                        va='center', transform=ax.transAxes)
                continue
            im = self._heatmap(ax, rows, cols, grid, ratio, 'coolwarm',
                               -cap, cap, fmt='{:+.0f}')
            fig.colorbar(im, ax=ax, label='CCW - CW efficiency (points)')
        fig.suptitle(f'{labels["title"]} -- direction asymmetry')
        fig.tight_layout()
        return fig

    def _plot_consistency(self, dwells, offset, applied_c, labels):
        """The calibration honesty plot.

        Left: loss torque against load, split by rotation direction. A constant
        cell offset displaces the two branches in opposite directions, so half
        the gap between their fits is the residual offset that the applied zeros
        did not remove -- and the branches only overlap when the cells agree.

        Right: where each dwell puts its bound on the combined offset, and the
        interval those bounds leave. An empty interval means the two cells
        cannot both be right whatever zeros are chosen.
        """
        import matplotlib.pyplot as plt

        fig, (axL, axM) = plt.subplots(1, 2, figsize=(14, 6))

        residual = {}
        for sign, color, tag in ((+1, 'tab:blue', 'w_out > 0'),
                                 (-1, 'tab:orange', 'w_out < 0')):
            sel = [d for d in dwells if np.sign(d['w_out']) == sign]
            if not sel:
                continue
            x = [d['load_Nm'] for d in sel]
            y = [d['loss_Nm'] for d in sel]
            axL.plot(x, y, 'o', color=color, markersize=4, alpha=0.7, label=tag)
            slope, intercept, _ = _linfit(x, y)
            if np.isfinite(slope):
                xs = np.array([0, max(x)])
                axL.plot(xs, slope * xs + intercept, '--', color=color, linewidth=1.2)
                residual[sign] = intercept
        axL.axhline(0, color='k', linewidth=0.8)
        axL.set_xlabel('|Output torque| measured (Nm)')
        axL.set_ylabel('Output-referred loss torque (Nm)')
        gap = (residual.get(+1, np.nan) - residual.get(-1, np.nan)) / 2
        axL.set_title('Loss torque by rotation direction\n'
                      f'residual combined cell offset = {gap:+.3f} Nm '
                      '(half the intercept gap)')
        axL.grid(True, alpha=0.3)
        axL.legend(fontsize=8)

        # Each dwell bounds the combined offset c; sorting turns the two families
        # of bounds into staircases whose crossing point is the feasible window.
        upper = sorted(d['loss_Nm'] + applied_c for d in dwells if d['w_out'] > 0)
        lower = sorted(applied_c - d['loss_Nm'] for d in dwells if d['w_out'] < 0)
        axM.plot(upper, range(len(upper)), '.', color='tab:blue', markersize=3,
                 label='upper bounds on c (w_out > 0)')
        axM.plot(lower, range(len(lower)), '.', color='tab:orange', markersize=3,
                 label='lower bounds on c (w_out < 0)')
        lo, hi = offset['lo'], offset['hi']
        if np.isfinite(lo) and np.isfinite(hi):
            axM.axvspan(min(lo, hi), max(lo, hi),
                        color='tab:green' if hi > lo else 'tab:red', alpha=0.18,
                        label='feasible' if hi > lo else 'infeasible: no zero works')
        axM.axvline(applied_c, color='k', linewidth=1.2,
                    label=f'applied c = {applied_c:+.3f} Nm')
        axM.set_xlabel('Combined cell offset c = ratio*z_in - z_out (Nm at output)')
        axM.set_ylabel('Dwell (sorted)')
        axM.set_title('Feasible offset interval\n' + (
            f'[{lo:+.3f}, {hi:+.3f}] Nm, width {hi - lo:.3f} Nm' if hi > lo else
            f'empty: dwells demand c > {lo:+.3f} and c < {hi:+.3f} at once'))
        axM.grid(True, alpha=0.3)
        axM.legend(fontsize=7, loc='lower right')

        fig.suptitle(f'{labels["title"]} -- torque-cell cross-calibration')
        fig.tight_layout()
        return fig

    # -- main --------------------------------------------------------------

    def run(self, segs, params):
        p = dict(params)
        res = Result()
        seg = segs[0]

        in_dev, out_dev = p['input_device'], p['output_device']
        if not (in_dev and out_dev):
            driver, loader = self._roles(segs)
            if driver is None or loader is None:
                raise ValueError('could not identify the speed-holding and '
                                 'torque-holding motors; set input_device / '
                                 'output_device')
            in_dev, out_dev = driver[0], loader[0]
        in_prefix, out_prefix = seg.prefix(in_dev), seg.prefix(out_dev)
        in_cell, out_cell = self._cells(seg, p, in_prefix, out_prefix)
        chans = {'v_cmd': f'{in_prefix}_velocity_command',
                 't_cmd': f'{out_prefix}_torque_command',
                 'w_in': f'{in_prefix}_velocity',
                 'w_out': f'{out_prefix}_velocity',
                 't_in': in_cell, 't_out': out_cell}

        dwells = self._dwells(segs, p, chans)
        if len(dwells) < 4:
            raise ValueError(f'only {len(dwells)} usable dwells in this run')

        # -- gear ratio: config value, cross-checked against the encoders -----
        cfg_ratio = self._config_ratio(seg, in_dev, out_dev)
        ratio = float(p['gear_ratio']) if p['gear_ratio'] else cfg_ratio
        ratio_source = ('gear_ratio parameter override' if p['gear_ratio']
                        else f'log resolved_config ({in_dev}.gear_ratio)')
        w_in = np.array([d['w_in'] for d in dwells])
        w_out = np.array([d['w_out'] for d in dwells])
        turning = np.abs(w_out) > p['min_output_speed']
        measured_ratio = (float(np.sum(w_in[turning] * w_out[turning])
                                / np.sum(w_out[turning] ** 2))
                          if np.any(turning) else float('nan'))
        if (np.isfinite(measured_ratio)
                and abs(measured_ratio - ratio) / abs(ratio) > p['ratio_tol_pct'] / 100):
            res.add('warn', 'gear_ratio_mismatch',
                    f'The two encoders imply {measured_ratio:.4f}:1, but the '
                    f'analysis is using {ratio:g}:1 from the {ratio_source}. '
                    f'Efficiency scales directly with this ratio -- reconcile it '
                    f'before quoting any number below.')
        else:
            res.add('info', 'gear_ratio_confirmed',
                    f'Using {ratio:g}:1 from the {ratio_source}; the encoders '
                    f'imply {measured_ratio:.4f}:1.')

        # -- torque cell zeros ------------------------------------------------
        zeros = {'input': {'zero': 0.0, 'n_pairs': 0, 'drift': float('nan'),
                           'spread': float('nan')},
                 'output': {'zero': 0.0, 'n_pairs': 0, 'drift': float('nan'),
                            'spread': float('nan')}}
        if p['torque_zero'] not in ('mirror', 'none'):
            raise ValueError("torque_zero must be 'mirror' or 'none', "
                             f"got {p['torque_zero']!r}")
        if p['torque_zero'] == 'mirror':
            for side, key in (('input', 't_in_raw'), ('output', 't_out_raw')):
                zeros[side] = self._mirror_zero(dwells, key, p['grid_decimals'])
        if p['input_torque_zero'] is not None:
            zeros['input'] = {'zero': float(p['input_torque_zero']), 'n_pairs': 0,
                              'drift': float('nan'), 'spread': float('nan')}
        if p['output_torque_zero'] is not None:
            zeros['output'] = {'zero': float(p['output_torque_zero']), 'n_pairs': 0,
                               'drift': float('nan'), 'spread': float('nan')}
        z_in, z_out = zeros['input']['zero'], zeros['output']['zero']

        if p['torque_zero'] == 'mirror' and zeros['input']['n_pairs'] == 0:
            res.add('warn', 'no_mirror_pairs',
                    'No mirrored (+speed, +torque) / (-speed, -torque) grid cells '
                    'were found, so the torque cells could not be tared. Light-load '
                    'efficiency carries the full cell offset.')
        for side, cell in (('input', in_cell), ('output', out_cell)):
            z = zeros[side]
            if z['n_pairs'] and z['drift'] > p['zero_drift_frac'] * abs(z['zero']):
                res.add('warn', f'{side}_zero_not_constant',
                        f'The {side} cell zero recovered from mirror symmetry is '
                        f'{z["zero"]:+.4f} Nm, but it drifts {z["drift"]:.4f} Nm '
                        f'across the load range. A true sensor offset does not '
                        f'depend on load, so part of this is a real CW/CCW '
                        f'asymmetry of the machine being absorbed into the tare '
                        f'-- see the direction-asymmetry figure.')
            elif z['n_pairs']:
                res.add('info', f'{side}_zero_applied',
                        f'{cell} tared by {z["zero"]:+.4f} Nm, from {z["n_pairs"]} '
                        f'mirror pairs (pair-to-pair spread {z["spread"]:.4f} Nm, '
                        f'drift across load {z["drift"]:.4f} Nm).')

        # -- per-dwell physics ------------------------------------------------
        # Two passes, because the hysteresis estimator groups dwells by `mode`
        # and only _derive() can assign it. The first pass exists solely to get
        # those labels; h is estimated from raw torque, so nothing it depends on
        # changes between the passes and there is no fixed point to chase.
        if p['torque_hysteresis'] not in ('auto', 'none'):
            raise ValueError("torque_hysteresis must be 'auto' or 'none', "
                             f"got {p['torque_hysteresis']!r}")
        live, standstill = self._derive(dwells, ratio, z_in, z_out,
                                        p['min_output_speed'])
        if not live:
            raise ValueError('every dwell is below min_output_speed; nothing '
                             'is transmitting power')
        if standstill:
            res.add('info', 'standstill_dwells',
                    f'{len(standstill)} dwell(s) held the output below '
                    f'{p["min_output_speed"]:g} rad/s and carry no power; dropped.')

        # -- direction-dependent output bias, then the second pass -------------
        hyst = {'h': 0.0, 'h_pos': float('nan'), 'h_neg': float('nan'),
                'n_keys': 0, 'spread': float('nan'), 'disagree': float('nan'),
                'agrees': False, 'by_speed': {}}
        if p['torque_hysteresis'] == 'auto':
            hyst = self._hysteresis(live, p['hyst_agree_tol'])
        forced_h = p['output_torque_hyst']
        # An estimate whose two halves disagree is evidence against the sign
        # model, so it is reported and NOT applied. A forced value is the
        # operator overriding that judgement and always wins.
        h_out = (float(forced_h) if forced_h is not None
                 else (hyst['h'] if hyst['agrees'] else 0.0))

        width_before = self._feasible_offset(live, ratio)['width']
        if h_out:
            live, standstill = self._derive(dwells, ratio, z_in, z_out,
                                            p['min_output_speed'], h_out)

        if forced_h is not None:
            res.add('info', 'output_hysteresis_forced',
                    f'Output cell hysteresis forced to h = {h_out:+.4f} Nm by '
                    f'output_torque_hyst; every dwell has h*sign(w_out) removed '
                    f'from {out_cell}.')
        elif hyst['n_keys'] and hyst['agrees']:
            res.add('info', 'output_hysteresis_applied',
                    f'{out_cell} carries a direction-dependent bias of '
                    f'h = {h_out:+.4f} Nm, recovered from {hyst["n_keys"]} grid '
                    f'cells that have all four (mode, rotation) combinations '
                    f'(key-to-key spread {hyst["spread"]:.4f} Nm). The two '
                    f'independent halves of the estimate agree: '
                    f'{hyst["h_pos"]:+.4f} Nm from the positive-torque group and '
                    f'{hyst["h_neg"]:+.4f} Nm from the negative-torque group, a '
                    f'difference of {hyst["disagree"]:.4f} Nm. This term flips '
                    f'sign with rotation, so the mirror tare cancels it in the '
                    f'zero estimate but leaves it in every dwell; it has been '
                    f'removed as h*sign(w_out).')
        elif hyst['n_keys']:
            res.add('warn', 'output_hysteresis_disagrees',
                    f'The two independent estimates of the output cell\'s '
                    f'direction-dependent bias disagree by '
                    f'{hyst["disagree"]:.4f} Nm ({hyst["h_pos"]:+.4f} Nm from the '
                    f'positive-torque group, {hyst["h_neg"]:+.4f} Nm from the '
                    f'negative-torque group, tolerance '
                    f'{p["hyst_agree_tol"]:g} Nm). Under the model these two '
                    f'measure the same quantity, so disagreement means the model '
                    f'is wrong -- something other than a rotation-dependent bias '
                    f'is moving with direction. NO correction has been applied; '
                    f'set output_torque_hyst to force one.')
        elif p['torque_hysteresis'] == 'auto':
            res.add('warn', 'output_hysteresis_unmeasurable',
                    'No grid cell has all four (mode, rotation) combinations, so '
                    'the output cell\'s direction-dependent bias could not be '
                    'separated from the torque. Any such bias is still in every '
                    'dwell below.')

        counts = {m: sum(1 for d in live if d['mode'] == m)
                  for m in ('forward', 'backdrive', 'dissipative', 'non_physical')}
        if counts['dissipative']:
            res.add('info', 'dissipative_dwells',
                    f'{counts["dissipative"]} dwell(s) had both shafts feeding '
                    f'power into the gearbox and none coming out -- the load '
                    f'could not backdrive it. Reported as 0% efficiency.')
        if counts['non_physical']:
            res.add('error', 'non_physical_dwells',
                    f'{counts["non_physical"]} dwell(s) show both shafts taking '
                    f'power OUT of the gearbox, which cannot happen. A torque '
                    f'cell sign or zero is wrong.')

        # -- consistency: what offset could reconcile the two cells? ----------
        offset = self._feasible_offset(live, ratio, h_out)
        applied_c = ratio * z_in - z_out
        median_loss = float(np.median([d['loss_Nm'] for d in live]))

        # The acceptance test for the whole hysteresis model. An uncorrected h
        # opens the feasible interval by 2h in the infeasible direction, so if
        # the model is right, removing h has to close most of that gap -- and if
        # it does not, h is not what was making the two cells irreconcilable.
        if h_out:
            recovered = offset['width'] - width_before
            # Only an ESTIMATED h is independent evidence. A forced one was
            # chosen by hand, so the recovery it produces is arithmetic (the
            # width always moves by 2h) and must not be quoted as confirmation.
            why = (' This is the independent check on the correction: h was '
                   'measured from the rotation sign structure alone and never '
                   'from this interval, so the two agreeing is evidence and not '
                   'bookkeeping.' if forced_h is None else
                   ' h was forced rather than measured here, so this recovery is '
                   'arithmetic -- the width moves by 2h whatever h is -- and is '
                   'not evidence that the value is right.')
            res.add('info', 'hysteresis_feasibility_recovery',
                    f'Removing the direction-dependent bias moves the feasible '
                    f'cell-offset interval from a width of {width_before:+.3f} Nm '
                    f'to {offset["width"]:+.3f} Nm, recovering {recovered:.3f} Nm '
                    f'({100 * recovered / abs(width_before):.0f}% of the '
                    f'infeasibility) of cross-cell disagreement.' + why)
            if abs(h_out) > p['hyst_loss_frac'] * abs(median_loss):
                res.add('warn', 'hysteresis_large_vs_loss',
                        f'The direction-dependent bias h = {h_out:+.4f} Nm is '
                        f'{100 * abs(h_out) / abs(median_loss):.0f}% of the median '
                        f'loss torque ({median_loss:.3f} Nm). This is not a '
                        f'cosmetic correction: uncorrected, it biases the forward '
                        f'and backdrive losses in opposite directions by that '
                        f'much, and any efficiency number taken from a log '
                        f'processed with torque_hysteresis=none carries it.')
        if hyst['n_keys'] and len(hyst['by_speed']) > 1:
            # Speed dependence is the physical signature: a Coulomb term is
            # roughly flat, a viscous one climbs. It also exposes the slow rows,
            # where the output turns under a revolution and the number is a
            # breakaway torque rather than steady friction.
            res.add('info', 'hysteresis_by_speed',
                    'Direction-dependent bias by input speed: '
                    + ', '.join(f'{v:g} rad/s -> {hv:+.3f} Nm'
                                for v, hv in hyst['by_speed'].items())
                    + '. A constant term is Coulomb friction in the measurement '
                      'path; the rise with speed is the viscous part. The slowest '
                      'rows turn the output less than a revolution per dwell, so '
                      'their value is a breakaway torque rather than a steady '
                      'one -- see the partial-revolution finding.')
        over = [d for d in live if d['mode'] in ('forward', 'backdrive')
                and d['eff'] > 1.0]
        if offset['width'] < 0:
            res.add('warn', 'cells_inconsistent',
                    f'No pair of torque-cell zeros can make this log physically '
                    f'consistent: the dwells demand c < {offset["hi"]:+.3f} Nm and '
                    f'c > {offset["lo"]:+.3f} Nm at once. The two cells therefore '
                    f'disagree by at least {abs(offset["width"]):.3f} Nm at the '
                    f'output, which is the floor on the error of every loss and '
                    f'efficiency number here. It matters most at light load, '
                    f'where it is a large fraction of the loss.')
        elif not (offset['lo'] <= applied_c <= offset['hi']):
            res.add('warn', 'offset_outside_feasible',
                    f'The applied cell zeros combine to c = {applied_c:+.3f} Nm, '
                    f'outside the physically feasible interval '
                    f'[{offset["lo"]:+.3f}, {offset["hi"]:+.3f}] Nm.')
        else:
            res.add('info', 'cells_consistent',
                    f'A combined cell offset anywhere in '
                    f'[{offset["lo"]:+.3f}, {offset["hi"]:+.3f}] Nm keeps every '
                    f'dwell physical; the applied zeros give c = {applied_c:+.3f} Nm. '
                    f'That interval width, {offset["width"]:.3f} Nm, is the '
                    f'calibration uncertainty on every loss torque below.')
        if over:
            worst = max(over, key=lambda d: d['eff'])
            res.add('warn', 'efficiency_above_100',
                    f'{len(over)} of {len(live)} dwells report efficiency above '
                    f'100% (worst {100 * worst["eff"]:.1f}% at '
                    f'{worst["v_cmd"]:g} rad/s input, {worst["t_cmd"]:g} Nm). They '
                    f'are left in the maps rather than clipped, because they are '
                    f'the visible part of the cell disagreement above.')

        # -- rotation coverage: is the mesh actually being averaged? ----------
        thin = []
        for v in sorted({abs(d['v_cmd']) for d in live}):
            revs = float(np.median([d['rev_out'] for d in live
                                    if abs(d['v_cmd']) == v]))
            if revs < p['min_output_revs']:
                thin.append((v, revs))
        if thin:
            res.add('warn', 'partial_output_revolution',
                    'At ' + ', '.join(f'{v:g} rad/s input ({r:.2f} rev)'
                                      for v, r in thin) +
                    f' the output turns less than {p["min_output_revs"]:g} '
                    f'revolution per dwell, so each of those points samples one '
                    f'patch of the mesh rather than averaging over it. Their '
                    f'scatter is mesh position, not repeatability.')

        # -- loss model -------------------------------------------------------
        # L = L0 + k*T_out: L0 is the no-load drag referred to the output and k
        # is the load-proportional share of the loss, so eta_forward tends to
        # 1/(1+k) at high load. Fitted above a minimum load because the
        # light-load points are where the cell disagreement lives.
        peak_load = max(abs(d['t_cmd']) for d in live)
        fit_floor = p['loss_fit_min_load_frac'] * peak_load
        fits = {}
        for mode in _MODES:
            sel = [d for d in live if d['mode'] == mode and d['load_Nm'] >= fit_floor]
            slope, intercept, r2 = _linfit([d['load_Nm'] for d in sel],
                                           [d['loss_Nm'] for d in sel])
            fits[mode] = {'slope': slope, 'intercept': intercept, 'r2': r2,
                          'n': len(sel)}

        # -- headline numbers -------------------------------------------------
        floor = p['headline_min_load_frac'] * peak_load
        head = {}
        for mode in _MODES:
            sel = [d for d in live if d['mode'] == mode and abs(d['t_cmd']) >= floor]
            effs = np.array([d['eff'] for d in sel], dtype=float)
            head[mode] = {
                'n': len(sel),
                'mean': float(np.nanmean(effs)) if effs.size else float('nan'),
                'std': _sd(effs),
                'peak': float(np.nanmax(effs)) if effs.size else float('nan'),
                'peak_at': (max(sel, key=lambda d: d['eff']) if sel else None),
            }

        # Direction asymmetry, restricted to the loads the headline trusts.
        asym = {}
        for mode in _MODES:
            pooled = self._pool(
                [d for d in live if d['mode'] == mode and abs(d['t_cmd']) >= floor],
                lambda d: (abs(d['v_cmd']), abs(d['t_cmd']), d['rotation']),
                lambda d: d['eff'])
            diffs = [pooled[(v, t, 'ccw')] - pooled[(v, t, 'cw')]
                     for (v, t, r) in pooled if r == 'cw'
                     and (v, t, 'ccw') in pooled]
            asym[mode] = float(np.mean(diffs)) if diffs else float('nan')

        # -- figures ----------------------------------------------------------
        attached = seg.attached_label('input') or seg.attached_label('output')
        labels = {'title': f'Gearbox Efficiency{" -- " + attached if attached else ""}'}
        res.figures.append(('map', self._plot_maps(live, ratio, labels)))
        res.figures.append(('vs_load', self._plot_curves(live, ratio, labels)))
        res.figures.append(('loss_torque',
                            self._plot_loss(live, fits, ratio, labels)))
        res.figures.append(('direction_asymmetry',
                            self._plot_asymmetry(live, ratio, labels)))
        res.figures.append(('cell_consistency',
                            self._plot_consistency(live, offset, applied_c,
                                                   labels)))

        # -- tables -----------------------------------------------------------
        cells = {}
        for d in live:
            cells.setdefault((d['v_cmd'], d['t_cmd']), []).append(d)
        rows = ['input_speed_cmd_rad_s,output_torque_cmd_Nm,n_dwells,'
                'output_speed_rad_s,input_torque_Nm,output_torque_Nm,'
                'input_power_W,output_power_W,mode,efficiency,'
                'loss_torque_Nm,loss_power_W,output_revs']
        for (v, t), ds in sorted(cells.items()):
            mean = lambda k: float(np.nanmean([d[k] for d in ds]))
            rows.append(
                f'{v:g},{t:g},{len(ds)},{mean("w_out"):.5f},{mean("t_in"):.5f},'
                f'{mean("t_out"):.4f},{mean("p_in"):.3f},{mean("p_out"):.3f},'
                f'{ds[0]["mode"]},{mean("eff"):.5f},{mean("loss_Nm"):.5f},'
                f'{mean("loss_W"):.3f},{mean("rev_out"):.3f}')
        res.tables.append(('per_cell', '\n'.join(rows) + '\n'))

        if p['per_dwell_csv']:
            rows = ['segment,setpoint,input_speed_cmd_rad_s,output_torque_cmd_Nm,'
                    'input_speed_rad_s,output_speed_rad_s,input_torque_Nm,'
                    'output_torque_Nm,input_torque_raw_Nm,output_torque_raw_Nm,'
                    'input_torque_sd_Nm,output_torque_sd_Nm,mode,rotation,'
                    'efficiency,loss_torque_Nm,output_revs']
            for d in sorted(live, key=lambda d: d['setpoint']):
                rows.append(
                    f'{d["raw_id"]},{d["setpoint"]},{d["v_cmd"]:g},{d["t_cmd"]:g},'
                    f'{d["w_in"]:.5f},{d["w_out"]:.6f},{d["t_in"]:.5f},'
                    f'{d["t_out"]:.4f},{d["t_in_raw"]:.5f},{d["t_out_raw"]:.4f},'
                    f'{d["t_in_sd"]:.5f},{d["t_out_sd"]:.4f},{d["mode"]},'
                    f'{d["rotation"]},{d["eff"]:.5f},{d["loss_Nm"]:.5f},'
                    f'{d["rev_out"]:.3f}')
            res.tables.append(('per_dwell', '\n'.join(rows) + '\n'))

        # -- metrics and summary ----------------------------------------------
        peak_fwd = head['forward']['peak_at']
        res.metrics = {
            'gear_ratio': ratio,
            'gear_ratio_source': ratio_source,
            'gear_ratio_from_encoders': measured_ratio,
            'input_torque_channel': in_cell,
            'output_torque_channel': out_cell,
            'torque_zero_method': p['torque_zero'],
            'input_torque_zero_Nm': z_in,
            'output_torque_zero_Nm': z_out,
            'input_zero_mirror_pairs': zeros['input']['n_pairs'],
            'output_zero_mirror_pairs': zeros['output']['n_pairs'],
            'input_zero_drift_Nm': zeros['input']['drift'],
            'output_zero_drift_Nm': zeros['output']['drift'],
            'torque_hysteresis_method': p['torque_hysteresis'],
            'output_hysteresis_applied_Nm': h_out,
            'output_hysteresis_estimate_Nm': hyst['h'],
            'output_hysteresis_pos_group_Nm': hyst['h_pos'],
            'output_hysteresis_neg_group_Nm': hyst['h_neg'],
            'output_hysteresis_group_disagreement_Nm': hyst['disagree'],
            'output_hysteresis_n_keys': hyst['n_keys'],
            'output_hysteresis_spread_Nm': hyst['spread'],
            'output_hysteresis_by_input_speed_Nm': hyst['by_speed'],
            'median_loss_torque_Nm': median_loss,
            'combined_cell_offset_feasible_width_uncorrected_Nm': width_before,
            'n_dwells': len(live),
            'n_grid_cells': len(cells),
            'n_forward': counts['forward'],
            'n_backdrive': counts['backdrive'],
            'n_dissipative': counts['dissipative'],
            'n_non_physical': counts['non_physical'],
            'n_efficiency_above_100': len(over),
            'speed_setpoints_rad_s': sorted({d['v_cmd'] for d in live}),
            'torque_setpoints_Nm': sorted({d['t_cmd'] for d in live}),
            'headline_min_load_Nm': floor,
            'efficiency_forward_mean': head['forward']['mean'],
            'efficiency_forward_std': head['forward']['std'],
            'efficiency_forward_peak': head['forward']['peak'],
            'efficiency_forward_peak_at_input_rad_s': (
                peak_fwd['v_cmd'] if peak_fwd else None),
            'efficiency_forward_peak_at_output_Nm': (
                peak_fwd['t_cmd'] if peak_fwd else None),
            'efficiency_backdrive_mean': head['backdrive']['mean'],
            'efficiency_backdrive_std': head['backdrive']['std'],
            'efficiency_backdrive_peak': head['backdrive']['peak'],
            'asymmetry_forward_ccw_minus_cw': asym['forward'],
            'asymmetry_backdrive_ccw_minus_cw': asym['backdrive'],
            'loss_no_load_forward_Nm': fits['forward']['intercept'],
            'loss_slope_forward': fits['forward']['slope'],
            'loss_fit_r2_forward': fits['forward']['r2'],
            'loss_no_load_backdrive_Nm': fits['backdrive']['intercept'],
            'loss_slope_backdrive': fits['backdrive']['slope'],
            'loss_fit_r2_backdrive': fits['backdrive']['r2'],
            'combined_cell_offset_applied_Nm': applied_c,
            'combined_cell_offset_feasible_lo_Nm': offset['lo'],
            'combined_cell_offset_feasible_hi_Nm': offset['hi'],
            'combined_cell_offset_feasible_width_Nm': offset['width'],
        }

        band = (f'a {offset["width"]:.2f} Nm window of cell offset is still '
                f'consistent with the data'
                if offset['width'] >= 0 else
                f'no pair of cell zeros reconciles them, so they disagree by at '
                f'least {abs(offset["width"]):.2f} Nm at the output')
        hyst_note = (
            f' The output cell also carries a {h_out:+.2f} Nm bias that flips '
            f'sign with rotation, which the mirror tare cannot see; removing it '
            f'closed {100 * (offset["width"] - width_before) / abs(width_before):.0f}% '
            f'of the gap between the two cells.' if h_out else '')
        res.summary = (
            f'{len(live)} dwells over {len(cells)} grid cells at {ratio:g}:1, '
            f'every operating point measured both forward and backdriven. Above '
            f'{floor:.0f} Nm of output torque efficiency is '
            f'{100 * head["forward"]["mean"]:.1f}% +/- '
            f'{100 * head["forward"]["std"]:.1f} forward, peaking at '
            f'{100 * head["forward"]["peak"]:.1f}%, and '
            f'{100 * head["backdrive"]["mean"]:.1f}% +/- '
            f'{100 * head["backdrive"]["std"]:.1f} backdriven. The forward loss '
            f'model is L = {fits["forward"]["intercept"]:.2f} Nm + '
            f'{100 * fits["forward"]["slope"]:.1f}% of output torque. Every '
            f'number here reads both cells, so it carries their disagreement, '
            f'and the light-load end is where that hurts: {band}, which is why '
            f'{len(over)} dwell(s) read above 100%.{hyst_note}')
        return res
