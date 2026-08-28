"""Gearbox backlash and torsional stiffness from a torque sawtooth at stepped
output positions.

The experiment: the output is held at a fixed position while the input runs a
bipolar torque sweep; the output then steps to a new position and the sweep
repeats, sampling backlash at many mesh positions. Between sweeps the input
torque command sits at zero, which both separates the sweeps and gives a rest
window to tare the torque cells against.

Either sweep shape works, but they are not equally good. A plain sawtooth
(0 -> +T -> -T -> 0) leaves the + flank traversed once more while unloading than
while loading, and friction then biases that flank's fit; a peak-anchored sweep
(0 -> +T, then +T <-> -T, then 0) covers both flanks in both ramp directions.
The findings below quantify the difference when it matters.

Everything lands in one logged span -- a trace CSV carries no log-flag column --
so this processor owns all the segmentation:

    1. find the zero-torque rest windows; the gaps between them are the sweeps
    2. within a sweep, split into ramp legs by the sign of d(torque_cmd)/dt
    3. drop the first leg: it engages from an unknown backlash state at rest
    4. quantify the loop per sweep, then aggregate across mesh positions

Frames. Positions are always reported in the OUTPUT frame: the input encoder
reads the motor shaft, so it is divided by the gear ratio before differencing
against the output encoder. Deflection is

    deflection = input_position / gear_ratio - output_position

i.e. positive when the input leads the output, which is the sign a positive
input torque produces. Torque is measured on the input cell and reported in
whichever frame `torque_frame` selects; 'output' multiplies by the gear ratio.
That is a kinematic conversion of the input cell, NOT the output cell's reading
-- see the note on `torque_frame` in the params block.

Quantifying the loop. The two engaged tooth flanks (|T| above a fraction of
peak) are fitted and the backlash is the gap between their zero-torque
intercepts, i.e. the loop is modelled as a parallelogram. Precise, and the only
source of torsional stiffness, but it extrapolates the stiff flanks across the
soft region near zero torque, so it reports total lost motion rather than the
geometric tooth gap. Refits over a narrower and a wider band bound how much
that model choice is moving the number.
"""

import numpy as np

from ..processor import Applicability, Processor, Result
from ..registry import register

RAD_TO_ARCMIN = 60.0 * 180.0 / np.pi        # 3437.75 arc-min per radian

# The input encoder is logged as '<prefix>_output_position' on this rig -- the
# drive's position *output*, which is the motor shaft, upstream of the gearbox.
# It is not the gearbox output. `_position` is the fallback for ports whose
# encoder is logged plainly. Order matters; the implied-ratio check below is
# what catches it if this assumption ever stops holding.
_POSITION_SUFFIXES = ('_output_position', '_position')


def _linfit(x, y):
    """Linear fit y = slope*x + intercept; returns (slope, intercept, r2)."""
    if x.size < 2 or np.ptp(x) == 0:
        return float('nan'), float('nan'), float('nan')
    slope, intercept = np.polyfit(x, y, 1)
    ss_res = float(np.sum((y - (slope * x + intercept)) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return (float(slope), float(intercept),
            1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan'))


def _smooth(x, n):
    n = max(int(n), 1)
    if n <= 1 or x.size < n:
        return x.astype(float)
    return np.convolve(x, np.ones(n) / n, mode='same')


def _runs(mask):
    """[(start, end), ...] for each contiguous True run in a boolean mask."""
    if mask.size == 0:
        return []
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    bounds = np.concatenate(([0], edges + 1, [mask.size]))
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if mask[a]]


@register
class Backlash(Processor):
    name = 'backlash'
    title = 'Gearbox Backlash & Torsional Stiffness'
    description = (
        'Measures gearbox backlash and output-referred torsional stiffness from '
        'an input torque sawtooth repeated at stepped output positions. Splits '
        'the span into per-position sweeps, builds the deflection-vs-torque '
        'hysteresis loop for each, and quantifies the loop by flank-fit '
        'extrapolation to zero torque.'
    )
    granularity = 'run'

    params = {
        # -- roles and channels (all auto-detected; override to force) -------
        'input_device':          (None, str, 'Device running the torque sweep (auto-detected)'),
        'output_device':         (None, str, 'Device holding position (auto-detected)'),
        'input_position_channel': (None, str, 'Input-side encoder channel (auto-detected)'),
        'output_position_channel': (None, str, 'Output-side encoder channel (auto-detected)'),
        'input_torque_channel':  (None, str, 'Input torque cell (default: the input port\'s cell)'),

        # -- frames ---------------------------------------------------------
        # Positions are unconditionally output-frame; only torque is optional.
        # 'output' scales the INPUT cell by the gear ratio rather than reading
        # the output cell, because the input cell is the one in the deflection
        # measurement's own load path and is the better-conditioned sensor here
        # (20 Nm FS carrying ~1.7 Nm, against 500 Nm FS carrying ~50 Nm). The
        # output cell is still read, but only to cross-check the ratio.
        'torque_frame':       ('input', str,   "Torque axis frame: 'input' or 'output' (input cell x gear ratio)"),
        'gear_ratio':         (None,   float, 'Input:output reduction (default: from the log config)'),

        # -- flank fit --------------------------------------------------------
        'edge_frac':          (0.25,   float, 'Fit flanks above this fraction of peak |torque|'),

        # -- segmentation ---------------------------------------------------
        'rest_torque_thresh': (0.005,  float, '|input torque command| below this counts as rest [Nm]'),
        'rest_min_s':         (0.5,    float, 'Minimum duration of a rest window [s]'),
        'ramp_smooth_s':      (0.2,    float, 'Smoothing window for ramp-direction detection [s]'),
        'min_leg_samples':    (200,    int,   'Minimum samples for a ramp leg to be kept'),
        'drop_first_leg':     (False,  bool,  'Drop each sweep\'s first ramp leg (first engagement from rest)'),
        'drop_first_sweep':   (True,   bool,  'Drop the first sweep (first-cycle settling artifact)'),

        # -- reporting ------------------------------------------------------
        'min_band_samples':   (20,     int,   'Minimum samples per branch for a flank to count as covered'),
        'position_tol_rad':   (0.05,   float, 'Tolerance for matching up-sweep and down-sweep mesh positions [rad]'),
        'ratio_tol_pct':      (2.0,    float, 'Config-vs-measured gear ratio disagreement that warns [%]'),
        'min_flank_r2':       (0.95,   float, 'Flank fit R2 below this warns that the loop is not a parallelogram'),
        'band_sensitivity_pct': (15.0, float, "Spread in 'edges' across fit bands that warns the model is wrong [%]"),
        'per_sweep_plots':    (True,   bool,  'Write a per-sweep loop plot into runs/'),
        'csv_decimate':       (10,     int,   'Keep every Nth sample in the exported loop CSV'),
    }

    # -- channel discovery -------------------------------------------------

    def _position_channel(self, seg, prefix):
        for suffix in _POSITION_SUFFIXES:
            if seg.has(prefix + suffix):
                return prefix + suffix
        return None

    def _roles(self, seg):
        """Identify the driving (torque-swept) port and the holding port.

        Detected from the command channels rather than read from `ports`,
        because a command channel is NaN whenever its mode is inactive -- so
        'which motor was in torque mode' is a fact in the data, not a
        convention. `ports` is consulted only for channel prefixes and labels.
        """
        driver = holder = None
        for dev in seg.devices():
            prefix = seg.prefix(dev)
            if seg.is_active(f'{prefix}_torque_command'):
                driver = driver or (dev, prefix)
            if seg.is_active(f'{prefix}_position_command'):
                holder = holder or (dev, prefix)
        return driver, holder

    def applies_to(self, segs):
        seg = segs[0]
        d = self.defaults()
        driver, holder = self._roles(seg)

        if driver is None:
            return Applicability('no', 'no motor is running a torque command')
        if holder is None:
            return Applicability('no', 'no motor is holding a position command; '
                                       'nothing is stepping the mesh position')
        if driver[0] == holder[0]:
            return Applicability('no', f'{driver[0]} carries both a torque and a '
                                       'position command; cannot assign roles')

        in_dev, in_prefix = driver
        out_dev, out_prefix = holder
        in_pos = self._position_channel(seg, in_prefix)
        out_pos = self._position_channel(seg, out_prefix)
        if in_pos is None or out_pos is None:
            return Applicability(
                'no', f'need a position channel on both ports; found '
                      f'{in_pos or "none"} / {out_pos or "none"}')

        ratio = self._config_ratio(seg, in_dev, out_dev)
        if ratio is None or not np.isfinite(ratio):
            return Applicability('no', f'no usable gear_ratio for {in_dev}')

        tcmd = np.concatenate([s[f'{in_prefix}_torque_command'] for s in segs])
        if not (np.nanmax(tcmd) > 0 > np.nanmin(tcmd)):
            return Applicability(
                'no', 'the input torque command never changes sign; a backlash '
                      'loop needs both tooth flanks loaded')

        rests = self._rest_windows(segs, in_prefix, d['rest_torque_thresh'],
                                   d['rest_min_s'])
        if len(rests) < 2:
            return Applicability(
                'maybe' if rests else 'no',
                f'found {len(rests)} zero-torque rest window(s); need at least 2 '
                'to bound a sweep')

        pos_cmd = np.concatenate([s[f'{out_prefix}_position_command'] for s in segs])
        span = float(np.nanmax(pos_cmd) - np.nanmin(pos_cmd))
        return Applicability(
            'yes',
            f'{in_dev} sweeps torque ({np.nanmin(tcmd):.2f} to {np.nanmax(tcmd):.2f} Nm) '
            f'while {out_dev} holds position over {span:.2f} rad; '
            f'{len(rests) - 1} sweeps at {ratio:g}:1',
            {'input_device': in_dev, 'output_device': out_dev,
             'input_position_channel': in_pos, 'output_position_channel': out_pos})

    # -- segmentation ------------------------------------------------------

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

    def _rest_windows(self, segs, in_prefix, thresh, min_s):
        """Sustained zero-torque-command windows, as (start, end) index pairs."""
        tcmd = np.concatenate([s[f'{in_prefix}_torque_command'] for s in segs])
        t = np.concatenate([s['time'] for s in segs])
        at_rest = np.abs(np.nan_to_num(tcmd, nan=np.inf)) < thresh
        return [(a, b) for a, b in _runs(at_rest) if t[b - 1] - t[a] >= min_s]

    def _legs(self, tcmd, smooth_n, min_leg):
        """Split a sweep into monotonic ramp legs.

        Returns (legs, moving) where legs is [(start, end, direction)] and
        `moving` marks the samples where the command is actually ramping.

        Sign of the smoothed command rate, with near-zero rates carried forward
        from the previous sample so that a momentary flat spot does not shatter
        a leg into fragments. The flat samples stay inside their leg for
        direction bookkeeping but are reported separately, because a dwell at
        peak torque is a dense cluster of points at one end of the flank and
        would drag a least-squares fit toward it. The test builder's sawtooth
        pattern has a `peak_dwell_s` that does exactly this.
        """
        rate = _smooth(np.gradient(tcmd), smooth_n)
        peak = float(np.max(np.abs(rate))) if rate.size else 0.0
        if peak == 0:
            return [], np.zeros(tcmd.size, dtype=bool)
        moving = np.abs(rate) > 0.1 * peak
        sign = np.where(moving, np.sign(rate), 0.0)
        if not np.any(moving):
            return [], moving
        # Fill the near-zero-rate samples from their neighbours so that a
        # momentary flat spot -- including the smoothing skirt at the very start
        # of a sweep -- does not shatter a leg into fragments or invent a
        # spurious zero-signed leg ahead of the first real one.
        idx = np.maximum.accumulate(np.where(moving, np.arange(sign.size), 0))
        sign = sign[idx]
        sign[:np.argmax(moving)] = sign[np.argmax(moving)]

        legs, start = [], 0
        for i in np.flatnonzero(np.diff(sign)) + 1:
            legs.append((start, int(i)))
            start = int(i)
        legs.append((start, int(tcmd.size)))
        return ([(a, b, float(sign[a])) for a, b in legs if b - a >= min_leg],
                moving)

    # -- loop quantification ------------------------------------------------

    def _fit_edges(self, torque, deflection, keep, edge_frac):
        """Flank fits, split by torque SIGN.

        Sign rather than ramp direction is correct here: once a tooth flank is
        engaged the loading and unloading passes ride the same elastic branch,
        so the loop's two long sides are the +flank and the -flank contacts.
        """
        peak = float(np.max(np.abs(torque)))
        thr = edge_frac * peak
        pos, neg = keep & (torque > thr), keep & (torque < -thr)
        sp, ip, r2p = _linfit(torque[pos], deflection[pos])
        sn, inn, r2n = _linfit(torque[neg], deflection[neg])
        return {'slope_pos': sp, 'intercept_pos': ip, 'r2_pos': r2p,
                'slope_neg': sn, 'intercept_neg': inn, 'r2_neg': r2n,
                'n_pos': int(pos.sum()), 'n_neg': int(neg.sum()),
                'threshold_Nm': thr,
                'backlash_arcmin': ip - inn,
                'compliance_arcmin_per_Nm': 0.5 * (sp + sn)}

    # -- plotting -----------------------------------------------------------

    def _plot_overlay(self, sweeps, p, labels):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))
        positions = [s['output_position_rad'] for s in sweeps]
        # Absolute output position is an arbitrary encoder offset; only the span
        # swept matters, so zero the scale at the first (lowest) hold position.
        pos0 = min(positions)
        norm = plt.Normalize(0.0, max(positions) - pos0)
        cmap = plt.cm.viridis
        for s in sweeps:
            ax.plot(s['torque'], s['deflection'], '-', linewidth=0.8, alpha=0.8,
                    color=cmap(norm(s['output_position_rad'] - pos0)))
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                     label='Output hold position (rad from first, output frame)')

        head = np.array([s['backlash_arcmin'] for s in sweeps], dtype=float)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlabel(labels['torque'])
        ax.set_ylabel('Deflection (arc-min, output frame)')
        ax.set_title(
            f'{labels["title"]}\n'
            f'{len(sweeps)} mesh positions, {p["gear_ratio"]:g}:1  |  '
            f'backlash = '
            f'{np.nanmean(head):.2f} +/- {np.nanstd(head, ddof=1) if head.size > 1 else 0:.2f} arc-min  |  '
            f'stiffness = {np.nanmean([s["stiffness_Nm_per_rad"] for s in sweeps]):.0f} Nm/rad')
        ax.grid(True, alpha=0.4)
        fig.tight_layout()
        return fig

    def _plot_summary(self, sweeps, p, labels):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
        pos = np.array([s['output_position_rad'] for s in sweeps])
        pos = pos - pos.min()  # zero at the first hold position, as in the overlay
        up = np.array([s['step_direction'] == 'up' for s in sweeps])

        for ax, key, ylabel in (
                (axes[0], 'backlash_arcmin', 'Backlash (arc-min)'),
                (axes[1], 'stiffness_Nm_per_rad',
                 'Output torsional\nstiffness (Nm/rad)')):
            v = np.array([s[key] for s in sweeps], dtype=float)
            ax.plot(pos[up], v[up], 'o-', color='tab:blue', label='stepping up')
            ax.plot(pos[~up], v[~up], 's--', color='tab:orange', label='stepping down')
            mean = np.nanmean(v)
            ax.axhline(mean, color='0.4', linewidth=1, linestyle=':',
                       label=f'mean = {mean:.3g}')
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.4)
            ax.legend(fontsize=8)

        axes[1].set_xlabel('Output hold position (rad from first, output frame)')
        axes[0].set_title(f'{labels["title"]} -- variation with mesh position')
        fig.tight_layout()
        return fig

    def _plot_sweep(self, s, p, labels):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 7))
        x, y, edges = s['torque'], s['deflection'], s['edges']
        ax.plot(x, y, '-', color='0.6', linewidth=0.7, alpha=0.8, label='loop')

        thr = edges['threshold_Nm']
        pos, neg = x > thr, x < -thr
        ax.scatter(x[pos], y[pos], s=5, color='limegreen', alpha=0.4, label='+ flank (fit)')
        ax.scatter(x[neg], y[neg], s=5, color='red', alpha=0.4, label='- flank (fit)')

        ip, inn = s['edges_centered']
        xr, xl = np.array([0, float(np.max(x))]), np.array([float(np.min(x)), 0])
        ax.plot(xr, edges['slope_pos'] * xr + ip, color='green', linewidth=2)
        ax.plot(xl, edges['slope_neg'] * xl + inn, color='darkred', linewidth=2)
        ax.plot([0, 0], [inn, ip], 'k-', linewidth=1)
        ax.plot([0, 0], [inn, ip], 'o', color='k', markersize=4)
        ax.annotate(f'backlash = {edges["backlash_arcmin"]:.2f} arc-min',
                    (0, 0.5 * (ip + inn)), textcoords='offset points',
                    xytext=(12, 6), fontsize=10, fontweight='bold')

        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_xlabel(labels['torque'])
        ax.set_ylabel('Deflection (arc-min, output frame)')
        ax.set_title(
            f'Sweep {s["index"]:02d} -- output {s["output_position_rad"]:.3f} rad '
            f'(stepping {s["step_direction"]})\n'
            f'stiffness = {s["stiffness_Nm_per_rad"]:.0f} Nm/rad, '
            f'R2(+) = {edges["r2_pos"]:.3f}  R2(-) = {edges["r2_neg"]:.3f}')
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    # -- main ---------------------------------------------------------------

    def run(self, segs, params):
        p = dict(params)
        res = Result()

        seg = segs[0]
        in_dev, out_dev = p['input_device'], p['output_device']
        in_pos_ch, out_pos_ch = p['input_position_channel'], p['output_position_channel']
        if not (in_dev and out_dev and in_pos_ch and out_pos_ch):
            raise ValueError('input/output device and position channels are '
                             'unset; applies_to normally supplies them')
        in_prefix, out_prefix = seg.prefix(in_dev), seg.prefix(out_dev)
        cat = lambda ch: np.concatenate([s[ch] for s in segs])

        if p['torque_frame'] not in ('input', 'output'):
            raise ValueError("torque_frame must be 'input' or 'output', "
                             f"got {p['torque_frame']!r}")

        # -- gear ratio: config value, cross-checked against the kinematics ---
        cfg_ratio = self._config_ratio(seg, in_dev, out_dev)
        ratio = float(p['gear_ratio']) if p['gear_ratio'] else cfg_ratio
        ratio_source = ('gear_ratio parameter override' if p['gear_ratio']
                        else f'log resolved_config ({in_dev}.gear_ratio)')

        in_pos_raw, out_pos = cat(in_pos_ch), cat(out_pos_ch)
        span_out = float(np.nanmax(out_pos) - np.nanmin(out_pos))
        implied = (float(np.nanmax(in_pos_raw) - np.nanmin(in_pos_raw)) / span_out
                   if span_out > 1e-6 else float('nan'))
        if np.isfinite(implied) and abs(implied - ratio) / ratio > p['ratio_tol_pct'] / 100:
            res.add('warn', 'gear_ratio_mismatch',
                    f'The travel in {in_pos_ch} vs {out_pos_ch} implies '
                    f'{implied:.3f}:1, but the analysis is using {ratio:g}:1 from '
                    f'the {ratio_source}. Every deflection, backlash and stiffness '
                    f'number below scales with this ratio -- reconcile it before '
                    f'quoting them.')
        else:
            res.add('info', 'gear_ratio_confirmed',
                    f'Using {ratio:g}:1 from the {ratio_source}; the logged '
                    f'travel implies {implied:.3f}:1.')

        # -- torque channel and frame ----------------------------------------
        in_cell = (p['input_torque_channel']
                   or (seg.port_by_role('input') or {}).get('cell')
                   or f'{in_prefix}_torque')
        out_cell = (seg.port_by_role('output') or {}).get('cell') or f'{out_prefix}_torque'
        if not seg.has(in_cell):
            raise ValueError(f'input torque cell {in_cell!r} is not in this log')

        # Tare on the FIRST rest window only. Later rest windows sit at a fresh
        # hold position with real wound-up holding torque in the train, so they
        # are not zero-torque references even though the command is zero.
        rests = self._rest_windows(segs, in_prefix, p['rest_torque_thresh'],
                                   p['rest_min_s'])
        torque_in = cat(in_cell)
        zero_ref = float(np.nanmean(torque_in[rests[0][0]:rests[0][1]]))
        torque_in = torque_in - zero_ref

        scale = ratio if p['torque_frame'] == 'output' else 1.0
        torque_axis = torque_in * scale
        torque_label = ('Input torque referred to output (Nm, = input cell x ratio)'
                        if p['torque_frame'] == 'output'
                        else f'Input torque (Nm, {in_cell}, tared)')

        # -- deflection, in the output frame ---------------------------------
        deflection = (in_pos_raw / ratio - out_pos) * RAD_TO_ARCMIN

        # -- per-sweep -------------------------------------------------------
        smooth_n = max(int(round(p['ramp_smooth_s'] / seg.dt)), 1)
        tcmd = cat(f'{in_prefix}_torque_command')
        torque_out = cat(out_cell) if seg.has(out_cell) else None
        sweeps, dropped = [], []
        prev_pos = None

        for i in range(len(rests) - 1):
            a, b = rests[i][1], rests[i + 1][0]
            if b - a < p['min_leg_samples']:
                continue
            x, y = torque_axis[a:b], deflection[a:b]
            legs, moving = self._legs(tcmd[a:b], smooth_n, p['min_leg_samples'])
            if len(legs) < 2:
                dropped.append({'index': i, 'why': f'{len(legs)} usable ramp legs'})
                continue

            keep = np.zeros(x.size, dtype=bool)
            direction = np.zeros(x.size)
            for j, (la, lb, sgn) in enumerate(legs):
                direction[la:lb] = sgn
                if j == 0 and p['drop_first_leg']:
                    continue
                keep[la:lb] = True
            # Dwells belong to no branch: at a peak they pile points onto one
            # end of a flank fit, and at a reversal the cell is still settling.
            keep &= moving

            if p['drop_first_sweep'] and i == 0:
                dropped.append({'index': i, 'why': 'first sweep (settling)'})
                prev_pos = float(np.mean(out_pos[a:b]))
                continue

            # Does each flank get traversed in both ramp directions? Friction
            # displaces the loading and unloading passes in opposite directions,
            # so a flank fitted from both passes has that displacement averaged
            # out, while one fitted from a single pass does not. Fitting the two
            # flanks under different coverage makes them incomparable -- and the
            # cure is a profile that covers both, not a reweighting here.
            thr_cov = p['edge_frac'] * float(np.max(np.abs(x)))
            balanced = all(
                (keep & side & (direction == d)).sum() >= p['min_band_samples']
                for side in (x > thr_cov, x < -thr_cov) for d in (1.0, -1.0))
            edges = self._fit_edges(x, y, keep, p['edge_frac'])
            # Refit the flanks over a narrower and a wider band. If the loop
            # really is a parallelogram the intercept gap does not care which
            # part of a straight flank produced it, so any spread here is the
            # model failing rather than the measurement scattering. This is the
            # cheapest available test of the parallelogram assumption.
            band_probe = [self._fit_edges(x, y, keep, f)['backlash_arcmin']
                          for f in (p['edge_frac'] / 2,
                                    min(2 * p['edge_frac'], 0.8))]
            if not np.isfinite(edges['backlash_arcmin']):
                dropped.append({'index': i, 'why': 'flank fit failed'})
                continue

            # Output-side torsional stiffness. compliance is arc-min per Nm on
            # the torque axis in use, so undo the frame scaling before
            # converting: K_out = T_out / theta_out = ratio * T_in / theta_out.
            #
            # This is a CHORD stiffness across the fitted flank band, not the
            # small-signal stiffness at zero load. Where the flanks stiffen with
            # load the two differ substantially, and the chord is the honest
            # answer because it is what the fit measured -- but it moves with
            # `edge_frac`, exactly as the backlash intercept does.
            compliance_rad_per_Nm_in = (edges['compliance_arcmin_per_Nm']
                                        / RAD_TO_ARCMIN * scale)
            stiffness = (ratio / compliance_rad_per_Nm_in
                         if compliance_rad_per_Nm_in != 0 else float('nan'))
            # Per-flank stiffness. The two tooth flanks are different contacts
            # and need not be equally stiff; when they are not, the loop sits
            # asymmetrically about zero deflection. Keeping them apart is what
            # distinguishes that from a torque-zero error, which would shift the
            # loop sideways without changing either slope.
            stiff_pair = tuple(
                ratio / (s / RAD_TO_ARCMIN * scale) if s else float('nan')
                for s in (edges['slope_pos'], edges['slope_neg']))

            position = float(np.mean(out_pos[a:b]))
            step_dir = 'up' if prev_pos is None or position >= prev_pos else 'down'
            prev_pos = position

            # Centre the loop on its elastic neutral axis so the overlay is not
            # smeared out by each sweep's encoder-alignment offset. This is a
            # plotting correction only; every metric is a difference and so is
            # already immune to it.
            centre = 0.5 * (edges['intercept_pos'] + edges['intercept_neg'])

            sweeps.append({
                'index': i,
                'output_position_rad': position,
                'step_direction': step_dir,
                'n': int(b - a),
                'edges': edges,
                'band_probe': band_probe,
                'branches_balanced': balanced,
                'edges_centered': (edges['intercept_pos'] - centre,
                                   edges['intercept_neg'] - centre),
                'stiffness_Nm_per_rad': stiffness,
                'stiffness_pos_Nm_per_rad': stiff_pair[0],
                'stiffness_neg_Nm_per_rad': stiff_pair[1],
                'backlash_arcmin': edges['backlash_arcmin'],
                'torque': x,
                'deflection': y - centre,
                'output_torque_span_Nm': (
                    float(np.nanmax(torque_out[a:b]) - np.nanmin(torque_out[a:b]))
                    if torque_out is not None else float('nan')),
                'input_torque_span_Nm': float(np.nanmax(x) - np.nanmin(x)) / scale,
            })

        if not sweeps:
            why = '; '.join(f'{d["index"]}: {d["why"]}' for d in dropped)
            raise ValueError(f'no usable sweeps ({why or "none found"})')

        # -- aggregate --------------------------------------------------------
        head = np.array([s['backlash_arcmin'] for s in sweeps])
        stiff = np.array([s['stiffness_Nm_per_rad'] for s in sweeps])
        positions = np.array([s['output_position_rad'] for s in sweeps])
        sd = lambda v: float(np.nanstd(v, ddof=1)) if np.sum(np.isfinite(v)) > 1 else float('nan')

        stiff_pos = float(np.nanmean([s['stiffness_pos_Nm_per_rad'] for s in sweeps]))
        stiff_neg = float(np.nanmean([s['stiffness_neg_Nm_per_rad'] for s in sweeps]))
        flank_asym = (200 * abs(stiff_pos - stiff_neg) / (stiff_pos + stiff_neg)
                      if stiff_pos + stiff_neg else float('nan'))

        probe_lo = float(np.nanmean([s['band_probe'][0] for s in sweeps]))
        probe_hi = float(np.nanmean([s['band_probe'][1] for s in sweeps]))
        head_mean = float(np.nanmean(head))
        band_sensitivity = (100 * abs(probe_hi - probe_lo) / abs(head_mean)
                            if head_mean else float('nan'))

        # Up-and-down position schedules revisit the same mesh positions, which
        # is a free repeatability check: the spread between the two visits is
        # measurement scatter, while the spread across positions is mesh
        # variation plus that scatter.
        repeats = []
        ups = [s for s in sweeps if s['step_direction'] == 'up']
        downs = [s for s in sweeps if s['step_direction'] == 'down']
        for u in ups:
            for dn in downs:
                if abs(u['output_position_rad'] - dn['output_position_rad']) <= p['position_tol_rad']:
                    repeats.append(abs(u['backlash_arcmin'] - dn['backlash_arcmin']))
                    break

        metrics = {
            'gear_ratio': ratio,
            'gear_ratio_source': ratio_source,
            'gear_ratio_implied_by_travel': implied,
            'input_torque_channel': in_cell,
            'torque_frame': p['torque_frame'],
            'input_position_channel': in_pos_ch,
            'output_position_channel': out_pos_ch,
            'input_torque_zero_Nm': zero_ref,
            # The commanded sawtooth amplitude. Recorded because the report
            # states it in prose and the alternative is parsing it back out of
            # `selection_reason`, which is a sentence and not an interface.
            'cmd_torque_min_Nm': float(np.nanmin(tcmd)),
            'cmd_torque_max_Nm': float(np.nanmax(tcmd)),
            'n_sweeps_found': len(rests) - 1,
            'n_sweeps_used': len(sweeps),
            'n_sweeps_dropped': len(dropped),
            'output_position_min_rad': float(positions.min()),
            'output_position_max_rad': float(positions.max()),
            'backlash_arcmin_mean': head_mean,
            'backlash_arcmin_std': sd(head),
            'backlash_arcmin_min': float(np.nanmin(head)),
            'backlash_arcmin_max': float(np.nanmax(head)),
            'backlash_arcmin_at_half_band': probe_lo,
            'backlash_arcmin_at_double_band': probe_hi,
            'edges_band_sensitivity_pct': band_sensitivity,
            'stiffness_Nm_per_rad_mean': float(np.nanmean(stiff)),
            'stiffness_Nm_per_rad_std': sd(stiff),
            'stiffness_pos_flank_Nm_per_rad': stiff_pos,
            'stiffness_neg_flank_Nm_per_rad': stiff_neg,
            'flank_stiffness_asymmetry_pct': flank_asym,
            'compliance_arcmin_per_Nm_mean': float(np.nanmean(
                [s['edges']['compliance_arcmin_per_Nm'] for s in sweeps])),
            'updown_repeatability_arcmin': (float(np.mean(repeats)) if repeats
                                            else None),
            'n_repeated_positions': len(repeats),
            'flank_r2_min': float(np.nanmin(
                [min(s['edges']['r2_pos'], s['edges']['r2_neg']) for s in sweeps])),
            'sweeps': [{
                'index': s['index'],
                'output_position_rad': s['output_position_rad'],
                'step_direction': s['step_direction'],
                'backlash_arcmin': s['backlash_arcmin'],
                'stiffness_Nm_per_rad': s['stiffness_Nm_per_rad'],
                'r2_pos': s['edges']['r2_pos'],
                'r2_neg': s['edges']['r2_neg'],
            } for s in sweeps],
        }

        for d in dropped:
            res.add('info', 'sweep_dropped',
                    f'Sweep {d["index"]} excluded: {d["why"]}.')

        # -- figures and tables ------------------------------------------------
        attached = seg.attached_label('input') or seg.attached_label('output')
        labels = {'torque': torque_label,
                  'title': f'Gearbox Backlash{" -- " + attached if attached else ""}'}
        res.figures.append(('overlay', self._plot_overlay(sweeps, dict(p, gear_ratio=ratio), labels)))
        if len(sweeps) > 1:
            res.figures.append(('by_position', self._plot_summary(sweeps, p, labels)))
        if p['per_sweep_plots']:
            for s in sweeps:
                res.figures.append((f'runs/sweep_{s["index"]:02d}',
                                    self._plot_sweep(s, p, labels)))

        rows = ['sweep,output_position_rad,step_direction,backlash_arcmin,'
                'stiffness_Nm_per_rad,compliance_arcmin_per_Nm,r2_pos,r2_neg']
        for s in sweeps:
            e = s['edges']
            rows.append(
                f'{s["index"]},{s["output_position_rad"]:.6f},{s["step_direction"]},'
                f'{e["backlash_arcmin"]:.5f},'
                f'{s["stiffness_Nm_per_rad"]:.2f},{e["compliance_arcmin_per_Nm"]:.6f},'
                f'{e["r2_pos"]:.5f},{e["r2_neg"]:.5f}')
        res.tables.append(('per_sweep', '\n'.join(rows) + '\n'))

        step = max(int(p['csv_decimate']), 1)
        loop_rows = [f'sweep,output_position_rad,torque_{p["torque_frame"]}_Nm,'
                     'deflection_arcmin']
        for s in sweeps:
            for x, y in zip(s['torque'][::step], s['deflection'][::step]):
                loop_rows.append(f'{s["index"]},{s["output_position_rad"]:.6f},'
                                 f'{x:.6f},{y:.6f}')
        res.tables.append(('loops', '\n'.join(loop_rows) + '\n'))

        res.summary = (
            f'{len(sweeps)} usable sweeps across '
            f'{positions.min():.2f}-{positions.max():.2f} rad of output travel at '
            f'{ratio:g}:1. Backlash from the flank fits is '
            f'{metrics["backlash_arcmin_mean"]:.2f} +/- {metrics["backlash_arcmin_std"]:.2f} '
            f'arc-min (halving and doubling the fit band reads '
            f'{probe_lo:.2f} / {probe_hi:.2f}); '
            f'output-referred torsional stiffness is '
            f'{metrics["stiffness_Nm_per_rad_mean"]:.0f} +/- '
            f'{metrics["stiffness_Nm_per_rad_std"]:.0f} Nm/rad. Positions are in '
            f'the output frame; the torque axis is the input cell'
            + (' referred to the output.' if p['torque_frame'] == 'output' else '.'))

        res.metrics = metrics
        return res
