"""Current -> torque characterization (blocked-rotor / stationary conditions).

With the rotor stationary the torque balance reduces to T = T_motor(I): the
inertial, viscous, and back-EMF terms all vanish. That condition -- not the name
of the test -- is what makes the characterization valid, so applicability is a
predicate over the data rather than a match on the behavior ID.

The test drives a continuous torque sawtooth, optionally repeated at several
park angles held by the other motor. Everything lands in one logged span
(a trace CSV carries no log-flag column; test_manager.py:188 stamps one constant
flag on every row), so this processor owns all the segmentation:

    1. split the span into park-angle levels
    2. drop the slews between levels, where the rotor is genuinely moving
    3. within each level, split by ramp direction
    4. fit per (level, direction), then aggregate

Direction splitting is mandatory, not cosmetic. Up and down ramps trace
different curves, and their offsets are near-symmetric -- so pooling them
cancels the hysteresis to a reassuring near-zero that is an averaging artifact.
See dyno/docs/post_processing_design.md section 7.2b.
"""

import numpy as np
from scipy.optimize import curve_fit

from ..processor import Applicability, Processor, Result
from ..registry import register


def tanh_model(current, A, B):
    """De-biased torque model: odd, through the origin."""
    return A * np.tanh(B * current)


def tanh_model_offset(current, A, B, C):
    """Measurement model including the zero-current torque bias C."""
    return A * np.tanh(B * current) + C


def _r2(y, fit):
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')


def _fit_branch(current, torque):
    """Linear and tanh fits over one (level, direction) branch.

    Keeps the covariance from curve_fit. On a sweep that never approaches
    saturation, A and B are near-degenerate -- only their product is constrained
    -- and the standard errors are the only thing that reveals it.
    """
    out = {'n': int(current.size)}

    slope, offset = np.polyfit(current, torque, 1)
    out['linear'] = {
        'kt_Nm_per_A': float(slope),
        'offset_Nm': float(offset),
        'r_squared': _r2(torque, np.polyval([slope, offset], current)),
    }

    try:
        A0 = float(np.max(np.abs(torque)))
        B0 = 1.0 / max(float(np.max(np.abs(current))), 1e-6)
        popt, pcov = curve_fit(tanh_model_offset, current, torque,
                               p0=[A0, B0, 0.0], maxfev=20000)
        A, B, C = (float(v) for v in popt)
        err = np.sqrt(np.diag(pcov))
        resid = torque - tanh_model_offset(current, A, B, C)
        # A-B correlation exposes the degeneracy directly.
        ab_corr = (float(pcov[0, 1] / (err[0] * err[1]))
                   if err[0] > 0 and err[1] > 0 else float('nan'))
        out['tanh'] = {
            'A_Nm': A, 'A_stderr_Nm': float(err[0]),
            'B_per_A': B, 'B_stderr_per_A': float(err[1]),
            'C_Nm': C,
            'small_signal_kt_Nm_per_A': A * B,
            'AB_correlation': ab_corr,
            'r_squared': _r2(torque, tanh_model_offset(current, A, B, C)),
            'rmse_Nm': float(np.sqrt(np.mean(resid ** 2))),
        }
    except (RuntimeError, ValueError) as exc:
        out['tanh'] = {'error': f'{type(exc).__name__}: {exc}'}

    return out


@register
class CurrentTorque(Processor):
    name = 'current_torque'
    title = 'Current -> Torque Characterization'
    description = (
        'Fits torque against phase current under stationary-rotor conditions. '
        'Splits the sweep by park angle and by ramp direction, reports kt from '
        'linear and tanh models with their uncertainties, measures the '
        'hysteresis loop width, and cross-checks the measured kt and its sign '
        'against the configured motor parameters.'
    )
    granularity = 'run'

    params = {
        'motor':              (None,  str,   'Device to characterize (auto-detected)'),
        'current_channel':    (None,  str,   'Current channel (auto-detected)'),
        'torque_channel':     (None,  str,   'Torque channel (auto-detected)'),
        'max_velocity':       (0.05,  float, 'Stationarity gate [rad/s]'),
        'min_stationary_frac':(0.5,   float, 'Fraction of samples that must be stationary'),
        'min_current_span':   (0.5,   float, 'Minimum current range to characterize [A]'),
        'ramp_smooth_s':      (0.5,   float, 'Smoothing window for ramp-direction detection [s]'),
        'min_branch_samples': (500,   int,   'Minimum samples for a branch to be fitted'),
        'level_tol_rad':      (0.02,  float, 'Position tolerance for grouping park angles [rad]'),
        'saturation_util':    (0.7,   float, 'Fraction of i_cont above which saturation counts as exercised'),
        'kt_tolerance_pct':   (5.0,   float, 'Measured-vs-configured kt delta that raises a warning [%]'),
        'csv_step_a':         (0.25,  float, 'Sampling density of the exported fit curve [A]'),
    }

    # -- applicability -----------------------------------------------------

    def _torque_channels(self, seg):
        # Sensors only: '*_torque' but not '*_torque_command'.
        return [c for c in seg.channels()
                if c.endswith('_torque') and seg.is_active(c)]

    def _stationary_mask(self, segs, max_v):
        """Per-sample 'every motor is at rest'.

        Deliberately a mask rather than a peak-over-the-span test: a multi-park-
        angle sweep *must* slew between levels, so judging the span by its peak
        velocity would reject the very experiment this processor exists for.
        The moving samples are level transitions, which level detection drops
        anyway -- what matters is that the bulk of the span is usable.
        """
        seg = segs[0]
        mask = None
        for dev in seg.devices():
            vch = f'{seg.prefix(dev)}_velocity'
            if not seg.has(vch):
                continue
            v = np.concatenate([s[vch] for s in segs])
            still = np.abs(np.nan_to_num(v, nan=np.inf)) <= max_v
            mask = still if mask is None else (mask & still)
        n = sum(len(s) for s in segs)
        return np.ones(n, dtype=bool) if mask is None else mask

    def applies_to(self, segs):
        seg = segs[0]
        d = self.defaults()
        cat = lambda ch: np.concatenate([s[ch] for s in segs])

        # 1. Stationarity gate -- the precondition that makes the fit valid.
        still = self._stationary_mask(segs, d['max_velocity'])
        frac = float(still.mean())
        if frac < d['min_stationary_frac']:
            peaks = '; '.join(
                f'{dev} peak |w| = {float(np.nanmax(np.abs(cat(f"{seg.prefix(dev)}_velocity")))):.3f} rad/s'
                for dev in seg.devices() if seg.has(f'{seg.prefix(dev)}_velocity'))
            return Applicability(
                'no', f'only {100 * frac:.1f}% of samples are stationary '
                      f'(need {100 * d["min_stationary_frac"]:.0f}%): {peaks}')

        # 2. Which motor is sweeping current? The rig runs characterizations in
        #    either direction (LOAD swept against a blocked DUT, or DUT swept
        #    against a position-holding LOAD), so this is detected, not assumed.
        #    Judged over the stationary samples only.
        candidates = []
        for dev in seg.devices():
            cch = f'{seg.prefix(dev)}_current'
            if seg.has(cch):
                cur = cat(cch)[still]
                if np.all(np.isnan(cur)):
                    continue
                candidates.append((float(np.nanmax(cur) - np.nanmin(cur)), dev, cch))
        if not candidates:
            return Applicability('no', 'no current channel found for any device')

        span, motor, cch = max(candidates)
        if span < d['min_current_span']:
            return Applicability(
                'no', f'largest current span is {span:.3f} A on {motor}; '
                      'nothing is being swept')

        # 3. Which torque cell carries the reaction? Correlation separates them
        #    unambiguously -- an idle cell reads pure noise.
        tchs = self._torque_channels(seg)
        if not tchs:
            return Applicability('no', 'no torque sensor channel found')
        cur = cat(cch)[still]
        scored = []
        for t in tchs:
            tq = cat(t)[still]
            ok = ~(np.isnan(cur) | np.isnan(tq))
            if ok.sum() < 2 or np.std(tq[ok]) == 0:
                continue
            scored.append((abs(float(np.corrcoef(cur[ok], tq[ok])[0, 1])), t))
        if not scored:
            return Applicability('no', 'no torque channel correlates with current')
        corr, tch = max(scored)
        if corr < 0.5:
            return Applicability(
                'maybe', f'best torque channel {tch} correlates only {corr:.3f} '
                         f'with {cch}; pairing is doubtful')

        moving_note = ('' if frac > 0.999 else
                       f', {100 * (1 - frac):.1f}% moving samples excluded')
        return Applicability(
            'yes',
            f'{100 * frac:.1f}% of samples stationary{moving_note}; {cch} spans '
            f'{span:.2f} A, {tch} correlates {corr:.4f}',
            {'motor': motor, 'current_channel': cch, 'torque_channel': tch})

    # -- segmentation ------------------------------------------------------

    def _find_levels(self, seg, params, still=None):
        """Split a span into park-angle levels.

        Command channels are NaN whenever that mode is inactive, so the holding
        motor is simply the one with a live position command. When no motor
        holds position -- the rotor is blocked by a mechanical fixture -- the
        whole span is a single level.
        """
        holder = None
        for dev in seg.devices():
            pch = f'{seg.prefix(dev)}_position_command'
            if seg.is_active(pch):
                holder = (dev, pch)
                break

        if holder is None:
            return [{'lo': 0, 'hi': len(seg), 'position_rad': None, 'holder': None}]

        dev, pch = holder
        cmd = seg[pch]
        tol = params['level_tol_rad']
        min_n = params['min_branch_samples']

        # Plateaus in a piecewise-constant command. Samples inside a slew are
        # dropped outright rather than left to fail the stationarity gate.
        stationary = np.zeros(len(cmd), dtype=bool)
        stationary[1:] = np.abs(np.diff(cmd)) < tol * 1e-2
        stationary[0] = stationary[1] if len(cmd) > 1 else True
        stationary &= ~np.isnan(cmd)

        levels, lo = [], None
        meas_ch = f'{seg.prefix(dev)}_position'
        for i in range(len(stationary)):
            if stationary[i] and lo is None:
                lo = i
            elif not stationary[i] and lo is not None:
                if i - lo >= min_n:
                    levels.append((lo, i))
                lo = None
        if lo is not None and len(stationary) - lo >= min_n:
            levels.append((lo, len(stationary)))

        out = []
        for lo, hi in levels:
            # Report the measured mean position, not the commanded value: the
            # holding servo sags under load, and the measurement is free.
            # Averaged over stationary samples only -- a command that steps to
            # its new value before the axis has slewed there would otherwise
            # drag the mean back toward the previous level.
            src = seg[meas_ch][lo:hi] if seg.has(meas_ch) else cmd[lo:hi]
            if still is not None:
                settled = src[still[lo:hi]]
                if settled.size:
                    src = settled
            out.append({'lo': lo, 'hi': hi, 'position_rad': float(np.nanmean(src)),
                        'holder': dev})
        return out or [{'lo': 0, 'hi': len(seg), 'position_rad': None, 'holder': dev}]

    def _split_ramps(self, seg, lo, hi, motor, current, params):
        """Classify samples inside one level as up-ramp or down-ramp.

        Uses the torque command when it is live (clean, monotonic) and falls
        back to the measured current otherwise. Samples near a reversal have
        near-zero slope and fall out of both masks, which also disposes of the
        dwell the trace holds at each peak.
        """
        drive = None
        tcmd = f'{seg.prefix(motor)}_torque_command'
        if seg.is_active(tcmd):
            drive = seg[tcmd][lo:hi]
        if drive is None or np.all(np.isnan(drive)):
            drive = current

        win = max(3, int(round(params['ramp_smooth_s'] / seg.dt)) | 1)
        kernel = np.ones(win) / win
        slope = np.convolve(np.gradient(np.nan_to_num(drive)), kernel, mode='same')

        # Threshold relative to the sweep's own typical rate, so this adapts to
        # the ramp rate instead of hard-coding one.
        scale = float(np.percentile(np.abs(slope), 75))
        thresh = max(scale * 0.25, 1e-12)
        return slope > thresh, slope < -thresh

    # -- run ---------------------------------------------------------------

    def run(self, segs, params):
        res = Result()
        seg = segs[0]
        motor = params['motor']
        cch, tch = params['current_channel'], params['torque_channel']

        mp = seg.device_params(motor)
        cfg_kt = mp.get('motor_params', {}).get('kt')
        cfg_flip = mp.get('flip_torque_sign')
        i_cont = mp.get('drive_params', {}).get('i_cont')
        t_cont = mp.get('motor_limits', {}).get('continuous_torque')

        # -- collect branches -------------------------------------------
        blocks = []
        n_moving = 0
        for s in segs:
            # Drop moving samples outright rather than trusting level detection
            # to have caught every slew: the fit is only valid where the rotor
            # is at rest.
            still = self._stationary_mask([s], params['max_velocity'])
            n_moving += int((~still).sum())
            for lvl in self._find_levels(s, params, still):
                lo, hi = lvl['lo'], lvl['hi']
                cur, tq = s[cch][lo:hi], s[tch][lo:hi]
                up, dn = self._split_ramps(s, lo, hi, motor, cur, params)
                valid = ~(np.isnan(cur) | np.isnan(tq)) & still[lo:hi]
                for name, mask in (('up', up & valid), ('down', dn & valid)):
                    if mask.sum() >= params['min_branch_samples']:
                        blocks.append({
                            'level_position_rad': lvl['position_rad'],
                            'direction': name,
                            'current': cur[mask], 'torque': tq[mask],
                        })

        if not blocks:
            res.add('error', 'no_branches',
                    'No ramp branch had enough samples to fit. Check '
                    'min_branch_samples and that the sweep actually ramps.')
            return res

        # -- sign normalisation, cross-checked against config -------------
        all_i = np.concatenate([b['current'] for b in blocks])
        all_t = np.concatenate([b['torque'] for b in blocks])
        raw_kt = float(np.polyfit(all_i, all_t, 1)[0])
        flip = -1.0 if raw_kt < 0 else 1.0

        if flip < 0:
            expected_negative = bool(cfg_flip)
            if expected_negative:
                res.add('info', 'sign_flipped',
                        f'kt is negative in the raw drive frame, which is what '
                        f'flip_torque_sign={cfg_flip} predicts for {motor} '
                        f'(mounted counter to the rig frame). Torque negated for '
                        f'reporting; all kt values below are positive.')
            else:
                res.add('warn', 'sign_unexpected',
                        f'Measured kt is negative but flip_torque_sign={cfg_flip} '
                        f'for {motor} predicts positive. Torque negated so the '
                        f'numbers below are usable, but check the torque-cell '
                        f'wiring and the absorbers.yaml entry -- a mounting '
                        f'convention and a wiring fault look identical here.')
            for b in blocks:
                b['torque'] = -b['torque']
            all_t = -all_t

        # -- fits ---------------------------------------------------------
        branches = []
        for b in blocks:
            fit = _fit_branch(b['current'], b['torque'])
            fit['direction'] = b['direction']
            fit['level_position_rad'] = b['level_position_rad']
            branches.append(fit)

        pooled = _fit_branch(all_i, all_t)
        kt = pooled['linear']['kt_Nm_per_A']

        # -- hysteresis: the reason branches are never pooled for reporting --
        ups = [b for b in branches if b['direction'] == 'up']
        dns = [b for b in branches if b['direction'] == 'down']
        hysteresis = None
        if ups and dns:
            up_off = float(np.mean([b['linear']['offset_Nm'] for b in ups]))
            dn_off = float(np.mean([b['linear']['offset_Nm'] for b in dns]))
            hysteresis = abs(up_off - dn_off)
            kt_up = float(np.mean([b['linear']['kt_Nm_per_A'] for b in ups]))
            kt_dn = float(np.mean([b['linear']['kt_Nm_per_A'] for b in dns]))
            kt_dir_pct = 100 * abs(kt_up - kt_dn) / abs(kt_up) if kt_up else float('nan')

            noise = seg.tare.get(tch, {}).get('stddev')
            noise_txt = (f' ({hysteresis / noise:.1f}x the {tch} tare noise of '
                         f'{noise:.3f} Nm)') if noise else ''
            res.add('info', 'hysteresis',
                    f'Up/down ramp offsets differ by {hysteresis:.3f} Nm'
                    f'{noise_txt}. kt differs by only {kt_dir_pct:.3f}% between '
                    f'directions, so the slope is unaffected. Candidate sources: '
                    f'magnetic hysteresis, fixture friction as windup reverses, '
                    f'torque-cell hysteresis. Not attributed here.')
            if abs(pooled['linear']['offset_Nm']) < hysteresis / 4:
                res.add('info', 'pooled_offset_cancels',
                        f'The pooled offset ({pooled["linear"]["offset_Nm"]:+.3f} Nm) '
                        f'is much smaller than the branch gap because the two '
                        f'branches cancel. Do not read it as an absence of bias.')
        else:
            res.add('warn', 'single_direction',
                    'Only one ramp direction was fitted, so hysteresis could not '
                    'be measured.')

        # -- park-angle coverage -----------------------------------------
        positions = sorted({b['level_position_rad'] for b in branches
                            if b['level_position_rad'] is not None})
        kt_spread = None
        if len(positions) > 1:
            per_level = []
            for p in positions:
                vals = [b['linear']['kt_Nm_per_A'] for b in branches
                        if b['level_position_rad'] == p]
                per_level.append(float(np.mean(vals)))
            kt_spread = float(np.max(per_level) - np.min(per_level))
            res.add('info', 'park_angle_spread',
                    f'kt across {len(positions)} park angles spans '
                    f'{kt_spread:.4f} Nm/A ({100 * kt_spread / kt:.2f}%). This '
                    f'spread is the cogging contribution to kt and is a more '
                    f'honest error bar than the fit standard error.')
        else:
            res.add('warn', 'single_park_angle',
                    'Only one rotor position was measured, so the cogging '
                    'contribution to kt is unquantified. kt here is specific to '
                    'this electrical angle.')

        # -- saturation ---------------------------------------------------
        max_i = float(np.max(np.abs(all_i)))
        util = max_i / i_cont if i_cont else None
        sat_ok = bool(util is not None and util >= params['saturation_util'])
        tanh_A = pooled['tanh'].get('A_Nm')
        if not sat_ok and tanh_A:
            reach = abs(tanh_A) / max(float(np.max(np.abs(all_t))), 1e-9)
            res.add('warn', 'saturation_not_exercised',
                    f'Peak current is {max_i:.2f} A, '
                    f'{100 * util:.0f}% of i_cont ({i_cont} A). '
                    f'The tanh saturation torque A = {tanh_A:.1f} Nm is '
                    f'extrapolated {reach:.1f}x beyond the measured range and '
                    f'must not be quoted. A and B are near-degenerate here '
                    f'(correlation {pooled["tanh"].get("AB_correlation", float("nan")):.4f}); '
                    f'only their product, the small-signal kt, is constrained.')

        lin_r2 = pooled['linear']['r_squared']
        tanh_r2 = pooled['tanh'].get('r_squared')
        if tanh_r2 is not None and not sat_ok:
            res.add('info', 'model_choice',
                    f'Linear R2 = {lin_r2:.5f} vs tanh R2 = {tanh_r2:.5f} '
                    f'(delta {tanh_r2 - lin_r2:+.2e}). The tanh does not earn its '
                    f'extra parameter on this data; use the linear kt.')

        # -- kt vs config: the headline ----------------------------------
        kt_delta_pct = None
        if cfg_kt:
            kt_delta_pct = 100 * (abs(kt) - cfg_kt) / cfg_kt
            level = ('warn' if abs(kt_delta_pct) > params['kt_tolerance_pct']
                     else 'info')
            res.add(level, 'kt_vs_config',
                    f'Measured kt = {abs(kt):.4f} Nm/A vs configured '
                    f'{cfg_kt} Nm/A for {motor}: {kt_delta_pct:+.2f}%.')

        # -- thermal ------------------------------------------------------
        temp_ch = next((c for c in seg.channels() if c.endswith('_stator_temp')), None)
        peak_t = float(np.max(np.abs(all_t)))
        over_cont = t_cont and peak_t > t_cont
        if temp_ch and seg.is_active(temp_ch):
            temps = np.concatenate([s[temp_ch] for s in segs])
            res.metrics['stator_temp_start_C'] = float(np.nanmin(temps))
            res.metrics['stator_temp_rise_C'] = float(np.nanmax(temps) - np.nanmin(temps))
        elif over_cont:
            res.add('warn', 'thermal_unverified',
                    f'Peak torque {peak_t:.1f} Nm is {peak_t / t_cont:.1f}x the '
                    f'{t_cont} Nm continuous rating for {motor}, and '
                    f'{temp_ch or "no stator temperature channel"} is '
                    f'{"all NaN" if temp_ch else "absent"}. Magnet remanence falls '
                    f'with temperature, so kt drifts downward as the motor heats. '
                    f'The trace path has no cooldown logic. Thermal drift cannot '
                    f'be separated from cogging spread until this channel works.')

        # -- torque cell headroom ----------------------------------------
        cell = seg.tare.get(tch, {})
        if cell.get('full_scale'):
            frac = 100 * peak_t / cell['full_scale']
            res.metrics['torque_cell_fs_used_pct'] = frac
            if frac < 20:
                res.add('info', 'low_cell_utilisation',
                        f'Peak torque uses {frac:.1f}% of {tch} full scale '
                        f'({cell["full_scale"]} Nm); tare noise was '
                        f'{cell.get("stddev", float("nan")):.3f} Nm.')

        # -- metrics ------------------------------------------------------
        res.metrics.update({
            'motor': motor,
            'current_channel': cch,
            'torque_channel': tch,
            'n_samples': int(all_i.size),
            'n_park_angles': len(positions) if positions else 1,
            'park_angles_rad': positions,
            'n_branches': len(branches),
            'sign_flipped': flip < 0,
            'configured_flip_torque_sign': cfg_flip,
            'kt_Nm_per_A': abs(kt),
            'configured_kt_Nm_per_A': cfg_kt,
            'kt_delta_pct': kt_delta_pct,
            'kt_park_angle_spread_Nm_per_A': kt_spread,
            'hysteresis_Nm': hysteresis,
            'peak_current_A': max_i,
            'i_cont_A': i_cont,
            'current_utilisation_pct': 100 * util if util else None,
            'saturation_exercised': sat_ok,
            'peak_torque_Nm': peak_t,
            'pooled': pooled,
            'branches': branches,
        })

        res.summary = (
            f'{motor}: kt = {abs(kt):.4f} Nm/A'
            + (f' ({kt_delta_pct:+.2f}% vs configured {cfg_kt})' if cfg_kt else '')
            + f', from {len(branches)} branches across '
              f'{len(positions) if positions else 1} park angle(s). '
            + (f'Hysteresis {hysteresis:.3f} Nm. ' if hysteresis else '')
            + (f'Saturation not exercised ({100 * util:.0f}% of i_cont).'
               if util and not sat_ok else '')
        )

        res.figures.extend(self._figures(branches, blocks, pooled, all_i, all_t,
                                         motor, cch, tch, flip, cfg_kt))
        res.tables.append(('fit_curve', self._csv(pooled, all_i, params)))
        return res

    # -- outputs -----------------------------------------------------------

    def _figures(self, branches, blocks, pooled, all_i, all_t, motor, cch, tch,
                 flip, cfg_kt):
        import matplotlib.pyplot as plt

        figs = []
        colors = {'up': 'tab:blue', 'down': 'tab:red'}
        note = ' (torque sign normalised)' if flip < 0 else ''

        # 1. The hysteresis loop. Plotted against the pooled trend removed:
        #    a sub-Nm loop is invisible on a full-scale torque axis, and this
        #    figure exists precisely to show it.
        kt_p = pooled['linear']['kt_Nm_per_A']
        fig, ax = plt.subplots(figsize=(11, 7))
        seen = set()
        for b in branches:
            d = b['direction']
            lbl = d if d not in seen else None
            seen.add(d)
            ax.axhline(b['linear']['offset_Nm'], color=colors.get(d, 'gray'),
                       lw=1.2, alpha=0.9, label=lbl)
        x = np.linspace(float(np.min(all_i)), float(np.max(all_i)), 300)
        for b in blocks:
            ax.scatter(b['current'], b['torque'] - kt_p * b['current'], s=1,
                       alpha=0.06, color=colors.get(b['direction'], 'k'), zorder=0)
        gap = None
        ups = [b['linear']['offset_Nm'] for b in branches if b['direction'] == 'up']
        dns = [b['linear']['offset_Nm'] for b in branches if b['direction'] == 'down']
        if ups and dns:
            gap = abs(float(np.mean(ups)) - float(np.mean(dns)))
            ax.annotate('', xy=(x[10], float(np.mean(ups))),
                        xytext=(x[10], float(np.mean(dns))),
                        arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
            ax.text(x[12], 0, f'  {gap:.3f} Nm', color='green', va='center')
        ax.set_xlabel(f'{cch} (A)')
        ax.set_ylabel(f'{tch} minus pooled trend (Nm){note}')
        ax.set_title(f'{motor}: hysteresis loop '
                     f'(torque with the {abs(kt_p):.4f} Nm/A trend removed)')
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        figs.append(('hysteresis', fig))

        # 2. Pooled fit with the configured kt overlaid for comparison.
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.scatter(all_i, all_t - pooled['linear']['offset_Nm'], s=2, alpha=0.15,
                   color='steelblue', label='measured (de-biased)')
        x = np.linspace(float(np.min(all_i)), float(np.max(all_i)), 400)
        kt = pooled['linear']['kt_Nm_per_A']
        ax.plot(x, kt * x, 'r-', lw=2,
                label=f'linear: kt = {abs(kt):.4f} Nm/A, '
                      f"R2 = {pooled['linear']['r_squared']:.5f}")
        if 'A_Nm' in pooled['tanh']:
            ax.plot(x, tanh_model(x, pooled['tanh']['A_Nm'], pooled['tanh']['B_per_A']),
                    'g--', lw=1.5,
                    label=f"tanh: kt0 = {abs(pooled['tanh']['small_signal_kt_Nm_per_A']):.4f}, "
                          f"R2 = {pooled['tanh']['r_squared']:.5f}")
        if cfg_kt:
            ax.plot(x, np.sign(kt) * cfg_kt * x, 'k:', lw=1.5,
                    label=f'configured kt = {cfg_kt} Nm/A')
        ax.set_xlabel(f'{cch} (A)')
        ax.set_ylabel(f'{tch} (Nm){note}')
        ax.set_title(f'{motor}: pooled fit vs configured')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        figs.append(('pooled_fit', fig))

        # 3. Residuals -- where a linear model actually fails, if it does.
        fig, ax = plt.subplots(figsize=(11, 5))
        resid = all_t - np.polyval([kt, pooled['linear']['offset_Nm']], all_i)
        ax.scatter(all_i, resid, s=1, alpha=0.1, color='k')
        ax.axhline(0, color='r', lw=1)
        ax.set_xlabel(f'{cch} (A)')
        ax.set_ylabel('residual (Nm)')
        ax.set_title('Linear fit residuals (curvature here = real saturation)')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        figs.append(('residuals', fig))

        # 4. kt per park angle, only meaningful with more than one.
        positions = sorted({b['level_position_rad'] for b in branches
                            if b['level_position_rad'] is not None})
        if len(positions) > 1:
            fig, ax = plt.subplots(figsize=(11, 5))
            for d in ('up', 'down'):
                xs = [b['level_position_rad'] for b in branches if b['direction'] == d]
                ys = [abs(b['linear']['kt_Nm_per_A']) for b in branches
                      if b['direction'] == d]
                if xs:
                    ax.plot(np.degrees(xs), ys, 'o-', color=colors[d], label=d)
            if cfg_kt:
                ax.axhline(cfg_kt, color='k', ls=':', label=f'configured {cfg_kt}')
            ax.set_xlabel('park angle (deg)')
            ax.set_ylabel('kt (Nm/A)')
            ax.set_title('kt vs park angle (spread = cogging contribution)')
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            figs.append(('kt_vs_park_angle', fig))

        return figs

    def _csv(self, pooled, all_i, params):
        """Fit-derived curve at fixed current density, de-biased."""
        step = params['csv_step_a']
        lo = np.floor(float(np.min(all_i)) / step) * step
        hi = np.ceil(float(np.max(all_i)) / step) * step
        xs = np.arange(lo, hi + step / 2, step)
        kt = pooled['linear']['kt_Nm_per_A']
        rows = ['Current (A),Torque linear (Nm),Torque tanh (Nm)']
        has_tanh = 'A_Nm' in pooled['tanh']
        for x in xs:
            t_lin = kt * x
            t_tanh = (tanh_model(x, pooled['tanh']['A_Nm'], pooled['tanh']['B_per_A'])
                      if has_tanh else float('nan'))
            rows.append(f'{x:.4f},{t_lin:.4f},{t_tanh:.4f}')
        return '\n'.join(rows) + '\n'
