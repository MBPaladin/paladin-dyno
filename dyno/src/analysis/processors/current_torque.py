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


def drive_params_from_tanh(tanh_fit):
    """Recast a fitted A*tanh(B*I) into the (kt, k_tanh) a drive configures.

    AKD and ELMO both invert torque to current with the same expression
    (devices.py:1236 and devices.py:1726):

        I = arctanh(T * k_tanh / kt) / k_tanh

    whose forward form is T = (kt / k_tanh) * tanh(k_tanh * I). Matching that
    term for term against A*tanh(B*I) gives

        k_tanh = B          kt = A * B

    so the drive's kt is exactly the small-signal slope and its k_tanh is
    exactly the tanh's curvature -- the config parameterization and the fit are
    the same two-parameter model in different coordinates, not an
    approximation. kt/k_tanh recovers A, the saturation asymptote, and the
    arctanh above diverges as |T| approaches it: a torque command at or beyond
    the asymptote yields inf or nan amps, so motor_limits.torque has to stay
    underneath. Returns None if the tanh fit failed or is unusable.
    """
    A, B = tanh_fit.get('A_Nm'), tanh_fit.get('B_per_A')
    if A is None or B is None or not np.isfinite(A) or not np.isfinite(B):
        return None
    # Sign lives in flip_torque_sign, which the drive applies itself, so the
    # config always carries positive constants.
    A, B = abs(float(A)), abs(float(B))
    if B <= 0 or A <= 0:
        return None
    return {'kt': A * B, 'k_tanh': B, 'asymptote_Nm': A}


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
        'vel_smooth_s':       (0.1,   float, 'Low-pass window applied before the stationarity gate [s]'),
        'min_stationary_frac':(0.5,   float, 'Fraction of samples that must be stationary'),
        'min_current_span':   (0.5,   float, 'Minimum current range to characterize [A]'),
        'ramp_smooth_s':      (0.5,   float, 'Smoothing window for ramp-direction detection [s]'),
        'min_branch_samples': (500,   int,   'Minimum samples for a branch to be fitted'),
        'level_tol_rad':      (0.02,  float, 'Position tolerance for grouping park angles [rad]'),
        'saturation_util':    (0.7,   float, 'Fraction of i_cont above which saturation counts as exercised'),
        'kt_tolerance_pct':   (5.0,   float, 'Measured-vs-configured kt delta that raises a warning [%]'),
        'csv_step_a':         (0.25,  float, 'Sampling density of the exported fit curve [A]'),
        'overlay_configured_kt': (True, bool, 'Overlay the configured kt on the fit '
                                              'and park-angle figures'),
        'overlay_linear_fit': (False,  bool,  'Overlay the pooled linear fit on the '
                                             'pooled-fit figure'),
        'also_characterize_absorber': (True, bool,
                                       'Additionally fit the same characterization '
                                       'for the motor holding the rotor, whose '
                                       'current is the reaction to the sweep'),
    }

    # -- applicability -----------------------------------------------------

    def _torque_channels(self, seg):
        # Sensors only: '*_torque' but not '*_torque_command'.
        return [c for c in seg.channels()
                if c.endswith('_torque') and seg.is_active(c)]

    def _score_torque_channels(self, seg, cat, cch, still):
        """[(|corr|, channel)] against `cch`, best first, over stationary samples.

        Correlation separates a loaded cell from an idle one unambiguously --
        an idle cell reads pure noise. Shared by the applicability predicate and
        the absorber pass so both pair a current with a cell the same way.
        """
        cur = cat(cch)[still]
        scored = []
        for t in self._torque_channels(seg):
            tq = cat(t)[still]
            ok = ~(np.isnan(cur) | np.isnan(tq))
            if ok.sum() < 2 or np.std(tq[ok]) == 0:
                continue
            scored.append((abs(float(np.corrcoef(cur[ok], tq[ok])[0, 1])), t))
        return sorted(scored, reverse=True)

    def _declared_cell(self, seg, dev):
        """The cell the bench's `ports:` block bolts to this device's shaft."""
        for port in seg.ports():
            if port.get('device') == dev and port.get('cell'):
                return str(port['cell'])
        return None

    def _stationary_mask(self, segs, max_v, smooth_s=0.1):
        """Per-sample 'every motor is at rest'.

        Deliberately a mask rather than a peak-over-the-span test: a multi-park-
        angle sweep *must* slew between levels, so judging the span by its peak
        velocity would reject the very experiment this processor exists for.
        The moving samples are level transitions, which level detection drops
        anyway -- what matters is that the bulk of the span is usable.

        The velocity is low-passed first, and that is not cosmetic. A drive
        reports velocity by differentiating the encoder at the bus rate, so at
        1 kHz the signal on a blocked rotor is dominated by quantization dither,
        not motion: a genuine blocked-rotor sweep logged 0.88 rad/s peak and
        1.6 rad of integral(|w|dt) while the shaft physically moved 0.016 rad.
        Thresholding the raw samples measures the encoder's LSB and rejects the
        very condition it is meant to confirm. `max_velocity` is a statement
        about the shaft, so it is applied to a shaft-scale estimate of speed.
        """
        seg = segs[0]
        win_req = max(1, int(round(smooth_s / seg.dt)) | 1) if smooth_s > 0 else 1
        mask = None
        for dev in seg.devices():
            vch = f'{seg.prefix(dev)}_velocity'
            if not seg.has(vch):
                continue
            raw = np.concatenate([s[vch] for s in segs])
            # NaN marks a dead channel sample, which can never count as at
            # rest -- but it is masked out afterwards rather than smuggled in
            # as inf, which the convolution would smear over a whole window.
            bad = np.isnan(raw)
            v = np.nan_to_num(raw, nan=0.0)
            # Clamp to the data: np.convolve(mode='same') returns
            # max(len(v), len(kernel)), so a window wider than a short span
            # would silently lengthen the mask and break the & below.
            win = min(win_req, len(v))
            win -= 1 - win % 2                      # 'same' needs an odd window
            if win >= 3:
                # Smooth the signed velocity: the mean of w over a window is
                # that window's net displacement divided by its length, which
                # is the shaft speed being asked about. Smoothing |w| instead
                # would preserve the dither's magnitude and change nothing.
                v = np.convolve(v, np.ones(win) / win, mode='same')
            still = (np.abs(v) <= max_v) & ~bad
            mask = still if mask is None else (mask & still)
        n = sum(len(s) for s in segs)
        return np.ones(n, dtype=bool) if mask is None else mask

    def _travel(self, seg, dev, cat):
        """Peak-to-peak shaft travel for a device, or None if unmeasurable.

        Devices name their position channel inconsistently -- a geared DUT logs
        the output side -- so both spellings are tried.
        """
        p = seg.prefix(dev)
        for ch in (f'{p}_output_position', f'{p}_position'):
            if seg.has(ch):
                v = cat(ch)
                if not np.all(np.isnan(v)):
                    return float(np.nanmax(v) - np.nanmin(v))
        return None

    def applies_to(self, segs):
        seg = segs[0]
        d = self.defaults()
        cat = lambda ch: np.concatenate([s[ch] for s in segs])

        # 1. Stationarity gate -- the precondition that makes the fit valid.
        #
        # A low fraction is not automatically disqualifying. The mask is applied
        # per sample, so moving samples are dropped from the fit either way; the
        # fit's real requirements are that enough stationary samples survive and
        # that they still span a usable current range, which steps 2-3 and
        # min_branch_samples check directly. A sweep that stick-slips in bursts
        # scattered across its range keeps full current coverage in the samples
        # that remain, and rejecting it on the fraction alone throws away a
        # perfectly fittable test. So a shortfall here degrades the verdict to
        # 'maybe' and is explained, rather than ending the run -- unless too few
        # stationary samples survive to populate the up and down branches at
        # all, which is a real dead end.
        still = self._stationary_mask(segs, d['max_velocity'], d['vel_smooth_s'])
        frac = float(still.mean())
        n_still = int(still.sum())
        # Both directions, at more than the bare per-branch minimum, or the
        # branch fits downstream will be starved.
        n_needed = 4 * d['min_branch_samples']
        low_frac = frac < d['min_stationary_frac']

        # Report travel alongside speed. A rotor that never went anywhere but
        # reads a high peak |w| is a noisy velocity channel, not a moving
        # shaft, and the reason line should say so plainly.
        peaks = '; '.join(
            f'{dev} peak |w| = '
            f'{float(np.nanmax(np.abs(cat(f"{seg.prefix(dev)}_velocity")))):.3f} rad/s'
            + (f', travel {travel:.3f} rad'
               if (travel := self._travel(seg, dev, cat)) is not None else '')
            for dev in seg.devices() if seg.has(f'{seg.prefix(dev)}_velocity'))

        if low_frac and n_still < n_needed:
            return Applicability(
                'no', f'only {100 * frac:.1f}% of samples are stationary '
                      f'(need {100 * d["min_stationary_frac"]:.0f}%), and the '
                      f'{n_still} that are will not fill the up and down '
                      f'branches ({n_needed} needed): {peaks}')

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

        # 3. Which torque cell carries the reaction?
        if not self._torque_channels(seg):
            return Applicability('no', 'no torque sensor channel found')
        scored = self._score_torque_channels(seg, cat, cch, still)
        if not scored:
            return Applicability('no', 'no torque channel correlates with current')
        corr, tch = scored[0]
        # Built before the first non-'no' return, not after it: a verdict that
        # lets run() proceed must always carry the detection forward, or run()
        # falls back on the None defaults and indexes the log with them.
        params = {'motor': motor, 'current_channel': cch, 'torque_channel': tch}
        if corr < 0.5:
            return Applicability(
                'maybe', f'best torque channel {tch} correlates only {corr:.3f} '
                         f'with {cch}; pairing is doubtful', params)

        moving_note = ('' if frac > 0.999 else
                       f', {100 * (1 - frac):.1f}% moving samples excluded')
        if low_frac:
            # Fittable, but the operator should see why it was let through: the
            # surviving samples still cover the sweep, and that -- not the
            # headcount -- is what the fit needs.
            return Applicability(
                'maybe',
                f'only {100 * frac:.1f}% of samples are stationary (below the '
                f'{100 * d["min_stationary_frac"]:.0f}% preferred), but the '
                f'{n_still} that are still span {span:.2f} A of {cch} and '
                f'correlate {corr:.4f} with {tch}, so the sweep is covered. '
                f'Motion is intermittent rather than a steady rotation: '
                f'{peaks}', params)
        return Applicability(
            'yes',
            f'{100 * frac:.1f}% of samples stationary{moving_note}; {cch} spans '
            f'{span:.2f} A, {tch} correlates {corr:.4f}', params)

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
        motor = params['motor']
        cch, tch = params['current_channel'], params['torque_channel']

        metrics, summary = self._characterize(segs, motor, cch, tch, params, res)
        if metrics is None:
            return res                      # already carries the error finding
        res.metrics.update(metrics)
        res.summary = summary

        # A rig-level check, not a per-motor one: it asks whether the fixture
        # held, so it runs once regardless of how many motors get fitted.
        self._check_holder(segs, params, motor, res)

        if params['also_characterize_absorber']:
            self._characterize_absorber(segs, params, motor, res)

        return res

    def _characterize_absorber(self, segs, params, motor, res):
        """Fit the same curve for the motor on the other end of the shaft.

        In a blocked-rotor sweep the holding motor is a servo: its current is
        whatever it takes to stand still against the swept motor. At rest the
        torque balance is T_holder = -T_swept, and the cell reads that
        transmitted torque -- so the cell plotted against the *holder's* current
        is a genuine kt characterization of the holder, free from the same data.

        Three things make it weaker than the primary fit, and each is reported
        rather than assumed away: the holder's current is slaved to the sweep
        instead of being commanded, so its range is whatever the sweep provoked;
        it is measured through torque it absorbs, which inverts the sign
        convention (see `reacting` in _characterize); and the balance holds only
        at rest, which is exactly the condition the stationarity gate enforces.
        """
        seg = segs[0]
        cat = lambda ch: np.concatenate([s[ch] for s in segs])
        still = self._stationary_mask(segs, params['max_velocity'],
                                      params['vel_smooth_s'])

        # Prefer the motor that held position -- that is the absorber in this
        # test -- and fall back to whichever other motor drew the most current.
        candidates = []
        for dev in seg.devices():
            if dev == motor:
                continue
            cch = f'{seg.prefix(dev)}_current'
            if not seg.has(cch):
                continue
            cur = cat(cch)[still]
            if np.all(np.isnan(cur)):
                continue
            holds = seg.is_active(f'{seg.prefix(dev)}_position_command')
            candidates.append((holds, float(np.nanmax(cur) - np.nanmin(cur)),
                               dev, cch))
        if not candidates:
            res.add('info', 'absorber_absent',
                    'also_characterize_absorber is set, but no second motor on '
                    'the rig has a live current channel. If the rotor was blocked '
                    'by a mechanical fixture there is no absorber to characterize.')
            return

        holds, span, dev, cch = max(candidates)
        if span < params['min_current_span']:
            res.add('info', 'absorber_not_loaded',
                    f'{dev} drew only {span:.3f} A across the sweep (below the '
                    f'{params["min_current_span"]} A minimum), so there is nothing '
                    f'to fit. It was not carrying the reaction -- check whether '
                    f'the rotor was held mechanically rather than by {dev}.')
            return

        scored = self._score_torque_channels(seg, cat, cch, still)
        if not scored:
            res.add('warn', 'absorber_no_cell',
                    f'No torque channel correlates with {cch}, so {dev} cannot be '
                    f'characterized.')
            return
        corr, tch = scored[0]

        # The bench's ports: block already states which cell is bolted to this
        # shaft. Trust it over the correlation ranking when it is usable -- on a
        # single-shaft rig every cell correlates and the ranking is a coin flip.
        by_ch = {c: v for v, c in scored}
        declared = self._declared_cell(seg, dev)
        if declared and declared != tch:
            if by_ch.get(declared, 0.0) >= 0.5:
                res.add('info', 'absorber_cell_declared',
                        f'Using {declared} for {dev} because the bench\'s ports: '
                        f'block bolts that cell to its shaft, though {tch} '
                        f'correlates slightly better ({corr:.4f} vs '
                        f'{by_ch[declared]:.4f}). On a single-shaft rig both read '
                        f'the same torque.')
                tch, corr = declared, by_ch[declared]
            else:
                # Not a silent substitution: the operator would otherwise have to
                # work out for themselves why this motor's kt came off the other
                # shaft's sensor.
                res.add('info', 'absorber_cell_substituted',
                        f'{declared} is the cell the bench\'s ports: block bolts to '
                        f'{dev}, but it correlates only '
                        f'{by_ch.get(declared, float("nan")):.4f} with {cch} -- too '
                        f'poorly to fit against. Using {tch} instead ({corr:.4f}), '
                        f'which is valid while the two motors share a shaft with no '
                        f'gearing between them and the cells. A cell whose full '
                        f'scale dwarfs this sweep reads mostly noise, which is the '
                        f'usual cause.')

        if corr < 0.5:
            res.add('warn', 'absorber_pairing_doubtful',
                    f'{tch} correlates only {corr:.3f} with {cch}; the numbers '
                    f'below are reported but the cell/current pairing for {dev} '
                    f'is doubtful.')

        # This processor's kt maps cell torque to current directly, with no gear
        # term anywhere in the fit, so a geared absorber would report its kt
        # multiplied by the reduction and the config comparison would be
        # nonsense. Direct drive is the assumption; say so when it is violated.
        ratio = seg.device_params(dev).get('gear_ratio')
        try:
            ratio = abs(float(ratio)) if ratio is not None else 1.0
        except (TypeError, ValueError):
            ratio = 1.0
        if ratio != 1.0:
            res.add('warn', 'absorber_geared',
                    f'{dev} is configured gear_ratio={ratio:g}, but this fit maps '
                    f'{tch} to {cch} with no gear term. If that reduction is real '
                    f'the kt below is the shaft-side value, larger than the '
                    f'motor-side {seg.device_params(dev).get("motor_params", {}).get("kt")} '
                    f'it is compared against by roughly that factor, and the '
                    f'kt_vs_config delta is meaningless.')

        res.add('info', 'absorber_characterized',
                f'{dev} held position while {motor} was swept, so its current is '
                f'the reaction to the sweep rather than an independent excitation. '
                f'Fitting {tch} against {cch} ({corr:.4f} correlation) gives {dev} '
                f'its own kt from the same data. Ramp direction is taken from '
                f'{motor}, so "up" and "down" mean the same physical sweep '
                f'direction in both characterizations.')

        metrics, summary = self._characterize(
            segs, dev, cch, tch, params, res,
            tag='absorber', reacting=True, ramp_motor=motor)
        if metrics is None:
            return
        res.metrics.update({f'absorber_{k}': v for k, v in metrics.items()})
        res.summary = f'{res.summary} Absorber -- {summary}'

    def _check_holder(self, segs, params, motor, res):
        """Did the fixture actually hold?

        The blocked-rotor condition is manufactured by the *other* motor holding
        position. When that motor runs out of current it stops holding and the
        shaft walks under load, which is exactly what the excluded 'moving'
        samples are. The fit never sees this -- it just quietly loses samples --
        so it has to be reported from the raw channels or it goes unnoticed
        until two runs disagree.
        """
        seg = segs[0]
        n_total = sum(len(s) for s in segs)
        n_moving = sum(int((~self._stationary_mask(
            [s], params['max_velocity'], params['vel_smooth_s'])).sum())
            for s in segs)

        for dev in seg.devices():
            if dev == motor:
                continue
            pch = f'{seg.prefix(dev)}_position_command'
            hcur_ch = f'{seg.prefix(dev)}_current'
            if not seg.is_active(pch) or not seg.has(hcur_ch):
                continue
            hp = seg.device_params(dev)
            h_icont = hp.get('drive_params', {}).get('i_cont')
            hcur = np.concatenate([s[hcur_ch] for s in segs])
            h_peak = float(np.nanmax(np.abs(hcur))) if not np.all(np.isnan(hcur)) else None
            err = self._travel(seg, dev, lambda ch: np.concatenate(
                [s[ch] for s in segs]))
            if h_peak and h_icont and h_peak > h_icont:
                res.add('warn', 'holder_saturated',
                        f'{dev} was holding position and drew {h_peak:.1f} A '
                        f'peak, {h_peak / h_icont:.1f}x its {h_icont} A '
                        f'continuous rating'
                        + (f', while its shaft still moved {err:.3f} rad'
                           if err else '')
                        + f'. It is not strong enough to block the rotor at '
                          f'this torque: {100 * n_moving / n_total:.0f}% of '
                          f'samples were dropped as moving. The kt fitted from '
                          f'what survived is usable, but reduce the sweep '
                          f'amplitude or fit a stiffer fixture before trusting '
                          f'the hysteresis and saturation numbers, which are '
                          f'the ones a slipping fixture distorts.')

    def _characterize(self, segs, motor, cch, tch, params, res,
                      tag='', reacting=False, ramp_motor=None):
        """Fit one motor's current -> torque curve and report on it.

        Returns (metrics, summary), or (None, None) when no branch was fittable.
        Findings, figures, and the CSV are appended to `res`; `tag` prefixes
        their slugs so a second motor's output cannot collide with the first's.

        `reacting` marks a motor measured through torque it absorbs rather than
        torque it drives, which inverts the expected sign. `ramp_motor` is the
        motor whose command defines up/down, which for an absorber is the motor
        being swept, not itself.
        """
        seg = segs[0]
        ramp_motor = ramp_motor or motor
        code = (lambda c: f'{tag}_{c}') if tag else (lambda c: c)
        slug = (lambda s: f'{tag}_{s}') if tag else (lambda s: s)

        mp = seg.device_params(motor)
        cfg_kt = mp.get('motor_params', {}).get('kt')
        cfg_flip = mp.get('flip_torque_sign')
        i_cont = mp.get('drive_params', {}).get('i_cont')
        t_cont = mp.get('motor_limits', {}).get('continuous_torque')

        # -- collect branches -------------------------------------------
        blocks = []
        for s in segs:
            # Drop moving samples outright rather than trusting level detection
            # to have caught every slew: the fit is only valid where the rotor
            # is at rest.
            still = self._stationary_mask([s], params['max_velocity'],
                                          params['vel_smooth_s'])
            for lvl in self._find_levels(s, params, still):
                lo, hi = lvl['lo'], lvl['hi']
                cur, tq = s[cch][lo:hi], s[tch][lo:hi]
                up, dn = self._split_ramps(s, lo, hi, ramp_motor, cur, params)
                valid = ~(np.isnan(cur) | np.isnan(tq)) & still[lo:hi]
                for name, mask in (('up', up & valid), ('down', dn & valid)):
                    if mask.sum() >= params['min_branch_samples']:
                        blocks.append({
                            'level_position_rad': lvl['position_rad'],
                            'direction': name,
                            'current': cur[mask], 'torque': tq[mask],
                        })

        if not blocks:
            res.add('error', code('no_branches'),
                    f'No ramp branch had enough samples to fit for {motor}. Check '
                    'min_branch_samples and that the sweep actually ramps.')
            return None, None

        # -- sign normalisation, cross-checked against config -------------
        all_i = np.concatenate([b['current'] for b in blocks])
        all_t = np.concatenate([b['torque'] for b in blocks])
        raw_kt = float(np.polyfit(all_i, all_t, 1)[0])
        flip = -1.0 if raw_kt < 0 else 1.0

        # A reacting motor is measured through torque it absorbs, so the cell
        # reads its output with the sign opposite to a motor driving that cell.
        # flip_torque_sign describes the driving case, so the prediction inverts.
        expected_negative = bool(cfg_flip) != bool(reacting)
        # Naming the actual cause matters here: with `reacting` set, a negative
        # kt at flip_torque_sign=False comes from the reaction, not the mounting,
        # and saying 'mounted counter to the rig frame' would send someone to
        # check bolts that are fine.
        because = (f'flip_torque_sign={cfg_flip} combined with the inversion from '
                   f'measuring a motor that absorbs this torque rather than '
                   f'driving it' if reacting else
                   f'flip_torque_sign={cfg_flip} (mounted counter to the rig frame)')

        if flip < 0:
            if expected_negative:
                res.add('info', code('sign_flipped'),
                        f'kt is negative in the raw drive frame, which is what '
                        f'{because} predicts for {motor}. Torque negated for '
                        f'reporting; all kt values below are positive.')
            else:
                res.add('warn', code('sign_unexpected'),
                        f'Measured kt is negative but {because} predicts positive '
                        f'for {motor}. Torque negated so the numbers below are '
                        f'usable, but check the torque-cell wiring and the '
                        f'absorbers.yaml entry -- a mounting convention and a '
                        f'wiring fault look identical here.')
            for b in blocks:
                b['torque'] = -b['torque']
            all_t = -all_t
        elif expected_negative:
            # The mirror case. Nothing needs negating, but the disagreement is
            # the same disagreement and stays just as worth knowing.
            res.add('warn', code('sign_unexpected_positive'),
                    f'Measured kt is positive but {because} predicts negative for '
                    f'{motor}. The numbers below are unaffected, but the config '
                    f'and the rig disagree about this motor\'s mounting -- check '
                    f'the absorbers.yaml entry against how it is actually bolted '
                    f'in.')

        # -- fits ---------------------------------------------------------
        branches = []
        for b in blocks:
            fit = _fit_branch(b['current'], b['torque'])
            fit['direction'] = b['direction']
            fit['level_position_rad'] = b['level_position_rad']
            branches.append(fit)

        pooled = _fit_branch(all_i, all_t)
        kt = pooled['linear']['kt_Nm_per_A']

        # Each branch's offset about the pooled *tanh* trend, alongside the
        # linear one it already carries. Detrending the hysteresis loop with a
        # straight line leaves the model's own curvature in the picture, which
        # on a saturating motor is the larger signal of the two.
        if 'A_Nm' in pooled['tanh']:
            tA, tB = pooled['tanh']['A_Nm'], pooled['tanh']['B_per_A']
            for blk, fit in zip(blocks, branches):
                fit['tanh_offset_Nm'] = float(np.mean(
                    blk['torque'] - tanh_model(blk['current'], tA, tB)))

        # -- hysteresis: the reason branches are never pooled for reporting --
        ups = [b for b in branches if b['direction'] == 'up']
        dns = [b for b in branches if b['direction'] == 'down']
        hysteresis = hysteresis_tanh = None
        if ups and dns:
            up_off = float(np.mean([b['linear']['offset_Nm'] for b in ups]))
            dn_off = float(np.mean([b['linear']['offset_Nm'] for b in dns]))
            hysteresis = abs(up_off - dn_off)
            # The same gap measured about the tanh trend. Agreement between the
            # two is the check that the loop is physical rather than an artifact
            # of detrending a curved characteristic with a straight line.
            if all('tanh_offset_Nm' in b for b in ups + dns):
                hysteresis_tanh = abs(
                    float(np.mean([b['tanh_offset_Nm'] for b in ups]))
                    - float(np.mean([b['tanh_offset_Nm'] for b in dns])))
            kt_up = float(np.mean([b['linear']['kt_Nm_per_A'] for b in ups]))
            kt_dn = float(np.mean([b['linear']['kt_Nm_per_A'] for b in dns]))
            kt_dir_pct = 100 * abs(kt_up - kt_dn) / abs(kt_up) if kt_up else float('nan')

            noise = seg.tare.get(tch, {}).get('stddev')
            noise_txt = (f' ({hysteresis / noise:.1f}x the {tch} tare noise of '
                         f'{noise:.3f} Nm)') if noise else ''
            agree_txt = (
                f' Measured about the tanh trend instead it is '
                f'{hysteresis_tanh:.3f} Nm, so the loop is not an artifact of '
                f'detrending a curved characteristic with a straight line.'
                if hysteresis_tanh is not None else '')
            res.add('info', code('hysteresis'),
                    f'Up/down ramp offsets differ by {hysteresis:.3f} Nm'
                    f'{noise_txt}.{agree_txt} kt differs by only {kt_dir_pct:.3f}% between '
                    f'directions, so the slope is unaffected. Candidate sources: '
                    f'magnetic hysteresis, fixture friction as windup reverses, '
                    f'torque-cell hysteresis. Not attributed here.')
            if abs(pooled['linear']['offset_Nm']) < hysteresis / 4:
                res.add('info', code('pooled_offset_cancels'),
                        f'The pooled offset ({pooled["linear"]["offset_Nm"]:+.3f} Nm) '
                        f'is much smaller than the branch gap because the two '
                        f'branches cancel. Do not read it as an absence of bias.')
        else:
            res.add('warn', code('single_direction'),
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
            res.add('info', code('park_angle_spread'),
                    f'kt across {len(positions)} park angles spans '
                    f'{kt_spread:.4f} Nm/A ({100 * kt_spread / kt:.2f}%). This '
                    f'spread is the cogging contribution to kt and is a more '
                    f'honest error bar than the fit standard error.')
        else:
            res.add('warn', code('single_park_angle'),
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
            res.add('warn', code('saturation_not_exercised'),
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
            res.add('info', code('model_choice'),
                    f'Linear R2 = {lin_r2:.5f} vs tanh R2 = {tanh_r2:.5f} '
                    f'(delta {tanh_r2 - lin_r2:+.2e}). The tanh does not earn its '
                    f'extra parameter on this data; use the linear kt.')

        # -- kt vs config: the headline ----------------------------------
        kt_delta_pct = None
        if cfg_kt:
            kt_delta_pct = 100 * (abs(kt) - cfg_kt) / cfg_kt
            level = ('warn' if abs(kt_delta_pct) > params['kt_tolerance_pct']
                     else 'info')
            res.add(level, code('kt_vs_config'),
                    f'Measured kt = {abs(kt):.4f} Nm/A vs configured '
                    f'{cfg_kt} Nm/A for {motor}: {kt_delta_pct:+.2f}%.')

        # -- drive-ready constants ----------------------------------------
        # The point of the tanh fit is that it is the drive's own torque model,
        # so report it in the drive's own coordinates rather than leaving the
        # A/B -> kt/k_tanh conversion as an exercise.
        drive = drive_params_from_tanh(pooled['tanh'])
        if drive:
            t_limit = mp.get('motor_limits', {}).get('torque')
            res.add('info', code('drive_params'),
                    f'Drive-ready constants for {motor}, from the pooled tanh '
                    f'fit (kt = A*B, k_tanh = B -- the drive solves '
                    f'I = arctanh(T*k_tanh/kt)/k_tanh, which is this same '
                    f'curve):\n\n'
                    f'      motor_params: {{ kt: {drive["kt"]:.4f}, '
                    f'k_tanh: {drive["k_tanh"]:.6f} }}\n')
            if not sat_ok:
                res.add('warn', code('drive_params_extrapolated'),
                        f'Those constants are only as good as the saturation '
                        f'they were fitted through, and this sweep did not '
                        f'reach it. k_tanh is the extrapolated curvature; A and '
                        f'B trade off almost perfectly here (correlation '
                        f'{pooled["tanh"].get("AB_correlation", float("nan")):.4f}), '
                        f'so their product kt is trustworthy but the split '
                        f'between them is not. Prefer the linear kt with '
                        f'k_tanh left near zero until a sweep drives this motor '
                        f'into saturation.')
            if t_limit and t_limit >= drive['asymptote_Nm']:
                res.add('error', code('torque_limit_above_asymptote'),
                        f'Do not apply these constants as-is: '
                        f'motor_limits.torque = {t_limit} Nm meets or exceeds '
                        f'the tanh asymptote kt/k_tanh = '
                        f'{drive["asymptote_Nm"]:.2f} Nm. The drive computes '
                        f'arctanh(T*k_tanh/kt), which diverges there -- a '
                        f'command at the limit would demand infinite or NaN '
                        f'current. Raise the asymptote or lower the limit.')
            elif t_limit:
                res.add('info', code('torque_limit_headroom'),
                        f'motor_limits.torque = {t_limit} Nm sits at '
                        f'{100 * t_limit / drive["asymptote_Nm"]:.0f}% of the '
                        f'{drive["asymptote_Nm"]:.2f} Nm asymptote, so the '
                        f'drive\'s arctanh stays in range across the commandable '
                        f'span.')

        # -- thermal ------------------------------------------------------
        # This motor's own sensor when the bench has one. The fallback to any
        # '*_stator_temp' is kept because most benches instrument only one
        # motor, but which sensor was used is recorded either way -- with two
        # motors characterized, an unlabelled temperature belonging to the other
        # one is a number that reads as fact and is not.
        own_temp = f'{seg.prefix(motor)}_stator_temp'
        temp_ch = own_temp if seg.has(own_temp) else next(
            (c for c in seg.channels() if c.endswith('_stator_temp')), None)
        peak_t = float(np.max(np.abs(all_t)))
        over_cont = t_cont and peak_t > t_cont
        out = {}
        if temp_ch and seg.is_active(temp_ch):
            temps = np.concatenate([s[temp_ch] for s in segs])
            out['stator_temp_channel'] = temp_ch
            out['stator_temp_start_C'] = float(np.nanmin(temps))
            out['stator_temp_rise_C'] = float(np.nanmax(temps) - np.nanmin(temps))
        elif over_cont:
            res.add('warn', code('thermal_unverified'),
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
            out['torque_cell_fs_used_pct'] = frac
            if frac < 20:
                res.add('info', code('low_cell_utilisation'),
                        f'Peak torque uses {frac:.1f}% of {tch} full scale '
                        f'({cell["full_scale"]} Nm); tare noise was '
                        f'{cell.get("stddev", float("nan")):.3f} Nm.')

        # -- metrics ------------------------------------------------------
        out.update({
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
            'hysteresis_about_tanh_Nm': hysteresis_tanh,
            # Drive-ready: paste straight into motor_params.
            'drive_kt_Nm_per_A': drive['kt'] if drive else None,
            'drive_k_tanh_per_A': drive['k_tanh'] if drive else None,
            'tanh_asymptote_Nm': drive['asymptote_Nm'] if drive else None,
            'peak_current_A': max_i,
            'i_cont_A': i_cont,
            'current_utilisation_pct': 100 * util if util else None,
            'saturation_exercised': sat_ok,
            'peak_torque_Nm': peak_t,
            'pooled': pooled,
            'branches': branches,
        })

        summary = (
            f'{motor}: kt = {abs(kt):.4f} Nm/A'
            + (f' ({kt_delta_pct:+.2f}% vs configured {cfg_kt})' if cfg_kt else '')
            + f', from {len(branches)} branches across '
              f'{len(positions) if positions else 1} park angle(s). '
            + (f'Hysteresis {hysteresis:.3f} Nm. ' if hysteresis else '')
            + (f'Saturation not exercised ({100 * util:.0f}% of i_cont).'
               if util and not sat_ok else '')
        )

        # The overlay is a plotting choice only -- kt_vs_config above still
        # reports the comparison either way.
        overlay_kt = cfg_kt if params['overlay_configured_kt'] else None
        res.figures.extend(
            (slug(name), fig) for name, fig in
            self._figures(branches, blocks, pooled, all_i, all_t, motor, cch,
                          tch, flip, overlay_kt, params['overlay_linear_fit']))
        res.tables.append((slug('fit_curve'), self._csv(pooled, all_i, params)))
        return out, summary

    # -- outputs -----------------------------------------------------------

    def _figures(self, branches, blocks, pooled, all_i, all_t, motor, cch, tch,
                 flip, cfg_kt, show_linear=True):
        import matplotlib.pyplot as plt

        figs = []
        colors = {'up': 'tab:blue', 'down': 'tab:red'}
        note = ' (torque sign normalised)' if flip < 0 else ''

        # 1. The hysteresis loop, once per model. Plotted with the pooled trend
        #    removed: a sub-Nm loop is invisible on a full-scale torque axis,
        #    and this figure exists precisely to show it. Which trend gets
        #    removed matters -- see _hysteresis_fig.
        kt_p = pooled['linear']['kt_Nm_per_A']
        models = [('', lambda b: b['linear']['offset_Nm'],
                   lambda i: kt_p * i, 'linear',
                   f'the {abs(kt_p):.4f} Nm/A linear trend')]
        if 'A_Nm' in pooled['tanh']:
            tA, tB = pooled['tanh']['A_Nm'], pooled['tanh']['B_per_A']
            models.append(('_tanh', lambda b: b.get('tanh_offset_Nm'),
                           lambda i: tanh_model(i, tA, tB), 'tanh',
                           f'the tanh trend (kt = '
                           f"{abs(pooled['tanh']['small_signal_kt_Nm_per_A']):.4f}, "
                           f'k_tanh = {abs(tB):.6f})'))

        for suffix, getoff, trend, name, trend_txt in models:
            if any(getoff(b) is None for b in branches):
                continue
            figs.append((f'hysteresis{suffix}', self._hysteresis_fig(
                plt, branches, blocks, getoff, trend, colors, all_i,
                motor, cch, tch, note, name, trend_txt)))

        # 2. Pooled fit with the configured kt overlaid for comparison.
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.scatter(all_i, all_t - pooled['linear']['offset_Nm'], s=2, alpha=0.15,
                   color='steelblue', label='measured (de-biased)')
        x = np.linspace(float(np.min(all_i)), float(np.max(all_i)), 400)
        kt = pooled['linear']['kt_Nm_per_A']
        if show_linear:
            ax.plot(x, kt * x, 'k:', lw=2,
                    label=f'linear: kt = {abs(kt):.4f} Nm/A, '
                          f"R2 = {pooled['linear']['r_squared']:.5f}")
        if 'A_Nm' in pooled['tanh']:
            ax.plot(x, tanh_model(x, pooled['tanh']['A_Nm'], pooled['tanh']['B_per_A']),
                    'g--', lw=1.5,
                    label=f"tanh: kt = {abs(pooled['tanh']['small_signal_kt_Nm_per_A']):.4f}, "
                          f"R2 = {pooled['tanh']['r_squared']:.5f}")
        if cfg_kt:
            ax.plot(x, np.sign(kt) * cfg_kt * x, 'r', lw=1.5,
                    label=f'specsheet kt = {cfg_kt} Nm/A')
        # The tanh curve restated in the drive's own coordinates, monospaced so
        # it can be read off the figure and pasted into the config unaltered.
        # The equation above the block is the forward form the drive inverts,
        # so the two constants are readable as more than bare numbers.
        drive = drive_params_from_tanh(pooled['tanh'])
        if drive:
            ax.text(0.02, 0.97,
                    f'T = (kt / k_tanh) * tanh(k_tanh * I)\n'
                    f'motor_params:\n'
                    f'  kt:     {drive["kt"]:.4f}\n'
                    f'  k_tanh: {drive["k_tanh"]:.6f}',
                    transform=ax.transAxes, va='top', ha='left',
                    family='monospace', fontsize=10,
                    bbox=dict(boxstyle='round', fc='lightyellow', ec='gray'))
        ax.set_xlabel(f'{cch} (A)')
        ax.set_ylabel(f'{tch} (Nm){note}')
        ax.set_title(f'{motor}: pooled fit vs configured')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        figs.append(('pooled_fit', fig))

        # 3. Residuals -- where each model actually fails, if it does. Both are
        #    plotted so the curvature the linear fit leaves behind can be read
        #    directly against what the tanh removes.
        resid_models = [
            ('', np.polyval([kt, pooled['linear']['offset_Nm']], all_i),
             'Linear fit residuals (curvature here = real saturation)')]
        if 'A_Nm' in pooled['tanh']:
            resid_models.append((
                '_tanh',
                tanh_model_offset(all_i, pooled['tanh']['A_Nm'],
                                  pooled['tanh']['B_per_A'],
                                  pooled['tanh']['C_Nm']),
                'Tanh fit residuals (flat here = saturation fully accounted for)'))

        for suffix, model_t, title in resid_models:
            figs.append((f'residuals{suffix}', self._residual_fig(
                plt, all_i, all_t - model_t, cch, title)))

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

    def _hysteresis_fig(self, plt, branches, blocks, getoff, trend, colors,
                        all_i, motor, cch, tch, note, name, trend_txt):
        """One hysteresis loop, with `trend` subtracted from the torque.

        The choice of trend is not cosmetic on a saturating motor. Removing a
        straight line from a curved characteristic leaves that curvature in the
        figure as a bow, and the bow can be larger than the loop it is drawn to
        show -- on the log this was built against, 0.636 Nm of linear-fit
        curvature against a 0.304 Nm gap. Removing the tanh instead leaves the
        loop as almost the only thing on the axes.
        """
        fig, ax = plt.subplots(figsize=(11, 7))
        seen = set()
        for b in branches:
            d = b['direction']
            lbl = d if d not in seen else None
            seen.add(d)
            ax.axhline(getoff(b), color=colors.get(d, 'gray'),
                       lw=1.2, alpha=0.9, label=lbl)
        x = np.linspace(float(np.min(all_i)), float(np.max(all_i)), 300)
        for b in blocks:
            ax.scatter(b['current'], b['torque'] - trend(b['current']), s=1,
                       alpha=0.06, color=colors.get(b['direction'], 'k'), zorder=0)
        ups = [getoff(b) for b in branches if b['direction'] == 'up']
        dns = [getoff(b) for b in branches if b['direction'] == 'down']
        if ups and dns:
            gap = abs(float(np.mean(ups)) - float(np.mean(dns)))
            ax.annotate('', xy=(x[10], float(np.mean(ups))),
                        xytext=(x[10], float(np.mean(dns))),
                        arrowprops=dict(arrowstyle='<->', color='green', lw=1.5))
            ax.text(x[12], float(np.mean(ups + dns)), f'  {gap:.3f} Nm',
                    color='green', va='center')
        ax.set_xlabel(f'{cch} (A)')
        ax.set_ylabel(f'{tch} minus {name} trend (Nm){note}')
        ax.set_title(f'{motor}: hysteresis loop (torque with {trend_txt} removed)')
        ax.axhline(0, color='k', lw=0.5, ls=':')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    def _residual_fig(self, plt, all_i, resid, cch, title):
        """Residual scatter with the binned mean overlaid.

        The scatter alone is a cloud; the binned mean is what separates a
        systematic modelling error from noise, and is the whole reason to look
        at the linear and tanh versions side by side.
        """
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.scatter(all_i, resid, s=1, alpha=0.1, color='k')
        edges = np.linspace(float(np.min(all_i)), float(np.max(all_i)), 25)
        idx = np.digitize(all_i, edges)
        centers, means = [], []
        for i in range(1, len(edges)):
            sel = idx == i
            if sel.sum() > 20:
                centers.append(0.5 * (edges[i - 1] + edges[i]))
                means.append(float(np.mean(resid[sel])))
        if means:
            ax.plot(centers, means, 'o-', color='tab:orange', ms=3, lw=1.5,
                    label=f'binned mean (systematic span '
                          f'{max(means) - min(means):.3f} Nm)')
            ax.legend()
        ax.axhline(0, color='r', lw=1)
        ax.set_xlabel(f'{cch} (A)')
        ax.set_ylabel('residual (Nm)')
        ax.set_title(f'{title}  --  RMS {float(np.sqrt(np.mean(resid ** 2))):.4f} Nm')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

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
