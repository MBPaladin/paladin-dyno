"""No-load running torque, forward-driven or back-driven, plus its ripple spectrum.

Physics
-------
One motor sweeps a velocity triangle; the shaft on the far side of the gearbox
is free. Nothing is transmitted, so the inline torque cell reads pure drag:

    T(w) = T_coulomb * sgn(w) + b * w + ripple(theta, w) + J * dw/dt

Binning by velocity is what makes the first two terms recoverable. Every speed
is visited four times (accelerating and decelerating, in both directions), so
the inertial term -- which is antisymmetric in dw/dt -- averages out of a bin
while the drag terms, which depend only on w, do not. The residual scatter
inside a bin is the ripple, and that is the thing the spectral half of this
processor is about.

Ripple has two populations that look identical in a single spectrum and are
trivially separable once speed is on an axis:

  * a **structural resonance** is a horizontal line -- fixed frequency, no
    matter how fast the shaft turns;
  * an **order** (mesh, cogging, imbalance) is a ray through the origin,
    f = n * |w| / 2pi, so it climbs with speed.

Amplitude peaks where a ray crosses a line. That crossing is the only reason
the two views (vs time, vs |w|) both exist: the time view shows *when* it
happened, the speed-folded view shows *at what speed*, and only the latter is
comparable between two runs that swept different peak speeds.

Ambient floor
-------------
A lead-in during which nothing is commanded gives a same-hardware, same-cell,
same-day noise floor. Every spectral figure is emitted twice: once in absolute
Nm with that floor drawn on it, and once divided by it per frequency bin, so
0 dB is ambient and only what the rotation added stands up. The pair is
deliberate -- a floor-referenced plot alone cannot show how much was removed,
and a subtraction artifact is indistinguishable from a real line in it.

Applicability is a predicate over the data (one motor sweeping speed, a torque
cell that responds to rotation), not a match on the behavior name.
"""

import numpy as np

from ..processor import Applicability, Processor, Result
from ..registry import register

_TWO_PI = 2.0 * np.pi


# -- signal helpers ---------------------------------------------------------

def sample_rate(t):
    """fs from the record span, never from median(diff(t)).

    `time` is logged float32. Sixty seconds in, its resolution is ~4 us, so a
    1 ms tick quantizes to 0.999 ms or 1.007 ms and `1/median(diff(t))`
    misreports fs by up to 0.8% -- by a *different* amount for each log, since
    it depends where in the float32 exponent the record happens to sit. Two
    runs of the same rig would then place the same physical resonance at
    different frequencies. The end-to-end span absorbs a single quantization
    step over the whole record, so it is right to ~1e-7.
    """
    return float((len(t) - 1) / (t[-1] - t[0]))


def moving_average(x, n):
    """Centered rolling mean, edge-padded so the output length is preserved."""
    n = max(1, int(n) | 1)                    # odd, so the window is centered
    if n <= 1:
        return x.astype(float, copy=True)
    pad = n // 2
    padded = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(padded, np.ones(n) / n, mode='valid')


def stft(sig, fs, win_n, hop):
    """Amplitude STFT. Returns (freqs, frames[n_frames, n_freq], start_idx).

    Scaled so a pure sinusoid reads its own peak amplitude in the signal's
    units, which is what makes the ripple numbers below quotable in Nm.

    The window is the PERIODIC Hann (np.hanning is the symmetric variant,
    which slightly biases amplitudes quoted in Nm), and the FFT is zero-padded
    2x: plain Hann scalloping loss reaches 1.4 dB for a line landing between
    bins, which would bias dominant_line_Nm directly.
    """
    win_n = min(int(win_n), len(sig))
    win = np.hanning(win_n + 1)[:-1]          # periodic Hann
    nfft = 2 * win_n
    fr = np.fft.rfftfreq(nfft, 1.0 / fs)
    starts = np.arange(0, len(sig) - win_n + 1, hop, dtype=int)
    frames = np.empty((len(starts), len(fr)))
    for i, k in enumerate(starts):
        seg = sig[k:k + win_n]
        frames[i] = np.abs(np.fft.rfft((seg - seg.mean()) * win,
                                       n=nfft)) * 2 / win.sum()
    return fr, frames, starts


def _parse_orders(text):
    out = []
    for tok in str(text).replace(';', ',').split(','):
        tok = tok.strip()
        if tok:
            try:
                out.append(float(tok))
            except ValueError:
                pass
    return out


def _db(x, ref):
    return 20.0 * np.log10(np.asarray(x) / ref + 1e-12)


def _nice_bin(span, target=50):
    """A readable bin width giving roughly `target` bins across `span`:
    the raw quotient snapped to a 1/2/2.5/5 x 10^k grid."""
    raw = max(span / target, 1e-9)
    mag = 10.0 ** np.floor(np.log10(raw))
    return float(min((mag * m for m in (1.0, 2.0, 2.5, 5.0, 10.0)),
                     key=lambda w: abs(w - raw)))


@register
class RunningTorque(Processor):
    name = 'running_torque'
    title = 'Running Torque (forward / backdrive)'
    description = (
        'No-load drag of a gearbox driven through a velocity sweep with the far '
        'shaft free. Fits Coulomb and viscous drag from the velocity-binned '
        'torque, then resolves the ripple against both time and |speed| so that '
        'fixed structural resonances separate from shaft orders. A stationary '
        'lead-in, when present, is used as the ambient noise floor.'
    )
    granularity = 'run'

    params = {
        # -- what to read (auto-detected; override to force) ----------------
        'motor':            (None,   str,   'Driving motor (auto-detected)'),
        'velocity_channel': (None,   str,   'Velocity channel (auto-detected)'),
        'torque_channel':   (None,   str,   'Torque cell channel (auto-detected)'),
        # -- rig description -------------------------------------------------
        'gear_ratio':       (1.0,    float, 'Reduction ratio, input:output'),
        'driven_shaft':     ('output', str, "Shaft the driving motor turns: 'output' "
                                            "(back-drive) or 'input' (forward drive)"),
        # -- drag curve ------------------------------------------------------
        'smooth_s':         (0.1,    float, 'Rolling-average width used to detrend '
                                            'torque into ripple [s]'),
        'vel_bin_rad_s':    (0.0,    float, 'Velocity bin width for the drag curve '
                                            '[rad/s]; 0 = derive from the sweep '
                                            'span (~50 occupied bins)'),
        'min_bin_samples':  (200,    int,   'Samples a velocity bin needs to be reported'),
        'fit_min_vel':      (1.0,    float, 'Ignore |w| below this in the Coulomb+viscous fit [rad/s]'),
        'tare_from_leadin': (True,   bool,  'Subtract the stationary lead-in mean from the torque cell'),
        'invert_torque_sign': (False, bool, 'Negate the torque cell against velocity, so '
                                            'drag reads negative when turning positive. '
                                            'Affects the drag curve and its fits only; '
                                            'the spectra are amplitude-only and unchanged'),
        'symmetric_drag':   (True, bool,  'Assume drag is the same in both directions, '
                                           'so the offset between the two flank '
                                           'intercepts is torque-cell bias rather than '
                                           'physics. Subtracts that common-mode bias, '
                                           'which puts the origin midway between the '
                                           'fitted flanks. Coulomb and viscous terms are '
                                           'unchanged; the per-direction intercepts and '
                                           'the folded curves are not'),
        # -- spectral --------------------------------------------------------
        'stft_win':         (0,      int,   'STFT window [samples]; 0 = derive '
                                            'from the measured sweep rate and '
                                            'the highest overlaid order'),
        'stft_overlap':     (0.75,   float, 'STFT window overlap, 0-1'),
        'fmax_hz':          (500.0,  float, 'Top of the plotted frequency axis [Hz]'),
        'fmin_hz':          (1.0,    float, 'Ignore below this when picking dominant lines [Hz]'),
        'speed_source':     ('measured', str, "Speed axis from 'measured' or 'command' velocity"),
        'speed_bin_rad_s':  (0.0,   float, 'Speed bin width for the folded '
                                           'spectrogram [rad/s]; 0 = derive '
                                           'from peak speed (~50 bins)'),
        'moving_min_speed': (0.5,   float, 'Frames slower than this are not part of '
                                            'the moving record [rad/s]'),
        'orders':           ('1,5,10', str,  'Driven-shaft orders to overlay'),
        'overlay_fast_shaft': (False, bool,  'Overlay the far-shaft (mesh/cogging) '
                                            'orders as well as the driven-shaft ones'),
        'db_floor':         (0.0,    float, 'Bottom of the colour scale [dB]; '
                                            '0 = derive from the map\'s actual '
                                            'dynamic range'),
        'resonance_overlay_hz': ('', str,   'Comma-separated frequencies to draw '
                                            'as horizontal lines on the '
                                            'spectrograms (and hyperbolas on the '
                                            'Campbell map); populate from '
                                            'multisine_frf\'s '
                                            'structural_resonances_hz metric'),
        'min_static_frames': (3,     int,   'STFT frames the lead-in needs to serve as a floor'),
        'order_min_speed_frac': (0.4, float, 'Order spectrum ignores frames below this '
                                             'fraction of peak speed, where order '
                                             'resolution collapses'),
        'n_order_peaks':    (4,      int,   'Order peaks to label and report'),
        # -- which figures to emit (all independently switchable) ------------
        'fig_drag':          (True,  bool,  'Drag torque vs velocity'),
        'fig_order_spectrum': (True, bool,  'Amplitude vs shaft order'),
        'fig_spectrogram':   (True,  bool,  'Spectrograms (vs time and vs |speed|)'),
        'fig_campbell':      (True,  bool,  'Campbell/order map: |speed| vs order, '
                                            'so orders are horizontal lines and '
                                            'resonances are hyperbolas'),
        'fig_waterfall':     (False,  bool,  'Waterfalls (vs time and vs |speed|)'),
        'fig_vs_time':       (True,  bool,  'Emit the time-axis members of the above'),
        'fig_vs_speed':      (True,  bool,  'Emit the |speed|-axis members of the above'),
        'fig_absolute':      (True,  bool,  'Emit the absolute-amplitude (Nm) variants'),
        'fig_floor_ref':     (True,  bool,  'Emit the ambient-floor-referenced variants'),
        'mean_spectrum_panel': (True, bool, 'Stack the record-averaged spectrum above each spectrogram'),
        'waterfall_n':       (14,    int,   'Traces drawn in a waterfall'),
    }

    # -- channel discovery --------------------------------------------------

    def _motors(self, seg):
        """Config devices that expose a velocity channel -- i.e. the motors."""
        return [d for d in seg.devices() if seg.has(f'{seg.prefix(d)}_velocity')]

    def _torque_channels(self, seg):
        # Sensors only: '*_torque' excludes '*_torque_command' by construction.
        return [c for c in seg.channels() if c.endswith('_torque') and seg.is_active(c)]

    def _lead_in(self, seg, vch, vcmd_ch, vel_tol=0.05, cmd_tol=1e-6):
        """Length of the stationary, uncommanded run of samples at the start.

        Read from the command rather than assumed from the test's `lead_in_s`:
        the builder's lead-in and the secondary axis' settle time both land in
        it, so the data knows the real number and the recipe does not.

        'Uncommanded' means NO torque or velocity command channel is active --
        not merely the sweep's own velocity command. A torque-commanded
        excitation is stationary in velocity terms and would otherwise pass as
        'at rest', silently folding real excitation into the noise floor.
        (Position commands are exempt: a held position is non-zero at rest.)
        """
        v = np.abs(np.nan_to_num(seg[vch], nan=np.inf))
        still = v <= vel_tol
        for ch in seg.channels():
            if ch.endswith(('_velocity_command', '_torque_command')) \
                    and seg.is_active(ch):
                cmd = np.nan_to_num(seg[ch], nan=0.0)
                still &= np.abs(cmd) <= cmd_tol
        if not still[0]:
            return 0
        stop = np.argmin(still)                  # first False
        return int(stop) if not still[stop] else int(len(still))

    # -- applicability ------------------------------------------------------

    def applies_to(self, segs):
        seg = segs[0]
        d = self.defaults()
        cat = lambda ch: np.concatenate([s[ch] for s in segs])

        # 1. One motor is turning under a velocity command. A torque-commanded
        #    sweep is the stiction test, not this; a stationary log is the
        #    blocked-rotor characterization.
        #
        #    Ranked by whether the axis is actually velocity-COMMANDED before
        #    how fast it spun. Peak speed alone identifies the driver only on a
        #    rig where the far shaft is free; on a directly-coupled bench both
        #    motors read the same sweep to within sensor noise, so a bare
        #    max-by-speed lets noise pick the driver, and the log then fails the
        #    command check below as 'torque-driven' when it is nothing of the
        #    sort. The command channel says who is driving; the tachometer only
        #    says what turned.
        spinning = []
        for dev in self._motors(seg):
            vch = f'{seg.prefix(dev)}_velocity'
            v = np.nan_to_num(cat(vch), nan=0.0)
            commanded = seg.is_active(f'{seg.prefix(dev)}_velocity_command')
            spinning.append((commanded, float(np.max(np.abs(v))), dev, vch))
        if not spinning:
            return Applicability('no', 'no motor exposes a velocity channel')
        _, peak, motor, vch = max(spinning)
        if peak < 4 * d['fit_min_vel']:
            return Applicability(
                'no', f'fastest axis is {motor} at {peak:.3f} rad/s peak; nothing '
                      f'sweeps a usable speed range')

        vcmd_ch = f'{seg.prefix(motor)}_velocity_command'
        if not seg.is_active(vcmd_ch):
            return Applicability(
                'no', f'{motor} turns but {vcmd_ch} is never commanded; this is a '
                      f'torque-driven test, not a velocity sweep')

        # 2. The sweep has to visit many speeds, or there is no drag curve to
        #    fit and no ray to distinguish from a resonance.
        v = np.nan_to_num(cat(vch), nan=0.0)
        bw = d['vel_bin_rad_s'] or _nice_bin(float(v.max() - v.min()))
        occupied = np.bincount(np.clip(
            ((v - v.min()) // bw).astype(int), 0, None))
        n_bins = int(np.count_nonzero(occupied >= d['min_bin_samples']))
        if n_bins < 6:
            return Applicability(
                'no', f'{vch} occupies only {n_bins} velocity bin(s) of width '
                      f'{bw} rad/s; a dwell, not a sweep')

        # 3. Which cell is in the load path? The one whose AC content grows when
        #    the shaft turns. A cell that is bolted out of the load path reads
        #    the same noise moving or not, which no amplitude threshold would
        #    catch -- an idle 500 Nm cell is noisier than a loaded 20 Nm one.
        tchs = self._torque_channels(seg)
        if not tchs:
            return Applicability('no', 'no torque sensor channel found')

        fs = sample_rate(cat('time'))
        win = max(3, int(round(d['smooth_s'] * fs)))
        n_lead = self._lead_in(seg, vch, vcmd_ch)
        moving = np.abs(v) > d['moving_min_speed']
        scored = []
        for ch in tchs:
            tq = np.nan_to_num(cat(ch), nan=0.0)
            ac = tq - moving_average(tq, win)
            run_rms = float(np.std(ac[moving])) if moving.any() else 0.0
            rest = (float(np.std(ac[:n_lead])) if n_lead > win
                    else float(np.std(ac[~moving])) if (~moving).any() else 0.0)
            scored.append((run_rms / max(rest, 1e-12), run_rms, ch))
        gain, run_rms, tch = max(scored)
        if gain < 2.0:
            return Applicability(
                'no', f'no torque cell responds to rotation (best is {tch}, AC RMS '
                      f'only {gain:.2f}x its at-rest value); nothing is measuring '
                      f'this shaft')

        lead_txt = (f'{n_lead / fs:.1f} s stationary lead-in available as a noise '
                    f'floor' if n_lead > win else 'no stationary lead-in in this log')
        return Applicability(
            'yes',
            f'{motor} sweeps {vch} to {peak:.2f} rad/s across {n_bins} velocity '
            f'bins under velocity command; {tch} AC RMS is {gain:.1f}x higher '
            f'turning than at rest ({run_rms:.3f} Nm); {lead_txt}',
            {'motor': motor, 'velocity_channel': vch, 'torque_channel': tch})

    # -- run ----------------------------------------------------------------

    def run(self, segs, params):
        res = Result()
        seg = segs[0]
        motor = params['motor']
        vch, tch = params['velocity_channel'], params['torque_channel']
        vcmd_ch = f'{seg.prefix(motor)}_velocity_command'
        ratio = float(params['gear_ratio'])
        backdrive = str(params['driven_shaft']).lower().startswith('out')

        fs = sample_rate(np.concatenate([s['time'] for s in segs]))
        naive_fs = 1.0 / seg.dt

        smooth_n = max(3, int(round(params['smooth_s'] * fs)))

        # Ports: label each stream side by what the config says is attached,
        # so two reports from different mechanical configurations stay
        # distinguishable at a glance (the baseline-vs-gearbox workflow).
        self._attached = {}
        for port in seg.ports():
            side = {'DUT': 'input', 'LOAD': 'output'}.get(port.get('device'))
            attached = port.get('attached')
            if side and attached and attached != 'none':
                self._attached[side] = str(attached)

        # No analog anti-alias filter sits ahead of the torque cell, so real
        # content above Nyquist folds down and appears as genuine lines.
        if float(params['fmax_hz']) >= 0.49 * fs:
            res.add('warn', 'aliasing_possible',
                    f'fmax_hz={params["fmax_hz"]:g} is essentially Nyquist '
                    f'({fs / 2:.0f} Hz) and there is no analog anti-alias '
                    f'filter ahead of the cell: real content above '
                    f'{fs / 2:.0f} Hz folds down and is indistinguishable from '
                    f'a genuine line. Treat unexplained lines near the top of '
                    f'the band with suspicion.')

        # -- gather, tare -------------------------------------------------
        v = np.concatenate([np.nan_to_num(s[vch], nan=0.0) for s in segs])
        tq_raw = np.concatenate([np.nan_to_num(s[tch], nan=0.0) for s in segs])

        n_lead = self._lead_in(segs[0], vch, vcmd_ch)
        lead_s = n_lead / fs
        tare = float(np.mean(tq_raw[:n_lead])) if n_lead > smooth_n else 0.0
        if params['tare_from_leadin'] and tare:
            tq = tq_raw - tare
            res.add('info', 'tared',
                    f'{tch} read {tare:+.4f} Nm during the {lead_s:.1f} s stationary '
                    f'lead-in with nothing commanded, so that is cell offset, not '
                    f'drag, and it has been subtracted. It is '
                    f'{100 * abs(tare) / max(float(np.std(tq_raw)), 1e-9):.0f}% of the '
                    f'running signal RMS -- large enough to bias the Coulomb term '
                    f'if left in.')
        else:
            tq = tq_raw
            if n_lead <= smooth_n:
                res.add('warn', 'no_lead_in',
                        f'No stationary lead-in found at the start of this log, so '
                        f'{tch} could not be tared and there is no ambient noise '
                        f'floor to reference the spectra against. Add a lead-in '
                        f'(the builder\'s `lead_in_s`) to get both.')

        # Applied AFTER the tare, and to the drag path only. The tare is a cell
        # offset in the cell's own units, so it has to come off before the
        # convention is changed; the spectra rebuild their own signal from the
        # segments and are amplitude-only, where a global negation is a phase
        # flip that nothing downstream reads.
        invert = bool(params['invert_torque_sign'])
        if invert:
            tq = -tq
            res.add('info', 'torque_sign_inverted',
                    f'invert_torque_sign is set, so {tch} has been negated after '
                    f'taring: drag now reads negative while turning positive. The '
                    f'Coulomb and viscous magnitudes are unchanged -- this flips '
                    f'the signed per-direction intercepts and slopes, and the drag '
                    f'curve\'s quadrants, nothing else.')

        # -- drag curve ----------------------------------------------------
        bw = params['vel_bin_rad_s']
        if not bw:
            # Target ~50 occupied bins whatever the record: a fixed width is
            # wrong for both a 20 s and a 200 s run.
            bw = _nice_bin(float(v.max() - v.min()))
            res.add('info', 'derived_vel_bin',
                    f'vel_bin_rad_s derived as {bw:g} rad/s: the sweep spans '
                    f'{float(v.max() - v.min()):.2f} rad/s and ~50 bins keeps '
                    f'each one above min_bin_samples while resolving the drag '
                    f'curve. Set vel_bin_rad_s explicitly to override.')
        edges = np.arange(np.floor(v.min() / bw) * bw,
                          np.ceil(v.max() / bw) * bw + bw, bw)
        centers = 0.5 * (edges[:-1] + edges[1:])
        idx = np.clip(np.digitize(v, edges) - 1, 0, len(centers) - 1)
        ripple_sig = tq - moving_average(tq, smooth_n)

        bmean = np.full(len(centers), np.nan)
        bripple = np.full(len(centers), np.nan)
        bcount = np.zeros(len(centers), dtype=int)
        for b in range(len(centers)):
            sel = idx == b
            bcount[b] = int(np.count_nonzero(sel))
            if bcount[b] >= params['min_bin_samples']:
                bmean[b] = float(np.mean(tq[sel]))
                bripple[b] = float(np.std(ripple_sig[sel]))

        fmin_v = params['fit_min_vel']
        fit = {}
        for tag, mask in (('pos', centers >= fmin_v), ('neg', centers <= -fmin_v)):
            m = mask & ~np.isnan(bmean)
            if np.count_nonzero(m) >= 2:
                slope, intercept = np.polyfit(centers[m], bmean[m], 1)
                fit[tag] = (float(slope), float(intercept))

        coulomb = viscous = float('nan')
        if 'pos' in fit and 'neg' in fit:
            coulomb = abs(0.5 * (fit['pos'][1] - fit['neg'][1]))
            viscous = 0.5 * (abs(fit['pos'][0]) + abs(fit['neg'][0]))
            # Drag must oppose motion. Same-signed intercepts are a stuck bias
            # or a mis-signed cell, and would still produce a plausible-looking
            # Coulomb number from the difference.
            if np.sign(fit['pos'][1]) == np.sign(fit['neg'][1]):
                res.add('warn', 'drag_does_not_oppose',
                        f'The two branch intercepts have the same sign '
                        f'({fit["pos"][1]:+.3f} and {fit["neg"][1]:+.3f} Nm). Drag '
                        f'reverses with direction, so this is a residual bias in '
                        f'{tch}, not friction. The Coulomb figure below is not '
                        f'trustworthy.')
            elif np.sign(fit['pos'][1]) > 0:
                res.add('info', 'cell_sign',
                        f'{tch} reads positive while turning positive, i.e. it '
                        f'reports the torque the motor applies rather than the '
                        f'drag reacting to it. Magnitudes are unaffected; set '
                        f'invert_torque_sign=true to flip the drag curve into the '
                        f'drag-opposes-motion convention.'
                        + (' (invert_torque_sign is already set, so the cell\'s raw '
                           'convention was the opposite of this.)' if invert else ''))
        else:
            res.add('warn', 'single_direction',
                    f'Only one rotation direction cleared |w| >= {fmin_v} rad/s, so '
                    f'Coulomb and viscous drag cannot be separated -- a one-sided '
                    f'fit cannot tell a friction offset from a cell offset.')

        # -- symmetric-drag correction --------------------------------------
        # Under the symmetry assumption the two flanks are T = +-Tc + b*w + q,
        # with q a common-mode cell bias that no tare caught (a lead-in tare
        # only removes the offset present at standstill; a cell that shifts
        # once it is loaded or once the shaft turns keeps its own). Coulomb
        # already comes from the DIFFERENCE of the intercepts, so it never saw
        # q; the SUM is what q lives in, and halving it recovers it exactly.
        # Everything the fit reports is therefore unchanged except the two
        # signed intercepts, which is the point -- the curve moves, the drag
        # numbers do not.
        #
        # This is an assumption, not a measurement: a genuinely direction-
        # dependent drag (a preloaded lip seal, a helical stage thrusting into
        # a bearing one way and off it the other) is silently absorbed into
        # `cell_bias_Nm`. It is quoted as its own metric so it can be checked
        # against a second run taken with the cell rotated in its mount, where
        # a real bias flips with the cell and real asymmetry does not.
        symmetric = bool(params['symmetric_drag'])
        cell_bias = 0.0
        if symmetric and 'pos' in fit and 'neg' in fit:
            cell_bias = 0.5 * (fit['pos'][1] + fit['neg'][1])
            tq = tq - cell_bias
            bmean = bmean - cell_bias
            fit = {tag: (slope, intercept - cell_bias)
                   for tag, (slope, intercept) in fit.items()}
            res.add('info', 'symmetric_bias_removed',
                    f'symmetric_drag is set, so the {cell_bias:+.4f} Nm common-mode '
                    f'offset between the two flank intercepts is read as {tch} bias, '
                    f'not as direction-dependent friction, and has been subtracted. '
                    f'The origin now sits midway between the flanks and they read '
                    f'{fit["pos"][1]:+.3f} / {fit["neg"][1]:+.3f} Nm. That bias is '
                    f'{100 * abs(cell_bias) / max(coulomb, 1e-9):.0f}% of the '
                    f'{coulomb:.3f} Nm Coulomb term; Coulomb and viscous drag are '
                    f'unchanged by it. If the drag really is asymmetric, this has '
                    f'hidden that asymmetry -- clear symmetric_drag to see it.')
        elif symmetric:
            res.add('warn', 'symmetric_bias_unavailable',
                    f'symmetric_drag is set but only one direction cleared '
                    f'|w| >= {fmin_v} rad/s, so there is no second flank to centre '
                    f'against and no bias could be removed. The drag curve is '
                    f'uncorrected.')

        vpeak = float(np.max(np.abs(v)))
        drag_at_peak = coulomb + viscous * vpeak

        # Where the ripple dwarfs the drag, a single bin's mean is not a
        # measurement. Binning cancels inertia (each bin sees accel and decel
        # alike) but it cannot cancel ripple that the velocity loop itself
        # responds to: the controller lets speed wobble in sympathy with the
        # torque, so w and T are correlated at the ripple frequency and sorting
        # samples by w sorts them by ripple phase too.
        fitted = ~np.isnan(bmean) & (np.abs(centers) >= fmin_v)
        if fitted.any():
            worst = float(np.nanmax(bripple[fitted] / np.abs(bmean[fitted])))
            if worst > 1.0:
                res.add('warn', 'ripple_limited_bins',
                        f'In the worst velocity bin the ripple RMS is {worst:.1f}x '
                        f'the mean torque there. Quote the fitted Coulomb and '
                        f'viscous terms, which pool every bin; do not read a drag '
                        f'number off a single bin at the top of the sweep.')

        # -- spectra --------------------------------------------------------
        win_n = int(params['stft_win'])
        if win_n <= 0:
            # A harmonic drifting at df/dt smears by (df/dt)*T_win across the
            # window; keeping that under one bin width (1/T_win) gives
            # T_win <= 1/sqrt(df/dt), with df/dt taken for the fastest
            # OVERLAID order (that is the line the map exists to resolve).
            sweep = np.abs(np.diff(moving_average(np.abs(v), smooth_n))) * fs
            sweep_rate = float(np.percentile(sweep, 95)) if len(sweep) else 0.0
            o_max = max(_parse_orders(params['orders']) or [1.0])
            if params['overlay_fast_shaft'] and ratio != 1:
                o_max *= ratio
            dfdt = o_max * sweep_rate / _TWO_PI
            t_win = 1.0 / np.sqrt(dfdt) if dfdt > 0 else 1.0
            win_n = int(2 ** np.clip(np.round(np.log2(t_win * fs)), 7, 11))
            res.add('info', 'derived_stft_win',
                    f'stft_win derived as {win_n} samples ({win_n / fs:.2f} s): '
                    f'the sweep moves |w| at ~{sweep_rate:.2f} rad/s^2, so the '
                    f'{o_max:g}x order drifts at {dfdt:.2f} Hz/s and a window '
                    f'longer than 1/sqrt(df/dt) = {t_win:.2f} s would smear it '
                    f'past one bin width. Set stft_win explicitly to override.')
        hop = max(1, int(round(win_n * (1.0 - float(params['stft_overlap'])))))
        spec = self._spectra(segs, tch, vch, vcmd_ch, tare, fs, win_n, hop, params)

        fr, frames, f_speed, f_time = (spec['fr'], spec['frames'],
                                       spec['speed'], spec['time'])
        floor = spec['floor']
        moving = f_speed > params['moving_min_speed']
        if not moving.any():
            res.add('error', 'no_moving_frames', 'No STFT frame is above the '
                    'moving-speed threshold; nothing to analyze spectrally.')
            return res

        fmax = min(float(params['fmax_hz']), float(fr[-1]))
        keep = fr <= fmax
        lo = int(np.searchsorted(fr, params['fmin_hz']))
        mean_spec = frames[moving].mean(axis=0)
        dom_i = lo + int(np.argmax(mean_spec[lo:]))
        dom_hz, dom_amp = float(fr[dom_i]), float(mean_spec[dom_i])

        if floor is None:
            res.add('warn', 'no_noise_floor',
                    f'The stationary lead-in held fewer than '
                    f'{params["min_static_frames"]} whole STFT windows '
                    f'({lead_s:.1f} s at a {win_n / fs:.2f} s window), so no ambient '
                    f'floor could be measured and the floor-referenced figures are '
                    f'skipped. Either lengthen the lead-in or shorten stft_win.')
        else:
            snr = dom_amp / max(float(floor[dom_i]), 1e-12)
            res.add('info', 'dominant_line',
                    f'The strongest ripple line while turning is {dom_hz:.1f} Hz at '
                    f'{dom_amp:.3f} Nm, {snr:.0f}x the ambient floor at that '
                    f'frequency ({float(floor[dom_i]):.4f} Nm). At peak speed '
                    f'({vpeak:.2f} rad/s) that is order {dom_hz / (vpeak / _TWO_PI):.1f} '
                    f'of the driven shaft'
                    + (f', order {dom_hz / (ratio * vpeak / _TWO_PI):.1f} of the '
                       f'{"input" if backdrive else "output"} shaft at {ratio:g}:1.'
                       if ratio != 1 else '.'))
            self._classify_dominant(res, fr, frames, f_speed, moving, lo, dom_hz)

        # -- speed-folded spectra -------------------------------------------
        sbw = params['speed_bin_rad_s']
        if not sbw:
            sbw = _nice_bin(vpeak)
            res.add('info', 'derived_speed_bin',
                    f'speed_bin_rad_s derived as {sbw:g} rad/s (~50 bins over '
                    f'the {vpeak:.2f} rad/s sweep). Set speed_bin_rad_s '
                    f'explicitly to override.')
        s_edges = np.arange(0.0, np.ceil(vpeak / sbw) * sbw + sbw, sbw)
        s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
        s_bin = np.clip(np.digitize(f_speed, s_edges) - 1, 0, len(s_centers) - 1)
        by_speed = np.full((len(s_centers), len(fr)), np.nan)
        s_count = np.zeros(len(s_centers), dtype=int)
        for b in range(len(s_centers)):
            sel = s_bin == b
            if sel.any():
                by_speed[b] = frames[sel].mean(axis=0)
                s_count[b] = int(np.count_nonzero(sel))

        # -- order spectrum ---------------------------------------------------
        ospec = self._order_spectrum(fr, frames[moving], f_speed[moving], vpeak,
                                     fmax, params)
        top_orders = []
        if ospec is None:
            res.add('warn', 'no_order_spectrum',
                    f'Fewer than 3 STFT frames ran above '
                    f'{params["order_min_speed_frac"]:.0%} of peak speed, so the '
                    f'order spectrum was skipped. Lengthen the dwell at speed, or '
                    f'lower order_min_speed_frac and accept coarser order bins.')
        else:
            peaks = self._order_peaks(ospec['orders'], ospec['amp'],
                                      int(params['n_order_peaks']))
            top_orders = [{'order': float(ospec['orders'][i]),
                           'amplitude_Nm': float(ospec['amp'][i]),
                           'order_of_other_shaft': (float(ospec['orders'][i] / ratio)
                                                    if ratio != 1 else None)}
                          for i in peaks]
            self._check_ratio(res, ospec, ratio, backdrive, top_orders)

        # -- metrics ---------------------------------------------------------
        res.metrics.update({
            'motor': motor,
            'torque_channel': tch,
            'velocity_channel': vch,
            'drive_mode': 'back-drive' if backdrive else 'forward drive',
            'gear_ratio': ratio,
            'sample_rate_hz': fs,
            'lead_in_s': lead_s,
            'tare_Nm': tare if params['tare_from_leadin'] else 0.0,
            'torque_sign_inverted': invert,
            'symmetric_drag_assumed': symmetric,
            'cell_bias_Nm': cell_bias,
            'peak_speed_rad_s': vpeak,
            'coulomb_drag_Nm': coulomb,
            'viscous_drag_Nm_per_rad_s': viscous,
            'drag_at_peak_speed_Nm': drag_at_peak,
            'ripple_rms_max_Nm': (float(np.nanmax(bripple))
                                  if np.any(~np.isnan(bripple)) else float('nan')),
            'dominant_line_hz': dom_hz,
            'dominant_line_Nm': dom_amp,
            'dominant_line_snr': (float(dom_amp / max(float(floor[dom_i]), 1e-12))
                                  if floor is not None else None),
            'stft_window_s': win_n / fs,
            'freq_resolution_hz': float(fr[1]),
            'fmax_hz': fmax,
            # Per-direction fits kept separate: agreement between them is the
            # check that the numbers above are friction and not a cell offset.
            'viscous_pos_Nm_per_rad_s': fit['pos'][0] if 'pos' in fit else None,
            'intercept_pos_Nm': fit['pos'][1] if 'pos' in fit else None,
            'viscous_neg_Nm_per_rad_s': fit['neg'][0] if 'neg' in fit else None,
            'intercept_neg_Nm': fit['neg'][1] if 'neg' in fit else None,
            'top_orders': top_orders,
        })
        res.summary = (
            f'{motor} {"back-drives" if backdrive else "drives"} to '
            f'{vpeak:.2f} rad/s: Coulomb {coulomb:.3f} Nm, viscous {viscous:.4f} '
            f'Nm/(rad/s), {drag_at_peak:.3f} Nm total at peak speed. Dominant '
            f'ripple line {dom_hz:.1f} Hz at {dom_amp:.3f} Nm.'
        )

        # -- figures and tables ---------------------------------------------
        if params['fig_drag']:
            res.figures.append(('drag_vs_velocity', self._fig_drag(
                centers, bmean, bripple, fit, coulomb, viscous, motor, tch,
                backdrive, ratio, invert, cell_bias)))

        if params['fig_order_spectrum'] and ospec is not None:
            band = keep & (fr >= params['fmin_hz'])
            res.figures.append(('order_spectrum', self._fig_orders(
                ospec, peaks, ratio, backdrive, tch,
                float(np.mean(floor[band])) if floor is not None else None)))

        if not params['db_floor']:
            # Bottom of the colour scale from the map's ACTUAL dynamic range: a
            # fixed -40 dB either crushes a quiet map or drowns a loud one.
            db = _db(np.maximum(frames[moving][:, keep], 1e-15),
                     float(np.nanmax(frames[moving][:, keep])))
            derived_floor = float(np.clip(np.nanpercentile(db, 5), -80.0, -20.0))
            params = dict(params, db_floor=derived_floor)
            res.add('info', 'derived_db_floor',
                    f'db_floor derived as {derived_floor:.0f} dB (5th '
                    f'percentile of the moving-record map, clipped to '
                    f'[-80, -20]). Set db_floor explicitly to override.')

        res.figures.extend(self._spectral_figures(
            fr, keep, frames, f_time, f_speed, by_speed, s_edges,
            s_centers, floor, mean_spec, ratio, backdrive, tch, params))

        if params['fig_campbell']:
            res.figures.append(('campbell', self._fig_campbell(
                fr, keep, by_speed, s_centers, s_edges,
                _parse_orders(params['orders']), ratio, backdrive, tch,
                params)))

        res.tables.append(('drag_curve', self._csv_drag(centers, bmean, bripple, bcount)))
        if ospec is not None:
            res.tables.append(('order_spectrum', self._csv_orders(ospec)))
        res.tables.append(('spectrum', self._csv_spectrum(fr, keep, mean_spec, floor)))
        res.tables.append(('spectrum_vs_speed',
                           self._csv_vs_speed(fr, keep, s_centers, by_speed, s_count)))
        return res

    # -- spectral assembly ---------------------------------------------------

    def _spectra(self, segs, tch, vch, vcmd_ch, tare, fs, win_n, hop, params):
        """STFT every span separately, then pool the frames.

        Per span, not on the concatenation: spans are not contiguous in time,
        and a window straddling a join would smear a discontinuity across the
        whole band. Each frame carries its own centre time and mean |speed|,
        which is all the later views need.
        """
        use_cmd = str(params['speed_source']).lower().startswith('c')
        fr = None
        all_frames, all_speed, all_time, static = [], [], [], []
        need_static = int(params['min_static_frames'])

        for i, s in enumerate(segs):
            sig = np.nan_to_num(s[tch], nan=0.0) - tare
            v = np.nan_to_num(s[vch], nan=0.0)
            vsp = (np.nan_to_num(s[vcmd_ch], nan=0.0)
                   if use_cmd and s.has(vcmd_ch) else v)
            t = s['time']
            if len(sig) <= win_n:
                continue
            fr, frames, starts = stft(sig, fs, win_n, hop)
            speed = np.array([np.mean(np.abs(vsp[k:k + win_n])) for k in starts])
            times = np.array([t[k + win_n // 2] for k in starts])
            all_frames.append(frames)
            all_speed.append(speed)
            all_time.append(times)
            # Only whole windows inside the lead-in are ambient; a window
            # overlapping the first motion would fold real ripple into the floor
            # and quietly raise it.
            if i == 0:
                n_lead = self._lead_in(s, vch, vcmd_ch)
                static.append(frames[(starts + win_n) <= n_lead])

        frames = np.concatenate(all_frames)
        static = np.concatenate(static) if static else np.empty((0, frames.shape[1]))
        floor = static.mean(axis=0) if len(static) >= need_static else None
        return {
            'fr': fr,
            'frames': frames,
            'speed': np.concatenate(all_speed),
            'time': np.concatenate(all_time) - all_time[0][0],
            'floor': floor,
            'n_static': len(static),
        }

    def _order_spectrum(self, fr, frames, speed, vpeak, fmax, params):
        """Resample every frame onto a shaft-order axis, then average.

        An order sits at a fixed multiple of shaft rate, so it lands in the same
        order bin at every speed and survives the average. A structural
        resonance sits at a fixed *frequency*, so it lands in a different order
        bin at every speed and smears into the background. This is the exact
        complement of the speed-folded spectrogram: the map shows which ridge is
        which, this one ranks the orders so they can be quoted as numbers.

        Only the fast end of the sweep is used. Order resolution is
        df * 2pi / w -- at a tenth of peak speed the bins are ten times coarser,
        and averaging those in would blur every peak this figure exists to find.
        """
        v_min = max(float(params['order_min_speed_frac']) * vpeak, 1e-6)
        use = speed >= v_min
        if np.count_nonzero(use) < 3:
            return None

        # The axis stops where the *fastest* frame's Nyquist-limited band ends,
        # so no frame is ever extrapolated beyond what it measured.
        o_max = fmax / (vpeak / _TWO_PI)
        o_step = float(fr[1]) * _TWO_PI / v_min          # the coarsest frame's bin
        grid = np.arange(0.0, o_max, o_step)
        if len(grid) < 8:
            return None

        acc = np.zeros(len(grid))
        for amp, w in zip(frames[use], speed[use]):
            acc += np.interp(grid, fr / (w / _TWO_PI), amp)
        return {'orders': grid, 'amp': acc / np.count_nonzero(use),
                'n_frames': int(np.count_nonzero(use)), 'min_speed': v_min,
                'resolution': o_step}

    @staticmethod
    def _split_on_grid(top_orders, base, step, offset=0.0, tol_frac=0.03):
        """Partition detected peaks into those landing on `base` * (n + offset).

        `offset=0` tests integer harmonics, `offset=0.5` half-integers. Tolerance
        grows with order because an order estimate inherits the speed estimate's
        error, so the 9th harmonic may miss by nine times what the 1st may.
        Returns (on_grid_amplitude, off_grid_amplitude).
        """
        on = off = 0.0
        for o in top_orders:
            n = round(o['order'] / base - offset)
            target = (n + offset) * base
            if (n >= 1 and target > 0
                    and abs(o['order'] - target) <= max(tol_frac * o['order'],
                                                        1.5 * step)):
                on += o['amplitude_Nm']
            else:
                off += o['amplitude_Nm']
        return on, off

    def _check_ratio(self, res, spec, ratio, backdrive, top_orders):
        """Which harmonics of the far shaft carry energy, and is the ratio right?

        The tallest peak is very often the SECOND harmonic -- a two-lobed
        eccentric, a misaligned coupling, an oval bearing race and a cycloidal
        drive's opposed eccentrics all put their energy at 2x rev, leaving 1x
        near the noise. Reading that peak as the fundamental halves the apparent
        ratio, which is a mistake the figures alone invite.

        Odd harmonics settle it. If order 3x, 5x, 7x of the configured ratio
        carry energy, then 2*ratio cannot be the shaft rate -- those peaks would
        sit at 1.5x, 2.5x, 3.5x of it, and a single rotating shaft has no
        half-integer orders. If only even harmonics show up, the two hypotheses
        are genuinely indistinguishable from this test and that is said plainly.
        """
        if ratio == 1:
            return
        orders, amp = spec['orders'], spec['amp']
        step = float(orders[1] - orders[0])
        background = float(np.median(amp))
        n_max = int(orders[-1] // ratio)
        if n_max < 3:
            res.add('info', 'ratio_unchecked',
                    f'The order axis only reaches {orders[-1]:.0f}, under three '
                    f'harmonics of the {ratio:g}:1 ratio, so the ratio could not be '
                    f'cross-checked. Sweep faster or raise fmax_hz.')
            return

        present, table = [], []
        for n in range(1, n_max + 1):
            i = int(round(n * ratio / step))
            if i >= len(amp):
                break
            # Take the strongest bin within one bin of the target: the order
            # estimate carries the speed estimate's error, which grows with n.
            lo, hi = max(0, i - 1), min(len(amp), i + 2)
            a = float(np.max(amp[lo:hi]))
            table.append((n, a))
            if a > 2.0 * background:
                present.append(n)

        if not present:
            res.add('warn', 'ratio_no_harmonics',
                    f'No harmonic of the configured {ratio:g}:1 ratio carries more '
                    f'than 2x the background of the order spectrum. Either the '
                    f'ratio is wrong or the ripple is not coming from the far '
                    f'shaft. Compare the peaks in the RIPPLE ORDERS table against '
                    f'candidate ratios before trusting any per-shaft number here.')
            return

        odd = [n for n in present if n % 2]
        shown = ', '.join(f'{n}x={a:.3f}' for n, a in table if a > 2.0 * background)
        weak_first = table[0][1] <= 2.0 * background

        if not odd:
            res.add('warn', 'ratio_ambiguous',
                    f'Only even harmonics of {ratio:g}:1 carry energy ({shown} Nm). '
                    f'That is equally consistent with {2 * ratio:g}:1, where the same '
                    f'peaks would be 1x, 2x, 3x. This test cannot separate the two; '
                    f'confirm the ratio from the gearbox itself.')

        # Coverage. The harmonic-presence test above only rules out doubling; it
        # says nothing about peaks the ratio leaves stranded, and a ratio that
        # explains the loudest peak while orphaning the rest still reads as a
        # pass. The one competing hypothesis worth pricing is halving, and it is
        # tested directly -- do the orphans sit at HALF-INTEGER multiples of the
        # configured ratio? -- rather than by searching submultiples. A search
        # always finds one: a grid k times denser has k times as many slots, so
        # resonance smear, which lands nowhere in particular, scores well on it.
        if not top_orders:
            return
        on, off = self._split_on_grid(top_orders, ratio, step)
        total = on + off
        if total <= 0:
            return
        res.metrics['ratio_order_coverage'] = on / total

        half_on, _ = self._split_on_grid(top_orders, ratio, step, offset=0.5)
        if off > 0 and half_on > 0.6 * off and half_on > 0.25 * total:
            res.add('warn', 'ratio_may_be_halved',
                    f'{ratio:g}:1 leaves {100 * off / total:.0f}% of the detected '
                    f'ripple-order amplitude unexplained, and '
                    f'{100 * half_on / off:.0f}% of that lands on HALF-integer '
                    f'multiples of {ratio:g}. A single rotating shaft has no '
                    f'half-integer orders, so the far shaft is probably turning at '
                    f'{ratio / 2:g}x the driven shaft, not {ratio:g}x -- with its 1x '
                    f'weak and 2x dominating, which is what made {ratio:g} look '
                    f'right. Try gear_ratio={ratio / 2:g}.')
        elif on / total < 0.6:
            res.add('warn', 'ratio_poor_coverage',
                    f'{ratio:g}:1 accounts for only {100 * on / total:.0f}% of the '
                    f'detected ripple-order amplitude, and the remainder is not at '
                    f'half-integers either, so halving would not help. Read the '
                    f'orders off the RIPPLE ORDERS table directly rather than '
                    f'trusting the per-shaft column.')

    def _order_peaks(self, orders, amp, n, min_order=1.0):
        """Tallest local maxima, one per resolution-limited neighbourhood.

        The local-maximum test is what stops the tall peaks being reported
        twice: a strong order's own skirt is several bins wide and its shoulder
        outranks every genuine smaller peak in the record.
        """
        step = float(orders[1] - orders[0])
        sep = max(int(round(3.0 / step)), 3)      # never resolve below the bin width
        is_max = np.zeros(len(amp), dtype=bool)
        is_max[1:-1] = (amp[1:-1] >= amp[:-2]) & (amp[1:-1] > amp[2:])

        picked = []
        for i in np.argsort(amp)[::-1]:
            if orders[i] < min_order or not is_max[i]:
                continue
            if any(abs(i - j) < sep for j in picked):
                continue
            picked.append(int(i))
            if len(picked) >= n:
                break
        return sorted(picked, key=lambda i: -amp[i])

    def _fig_orders(self, spec, peaks, ratio, backdrive, tch, floor_amp):
        import matplotlib.pyplot as plt

        orders, amp = spec['orders'], spec['amp']
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.semilogy(orders, amp + 1e-12, color='tab:blue', lw=1.0)
        if floor_amp is not None:
            ax.axhline(floor_amp, color='0.5', ls='--', lw=1.0,
                       label=f'ambient floor, band mean ({floor_amp:.4f} Nm)')
        other = self._order_label(ratio, backdrive).split()[0]
        for rank, i in enumerate(peaks):
            ax.plot(orders[i], amp[i], 'v', color='tab:red', ms=6)
            label = f'{orders[i]:.1f}'
            if ratio != 1:
                label += f'\n({orders[i] / ratio:.2f}x {other})'
            # Alternate the offset: neighbouring peaks are common here and
            # overlapping labels are the one thing that makes this figure useless.
            ax.annotate(label, (orders[i], amp[i]), textcoords='offset points',
                        xytext=(0, 10 if rank % 2 == 0 else -26), ha='center',
                        fontsize=7, color='tab:red')
        ax.set_xlim(0, float(orders[-1]))
        ax.set_xlabel('Order (multiples of driven-shaft rate)')
        ax.set_ylabel(f'{tch} amplitude (Nm)')
        ax.set_title(
            f'Order spectrum -- {spec["n_frames"]} frames above '
            f'{spec["min_speed"]:.1f} rad/s, resolution {spec["resolution"]:.2f} order. '
            f'A peak here is an order; a resonance smears out.')
        ax.grid(True, which='both', alpha=0.3)
        if ratio != 1:
            top = ax.secondary_xaxis(
                'top', functions=(lambda o: o / ratio, lambda o: o * ratio))
            top.set_xlabel(f'Order of the {self._order_label(ratio, backdrive)} '
                           f'({ratio:g}:1)')
        if floor_amp is not None:
            ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    def _classify_dominant(self, res, fr, frames, speed, moving, lo, dom_hz):
        """Resonance or order? Decided by whether the ridge frequency tracks speed.

        Split the moving frames into a slow and a fast half and find each half's
        strongest line. A resonance sits still; an order moves in proportion to
        speed. Reported rather than acted on -- the two coexist, and the figures
        are where the reader confirms it.
        """
        sp = speed[moving]
        if sp.size < 8:
            return
        cut = float(np.median(sp))
        slow = frames[moving][sp <= cut]
        fast = frames[moving][sp > cut]
        if len(slow) < 2 or len(fast) < 2:
            return
        f_slow = float(fr[lo + int(np.argmax(slow.mean(axis=0)[lo:]))])
        f_fast = float(fr[lo + int(np.argmax(fast.mean(axis=0)[lo:]))])
        v_slow = float(np.mean(sp[sp <= cut]))
        v_fast = float(np.mean(sp[sp > cut]))
        if f_slow <= 0 or v_slow <= 0:
            return
        # An order would scale by v_fast/v_slow; a resonance by 1.
        expected = v_fast / v_slow
        got = f_fast / f_slow
        if abs(got - 1.0) < 0.15 * abs(expected - 1.0) + 0.05:
            res.add('info', 'dominant_is_resonance',
                    f'The strongest line barely moves with speed ({f_slow:.1f} Hz '
                    f'at {v_slow:.1f} rad/s vs {f_fast:.1f} Hz at {v_fast:.1f} rad/s, '
                    f'a {got:.2f}x shift where an order would give {expected:.2f}x). '
                    f'That is a fixed structural resonance being excited, not a '
                    f'gearbox order. Its amplitude vs speed therefore says where '
                    f'an order crosses it, not how bad the mesh is.')
        elif abs(got - expected) < 0.15 * expected:
            res.add('info', 'dominant_is_order',
                    f'The strongest line tracks speed ({f_slow:.1f} Hz at '
                    f'{v_slow:.1f} rad/s -> {f_fast:.1f} Hz at {v_fast:.1f} rad/s, '
                    f'{got:.2f}x against the {expected:.2f}x speed change), so it '
                    f'is a shaft order at roughly '
                    f'{f_fast / (v_fast / _TWO_PI):.1f}x the driven shaft.')
        else:
            res.add('info', 'dominant_ambiguous',
                    f'The strongest line moves {got:.2f}x between the slow and fast '
                    f'halves of the sweep where a pure order would move {expected:.2f}x '
                    f'and a pure resonance 1.00x ({dom_hz:.1f} Hz overall). Most '
                    f'likely an order sweeping through a resonance; read it off the '
                    f'speed-folded figures rather than from this number.')

    # -- figures -------------------------------------------------------------

    def _order_label(self, ratio, backdrive):
        # Far-shaft label; the `attached` text from the bench's ports block
        # rides along so two reports from different mechanical configurations
        # (bare vs gearbox bolted up) stay distinguishable at a glance.
        side = 'input' if backdrive else 'output'
        attached = getattr(self, '_attached', {}).get(side)
        return f'{side} shaft ({attached})' if attached else f'{side} shaft'

    def _draw_orders(self, ax, orders, ratio, backdrive, x_plot, speed, fmax,
                     fast_shaft=True):
        """Overlay f = n*|w|/2pi against whatever the x-axis is.

        `x_plot` is the axis quantity (time, or |w| itself) and `speed` is the
        |w| at each of those x values, so the same code draws straight rays on
        the speed-folded map and speed-profile-shaped curves on the time map.

        Fast-shaft orders are drawn too when there is a reduction: mesh and
        cogging live there, and at a ratio like 26:1 they are the only orders
        that reach the frequencies where structural resonances sit. At a large
        ratio they also crowd the map, so `overlay_fast_shaft` drops them and
        leaves the driven-shaft rays -- the underlying map is unchanged, and
        the reported orders in results.json still cover both shafts.
        """
        x_plot = np.asarray(x_plot, dtype=float)
        speed = np.asarray(speed, dtype=float)
        order = np.argsort(x_plot)
        families = [(1.0, 'driven shaft', 'cyan', '-')]
        if fast_shaft and ratio and abs(ratio - 1.0) > 1e-9:
            families.append((ratio, self._order_label(ratio, backdrive), 'lime', '--'))

        for mult, label, color, style in families:
            shown = False
            for n in orders:
                f = n * mult * speed / _TWO_PI
                # An order that never rises off the bottom of the axis is
                # clutter, not information.
                if f.size == 0 or np.nanmax(f) < 0.02 * fmax:
                    continue
                f = np.where(f <= fmax, f, np.nan)
                lbl = None if shown else (
                    f'{label} orders ({", ".join(f"{o:g}x" for o in orders)})')
                ax.plot(x_plot[order], f[order], style, color=color, lw=1.0,
                        alpha=0.7, label=lbl)
                shown = True

    @staticmethod
    def _draw_resonances(ax, overlays, fmax):
        """Known structural resonances (from multisine_frf) as horizontal
        dashed lines on a frequency-axis map."""
        shown = False
        for f in overlays:
            if 0 < f <= fmax:
                ax.axhline(f, color='w', ls=':', lw=1.0, alpha=0.8,
                           label=None if shown else 'known resonances')
                shown = True

    @staticmethod
    def _order_legend(ax):
        """Legend for the order rays, drawn only when any ray survived.

        Every family can be culled -- switched off, or too low to clear the
        clutter threshold -- and an unconditional legend() warns and stamps an
        empty box on the map in that case.
        """
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc='upper right', fontsize=8, facecolor='0.25',
                      labelcolor='w', framealpha=0.6)

    @staticmethod
    def _map_db(z, floor_ref, db_floor):
        """Map amplitudes to dB plus colour limits, per variant.

        Absolute: dB re the map's own peak, top of the scale at 0 -- the
        conventional reading, and the only one available without a lead-in.

        Floor-referenced: dB re ambient, so the bottom of the colour scale is
        pinned at 0 dB = the noise floor. Everything the rig does at rest goes
        black and only what rotation added has colour, which is the whole point
        of recording the lead-in. Using `db_floor` here instead would spend most
        of the scale below ambient showing nothing.
        """
        z = np.ma.masked_invalid(z)
        peak = float(np.nanmax(z))
        if not floor_ref:
            return _db(z, peak), db_floor, 0.0
        db = _db(z, 1.0)
        return db, 0.0, max(float(np.nanmax(db)), 6.0)

    def _spectral_figures(self, fr, keep, frames, f_time, f_speed,
                          by_speed, s_edges, s_centers, floor, mean_spec,
                          ratio, backdrive, tch, params):
        import matplotlib.pyplot as plt

        figs = []
        orders = _parse_orders(params['orders'])
        overlays = _parse_orders(params['resonance_overlay_hz'])
        fmax = float(fr[keep][-1])
        db_floor = float(params['db_floor'])

        # Every spectral figure is built twice: raw Nm, then divided by the
        # ambient floor per frequency bin. Same data, same code path; only the
        # normalization and the colour limits differ.
        variants = []
        if params['fig_absolute']:
            variants.append((False, '', 'Amplitude (dB re map peak)',
                             f'{tch} amplitude (Nm)', ''))
        if params['fig_floor_ref'] and floor is not None:
            variants.append((True, '_vs_floor', 'Amplitude (dB re ambient floor)',
                             f'{tch} amplitude / ambient floor',
                             ' (referenced to the ambient floor)'))

        for floor_ref, suffix, cbar_label, spec_ylabel, title_tag in variants:
            norm = np.maximum(floor, 1e-12) if floor_ref else np.ones(len(fr))
            panel_spec = mean_spec / norm
            panel_floor = (np.ones(len(fr)) if floor_ref else floor)

            # -- spectrogram vs time -------------------------------------
            if params['fig_spectrogram'] and params['fig_vs_time']:
                fig, ax = self._spec_canvas(plt, fr, keep, panel_spec, panel_floor,
                                            spec_ylabel, params)
                db, vmin, vmax = self._map_db(frames[:, keep] / norm[keep],
                                              floor_ref, db_floor)
                pcm = ax.pcolormesh(f_time, fr[keep], db.T, cmap='magma',
                                    vmin=vmin, vmax=vmax, shading='nearest')
                self._draw_orders(ax, orders, ratio, backdrive, f_time, f_speed,
                                  fmax, fast_shaft=params['overlay_fast_shaft'])
                self._draw_resonances(ax, overlays, fmax)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_ylim(0, fmax)
                ax.set_title(f'{tch} spectrogram vs time{title_tag}')
                self._order_legend(ax)
                fig.colorbar(pcm, ax=ax, label=cbar_label)
                fig.tight_layout()
                figs.append((f'spectrogram_time{suffix}', fig))

            # -- spectrogram vs |speed| ----------------------------------
            if params['fig_spectrogram'] and params['fig_vs_speed']:
                fig, ax = self._spec_canvas(plt, fr, keep, panel_spec, panel_floor,
                                            spec_ylabel, params)
                db, vmin, vmax = self._map_db(by_speed[:, keep] / norm[keep],
                                              floor_ref, db_floor)
                f_edges = np.append(fr[keep], fr[keep][-1] + float(fr[1]))
                pcm = ax.pcolormesh(s_edges, f_edges, db.T, cmap='magma',
                                    vmin=vmin, vmax=vmax)
                self._draw_orders(ax, orders, ratio, backdrive, s_centers,
                                  s_centers, fmax,
                                  fast_shaft=params['overlay_fast_shaft'])
                self._draw_resonances(ax, overlays, fmax)
                ax.set_xlabel('|Velocity| of the driven shaft (rad/s)')
                ax.set_ylabel('Frequency (Hz)')
                ax.set_xlim(s_edges[0], s_edges[-1])
                ax.set_ylim(0, fmax)
                ax.set_title(f'{tch} spectrogram folded on |speed|{title_tag}')
                self._order_legend(ax)
                fig.colorbar(pcm, ax=ax, label=cbar_label)
                fig.tight_layout()
                figs.append((f'spectrogram_speed{suffix}', fig))

            # -- waterfalls ------------------------------------------------
            if params['fig_waterfall'] and params['fig_vs_time']:
                groups, labels = self._group_by(f_time, frames, params['waterfall_n'])
                figs.append((f'waterfall_time{suffix}', self._fig_waterfall(
                    plt, fr, keep, groups, labels, norm, floor_ref,
                    f'{tch} spectra vs time{title_tag}')))
            if params['fig_waterfall'] and params['fig_vs_speed']:
                sel = ~np.isnan(by_speed[:, 0])
                groups, labels = self._pick(by_speed[sel], s_centers[sel],
                                            params['waterfall_n'], 'rad/s')
                figs.append((f'waterfall_speed{suffix}', self._fig_waterfall(
                    plt, fr, keep, groups, labels, norm, floor_ref,
                    f'{tch} spectra vs |speed|{title_tag}')))
        return figs

    def _fig_campbell(self, fr, keep, by_speed, s_centers, s_edges, orders,
                      ratio, backdrive, tch, params):
        """Campbell/order map: the speed-folded spectrogram with the frequency
        axis regridded onto shaft order PER SPEED ROW, keeping the speed axis
        instead of averaging it away (this is `by_speed` + the order spectrum's
        regrid, combined). Orders become horizontal lines; fixed structural
        resonances become hyperbolas order = 2*pi*f / |w| -- the exact
        complement of the frequency-axis map, and the view in which an order
        crossing a resonance is unmistakable.
        """
        import matplotlib.pyplot as plt

        fmax = float(fr[keep][-1])
        overlays = _parse_orders(params['resonance_overlay_hz'])
        db_floor = float(params['db_floor']) or -40.0

        valid = ~np.isnan(by_speed[:, 0]) & (s_centers > 1e-6)
        fig, ax = plt.subplots(figsize=(12, 8))
        if np.count_nonzero(valid) < 2:
            ax.text(0.5, 0.5, 'not enough speed bins', ha='center',
                    transform=ax.transAxes)
            return fig

        # Order resolution collapses at low speed (df*2pi/w); the axis top is
        # set where the fastest row's band ends so nothing is extrapolated.
        vpeak = float(s_centers[valid][-1])
        o_max = fmax * _TWO_PI / vpeak
        o_step = float(fr[1]) * _TWO_PI / vpeak
        grid = np.arange(0.0, o_max, o_step)
        z = np.full((np.count_nonzero(valid), len(grid)), np.nan)
        for i, (row, w) in enumerate(zip(by_speed[valid], s_centers[valid])):
            o_row = fr * _TWO_PI / w
            z[i] = np.interp(grid, o_row[keep], row[keep],
                             left=np.nan, right=np.nan)

        db, vmin, vmax = self._map_db(z, False, db_floor)
        pcm = ax.pcolormesh(s_centers[valid], grid, db.T, cmap='magma',
                            vmin=vmin, vmax=vmax, shading='nearest')
        for n in orders:
            ax.axhline(n, color='cyan', lw=1.0, alpha=0.7)
            ax.text(vpeak, n, f' {n:g}x', color='cyan', fontsize=7, va='bottom')
        shown = False
        for f_res in overlays:
            hyp = np.where(s_centers[valid] > 0,
                           f_res * _TWO_PI / s_centers[valid], np.nan)
            ax.plot(s_centers[valid], np.where(hyp <= grid[-1], hyp, np.nan),
                    'w:', lw=1.2, alpha=0.85,
                    label=None if shown else 'known resonances')
            shown = True
        ax.set_ylim(0, min(grid[-1], 4.0 * max(orders or [1.0]) * max(ratio, 1)))
        ax.set_xlabel('|Velocity| of the driven shaft (rad/s)')
        ax.set_ylabel('Order (multiples of driven-shaft rate)')
        ax.set_title(f'{tch} Campbell map -- orders are horizontal, fixed '
                     f'resonances are hyperbolas'
                     + (f'; far shaft: {self._order_label(ratio, backdrive)}'
                        if ratio != 1 else ''))
        if shown:
            ax.legend(loc='upper right', fontsize=8, facecolor='0.25',
                      labelcolor='w', framealpha=0.6)
        fig.colorbar(pcm, ax=ax, label='Amplitude (dB re map peak)')
        fig.tight_layout()
        return fig

    def _spec_canvas(self, plt, fr, keep, mean_spec, floor, ylabel, params):
        """One figure: optional record-averaged spectrum on top, map below."""
        if not params['mean_spectrum_panel']:
            fig, ax = plt.subplots(figsize=(12, 6))
            return fig, ax
        fig, (axT, axB) = plt.subplots(
            2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})
        axT.semilogy(fr[keep], np.asarray(mean_spec)[keep] + 1e-12,
                     color='tab:blue', lw=1.0, label='moving-record average')
        if floor is not None:
            axT.semilogy(fr[keep], np.asarray(floor)[keep] + 1e-12, color='0.5',
                         lw=1.0, ls='--', label='ambient floor (stationary lead-in)')
        axT.set_xlim(0, float(fr[keep][-1]))
        axT.set_xlabel('Frequency (Hz)')
        axT.set_ylabel(ylabel)
        axT.grid(True, which='both', alpha=0.35)
        axT.legend(loc='upper right', fontsize=8)
        return fig, axB

    def _group_by(self, x, frames, n):
        """Average frames into `n` consecutive equal-width slices of x."""
        n = max(2, int(n))
        edges = np.linspace(float(np.min(x)), float(np.max(x)) + 1e-9, n + 1)
        out, labels = [], []
        for i in range(n):
            sel = (x >= edges[i]) & (x < edges[i + 1])
            if sel.any():
                out.append(frames[sel].mean(axis=0))
                labels.append(f'{0.5 * (edges[i] + edges[i + 1]):.1f} s')
        return np.array(out), labels

    def _pick(self, rows, values, n, unit):
        """Evenly spaced subset of already-binned rows."""
        n = max(2, int(n))
        if len(rows) <= n:
            take = np.arange(len(rows))
        else:
            take = np.unique(np.linspace(0, len(rows) - 1, n).round().astype(int))
        return rows[take], [f'{values[i]:.2f} {unit}' for i in take]

    def _fig_waterfall(self, plt, fr, keep, groups, labels, norm, floor_ref, title):
        """Stacked line spectra, one trace per time slice or speed bin.

        The spectrogram answers 'where is the energy'; this answers 'how much'.
        A colour map cannot be read back to a number, and the two ridges that
        matter here -- one flat, one climbing -- are easier to follow as lines
        when they sit only a few dB apart.
        """
        fig, ax = plt.subplots(figsize=(12, 9))
        if len(groups) == 0:
            ax.text(0.5, 0.5, 'no data', ha='center', transform=ax.transAxes)
            return fig

        zdb = _db(np.asarray(groups)[:, keep] / norm[keep], 1.0)
        span = float(np.nanpercentile(zdb, 99) - np.nanpercentile(zdb, 5))
        step = max(span / 3.0, 6.0)             # overlap a little; pure ridges read better
        cmap = plt.get_cmap('viridis')
        for i, (row, lbl) in enumerate(zip(zdb, labels)):
            color = cmap(i / max(len(zdb) - 1, 1))
            base = i * step
            ax.plot(fr[keep], row + base, lw=0.9, color=color)
            if floor_ref:
                # With 0 dB = ambient, each trace's own baseline is meaningful:
                # anything above the grey line is above the noise floor.
                ax.axhline(base, color='0.85', lw=0.5, zorder=0)
            ax.text(float(fr[keep][-1]), base, f'  {lbl}', fontsize=7,
                    va='center', color=color)
        ax.set_xlim(0, float(fr[keep][-1]))
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(('dB re ambient floor' if floor_ref else 'dB re 1 Nm')
                      + f', traces offset {step:.0f} dB')
        ax.set_title(title)
        ax.grid(True, axis='x', alpha=0.3)
        fig.tight_layout()
        return fig

    def _fig_drag(self, centers, bmean, bripple, fit, coulomb, viscous, motor,
                  tch, backdrive, ratio, invert=False, cell_bias=0.0):
        import matplotlib.pyplot as plt

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 7))
        axL.plot(centers, bmean, 'b.-', ms=5, label='mean torque per velocity bin')
        for tag, color in (('pos', 'tab:red'), ('neg', 'tab:green')):
            if tag in fit:
                lim = np.nanmax(centers) if tag == 'pos' else np.nanmin(centers)
                xs = np.linspace(0, lim, 50)
                axL.plot(xs, fit[tag][0] * xs + fit[tag][1], '--', color=color, lw=1.5,
                         label=f'{tag}: {fit[tag][0]:+.4f}*w {fit[tag][1]:+.3f}')
        axL.plot([], [], ' ',
                 label=f'Coulomb {coulomb:.3f} Nm, viscous {viscous:.4f} Nm/(rad/s)')
        axL.axhline(0, color='k', lw=0.5)
        axL.axvline(0, color='k', lw=0.5)
        axL.set_xlabel('Driven-shaft velocity (rad/s)')
        axL.set_ylabel(f'{tch} (Nm, tared{", sign inverted" if invert else ""}'
                       f'{", bias removed" if cell_bias else ""})')
        axL.set_title(f'{motor} {"back-drive" if backdrive else "forward"} running '
                      f'torque' + (f' ({ratio:g}:1)' if ratio != 1 else ''))
        axL.grid(True, alpha=0.35)
        axL.legend(fontsize=8)

        # Folded: the two directions on top of each other. Agreement is the
        # check that this is friction and not a cell offset masquerading as it.
        #
        # With symmetric_drag set that check has been spent -- the bias was
        # chosen to make the flanks agree at w=0 -- so the uncorrected pair is
        # drawn behind it. The gap between ghost and solid IS the correction,
        # and what remains between the two solid curves is the part symmetry
        # could not explain: a slope difference, which no constant offset can
        # produce. That residual is the only thing left on this panel worth
        # reading as physics once the origin has been centred.
        pos = centers >= 0
        neg = centers < 0
        if cell_bias:
            raw = bmean + cell_bias
            axR.plot(centers[pos], np.abs(raw[pos]), ':', lw=1.0, color='tab:red',
                     alpha=0.55,
                     label=f'before removing {cell_bias:+.4f} Nm cell bias')
            axR.plot(-centers[neg], np.abs(raw[neg]), ':', lw=1.0, color='tab:green',
                     alpha=0.55)
        axR.plot(centers[pos], np.abs(bmean[pos]), 'o-', ms=4, color='tab:red',
                 label='positive direction')
        axR.plot(-centers[neg], np.abs(bmean[neg]), 'o-', ms=4, color='tab:green',
                 label='negative direction')
        axR.fill_between(centers[pos], np.abs(bmean[pos]) - bripple[pos],
                         np.abs(bmean[pos]) + bripple[pos], color='tab:red', alpha=0.15,
                         label='+/- ripple RMS')
        axR.fill_between(-centers[neg], np.abs(bmean[neg]) - bripple[neg],
                         np.abs(bmean[neg]) + bripple[neg], color='tab:green', alpha=0.15)
        axR.set_xlabel('|Driven-shaft velocity| (rad/s)')
        axR.set_ylabel('|Running torque| (Nm)')
        axR.set_title('Both directions folded, with the ripple band'
                      + ('\nsymmetric_drag: origin centred between the flanks, so '
                         'only a slope difference is real' if cell_bias else ''))
        axR.grid(True, alpha=0.35)
        axR.legend(fontsize=8)
        fig.tight_layout()
        return fig

    # -- tables --------------------------------------------------------------

    def _csv_drag(self, centers, bmean, bripple, bcount):
        rows = ['velocity_rad_s,mean_torque_Nm,ripple_rms_Nm,n_samples']
        for c, m, r, n in zip(centers, bmean, bripple, bcount):
            rows.append(f'{c:.4f},{"" if np.isnan(m) else f"{m:.6f}"},'
                        f'{"" if np.isnan(r) else f"{r:.6f}"},{n}')
        return '\n'.join(rows) + '\n'

    def _csv_orders(self, spec):
        rows = ['order,amplitude_Nm']
        for o, a in zip(spec['orders'], spec['amp']):
            rows.append(f'{o:.4f},{a:.8f}')
        return '\n'.join(rows) + '\n'

    def _csv_spectrum(self, fr, keep, mean_spec, floor):
        has = floor is not None
        rows = ['frequency_hz,moving_amplitude_Nm'
                + (',ambient_floor_Nm,ratio' if has else '')]
        for i in np.flatnonzero(keep):
            line = f'{fr[i]:.4f},{mean_spec[i]:.8f}'
            if has:
                line += f',{floor[i]:.8f},{mean_spec[i] / max(floor[i], 1e-12):.4f}'
            rows.append(line)
        return '\n'.join(rows) + '\n'

    def _csv_vs_speed(self, fr, keep, s_centers, by_speed, s_count):
        idx = np.flatnonzero(keep)
        rows = ['speed_rad_s,n_frames,' + ','.join(f'{fr[i]:.2f}Hz' for i in idx)]
        for c, n, row in zip(s_centers, s_count, by_speed):
            if not n:
                continue
            rows.append(f'{c:.4f},{n},'
                        + ','.join('' if np.isnan(row[i]) else f'{row[i]:.8f}'
                                   for i in idx))
        return '\n'.join(rows) + '\n'
