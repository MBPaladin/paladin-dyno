"""FRF, resonances, and honest distortion bounds from a multisine preamble.

Physics
-------
At standstill, a periodic multisine is injected into ONE shaft's torque
command; the assembly as seen from that shaft responds. The excitation is
periodic (integer number of periods at the logging rate), so a rectangular
window is EXACT -- no leakage, no window correction -- and periods can be
averaged coherently (complex average). That average is the measurement:

  * random noise is incoherent across periods and averages down as 1/sqrt(P);
  * nonlinear distortion is phase-locked to the excitation and does NOT
    average down.

The excited lines sit only on odd FFT bins, with a seeded ~25% of them
silenced as *detection lines*. That taxonomy turns the unexcited bins into
instruments: energy on the silenced odd bins is odd-order distortion (the
signature of friction and backlash), energy on even bins is even-order
distortion, and the period-to-period variance at the excited bins is the true
noise floor of the estimate.

What this actually measures
---------------------------
On a free-far-shaft rig, noise is not the binding constraint -- backlash is,
and with no DC preload available the excitation either traverses the deadband
(rattling between flanks; nonlinear) or stays inside it (linear, but then the
far side is effectively disconnected and the FRF describes the exciter's own
rotor, cell, and fixture). The honest deliverable is therefore: the ambient
floor, the fixture/rotor structural modes, and a QUANTIFIED backlash
nonlinearity -- not a clean gearbox FRF. This is why the distortion bands are
the primary figure, not a quality-control nicety: they say which regime a run
landed in. Run at gentle and at normal and compare -- if resonances move or
the bands jump, the excitation crossed out of the deadband.

Figures
-------
  * bode           |H| over log frequency with phase beneath. Marker opacity is
                   modulated by per-line coherence (SNR of the averaged
                   estimate), so the curve visibly fades where it stops being
                   trustworthy. Three shaded bands under the magnitude trace --
                   even-bin level, odd detection-line level, and the
                   period-to-period noise floor -- are the primary readout on
                   this rig (see above).
  * receptance     excited-port position / torque command. Resonances read
                   cleanest here; gated off by default because the position
                   channel's quantization dominates at high frequency.

Applicability is a predicate over the data (a torque-commanded oscillation
with nothing turning), not a match on the behavior name -- a log with no
preamble simply skips this processor, which is the whole backwards
compatibility story.
"""

import numpy as np

from ..processor import Applicability, Processor, Result
from ..registry import register
from .running_torque import sample_rate

_TWO_PI = 2.0 * np.pi

# Matches test_builder.MULTISINE_PERIOD_S; duplicated (with this note) so the
# analysis package keeps importing nothing from the control side.
PERIOD_S = 2.0
DISCARD_PERIODS = 2


@register
class MultisineFRF(Processor):
    name = 'multisine_frf'
    title = 'Multisine preamble FRF'
    description = (
        'Frequency response, structural resonances, and noise/odd/even '
        'distortion bounds from a standstill multisine torque excitation. '
        'Periods are split on the logged generator sample index, averaged '
        'coherently, and the unexcited bins are read as distortion.')
    granularity = 'run'

    params = {
        # -- what to read (auto-detected; override to force) ----------------
        'role':             (None,  str,  'Excited port role (auto-detected)'),
        'command_channel':  (None,  str,  'Torque command channel (auto-detected)'),
        'response_channel': (None,  str,  'Torque cell channel (auto-detected)'),
        'position_channel': (None,  str,  'Excited-port position channel '
                                          '(auto-detected, receptance only)'),
        # -- analysis --------------------------------------------------------
        'discard_periods':  (DISCARD_PERIODS, int,
                             'Leading periods dropped (ramp-in + settling)'),
        'min_line_snr':     (3.0,   float, 'Coherence gate: lines below this '
                                           'SNR are faded and never called '
                                           'resonances'),
        'resonance_min_db': (3.0,   float, 'Prominence a |H| peak needs over '
                                           'its neighbours to be reported'),
        # -- figures ---------------------------------------------------------
        'fig_bode':         (True,  bool, 'Bode with coherence-faded trace and '
                                          'distortion bands'),
        'fig_receptance':   (False, bool, 'Position/torque FRF (resonances '
                                          'read cleanest here)'),
    }

    # -- channel discovery ---------------------------------------------------

    def _motor_channels(self, seg):
        """[(device, role, torque_cmd_ch, velocity_ch)] for devices that
        expose both, resolved through the bench's ports block."""
        out = []
        for dev in seg.devices():
            prefix = seg.prefix(dev)
            cmd, vel = f'{prefix}_torque_command', f'{prefix}_velocity'
            if seg.has(cmd) and seg.has(vel):
                role = next((p.get('role') for p in seg.ports()
                             if p.get('device') == dev), prefix)
                out.append((dev, role, cmd, vel))
        return out

    def _torque_cells(self, seg):
        # Sensors only: '*_torque' excludes '*_torque_command' by construction.
        return [c for c in seg.channels() if c.endswith('_torque')
                and seg.is_active(c)]

    @staticmethod
    def _ac_rms(x):
        x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0)
        return float(np.std(x - x.mean()))

    def _classify(self, segs):
        """Split a span group into (quiet_segs, [(seg, role, cmd, vel)]).

        An excitation span is one where a torque command is active,
        oscillatory (many sign crossings), and its motor's measured velocity
        stays near zero -- standstill excitation, not a running test. A quiet
        span has no torque or velocity command active at all.
        """
        quiet, excite = [], []
        for seg in segs:
            found = None
            for dev, role, cmd, vel in self._motor_channels(seg):
                if not seg.is_active(cmd):
                    continue
                u = np.nan_to_num(seg[cmd], nan=0.0)
                v = np.nan_to_num(seg[vel], nan=0.0)
                crossings = int(np.count_nonzero(np.diff(np.sign(u)) != 0))
                if (self._ac_rms(u) > 1e-6 and crossings > 20
                        and float(np.max(np.abs(v))) < 1.0):
                    found = (seg, role, cmd, vel)
                    break
            if found:
                excite.append(found)
                continue
            cmds = [c for c in seg.channels()
                    if c.endswith(('_torque_command', '_velocity_command'))]
            if not any(seg.is_active(c)
                       and self._ac_rms(seg[c]) + abs(np.nanmean(seg[c])) > 1e-6
                       for c in cmds):
                quiet.append(seg)
        return quiet, excite

    # -- applicability -------------------------------------------------------

    def applies_to(self, segs):
        quiet, excite = self._classify(segs)
        if not excite:
            return Applicability(
                'no', 'no span holds an oscillatory torque command at '
                      'standstill; this log has no multisine preamble')

        seg, role, cmd, vel = excite[0]
        # The responding cell: the one whose AC content at the excitation grows
        # over its quiet-hold value -- directly analogous to running_torque's
        # "which cell responds to rotation" test. An idle 500 Nm cell is
        # noisier than a driven 20 Nm one, so no absolute threshold would do.
        cells = self._torque_cells(seg)
        if not cells:
            return Applicability('no', 'no torque sensor channel found')
        scored = []
        for ch in cells:
            run_rms = self._ac_rms(seg[ch])
            rest = (max(self._ac_rms(q[ch]) for q in quiet if q.has(ch))
                    if quiet and any(q.has(ch) for q in quiet) else 1e-12)
            scored.append((run_rms / max(rest, 1e-12), run_rms, ch))
        gain, run_rms, cell = max(scored)
        if quiet and gain < 1.5:
            return Applicability(
                'no', f'no torque cell responds to the excitation (best is '
                      f'{cell}, AC RMS only {gain:.2f}x its quiet-hold value); '
                      f'nothing is measuring the excited shaft')

        prefix = cmd[:-len('_torque_command')]
        pos = next((c for c in (f'{prefix}_position',
                                f'{prefix}_output_position') if seg.has(c)),
                   None)
        quiet_txt = (f'{len(quiet)} quiet span(s) for the ambient floor'
                     if quiet else 'no quiet span (floor from period variance only)')
        return Applicability(
            'yes',
            f"role '{role}' is torque-excited at standstill over "
            f'{len(excite)} block(s); {cell} responds at {run_rms:.4f} Nm AC '
            f'RMS'
            + (f' ({gain:.1f}x quiet)' if quiet else '') + f'; {quiet_txt}',
            {'role': role, 'command_channel': cmd, 'response_channel': cell,
             'position_channel': pos})

    # -- run ------------------------------------------------------------------

    @staticmethod
    def _periods(seg, n_period, discard):
        """Whole periods of the span as (n_periods, n_period), split on the
        logged generator sample index when present."""
        idx_ch = 'preamble_sample'
        n = len(seg)
        if seg.has(idx_ch) and not np.all(np.isnan(seg[idx_ch])):
            rel = seg[idx_ch] - seg[idx_ch][0]
            start = int(np.argmax(rel >= discard * n_period))
        else:
            start = discard * n_period
        usable = (n - start) // n_period
        return start, usable

    def run(self, segs, params):
        res = Result()
        quiet, excite = self._classify(segs)
        seg0 = excite[0][0]
        fs = sample_rate(seg0['time'])
        n_period = int(round(PERIOD_S * fs))
        discard = int(params['discard_periods'])
        min_snr = float(params['min_line_snr'])

        if not seg0.has('preamble_sample'):
            res.add('warn', 'no_sample_index',
                    "'preamble_sample' is not logged on this bench, so period "
                    'boundaries were taken from raw sample counts. A single '
                    'dropped frame shifts every later period and decoheres the '
                    'average; add the preamble_sample log key to the bench '
                    'config.')

        attached = seg0.attached_label(params['role'])
        tag = f" [{attached}]" if attached else ''
        all_res = []

        for block_i, (seg, role, cmd_ch, vel_ch) in enumerate(excite):
            cmd_ch = params['command_channel'] if len(excite) == 1 else cmd_ch
            resp_ch = params['response_channel']
            # Use the LOGGED command as the FRF input, not a regenerated
            # waveform: it reflects any clipping that actually occurred.
            u = np.nan_to_num(seg[cmd_ch], nan=0.0)
            y = np.nan_to_num(seg[resp_ch], nan=0.0)

            start, n_avg = self._periods(seg, n_period, discard)
            if n_avg < 3:
                res.add('error', 'too_short',
                        f'excitation block {block_i} holds only {n_avg} whole '
                        f'periods after discarding {discard}; nothing to average')
                continue

            def spectra(x):
                frames = x[start:start + n_avg * n_period].reshape(n_avg, n_period)
                # Rectangular window: exact for a periodic signal. Scaled so a
                # pure line reads its own amplitude in signal units.
                return np.fft.rfft(frames, axis=1) * 2.0 / n_period

            U = spectra(u)
            Y = spectra(y)
            U_avg, Y_avg = U.mean(axis=0), Y.mean(axis=0)
            fr = np.fft.rfftfreq(n_period, 1.0 / fs)

            # Excited bins straight from the data (the stored design in the
            # test yaml agrees, but the data needs no cross-file plumbing).
            mag_u = np.abs(U_avg)
            floor_u = float(np.median(mag_u[1:]))
            excited = np.flatnonzero(mag_u > 50.0 * max(floor_u, 1e-15))
            excited = excited[excited > 0]
            if len(excited) < 5:
                res.add('error', 'no_lines',
                        f'block {block_i}: only {len(excited)} excited lines '
                        f'detected in the command; not a multisine')
                continue

            H = Y_avg[excited] / U_avg[excited]
            # Noise of the averaged estimate: period-to-period scatter / sqrt(P).
            noise = np.std(Y[:, excited], axis=0) / np.sqrt(n_avg)
            snr = np.abs(Y_avg[excited]) / np.maximum(noise, 1e-15)

            # Bin taxonomy for the distortion bands, inside the excited band.
            lo_bin, hi_bin = excited[0], excited[-1]
            band = np.arange(lo_bin, hi_bin + 1)
            is_excited = np.isin(band, excited)
            odd_det = band[(band % 2 == 1) & ~is_excited]
            even = band[band % 2 == 0]
            odd_level = np.abs(Y_avg[odd_det])
            even_level = np.abs(Y_avg[even])

            # Regime call: the whole reason these bands exist on this rig.
            med_line = float(np.median(np.abs(Y_avg[excited])))
            med_odd = float(np.median(odd_level)) if len(odd_det) else 0.0
            med_noise = float(np.median(noise))
            if med_odd > 5.0 * med_noise:
                res.add('info', 'distortion_dominated',
                        f'block {block_i}{tag}: odd-order distortion on the '
                        f'detection lines is {med_odd / max(med_noise, 1e-15):.0f}x '
                        f'the noise floor ({med_odd * 1e3:.2f} mNm vs '
                        f'{med_noise * 1e3:.3f} mNm) -- the excitation is '
                        f'traversing backlash/friction and the FRF is a '
                        f'describing function, not a linear measurement. '
                        f'Compare a gentle-level run: if resonances move, the '
                        f'deadband is being crossed.')
            else:
                res.add('info', 'near_linear',
                        f'block {block_i}{tag}: distortion lines sit near the '
                        f'noise floor, so the response is effectively linear -- '
                        f'which at standstill usually means the excitation '
                        f'stayed inside the backlash deadband and the FRF '
                        f'describes the exciter rotor, cell, and fixture with '
                        f'the far side disconnected.')

            # Resonances: |H| local maxima over the sparse line grid, gated on
            # coherence and prominence.
            hmag = np.abs(H)
            good = snr >= min_snr
            peaks = []
            for i in range(1, len(excited) - 1):
                if not (good[i] and hmag[i] >= hmag[i - 1]
                        and hmag[i] > hmag[i + 1]):
                    continue
                prom_db = 20 * np.log10(
                    hmag[i] / max(min(hmag[i - 1], hmag[i + 1]), 1e-15))
                if prom_db >= float(params['resonance_min_db']):
                    peaks.append(float(fr[excited[i]]))
            all_res.extend(peaks)

            block = {
                'role': role, 'attached': attached or None,
                'n_periods_averaged': n_avg,
                'excited_lines': len(excited),
                'freq_hz': fr[excited].tolist(),
                'noise_floor_Nm_median': med_noise,
                'odd_distortion_Nm_median': med_odd,
                'even_distortion_Nm_median': float(np.median(even_level)),
                'line_level_Nm_median': med_line,
                'resonances_hz': peaks,
            }
            res.metrics.setdefault('blocks', []).append(block)

            if params['fig_bode']:
                res.figures.append((
                    f'bode_{role}',
                    self._fig_bode(fr, excited, H, snr, min_snr, band,
                                   odd_det, odd_level, even, even_level,
                                   noise, np.abs(U_avg), cmd_ch, resp_ch,
                                   role, tag, n_avg)))
            if params['fig_receptance'] and params['position_channel'] \
                    and seg.has(params['position_channel']):
                p = np.nan_to_num(seg[params['position_channel']], nan=0.0)
                P_avg = spectra(p).mean(axis=0)
                res.figures.append((
                    f'receptance_{role}',
                    self._fig_receptance(fr, excited,
                                         P_avg[excited] / U_avg[excited],
                                         snr, min_snr, resp_ch, role, tag)))

            res.tables.append((f'frf_{role}', self._csv(
                fr, excited, H, snr, noise)))

        resonances = sorted(set(round(f, 2) for f in all_res))
        res.metrics['structural_resonances_hz'] = resonances
        res.metrics['role'] = params['role']
        res.summary = (
            f"Multisine FRF{tag}: {len(excite)} excitation block(s), "
            f'resonances at '
            + (', '.join(f'{f:g} Hz' for f in resonances) if resonances
               else 'none above the prominence gate') + '.')
        if resonances:
            res.add('info', 'resonances',
                    'Structural resonances at '
                    + ', '.join(f'{f:g} Hz' for f in resonances)
                    + ' -- these are the horizontal lines worth overlaying on '
                      "running_torque's spectrograms (resonance_overlay_hz).")
        return res

    # -- figures --------------------------------------------------------------

    def _fig_bode(self, fr, excited, H, snr, min_snr, band, odd_det, odd_level,
                  even, even_level, noise, mag_u, cmd_ch, resp_ch, role, tag,
                  n_avg):
        import matplotlib.pyplot as plt

        fig, (axM, axP) = plt.subplots(
            2, 1, figsize=(14, 9), sharex=True,
            gridspec_kw={'height_ratios': [2, 1]})
        f = fr[excited]
        # Opacity carries coherence: the trace fades where the estimate stops
        # being trustworthy instead of silently looking as confident as ever.
        alpha = np.clip(snr / (4.0 * min_snr), 0.15, 1.0)
        axM.plot(f, np.abs(H), color='tab:blue', lw=0.8, alpha=0.35)
        axM.scatter(f, np.abs(H), c=[(0.12, 0.47, 0.71, a) for a in alpha],
                    s=18, zorder=3, label='|H| (opacity = coherence)')

        # Distortion bands in |H| units: divide each bin's response-side level
        # by the excitation magnitude interpolated to that bin, so all four
        # curves share one axis.
        u_at = lambda bins: np.interp(fr[bins], f, mag_u[excited])
        axM.fill_between(fr[even], 1e-15, even_level / u_at(even),
                         color='tab:orange', alpha=0.3, label='even-order distortion')
        if len(odd_det):
            axM.fill_between(fr[odd_det], 1e-15, odd_level / u_at(odd_det),
                             color='tab:red', alpha=0.3,
                             label='odd-order distortion (detection lines)')
        axM.fill_between(f, 1e-15, noise / mag_u[excited], color='0.6',
                         alpha=0.4, label=f'noise floor ({n_avg} periods averaged)')
        axM.set_xscale('log')
        axM.set_yscale('log')
        # The bands fill down from their level; without a floor on the axis
        # the log scale stretches to the fill's 1e-15 base and everything
        # interesting collapses into the top decade.
        y_lo = 0.1 * float(np.min(noise / mag_u[excited]))
        y_hi = 3.0 * float(np.max(np.abs(H)))
        axM.set_ylim(max(y_lo, 1e-9), y_hi)
        axM.set_ylabel(f'|{resp_ch} / {cmd_ch}| (Nm/Nm)')
        axM.set_title(
            f"Multisine FRF, role '{role}'{tag} -- the shaded bands are the "
            f'primary readout: distortion >> noise means backlash is being '
            f'traversed')
        axM.grid(True, which='both', alpha=0.3)
        axM.legend(fontsize=8, loc='lower left')

        phase = np.unwrap(np.angle(H))
        axP.scatter(f, np.degrees(phase),
                    c=[(0.12, 0.47, 0.71, a) for a in alpha], s=14)
        axP.set_xscale('log')
        axP.set_xlabel('Frequency (Hz)')
        axP.set_ylabel('Phase (deg)')
        axP.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        return fig

    def _fig_receptance(self, fr, excited, R, snr, min_snr, resp_ch, role, tag):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 7))
        alpha = np.clip(snr / (4.0 * min_snr), 0.15, 1.0)
        f = fr[excited]
        ax.plot(f, np.abs(R), color='tab:green', lw=0.8, alpha=0.35)
        ax.scatter(f, np.abs(R), c=[(0.17, 0.63, 0.17, a) for a in alpha], s=18)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('|position / torque command| (rad/Nm)')
        ax.set_title(f"Receptance, role '{role}'{tag} -- resonances read "
                     f'cleanest here; ignore the quantization shelf at high '
                     f'frequency')
        ax.grid(True, which='both', alpha=0.3)
        fig.tight_layout()
        return fig

    def _csv(self, fr, excited, H, snr, noise):
        rows = ['frequency_hz,H_real,H_imag,H_mag,H_phase_deg,line_snr,noise_Nm']
        for i, k in enumerate(excited):
            rows.append(f'{fr[k]:.4f},{H[i].real:.8g},{H[i].imag:.8g},'
                        f'{np.abs(H[i]):.8g},{np.degrees(np.angle(H[i])):.3f},'
                        f'{snr[i]:.2f},{noise[i]:.8g}')
        return '\n'.join(rows) + '\n'
