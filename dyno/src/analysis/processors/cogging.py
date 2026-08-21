"""Cogging / torque-ripple analysis over a (speed x load) gridpoint test.

One motor velocity-controls the shaft through a grid of speed plateaus while
the other holds a grid of torque commands. Each dwell spans several
revolutions, so folding the measured torque over shaft angle averages away
everything that is not angle-locked and leaves the cogging/ripple pattern.

Views produced (each with a CSV):

  1. Cogging vs mechanical angle -- torque folded over one full revolution at
     the slowest speed, one trace per load level. Slowest speed only: the cell
     and its filtering attenuate angle-locked ripple as its temporal frequency
     rises, so the slow dwell is the least-filtered look at the pattern.
  2. Ripple vs ripple-order phase -- the same data folded over one period of
     the dominant angle-locked ripple order (2pi / ripple_order), pooling every
     period. `ripple_order` can be given, or auto-detected from the dominant
     spatial order of the no-load trace. It is a property of the measured
     ripple, not of the motor: the fold period need not equal the pole-pair
     count, and a peak-picking detector cannot tell a fundamental from one of
     its harmonics. Label these views as an order, never as a pole count.
  3. Ripple vs load -- ripple peak-to-peak (from the ripple-order fold) against
     load level, one line per speed.
  4. Torque ratio vs speed -- measured torque / commanded torque per dwell,
     one line per load level: how speed erodes the ability to achieve the
     commanded torque (back-EMF headroom, drag, bandwidth).

Future views worth adding (done previously in
dyno/src/post_processors/mac_motor_post_processing.py, deliberately left out
of this first pass):
  * Campbell diagram: torque-amplitude spectrum vs (frequency, speed) with
    cogging-order rays overlaid -- separates orders from fixed resonances.
  * Per-speed FFT overlay with automatic structural-resonance detection and
    order-crossing speed annotations.
  * Ripple-change-per-Nm shape: each load's fold minus the no-load fold,
    normalized per Nm and spline-fitted, predicting ripple at any load.
  * Speed dependency of the folded shape itself (one fold per speed overlaid).
"""

import numpy as np

from ..processor import Applicability, Processor, Result
from ..registry import register

_TWO_PI = 2.0 * np.pi


def _fold(frac, val, nbins):
    """Bin `val` by `frac` in [0,1); returns (centers, mean, std, count)."""
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(frac, edges) - 1, 0, nbins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(nbins, np.nan)
    std = np.full(nbins, np.nan)
    cnt = np.zeros(nbins, dtype=int)
    for b in range(nbins):
        sel = val[idx == b]
        cnt[b] = sel.size
        if sel.size:
            mean[b] = float(sel.mean())
            std[b] = float(sel.std())
    return centers, mean, std, cnt


def _wide_csv(index_name, index, columns):
    """Wide CSV text: index column + one column per (name, array)."""
    lines = [','.join([index_name] + [c[0] for c in columns])]
    for i in range(len(index)):
        cells = [f'{index[i]:.5f}']
        for _, arr in columns:
            v = arr[i]
            cells.append('' if v is None or (isinstance(v, float) and np.isnan(v))
                         else f'{v:.5f}')
        lines.append(','.join(cells))
    return '\n'.join(lines) + '\n'


@register
class Cogging(Processor):
    name = 'cogging'
    title = 'Cogging / Torque Ripple (speed x load grid)'
    description = (
        'Folds measured torque over shaft angle across a grid of speed '
        'plateaus and load levels: ripple vs mechanical angle, vs phase '
        'within the dominant ripple order, ripple amplitude vs load, and '
        'achieved/commanded torque ratio vs speed.'
    )
    granularity = 'run'

    params = {
        'velocity_motor':  (None, str,   'Motor holding the speed plateaus (auto-detected)'),
        'torque_motor':    (None, str,   'Motor holding the load levels (auto-detected)'),
        'position_channel': (None, str,  'Continuous shaft angle channel (auto-detected)'),
        'torque_channel':  (None, str,   'Torque cell on the torque motor shaft (auto-detected)'),
        'ripple_order':    (0,    int,   'Ripple periods per mechanical revolution to fold '
                                         'over; 0 = auto-detect from the dominant spatial '
                                         'order of the slowest lightest-load trace'),
        'mech_bins':       (360,  int,   'Angle bins over one full revolution'),
        'phase_bins':      (32,   int,   'Angle bins over one ripple period'),
        'min_revs':        (2.0,  float, 'Revolutions a dwell needs for its fold to be used'),
        'ratio_min_cmd':   (0.5,  float, 'Skip |torque command| below this in the '
                                         'torque-ratio view [Nm]'),
        'max_auto_order':  (256,  int,   'Ceiling for the auto-detected ripple_order'),
    }

    # -- detection -----------------------------------------------------------

    def _grid(self, segs, params):
        """Classify each dwell as one (speed, load) cell of the grid.

        Speed comes from the velocity command and load from the torque
        command: the commands are the grid definition, the measurements are
        what is being judged against it.
        """
        seg = segs[0]
        vmotor, tmotor = params.get('velocity_motor'), params.get('torque_motor')
        if not (vmotor and tmotor):
            vmotor = tmotor = None
            for dev in seg.devices():
                p = seg.prefix(dev)
                if seg.is_active(f'{p}_velocity_command'):
                    vmotor = vmotor or dev
                if seg.is_active(f'{p}_torque_command'):
                    tmotor = tmotor or dev
        if not vmotor or not tmotor or vmotor == tmotor:
            return None, vmotor, tmotor

        vcmd = f'{seg.prefix(vmotor)}_velocity_command'
        tcmd = f'{seg.prefix(tmotor)}_torque_command'
        cells = []
        for s in segs:
            if not (s.is_active(vcmd) and s.is_active(tcmd)):
                continue
            cells.append({
                'seg': s,
                'speed': round(float(np.nanmean(s[vcmd])), 3),
                'load': round(float(np.nanmean(s[tcmd])), 3),
            })
        return cells, vmotor, tmotor

    def _position_channel(self, seg, tmotor):
        # The cogging source is the torque motor, so its own encoder is the
        # angle reference. A geared device logs the output side.
        p = seg.prefix(tmotor)
        for ch in (f'{p}_output_position', f'{p}_position'):
            if seg.has(ch) and seg.is_active(ch):
                return ch
        return None

    def _torque_channel(self, seg, tmotor):
        # The bench's ports block names the cell on each shaft; that beats any
        # data-driven guess. Fall back to the only active sensor if absent.
        for port in seg.ports():
            if port.get('device') == tmotor and port.get('cell'):
                cell = str(port['cell'])
                if seg.has(cell) and seg.is_active(cell):
                    return cell
        active = [c for c in seg.channels()
                  if c.endswith('_torque') and seg.is_active(c)]
        return active[0] if len(active) == 1 else None

    # -- applicability --------------------------------------------------------

    def applies_to(self, segs):
        d = self.defaults()
        seg = segs[0]
        cells, vmotor, tmotor = self._grid(segs, d)
        if cells is None:
            return Applicability(
                'no', 'need one motor under velocity command and a different '
                      'one under torque command; found '
                      f'velocity={vmotor!r}, torque={tmotor!r}')
        if not cells:
            return Applicability('no', 'no dwell has both commands active')

        speeds = sorted({c['speed'] for c in cells})
        loads = sorted({c['load'] for c in cells})
        if len(speeds) < 2 and len(loads) < 2:
            return Applicability(
                'no', f'single grid cell (speed {speeds}, load {loads}); '
                      'nothing to compare across')

        pch = self._position_channel(seg, tmotor)
        if pch is None:
            return Applicability('no', f'no position channel found for {tmotor}')
        tch = d['torque_channel'] or self._torque_channel(seg, tmotor)
        if tch is None:
            return Applicability(
                'no', f'no torque cell resolvable for the {tmotor} shaft '
                      '(no ports.cell entry and not exactly one active sensor)')

        # The fold needs whole revolutions somewhere on the grid.
        slow = min(c['speed'] for c in cells if c['speed'] > 0)
        revs = max(abs(float(c['seg'][pch][-1] - c['seg'][pch][0])) / _TWO_PI
                   for c in cells if c['speed'] == slow)
        if revs < d['min_revs']:
            return Applicability(
                'no', f'slowest dwell covers only {revs:.2f} revolutions '
                      f'(need {d["min_revs"]:g}) -- folding cannot average')

        return Applicability(
            'yes',
            f'{vmotor} holds {len(speeds)} speed plateau(s) '
            f'({speeds[0]:g}..{speeds[-1]:g} rad/s) against {len(loads)} '
            f'{tmotor} load level(s) ({loads[0]:g}..{loads[-1]:g} Nm); '
            f'{tch} on the {tmotor} shaft, {revs:.1f} revs per slow dwell',
            {'velocity_motor': vmotor, 'torque_motor': tmotor,
             'position_channel': pch, 'torque_channel': tch})

    # -- ripple-order auto-detection -------------------------------------------

    def _detect_ripple_order(self, pos, tq, max_order):
        """Dominant spatial order of the torque-vs-angle signal.

        Resample torque onto a uniform angle grid (the dwell's speed ripple
        makes the raw samples non-uniform in angle), FFT over angle, and take
        the strongest peak in cycles/rev.

        This is the order to fold over, and nothing more. It is NOT a pole-pair
        count: peak-picking a single bin cannot separate a fundamental from its
        harmonics (a machine with a strong 3rd order reads as 3x its pole
        pairs), cannot separate angle-locked ripple from time-locked content
        that happens to land on an order at this one speed, and reports the
        winner even when the runner-up is a hair behind. Cross-check a detected
        order against a second speed plateau -- angle-locked content holds its
        order, time-locked content does not -- before reading physics into it.
        """
        theta = pos - pos[0]
        span = float(abs(theta[-1]))
        if span < _TWO_PI:
            return 0
        n = len(theta)
        grid = np.linspace(0.0, theta[-1], n)
        # np.interp needs increasing x; a negative-direction dwell is mirrored.
        if theta[-1] < 0:
            theta, grid = -theta, -grid
        uni = np.interp(np.abs(grid), np.abs(theta), tq)
        uni = uni - uni.mean()
        amp = np.abs(np.fft.rfft(uni * np.hanning(n)))
        orders = np.fft.rfftfreq(n, d=span / n) * _TWO_PI   # cycles per rev
        band = (orders >= 2.0) & (orders <= max_order)
        if not band.any():
            return 0
        return int(round(float(orders[band][np.argmax(amp[band])])))

    # -- run --------------------------------------------------------------------

    def run(self, segs, params):
        import matplotlib.pyplot as plt

        res = Result()
        cells, vmotor, tmotor = self._grid(segs, params)
        pch, tch = params['position_channel'], params['torque_channel']
        speeds = sorted({c['speed'] for c in cells})
        loads = sorted({c['load'] for c in cells})
        slow = min(s for s in speeds if s > 0)
        by_cell = {(c['speed'], c['load']): c['seg'] for c in cells}
        cmap = plt.cm.viridis

        def trace(speed, load):
            s = by_cell.get((speed, load))
            if s is None:
                return None, None
            return s[pch], s[tch]

        # -- ripple order ----------------------------------------------------
        order = int(params['ripple_order'])
        order_source = 'specified'
        base_load = min(loads, key=abs)
        if order <= 0:
            order_source = 'detected'
            p0, q0 = trace(slow, base_load)
            order = self._detect_ripple_order(p0, q0, params['max_auto_order'])
            if order <= 0:
                res.add('warn', 'ripple_order_unknown',
                        'ripple_order could not be auto-detected (no dominant '
                        'spatial order in the no-load slow dwell); the '
                        'ripple-order views are skipped. Set ripple_order '
                        'explicitly to get them.')
            else:
                res.add('info', 'ripple_order_detected',
                        f'ripple_order auto-detected as {order}: the dominant '
                        f'spatial order of {tch} vs angle in the {slow:g} '
                        f'rad/s, {base_load:g} Nm dwell. This is a fold period, '
                        f'not a motor property -- it may be a harmonic of the '
                        f'true fundamental, and a single peak pick cannot rule '
                        f'out time-locked content that only looks angle-locked '
                        f'at this speed. Confirm against another speed plateau '
                        f'before quoting it. Set ripple_order to override.')

        res.add('info', 'slow_fold_only',
                f'Angle-folded views use the {slow:g} rad/s plateau only: the '
                f'torque path low-passes angle-locked ripple harder as speed '
                f'(and hence its temporal frequency) rises, so faster dwells '
                f'understate the pattern rather than re-measure it.')

        # -- 1) cogging vs mechanical angle @ slowest speed --------------------
        fig, ax = plt.subplots(figsize=(13, 7))
        mech_cols, mech_pkpk = [], {}
        for i, load in enumerate(loads):
            p, q = trace(slow, load)
            if p is None:
                continue
            c, m, sd, _ = _fold((p % _TWO_PI) / _TWO_PI, q, params['mech_bins'])
            m = m - np.nanmean(m)          # ripple about the load's own mean
            color = cmap(i / max(len(loads) - 1, 1))
            ax.plot(c * 360, m, color=color, lw=1.2, label=f'{load:g} Nm')
            ax.fill_between(c * 360, m - sd, m + sd, color=color, alpha=0.15)
            mech_cols += [(f'{load:g}Nm_mean', m), (f'{load:g}Nm_std', sd)]
            mech_pkpk[load] = float(np.nanmax(m) - np.nanmin(m))
        ax.set_xlabel('Mechanical angle (deg)')
        ax.set_ylabel(f'{tch} ripple (Nm, mean-subtracted)')
        ax.set_title(f'{tmotor} cogging vs mechanical angle @ {slow:g} rad/s')
        ax.axhline(0, color='k', lw=0.5)
        ax.grid(alpha=0.4)
        ax.legend(title='Load')
        fig.tight_layout()
        res.figures.append(('vs_mech_angle', fig))
        res.tables.append(('vs_mech_angle',
                           _wide_csv('angle_deg', c * 360, mech_cols)))

        # -- 2) ripple vs ripple-order phase + 3) ripple vs load ---------------
        # Peak-to-peak per (speed, load) comes from the ripple-order fold, which
        # pools every period and so is the lowest-noise ripple estimate.
        pkpk = {}          # {(speed, load): pkpk}
        if order > 0:
            period = _TWO_PI / order
            # Phase reference: put the no-load slow trace's peak at 0.
            p0, q0 = trace(slow, base_load)
            c0, m0, _, _ = _fold((p0 / period) % 1.0, q0, params['phase_bins'])
            offset = float(c0[np.nanargmax(m0 - np.nanmean(m0))])

            fig, ax = plt.subplots(figsize=(13, 7))
            phase_cols = []
            for i, load in enumerate(loads):
                for speed in speeds:
                    p, q = trace(speed, load)
                    if p is None:
                        continue
                    c, m, sd, _ = _fold(((p / period) - offset) % 1.0, q,
                                        params['phase_bins'])
                    m = m - np.nanmean(m)
                    pkpk[(speed, load)] = float(np.nanmax(m) - np.nanmin(m))
                    if speed != slow:
                        continue
                    color = cmap(i / max(len(loads) - 1, 1))
                    ax.plot(c * 100, m, color=color, lw=1.6, label=f'{load:g} Nm')
                    ax.fill_between(c * 100, m - sd, m + sd, color=color,
                                    alpha=0.15)
                    phase_cols += [(f'{load:g}Nm_mean', m),
                                   (f'{load:g}Nm_std', sd)]
            ax.set_xlabel(f'Phase within one 1/{order} rev period (%)')
            ax.set_ylabel(f'{tch} ripple (Nm, mean-subtracted)')
            ax.set_title(f'{tmotor} torque ripple folded at order {order} '
                         f'({order_source}) @ {slow:g} rad/s')
            ax.axhline(0, color='k', lw=0.5)
            ax.grid(alpha=0.4)
            ax.legend(title='Load')
            fig.tight_layout()
            res.figures.append(('vs_ripple_order', fig))
            res.tables.append(('vs_ripple_order',
                               _wide_csv('phase_percent', c * 100, phase_cols)))

            fig, ax = plt.subplots(figsize=(11, 7))
            load_cols = []
            for si, speed in enumerate(speeds):
                ys = np.array([pkpk.get((speed, ld), np.nan) for ld in loads])
                color = cmap(si / max(len(speeds) - 1, 1))
                ax.plot(loads, ys, 'o-', color=color, lw=1.5,
                        label=f'{speed:g} rad/s')
                load_cols.append((f'{speed:g}rad_s_pkpk_Nm', ys))
            ax.set_xlabel('Load / commanded torque (Nm)')
            ax.set_ylabel(f'Ripple peak-to-peak (Nm, order-{order} fold)')
            ax.set_title(f'{tmotor} ripple amplitude vs load')
            ax.grid(alpha=0.4)
            ax.legend(title='Speed')
            fig.tight_layout()
            res.figures.append(('vs_load', fig))
            res.tables.append(('vs_load',
                               _wide_csv('load_Nm', np.asarray(loads, float),
                                         load_cols)))

        # -- 4) torque ratio vs speed ------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 7))
        ratio_cols, ratios = [], {}
        used_loads = [ld for ld in loads if abs(ld) >= params['ratio_min_cmd']]
        for i, load in enumerate(used_loads):
            ys = []
            for speed in speeds:
                s = by_cell.get((speed, load))
                r = (float(np.nanmean(s[tch])) / load) if s is not None else np.nan
                ys.append(r)
                if s is not None:
                    ratios[(speed, load)] = r
            ys = np.array(ys)
            color = cmap(i / max(len(used_loads) - 1, 1))
            ax.plot(speeds, ys, 'o-', color=color, lw=1.6, label=f'{load:g} Nm')
            ratio_cols.append((f'{load:g}Nm', ys))
        ax.axhline(1.0, color='k', lw=0.8, ls=':')
        ax.set_xlabel('Speed (rad/s)')
        ax.set_ylabel(f'Achieved / commanded torque ({tch} / command)')
        ax.set_title(f'{tmotor}: how speed affects torque tracking')
        ax.grid(alpha=0.4)
        ax.legend(title='Commanded torque')
        fig.tight_layout()
        res.figures.append(('torque_ratio_vs_speed', fig))
        res.tables.append(('torque_ratio_vs_speed',
                           _wide_csv('speed_rad_s', np.asarray(speeds, float),
                                     ratio_cols)))

        if ratios:
            worst = min(ratios, key=lambda k: ratios[k])
            res.add('info', 'torque_tracking',
                    f'Worst torque tracking is {ratios[worst]:.3f} of the '
                    f'command at {worst[0]:g} rad/s, {worst[1]:g} Nm. A ratio '
                    f'falling with speed is back-EMF eating voltage headroom '
                    f'plus speed-dependent drag; a constant offset from 1.0 is '
                    f'a kt or cell-scale error instead.')

        # -- metrics -----------------------------------------------------------
        slow_pkpk = {f'{ld:g}Nm': pkpk.get((slow, ld)) for ld in loads} if order else {}
        res.metrics.update({
            'velocity_motor': vmotor,
            'torque_motor': tmotor,
            'position_channel': pch,
            'torque_channel': tch,
            'ripple_order': order if order > 0 else None,
            'ripple_order_source': order_source if order > 0 else None,
            'speeds_rad_s': speeds,
            'loads_Nm': loads,
            'slowest_speed_rad_s': slow,
            'ripple_pkpk_at_slowest_Nm': slow_pkpk,
            'mech_ripple_pkpk_Nm': {f'{ld:g}Nm': v for ld, v in mech_pkpk.items()},
            'torque_ratio': {f'{sp:g}rad_s_{ld:g}Nm': r
                             for (sp, ld), r in sorted(ratios.items())},
        })
        headline = (f'{np.nanmax(list(slow_pkpk.values())):.3f} Nm pk-pk ripple '
                    f'(folded at order {order})'
                    if slow_pkpk and any(v is not None for v in slow_pkpk.values())
                    else 'ripple-order fold unavailable')
        res.summary = (
            f'{tmotor} over {len(speeds)} speeds x {len(loads)} loads: '
            f'{headline} at {slow:g} rad/s; torque tracking spans '
            + (f'{min(ratios.values()):.3f}..{max(ratios.values()):.3f} '
               f'of command across the grid.' if ratios else 'n/a.'))
        return res
