"""Spectra of the load cell torque in time (Hz) and in angle (orders), and how
each order behaves across speed and current.

Time domain separates what is fixed in Hz (rig resonances, control loops,
electrical pickup) from what moves with speed. Angle domain (torque resampled
uniformly in motor or output angle) turns everything locked to the actuator's
geometry into sharp lines at fixed orders, whatever the speed ripple and
stick-slip did to the time axis.

Order bookkeeping (26:1, 21 pole pairs):
  per motor rev:  1 = eccentric/input; 21 = electrical; 42 = 2x electrical;
                  36 = 36 per rev (slot-count candidate); 27 = pin mesh if 27 pins
  per output rev: 26 = once per motor rev; 25 / 27 = cycloid lobe/pin sidebands;
                  1..8 = output side (disc, output bearing, LOAD motor, coupling)

Outputs (results/05_spectra/):
  Skip_spectrum_vs_frequency.png
  Skip_spectrum_vs_motor_order.png
  Skip_spectrum_vs_output_order.png
  Skip_order_amplitude_vs_current.png
  Skip_order_phase_repeatability.png
  Skip_fixed_frequency_lines.csv
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import signal

import skip_common as c
import skip_orders as so

SUB = '05_spectra'
NPERSEG = 16384
MOTOR_MARKS = {1: '1', 2: '2', 6: '6', 21: '21 (1x elec)', 36: '36', 42: '42 (2x elec)',
               63: '63', 84: '84', 126: '126 (6x elec)'}
OUTPUT_STEP = c.TWO_PI / 2048


def welch_asd(x):
    f, p = signal.welch(x - np.mean(x), fs=c.FS_HZ, nperseg=NPERSEG)
    return f, np.sqrt(p)


def fixed_lines(freqs, spectra, fmin=20.0, tol=0.35, n=8):
    """Peaks present at the same frequency at every speed = fixed in Hz."""
    found = []
    for asd in spectra:
        lg = np.log10(asd)
        pk, pr = signal.find_peaks(lg, prominence=0.35)
        found.append([(freqs[i], asd[i]) for i in pk if freqs[i] >= fmin])
    common = []
    for f0, a0 in found[0]:
        others = [min(fl, key=lambda fa: abs(fa[0] - f0)) for fl in found[1:] if fl]
        if len(others) == len(found) - 1 and all(abs(o[0] - f0) < tol for o in others):
            common.append((f0, np.mean([a0] + [o[1] for o in others])))
    common.sort(key=lambda fa: -fa[1])
    merged = []   # a split peak (308.8 / 310.1 Hz) is one line for labelling
    for f0, a0 in common:
        if all(abs(f0 - m[0]) > 3.0 for m in merged):
            merged.append((f0, a0))
    return sorted(merged[:n])


def motor_order_spectrum(d, w):
    rs = so.resample_dwell(d, w)
    f = so.fit_dwell(d, w)
    return c.order_spectrum(rs['y'] - f['slow'], so.STEP_M)


def output_order_series(d, dwells):
    ys = []
    for w in dwells:
        sl = c.dwell_slice(w)
        y, _ = c.angle_resample(d['th_o'][sl], d['T_cell'][sl], OUTPUT_STEP)
        x = np.arange(len(y))
        ys.append(y - np.polyval(np.polyfit(x, y, 1), x))
    return np.concatenate(ys)


def run(d):
    c.apply_style()
    dw = d['dwells']
    zero = [w for w in dw if w['quadrant'] == 'zero']

    # -- A: time-domain spectra ----------------------------------------------------
    asd, asd_lm, asd_in = {}, {}, None
    for v in c.speeds(d):
        sp, sl_, si = [], [], []
        for w in [w for w in zero if w['speed_abs'] == v]:
            s = c.dwell_slice(w)
            f, a = welch_asd(d['T_cell'][s])
            sp.append(a)
            sl_.append(welch_asd(d['T_lm'][s])[1])
            # input cell in load-cell-equivalent Nm: same volts at the ADC
            si.append(welch_asd(d['input_torque'][s] * (10.0 / 20.0) / c.CELL_V_PER_NM)[1])
        asd[v] = np.mean(sp, 0)
        asd_lm[v] = np.mean(sl_, 0)
        asd_in = np.mean(si, 0) if asd_in is None else 0.5 * (asd_in + np.mean(si, 0))
    lines = fixed_lines(f, [asd[v] for v in c.speeds(d)])
    rows = []
    for f0, a0 in lines:
        i = int(np.argmin(np.abs(f - f0)))
        j = slice(max(0, i - 3), i + 4)
        lm = float(np.mean([asd_lm[v][j].max() / np.median(asd_lm[v][max(0, i - 60):i + 60]) for v in asd_lm]))
        ic = float(asd_in[j].max() / np.median(asd_in[max(0, i - 60):i + 60]))
        rows.append([round(f0, 2), a0, lm, ic])
    c.save_csv(SUB, 'Skip_fixed_frequency_lines.csv',
               ['freq_Hz', 'load_cell_asd_Nm_rtHz', 'in_LOAD_current_peak_over_floor',
                'in_input_cell_peak_over_floor'], rows)

    fig, ax = plt.subplots(2, 1, figsize=(14, 9))
    for v in c.speeds(d):
        ax[0].semilogy(f, asd[v], color=c.SPEED_COLORS[v], lw=0.8, label=f'load cell, {v:g} rad/s, 0 A')
        ax[1].semilogy(f, asd[v], color=c.SPEED_COLORS[v], lw=1.0, label=f'{v:g} rad/s')
    ax[0].semilogy(f, asd_in, color=c.INK, lw=0.6, label='input cell (disconnected), same ADC volts')
    for f0, a0 in lines:
        ax[0].annotate(f'{f0:.1f} Hz', xy=(f0, a0), xytext=(f0, a0 * 3.0), fontsize=8, color=c.INK_2,
                       ha='center', arrowprops=dict(arrowstyle='-', color=c.INK_3, lw=0.6))
    ax[0].set_xlim(0, 500)
    ax[0].set_title('0-500 Hz: what is fixed in frequency')
    ax[0].legend(loc='upper right', ncol=2)
    for v in c.speeds(d):
        fm = c.GEAR_RATIO * v / c.TWO_PI
        for k in (1, 6, 21, 36, 42):
            if k * fm < 40:
                ax[1].axvline(k * fm, color=c.SPEED_COLORS[v], lw=0.6, ls=':', alpha=0.9)
                ax[1].text(k * fm, 0.98, str(k), transform=ax[1].get_xaxis_transform(), fontsize=7.5,
                           color=c.SPEED_COLORS[v], ha='center', va='top')
    ax[1].set_xlim(0, 40)
    ax[1].set_title('0-40 Hz: lines that move with speed (dotted = orders 1, 6, 21, 36, 42 per motor rev)')
    ax[1].legend(loc='upper right')
    for a in ax:
        a.set_ylabel('Amplitude density (Nm/sqrt(Hz))')
        a.set_xlabel('Frequency (Hz)')
    fig.suptitle('Skip load cell spectrum, zero-current dwells (both directions averaged)')
    c.note(fig, 'Welch, 16384-sample segments (0.06 Hz). The disconnected input cell is shown at the load cell\'s '
                'V/Nm so the two noise floors compare at the ADC. Fixed lines are listed with their presence '
                'in the LOAD current and input cell in Skip_fixed_frequency_lines.csv.', y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    c.save(fig, SUB, 'Skip_spectrum_vs_frequency.png')

    # -- B: motor-order spectra ----------------------------------------------------
    grid = np.arange(0.0, 135.0, 0.01)
    fig, ax = plt.subplots(3, 1, figsize=(14, 11), sharex=True, sharey=True)
    rows_q = (('backdrive', -15, 'LOAD backdrives DUT, 15 A'), ('zero', 0, 'Zero current'),
              ('forward', 15, 'DUT drives, 15 A'))
    for a, (q, ia, title) in zip(ax, rows_q):
        for v in c.speeds(d):
            specs = []
            for w in [w for w in dw if w['speed_abs'] == v and w['i_along_nom'] == ia]:
                o, A = motor_order_spectrum(d, w)
                specs.append(np.interp(grid, o, A))
            a.semilogy(grid, np.mean(specs, 0), color=c.SPEED_COLORS[v], lw=0.7, label=f'{v:g} rad/s')
        for k, lab in MOTOR_MARKS.items():
            a.axvline(k, color=c.INK_3, lw=0.6, ls=':', zorder=0)
            if a is ax[0]:
                a.text(k, 1.02, lab, transform=a.get_xaxis_transform(), fontsize=8, color=c.INK_2,
                       ha='center', va='bottom', rotation=0)
        a.set_title(title, loc='left')
        a.set_ylabel('Amplitude (Nm)')
        a.set_ylim(1e-4, 3)
    ax[0].legend(loc='upper right', ncol=3)
    ax[-1].set_xlabel('Order (cycles per MOTOR revolution)')
    ax[-1].set_xlim(0, 135)
    fig.suptitle('Skip load cell torque vs motor angle: order spectrum', y=0.995)
    c.note(fig, 'Torque resampled uniformly in motor angle (1024 per rev), slow drift removed, Hann window; '
                'average of the two directions. Sharp lines at the same order at all speeds are locked to the motor '
                'shaft (rotor, eccentric, input-side gearbox parts).', y=0.0)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    c.save(fig, SUB, 'Skip_spectrum_vs_motor_order.png')

    # -- C: output-order spectra -----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={'width_ratios': [1.6, 1]})
    out_rec = {}
    for s in (-1, 1):
        y = output_order_series(d, [w for w in dw if w['speed_abs'] == 0.1 and w['dir'] == s])
        o, A = c.order_spectrum(y, OUTPUT_STEP)
        col = '#2a78d6' if s < 0 else '#eb6834'
        for a in ax:
            a.semilogy(o, A, color=col, lw=0.8, label=f'{c.DIR_LABEL[s]} ({len(y) * OUTPUT_STEP / c.TWO_PI:.0f} output revs)')
        out_rec[f'dir{s:+d}'] = {str(k): float(A[np.abs(o - k) < 0.3].max()) for k in (1, 2, 3, 4, 6, 24, 25, 26, 27, 28, 52, 156)}
    for k in (1, 25, 26, 27, 52):
        for a in ax:
            a.axvline(k, color=c.INK_3, lw=0.6, ls=':', zorder=0)
    for k, lab in ((1, '1'), (26, '26 = once per motor rev'), (52, '52')):
        ax[0].text(k, 1.01, lab, transform=ax[0].get_xaxis_transform(), fontsize=8, color=c.INK_2, ha='center')
    for k in (25, 26, 27):
        ax[1].text(k, 1.01, str(k), transform=ax[1].get_xaxis_transform(), fontsize=8.5, color=c.INK_2, ha='center')
    ax[0].set_xlim(0, 60)
    ax[1].set_xlim(20, 32)
    ax[0].set_title('Orders 0-60 per output rev', pad=16)
    ax[1].set_title('Zoom: the motor-rev line and its sidebands', pad=16)
    for a in ax:
        a.set_xlabel('Order (cycles per OUTPUT revolution)')
        a.set_ylabel('Amplitude (Nm)')
        a.set_ylim(1e-3, 2)
    ax[0].legend(loc='upper right')
    fig.suptitle('Skip load cell torque vs output angle: order spectrum, 0.1 rad/s, all seven currents pooled')
    c.note(fig, 'The lower sideband (25) is 2-3x the upper (27): a feature that turns with the output and is '
                'swept by the eccentric 25 times per output rev, rather than plain amplitude modulation, which '
                'would raise 25 and 27 equally.', y=0.0)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    c.save(fig, SUB, 'Skip_spectrum_vs_output_order.png')

    # -- D: order amplitude vs current ------------------------------------------------
    picks = (('m1', 'Once per motor rev (eccentric)'), ('m2', '2 per motor rev'),
             ('m6', '6 per motor rev'), ('m36', '36 per motor rev'),
             ('e21', '21 per motor rev (1x electrical)'), ('e42', '42 per motor rev (2x electrical)'),
             ('e126', '126 per motor rev (6x electrical)'), ('m27', '27 per motor rev (pin-mesh candidate)'))
    fig, ax = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
    amp_rec = {}
    for a, (key, title) in zip(ax.flat, picks):
        for v in c.speeds(d):
            for s in (-1, 1):
                pts = sorted([w for w in dw if w['speed_abs'] == v and w['dir'] == s], key=lambda w: w['i_along_nom'])
                ys = [so.fit_dwell(d, w)['amps'][key][0] for w in pts]
                a.plot([w['i_along_nom'] for w in pts], ys, color=c.SPEED_COLORS[v], marker=c.DIR_MARKER[s],
                       ls=c.DIR_LS[s], lw=1.1, ms=5)
        a.set_title(title, fontsize=10)
        a.set_ylim(bottom=0)
        a.axvline(0, color=c.INK_3, lw=0.8)
        amp_rec[key] = {str(ia): float(np.mean([so.fit_dwell(d, w)['amps'][key][0] for w in dw if w['i_along_nom'] == ia]))
                        for ia in c.currents_along(d)}
    for a in ax[:, 0]:
        a.set_ylabel('Amplitude (Nm, output)')
    for a in ax[1]:
        a.set_xlabel('DUT current along motion (A)')
        a.set_xticks(c.currents_along(d))
    handles = [Line2D([], [], color=col, lw=2, label=f'{v:g} rad/s') for v, col in c.SPEED_COLORS.items()]
    handles += [Line2D([], [], color=c.INK_2, marker=c.DIR_MARKER[s], ls=c.DIR_LS[s], label=c.DIR_LABEL[s]) for s in (-1, 1)]
    fig.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Skip ripple orders vs current: which parts scale with load, which do not')
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    c.save(fig, SUB, 'Skip_order_amplitude_vs_current.png')

    # -- E: phase repeatability ----------------------------------------------------------
    keys = (('m1', 'Once per motor rev'), ('m6', '6 per motor rev'), ('m36', '36 per motor rev'),
            ('e42', '2x electrical'))
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.8), subplot_kw={'projection': 'polar'})
    for a, (key, title) in zip(ax, keys):
        for w in dw:
            A, ph = so.fit_dwell(d, w)['amps'][key]
            # Torque along the motion: the cell-frame phasor flipped for the - direction.
            if w['dir'] < 0:
                ph += np.pi
            a.plot(ph, A, c.DIR_MARKER[w['dir']], color=c.current_color(w['i_along_nom']), ms=6, alpha=0.9)
        a.set_title(title, pad=14, fontsize=10)
        a.tick_params(labelsize=7.5)
        a.grid(color=c.GRID)
    handles = [Line2D([], [], color=c.current_color(i), marker='o', ls='', label=c.current_label(i))
               for i in c.currents_along(d)]
    handles += [Line2D([], [], color=c.INK_2, marker=c.DIR_MARKER[s], ls='', label=c.DIR_LABEL[s]) for s in (-1, 1)]
    fig.legend(handles=handles, loc='lower center', ncol=9, bbox_to_anchor=(0.5, -0.02), fontsize=8)
    fig.suptitle('Skip ripple phasors per dwell (torque along the motion): tight clusters = repeatable, compensable')
    c.note(fig, 'Radius = amplitude (Nm), angle = phase of A cos(k theta + phase) against the motor encoder. '
                'Friction-like ripple clusters in this frame for both directions; a magnetic (conservative) ripple '
                'splits into two clusters 180 deg apart.', y=-0.06)
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    c.save(fig, SUB, 'Skip_order_phase_repeatability.png')

    c.record('spectra', {
        'fixed_lines_Hz': [[float(r[0]), float(r[1]), float(r[2]), float(r[3])] for r in rows],
        'fixed_lines_columns': ['freq_Hz', 'load_cell_asd', 'LOAD_current_peak_over_floor', 'input_cell_peak_over_floor'],
        'output_order_amplitudes_0p1': out_rec,
        'motor_order_mean_amplitude_by_current': amp_rec,
        'load_cell_floor_Nm_rtHz_100_300Hz': float(np.median(asd[0.1][(f > 100) & (f < 300)])),
        'input_cell_floor_equiv_Nm_rtHz_100_300Hz': float(np.median(asd_in[(f > 100) & (f < 300)])),
    })


if __name__ == '__main__':
    run(c.load())
