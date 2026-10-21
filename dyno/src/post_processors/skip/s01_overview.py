"""Test overview: what the 85-minute grid did, and a per-dwell table.

Outputs (results/01_overview/):
  Skip_test_overview.png  speed command, DUT current, cell torque, motor speed
                          and temperature against time, dwells shaded by
                          power-flow quadrant
  Skip_dwell_table.csv    one row per settled dwell
"""

import numpy as np
import matplotlib.pyplot as plt

import skip_common as c

SUB = '01_overview'


def _block_stats(x, n):
    m = len(x) // n
    b = x[:m * n].reshape(m, n)
    return b.mean(1), b.min(1), b.max(1)


def run(d):
    c.apply_style()
    t_min = d['t'] / 60.0
    n = 1000  # 1 s blocks
    tb, _, _ = _block_stats(t_min, n)

    fig, ax = plt.subplots(5, 1, figsize=(13, 11), sharex=True,
                           gridspec_kw={'height_ratios': [1, 1.1, 1.6, 1.4, 0.7]})
    for a in ax:
        for w in d['dwells']:
            fill = c.QUADRANT_FILL[w['quadrant']]
            if fill:
                a.axvspan(w['t0'] / 60, w['t1'] / 60, color=fill, lw=0, zorder=0)

    ax[0].plot(t_min[::200], d['load_velocity_command'][::200], color=c.INK, lw=1.4)
    ax[0].set_ylabel('Output speed\ncommand (rad/s)')

    m, lo, hi = _block_stats(d['dut_current'], n)
    ax[1].plot(tb, m, color=c.INK, lw=1.2)
    ax[1].set_ylabel('DUT current (A)')
    ax[1].set_yticks([-15, -10, -5, 0, 5, 10, 15])

    m, lo, hi = _block_stats(d['T_cell'], n)
    ax[2].fill_between(tb, lo, hi, color='#b7d3f6', lw=0, label='1 s min..max')
    ax[2].plot(tb, m, color='#1c5cab', lw=1.2, label='1 s mean')
    ax[2].set_ylabel('Load cell torque (Nm)')
    ax[2].legend(loc='upper right', ncol=2)

    ratio = d['w_m'] / (c.GEAR_RATIO * np.where(np.abs(d['load_velocity_command']) > 0,
                                                d['load_velocity_command'], np.nan))
    m, lo, hi = _block_stats(np.nan_to_num(ratio, nan=1.0), n)
    ax[3].fill_between(tb, np.clip(lo, -1, 8), np.clip(hi, -1, 8), color='#f6c9b5', lw=0,
                       label='1 s min..max')
    ax[3].plot(tb, m, color='#c14d1f', lw=1.0, label='1 s mean')
    ax[3].axhline(0, color=c.INK_3, lw=0.8)
    ax[3].axhline(1, color=c.INK_3, lw=0.8, ls=':')
    ax[3].set_ylim(-1, 8)
    ax[3].set_ylabel('Motor speed /\ncommanded (-)')
    ax[3].legend(loc='upper right', ncol=2)

    ax[4].plot(t_min[::1000], d['pt100_rtd_top'][::1000], color=c.INK_2, lw=1.2)
    ax[4].set_ylabel('RTD top (degC)')
    ax[4].set_xlabel('Time (min)')

    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(color=c.QUADRANT_FILL['forward'], label='DUT drives load (forward)'),
                          Patch(color=c.QUADRANT_FILL['backdrive'], label='Load backdrives DUT'),
                          Patch(facecolor=c.SURFACE, edgecolor=c.INK_3, label='Zero current')],
                 loc='upper left', ncol=3)
    fig.suptitle('Skip cogging/backdrive grid: 6 speeds x 7 currents, 120 s per dwell')
    c.note(fig, 'Motor speed is differentiated from the motor encoder (25 ms window) and divided by '
                '26 x the commanded output speed: 1 = smooth tracking, 0 = stuck, spikes = slip bursts.',
           y=0.0)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    c.save(fig, SUB, 'Skip_test_overview.png')

    rows = []
    for w in d['dwells']:
        sl = c.dwell_slice(w)
        wm = d['w_m'][sl] * w['dir'] / w['motor_speed_nom']
        dpm = d['th_m'][w['i1'] - 1] - d['th_m'][w['i0']]
        dpo = d['th_o'][w['i1'] - 1] - d['th_o'][w['i0']]
        rows.append([w['n'], round(w['t0'], 1), round(w['t1'], 1), w['speed'], w['tcmd'],
                     w['current'], w['i_along'], w['quadrant'], w['T_cell'],
                     float(np.mean(d['cell_raw'][sl] + d['tare_bias'])),
                     float(np.std(d['T_cell'][sl])), w['P'], w['T_lm'],
                     float(np.mean(d['load_velocity'][sl])), float(np.mean(d['w_m'][sl])),
                     float(dpm / dpo), float(np.mean(np.abs(wm) < c.STUCK_FRAC)),
                     float(abs(dpo) / c.TWO_PI), float(abs(dpm) / c.TWO_PI), w['temp_C']])
    c.save_csv(SUB, 'Skip_dwell_table.csv',
               ['dwell', 't_start_s', 't_end_s', 'output_speed_cmd_rad_s', 'dut_torque_cmd_Nm',
                'dut_current_A', 'current_along_motion_A', 'quadrant',
                'cell_torque_corrected_Nm', 'cell_torque_as_logged_Nm', 'cell_torque_std_Nm',
                'load_push_along_motion_Nm', 'load_motor_torque_Nm', 'output_speed_rad_s',
                'motor_speed_rad_s', 'motor_over_output_travel', 'stuck_fraction',
                'output_revs', 'motor_revs', 'rtd_top_C'], rows)

    ratios = np.array([r[15] for r in rows])
    c.record('overview', {
        'n_dwells': len(rows),
        'speeds_rad_s': c.speeds(d),
        'gear_ratio_from_travel_mean': float(ratios.mean()),
        'gear_ratio_from_travel_sd': float(ratios.std()),
        'dut_velocity_over_position_derivative': float(np.median(
            [np.mean(d['dut_velocity'][c.dwell_slice(w)]) / np.mean(d['w_m'][c.dwell_slice(w)])
             for w in d['dwells']])),
        'rtd_top_range_C': [float(np.nanmin(d['pt100_rtd_top'])), float(np.nanmax(d['pt100_rtd_top']))],
        'cell_zero_raw_Nm': float(d['cell_zero']),
        'tare_bias_Nm': float(d['tare_bias']),
    })


if __name__ == '__main__':
    run(c.load())
