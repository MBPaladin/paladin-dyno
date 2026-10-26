"""Skip forward/backdrive efficiency maps from the 2026-10-22 efficiency run,
drawn by the rig's own efficiency processor so they look exactly like every
other gearbox map.

The run: DUT (Elmo) in velocity mode at 1.3-104 rad/s motor (0.05-4 rad/s
output), both directions; LOAD (AKD) in torque mode at +-2 ... +-20 Nm; short
holds (>= 3 motor revs or ~6 s); DUT i_cont raised to 50 A.

src/analysis/processors/efficiency.py wants an input torque cell. Skip has
none, so the input torque is the motor torque kt x I_q. Everything else goes
through the processor's own code: dwells are built in its record format and
handed to Efficiency._derive (powers, mode from the power signs, efficiency,
loss torque), _mirror_zero (the output cell zero) and _plot_maps /
_plot_curves / _plot_loss (the figures). So the definitions are the
processor's own:

    forward   (P_in > 0, P_out > 0):  eta = P_out / P_in
    backdrive (P_in < 0, P_out < 0):  eta = P_in / P_out

kt. It is not measured on Skip, and it decides which way the forward/backdrive
comparison goes: eta_b / eta_f scales as kt^2, while eta_f x eta_b does not
depend on it. Every cell must satisfy eta <= 1, which brackets kt (0.024-0.039
Nm/A on the 2026-10-22 run). KT_NM_PER_A = None estimates kt from this run's
own data under FRICTION_MODEL. At each speed and |T|, forward and backdrive need
I_f and I_b, and their ratio measures p = eta_f x eta_b = (I_b/T_b)/(I_f/T_f).
A closure then splits p into eta_f and eta_b:

    'worm'       sliding contact with one friction angle phi and lead angle
                 lambda (worm, lead screw); small-angle form of
                 eta_f = tan(l)/tan(l+phi), eta_b = tan(l-phi)/tan(l):
                 eta_b = 2 - 1/eta_f  ->  eta_f = (1 + p)/2,
                 kt = T_f / (26 I_f eta_f)
    'symmetric'  eta_f = eta_b = sqrt(p),  kt = T / (26 sqrt(I_f I_b))

kt is the median over loads >= KT_FIT_MIN_T, then used everywhere, so the input
side stays kt x I as for a cell. The forward map scales as 1/kt and the
backdrive map as kt; the third figure,
sqrt(eta_f x eta_b) = sqrt(I_b / I_f), needs no kt at all and is the check on
whatever kt is used. Replace KT_NM_PER_A with a back-EMF value when there is one.

Dwell means are taken over a whole number of motor revs after SETTLE_S, so the
+-1-2 A once-per-motor-rev current ripple does not bias them.

    .venv/bin/python dyno/src/post_processors/skip/s13_efficiency_map.py

Outputs (results/efficiency/):
  Skip_efficiency_map.png             forward | backdrive, the processor's map
  Skip_efficiency_vs_load.png         the processor's efficiency-vs-load curves
  Skip_loss_torque.png                the processor's loss-torque view
  Skip_efficiency_round_trip_map.png  sqrt(eta_f x eta_b), kt-free, same style
  Skip_efficiency_table.csv
"""

import json
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import patheffects  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from analysis.processors.efficiency import Efficiency, _linfit  # noqa: E402

import skip_common as c  # noqa: E402

LOG = os.path.join(c.SKIP_LOGS, 'efficiency', 'efficiency', 'skip_efficiency.hdf5')
OUT_DIR = os.path.join(c.SKIP_LOGS, 'results', 'efficiency')
KT_NM_PER_A = None         # None = estimate from this run under FRICTION_MODEL
FRICTION_MODEL = 'worm'    # 'worm': sliding contact, eta_b = 2 - 1/eta_f; 'symmetric': eta_f = eta_b
TITLE = 'Skip Actuator Mechanical Efficiency (Air Gap to Output) - Sliding Contact Friction Model'
KT_FIT_MIN_T = 8.0         # Nm; kt estimate uses loads at or above this
SETTLE_S = 0.5             # trimmed off the front of each hold
MIN_HOLD_S = 3.0
GRID_DECIMALS = 3
LOSS_FIT_MIN_LOAD_FRAC = 0.1    # processor defaults
HEADLINE_MIN_LOAD_FRAC = 0.5
_CH = ('time', 'load_torque_command', 'dut_velocity_command', 'dut_current', 'load_torque',
       'dut_output_position', 'load_position', 'pt100_rtd_top')


def build_dwells(path):
    """Holds as records in Efficiency._dwells' format, with t_in_raw = motor
    current (A) for now; it becomes kt x I once kt is chosen."""
    with h5py.File(path, 'r') as f:
        d = {k: f[k][:].astype(np.float64) for k in _CH}
        tare = json.loads(f.attrs['tare'])['load_torque']['bias'] if 'tare' in f.attrs else 0.0
    t = d['time'] - d['time'][0]
    tc, vc = d['load_torque_command'], d['dut_velocity_command']
    key = np.round(np.nan_to_num(tc), 3) * 1e4 + np.round(np.nan_to_num(vc), 3)
    edges = np.flatnonzero(np.diff(key) != 0) + 1
    out, idle = [], None
    for s, e in zip(np.r_[0, edges], np.r_[edges, len(t)]):
        if t[e - 1] - t[s] < MIN_HOLD_S:
            continue
        if not np.isfinite(vc[s]) or vc[s] == 0:
            if idle is None:                              # opening idle: an independent zero check
                idle = float(np.mean(d['load_torque'][s + 2000:e - 500]))
            continue
        i0 = s + int(SETTLE_S * c.FS_HZ)
        th = d['dut_output_position']
        # last sample that closes a whole number of motor revs from i0
        revs = np.abs(th[i0:e] - th[i0]) / c.TWO_PI
        n_rev = int(np.floor(revs[-1]))
        i1 = i0 + int(np.searchsorted(revs, n_rev)) if n_rev >= 1 else e
        sl = slice(i0, i1)
        dur = (i1 - i0) / c.FS_HZ
        # The DUT logs velocity flipped against current (flip_direction_sign):
        # -I is the current in the frame where it times the motor speed is the
        # power the motor puts in.
        out.append({
            'raw_id': len(out), 'setpoint': (float(vc[s]), float(tc[s])), 'n': i1 - i0, 'duration': dur,
            'v_cmd': round(float(vc[s]), GRID_DECIMALS), 't_cmd': round(float(tc[s]), GRID_DECIMALS),
            'w_in': (th[i1 - 1] - th[i0]) / dur,
            'w_out': (d['load_position'][i1 - 1] - d['load_position'][i0]) / dur,
            'i_in': float(-np.mean(d['dut_current'][sl])),
            't_out_raw': float(np.mean(d['load_torque'][sl])),
            't_out_sd': float(np.std(d['load_torque'][sl])),
            'i_sd': float(np.std(d['dut_current'][sl])),
            'i_peak': float(np.max(np.abs(d['dut_current'][sl]))),
            'motor_revs': n_rev, 'temp_C': float(np.nanmean(d['pt100_rtd_top'][sl])),
        })
    for r in out:
        r['rev_out'] = abs(r['w_out']) * r['duration'] / c.TWO_PI
    return out, idle, tare, (float(np.nanmin(d['pt100_rtd_top'])), float(np.nanmax(d['pt100_rtd_top'])))


def kt_estimates(dwells, z_out):
    """Per (|v|, |T|) cell, both rotations pooled: kt under each friction
    closure, and the measured round trip p = eta_f x eta_b."""
    cells = {}
    for r in dwells:
        T = (r['t_out_raw'] - z_out) * np.sign(r['w_out'])   # + = output delivers power (forward)
        p_in = r['i_in'] * np.sign(r['w_in'])                # + = motor puts power in
        # only holds whose two power signs agree with a real quadrant; at light
        # backdrive load the drag can win and the DUT still pushes (the
        # processor calls those 'dissipative' and greys them out)
        if T > 0 and p_in > 0:
            q = 'f'
        elif T < 0 and p_in < 0:
            q = 'b'
        else:
            continue
        cells.setdefault((abs(r['v_cmd']), abs(r['t_cmd'])), {}).setdefault(q, []).append((abs(T), abs(r['i_in'])))
    kts, rt = {'symmetric': {}, 'worm': {}}, {}
    for (v, tq), qd in cells.items():
        if 'f' in qd and 'b' in qd and tq > 0:
            Tf, If = np.mean(qd['f'], axis=0)
            Tb, Ib = np.mean(qd['b'], axis=0)
            # torque-normalised so a small mismatch in the achieved |T| does not leak in
            p = (Ib / Tb) / (If / Tf)
            rt[(v, tq)] = p
            kts['symmetric'][(v, tq)] = 0.5 * (Tf + Tb) / (c.GEAR_RATIO * np.sqrt(If * Ib))
            kts['worm'][(v, tq)] = Tf / (c.GEAR_RATIO * If * 0.5 * (1.0 + p))
    return kts, rt


def plain_labels(ax):
    """Cell labels in black with a thin white outline, whatever the cell colour
    (the processor picks black or white per cell from its luminance)."""
    for t in ax.texts:
        t.set_color('black')
        t.set_path_effects([patheffects.withStroke(linewidth=1.5, foreground='white')])


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches='tight')
    print(f'  wrote {os.path.relpath(os.path.join(OUT_DIR, name), c.DYNO_DIR)}')


def run():
    import matplotlib.pyplot as plt
    proc = Efficiency()
    dwells, idle, tare, temps = build_dwells(LOG)
    ratio = float(c.GEAR_RATIO)

    # Output cell zero exactly as the processor does it: mirror pairs (+v,+T) / (-v,-T).
    mz = proc._mirror_zero(dwells, 't_out_raw', GRID_DECIMALS)
    z_out = mz['zero']

    kts, rt = kt_estimates(dwells, z_out)
    kt_med = {m: float(np.median([k for (v, tq), k in kk.items() if tq >= KT_FIT_MIN_T])) for m, kk in kts.items()}
    kt = KT_NM_PER_A or kt_med[FRICTION_MODEL]
    for r in dwells:
        r['t_in_raw'] = kt * r['i_in']

    live, dropped = proc._derive(dwells, ratio, 0.0, z_out, min_speed=0.01)
    model_txt = {'worm': 'sliding contact, eta_b = 2 - 1/eta_f',
                 'symmetric': 'symmetric, eta_f = eta_b'}[FRICTION_MODEL]
    kt_note = (f'kt = {kt:.4f} Nm/A from this run under the {model_txt} closure' if KT_NM_PER_A is None
               else f'kt = {kt:.4f} Nm/A, set by hand')
    labels = {'title': TITLE}

    fig_map = proc._plot_maps(live, ratio, labels)
    fig_map.suptitle(TITLE)
    for ax, name in zip(fig_map.axes[:2], ('Forward Driving', 'Back Driving')):
        ax.set_title(name)
        plain_labels(ax)
    fig_map.text(0.01, -0.01, f'Input power = kt x DUT current x motor speed; {kt_note}. Rotations averaged; '
                              'grey = dissipative (drag exceeds the backdriving torque).',
                 ha='left', va='top', fontsize=8.5, color='#52514e')
    figs = {'Skip_efficiency_map.png': fig_map,
            'Skip_efficiency_vs_load.png': proc._plot_curves(live, ratio, labels)}
    peak_load = max(abs(r['t_cmd']) for r in live)
    fits = {}
    for mode in ('forward', 'backdrive'):
        sel = [r for r in live if r['mode'] == mode and r['load_Nm'] >= LOSS_FIT_MIN_LOAD_FRAC * peak_load]
        s_, i_, r2 = _linfit([r['load_Nm'] for r in sel], [r['loss_Nm'] for r in sel])
        fits[mode] = {'slope': s_, 'intercept': i_, 'r2': r2, 'n': len(sel)}
    figs['Skip_loss_torque.png'] = proc._plot_loss(live, fits, ratio, labels)

    # kt-free round trip, drawn with the processor's own heatmap and scale rule
    speeds = sorted({v for v, _ in rt})
    loads = sorted({tq for _, tq in rt})
    grid = np.full((len(speeds), len(loads)), np.nan)
    for (v, tq), val in rt.items():
        grid[speeds.index(v), loads.index(tq)] = 100 * np.sqrt(max(val, 0.0))
    fin = grid[np.isfinite(grid)]
    vmin, vmax = max(0.0, np.floor(fin.min() / 5) * 5), max(100.0, np.ceil(fin.max()))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = proc._heatmap(ax, speeds, loads, grid, ratio, 'viridis', vmin, vmax)
    plain_labels(ax)
    fig.colorbar(im, ax=ax, label='Efficiency (%)')
    ax.set_title('sqrt(eta_f x eta_b) = sqrt(I_backdrive / I_forward): needs no kt')
    ax.set_xlabel('|Output torque| setpoint (Nm)')
    ax.set_ylabel('|Output speed| (rad/s)')
    fig.suptitle(f'Skip actuator -- symmetric efficiency at {ratio:g}:1 (rotations averaged)')
    fig.tight_layout()
    figs['Skip_efficiency_round_trip_map.png'] = fig

    for name, fig in figs.items():
        save(fig, name)
        plt.close(fig)

    # table, in the processor's column layout plus the current columns
    cells = {}
    for r in live:
        cells.setdefault((r['v_cmd'], r['t_cmd']), []).append(r)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'Skip_efficiency_table.csv'), 'w') as fh:
        fh.write('input_speed_cmd_rad_s,output_torque_cmd_Nm,output_speed_rad_s,motor_current_A,input_torque_Nm,'
                 'output_torque_Nm,input_power_W,output_power_W,mode,efficiency,loss_torque_Nm,loss_power_W,'
                 'motor_revs,current_peak_A,rtd_top_C\n')
        for (v, tq), ds in sorted(cells.items()):
            m = lambda k: float(np.mean([x[k] for x in ds]))
            fh.write(f'{v:g},{tq:g},{m("w_out"):.5f},{m("i_in"):.3f},{m("t_in"):.5f},{m("t_out"):.4f},'
                     f'{m("p_in"):.3f},{m("p_out"):.3f},{ds[0]["mode"]},{m("eff"):.5f},{m("loss_Nm"):.5f},'
                     f'{m("loss_W"):.3f},{m("motor_revs"):.0f},{m("i_peak"):.2f},{m("temp_C"):.2f}\n')
    print(f'  wrote {os.path.relpath(os.path.join(OUT_DIR, "Skip_efficiency_table.csv"), c.DYNO_DIR)}')

    def pooled(mode):
        return proc._pool([r for r in live if r['mode'] == mode],
                          lambda r: (abs(r['v_cmd']) / ratio, abs(r['t_cmd'])), lambda r: r['eff'])
    c.RESULTS_DIR, c.KEY_NUMBERS_PATH = OUT_DIR, os.path.join(OUT_DIR, 'skip_key_numbers.json')
    head = {}
    for mode in ('forward', 'backdrive'):
        sel = [r['eff'] for r in live if r['mode'] == mode and abs(r['t_cmd']) >= HEADLINE_MIN_LOAD_FRAC * peak_load]
        head[mode] = {'mean': float(np.mean(sel)), 'min': float(np.min(sel)), 'max': float(np.max(sel)), 'n': len(sel)}
    c.record('efficiency_map', {
        'kt_used': kt, 'friction_model': FRICTION_MODEL if KT_NM_PER_A is None else 'kt set by hand',
        'kt_estimates': kt_med,
        'kt_by_cell_range': {m: [float(min(kk.values())), float(max(kk.values()))] for m, kk in kts.items()},
        'z_out_mirror_Nm': z_out, 'z_out_mirror_pairs': mz['n_pairs'], 'z_out_mirror_drift_Nm': mz['drift'],
        'idle_reading_Nm': idle, 'tare_bias_Nm': tare, 'rtd_top_C': list(temps),
        'modes': {m: sum(1 for r in live if r['mode'] == m) for m in ('forward', 'backdrive', 'dissipative', 'non_physical')},
        'headline_above_half_load': head,
        'loss_fits': fits,
        'map_forward': {f'{v:g}|{tq:g}': e for (v, tq), e in pooled('forward').items()},
        'map_backdrive': {f'{v:g}|{tq:g}': e for (v, tq), e in pooled('backdrive').items()},
        'round_trip_sqrt': {f'{v / ratio:g}|{tq:g}': float(np.sqrt(val)) for (v, tq), val in rt.items()},
        'max_current_peak_A': max(r['i_peak'] for r in live),
    })
    return live


if __name__ == '__main__':
    run()
