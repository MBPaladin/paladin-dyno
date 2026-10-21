"""Shared plumbing for the Skip actuator one-off analysis.

Skip is a client-supplied motor (21 pole pairs) on a custom 26:1 slip-driven
cycloidal gearbox. On the in-house dyno it sits as

    LOAD (AKD, velocity mode) -> load_torque cell -> gearbox output -> gearbox -> DUT motor (Elmo, torque mode)

The input torque cell and SMALL_INHOUSE_MOTOR are out of the loop; the input
cell is kept only as an electrical noise reference.

Facts about this log that every script relies on (each was checked against the
data; the writeup has the numbers):

  * FRAMES. `dut_output_position` and `dut_velocity` are MOTOR side despite the
    name: per dwell, d(dut_output_position)/d(load_position) = 26.00 +/- 0.01.
  * `dut_velocity` reads ~2% high against the derivative of
    `dut_output_position`, and is quantised to one count per sample
    (0.77 rad/s). Motor speed is therefore always differentiated from position.
  * Positions are stored as float32 near 2e4 rad, so both are quantised to
    2^-9 rad (~2 mrad): ~2.5 encoder counts on the motor, 2 mrad on the output.
    Folds and dwell means average this away; single samples do not.
  * SIGN. The DUT logs velocity flipped against torque (flip_direction_sign),
    so in logged signs the DUT drives the load (forward power flow) when
    dut_torque_command and load_velocity_command have OPPOSITE signs.
  * LOAD CELL. The load cell chain has a ~0.4 Nm split at 0 V: raw readings are
    pushed ~0.2 Nm away from zero on either side, leaving a gap in the
    histogram and a sign-dependent offset on every reading. `split_correct`
    undoes it; see s08_load_cell.py for the evidence.

Run any script directly, or everything via run_all.py, with the anaconda base
interpreter:

    "C:/Users/Nathan Justus/anaconda3/python.exe" dyno/src/post_processors/skip/run_all.py
"""

import json
import os
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless: save figures, never block on display
import matplotlib.pyplot as plt

# --- Paths -------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DYNO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
SKIP_LOGS = os.path.join(DYNO_DIR, 'logs', 'skip')
COGGING_LOG = os.path.join(SKIP_LOGS, 'cogging', 'cogging_vs_torque_and_speed',
                           'skip_cogging_vs_torque_long.hdf5')
TORQUE_TEST_LOGS = {
    'torque_test_15': os.path.join(SKIP_LOGS, 'torque_tests', 'torque_test_15',
                                   'skip_torque_holds.hdf5'),
    'torque_test_17': os.path.join(SKIP_LOGS, 'torque_tests', 'torque_test_17',
                                   'skip_torque_holds_samePositionDiffRamps.hdf5'),
}
RESULTS_DIR = os.path.join(SKIP_LOGS, 'results')
KEY_NUMBERS_PATH = os.path.join(RESULTS_DIR, 'skip_key_numbers.json')

# --- Actuator and rig constants -----------------------------------------------
GEAR_RATIO = 26
POLE_PAIRS = 21
FS_HZ = 1000.0
KT_CONFIGURED = 0.02      # Nm/A the DUT drive used to turn torque commands into current
KT_LOAD = 7.79            # Nm/A, LOAD (AKD); torque_test_15 measured 7.75
CELL_V_PER_NM = 5.0 / 500.0

# --- Analysis tunables ---------------------------------------------------------
DWELL_MIN_S = 100.0       # command plateaus shorter than this are transitions
DWELL_SETTLE_S = 3.0      # trimmed off the front of each dwell (command ramp + catch-up)
DWELL_TAIL_S = 0.2        # trimmed off the end
MOTOR_SPEED_WINDOW = 25   # samples; centred position difference for motor speed
CELL_SPLIT_NM = 0.40      # load cell split at 0 V (s08 measures 0.39-0.42)
STUCK_FRAC = 0.2          # |motor speed| below this x nominal counts as stuck
SLIDE_BAND = (0.3, 3.0)   # |motor speed| within this x nominal counts as steady sliding

TWO_PI = 2.0 * np.pi

_CHANNELS = ('time', 'dut_torque_command', 'load_velocity_command', 'dut_current',
             'dut_output_position', 'dut_velocity', 'load_position',
             'load_velocity', 'load_torque', 'load_current', 'input_torque',
             'pt100_rtd_top', 'cycle_time_us')


# --- Load-cell correction -------------------------------------------------------
def split_correct(raw, split=CELL_SPLIT_NM):
    """Undo the load cell's 0 V split: y = x + (g/2) sign(x)  ->  x = y - (g/2) sign(y).

    `raw` must be the cell reading BEFORE the session tare is applied, because
    the split sits at 0 V on the wire, not at the tared zero.
    """
    raw = np.asarray(raw, dtype=float)
    return raw - np.sign(raw) * 0.5 * split


def cell_split_mv(f, name='load_torque'):
    """split_mv the logger already corrected this cell for, or None.

    From 2026-10-21 the in-house rig corrects the split in the sensor scaling
    (devices.ScaledChannels._scale), so `load_torque` in those logs is already
    split-free and must not go through split_correct again.
    """
    try:
        rc = json.loads(f.attrs['resolved_config'])
    except (KeyError, ValueError):
        return None
    for spec in rc.get('devices', {}).values():
        for p in (spec.get('params') or {}).values():
            if isinstance(p, dict) and p.get('name') == name and p.get('split_mv'):
                return float(p['split_mv'])
    return None


def cell_tare_bias(f):
    try:
        return float(json.loads(f.attrs['tare'])['load_torque']['bias'])
    except (KeyError, ValueError):
        return 0.0


# --- Loading --------------------------------------------------------------------
def speed_from_position(theta, window=MOTOR_SPEED_WINDOW, fs=FS_HZ):
    """Centred difference over `window` samples. Plain np.diff of a float32
    position at 2e4 rad is 2 rad/s of quantisation noise per sample."""
    h = max(1, int(window) // 2)
    w = np.empty_like(theta)
    w[h:-h] = (theta[2 * h:] - theta[:-2 * h]) * fs / (2 * h)
    w[:h] = w[h]
    w[-h:] = w[-h - 1]
    return w


def load(path=COGGING_LOG, verbose=True):
    """Read the cogging log into a dict of float64 arrays plus derived signals."""
    with h5py.File(path, 'r') as f:
        d = {k: f[k][:].astype(np.float64) for k in _CHANNELS}
        d['tare_bias'] = cell_tare_bias(f)
        d['logged_split_mv'] = cell_split_mv(f)
    d['t'] = d['time'] - d['time'][0]
    d['th_m'] = d['dut_output_position']               # motor angle [rad]
    d['th_o'] = d['load_position']                     # output angle [rad]
    d['th_e'] = POLE_PAIRS * d['th_m']                 # electrical angle [rad]
    d['w_m'] = speed_from_position(d['th_m'])          # motor speed [rad/s]
    d['cell_raw'] = d['load_torque'] - d['tare_bias']  # untared cell [Nm]
    # Logs from 2026-10-21 on are corrected at acquisition; never correct twice.
    d['cell_corr'] = (d['cell_raw'] if d['logged_split_mv'] else split_correct(d['cell_raw']))
    d['T_lm'] = KT_LOAD * d['load_current']            # LOAD motor torque [Nm]
    d['dwells'] = find_dwells(d)

    # Mechanical zero of the corrected cell: midpoint of the zero-current
    # dwells in the two directions (the gearbox's no-load drag is taken to be
    # symmetric -- the cell zero and an asymmetric drag cannot be separated).
    zero_by_dir = {-1: [], 1: []}
    for w in d['dwells']:
        if w['quadrant'] == 'zero':
            zero_by_dir[w['dir']].append(np.mean(d['cell_corr'][w['i0']:w['i1']]))
    d['cell_zero'] = 0.5 * (np.mean(zero_by_dir[-1]) + np.mean(zero_by_dir[1]))
    d['T_cell'] = d['cell_corr'] - d['cell_zero']      # corrected, re-zeroed cell [Nm]
    for w in d['dwells']:
        sl = slice(w['i0'], w['i1'])
        w['T_cell'] = float(np.mean(d['T_cell'][sl]))
        # Torque LOAD pushes along the direction of motion: positive = LOAD
        # drives the gearbox (backdrive), negative = LOAD brakes it.
        w['P'] = -w['dir'] * w['T_cell']
        w['T_lm'] = float(np.mean(d['T_lm'][sl]))
        w['temp_C'] = float(np.nanmean(d['pt100_rtd_top'][sl]))
    if verbose:
        print(f'loaded {os.path.basename(path)}: {len(d["t"]):,} samples, '
              f'{len(d["dwells"])} dwells, cell zero {d["cell_zero"]:+.3f} Nm (raw)')
    return d


def find_dwells(d):
    """Command plateaus of the speed x torque grid, trimmed to the settled part.

    The grid is (LOAD speed command) x (DUT torque command); the commands ramp
    between plateaus, so plateaus are runs of constant (speed, torque) longer
    than DUT_MIN_S.
    """
    t, tc, vc = d['t'], d['dut_torque_command'], d['load_velocity_command']
    key = np.round(np.nan_to_num(tc), 4) * 1000 + np.round(np.nan_to_num(vc), 4)
    edges = np.flatnonzero(np.diff(key) != 0) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, len(t)]
    dwells = []
    for s, e in zip(starts, ends):
        if t[e - 1] - t[s] < DWELL_MIN_S or abs(vc[s]) < 1e-6:
            continue
        i0 = s + int(DWELL_SETTLE_S * FS_HZ)
        i1 = e - int(DWELL_TAIL_S * FS_HZ)
        speed = round(float(vc[s]), 4)
        tcmd = round(float(tc[s]), 3)
        current = float(np.mean(d['dut_current'][i0:i1]))
        direction = int(np.sign(speed))
        if tcmd == 0:
            quadrant = 'zero'
        elif tcmd * speed < 0:
            quadrant = 'forward'      # DUT torque along the motion: DUT drives LOAD
        else:
            quadrant = 'backdrive'    # DUT torque against the motion: LOAD drives DUT
        # DUT current along the motion: + = DUT pushes with the motion.
        i_along = -direction * current
        dwells.append({
            'n': len(dwells), 'i0': int(i0), 'i1': int(i1),
            't0': float(t[i0]), 't1': float(t[i1 - 1]),
            'speed': speed, 'speed_abs': abs(speed), 'dir': direction,
            'tcmd': tcmd, 'current': current, 'i_along': float(i_along),
            'i_along_nom': int(round(-direction * tcmd / KT_CONFIGURED)),
            'quadrant': quadrant,
            'motor_speed_nom': GEAR_RATIO * abs(speed),
        })
    return dwells


def dwell_slice(w):
    return slice(w['i0'], w['i1'])


def speeds(d):
    return sorted({w['speed_abs'] for w in d['dwells']})


def currents_along(d):
    return sorted({w['i_along_nom'] for w in d['dwells']})


def sliding_mask(d, w):
    """Samples of dwell `w` where the motor is in steady sliding (not stuck,
    not in a slip burst)."""
    wn = w['motor_speed_nom']
    r = (d['w_m'][dwell_slice(w)] * w['dir']) / wn
    return (r > SLIDE_BAND[0]) & (r < SLIDE_BAND[1])


# --- Signal tools -------------------------------------------------------------------
def fold(phase, x, nbins, min_count=5):
    """Bin x by phase (rad, any range) over one period. Returns (centers_deg, mean, count)."""
    k = np.floor((np.mod(phase, TWO_PI) / TWO_PI) * nbins).astype(int)
    k = np.clip(k, 0, nbins - 1)
    cnt = np.bincount(k, minlength=nbins)
    s = np.bincount(k, weights=x, minlength=nbins)
    m = np.full(nbins, np.nan)
    ok = cnt >= min_count
    m[ok] = s[ok] / cnt[ok]
    centers = (np.arange(nbins) + 0.5) * 360.0 / nbins
    return centers, m, cnt


def angle_resample(theta, x, step):
    """Bin-average x onto a uniform grid in |theta| (travel), filling empty
    bins by interpolation. Turns a time series with speed ripple and stick-slip
    into one sampled uniformly in angle, so an FFT gives orders, not Hz."""
    th = np.abs(theta - theta[0])
    k = np.floor(th / step).astype(int)
    n = int(k.max()) + 1
    cnt = np.bincount(k, minlength=n)
    s = np.bincount(k, weights=x, minlength=n)
    y = np.full(n, np.nan)
    ok = cnt > 0
    y[ok] = s[ok] / cnt[ok]
    idx = np.arange(n)
    y = np.interp(idx, idx[ok], y[ok])
    return y, float(1.0 - ok.mean())


def order_spectrum(y, step, window=True):
    """Single-sided amplitude spectrum of a uniformly angle-sampled series.
    Returns (orders [cycles per 2pi of the sampled angle], amplitude [units of y])."""
    y = y - np.mean(y)
    n = len(y)
    win = np.hanning(n) if window else np.ones(n)
    amp = 2.0 * np.abs(np.fft.rfft(y * win)) / (n * win.mean())
    orders = np.fft.rfftfreq(n, d=step / TWO_PI)
    return orders, amp


def lsq_orders(columns_theta, x, nuisance=None):
    """Least-squares fit of x to sinusoids.

    columns_theta: list of (name, theta_array, order) -- one cos/sin pair each.
    nuisance: optional (n x k) design matrix of extra columns (offsets, drifts).
    Returns ({name: (amplitude, phase_rad)}, fitted_components {name: series},
    nuisance_fit series, residual).
    Phase follows x ~ A cos(order*theta + phase).
    """
    cols = []
    for _, th, k in columns_theta:
        cols.append(np.cos(k * th))
        cols.append(np.sin(k * th))
    nn = 0 if nuisance is None else nuisance.shape[1]
    X = np.column_stack(cols + ([nuisance] if nuisance is not None else []))
    coef, *_ = np.linalg.lstsq(X, x, rcond=None)
    out, comps = {}, {}
    for i, (name, th, k) in enumerate(columns_theta):
        a, b = coef[2 * i], coef[2 * i + 1]
        out[name] = (float(np.hypot(a, b)), float(np.arctan2(-b, a)))
        comps[name] = a * X[:, 2 * i] + b * X[:, 2 * i + 1]
    nuis = X[:, len(cols):] @ coef[len(cols):] if nn else np.zeros_like(x)
    resid = x - X @ coef
    return out, comps, nuis, resid


def legendre_nuisance(n, degree):
    """Offset plus slow polynomial drift over a dwell, as design columns."""
    u = np.linspace(-1.0, 1.0, n)
    return np.column_stack([np.polynomial.legendre.legval(u, np.eye(degree + 1)[j])
                            for j in range(degree + 1)])


# --- Plot style ------------------------------------------------------------------------
INK = '#0b0b0b'
INK_2 = '#52514e'
INK_3 = '#8a8984'
GRID = '#e4e3de'
SURFACE = '#fcfcfb'

# Three speed magnitudes -> the first three categorical slots (validated all-pairs).
SPEED_COLORS = {0.025: '#2a78d6', 0.05: '#eb6834', 0.1: '#1baf7a'}
DIR_MARKER = {-1: 'v', 1: '^'}
DIR_LS = {-1: '--', 1: '-'}
DIR_LABEL = {-1: 'output turning -', 1: 'output turning +'}

# Signed DUT current along the motion: diverging red (backdrive) <-> gray <-> blue (forward).
_CURRENT_COLORS = {-15: '#a8302f', -10: '#e34948', -5: '#f2a19f',
                   0: '#8a8984',
                   5: '#86b6ef', 10: '#3987e5', 15: '#1c5cab'}
QUADRANT_FILL = {'forward': '#e8f1fc', 'backdrive': '#fcebea', 'zero': None}


def current_color(i_along_nom):
    return _CURRENT_COLORS.get(int(i_along_nom), INK_3)


def current_label(i_along_nom):
    i = int(i_along_nom)
    if i == 0:
        return '0 A'
    return f'{"backdrive" if i < 0 else "forward"} {abs(i)} A'


def speed_label(v):
    return f'{v:g} rad/s out ({GEAR_RATIO * v * 60 / TWO_PI:.1f} rpm motor)'


def apply_style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE,
        'axes.edgecolor': INK_3, 'axes.labelcolor': INK, 'axes.titlecolor': INK,
        'axes.titlesize': 11, 'axes.labelsize': 10, 'axes.titleweight': 'bold',
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
        'xtick.color': INK_2, 'ytick.color': INK_2,
        'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 8.5, 'legend.frameon': False,
        'lines.linewidth': 1.6, 'lines.markersize': 6,
        'font.family': 'DejaVu Sans',
        'figure.titlesize': 13, 'figure.titleweight': 'bold',
    })


def save(fig, subdir, name, dpi=150):
    out_dir = os.path.join(RESULTS_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    # On Windows a viewer or file watcher can hold a just-viewed PNG for a
    # moment, and the overwrite then fails with EINVAL/EACCES. Retry briefly.
    for attempt in range(6):
        try:
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
            break
        except OSError:
            if attempt == 5:
                raise
            time.sleep(0.5 * (attempt + 1))
    plt.close(fig)
    print(f'  wrote {os.path.relpath(path, DYNO_DIR)}')
    return path


def save_csv(subdir, name, header, rows):
    out_dir = os.path.join(RESULTS_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, 'w') as fh:
        fh.write(','.join(header) + '\n')
        for r in rows:
            fh.write(','.join('' if v is None else (f'{v:.6g}' if isinstance(v, float) else str(v))
                              for v in r) + '\n')
    print(f'  wrote {os.path.relpath(path, DYNO_DIR)}')
    return path


def record(section, values):
    """Merge `values` into results/skip_key_numbers.json under `section`."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    data = {}
    if os.path.exists(KEY_NUMBERS_PATH):
        with open(KEY_NUMBERS_PATH) as fh:
            data = json.load(fh)
    data[section] = values
    with open(KEY_NUMBERS_PATH, 'w') as fh:
        json.dump(data, fh, indent=2, default=float)


def note(fig, text, y=-0.02):
    """Small caption under a figure."""
    fig.text(0.01, y, text, ha='left', va='top', fontsize=8.5, color=INK_2, wrap=True)
