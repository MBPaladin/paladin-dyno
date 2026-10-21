"""Angle-domain decomposition of the load cell torque, shared by the fold,
spectrum and decomposition scripts.

Why angle domain: with stick-slip, a stuck motor sits at one angle while the
torque ramps for up to a second; fitted in time, those samples would teach an
angle-locked model the stick ramp. Bin-averaging each dwell onto a uniform
motor-angle grid first gives every angle equal weight, so what is fitted is
torque as a function of angle, which is what "cogging" means.

Per dwell (motor-angle grid, 1024 points per motor rev):

    T(theta) = slow(travel) + MECH(theta_m) + ELEC(theta_m) + residual

  slow   Legendre polynomial over the dwell's travel (offset, thermal drift,
         and the output-angle-locked content, which is far below 1/motor rev)
  MECH   motor-locked orders that are not multiples of 21 (eccentric, pins,
         bearings, output mechanism, motor mechanical)
  ELEC   multiples of 21 per motor rev (1x..6x electrical)

The output-angle-locked part (orders per output rev, i.e. the gearbox's slow
phase) is fitted afterwards, jointly across dwells, on what MECH and ELEC
leave behind -- one dwell covers at most 1.9 output revs, too few on its own.
"""

import numpy as np

import skip_common as c

STEP_M = c.TWO_PI / 1024            # motor-angle grid step [rad]; ~2.4 samples per bin at 0.1 rad/s
SLOW_DEGREE = 8
ELEC_ORDERS = tuple(c.POLE_PAIRS * k for k in range(1, 7))          # 21..126
MECH_ORDERS = tuple(list(range(1, 21)) + [27, 35, 36, 37, 57, 72])  # per motor rev
OUTPUT_ORDERS = tuple(range(1, 9))                                  # per output rev


def resample_dwell(d, w, signal='T_cell'):
    """Dwell `w` on a uniform motor-angle grid. Returns dict of grid arrays."""
    if '_rs' not in d:
        d['_rs'] = {}
    key = (w['n'], signal)
    if key in d['_rs']:
        return d['_rs'][key]
    sl = c.dwell_slice(w)
    th = d['th_m'][sl]
    y, gap = c.angle_resample(th, d[signal][sl], STEP_M)
    to, _ = c.angle_resample(th, d['th_o'][sl], STEP_M)
    n = len(y)
    th_grid = th[0] + w['dir'] * STEP_M * (np.arange(n) + 0.5)
    rs = {'y': y, 'th_m': th_grid, 'th_o': to, 'th_e': c.POLE_PAIRS * th_grid,
          'gap': gap, 'n': n, 'motor_revs': n * STEP_M / c.TWO_PI}
    d['_rs'][key] = rs
    return rs


def fit_dwell(d, w, signal='T_cell'):
    """Per-dwell motor-locked fit. Cached in d['_fit']."""
    if '_fit' not in d:
        d['_fit'] = {}
    key = (w['n'], signal)
    if key in d['_fit']:
        return d['_fit'][key]
    rs = resample_dwell(d, w, signal)
    cols = [(f'm{k}', rs['th_m'], k) for k in MECH_ORDERS]
    cols += [(f'e{k}', rs['th_m'], k) for k in ELEC_ORDERS]
    amps, comps, slow, resid = c.lsq_orders(cols, rs['y'], c.legendre_nuisance(rs['n'], SLOW_DEGREE))
    mech = np.sum([comps[f'm{k}'] for k in MECH_ORDERS], axis=0)
    elec = np.sum([comps[f'e{k}'] for k in ELEC_ORDERS], axis=0)
    fit = {'amps': amps, 'mech': mech, 'elec': elec, 'slow': slow, 'resid': resid,
           # what is left for the output-locked fit: everything but MECH/ELEC
           'for_output': rs['y'] - mech - elec}
    d['_fit'][key] = fit
    return fit


def fit_output_group(d, dwells, signal='T_cell', orders=OUTPUT_ORDERS):
    """Shared output-angle-locked profile for a group of dwells, with a free
    offset per dwell. Returns (coef dict {k: (a, b)}, per-dwell offsets, eval fn)."""
    ys, ths, idx = [], [], []
    for j, w in enumerate(dwells):
        f = fit_dwell(d, w, signal)
        rs = resample_dwell(d, w, signal)
        ys.append(f['for_output'])
        ths.append(rs['th_o'])
        idx.append(np.full(rs['n'], j))
    y = np.concatenate(ys)
    th = np.concatenate(ths)
    idx = np.concatenate(idx)
    cols = []
    for k in orders:
        cols += [np.cos(k * th), np.sin(k * th)]
    offs = np.zeros((len(y), len(dwells)))
    offs[np.arange(len(y)), idx] = 1.0
    X = np.column_stack(cols + [offs])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    ab = {k: (float(coef[2 * i]), float(coef[2 * i + 1])) for i, k in enumerate(orders)}
    offsets = coef[2 * len(orders):]

    def evaluate(theta_o):
        out = np.zeros_like(theta_o, dtype=float)
        for k, (a, b) in ab.items():
            out += a * np.cos(k * theta_o) + b * np.sin(k * theta_o)
        return out
    return ab, offsets, evaluate


def recon(amps, orders, prefix, theta):
    """Rebuild a Fourier series from lsq_orders amplitudes/phases."""
    out = np.zeros_like(theta, dtype=float)
    for k in orders:
        A, ph = amps[f'{prefix}{k}']
        out += A * np.cos(k * theta + ph)
    return out


def pkpk(x):
    return float(np.nanmax(x) - np.nanmin(x))
