"""Logic-only checks of the open-loop DUT compensation law (dyno/src/dut_compensation.py).

Run from repo root:
  PYTHONPATH=. .venv/bin/python dyno/sim/dut_compensation_logic_test.py

No bus, no controller. Uses the fitted block from
logs/skip/results/comp_fit/dut_compensation_fit.yaml (s15_comp_fit.py) when it
exists, so the numbers are the ones the rig will run. Checks:
  - the base law inverts its own forward model in all four quadrants, and the
    current is continuous through the quadrant switch;
  - the naive profile is tau / naive_nm_per_a and ignores everything else;
  - ripple terms are off at standstill and fully on once sliding;
  - against a plant whose ripple IS the fitted ripple, each harmonic group
    removes its own order, and an 11 deg phase error (the size seen after an
    Elmo power cycle) costs most of the 6/rev benefit;
  - the velocity estimate tracks a constant speed;
  - config validation rejects unknown profile terms.
"""
import math
import os

import yaml

from dyno.src.dut_compensation import DutCompensation, compile_dut_compensation

HERE = os.path.dirname(os.path.abspath(__file__))
FIT = os.path.join(HERE, '..', 'logs', 'skip', 'results', 'comp_fit', 'dut_compensation_fit.yaml')
G = 26.0
DT = 0.001

ok = True


def check(cond, msg):
    global ok
    if not cond:
        ok = False
        print('FAIL:', msg)


def load_cfg(phase_offset=0.0):
    if os.path.exists(FIT):
        with open(FIT) as f:
            spec = yaml.safe_load(f)
    else:  # minimal stand-in so the logic still runs on a fresh checkout
        spec = {'dut_compensation': {
            'base': {'K_f': 0.54, 'K_b': 1.39, 'drag_nm': 0.64},
            'harmonics': [{'group': 'h1', 'order': 1, 'cells': {
                q: {'a0': 0.08, 'a1': 0.05, 'phase': 1.0}
                for q in ('pos_forward', 'pos_backdrive', 'neg_forward', 'neg_backdrive')}}],
            'profiles': {'C0_naive': [], 'C1_base': ['base'], 'C3_h1': ['base', 'h1'],
                         'C6_all': ['base', 'h1']}}}
    spec['dut_compensation']['phase_offset_rad'] = phase_offset
    return compile_dut_compensation(spec)


def plant_base(cfg, I, w_m, dir_idx):
    """The base model forward: output torque for current I (no ripple)."""
    b = cfg['base']
    s = math.tanh(w_m / G / cfg['v_blend_rad_s'])
    K = b['K_f'][dir_idx] if s * I > 0 else b['K_b'][dir_idx]
    return K * I - b['drag_nm'][dir_idx] * s


# --- base inversion and continuity ---------------------------------------------------
cfg = load_cfg()
comp = DutCompensation(cfg, G, DT)
for w_m in (2.6, -2.6):
    dir_idx = 0 if w_m > 0 else 1
    prev = None
    for i in range(-200, 201):
        tau = i * 0.1
        I = comp.update(tau, None, 'C1_base', omega_m=w_m)
        back = plant_base(cfg, I, w_m, dir_idx)
        if abs(I) >= cfg['max_current_a'] - 1e-9:
            check(abs(back) < abs(tau), f'clipped at tau {tau} but plant overshoots: {back}')
        else:
            check(abs(back - tau) < 1e-9, f'base inverse at w {w_m}, tau {tau}: plant gives {back}')
        if prev is not None:
            # max step between 0.1 Nm targets is 0.1 / K_f
            check(abs(I - prev) <= 0.1 / min(cfg['base']['K_f']) + 1e-9,
                  f'current jumps {I - prev:+.3f} A at tau {tau} (w {w_m})')
        prev = I

# known point: +v, 12 Nm forward needs (12 + D)/K_f
I = comp.update(12.0, None, 'C1_base', omega_m=2.6)
expect = (12.0 + cfg['base']['drag_nm'][0]) / cfg['base']['K_f'][0]
check(abs(I - expect) < 1e-6, f'12 Nm forward +v: {I:.3f} A, expected {expect:.3f}')
print(f'12 Nm targets at 0.1 rad/s: forward {I:.1f} A, backdrive '
      f'{comp.update(-12.0, None, "C1_base", omega_m=2.6):.1f} A')

# --- naive ---------------------------------------------------------------------------
for tau in (-12.0, -1.0, 0.0, 3.0, 12.0):
    I = comp.update(tau, 1.234, 'C0_naive', omega_m=-50.0)
    check(abs(I - tau / cfg['naive_nm_per_a']) < 1e-12, f'naive tau {tau}: {I}')
    check(comp.last['ripple_nm'] == 0.0, 'naive profile must not add ripple')

# --- gate --------------------------------------------------------------------------------
all_profile = 'C6_all'
comp.update(-12.0, 0.7, all_profile, omega_m=0.0)
check(comp.last['ripple_nm'] == 0.0 and comp.last['gate'] == 0.0, 'ripple must be off at standstill')
comp.update(-12.0, 0.7, all_profile, omega_m=2.6)
check(comp.last['gate'] == 1.0, 'gate must be fully on at 0.1 rad/s output')

# --- speed fade (only when configured) ------------------------------------------------
fcfg = dict(cfg, ripple_fade_rad_s=(0.25, 1.0))
fc = DutCompensation(fcfg, G, DT)
for w_out, expect in ((0.1, 1.0), (0.25, 1.0), (0.5, 2 / 3), (1.0, 0.0), (2.0, 0.0)):
    check(abs(fc.gate(w_out * G) - expect) < 1e-9, f'fade at {w_out} rad/s: {fc.gate(w_out * G)}')
fc.update(-12.0, 0.7, all_profile, omega_m=2.0 * G)
check(fc.last['ripple_nm'] == 0.0, 'ripple must be faded out at 2 rad/s')

# --- qdrag: separate forward / backdrive lines ------------------------------------------
for D_f, D_b in ((0.3, 0.6), (0.6, 0.3), (0.45, 0.45)):   # gap, overlap, shared
    qspec = {'dut_compensation': {
        'base': {'K_f': 0.5, 'K_b': 1.3, 'drag_nm': 0.45},
        'quadrant_lines': {'speed_rad_s': [0.1], 'K_f': [[0.51, 0.52]], 'D_f': [D_f],
                           'K_b': [[1.33, 1.37]], 'D_b': [D_b]},
        'profiles': {'Q': ['base', 'qdrag']}}}
    qc = DutCompensation(compile_dut_compensation(qspec), G, DT)
    for w_m in (2.6, -2.6):
        di = 0 if w_m > 0 else 1
        sgn = math.tanh(w_m / G / qc.cfg['v_blend_rad_s'])
        K_f, K_b = (0.51, 0.52)[di], (1.33, 1.37)[di]
        prev = None
        for i in range(-140, 141):          # 15 Nm forward would hit the 30 A cap
            tau = i * 0.1
            I = qc.update(tau, None, 'Q', omega_m=w_m)
            plant = K_f * I - D_f * sgn if sgn * I > 0 else K_b * I - D_b * sgn
            in_band = min(-D_f * sgn, -D_b * sgn) - 1e-9 <= tau <= max(-D_f * sgn, -D_b * sgn) + 1e-9
            if not in_band:
                check(abs(plant - tau) < 1e-9, f'qdrag D_f {D_f} D_b {D_b} w {w_m} tau {tau}: plant {plant}')
            if prev is not None:
                check(I >= prev - 1e-12, f'qdrag current not monotone at tau {tau} (D_f {D_f}, D_b {D_b})')
                # continuous; across an overlap it is steeper than either line
                check(I - prev <= 0.5, f'qdrag current jumps {I - prev:.3f} A at tau {tau}')
            prev = I
# shared drag reproduces the base law exactly
qspec['dut_compensation']['quadrant_lines'].update({'K_f': [0.5], 'K_b': [1.3], 'D_f': [0.45], 'D_b': [0.45]})
qspec['dut_compensation']['profiles']['B'] = ['base']
qc = DutCompensation(compile_dut_compensation(qspec), G, DT)
for tau in (-12.0, -0.2, 0.0, 0.3, 12.0):
    check(abs(qc.update(tau, None, 'Q', omega_m=2.6) - qc.update(tau, None, 'B', omega_m=2.6)) < 1e-12,
          f'qdrag with shared lines must equal base at tau {tau}')
try:
    compile_dut_compensation({'dut_compensation': {'base': {'K_f': 0.5, 'K_b': 1.4},
                                                   'profiles': {'X': ['base', 'qdrag']}}})
    check(False, 'qdrag without quadrant_lines should be rejected')
except ValueError:
    pass

# --- velocity estimate ---------------------------------------------------------------
est = DutCompensation(cfg, G, DT)
theta = 0.0
for _ in range(500):
    theta += 2.6 * DT
    est.update(1.0, theta, 'C1_base')
check(abs(est.omega_m - 2.6) < 0.01, f'velocity estimate {est.omega_m:.3f}, expected 2.6')


# --- closed-form plant with the fitted ripple --------------------------------------------
def residual_rms(profile, comp_offset, tau, w_m, true_offset=0.0):
    """rms of (plant output - target) over 5 motor revs, plant ripple = the fitted
    ripple in the frame where the true encoder offset is `true_offset`."""
    c_plant = DutCompensation(load_cfg(true_offset), G, DT)
    c_ctrl = DutCompensation(load_cfg(comp_offset), G, DT)
    terms_all = set(c_plant.cfg['profiles'].get('C6_all', ())) | {'h1', 'h6', 'hx'}
    dir_idx = 0 if w_m > 0 else 1
    n = int(5 * 2 * math.pi / abs(w_m) / DT)
    acc = 0.0
    for i in range(n):
        th = w_m * i * DT
        I = c_ctrl.update(tau, th, profile, omega_m=w_m)
        cell = c_ctrl.last and (('pos_' if dir_idx == 0 else 'neg_')
                                + ('forward' if c_ctrl.last['quadrant'] > 0 else 'backdrive'))
        # plant: base model plus the true ripple at this angle and load (no lead in the plant)
        r_true = 0.0
        for h in c_plant.cfg['harmonics']:
            if h['group'] in terms_all and cell in h['cells']:
                a0, a1, phi = h['cells'][cell]
                r_true += (a0 + a1 * abs(tau)) * math.cos(h['order'] * (th + true_offset) + phi)
        out = plant_base(c_ctrl.cfg, I, w_m, dir_idx) + r_true
        acc += (out - tau) ** 2
    return math.sqrt(acc / n)


if any(h['group'] == 'h6' for h in cfg['harmonics']):
    # backdrive -12 Nm at +0.1 rad/s output, where the 6/rev is largest
    w = 2.6
    base_r = residual_rms('C1_base', 0.0, -12.0, w)
    profiles = sorted(cfg['profiles'])
    print('plant = fitted ripple, backdrive 12 Nm, +0.1 rad/s; rms torque error (Nm):')
    results = {}
    for p in profiles:
        results[p] = residual_rms(p, 0.0, -12.0, w)
        print(f'  {p:10s} {results[p]:.3f}')
    check(results['C6_all'] < 0.2 * base_r, 'all terms should cancel a plant made of the fitted ripple')
    check(results['C4_h6'] < base_r, '6/rev term should reduce the error')
    # Lead: with t_lead the controller looks w*t_lead ahead; the plant has no
    # delay, so that is itself a small phase error. Measure the true-offset cost
    # with the same lead in both, by giving the plant the lead as its offset.
    lead = w * cfg['t_lead_s']
    ok11 = residual_rms('C6_all', math.radians(11), -12.0, w, true_offset=lead)
    ok0 = residual_rms('C6_all', 0.0, -12.0, w, true_offset=lead)
    print(f'  C6_all with an 11 deg encoder error: {ok11:.3f} (correct offset {ok0:.3f})')
    check(ok11 > 2 * ok0, 'an 11 deg phase error should visibly hurt')

# --- validation --------------------------------------------------------------------------
try:
    compile_dut_compensation({'dut_compensation': {'base': {'K_f': 0.5, 'K_b': 1.4},
                                                   'profiles': {'X': ['base', 'h9']}}})
    check(False, 'unknown term h9 should be rejected')
except ValueError:
    pass
check(compile_dut_compensation({}) is None, 'no block -> None')
check(compile_dut_compensation({'dut_compensation': {'enabled': False}}) is None, 'disabled -> None')

print('PASS' if ok else 'FAILED')
raise SystemExit(0 if ok else 1)
