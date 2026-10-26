"""Refit the DUT compensation from the session's own A/B log (R3), the in-situ
calibration Skip_Next_Steps.pdf section 5 describes, and write R3b to test it.

From every profile-flagged hold (s16_comp_ab.holds) of the given log(s):

  base / speed  per direction and speed, T = K_f*I (forward) | K_b*I (backdrive)
                - D*s on (mean commanded current, mean cell torque). The ripple
                terms average to zero over the whole motor revs each hold is
                cut to, so every profile's holds count; C0's different gain
                spreads the current range. base = the lowest speed.
  h1, h6        from the holds with no ripple term on (C0, C1, C2, C1_repeat),
                where the current is constant: cell torque folded per
                (direction, quadrant), a0 + a1*|T| with one phase, in v2's frame
                (theta + the config's phase_offset_rad), as s15 does for v2.
  hx (2/rev)    from the ERROR of the C3_h1 holds: R2 showed the 1/rev
                injection itself makes 2/rev (load-dependent 1/rev friction
                times a 1/rev current), so the 2/rev to cancel is the one left
                with h1 on, not the constant-current one.
  naive         pinned to the value the config already has, so C0 is the same
                law in R3 and R3b and measures the drift between the runs.

Prints the refit against the v2 values, the linearity of each branch (residual
per target), and the C1 -> C1_repeat drift.

Run:
    .venv/bin/python dyno/src/post_processors/skip/s17_session_refit.py R3.hdf5 [more.hdf5]

Outputs (results/comp_fit/):
  dut_compensation_session.yaml   the block, ready for apply (see --apply)
  --apply CONFIG                  replace the dut_compensation block in CONFIG
                                  as text (never yaml.safe_dump a rig config)
"""

import argparse
import os
import re

import h5py
import numpy as np
import yaml

import skip_common as c
import s15_comp_fit as fit15
import s16_comp_ab as ab
from dyno.src.dut_compensation import compile_dut_compensation

CONFIG = os.path.join(c.DYNO_DIR, 'config', 'inhouse_dyno_config.yaml')
OUT = os.path.join(c.SKIP_LOGS, 'results', 'comp_fit', 'dut_compensation_session.yaml')
NO_RIPPLE = ('C0_naive', 'C1_base', 'C2_speed', 'C1_repeat')
RIPPLE_FULL = 0.25           # output rad/s; ripple terms fade to 0 by RIPPLE_ZERO
RIPPLE_ZERO = 1.0
# qdrag on top of the proven speed term, alone and in the full stack, so that
# C2 -> C7 and C6 -> C8 each isolate it.
QDRAG_PROFILES = {'C7_qdrag': ['base', 'speed', 'qdrag'],
                  'C8_all_q': ['base', 'speed', 'qdrag', 'h1', 'h6', 'hx']}
MIN_KEPT = 0.70            # below this a refit harmonic cell keeps its previous value
DIR_NAME = {1: 'pos', -1: 'neg'}


def branch_fit(rows):
    """K_f, K_b, D, rms for one direction from hold rows (current_mean, T)."""
    X, y = [], []
    for r in rows:
        I, s = r['current_mean'], int(np.sign(r['speed']))
        T = r['target'] + r['bias']
        X.append([I * (s * I > 0), I * (s * I <= 0), -s])
        y.append(T)
    X, y = np.array(X), np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ coef
    return float(coef[0]), float(coef[1]), float(coef[2]), float(np.sqrt(np.mean(res ** 2))), res


def line_fit(rows):
    """One quadrant's own line T = K*I - D*s: (K, D, rms, residuals by |target|)."""
    I = np.array([r['current_mean'] for r in rows])
    s = np.array([np.sign(r['speed']) for r in rows])
    T = np.array([r['target'] + r['bias'] for r in rows])
    X = np.column_stack([I, -s])
    coef, *_ = np.linalg.lstsq(X, T, rcond=None)
    res = T - X @ coef
    by_t = {}
    for r, e in zip(rows, res):
        by_t.setdefault(abs(round(r['target'])), []).append(e)
    return float(coef[0]), float(coef[1]), float(np.sqrt(np.mean(res ** 2))), by_t


def spans(path, offset):
    """Raw arrays of every flagged hold, angle in v2's frame, whole revs."""
    out = []
    with h5py.File(path, 'r') as f:
        ids = [b.decode() if isinstance(b, bytes) else str(b) for b in f['behavior_ids'][:]]
        idx = f['behavior_indices'][:]
        cell = f['load_torque'][:].astype(float)
        tgt = f['dut_comp_target'][:].astype(float)
        th = f['dut_output_position'][:].astype(float)
        vc = f['load_velocity_command'][:].astype(float)
    for bid, (i0, i1) in zip(ids, idx):
        m = ab.FLAG.match(bid)
        if not m:
            continue
        sl = slice(int(i0), int(i1) + 1)
        a = th[sl]
        revs = np.abs(a - a[0]) / c.TWO_PI
        n_rev = int(np.floor(revs.max()))
        if n_rev < 2:
            continue
        keep = revs <= n_rev
        out.append({'profile': m['profile'], 'speed': float(np.median(vc[sl])),
                    'target': float(np.median(tgt[sl])), 'theta': a[keep] + offset,
                    'cell': cell[sl][keep], 'err': cell[sl][keep] - tgt[sl][keep]})
    return out


def order_coef(theta, y, k_list, direction):
    yr, _ = c.angle_resample(theta, y, c.TWO_PI / 512)
    n = len(yr)
    g = theta[0] + direction * c.TWO_PI / 512 * (np.arange(n) + 0.5)
    revs = n / 512
    amps, _, _, _ = c.lsq_orders([(k, g, k) for k in k_list], yr,
                                 c.legendre_nuisance(n, min(4, max(1, int(revs) - 1))))
    return {k: amps[k][0] * np.exp(1j * amps[k][1]) for k in k_list}


def harmonic_fit(spans_, profiles, signal, orders, max_speed):
    """{(dir, quadrant): {k: (a0, a1, phi, kept)}} from the given profiles' holds."""
    acc = {}
    for sp in spans_:
        if sp['profile'] not in profiles or abs(sp['speed']) > max_speed + 1e-9:
            continue
        s = int(np.sign(sp['speed']))
        q = 'forward' if float(np.mean(sp['cell'])) * s > 0 else 'backdrive'
        C = order_coef(sp['theta'], sp[signal], orders, s)
        # Ripple scales with the torque actually transmitted, not the target:
        # C0's holds miss their target by up to ~10 Nm.
        acc.setdefault((s, q), []).append((abs(float(np.mean(sp['cell']))), C))
    out = {}
    for key, pts in acc.items():
        load = np.array([p[0] for p in pts])
        out[key] = {}
        for k in orders:
            C = np.array([p[1][k] for p in pts])
            if np.ptp(load) < 1.0:
                out[key][k] = None   # one load level cannot give a0 and a1
                continue
            X = np.column_stack([np.ones_like(load), load]).astype(complex)
            uv, *_ = np.linalg.lstsq(X, C, rcond=None)
            phi = float(np.angle(uv[0] + uv[1] * load.max()))
            rot = np.exp(-1j * phi)
            a0, a1 = float((uv[0] * rot).real), float((uv[1] * rot).real)
            model = (a0 + a1 * load) * np.exp(1j * phi)
            kept = 1.0 - float(np.sum(np.abs(C - model) ** 2) / np.sum(np.abs(C) ** 2))
            out[key][k] = (a0, a1, phi, kept)
    return out


def replace_block(config_path, block_text):
    s = open(config_path).read()
    start = s.index('\ndut_compensation:\n') + 1
    end = s.index('\n# --- 0. PORTS ---')
    s = s[:start] + block_text.rstrip() + '\n' + s[end:]
    open(config_path, 'w').write(s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--apply', metavar='CONFIG', help='write the refit into this rig config')
    args = ap.parse_args()

    raw = open(CONFIG).read()
    old_spec = yaml.safe_load(raw)['dut_compensation']
    old = compile_dut_compensation({'dut_compensation': old_spec})
    offset = old['phase_offset_rad']

    rows, sps = [], []
    for p in args.logs:
        rows += ab.holds(p)
        sps += spans(p, offset)
    print(f'{len(rows)} holds from {len(args.logs)} log(s)')

    # --- base / speed ---------------------------------------------------------------
    speeds = sorted({round(abs(r['speed']), 4) for r in rows})
    table = {}
    print('branch fit per speed (+v / -v), today vs v2 base:')
    for v in speeds:
        table[v] = {}
        for s in (1, -1):
            rs = [r for r in rows if round(abs(r['speed']), 4) == v and np.sign(r['speed']) == s]
            quads = {('f' if r['current_mean'] * s > 0 else 'b') for r in rs}
            if quads != {'f', 'b'}:
                table[v][s] = None
                continue
            K_f, K_b, D, rms, res = branch_fit(rs)
            table[v][s] = (K_f, K_b, D, rms)
            # linearity: mean residual per |target| and quadrant
            lin = {}
            for r, e in zip(rs, res):
                lin.setdefault((('fwd' if r['current_mean'] * s > 0 else 'bd'), abs(round(r['target']))), []).append(e)
            lin_txt = ' '.join(f'{q}{t}:{np.mean(e):+.2f}' for (q, t), e in sorted(lin.items()))
            d = DIR_NAME[s]
            i = 0 if s > 0 else 1
            print(f'  {v:4.2f} {d}: K_f {K_f:.3f} (v2 {old["base"]["K_f"][i]:.3f})  '
                  f'K_b {K_b:.3f} (v2 {old["base"]["K_b"][i]:.3f})  D {D:+.2f} (v2 '
                  f'{old["base"]["drag_nm"][i]:+.2f})  rms {rms:.2f}  resid {lin_txt}')
    good = [v for v in speeds if all(table[v][s] for s in (1, -1))]

    # --- quadrant lines (qdrag): forward and backdrive each get their own line --
    qlines = {}
    print('quadrant lines (qdrag), per speed and direction: K, D, rms, residual by |target|:')
    for v in good:
        qlines[v] = {}
        for s in (1, -1):
            rs = [r for r in rows if round(abs(r['speed']), 4) == v and np.sign(r['speed']) == s]
            fw = [r for r in rs if r['current_mean'] * s > 0]
            bd = [r for r in rs if r['current_mean'] * s <= 0]
            K_f, D_f, rf, tf = line_fit(fw)
            K_b, D_b, rb, tb = line_fit(bd)
            qlines[v][s] = (K_f, D_f, K_b, D_b)
            txt = lambda t: ' '.join(f'{k}:{np.mean(e):+.2f}' for k, e in sorted(t.items()))
            print(f'  {v:4.2f} {DIR_NAME[s]}: fwd K {K_f:.3f} D {D_f:+.2f} rms {rf:.2f} [{txt(tf)}]   '
                  f'bd K {K_b:.3f} D {D_b:+.2f} rms {rb:.2f} [{txt(tb)}]')
    if not good:
        raise SystemExit('no speed has both quadrants in both directions: cannot refit the base')

    # --- drift ------------------------------------------------------------------------
    pair = {}
    for r in rows:
        if r['profile'] in ('C1_base', 'C1_repeat'):
            pair.setdefault((r['segment'], r['point']), {})[r['profile']] = r['bias']
    drift = [p['C1_repeat'] - p['C1_base'] for p in pair.values() if len(p) == 2]
    if drift:
        print(f'C1 -> C1_repeat bias drift within a point: mean {np.mean(drift):+.3f}, '
              f'rms {np.sqrt(np.mean(np.square(drift))):.3f} Nm over {len(drift)} points')

    # --- harmonics --------------------------------------------------------------------
    # Only where the ripple fade is 1: above ~0.25 rad/s the ripple is not the
    # table's (R3: ~0.7x at 0.5 rad/s, phase-scrambled at 2).
    h16 = harmonic_fit(sps, NO_RIPPLE, 'cell', (1, 6), RIPPLE_FULL)
    hx = harmonic_fit(sps, ('C3_h1',), 'err', (2,), RIPPLE_FULL)
    print('harmonics (today, v2 frame): a0 + a1|T| @ phase, kept')
    for key in sorted(h16):
        parts = []
        for k, src in ((1, h16), (6, h16), (2, hx)):
            v = src.get(key, {}).get(k)
            parts.append(f'k{k}: ' + ('n/a' if v is None else
                                      f'{v[0]:+.2f}{v[1]:+.3f}|T| @{np.degrees(v[2]):6.1f}° {v[3]:.0%}'))
        print(f'  {DIR_NAME[key[0]]}_{key[1]:9s} ' + '  '.join(parts))

    # --- assemble -----------------------------------------------------------------------
    base_v = good[0]
    pairv = lambda v, j: [round(table[v][1][j], 4), round(table[v][-1][j], 4)]
    spec = {
        'enabled': True,
        'phase_offset_rad': round(offset, 4),
        'max_current_a': old['max_current_a'],
        'naive_nm_per_a': round(old['naive_nm_per_a'], 4),
        'ripple_fade_rad_s': [RIPPLE_FULL, RIPPLE_ZERO],
        'base': {'K_f': pairv(base_v, 0), 'K_b': pairv(base_v, 1), 'drag_nm': pairv(base_v, 2)},
        'speed_table': {'speed_rad_s': [float(v) for v in good],
                        'K_f': [pairv(v, 0) for v in good],
                        'K_b': [pairv(v, 1) for v in good],
                        'drag_nm': [pairv(v, 2) for v in good]} if len(good) >= 2 else None,
        'quadrant_lines': {
            'speed_rad_s': [float(v) for v in good],
            **{k: [[round(qlines[v][1][j], 4), round(qlines[v][-1][j], 4)] for v in good]
               for j, k in enumerate(('K_f', 'D_f', 'K_b', 'D_b'))}},
        'harmonics': [],
        'profiles': dict(old_spec['profiles'], **QDRAG_PROFILES),
    }
    if spec['speed_table'] is None:
        del spec['speed_table']
    for group, k, src in (('h1', 1, h16), ('h6', 6, h16), ('hx', 2, hx)):
        cells = {}
        for (s, q), byk in src.items():
            v = byk.get(k)
            # A cell the shared-phase model explains poorly keeps its old value
            # (R3: the forward 2/rev cells, 47-65%, with a sign-flipping a0).
            if v is not None and v[3] >= MIN_KEPT:
                cells[f'{DIR_NAME[s]}_{q}'] = {'a0': round(v[0], 4), 'a1': round(v[1], 5),
                                              'phase': round(v[2], 4)}
        # a cell today's data cannot fit keeps its v2 value
        for h in old_spec.get('harmonics', []):
            if h['order'] == k:
                for name, cv in h['cells'].items():
                    cells.setdefault(name, cv)
        spec['harmonics'].append({'group': group, 'order': k, 'cells': cells})
    compile_dut_compensation({'dut_compensation': spec})   # validates

    head = ('# Session refit by dyno/src/post_processors/skip/s17_session_refit.py from\n'
            f'# {", ".join(os.path.relpath(p, c.DYNO_DIR) for p in args.logs)}.\n'
            '# naive_nm_per_a is pinned to the previous block so C0 is the same law.\n')
    text = yaml.safe_dump({'dut_compensation': spec}, sort_keys=False,
                          default_flow_style=None, width=100)
    with open(OUT, 'w') as f:
        f.write(head + text)
    print(f'wrote {OUT}')
    if args.apply:
        replace_block(args.apply, head + text)
        compile_dut_compensation(yaml.safe_load(open(args.apply)))
        print(f'replaced the dut_compensation block in {args.apply}')


if __name__ == '__main__':
    main()
