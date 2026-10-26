"""Open-loop DUT torque compensation: target OUTPUT torque -> DUT current.

Built for the Skip actuator (26:1 cycloid, Elmo DUT) to test, term by term,
how far feedforward alone gets the output torque to follow a target with no
torque loop closed (logs/skip/results/Skip_Next_Steps.pdf, section 2). Each
term can be switched on and off per test segment through named profiles, so
one test can compare them back to back at the same operating point.

The law, for a target output torque tau (Nm, the same sign the cell reads):

    tau_eff = tau - g(w) * R(theta + offset + w * t_lead, tau, cell)     ripple terms
    I       = (tau_eff + D * s) / K_q                                    base / speed

  s      = tanh(w_out / v_blend): smooth sign of the output speed.
  K_q    = K_f when the DUT drives the output (s * (tau_eff + D*s) > 0, forward),
           K_b when the output drives the DUT (backdrive). Output Nm per DUT amp.
           The current is continuous across the switch: both branches give
           I = 0 at tau_eff = -D*s, only the slope changes.
  D      = no-load drag at the output (Nm), opposing the motion.
  R      = sum over enabled harmonics of A * cos(k * theta_ref + phi), with
           A = a0 + a1 * |tau| per (direction, quadrant) cell. This is the
           angle-locked torque measured at constant current; subtracting it
           from the target before inverting the base law is the exact inverse,
           so the quadrant slope is applied to the ripple too.
  g(w)   = 0 below gate_lo and 1 above gate_hi (motor rad/s): ripple terms act
           only while the gearbox slides, never at standstill or reversals.
           Times the speed fade: 1 up to ripple_fade_rad_s[0], 0 from
           ripple_fade_rad_s[1] (output rad/s). The table is measured at
           <= 0.25 rad/s; R3 (2026-10-26) found the real ripple at 0.5 rad/s
           ~0.7x the table and at 2 rad/s phase-scrambled by rig dynamics, where
           injecting it made the 1/rev and 6/rev worse than nothing.

This is the client's existing law, tau_m = tau + sign(w)(F_c + F_g |tau|),
written in branch form: Kt*N = 2 / (1/K_f + 1/K_b), F_g = (K_b - K_f)/(K_b + K_f).

Terms, as named in a profile:
  base     the branch law with constant K_f, K_b, D (the low-speed fit).
  speed    K_f, K_b, D interpolated from `speed_table` at |w_out|.
           Without base, speed has nothing to act on, so it implies base.
  qdrag    separate forward and backdrive lines, each with its own slope
           and drag (`quadrant_lines`, per direction and speed): T = K_f*I -
           D_f*s forward, K_b*I - D_b*s backdrive. Taken at the table's lowest
           speed, or interpolated in speed when `speed` is on too. Added after
           R3b (2026-10-26) showed a constant backdrive offset and a forward
           slope error that one shared drag cannot fit.
  <group>  every harmonic whose `group` matches (h1, h6, hx, ...).
  An empty profile is the naive law I = tau / naive_nm_per_a.

Frames. theta is the DUT motor angle (rad, DUT.position: the Elmo's own
counter, untouched by any software offset). `phase_offset_rad` maps it onto the
frame the harmonic phases were fitted in (v2): theta_ref = theta + offset,
measured after every Elmo power cycle with post_processors/skip/s14_phase_cal.py.
Velocity w is the motor speed (rad/s), differentiated from theta here and
low-passed. Positive tau, positive current and positive w are the DUT command
frame. On the in-house rig (DUT flip_torque_sign and flip_direction_sign both
True) a positive command reads positive on the cell, and the motor position
increases with load_position, so forward is tau * w > 0.

Pure: no drive or controller access. Controller.step feeds it and turns the
current into a DUT torque command through the drive's configured kt.
"""

import math

TERMS_BUILTIN = ('base', 'speed', 'qdrag')


def _pair(value, name):
    """A parameter given as one magnitude or as [positive dir, negative dir]."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f'dut_compensation.{name}: give one value or [pos, neg]')
        return (float(value[0]), float(value[1]))
    return (float(value), float(value))


def compile_dut_compensation(dyno_params):
    """The `dut_compensation:` block, validated, or None if absent/disabled.
    Pure function of the config so Master can record it before the controller
    exists, like compile_feedforward."""
    spec = dyno_params.get('dut_compensation')
    if not spec or not spec.get('enabled', True):
        return None
    base = spec.get('base') or {}
    for k in ('K_f', 'K_b'):
        if k not in base:
            raise ValueError(f'dut_compensation.base needs {k}')
    K_f, K_b = _pair(base['K_f'], 'base.K_f'), _pair(base['K_b'], 'base.K_b')
    # Default naive gain: the client law's Kt*N, harmonic mean of the branches,
    # averaged over the two directions.
    naive = sum(2.0 / (1.0 / f + 1.0 / b) for f, b in zip(K_f, K_b)) / 2.0
    out = {
        'phase_offset_rad': float(spec.get('phase_offset_rad', 0.0)),
        'naive_nm_per_a': float(spec.get('naive_nm_per_a', naive)),
        'base': {'K_f': K_f,
                 'K_b': K_b,
                 'drag_nm': _pair(base.get('drag_nm', 0.0), 'base.drag_nm')},
        'v_blend_rad_s': float(spec.get('v_blend_rad_s', 0.01)),      # output side
        'velocity_tau_s': float(spec.get('velocity_tau_s', 0.01)),
        'gate_rad_s': tuple(float(x) for x in spec.get('gate_rad_s', (0.5, 1.5))),  # motor side
        't_lead_s': float(spec.get('t_lead_s', 0.0015)),
        'ripple_fade_rad_s': (tuple(float(x) for x in spec['ripple_fade_rad_s'])
                              if spec.get('ripple_fade_rad_s') is not None else None),
        'max_current_a': float(spec.get('max_current_a', 30.0)),
        'max_output_nm': float(spec.get('max_output_nm', 30.0)),
        'load_ff_ratio': float(spec.get('load_ff_ratio', 1.0)),
        'speed_table': None,
        'harmonics': [],
        'profiles': {},
    }
    if out['v_blend_rad_s'] <= 0 or out['velocity_tau_s'] < 0:
        raise ValueError('dut_compensation: v_blend_rad_s must be > 0, velocity_tau_s >= 0')
    lo, hi = out['gate_rad_s']
    if not 0 <= lo < hi:
        raise ValueError('dut_compensation.gate_rad_s must be [lo, hi] with 0 <= lo < hi')
    fade = out['ripple_fade_rad_s']
    if fade is not None and not (len(fade) == 2 and 0 <= fade[0] < fade[1]):
        raise ValueError('dut_compensation.ripple_fade_rad_s must be [full, zero] with 0 <= full < zero')
    for k in ('K_f', 'K_b'):
        if min(out['base'][k]) <= 0:
            raise ValueError(f'dut_compensation.base.{k} must be > 0')

    st = spec.get('speed_table')
    if st:
        speeds = [float(v) for v in st['speed_rad_s']]
        if speeds != sorted(speeds) or len(speeds) < 2:
            raise ValueError('dut_compensation.speed_table.speed_rad_s must be ascending, >= 2 points')
        table = {'speed_rad_s': speeds}
        for k in ('K_f', 'K_b', 'drag_nm'):
            rows = st[k]
            if len(rows) != len(speeds):
                raise ValueError(f'dut_compensation.speed_table.{k} needs one entry per speed')
            table[k] = [_pair(r, f'speed_table.{k}') for r in rows]
        out['speed_table'] = table

    ql = spec.get('quadrant_lines')
    out['quadrant_lines'] = None
    if ql:
        speeds = [float(v) for v in ql['speed_rad_s']]
        if speeds != sorted(speeds) or not speeds:
            raise ValueError('dut_compensation.quadrant_lines.speed_rad_s must be ascending, >= 1 point')
        lines = {'speed_rad_s': speeds}
        for k in ('K_f', 'D_f', 'K_b', 'D_b'):
            rows = ql[k]
            if len(rows) != len(speeds):
                raise ValueError(f'dut_compensation.quadrant_lines.{k} needs one entry per speed')
            lines[k] = [_pair(r, f'quadrant_lines.{k}') for r in rows]
            if k.startswith('K') and min(min(r) for r in lines[k]) <= 0:
                raise ValueError(f'dut_compensation.quadrant_lines.{k} must be > 0')
        out['quadrant_lines'] = lines

    groups = set()
    for h in spec.get('harmonics') or []:
        cells = {}
        for name, c in (h.get('cells') or {}).items():
            if name not in ('pos_forward', 'pos_backdrive', 'neg_forward', 'neg_backdrive'):
                raise ValueError(f'dut_compensation harmonic cell {name!r}: use '
                                 '{pos,neg}_{forward,backdrive}')
            cells[name] = (float(c.get('a0', 0.0)), float(c.get('a1', 0.0)), float(c['phase']))
        out['harmonics'].append({'group': str(h['group']), 'order': int(h['order']), 'cells': cells})
        groups.add(str(h['group']))

    known = set(TERMS_BUILTIN) | groups
    for name, terms in (spec.get('profiles') or {}).items():
        terms = list(terms or [])
        bad = [t for t in terms if t not in known]
        if bad:
            raise ValueError(f'dut_compensation profile {name!r}: unknown terms {bad}; '
                             f'known: {sorted(known)}')
        if 'qdrag' in terms and out['quadrant_lines'] is None:
            raise ValueError(f'dut_compensation profile {name!r} uses qdrag, but there is '
                             'no quadrant_lines table')
        out['profiles'][str(name)] = tuple(terms)
    return out


def _interp(x, xs, ys):
    """Piecewise-linear, clamped at both ends."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            u = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + u * (ys[i] - ys[i - 1])
    return ys[-1]


class DutCompensation:
    """Stateful only for the velocity estimate. Call update() once per cycle."""

    def __init__(self, cfg, gear_ratio, dt):
        self.cfg = cfg
        self.G = abs(float(gear_ratio))
        self.dt = float(dt)
        self._theta_prev = None
        self.omega_m = 0.0
        tau = cfg['velocity_tau_s']
        self._alpha = 1.0 if tau <= 0 else self.dt / (tau + self.dt)
        self.profile_index = {name: i for i, name in enumerate(sorted(cfg['profiles']))}
        self.last = {}

    def reset(self):
        self._theta_prev = None
        self.omega_m = 0.0

    def terms(self, profile):
        """Enabled term names for a profile name (KeyError if unknown)."""
        return set(self.cfg['profiles'][profile])

    def _velocity(self, theta):
        if theta is None or not math.isfinite(theta):
            return self.omega_m
        if self._theta_prev is not None:
            raw = (theta - self._theta_prev) / self.dt
            self.omega_m += self._alpha * (raw - self.omega_m)
        self._theta_prev = theta
        return self.omega_m

    def _branch_params(self, terms, dir_idx, w_out_abs):
        """(K_f, D_f, K_b, D_b) for this cycle. Without qdrag D_f = D_b."""
        ql = self.cfg['quadrant_lines']
        if 'qdrag' in terms and ql is not None:
            xs = ql['speed_rad_s']
            x = w_out_abs if 'speed' in terms else xs[0]
            return tuple(_interp(x, xs, [row[dir_idx] for row in ql[k]])
                         for k in ('K_f', 'D_f', 'K_b', 'D_b'))
        b = self.cfg['base']
        st = self.cfg['speed_table']
        if 'speed' in terms and st is not None:
            xs = st['speed_rad_s']
            K_f, K_b, D = (_interp(w_out_abs, xs, [row[dir_idx] for row in st[k]])
                           for k in ('K_f', 'K_b', 'drag_nm'))
        else:
            K_f, K_b, D = b['K_f'][dir_idx], b['K_b'][dir_idx], b['drag_nm'][dir_idx]
        return K_f, D, K_b, D

    @staticmethod
    def _invert(tau, s, K_f, D_f, K_b, D_b):
        """Current giving output torque tau on the branch model.

        Forward (s*I > 0): T = K_f*I - D_f*s. Backdrive: T = K_b*I - D_b*s.
        With D_f == D_b the two meet at I = 0 and this is the plain branch
        inverse. With D_f < D_b they leave a gap of targets no current reaches
        (I = 0 there, continuous with both sides); with D_f > D_b they overlap,
        and the current is interpolated across the overlap so it stays
        continuous and monotone in tau.
        """
        I_f = (tau + D_f * s) / K_f
        I_b = (tau + D_b * s) / K_b
        fwd_ok = s * I_f > 0
        bd_ok = s * I_b < 0
        if fwd_ok and not bd_ok:
            return I_f
        if bd_ok and not fwd_ok:
            return I_b
        if not fwd_ok and not bd_ok:
            return 0.0
        t_f, t_b = -D_f * s, -D_b * s          # where each branch reaches I = 0
        u = (tau - t_f) / (t_b - t_f)
        return (1 - u) * (t_f + D_b * s) / K_b + u * (t_b + D_f * s) / K_f

    def ripple(self, terms, theta, w_m, tau, cell):
        """Predicted angle-locked output torque (Nm) for the enabled groups."""
        th = theta + self.cfg['phase_offset_rad'] + w_m * self.cfg['t_lead_s']
        r = 0.0
        for h in self.cfg['harmonics']:
            if h['group'] not in terms:
                continue
            c = h['cells'].get(cell)
            if c is None:
                continue
            a0, a1, phi = c
            r += (a0 + a1 * abs(tau)) * math.cos(h['order'] * th + phi)
        return r

    def gate(self, w_m):
        lo, hi = self.cfg['gate_rad_s']
        g = min(1.0, max(0.0, (abs(w_m) - lo) / (hi - lo)))
        fade = self.cfg['ripple_fade_rad_s']
        if fade is not None:
            w_out = abs(w_m) / self.G
            g *= min(1.0, max(0.0, (fade[1] - w_out) / (fade[1] - fade[0])))
        return g

    def update(self, tau, theta, profile, omega_m=None):
        """DUT current (A, command frame) for target output torque `tau`.

        theta: DUT motor angle (rad) this cycle. omega_m: optional motor speed
        to use instead of the internal estimate (the estimate still tracks).
        Returns the current; the breakdown is left in self.last for logging.
        """
        w_est = self._velocity(theta)
        w_m = w_est if omega_m is None else float(omega_m)
        terms = self.terms(profile)
        tau = max(-self.cfg['max_output_nm'], min(self.cfg['max_output_nm'], float(tau)))
        w_out = w_m / self.G
        s = math.tanh(w_out / self.cfg['v_blend_rad_s'])
        dir_idx = 0 if w_m >= 0 else 1
        use_base = 'base' in terms or 'speed' in terms

        if use_base:
            K_f, D_f, K_b, D_b = self._branch_params(terms, dir_idx, abs(w_out))
        else:
            K_f = K_b = self.cfg['naive_nm_per_a']
            D_f = D_b = 0.0

        # Quadrant from the target before ripple, so a ripple swing near the
        # kink cannot flip the cell it is looked up in.
        forward = s * self._invert(tau, s, K_f, D_f, K_b, D_b) > 0
        cell = ('pos_' if dir_idx == 0 else 'neg_') + ('forward' if forward else 'backdrive')
        g = self.gate(w_m)
        R = g * self.ripple(terms, theta if theta is not None else 0.0, w_m, tau, cell) if g > 0 else 0.0

        current = self._invert(tau - R, s, K_f, D_f, K_b, D_b)
        cap = self.cfg['max_current_a']
        current = max(-cap, min(cap, current))
        self.last = {'tau': tau, 'current': current, 'ripple_nm': R, 'omega_m': w_m,
                     'quadrant': 1.0 if forward else -1.0, 'gate': g,
                     'profile': float(self.profile_index.get(profile, -1))}
        return current
