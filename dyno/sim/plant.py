"""Single-DOF mechanical + thermal model of the dyno shaft for the sim sandbox.

One rotational state (the output shaft): drives push torques in (already
converted to output-frame by their behaviors), the plant integrates
J*d(omega)/dt = sum(tau) - friction, and sensor behaviors read named
quantities back out. Constants can be tuned in dyno/sim/sim_params.yaml
(optional) without touching code.

Friction has two regimes. Sliding is the usual viscous + Coulomb pair. At rest
the shaft STICKS: it is pinned at exactly zero velocity until the applied
torque exceeds `stiction_nm`, then breaks free against the smaller Coulomb
term and lurches. A plain tanh-Coulomb model has no such threshold -- the shaft
creeps under any torque at all -- so it cannot be used to exercise anything
that measures or detects breakaway (test_manager.RampBreak, the stiction
analyses). Set `stiction_nm: 0` to get the old creep-everywhere behavior back.
"""
import json
import math
import os

import yaml

DEFAULTS = {
    'inertia_kgm2': 25.0,      # reflected inertia at the output shaft
    'viscous_nms': 1.5,        # viscous friction, Nm per rad/s
    'coulomb_nm': 2.0,         # coulomb (kinetic) friction magnitude
    # Static friction. Meaningful only at or above coulomb_nm -- below it the
    # shaft would break free into a larger resisting torque and immediately
    # re-stick. 0 disables the stick model entirely.
    'stiction_nm': 3.0,
    'stick_band_radps': 0.02,  # |omega| under this counts as at rest
    'ambient_c': 25.0,
    'thermal_tau_s': 120.0,    # stator temp first-order time constant
    'heat_c_per_nm': 0.15,     # steady-state stator rise per Nm of load torque
    'adc_noise_counts': 4000,  # gaussian noise on unmapped/mapped ADC channels
    'initial_theta_rad': 0.0,  # shaft angle at power-up
    # Rubber endstops on the output shaft at theta = endstop_centre_rad +/-
    # endstop_half_span_rad (None = no endstops). A one-sided spring-damper past
    # each stop. Their reaction torque reads on the output torque cell, which
    # sits between the absorber and the stops.
    'endstop_half_span_rad': None,
    'endstop_centre_rad': 0.0,
    'endstop_stiffness_nm_per_rad': 300.0,
    'endstop_damping_nms': 20.0,
}


def load_sim_params():
    """DEFAULTS < sim_params.yaml top level < its `modes: {<DYNO_SIM mode>: }`
    block < JSON in DYNO_SIM_PARAMS (for one-off test cases)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim_params.yaml')
    params = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path, 'r') as f:
            loaded = yaml.safe_load(f) or {}
        modes = loaded.pop('modes', None) or {}
        params.update(loaded)
        params.update(modes.get(os.environ.get('DYNO_SIM'), None) or {})
    if os.environ.get('DYNO_SIM_PARAMS'):
        params.update(json.loads(os.environ['DYNO_SIM_PARAMS']))
    return params


class Plant:
    def __init__(self, params=None):
        p = dict(DEFAULTS)
        p.update(params or {})
        self.J = p['inertia_kgm2']
        self.b = p['viscous_nms']
        self.coulomb = p['coulomb_nm']
        self.stiction = p['stiction_nm']
        self.stick_band = p['stick_band_radps']
        self.ambient_c = p['ambient_c']
        self.thermal_tau = p['thermal_tau_s']
        self.heat_c_per_nm = p['heat_c_per_nm']

        self.endstop = p['endstop_half_span_rad']
        self.endstop_centre = float(p['endstop_centre_rad'])
        self.endstop_k = p['endstop_stiffness_nm_per_rad']
        self.endstop_c = p['endstop_damping_nms']
        self.tau_endstop = 0.0

        self.omega = 0.0  # output shaft, rad/s
        self.theta = float(p['initial_theta_rad'])  # output shaft, rad
        self.stator_temp_c = self.ambient_c

        self._drive_torques = {}   # name -> output-frame Nm
        self.tau_dut_out = 0.0     # what the input side delivers at the output
        self.tau_dut_motor = 0.0   # input side, motor frame (input torque cell)

    def set_drive_torque(self, name, tau_out_frame, tau_motor_frame=None,
                         is_input_side=False):
        self._drive_torques[name] = tau_out_frame
        if is_input_side:
            self.tau_dut_out = tau_out_frame
            if tau_motor_frame is not None:
                self.tau_dut_motor = tau_motor_frame

    def _endstop_torque(self):
        if self.endstop is None:
            return 0.0
        x = self.theta - self.endstop_centre
        over = abs(x) - abs(self.endstop)
        if over <= 0:
            return 0.0
        push = self.endstop_k * over
        # Damping only resists motion into the stop; the spring never pulls.
        if self.omega * x > 0:
            push += self.endstop_c * abs(self.omega)
        return -math.copysign(push, x)

    def step(self, dt):
        self.tau_endstop = self._endstop_torque()
        tau_sum = sum(self._drive_torques.values()) + self.tau_endstop
        at_rest = self.stiction > 0 and abs(self.omega) <= self.stick_band
        if at_rest and abs(tau_sum) <= self.stiction:
            # Stuck. Velocity is pinned to zero outright rather than merely
            # damped hard: a stiff damper still creeps, and creep is exactly
            # what must not happen here -- the measurement this model exists to
            # support is the torque at FIRST motion, so any motion below the
            # breakaway threshold is a false reading, not a small error.
            self.omega = 0.0
        else:
            if at_rest:
                # Breaking free: past the static threshold only the kinetic
                # term resists, and that step down from stiction to coulomb is
                # what makes the shaft lurch instead of easing into motion.
                friction = math.copysign(self.coulomb, tau_sum)
            else:
                friction = (self.b * self.omega
                            + self.coulomb * math.tanh(self.omega / 0.05))
            self.omega += (tau_sum - friction) / self.J * dt
            self.theta += self.omega * dt

        # Stator temperature: first-order approach to ambient + load-dependent rise
        target = self.ambient_c + self.heat_c_per_nm * abs(self.tau_dut_out)
        self.stator_temp_c += (target - self.stator_temp_c) * dt / self.thermal_tau

    def quantity(self, sensor_name):
        """Named engineering quantities that sensor behaviors can read.
        Unknown names read 0 (noise-only channel)."""
        if sensor_name in ('load_torque', 'output_torque'):
            return self.tau_dut_out + self.tau_endstop
        if sensor_name == 'input_torque':
            return self.tau_dut_motor
        if sensor_name == 'load_stator_temp':
            return self.stator_temp_c
        return 0.0
