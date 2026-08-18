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
}


def load_sim_params():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sim_params.yaml')
    params = dict(DEFAULTS)
    if os.path.exists(path):
        with open(path, 'r') as f:
            params.update(yaml.safe_load(f) or {})
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

        self.omega = 0.0  # output shaft, rad/s
        self.theta = 0.0  # output shaft, rad
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

    def step(self, dt):
        tau_sum = sum(self._drive_torques.values())
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
            return self.tau_dut_out
        if sensor_name == 'input_torque':
            return self.tau_dut_motor
        if sensor_name == 'load_stator_temp':
            return self.stator_temp_c
        return 0.0
