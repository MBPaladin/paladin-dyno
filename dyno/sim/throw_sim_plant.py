"""Stand-in plant and behavior factory shared by the throw sim tests."""
import sys

from dyno.src.test_manager import Throw

HALF_WINDOW = 1.5
RATIO = 43.88
LIMITS = {'input': {'velocity': 500.0, 'acceleration': 7500.0, 'torque': 15.0,
                    'gear_ratio': -RATIO},
          'output': {'velocity': 60.0, 'acceleration': 2000.0, 'torque': 200.0},
          'coupled': True}


class Plant:
    """Output shaft with a velocity loop: slews toward the commanded speed,
    braking at `true_decel`, speeding up at `true_accel`. `gain` > 1 makes the
    shaft overshoot its command the way velocity_mode_6 did (3.0 -> 4.2)."""

    def __init__(self, drive='output', input_sign=1, centre=2.0, start_rel=0.0,
                 true_decel=70.0, true_accel=1000.0, gain=1.0, dt=0.001):
        self.drive, self.sign, self.centre = drive, input_sign, centre
        self.pos = centre + start_rel
        self.v, self.true_decel, self.true_accel = 0.0, true_decel, true_accel
        self.gain, self.dt = gain, dt
        self.peak = abs(start_rel)
        self.frozen = False

    def step(self, cmd):
        if self.frozen:
            return
        if self.drive == 'output':
            target = cmd['output_command']
        else:
            target = cmd['input_command'] * self.sign / RATIO
        target *= self.gain
        braking = abs(target) < abs(self.v) or target * self.v < 0
        rate = (self.true_decel if braking else self.true_accel) * self.dt
        self.v += max(-rate, min(rate, target - self.v))
        self.pos += self.v * self.dt
        self.peak = max(self.peak, abs(self.pos - self.centre))

    def reader(self, centre='plant', input_sign='plant'):
        def read():
            return {'torque': {}, 'velocity': {'input': 0.0, 'output': self.v},
                    'position': {'input': 0.0, 'output': self.pos},
                    'centre': self.centre if centre == 'plant' else centre,
                    'input_sign': self.sign if input_sign == 'plant' else input_sign,
                    'ratio': RATIO}
        return read


def make(plant=None, reader=None, rig=None, **settings):
    s = {'drive_motor': 'output', 'speed_rad_s': 3.0, 'target_rad': 1.0,
         'ramp_accel': 100.0, 'n_throws': 4, 'settle_s': 0.05}
    s.update(settings)
    rig = rig or {'window_enabled': True, 'half_window': HALF_WINDOW,
                  'window_decel': 70.0, 'stop_decel': {'output': 70.0, 'input': 70.0}}

    class T(Throw):
        def _load_rig(self):
            return rig

    if reader is None and plant is not None:
        reader = plant.reader()
    return T({'id': 'TH', 'settings': s}, 'inhouse_archimedes', LIMITS, reader)


def run(beh, plant=None, max_cycles=400000):
    cmds = []
    for c in beh.commands():
        cmds.append(c)
        if plant is not None:
            plant.step(c)
        if len(cmds) >= max_cycles:
            break
    return cmds
