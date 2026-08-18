import math

import yaml
import numpy as np
import pandas as pd
import time
from deployment import dyno_paths
# test_builder imports nothing from this package, so this stays acyclic. The
# shared check lives there because the builder must run without numpy/pandas.
from dyno.src.test_builder import design_multisine, reaction_torque_issue

# iterator that yields loops of behaviors
def loop_iterator(loop_definition):
    for i in range(loop_definition['settings']['loop_count']):
        for behavior in loop_definition['behaviors']:
            if not behavior['type'] == 'loop':
                yield behavior
            else:
                yield from loop_iterator(behavior)

# iterator that yield behaviors from the test definition
def behavior_iterator(test_config):
    for behavior in test_config['behaviors']:
        if behavior['type'] == 'loop':
            yield from loop_iterator(behavior)
        else:
            yield behavior

# iterator that yields from a grid
def grid_iterator(list1, list2):
    for i in list1:
        for j in list2:
            yield i, j

# class that iterpolates over perscribed torque / velocity / position traces
class TestTrace:
    def __init__(self, parameters, mode, limits):
        self.mode = mode
        self.parameters = parameters
        self.limits = limits
        self.settings = self.parameters['settings']
        self._load_trace() # We want to do all loading of files in the __init__ as that happens before the test definition is placed inside of the control thread
        self.log_id_base = self.parameters['id']+'-RUN' # self.run gets appended to this string to capture how many times a behavior has run within a log
        self.run = 0

    def _load_trace(self):
        f_name = self.settings['trace_file']
        self.input_mode = self.settings['input_motor']['control_mode']
        self.output_mode = self.settings['output_motor']['control_mode']
        # Ensure the 'tests/traces/' directory exists for your CSVs
        self.trace = pd.read_csv(f"{dyno_paths.dyno_test_directory}/traces/{f_name}")

        # Apply scaling to the trace, if present
        if self.settings.get('use_relative_command', False):
            print('\nApplying relative scaling')
            # Relative 1.0 means "this motor's own rating", not the weaker of
            # the two -- self.limits is keyed by motor first (see
            # dyno_controller._get_limits).
            if self.input_mode in ['velocity', 'torque']:
                self.trace[f"input_motor_{self.input_mode}"] *= self.limits['input'][self.input_mode]
                print('\tScaling input trace by ',self.limits['input'][self.input_mode])

                if 'scale_override' in self.settings['input_motor'].keys():
                    self.trace[f"input_motor_{self.input_mode}"] *= min(1, abs(self.settings['input_motor']['scale_override']))

            if self.output_mode in ['velocity', 'torque']:
                self.trace[f"output_motor_{self.output_mode}"] *= self.limits['output'][self.output_mode]
                print('\tScaling output trace by ',self.limits['output'][self.output_mode])


                if 'scale_override' in self.settings['output_motor'].keys():
                    self.trace[f"output_motor_{self.output_mode}"] *= min(1, abs(self.settings['output_motor']['scale_override']))

        # Ensure right modes are present
        assert len(self.trace.keys()) == 3, 'Trace csv file must contain 3 keys'
        assert 'time' in self.trace.keys(), '"time" trace must be present in csv file'

        for trace_key in self.trace.keys():
            assert any(word in trace_key for word in ['time', 'torque', 'velocity', 'position']), ' trace key:'+trace_key+' is not one o[time, torque, velocity, position]'

        assert any(self.input_mode in key for key in self.trace.keys()), 'Input motor control mode: '+self.input_mode+' not found in trace keys: '+str(self.trace.keys()) # Convert keys to string for error message
        assert any(self.output_mode in key for key in self.trace.keys()), 'Output motor control mode: '+self.output_mode+' not found in trace keys: '+str(self.trace.keys())
        assert set([self.output_mode, self.input_mode]) != set(['velocity', 'position']), 'Velocity and position control modes should not be used at the same time'
        assert set([self.output_mode, self.input_mode]) != set(['position', 'position']), 'Posotion and position control modes should not be used at the same time'
        assert set([self.output_mode, self.input_mode]) != set(['velocity', 'velocity']), 'Velocity and velocity control modes should not be used at the same time'
        #assert self.output_mode != self.input_mode, 'Same control mode assigned to both motors: ' + self.output_mode

        # check that time increases and that all traces start at 0
        assert (self.trace['time'][:] == np.sort(self.trace['time'][:])).all(), 'Time trace must be increasing'
        for trace in self.trace.keys():
            assert self.trace[trace][0] == 0, trace+' trace must start at 0'


        d_dt = np.array(self.trace['time'].iloc[1:] - np.array(self.trace['time'].iloc[:-1])) # time delta between points

        rates = {
            'input_motor': (self.trace[f"input_motor_{self.input_mode}"][1:] - np.array(self.trace[f"input_motor_{self.input_mode}"][:-1])) / d_dt,
            'output_motor': (self.trace[f"output_motor_{self.output_mode}"][1:] - np.array(self.trace[f"output_motor_{self.output_mode}"][:-1])) / d_dt
            }
        
        for motor in ['input', 'output']:
            motor_key = f'{motor}_motor'
            # Each motor is held to its own ratings, not the weaker of the pair.
            limits = self.limits[motor]
            if self.settings[motor_key]['control_mode'] == 'torque':
                max_abs_test_torque = max(abs(self.trace[f"{motor_key}_torque"][:]))
                limit_val = limits['torque']
                assert max_abs_test_torque <= limit_val, \
                    (f'Max {motor_key} trace torque of {max_abs_test_torque} exceeds '
                     f'its limit of {limit_val}')
                assert self.trace[f"{motor_key}_torque"][len(self.trace['time'])-1] == 0, 'Torque trace must end at 0'
                assert max(abs(rates[motor_key])) <= limits['rotatum'], motor_key+' trace rotatum (d_torque/dt) of '+str(max(abs(rates[motor_key])))+' Nm/s exceeds its limit of '+str(limits['rotatum'])
                if motor == 'output':
                    reaction = reaction_torque_issue(max_abs_test_torque, self.limits)
                    assert reaction is None, 'output_motor trace torque '+str(reaction)


            if self.settings[motor_key]['control_mode'] == 'velocity':
                max_abs_test_velocity = max(abs(self.trace[f"{motor_key}_velocity"][:]))
                limit_val = limits['velocity']
                #assert max_abs_test_velocity <= limit_val, f'Max trace velocity of {max_abs_test_velocity} exceeds system limit of {limit_val}'
                assert max(abs(rates[motor_key])) <= limits['acceleration'], motor_key+' trace acceleration of '+str(max(abs(rates[motor_key])) )+' exceeds its limit of '+str(limits['acceleration'])
                assert self.trace[f"{motor_key}_velocity"][len(self.trace['time'])-1] == 0, 'Velocity trace must end at 0'

            if self.settings[motor_key]['control_mode'] == 'position':
                assert max(abs(rates[motor_key]))  <= limits['velocity'], motor_key+' trace velocity of '+str(max(abs(rates[motor_key])))+' exceeds its limit of '+str(limits['velocity'])

                # Corner check: a slope change between adjacent keyframes is a
                # velocity step the position loop absorbs in ~one cycle -- an
                # acceleration impulse that spikes torque (inertia * dv/dt).
                # Flag corners bigger than the accel limit could produce in
                # CORNER_DT. This is a warning, not an assert, because several
                # long-standing traces (e.g. the cogging trapezoids discretized
                # at 0.1 s) technically exceed it; the test builder emits
                # accel-blended traces that pass cleanly.
                corner_dt = 0.05  # [s] matches test_builder.CORNER_DT
                slopes = np.asarray(rates[motor_key], dtype=float)
                corner_steps = np.abs(np.diff(slopes))
                step_limit = limits['acceleration'] * corner_dt
                if corner_steps.size and corner_steps.max() > step_limit:
                    worst = int(np.argmax(corner_steps))
                    print(f"WARNING: {motor_key} position trace has a corner velocity step of "
                          f"{corner_steps.max():.3f} rad/s at t={self.trace['time'].iloc[worst+1]:.2f}s "
                          f"(> {step_limit:.3f} = acceleration limit x {corner_dt}s). "
                          "Sharp corners cause torque spikes; use accel-limited blends "
                          "(the test builder applies these automatically).")


            # if self.settings[motor_key]['control_mode'] == 'position':
            #     d_dt = self.trace['time'][1:] - self.trace['time'][:len(self.trace['time'])-2] # time delta between points

            #     velocity = (self.trace['position'][1:] - self.trace['position'][:len(self.trace['time'])-2]) / d_dt

            #     assert max(abs(velocity)) < self.limits['acceleration'], 'Trace acceleration exceeds system limits'

        # Cache hot-path columns as plain numpy arrays. commands() runs at 1kHz,
        # and pandas Series indexing per cycle was the dominant source of jitter.
        self._t_arr = self.trace['time'].to_numpy()
        self._in_arr = self.trace[f"input_motor_{self.input_mode}"].to_numpy()
        self._out_arr = self.trace[f"output_motor_{self.output_mode}"].to_numpy()
        self._trace_max_time = float(self._t_arr[-1])
        self._in_last = float(self._in_arr[-1])
        self._out_last = float(self._out_arr[-1])



    # method that yields out commands to the test manager. before yielding the last command the class should be in a state from which it can be run again.
    def commands(self):
        # Yield the initial command to set op modes. This command should be yielded before the timer is started, as mode changes may take some time to process

        yield {
            'input_mode': self.input_mode,
            'output_mode': self.output_mode,
            'input_command': 0,
            'output_command': 0
        }


        # Use perf_counter (monotonic) instead of time.time(); cached arrays avoid pandas in the hot path.
        start_time = time.perf_counter()
        while True:
            current_time_in_trace = time.perf_counter() - start_time
            if current_time_in_trace < self._trace_max_time:
                command = {
                    'input_mode': self.input_mode,
                    'output_mode': self.output_mode,
                    'input_command':  np.interp(current_time_in_trace, self._t_arr, self._in_arr),
                    'output_command': np.interp(current_time_in_trace, self._t_arr, self._out_arr),
                }
                command['log_flag'] = self.log_id_base + str(self.run)
                yield command
            else:
                for i in range(250): # hold constant cmd to stabilize system
                    yield {
                        'input_mode': self.input_mode,
                        'output_mode': self.output_mode,
                        'input_command': self._in_last,
                        'output_command': self._out_last,
                    }

                if (self.input_mode == 'position' or self.output_mode == 'position') and not (self.trace[f'{motor_key}_position'].iloc[-1] == 0 for motor_key in ['input_motor', 'output_motor']): # toggle output mode so that the next time a motor switches to position mode it reset
                    yield {
                        'input_mode': 'torque',
                        'output_mode': 'torque',
                        'input_command': 0,
                        'output_command': 0
                    }
                self.run += 1
                return # Exits the generator

class GridSearch:
    def __init__(self, parameters, mode, limits):
        self.parameters = parameters
        self.mode = mode
        self.limits = limits
        self.settings = self.parameters['settings']
        self.log_id_base = self.parameters['id']+'-RUN'

        self.input_mode = self.settings['input_motor']['control_mode']
        self.output_mode = self.settings['output_motor']['control_mode']

        # Apply scaling to the trace, if present
        if self.settings.get('use_relative_command', False):
            if self.input_mode in ['velocity', 'torque']:
                self.settings['input_motor']['command_list'] = [command * self.limits['input'][self.input_mode] for command in self.settings['input_motor']['command_list']]

            if self.output_mode in ['velocity', 'torque']:
                self.settings['output_motor']['command_list'] = [command * self.limits['output'][self.output_mode] for command in self.settings['output_motor']['command_list']]

        with open(f"{dyno_paths.dyno_config_directory}/master_config.yaml", 'r') as f:
            master_params = yaml.safe_load(f)

        # Each axis ramps at a fraction of the rating of the motor that drives
        # it; the asserts below establish that exactly one motor holds each mode.
        self.rate_limits = {}
        for motor in ['input', 'output']:
            control_mode = self.settings[f'{motor}_motor']['control_mode']
            if control_mode == 'torque':
                self.rate_limits['torque'] = self.settings['transition_rate'] * self.limits[motor]['rotatum']
            if control_mode == 'velocity':
                self.rate_limits['velocity'] = self.settings['transition_rate'] * self.limits[motor]['acceleration']

        self.timestep = master_params['cycle_time_us'] / 1e6 # Convert microseconds to seconds

        self.sweep_axes = {
            self.settings[key]['control_mode']: self.settings[key]['command_list']
            for key in ['input_motor', 'output_motor']
        }
        
        # Continuous-torque rating used by the cooldown logic below. Builder
        # generated tests bake the resolved rating into settings; hand-written
        # yamls without it fall back to the historical hardcodes.
        if 'continuous_torque' in self.settings:
            self.t_limit = abs(float(self.settings['continuous_torque']))
        elif self.input_mode == 'torque':
            self.t_limit = 4
        elif self.output_mode == 'torque':
            self.t_limit = 110

        for motor in ['input', 'output']:
            motor_key = f'{motor}_motor'
            peak = max(abs(np.array(self.settings[motor_key]['command_list'])))
            if self.settings[motor_key]['control_mode'] == 'torque':
                assert peak <= self.limits[motor]['torque'], 'Max commanded '+motor_key+' torque of '+str(peak)+' exceeds its limit of '+str(self.limits[motor]['torque'])
                if motor == 'output':
                    reaction = reaction_torque_issue(peak, self.limits)
                    assert reaction is None, 'Commanded output_motor torque '+str(reaction)


            if self.settings[motor_key]['control_mode'] == 'velocity':
                assert peak <= self.limits[motor]['velocity'], 'Max commanded '+motor_key+' velocity of '+str(peak)+' exceeds its limit of '+str(self.limits[motor]['velocity'])

        assert set(self.settings['loop_order']) == set([self.input_mode, self.output_mode]), 'Provided input/output control modes do not match with given loop order keys'
        assert set([self.output_mode, self.input_mode]) == set(['velocity', 'torque']), 'The grid search behavior should only be used with velocity and torque control modes'
        assert self.output_mode != self.input_mode, 'Same control mode is assigned to both motors: ' + self.output_mode
        assert self.settings['settle_time_s'] >= 0, 'Perscribed settle time of '+str(self.settings['settle_time_s'])+' must be greater than or equal to 0'
        assert self.settings['transition_rate'] > 0, 'Perscribed transition rate of '+str(self.settings['transition_rate'])+' must be greater than 0'
        assert self.settings['transition_rate'] <= 1, 'Perscribed transition rate of '+str(self.settings['transition_rate'])+' must not exceed 1.0'
        assert self.settings['duration_per_point_s'] > 0, 'Perscribed duration of '+str(self.settings['duration_per_point_s'])+' must be greater than 0.0'

        # _setpoint_generator, _active_setpoint, _state, etc. will be initialized
        # when .commands() is called.
        self._setpoint_generator = None
        self._active_setpoint = None
        self._state = {'velocity': 0.0, 'torque': 0.0}
        self._point_ct = 0
        self.run = 0
        self._transition_complete_time = None

    def ramp(self, setpoint):
        starting_velocity = self._state['velocity']
        starting_torque = self._state['torque']

        velocity_change = setpoint['velocity'] - self._state['velocity']
        torque_change = setpoint['torque'] - self._state['torque']

        time_to_change_velocity = abs(velocity_change) / self.rate_limits['velocity']
        time_to_change_torque = abs(torque_change) / self.rate_limits['torque']

        transition_time = max([time_to_change_velocity, time_to_change_torque])

        start_time = time.perf_counter()

        while (time.perf_counter() - start_time) <= transition_time:
            transition_ratio = (time.perf_counter() - start_time) / transition_time
            self._state['velocity'] = (1 - transition_ratio)*starting_velocity + transition_ratio*setpoint['velocity']
            self._state['torque'] = (1 - transition_ratio)*starting_torque + transition_ratio*setpoint['torque']

            yield {
                'input_mode': self.input_mode,
                'output_mode': self.output_mode,
                'input_command': self._state[self.input_mode],
                'output_command': self._state[self.output_mode]
            }
        
        yield {
            'input_mode': self.input_mode,
            'output_mode': self.output_mode,
            'input_command': setpoint[self.input_mode],
            'output_command': setpoint[self.output_mode]
        }

    def hold(self, setpoint, duration, flag = None):
        start = time.perf_counter()
        while time.perf_counter() - start < duration:
            cmd = {
                'input_mode': self.input_mode,
                'output_mode': self.output_mode,
                'input_command': setpoint[self.input_mode],
                'output_command': setpoint[self.output_mode]
            }
            if not flag == None:
                cmd['log_flag'] = flag
            yield cmd


    def commands(self):
        # --- Initialize state for this specific run of the generator ---
        self._state = {'velocity': 0.0, 'torque': 0.0}
        self._point_ct = 0
        self._transition_complete_time = None
        
        # yield initial command before timers are started
        yield {
            'input_mode': self.input_mode,
            'output_mode': self.output_mode,
            'input_command': self._state[self.input_mode],
            'output_command': self._state[self.output_mode]
        }

        # configure setpoint generator
        self._setpoint_generator = grid_iterator(
            self.sweep_axes[self.settings['loop_order'][0]],
            self.sweep_axes[self.settings['loop_order'][1]]
        )

        for raw_setpoint in self._setpoint_generator:
            log_flag = self.log_id_base + str(self.run)+'-SETPOINT'+str(self._point_ct)
            self._point_ct += 1

            self._active_setpoint = {
                self.settings['loop_order'][0]: raw_setpoint[0],
                self.settings['loop_order'][1]: raw_setpoint[1]
            }

            print("")
            print(f"GridSearch setpoint: {self._active_setpoint}")

            # Ramp to setpoint
            yield from self.ramp(self._active_setpoint)

            # settle at setpoint
            yield from self.hold(self._active_setpoint, self.settings['settle_time_s'])

            # hold at setpoint
            yield from self.hold(self._active_setpoint, self.settings['duration_per_point_s'], log_flag)

            # if setpoint is above cont. torque rating, ramp and hold at 0 torque to cool down
            if abs(self._active_setpoint['torque']) > self.t_limit:
                cooldown_time = (self.settings['settle_time_s']+self.settings['settle_time_s']) * abs(self._active_setpoint['torque'] / self.t_limit)**3.5
                self._active_setpoint['torque'] = 0 #override torque setpoint to 0

                yield from self.ramp(self._active_setpoint)

                yield from self.hold(self._active_setpoint, cooldown_time)

            
        # ramp down
        zero_setpoint = {self.settings['loop_order'][i]:0.0 for i in range(2)}

        yield from self.ramp(zero_setpoint)

        yield from self.hold(zero_setpoint, 0.1)
        self.run += 1

class Multisine:
    """Optional test preamble: a silent hold, optionally followed by a periodic
    multisine torque excitation at standstill (behavior type 'preamble').

    Deliberately a separate behavior from TestTrace -- it reads no trace file
    and generates entirely from parameters, so existing tests carry zero
    regression risk. Everything here is SAMPLE-paced (one yield = one control
    cycle), never wall-clock paced: coherent averaging in analysis needs exact
    period boundaries, which perf_counter cannot provide under cycle jitter.
    For the same reason every command carries a monotonic 'sample_index' that
    the controller exposes for logging (log key 'preamble_sample').

    Log flags reuse the grid-search span scheme so analysis granularity='run'
    groups all phases together:
        <id>-RUN<r>-SETPOINT0    the quiet hold
        <id>-RUN<r>-SETPOINT<n>  one excitation block per excited role
    Which span is which phase is left to the data (the quiet span has no
    active command), matching the applies_to-is-a-data-predicate convention.

    Excitation is torque-mode only (velocity mode would put the drive's
    velocity loop inside the measurement, and is gated by the far more
    restrictive acceleration limit). Amplitude is anchored to the noise floor
    measured during the quiet hold -- never to a fraction of a motor rating,
    which would be two orders of magnitude wrong across the two ports.
    """

    # The controller's fixed wiring: DUT always receives input_command, LOAD
    # always receives output_command. Roles map to devices via the bench's
    # `ports:` block; this maps the device on to a command-stream side.
    DEVICE_TO_STREAM = {'DUT': 'input', 'LOAD': 'output'}
    DRIFT_LIMIT_RAD = 2.0       # |position - start| that aborts an excitation
    CEILING_FRAC_CONT = 0.05    # absolute safety stop: 5% of continuous torque
    ROTATUM_MARGIN = 0.9        # keep commanded slew inside 90% of the limit
    RAMP_DOWN_SAMPLES = 250     # matches TestTrace's stabilizing tail
    SIGMA_FALLBACK_NM = 0.1     # used when no sensor_reader (preview / no cell)

    def __init__(self, parameters, mode, limits, sensor_reader=None):
        self.parameters = parameters
        self.settings = parameters['settings']
        self.mode = mode                    # bench name, e.g. 'inhouse'
        self.limits = limits
        self.sensor_reader = sensor_reader
        self.log_id_base = parameters['id'] + '-RUN'
        self.run = 0

        self.preamble_mode = self.settings['mode']
        assert self.preamble_mode in ('hold', 'multisine'), \
            f"preamble mode must be 'hold' or 'multisine', got {self.preamble_mode!r}"
        self.quiet_s = float(self.settings['quiet_s'])
        assert self.quiet_s >= 1.0, 'preamble quiet_s must be >= 1.0 s'

        with open(f"{dyno_paths.dyno_config_directory}/master_config.yaml") as f:
            self.dt = yaml.safe_load(f)['cycle_time_us'] / 1e6
        with open(f"{dyno_paths.dyno_config_directory}/{mode}_dyno_config.yaml") as f:
            bench = yaml.safe_load(f)

        # Continuous-torque ratings per stream side, for the absolute ceiling.
        self._cont_torque = {'input': 4.0, 'output': 110.0}  # legacy fallbacks
        for slave in bench.get('expected_slave_layout', []):
            side = self.DEVICE_TO_STREAM.get(slave.get('name'))
            value = (slave.get('params', {}).get('motor_limits', {})
                     .get('continuous_torque'))
            if side and value is not None:
                self._cont_torque[side] = abs(float(value))

        # Resolve the requested role(s) to targets through the bench's ports.
        ports = bench.get('ports') or [
            {'role': side, 'device': dev, 'cell': None}
            for dev, side in self.DEVICE_TO_STREAM.items()]
        commandable = [p for p in ports
                       if p.get('device') in self.DEVICE_TO_STREAM]
        self.targets = []
        if self.preamble_mode == 'multisine':
            requested = self.settings.get('role', 'output')
            chosen = (commandable if requested == 'all' else
                      [p for p in commandable if p.get('role') == requested])
            assert chosen, (f"preamble role {requested!r} matches no commandable "
                            f"port on bench {mode!r} (declared: "
                            f"{[p.get('role') for p in commandable]})")
            for p in chosen:
                self.targets.append({
                    'role': p.get('role'),
                    'side': self.DEVICE_TO_STREAM[p['device']],
                    'cell': p.get('cell'),
                })
            self.design = self.settings.get('design') or design_multisine(
                int(self.settings.get('seed', 1234)), self.dt)
            self.snr_db = {'gentle': 20.0, 'normal': 30.0, 'hard': 40.0}[
                self.settings.get('level', 'normal')]

    # -- waveform synthesis --------------------------------------------------

    def _unit_period(self, weights):
        """One period of the multisine with per-line amplitudes `weights`,
        explicitly DC-nulled. Returns a float array of samples_per_period."""
        N = int(self.design['samples_per_period'])
        n = np.arange(N)
        u = np.zeros(N)
        for line, w in zip(self._excited_lines(), weights):
            k = line['bin']
            u += w * np.cos(2 * np.pi * k * n / N + line['phase'])
        # There is no DC bin by construction, but any numeric DC in a torque
        # command integrates straight into position runaway on a free shaft.
        return u - u.mean()

    def _excited_lines(self):
        return [l for l in self.design['lines'] if not l['detection']]

    def _scaled_period(self, side, sigma):
        """Synthesize and numerically scale one period for `side`.

        The shaping rule (comment kept ON PURPOSE -- this looks arbitrary
        later and gets 'simplified' into a bug): shape to equalize SNR per
        line, which for a roughly white noise floor means FLAT amplitude; then
        scale to fit the torque ceiling and the rotatum budget; tilt toward
        1/f ONLY if the slew budget is what binds, and only if that actually
        buys a better worst-line amplitude. Scaling is numeric (measure the
        synthesized waveform's real peak and peak slew), not analytic.

        Returns (period_array, report_dict).
        """
        N = int(self.design['samples_per_period'])
        n_avg = (self.design['periods'] - self.design['discard_periods'])
        # Per-bin amplitude noise of a rectangular-window FFT of N samples,
        # after coherent averaging across the retained periods.
        floor = 2.0 * sigma / math.sqrt(N) / math.sqrt(n_avg)
        target = floor * 10 ** (self.snr_db / 20.0)   # per-line amplitude [Nm]

        ceiling = self.CEILING_FRAC_CONT * self._cont_torque[side]
        rotatum = self.ROTATUM_MARGIN * self.limits[side]['rotatum']

        lines = self._excited_lines()
        f_lo = lines[0]['freq_hz']

        def scaled(weights):
            u = self._unit_period(weights)
            peak = float(np.max(np.abs(u)))
            # Periodic waveform: include the wrap-around step in the slew.
            slew = float(np.max(np.abs(np.diff(np.append(u, u[0]))))) / self.dt
            s_ceiling = ceiling / peak
            s_rotatum = rotatum / slew
            s = min(target, s_ceiling, s_rotatum)
            bound = ('target SNR' if s == target else
                     'torque ceiling' if s == s_ceiling else 'rotatum')
            worst = s * min(weights)      # weakest line's amplitude
            return u * s, {'scale': s, 'peak_Nm': s * peak,
                           'peak_slew_Nm_s': s * slew, 'bound': bound,
                           'worst_line_Nm': worst}

        flat, flat_rep = scaled([1.0] * len(lines))
        if flat_rep['bound'] != 'rotatum':
            return flat, dict(flat_rep, shaping='flat', target_line_Nm=target,
                              floor_Nm=floor, sigma_Nm=sigma)
        # Slew binds: a 1/f tilt moves amplitude down-band where slew is cheap.
        tilt, tilt_rep = scaled([f_lo / l['freq_hz'] for l in lines])
        if tilt_rep['worst_line_Nm'] > flat_rep['worst_line_Nm']:
            return tilt, dict(tilt_rep, shaping='1/f', target_line_Nm=target,
                              floor_Nm=floor, sigma_Nm=sigma)
        return flat, dict(flat_rep, shaping='flat', target_line_Nm=target,
                          floor_Nm=floor, sigma_Nm=sigma)

    # -- command generation ----------------------------------------------------

    def _cmd(self, side, value, flag=None, sample_index=None):
        cmd = {'input_mode': 'torque', 'output_mode': 'torque',
               'input_command': 0.0, 'output_command': 0.0}
        if side is not None:
            cmd[f'{side}_command'] = float(value)
        if flag is not None:
            cmd['log_flag'] = flag
        if sample_index is not None:
            cmd['sample_index'] = float(sample_index)
        return cmd

    def _read(self):
        if self.sensor_reader is None:
            return None
        try:
            return self.sensor_reader()
        except Exception:
            return None

    def commands(self):
        # Initial command sets both drives to torque mode before timers start.
        yield self._cmd(None, 0.0)

        sample = 0
        setpoint = 0
        flag_base = self.log_id_base + str(self.run)

        # --- quiet hold: ambient floor, and sigma for the amplitude anchor ---
        # Runs FIRST in every mode: the floor is needed both to interpret the
        # excitation and to set its amplitude. Structural, not a user choice.
        n_quiet = int(round(self.quiet_s / self.dt))
        n_settle = min(n_quiet // 3, int(round(0.5 / self.dt)))
        sums, sumsqs, counts = {}, {}, 0
        flag = f'{flag_base}-SETPOINT{setpoint}'
        for i in range(n_quiet):
            yield self._cmd(None, 0.0, flag, sample)
            sample += 1
            if i >= n_settle:
                reading = self._read()
                if reading:
                    for name, value in reading.get('torque', {}).items():
                        sums[name] = sums.get(name, 0.0) + value
                        sumsqs[name] = sumsqs.get(name, 0.0) + value * value
                    counts += 1
        setpoint += 1

        sigmas = {}
        for name in sums:
            mean = sums[name] / counts
            sigmas[name] = math.sqrt(max(sumsqs[name] / counts - mean * mean, 0.0))
        if sigmas:
            print('Preamble: quiet-hold torque cell sigma: '
                  + ', '.join(f'{k}={v:.4f} Nm' for k, v in sorted(sigmas.items())))

        if self.preamble_mode == 'hold':
            self.run += 1
            return

        # --- one excitation block per target role ---------------------------
        for target in self.targets:
            side = target['side']
            sigma = sigmas.get(target['cell'])
            if sigma is None:
                # No reader (preview) or no cell on this shaft: fall back and
                # say so. The fallback is conservative -- amplitudes stay small.
                sigma = self.SIGMA_FALLBACK_NM
                print(f"Preamble: no measured sigma for role '{target['role']}'"
                      f' (cell {target["cell"]!r}); using fallback '
                      f'{sigma:g} Nm')
            period, rep = self._scaled_period(side, sigma)
            print(f"Preamble: role '{target['role']}' ({side} side): "
                  f"{rep['shaping']} shaping, peak {rep['peak_Nm']:.3f} Nm, "
                  f"peak slew {rep['peak_slew_Nm_s']:.1f} Nm/s, bound by "
                  f"{rep['bound']} (per-line target {rep['target_line_Nm'] * 1e3:.2f} mNm "
                  f"over a {rep['floor_Nm'] * 1e3:.3f} mNm floor). Sanity-check "
                  f"the peak Nm figure against expected drag.")

            N = len(period)
            flag = f'{flag_base}-SETPOINT{setpoint}'
            setpoint += 1
            reading = self._read()
            pos0 = (reading or {}).get('position', {}).get(side)
            aborted = False
            last = 0.0
            for p in range(int(self.design['periods'])):
                # First period is an amplitude ramp-in (which also satisfies
                # the start-at-zero invariant); a taper window would destroy
                # the periodicity the whole scheme depends on.
                env = None if p else np.arange(N) / N
                for i in range(N):
                    last = period[i] * (env[i] if env is not None else 1.0)
                    yield self._cmd(side, last, flag, sample)
                    sample += 1
                # Free-far-shaft safety: zero-mean by construction, but any DC
                # in command or cell integrates into position runaway. The
                # config's velocity safety would catch it late; this is early.
                reading = self._read()
                pos = (reading or {}).get('position', {}).get(side)
                if pos0 is not None and pos is not None and \
                        abs(pos - pos0) > self.DRIFT_LIMIT_RAD:
                    print(f'Preamble: position drifted {abs(pos - pos0):.2f} rad '
                          f'during excitation of {side}; aborting this block')
                    aborted = True
                    break
            # Ramp the command back to zero (untagged, like GridSearch ramps).
            for i in range(self.RAMP_DOWN_SAMPLES):
                frac = 1.0 - (i + 1) / self.RAMP_DOWN_SAMPLES
                yield self._cmd(side, last * frac, None, sample)
                sample += 1
            if aborted:
                break

        self.run += 1


class RampBreak:
    """Event-terminated torque ramp: the stiction / breakaway test
    (behavior type 'ramp_break').

    One motor ramps torque slowly from zero while the other holds; the moment
    the shaft is seen to move, the ramp stops and releases back to zero, rests,
    and the next ramp starts (in the opposite direction, if bipolar). The
    breakaway torque is the static friction.

    WHY THIS IS A BEHAVIOR AND NOT A TRACE PATTERN. A trace is precomputed and
    replayed open-loop by TestTrace, so its release point is a wall-clock time
    fixed before the test ran. Releasing on the *event* needs live feedback in
    the command loop, which only a generator can do. The plumbing already
    existed for the multisine preamble: TestManager passes `sensor_reader`
    straight through, and the controller's snapshot is read inside the
    generator, on the control thread, one cycle behind the command.

    TRIGGER IS NOT MEASUREMENT. The detector here only decides *when to
    release*. The breakaway torque itself is still read off the log afterwards
    from the window just before motion onset (see the stiction analysis), which
    is why the online detector can afford to be conservative: a trigger a few
    milliseconds late costs nothing, a trigger on sensor noise costs a whole
    ramp cycle. The predicate matches the offline one -- |velocity| over
    threshold after having been below it for `stuck_s` -- which is causal (the
    stuck window looks backwards only), so it ports to a running counter with
    no change of meaning.

    NO SENSOR = NO DETECTION, BY DESIGN. With `sensor_reader` None (preview,
    offline expansion) or a reading that does not carry velocity, every ramp
    runs to `amplitude` and releases on the ceiling. That is the worst case the
    hardware could be asked for, which is exactly what the limit checks and the
    preview plot should be showing. `amplitude` is therefore always a real
    ceiling, never merely a hint -- on the rig too, a shaft that never breaks
    free stops the ramp there.

    Everything is SAMPLE-paced (one yield = one control cycle), not wall-clock
    paced: the commanded torque at the trigger is then an exact function of the
    sample count, so cycle jitter cannot smear the number the whole test exists
    to measure.

    Log flags reuse the grid-search span scheme, one span per ramp attempt,
    covering ramp + release + rest so a span holds one complete breakaway
    story:  <id>-RUN<r>-SETPOINT<n>. The moves between hold levels are
    untagged, like GridSearch's ramps.
    """

    RAMP_DOWN_SAMPLES = 250     # matches TestTrace's stabilizing tail

    def __init__(self, parameters, mode, limits, sensor_reader=None):
        self.parameters = parameters
        self.settings = parameters['settings']
        self.mode = mode
        self.limits = limits
        self.sensor_reader = sensor_reader
        self.log_id_base = parameters['id'] + '-RUN'
        self.run = 0

        s = self.settings
        self.ramp_motor = s['ramp_motor']
        assert self.ramp_motor in ('input', 'output'), \
            "ramp_break ramp_motor must be 'input' or 'output', got %r" % (self.ramp_motor,)
        self.hold_motor = 'output' if self.ramp_motor == 'input' else 'input'
        self.hold_mode = s.get('hold_mode', 'torque')
        assert self.hold_mode in ('torque', 'velocity', 'position'), \
            'ramp_break hold_mode %r is not a control mode' % (self.hold_mode,)

        self.hold_levels = [float(v) for v in s.get('hold_levels', [0.0])] or [0.0]
        self.hold_rate = max(abs(float(s.get('hold_rate', 1.0))), 1e-12)
        self.hold_settle_s = max(float(s.get('hold_settle_s', 0.0)), 0.0)
        self.lead_in_s = max(float(s.get('lead_in_s', 0.0)), 0.0)

        self.amplitude = abs(float(s['amplitude']))
        self.rate = abs(float(s['rate']))
        assert self.amplitude > 0, 'ramp_break amplitude must be > 0'
        assert self.rate > 0, 'ramp_break rate must be > 0'
        self.release_s = max(float(s.get('release_s', 1.0)), 1e-3)
        self.rest_s = max(float(s.get('rest_s', 0.0)), 0.0)
        self.bipolar = bool(s.get('bipolar', True))
        self.cycles = max(1, int(s.get('cycles', 1)))

        detect = s.get('detect') or {}
        self.v_thresh = abs(float(detect.get('velocity_threshold', 0.5)))
        self.stuck_s = max(float(detect.get('stuck_s', 0.25)), 0.0)
        self.debounce = max(1, int(detect.get('debounce_cycles', 5)))
        self.arm_fraction = min(max(float(detect.get('arm_fraction', 0.05)), 0.0), 1.0)
        assert self.v_thresh > 0, 'ramp_break velocity_threshold must be > 0'

        with open(dyno_paths.dyno_config_directory + '/master_config.yaml') as f:
            self.dt = yaml.safe_load(f)['cycle_time_us'] / 1e6
        self.stuck_samples = int(round(self.stuck_s / self.dt))

        # The same ceilings TestTrace enforces, applied to the plan rather than
        # to a trace file. The release slew is checked at `amplitude` because
        # that is the largest value a release can ever start from.
        ramp_limits = self.limits[self.ramp_motor]
        assert self.amplitude <= ramp_limits['torque'], \
            ('ramp_break amplitude %g Nm exceeds the %s motor torque limit %g'
             % (self.amplitude, self.ramp_motor, ramp_limits['torque']))
        assert self.rate <= ramp_limits['rotatum'], \
            ('ramp_break rate %g Nm/s exceeds the %s motor rotatum limit %g'
             % (self.rate, self.ramp_motor, ramp_limits['rotatum']))
        assert self.amplitude / self.release_s <= ramp_limits['rotatum'], \
            ('ramp_break release slew %g Nm/s exceeds the %s motor rotatum '
             'limit %g; lengthen release_s'
             % (self.amplitude / self.release_s, self.ramp_motor,
                ramp_limits['rotatum']))
        if self.ramp_motor == 'output':
            reaction = reaction_torque_issue(self.amplitude, self.limits)
            assert reaction is None, 'ramp_break amplitude: %s' % (reaction,)

        hold_peak = max(abs(v) for v in self.hold_levels)
        hold_limit = self.limits[self.hold_motor].get(self.hold_mode)
        if hold_limit is not None:
            assert hold_peak <= hold_limit, \
                ('ramp_break hold level %g exceeds the %s motor %s limit %g'
                 % (hold_peak, self.hold_motor, self.hold_mode, hold_limit))
        if self.hold_mode == 'torque' and self.hold_motor == 'output':
            reaction = reaction_torque_issue(hold_peak, self.limits)
            assert reaction is None, 'ramp_break hold level: %s' % (reaction,)

        self._peak = 0.0        # command value the last ramp ended on

    # -- command helpers -----------------------------------------------------

    def _cmd(self, ramp_value, hold_value, flag=None, breakaway=None):
        cmd = {self.ramp_motor + '_mode': 'torque',
               self.ramp_motor + '_command': float(ramp_value),
               self.hold_motor + '_mode': self.hold_mode,
               self.hold_motor + '_command': float(hold_value)}
        if flag is not None:
            cmd['log_flag'] = flag
        if breakaway is not None:
            # Marks the single sample the detector fired on, carrying the
            # command value it fired at (log key 'breakaway_torque', NaN
            # everywhere else). Without it a reader cannot tell a release that
            # was triggered from one that merely hit the ceiling.
            cmd['breakaway'] = float(breakaway)
        return cmd

    def _read_speed(self):
        """|velocity| of the ramping motor, or None when it cannot be read --
        which the caller must treat as 'no detection possible', never as zero
        (zero would mean 'definitely stuck' and arm the detector on nothing)."""
        if self.sensor_reader is None:
            return None
        try:
            reading = self.sensor_reader()
        except Exception:
            return None
        velocity = (reading or {}).get('velocity') or {}
        value = velocity.get(self.ramp_motor)
        if value is None:
            return None
        return abs(float(value))

    def _dwell(self, ramp_value, hold_value, duration, flag=None):
        for _ in range(int(round(duration / self.dt))):
            yield self._cmd(ramp_value, hold_value, flag)

    def _move_hold(self, start, target):
        """Rate-limited move of the hold motor, ramp motor at zero, untagged."""
        n = int(round(abs(target - start) / self.hold_rate / self.dt))
        for i in range(n):
            yield self._cmd(0.0, start + (target - start) * (i + 1) / n)

    # -- the ramp ------------------------------------------------------------

    def _ramp_to_break(self, sign, hold_value, flag):
        """Ramp torque until the shaft breaks free, or until `amplitude`.

        Leaves the command value it stopped at in self._peak, for the release.
        """
        step = self.rate * self.dt
        max_samples = int(math.ceil(self.amplitude / step))
        arm_floor = self.arm_fraction * self.amplitude
        stuck = 0               # consecutive samples seen below threshold
        stuck_at_onset = 0      # how long it had been stuck when motion started
        moving_run = 0          # consecutive samples seen above threshold
        value = 0.0

        for i in range(1, max_samples + 1):
            value = sign * min(i * step, self.amplitude)
            speed = self._read_speed()
            if speed is not None and speed > self.v_thresh:
                if moving_run == 0:
                    stuck_at_onset = stuck
                moving_run += 1
                stuck = 0
            else:
                # An unreadable sensor lands here too, but it can never trigger
                # a release: moving_run stays 0, so the ramp runs to the ceiling.
                moving_run = 0
                stuck += 1
            # `stuck` starts at 0 on every ramp rather than carrying over from
            # the rest period: after a lurch the shaft may still be coasting
            # when the next ramp begins, and re-earning the stuck window is what
            # stops that coast from reading as an instant breakaway. It costs
            # stuck_s of ramp, which at these rates is a few mNm.
            if (moving_run >= self.debounce
                    and stuck_at_onset >= self.stuck_samples
                    and abs(value) >= arm_floor):
                self._peak = value
                print('RampBreak: breakaway at %+.4g Nm (%s motor)'
                      % (value, self.ramp_motor))
                yield self._cmd(value, hold_value, flag, breakaway=value)
                return
            yield self._cmd(value, hold_value, flag)

        self._peak = value
        print('RampBreak: no breakaway; ramp hit the %+.4g Nm ceiling' % (value,))

    def _release(self, hold_value, flag):
        n = max(1, int(round(self.release_s / self.dt)))
        start = self._peak
        for i in range(1, n + 1):
            yield self._cmd(start * (1.0 - i / n), hold_value, flag)

    # -- command generation --------------------------------------------------

    def commands(self):
        # Initial command sets op modes before any timing starts.
        yield self._cmd(0.0, 0.0)

        flag_base = self.log_id_base + str(self.run)
        span = 0
        hold_value = 0.0
        directions = (1.0, -1.0) if self.bipolar else (1.0,)

        if self.lead_in_s > 0:
            yield from self._dwell(0.0, 0.0, self.lead_in_s)

        for level in self.hold_levels:
            if level != hold_value:
                yield from self._move_hold(hold_value, level)
                hold_value = level
            if self.hold_settle_s > 0:
                yield from self._dwell(0.0, hold_value, self.hold_settle_s)
            for _ in range(self.cycles):
                for sign in directions:
                    flag = flag_base + '-SETPOINT' + str(span)
                    span += 1
                    yield from self._ramp_to_break(sign, hold_value, flag)
                    yield from self._release(hold_value, flag)
                    if self.rest_s > 0:
                        yield from self._dwell(0.0, hold_value, self.rest_s, flag)

        if hold_value != 0.0:
            yield from self._move_hold(hold_value, 0.0)
        for _ in range(self.RAMP_DOWN_SAMPLES):
            yield self._cmd(0.0, 0.0)
        if self.hold_mode == 'position':
            # The same reset TestTrace does: toggling out of position mode makes
            # the next entry into it re-zero against the current shaft angle.
            yield {'input_mode': 'torque', 'output_mode': 'torque',
                   'input_command': 0.0, 'output_command': 0.0}

        self.run += 1


class TestManager:
    def __init__(self, test_file, mode, limits, sensor_reader=None):
        # sensor_reader: optional zero-arg callable returning
        # {'torque': {sensor_name: Nm}, 'position': {'input': rad, 'output': rad},
        #  'velocity': {'input': rad/s, 'output': rad/s}}
        # snapshotted from live telemetry. Used by the preamble (noise-floor
        # anchoring + drift supervision) and by ramp_break (breakaway
        # detection); None (as in preview / offline expansion) degrades to
        # documented fallbacks -- for ramp_break, ramping to the ceiling.
        self.behaviors = {}
        self.test_config = None
        self.mode = mode
        self.limits = limits
        self.sensor_reader = sensor_reader
        self.name = test_file.split('.')[0]

        with open(f"{dyno_paths.dyno_test_directory}/{test_file}", "r") as f:
            self.test_config = yaml.safe_load(f)

        self._load_yaml(test_file)
        print('Test loaded.')

    def _load_yaml(self, file):
        with open(f"{dyno_paths.dyno_test_directory}/{file}", "r") as f:
            test_config = yaml.safe_load(f)

        if 'imports' in test_config.keys():
            for referenced_test in test_config['imports']:
                self._load_yaml(referenced_test)

        for behavior in test_config['behaviors']:
            self._load_behavior(behavior)

    def _load_behavior(self, behavior):
        if behavior['type'] == 'test_trace' and behavior['id'] not in self.behaviors:
            self.behaviors[behavior['id']] = TestTrace(behavior, self.mode, self.limits)
        elif behavior['type'] == 'grid_search' and behavior['id'] not in self.behaviors:
            self.behaviors[behavior['id']] = GridSearch(behavior, self.mode, self.limits)
        elif behavior['type'] == 'preamble' and behavior['id'] not in self.behaviors:
            self.behaviors[behavior['id']] = Multisine(
                behavior, self.mode, self.limits, self.sensor_reader)
        elif behavior['type'] == 'ramp_break' and behavior['id'] not in self.behaviors:
            self.behaviors[behavior['id']] = RampBreak(
                behavior, self.mode, self.limits, self.sensor_reader)
        elif behavior['type'] == 'loop':
            for looped_behavior in behavior['behaviors']:
                self._load_behavior(looped_behavior)
        else:
            raise ValueError('Print invalid test "type" specified: '+behavior['type'])

    def reset(self):
        self._test_start_real_time = time.time() # Capture new start time for the run
        self._test_complete = False              # Test is no longer complete

        # Re-create the top-level behavior generator
        self._behavior_gen = behavior_iterator(self.test_config)

        # Reset individual behavior instances (though their .commands() also re-initializes)

        # Create the master command generator for this run
        self._master_command_generator = self._create_master_command_generator()

    def _create_master_command_generator(self):
        # Iterate through the sequence of behavior definitions
        for behavior_definition in self._behavior_gen:
            behavior_id = behavior_definition['id']
            behavior_instance = self.behaviors[behavior_id]

            print(f"\n--- Starting Behavior: '{behavior_id}' ({behavior_definition['type']}) ---")
            
            # Get the command generator for the current behavior instance
            # Pass the test_start_real_time to allow the behavior to calculate its own internal time
            current_behavior_command_gen = behavior_instance.commands()
            
            # Yield all commands from the current behavior until it's exhausted.
            # This is where the magic happens: this generator delegates to the sub-generator.
            yield from current_behavior_command_gen
            

    def next_command(self):
        if self._test_complete:
            return None # Test has already finished

        try:
            cmd = next(self._master_command_generator)
            return cmd
        except StopIteration:
            self._test_complete = True # Mark the test as complete
            print(f"Test Completed: {self.name}")
            return None # Signal that the test is finished

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import tkinter as tk
    from tkinter import filedialog

    print('Select test to validate: ')

    root = tk.Tk()
    root.withdraw()

    test_name = filedialog.askopenfilename().split('/')[-1]
    TM = TestManager(test_name)
    TM.reset()

    input_cmds = []
    output_cmds = []
    time_arr = [None]

    while True:
        time.sleep(0.001)
        next_cmd = TM.next_command()
        if next_cmd == None:
            break
        else:
            print(next_cmd)
            input_cmds.append(next_cmd['input_command'])
            output_cmds.append(next_cmd['output_command'])
            if time_arr[0] == None:
                t_offset = time.time()
                time_arr[0] = 0
            else:
                time_arr.append(time.time() - t_offset)

    plt.figure()
    plt.title('Input commands')
    plt.plot(time_arr, input_cmds)
    plt.xlabel('Time (s)')
    plt.figure()
    plt.title('Output commands')
    plt.plot(time_arr, output_cmds)
    plt.xlabel('Time (s)')
    plt.show()
