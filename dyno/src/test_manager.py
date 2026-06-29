import yaml
import numpy as np
import pandas as pd
import time
from deployment import dyno_paths

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
            if self.input_mode in ['velocity', 'torque']:
                self.trace[f"input_motor_{self.input_mode}"] *= self.limits[self.input_mode]
                print('\tScaling input trace by ',self.limits[self.input_mode])

                if 'scale_override' in self.settings['input_motor'].keys():
                    self.trace[f"input_motor_{self.input_mode}"] *= min(1, abs(self.settings['input_motor']['scale_override']))

            if self.output_mode in ['velocity', 'torque']:
                self.trace[f"output_motor_{self.output_mode}"] *= self.limits[self.output_mode]
                print('\tScaling output trace by ',self.limits[self.output_mode])


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
        
        for motor_key in ['input_motor', 'output_motor']:
            if self.settings[motor_key]['control_mode'] == 'torque':
                test_torque = self.trace[f"{motor_key}_torque"][:]
                abs_test_torque = abs(test_torque)
                max_abs_test_torque = max(abs_test_torque)
                print(f"TORQUE: {max_abs_test_torque}")
                limit_val = self.limits['torque']
                print(f"LIMIT: {limit_val}")
                assert max(abs(self.trace[f"{motor_key}_torque"][:])) <= self.limits['torque'], 'Max trace torque exceeds system limits'
                assert self.trace[f"{motor_key}_torque"][len(self.trace['time'])-1] == 0, 'Torque trace must end at 0'
                assert max(abs(rates[motor_key])) <= self.limits['rotatum'], 'Trace rotatum (d_torque/dt) of '+str(max(abs(rates[motor_key])))+' Nm/s exceeds system limits'

                
            if self.settings[motor_key]['control_mode'] == 'velocity':
                velocity = self.trace[f"{motor_key}_velocity"][:]
                abs_test_velocity = abs(velocity)
                max_abs_test_velocity = max(abs_test_velocity)
                print(f"VELOCITY: {max_abs_test_velocity}")
                limit_val = self.limits['velocity']
                print(f"LIMIT: {limit_val}")
                #assert max(abs(self.trace[f"{motor_key}_velocity"][:])) <= self.limits['velocity'], 'Max trace velocity exceeds system limits'
                assert max(abs(rates[motor_key])) <= self.limits['acceleration'], 'Trace acceleration of '+str(max(abs(rates[motor_key])) )+' exceeds system limits'
                assert self.trace[f"{motor_key}_velocity"][len(self.trace['time'])-1] == 0, 'Velocity trace must end at 0'

            if self.settings[motor_key]['control_mode'] == 'position':
                assert max(abs(rates[motor_key]))  <= self.limits['velocity'], 'Trace velocity of '+str(max(abs(rates[motor_key])))+' exceeds system limits'


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
                self.settings['input_motor']['command_list'] = [command * self.limits[self.input_mode] for command in self.settings['input_motor']['command_list']]

            if self.output_mode in ['velocity', 'torque']:
                self.settings['output_motor']['command_list'] = [command * self.limits[self.output_mode] for command in self.settings['output_motor']['command_list']]

        with open(f"{dyno_paths.dyno_config_directory}/master_config.yaml", 'r') as f:
            master_params = yaml.safe_load(f)

        self.rate_limits = {}
        for motor_key in ['input_motor', 'output_motor']:
            control_mode = self.settings[motor_key]['control_mode']
            if control_mode == 'torque':
                self.rate_limits['torque'] = self.settings['transition_rate'] * self.limits['rotatum']
            if control_mode == 'velocity':
                self.rate_limits['velocity'] = self.settings['transition_rate'] * self.limits['acceleration']

        self.timestep = master_params['cycle_time_us'] / 1e6 # Convert microseconds to seconds

        self.sweep_axes = {
            self.settings[key]['control_mode']: self.settings[key]['command_list']
            for key in ['input_motor', 'output_motor']
        }
        
        # Hard coded values to limit RMS torques during a test
        if self.input_mode == 'torque':
            self.t_limit = 4
        elif self.output_mode == 'torque':
            self.t_limit = 110

        for motor_key in ['input_motor', 'output_motor']:
            if self.settings[motor_key]['control_mode'] == 'torque':
                assert max(abs(np.array(self.settings[motor_key]['command_list']))) <= self.limits['torque'], 'Max commanded '+motor_key+' torque exceeds system limits'

        
            if self.settings[motor_key]['control_mode'] == 'velocity':
                assert max(abs(np.array(self.settings[motor_key]['command_list']))) <= self.limits['velocity'], 'Max commanded '+motor_key+' velocity exceeds system limits'

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

class TestManager:
    def __init__(self, test_file, mode, limits):
        self.behaviors = {}
        self.test_config = None
        self.mode = mode
        self.limits = limits
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
