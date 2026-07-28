from dyno.src.master import Master
import yaml
import time
import math
import os
import signal
import threading
from operator import attrgetter
from deployment import dyno_paths
from dyno.src.test_manager import TestManager

SCHED_POLICY = os.SCHED_FIFO
SCHED_PRIO = 50

class Controller(Master):
    def __init__(self, telemetry_queue=None, command_queue=None,mode=None):
        # Runs in a child process: ignore SIGINT so a terminal Ctrl-C can't kill
        # the real-time control loop mid-cycle (which would orphan it with the
        # drives still enabled). The parent GUI orchestrates an orderly shutdown
        # via the command queue instead.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.mode = mode

        with open(f"{dyno_paths.dyno_config_directory}/{mode}_dyno_config.yaml", 'r') as f:
            self.dyno_params = yaml.safe_load(f)

        self._expected_slave_layout = self.dyno_params['expected_slave_layout']
        
        super().__init__(slave_layout = self._expected_slave_layout)

        self._telemetry_queue = telemetry_queue
        self._command_queue = command_queue

        self.data_counter = 0
        self._safe_default_command = {'input_mode':'torque','output_mode':'torque','input_command': 0,'output_command': 0}

        self.t_offset = time.perf_counter()

        self._current_input_mode = None
        self._current_output_mode = None
        self._test_active = False
        self.generated_cmd = None
        self.test_definition = None
        self._test_init_thread = None
        # Handoff slot for a test loaded on the init thread: (generation,
        # TestManager). Only _cmd_check (control thread) moves it into
        # test_definition — see _cmd_check.
        self._pending_test_definition = None
        # Bumped every time the armed test changes. A load still running when
        # the operator re-arms or disarms carries a stale generation and is
        # dropped, rather than resurfacing a test that was already dismissed.
        self._test_load_generation = 0
        # (generation, test_file) currently loading, and (generation, message)
        # for one that failed. Both are reported to the GUI and both go stale
        # on their own when the generation moves past them.
        self._loading_test = None
        self._test_load_error = None

        self._aux_funcs = []
        if self.mode == 'actuator_production':
            self._aux_funcs.append(self._aux_func_A3_Dyno)

        # Compile safety checks from config: each entry names a telemetry
        # source (attrgetter path, same idiom as log_keys) and a limit.
        # safeties: { output_torque: { source: devices.ADC.output_torque, limit: 375 } }
        self._safety_checks = []
        for check_name, spec in self.dyno_params.get('safeties', {}).items():
            if not isinstance(spec, dict) or 'source' not in spec or 'limit' not in spec:
                raise ValueError(
                    f"safeties entry '{check_name}' in {self.mode}_dyno_config.yaml "
                    f"must be a dict with 'source' and 'limit' keys, got: {spec!r}")
            self._safety_checks.append(
                (check_name, attrgetter(spec['source']), abs(spec['limit'])))

        try:
            os.sched_setscheduler(0, SCHED_POLICY, os.sched_param(SCHED_PRIO))
            cpu_set = {1, 2, 3, 4}
            os.sched_setaffinity(0, cpu_set)
            print("Controller: Real-time scheduling enabled")
        except PermissionError:
            print("Controller: Real-time scheduling not permitted, running normally")

        self.run()

    def _stop_test(self):
        if not self.test_definition == None:
            self.test_definition.reset()
        self.devices.DUT.sw_enable = False
        self.devices.LOAD.sw_enable = False
        self._test_active = False

        # self.devices.DUT.command_operating_mode('torque')
        self.devices.LOAD.command_operating_mode('torque')
        self._safe_default_command['input_mode'] = 'torque'
        self._safe_default_command['output_mode'] = 'torque'
        self._safe_default_command['input_command'] = 0
        self._safe_default_command['output_command'] = 0

    def _control_state(self):
        """Rig state for the GUI's status indicator (telemetry slot -1).

        The GUI can see none of this on its own: test_definition lives in this
        process and drive faults are not in log_keys, so the GUI could only
        report what it ASKED for -- which is wrong for as long as a load is in
        flight, and never learns that a load failed here. Slot -1 was
        previously always None and both consumers slice it off (sample[:-2]),
        so filling it in cannot shift plot or log columns.

        `_loading_test` / `_test_load_error` are reported only while their
        generation is current, so a superseded load reports nothing.
        """
        current = self._test_load_generation
        return {
            'armed': self.test_definition.name if self.test_definition else None,
            'loading': (self._loading_test[1]
                        if self._loading_test and self._loading_test[0] == current
                        else None),
            'load_error': (self._test_load_error[1]
                           if self._test_load_error
                           and self._test_load_error[0] == current else None),
            'test_active': self._test_active,
            'fault': bool(self.devices.DUT.fault or self.devices.LOAD.fault),
        }

    def _send_telemetry(self):
        self.logging_state = {'log': False} # Default to not logging
        if self._test_active and 'log_flag' in self.current_cmd:
            self.logging_state = {'log': True, 'behavior_id': self.current_cmd['log_flag']}
        elif self._test_active: # If test is active but no specific log_flag
            self.logging_state = {'log': True}

        self.control_state = self._control_state()

        self.time = time.perf_counter() - self.t_offset

        telemetry = [getter(self) for getter in self._telemetry_compiled]
        telemetry.append(self.logging_state)
        telemetry.append(self.control_state)

        self._telemetry_queue.put_nowait(telemetry)

    # recieves and manages commands from the GUI
    def _cmd_check(self):
        # Publish a background-loaded test, on this (the control) thread.
        # TestManager.__init__ leaves the instance unusable until reset(), and
        # the init thread can finish AFTER start_test has already reset the
        # previous test -- assigning from that thread would hand step() a test
        # with no command generator and kill the control loop mid-run. Holding
        # the swap here means test_definition only ever changes between tests.
        if self._pending_test_definition is not None and not self._test_active:
            generation, test = self._pending_test_definition
            self._pending_test_definition = None
            if generation == self._test_load_generation:
                self.test_definition = test
                self._loading_test = None
                print(f'Controller: {test.name} armed')
            else:
                print(f'Controller: dropped superseded load of {test.name}')

        read_queue = True
        while read_queue:
            try:
                cmd = self._command_queue.get_nowait()
                if cmd[0] == 'start_test':
                    if self._test_active:
                        print('Unable to start test: Test already active')
                    elif self.devices.DUT.fault:
                        print('Unable to start test: DUT in fault state')
                    elif self.devices.LOAD.fault:
                        print('Unable to start test: LOAD in fault state')
                    else:
                        self.pull_cmd = True
                        self.devices.DUT.sw_enable = True
                        self.devices.LOAD.sw_enable = True
                        if not self.test_definition == None:
                            self._test_active = True
                            self.test_definition.reset()
                        print('Starting test')
                        
                elif cmd[0] == 'stop_test':
                    self._stop_test()
                    print('attempting to stop test, in / out motor commanded to torque mode')

                elif cmd[0] == 'test_def':
                    if not self._test_active:
                        self._get_limits()
                        self._test_load_generation += 1
                        generation = self._test_load_generation
                        self._loading_test = (generation, cmd[1][0])

                        # Define the target function for the thread
                        def load_test(file, mode, limits):
                            # This runs off the control loop, so an exception
                            # here would otherwise die with the thread and
                            # leave the GUI waiting forever. Report it instead
                            # -- these are the limit asserts checked against
                            # the drives' real limits, which the GUI's
                            # config-based pre-check cannot see.
                            try:
                                test = TestManager(file, mode, limits)
                                test.reset()  # usable before the loop sees it
                            except Exception as e:
                                self._test_load_error = (generation,
                                                         f'{type(e).__name__}: {e}')
                                print(f'Controller: FAILED to load {file}: {e}')
                                return
                            self._pending_test_definition = (generation, test)
                            print("Controller: TestManager ready.")

                        self._test_init_thread = threading.Thread(target=load_test, args=(cmd[1][0],cmd[1][1], self.limits))
                        self._test_init_thread.start()
                    else:
                        print('Please re-select test when dyno is not active')

                elif cmd[0] == 'clear_test':
                    # Disarm. The GUI sends this whenever it stops vouching for
                    # the armed test (a new pick, a failed load), because it
                    # cannot clear test_definition itself — without this the
                    # previous test stays loaded here and Start would run it.
                    if not self._test_active:
                        self._test_load_generation += 1  # drop any load in flight
                        self._pending_test_definition = None
                        if self.test_definition is not None:
                            print(f'Controller: {self.test_definition.name} disarmed')
                        self.test_definition = None
                    else:
                        print('Unable to disarm: test already active')

                elif cmd[0] == 'shutdown':
                    print('Shutdown command recieved by control loop')
                    self.shutdown = True
                    for device_name in vars(self.devices).keys():
                        device_instance = getattr(self.devices, device_name)
                        if hasattr(device_instance, 'shutdown'):
                            device_instance.shutdown = True

                    self.devices.LOAD.shutdown = True
                    self.devices.DUT.shutdown = True

                    try:
                        self.devices.input_motor.shutdown = True
                    except:
                        pass


            except:
                read_queue = False
                pass

    # stops the test if measured values are outside of an acceptable range.
    # Checks are declared in the config's `safeties:` section (source + limit);
    # add entries there rather than here.
    def _safety_trigger(self):
        for check_name, getter, limit in self._safety_checks:
            value = abs(getter(self))
            if value > limit:
                print(f'Safety triggered, {check_name} of {value} exceeds limit of {limit}')
                return True

        if self.devices.DUT.fault:
            print('Safety triggered, DUT is in fault state')
            return True

        if self.devices.LOAD.fault:
            print('Safety triggered, LOAD is in fault state')
            return True

        return False
    
    def _get_limits(self):
        self.limits = {
            'torque': min(abs(self.devices.DUT.torque_limit),abs(self.devices.LOAD.torque_limit)),
            'velocity': min(abs(self.devices.DUT.velocity_limit),abs(self.devices.LOAD.velocity_limit)),
            'acceleration': abs(self.devices.LOAD.acceleration_limit),
            'rotatum': min(abs(self.devices.LOAD.rotatum_limit), abs(self.devices.DUT.torque_limit)*4)
        }
       
    def step(self):
        self.data_counter += 1

        if self.devices.LOAD.position_offset == 0 and not self.devices.DUT.position == 0:
            self.devices.LOAD.position_offset = self.devices.DUT.position - self.devices.LOAD.position

        if self._test_active:
            if self._safety_trigger():
                self._stop_test()

            if self.pull_cmd:
                self.generated_cmd = self.test_definition.next_command()
                self.pull_cmd = False

            if not self.generated_cmd == None and not self.shutdown:
                if self.generated_cmd['input_mode'] != self.devices.DUT.mode:
                    if not self.devices.DUT.switching_modes:
                        self.devices.DUT.command_operating_mode(self.generated_cmd['input_mode'])
                        self._safe_default_command['input_mode'] = self.current_cmd['input_mode']
                        self._safe_default_command['input_command'] = self.current_cmd['input_command']
                        

                if self.generated_cmd['output_mode'] != self.devices.LOAD.mode:
                    if not self.devices.LOAD.switching_modes:
                        self.devices.LOAD.command_operating_mode(self.generated_cmd['output_mode'])
                        self._safe_default_command['output_mode'] = self.current_cmd['output_mode']
                        self._safe_default_command['output_command'] = self.current_cmd['output_command']
                        

                if self.devices.DUT.mode == self.generated_cmd['input_mode'] and self.devices.LOAD.mode == self.generated_cmd['output_mode'] and not self.devices.DUT.switching_modes and not self.devices.LOAD.switching_modes:
                    self.current_cmd = self.generated_cmd # Use the command from the test
                    self.pull_cmd = True

            else:
                self._stop_test()

        else:
            self.current_cmd = self._safe_default_command

            if self.devices.DUT.sw_enable or self.devices.LOAD.sw_enable:
                self.devices.DUT.sw_enable = False
                self.devices.LOAD.sw_enable = False

        # Both are rebuilt in _send_telemetry, below.
        self.control_state = None
        self.logging_state = None

        ff_ratio = 0.8

        if self.devices.DUT.mode == 'torque' and not self.devices.LOAD.mode == 'torque':
            torque_ff = ff_ratio*self.current_cmd['input_command'] * self.devices.DUT.params['gear_ratio']
            self.devices.LOAD.send_command(self.current_cmd['output_command'], torque_ff)
        else:
            self.devices.LOAD.send_command(self.current_cmd['output_command'])

        if self.devices.LOAD.mode == 'torque' and not self.devices.DUT.mode == 'torque':
            torque_ff = ff_ratio*self.current_cmd['output_command'] / self.devices.DUT.params['gear_ratio']
            self.devices.DUT.send_command(self.current_cmd['input_command'], torque_ff)
        else:
            self.devices.DUT.send_command(self.current_cmd['input_command'])

        if self.current_cmd['input_command'] != getattr(self, '_last_dut_cmd', None):
            self._last_dut_cmd = self.current_cmd['input_command']

        for aux_func in self._aux_funcs:
            aux_func()

        self._send_telemetry()
        self._cmd_check()
        time.sleep(0) #momentarily yeilds the GIL

    def _write_led(self, ch, mode):

        # LED pins on AKD (LOAD)
        pins = {'g':1, 'r':2}
        """Mode can be 'on', 'off', or 'blink'"""
        blink_state = (self.data_counter % 1000) < 500
        val = False
        if mode == 'on': val = True
        elif mode == 'blink': val = blink_state

        try:
            dev = getattr(self.devices, 'LOAD')
            if hasattr(dev, 'set_channel'):
                dev.set_channel(pins[ch], val)
            elif hasattr(dev, 'set_dout'):
                dev.set_dout(pins[ch], val)
        except AttributeError:
            pass

    def _aux_func_A3_Dyno(self):
        # Run specific logic at lower cycle rate
        if self.data_counter % 20 == 0:
            # Stator temperature control
            stator_temp = self.devices.rtd_module.load_stator_temp

            # Grab the 24v_power3 module
            pwr3 = getattr(self.devices, '24v_power3')
            if math.isnan(stator_temp) or stator_temp >= 100:
                pwr3.set_channel(4, False)
                self._stop_test()
                if math.isnan(stator_temp):
                    print('CRITICAL: Stator RTD Invalid (NaN). Test Stopped')
                else:
                    print(f'CRITICAL: Stator Temp {stator_temp:.1f}C >= 100C. Test Stopped')
            elif stator_temp >= 60:
                pwr3.set_channel(4, True)
            else:
                pwr3.set_channel(4, False)

            # Status Indication
            if self.devices.DUT.fault or self.devices.LOAD.fault:
                self._write_led('r', 'on')
                self._write_led('g', 'off')
            elif self._test_active:
                self._write_led('r', 'off')
                self._write_led('g', 'on')
            else:
                self._write_led('r', 'off')
                self._write_led('g', 'blink')