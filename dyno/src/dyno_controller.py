from dyno.src.master import Master
import yaml
import time
import math
import os
import threading
from deployment import dyno_paths
from dyno.src.test_manager import TestManager

SCHED_POLICY = os.SCHED_FIFO
SCHED_PRIO = 50

class Controller(Master):
    def __init__(self, telemetry_queue=None, command_queue=None,mode=None):
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

        self._aux_funcs = []
        assert self.mode in ['gearbox', 'actuator', 'actuator_production']
        if self.mode == 'actuator_production':
            self._aux_funcs.append(self._aux_func_A3_Dyno)

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

    def _send_telemetry(self):
        self.logging_state = {'log': False} # Default to not logging
        if self._test_active and 'log_flag' in self.current_cmd:
            self.logging_state = {'log': True, 'behavior_id': self.current_cmd['log_flag']}
        elif self._test_active: # If test is active but no specific log_flag
            self.logging_state = {'log': True}

        self.time = time.perf_counter() - self.t_offset

        telemetry = [getter(self) for getter in self._telemetry_compiled]
        telemetry.append(self.logging_state)
        telemetry.append(self.control_state)

        self._telemetry_queue.put_nowait(telemetry)

    # recieves and manages commands from the GUI
    def _cmd_check(self):
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
                        # Define the target function for the thread
                        def load_test(file, mode, limits):
                            self.test_definition = TestManager(file, mode, limits)
                            print("Controller: TestManager ready.")

                        self._test_init_thread = threading.Thread(target=load_test, args=(cmd[1][0],cmd[1][1], self.limits))
                        self._test_init_thread.start()
                    else:
                        print('Please re-select test when dyno is not active')

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

    # stops the test if measured values are outside of an acceptable range. more entries can be added as desired
    def _safety_trigger(self):
        if self.mode == 'actuator_production':
            output_torque = abs(self.devices.adc_1.load_torque)
        else:
            output_torque = abs(self.devices.ADC.output_torque)
            input_torque = abs(self.devices.ADC.input_torque)

            if input_torque > self.dyno_params['safeties']['input_torque']:
                print('Safety triggered, input torque of ',input_torque, ' exceeds limit')
                return True

        output_velocity = abs(self.devices.LOAD.velocity)
        output_gear_ratio = self.devices.LOAD.params['gear_ratio']
        input_velocity = abs(self.devices.DUT.velocity)
        input_gear_ratio = self.devices.DUT.params['gear_ratio']

        if output_torque > self.dyno_params['safeties']['output_torque']:
            print('Safety triggered, output torque of ',output_torque, ' exceeds limit')
            return True
        
        if input_velocity > self.dyno_params['safeties']['input_velocity']:
            print('Safety triggered, input velocity of ',input_velocity, ' exceeds limit')
            return True

        if output_velocity > self.dyno_params['safeties']['output_velocity']:
            print('Safety triggered, output velocity of ',output_velocity, ' exceeds limit')
            return True
        
        # 2 rad/s velocity difference, in output coordinates, between the two motors will kill the test
        if abs((input_velocity / input_gear_ratio) - (output_velocity / output_gear_ratio)) > 2: 
            print('Safety triggered, diverging velocities detected. Check the configured gear ratio for each motor')
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