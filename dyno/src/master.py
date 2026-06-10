import pysoem
import time
import threading
import yaml
from dyno.src.timing import nano_sleep
import os
from dyno.src.devices import DEVICE_CLASSES, EL2004
import types
from operator import attrgetter
from deployment import dyno_paths
import numpy as np

class Master:
    def __init__(self, slave_layout):
        with open(f"{dyno_paths.dyno_config_directory}/master_config.yaml", 'r') as f:
            self.master_params = yaml.safe_load(f)

        self._ifname = self.master_params['interface']
        self.process_data_cycle_time_us = self.master_params['cycle_time_us'] # 1ms in microseconds
        self._ifname_red = None
        self._pd_thread_stop_event = threading.Event()
        self._ch_thread_stop_event = threading.Event()
        self._actual_wkc = 0
        self._master = pysoem.Master()
        self._master.in_op = False
        self._master.do_check_state = False
        self._last_master_time = 0

        self.in_op = False
        self.shutdown = False

        # Open network interface
        print(len(slave_layout), ' Ethercat slaves expected')
        self._master.open(self._ifname, self._ifname_red)
        online_slave_count = self._master.config_init()

        while online_slave_count != len(slave_layout):
            self._master.close()
            time.sleep(1)
            self._master.open(self._ifname, self._ifname_red)
            print('Waiting for slaves to initialize ',online_slave_count,' are online, ',len(slave_layout),' expected')
            online_slave_count = self._master.config_init()
       
    def step(self):
        pass

    def _processdata_loop(self):
        '''Background thread that sends and receives the process-data frame.'''
        cycle_time_sec = self.process_data_cycle_time_us / 1_000_000.0
        print(f'Process data thread cycle time: {cycle_time_sec*1000:.2f} ms')

        wkc_error_count = 0
        max_wkc_errors_before_warning = 10 # Only print warning after 10 consecutive errors

        target_dc_modulo = 500000
        master_time_offset = None
        jitter_arr = np.zeros(1000)
        cycle_start_time = time.perf_counter_ns()
        while not self._pd_thread_stop_event.is_set():

            pdo_send_time = time.perf_counter_ns()
            self._master.send_processdata()
            self._actual_wkc = self._master.receive_processdata(timeout=int(cycle_time_sec * 1_000_000 * 0.9)) # Timeout in microseconds, 90% of cycle

            dc_modulo = self._master.dc_time % 1000000 # determine where in the ethercat cycle we've sent data
            if master_time_offset == None:
                master_time_offset = self._master.dc_time - pdo_send_time

            if self.data_counter > 500:
                jitter_arr[self.data_counter % 1000] = pdo_send_time - last_dc_time

            last_dc_time = pdo_send_time

            start_time_shift = int( ((self._master.dc_time - pdo_send_time) - master_time_offset)/30) + int((dc_modulo - target_dc_modulo)/30)
            master_time_offset += start_time_shift

            if self.in_op:
                # Process inbound PDO data
                for device_name in vars(self.devices).keys():
                    device_instance = getattr(self.devices, device_name)
                    if hasattr(device_instance, 'process_txpdo'):
                        device_instance.process_txpdo()

                self.step()

                # Process outbound PDO data
                for device_name in vars(self.devices).keys():
                    device_instance = getattr(self.devices, device_name)
                    if hasattr(device_instance, 'write_rxpdo'):
                        device_instance.write_rxpdo()

            if self._actual_wkc < self._master.expected_wkc:
                wkc_error_count += 1
                if wkc_error_count >= max_wkc_errors_before_warning and not self.shutdown:
                    print(f'WARNING: Incorrect WKC. Expected: {self._master.expected_wkc}, Actual: {self._actual_wkc}')
                    wkc_error_count = 0 # Reset counter after printing warning
            else:
                wkc_error_count = 0 # Reset if WKC is correct

            cycle_start_time += self.process_data_cycle_time_us * 1000 - start_time_shift

            sleep_duration = cycle_start_time - time.perf_counter_ns()
            if sleep_duration > 0:
                nano_sleep(int(sleep_duration))

    # PDO update loop, taken from basic example. For each ethercat device, if the wrapping class has an 'update' method, that update method is run
    def run(self):
        """
        Main orchestrator for the EtherCAT Master. 
        Follows the sequence: Hardware Init -> Sensor Routing -> SDO Setup -> OP Transition.
        """
        # 0. Request PREOP immediately to allow SDO access during init/setup
        self._master.state = pysoem.PREOP_STATE
        self._master.write_state()
        self._master.state_check(pysoem.PREOP_STATE, timeout=500_000)

        # --- 1. Hardware Instantiation ---
        EL2004._output_bit_counter = 0 # Reset bit-packing
        self.devices = {}

        print('\nStep 1: Instantiating hardware devices...')
        for i, slave in enumerate(self._master.slaves):
            layout_entry = self._expected_slave_layout[i]
            model = layout_entry['model']
            name = layout_entry['name']

            if slave.id != DEVICE_CLASSES[model]['id']:
                self._master.close()
                raise Exception(f"Hardware Mismatch at Slave {i}. Expected {model} "
                                f"(ID {DEVICE_CLASSES[model]['id']}), got ID {slave.id}")
            
            params = layout_entry.get('params', {})
            # Initialize hardware class
            self.devices[name] = DEVICE_CLASSES[model]['class'](slave, name, params)
            print(f"\tInitialized {name} ({model})")

        self.devices = types.SimpleNamespace(**self.devices)

        # --- 2. Logical Sensor & Power Routing ---
        print('\nStep 2: Routing logical sensors and activating power supplies...')
        if 'sensors' in self.dyno_params:
            for sensor_name, config in self.dyno_params['sensors'].items():
                config['name'] = sensor_name 
                
                if 'port' in config:
                    port_mapping = self.dyno_params.get('panel_ports', {}).get(config['port'])
                    if port_mapping:
                        sig_mod = getattr(self.devices, port_mapping['signal_module'])
                        sig_ch = port_mapping['signal_channel']
                        # Inject config into hardware params
                        sig_mod.params[sig_ch] = config
                        setattr(sig_mod, sensor_name, 0.0)
                        supply_req = config.get('supply', 'NA')
                        power_key = f'power_{supply_req}'
                        if power_key in port_mapping:
                            p_route = port_mapping[power_key]
                            p_mod = getattr(self.devices, p_route['module'])
                            p_mod.set_channel(p_route['channel'], True)
                            print(f"\tPowered ON '{sensor_name}' via {p_route['module']} (Pin {p_route['channel']})")

                elif 'signal_module' in config:
                    sig_mod = getattr(self.devices, config['signal_module'])
                    sig_ch = config.get('signal_channel', 'ch1')
                    sig_mod.params[sig_ch] = config
                    setattr(sig_mod, sensor_name, 0.0)

        # --- 3. SDO Configuration ---
        print('\nStep 3: Executing device setups...')
        for device in vars(self.devices).values():
            if hasattr(device, 'setup'):
                device.setup()

        # --- 3.5 Auto-Add Sensors to Log Keys ---
        print('Adding sensors to log keys')
        if 'sensors' in self.dyno_params:
            for sensor_name, config in self.dyno_params['sensors'].items():
                
                # Determine the module name (either from port or direct signal_module)
                if 'port' in config:
                    port_map = self.dyno_params.get('panel_ports', {}).get(config['port'])
                    module_name = port_map['signal_module']
                else:
                    module_name = config.get('signal_module')

                # Construct the dot-notation path for attrgetter
                # Format: devices.<module_name>.<sensor_name>
                log_path = f"devices.{module_name}.{sensor_name}"
                
                # Check if this sensor is already in log_keys to avoid duplicates
                existing_keys = [k[0] for k in self.dyno_params.get('log_keys', [])]
                
                if sensor_name not in existing_keys:
                    self.dyno_params['log_keys'].append([sensor_name, log_path])
                    print(f"\tAuto-logged sensor: {sensor_name} -> {log_path}")

        # Now compile the telemetry list (this remains the same)
        self._telemetry_compiled = [attrgetter(attr[1]) for attr in self.dyno_params.get('log_keys', [])]

        print('\nCompiled telemetry check: item count = ',len(self._telemetry_compiled))
        for item in self._telemetry_compiled:
            print('\t',item)

        # --- 4. Transition to OP ---
        self._master.config_map()
        self._master.config_dc()

        for i, slave in enumerate(self._master.slaves):
            model = self._expected_slave_layout[i]['model']
            if DEVICE_CLASSES[model].get('has_dc', False):
                slave.dc_sync(True, self.process_data_cycle_time_us * 1000)

        self._pd_thread_stop_event.clear()
        control_thread = threading.Thread(target=self._processdata_loop)
        control_thread.start()

        self._master.state = pysoem.OP_STATE
        self._master.write_state()
        
        if self._master.state_check(pysoem.OP_STATE, timeout=5_000_000) == pysoem.OP_STATE:
            print('SUCCESS: Systems OPERATIONAL.')
            self.in_op = True
            
            check_thread = threading.Thread(target=self._check_thread)
            check_thread.start()

            while not self.shutdown:
                time.sleep(0.1)
        else:
            print('\nERROR: Failed to reach OP state.')
            self.shutdown = True

        # --- 5. Shutdown ---
        self.in_op = False
        self._master.state = pysoem.INIT_STATE
        self._master.write_state()
        self._pd_thread_stop_event.set()
        self._ch_thread_stop_event.set()
        control_thread.join()
        self._master.close()

    @staticmethod
    def _check_slave(slave, pos):
        if slave.state == (pysoem.SAFEOP_STATE + pysoem.STATE_ERROR):
            print(f'ERROR : slave {pos} is in SAFE_OP + ERROR (AL status code: {slave.state}), attempting ack.')
            slave.state = pysoem.SAFEOP_STATE + pysoem.STATE_ACK
            slave.write_state()
        elif slave.state == pysoem.SAFEOP_STATE:
            print(f'WARNING : slave {pos} is in SAFE_OP (AL status code: {slave.state}), try change to OPERATIONAL.')
            slave.state = pysoem.OP_STATE
            slave.write_state()
        elif slave.state > pysoem.NONE_STATE:
            if slave.reconfig():
                slave.is_lost = False
                print(f'MESSAGE : slave {pos} reconfigured')
        elif not slave.is_lost:
            slave.state_check(pysoem.OP_STATE)
            if slave.state == pysoem.NONE_STATE:
                slave.is_lost = True
                print(f'ERROR : slave {pos} lost')
        if slave.is_lost:
            if slave.state == pysoem.NONE_STATE:
                if slave.recover():
                    slave.is_lost = False
                    print(f'MESSAGE : slave {pos} recovered')
            else:
                slave.is_lost = False
                print(f'MESSAGE : slave {pos} found')

    def _check_thread(self):
        while not self._ch_thread_stop_event.is_set():
            if self._master.in_op and ((self._actual_wkc < self._master.expected_wkc) or self._master.do_check_state):
                self._master.do_check_state = False
                self._master.read_state()
                if not self.shutdown:
                    for i, slave in enumerate(self._master.slaves):
                        if slave.state != pysoem.OP_STATE:
                            self._master.do_check_state = True
                            self._check_slave(slave, i)
                    if not self._master.do_check_state:
                        pass
            time.sleep(0.01)

if __name__ == '__main__':
    Master()