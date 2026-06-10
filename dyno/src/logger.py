import h5py
import os
import time
from PySide6.QtCore import QObject
from PySide6.QtCore import *
from PySide6.QtWidgets import *
import numpy as np
import time
import yaml
from deployment import dyno_paths

class Logger:
    def __init__(self, telemetry_queue, mode):

        self.telemetry_queue = telemetry_queue

        with open(f"{dyno_paths.dyno_config_directory}/{mode}_dyno_config.yaml", 'r') as f:
            self.dyno_params = yaml.safe_load(f)

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

        self.log_keys = [entry[0] for entry in self.dyno_params['log_keys']]

        self.telemetry_samples = []

        ideal_chunk_size = 2**13 #Approximately 100 KiB / 8 byte / stored value
        self.chunk_length = ideal_chunk_size / len(self.dyno_params['log_keys'])
        self.chunk_length = 2**(1+int(np.log2(self.chunk_length)))

        self.file = None
        self.data_dset = None
        self.save = False
        self.active_id = None
        self.data_counter = 0

        print('#'*32)
        print('Logging Initialization')
        print('\tBuffer Length / Channel = ', self.chunk_length)
        print('\n')

        self.run()

    def run(self):
        mode_offset = {4: 0, 3: 1, 7: 2}
        while True:
            time.sleep(0.1)

            read_queue = True
            while read_queue:
                try:
                    sample = self.telemetry_queue.get_nowait()
                except:
                    sample = None
                    read_queue = False
                    pass

                if not sample == None:
                    # starts logging
                    if sample[-2]['log'] == True and self.save == False:
                        self.save = True
                    # stop logging and close out the active file
                    elif sample[-2]['log'] == False and self.save == True:
                        self.save = False
                        self.stop_logging()
                        print('Logging stopped')

                    if self.save == True:
                        if self.file == None:
                            self.start_logging()
                        self.data_counter += 1
                        self.telemetry_samples.append(sample[:-2])

                    # first time receiving a behavior ID callouts
                    if 'behavior_id' in sample[-2] and self.active_id != sample[-2]['behavior_id']:
                        self.active_id = sample[-2]['behavior_id']
                        self.dsets['behavior_ids'][self.keys_written] = sample[-2]['behavior_id']
                        self.dsets['behavior_indices'][self.keys_written, 0] = self.data_counter

                        if self.keys_written+1 == self.dsets['behavior_ids'].shape[0]:
                            self.dsets['behavior_ids'].resize(self.dsets['behavior_ids'].shape[0]+1, axis=0)
                            self.dsets['behavior_indices'].resize(self.dsets['behavior_ids'].shape[0]+1, axis=0)
                         
                    # stopped receiving a behavior ID callout
                    elif not 'behavior_id' in sample[-2] and not self.active_id == None:
                        self.active_id = None
                        if not self.file == None:
                            self.dsets['behavior_indices'][self.keys_written,1] = self.data_counter - 1
                        self.keys_written += 1

            # if enough data has buffered to fill a hdf5 chunk
            if len(self.telemetry_samples) > self.chunk_length:
                chunks_to_write = int(len(self.telemetry_samples) / self.chunk_length)
                new_data = np.array(self.telemetry_samples[:chunks_to_write*self.chunk_length]).transpose()

                self.save_cache(new_data)

                if len(self.telemetry_samples) == chunks_to_write*self.chunk_length:
                    self.telemetry_samples = []
                else:
                    self.telemetry_samples = self.telemetry_samples[chunks_to_write*self.chunk_length:]

    def start_logging(self):
        folder_dir = f"{dyno_paths.dyno_logs_directory}/{str(int(time.time()))}"
        os.mkdir(folder_dir)
        f_name = folder_dir+'/log.hdf5'
        self.file = h5py.File(f_name,'w')
        
        # makes the HDF5 file and the datasets within it that are needed
        self.dsets = {}
        self.dsets['behavior_ids'] = self.file.create_dataset('behavior_ids', shape=(10,), maxshape=(None,), dtype=h5py.string_dtype())
        self.dsets['behavior_indices'] = self.file.create_dataset('behavior_indices', shape=(10, 2), maxshape=(None, 2), dtype=np.int32, chunks=True)
        self.keys_written = 0
        for key in self.log_keys:
            self.dsets[key] = self.file.create_dataset(key,shape=(0,), chunks=(self.chunk_length,), maxshape=(None,))


    def save_cache(self, data):
        # to save a chunk of data, just resize the dataset and write in the data
        for ui, key in enumerate(self.log_keys):
            self.dsets[key].resize((self.dsets[key].shape[0]+data.shape[1],))
            self.dsets[key][-data.shape[1]:] = data[ui, :]
                

    def stop_logging(self):
        new_data = np.array(self.telemetry_samples).transpose()
        if not self.active_id == None:
            self.dsets['behavior_indices'][self.keys_written,1] = self.data_counter - 1

        # Get rid of null ID's in dataset
        setpoint_strings = [string for string in self.dsets['behavior_ids'][:] if len(string) > 0]
        self.dsets['behavior_ids'].resize(len(setpoint_strings), axis=0)
        self.dsets['behavior_indices'].resize(len(setpoint_strings), axis=0)

        # save any data in the buffer and close the file
        self.save_cache(new_data)
        self.telemetry_samples = []
        self.file.close()
        self.file = None
        self.data_dset = None
        self.data_counter = 0