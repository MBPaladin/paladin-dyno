import h5py
import os
import shutil
import signal
import time
from PySide6.QtCore import QObject
from PySide6.QtCore import *
from PySide6.QtWidgets import *
import numpy as np
import time
import yaml
from deployment import dyno_paths
from dyno.src.config_utils import augment_log_keys, chown_to_invoking_user


def log_dir_name(sim=None):
    """Log folder naming convention: (sim_)yyyy_mm_dd_hh_mm_ss, stamped in
    Pacific time regardless of the machine's timezone. Shared by the Logger
    (folder creation) and the GUI (experiment-notes path)."""
    if sim is None:
        sim = bool(os.environ.get('DYNO_SIM'))
    try:
        from zoneinfo import ZoneInfo
        from datetime import datetime
        stamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d_%H%M%S')
    except Exception:  # tz database unavailable: fall back to machine-local
        stamp = time.strftime('%Y_%m_%d_%H_%M_%S')
    return ('sim_' if sim else '') + stamp


class Logger:
    def __init__(self, telemetry_queue, mode):
        # Child process: ignore SIGINT and let the parent GUI shut us down
        # cleanly (via terminate() in close_processes), so a Ctrl-C doesn't kill
        # us mid-write and truncate the log file.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.telemetry_queue = telemetry_queue

        with open(f"{dyno_paths.dyno_config_directory}/{mode}_dyno_config.yaml", 'r') as f:
            self.dyno_params = yaml.safe_load(f)

        # shared builder — must produce the same ordering as GUI/Controller
        self.log_keys = augment_log_keys(self.dyno_params)

        self.telemetry_samples = []

        ideal_chunk_size = 2**13 #Approximately 100 KiB / 8 byte / stored value
        self.chunk_length = ideal_chunk_size / len(self.dyno_params['log_keys'])
        self.chunk_length = 2**(1+int(np.log2(self.chunk_length)))

        self.file = None
        self.data_dset = None
        self.save = False
        self.active_id = None
        self.data_counter = 0
        # Naming metadata pushed by the GUI through the logging queue (dict
        # sentinels among the list-typed telemetry samples): 'test_name' when a
        # test is armed, 'log_dir' just before the first logged sample.
        self.meta = {}

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

                if isinstance(sample, dict):
                    # metadata sentinel from the GUI, not a telemetry sample
                    self.meta.update(sample)
                    sample = None

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
        # Folder: (sim_)yyyy_mm_dd_hh_mm_ss. The GUI stamps 'log_dir' just
        # before the first logged sample so the folder it shows (and writes
        # experiment notes into) matches ours; fall back to stamping locally.
        folder = self.meta.get('log_dir') or log_dir_name()
        folder_dir = f"{dyno_paths.dyno_logs_directory}/{folder}"
        if os.path.exists(folder_dir):  # same-second rerun: replace the old run
            shutil.rmtree(folder_dir)
        os.makedirs(folder_dir)
        chown_to_invoking_user(dyno_paths.dyno_logs_directory, folder_dir)

        # File: named after the test yaml that ran, so the log type is
        # readable at a glance.
        test_name = self.meta.get('test_name') or 'log'
        base = os.path.splitext(os.path.basename(test_name))[0] or 'log'
        f_name = f"{folder_dir}/{base}.hdf5"
        self.file = h5py.File(f_name,'w')
        chown_to_invoking_user(f_name)

        # Attach the resolved device configuration (written by the master at
        # bring-up) so every log records exactly what parameters ran.
        resolved_path = f"{dyno_paths.dyno_logs_directory}/resolved_config.json"
        if os.path.exists(resolved_path):
            with open(resolved_path, 'r') as f:
                self.file.attrs['resolved_config'] = f.read()
        
        # makes the HDF5 file and the datasets within it that are needed
        self.dsets = {}
        self.dsets['behavior_ids'] = self.file.create_dataset('behavior_ids', shape=(10,), maxshape=(None,), dtype=h5py.string_dtype())
        self.dsets['behavior_indices'] = self.file.create_dataset('behavior_indices', shape=(10, 2), maxshape=(None, 2), dtype=np.int32, chunks=True)
        self.keys_written = 0
        for key in self.log_keys:
            self.dsets[key] = self.file.create_dataset(key,shape=(0,), chunks=(self.chunk_length,), maxshape=(None,), dtype='f4')


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