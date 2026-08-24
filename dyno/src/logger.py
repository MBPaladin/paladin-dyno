import h5py
import json
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
from dyno.src import setup_summary
from dyno.src.config_utils import augment_log_keys


def log_dir_name(sim=None):
    """Log folder naming convention: (sim_)yyyy-mm-dd_hhmmss, stamped in the
    machine's local timezone. Shared by the Logger (folder creation) and the
    GUI (experiment-notes path)."""
    if sim is None:
        sim = bool(os.environ.get('DYNO_SIM'))
    return ('sim_' if sim else '') + time.strftime('%Y-%m-%d_%H%M%S')


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
        # Set per run in start_logging; used to write the companion report.
        self.resolved = None
        self.log_folder = None
        self.log_base = None
        self.save = False
        self.active_id = None
        self.data_counter = 0
        # Session id off the live telemetry, checked against the one baked into
        # resolved_config.json at start_logging; and the resulting complaint, if
        # they disagree, held for the run's setup report.
        self.live_session_id = None
        self.config_warning = None
        # Session tare off the live telemetry, stamped onto each log as it
        # opens. The controller refuses to tare while a test is running, so the
        # bias cannot change mid-file and one scalar per sensor describes the
        # whole log.
        self.live_tare = None
        # How the run being closed ended, set by stop_logging for _report_meta.
        self.stop_reason = None
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
                    # Slot -1 is the controller's state dict. The id only ever
                    # changes at bring-up, so keeping the latest is enough to
                    # have it in hand whenever a log opens.
                    if isinstance(sample[-1], dict):
                        self.live_session_id = sample[-1].get('session_id')
                        self.live_tare = sample[-1].get('tare')

                    # starts logging
                    if sample[-2]['log'] == True and self.save == False:
                        self.save = True
                    # stop logging and close out the active file
                    elif sample[-2]['log'] == False and self.save == True:
                        self.save = False
                        self.stop_logging(sample[-2].get('stop_reason'))
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

        # File: named after the test yaml that ran, so the log type is
        # readable at a glance.
        test_name = self.meta.get('test_name') or 'log'
        base = os.path.splitext(os.path.basename(test_name))[0] or 'log'
        f_name = f"{folder_dir}/{base}.hdf5"
        self.file = h5py.File(f_name,'w')
        self.log_folder = folder_dir
        self.log_base = base

        # Attach the resolved device configuration (written by the master at
        # bring-up) so every log records exactly what parameters ran. Kept
        # parsed as well, to render the companion report when the log closes.
        resolved_path = f"{dyno_paths.dyno_logs_directory}/resolved_config.json"
        self.resolved = None
        self.config_warning = None
        if os.path.exists(resolved_path):
            with open(resolved_path, 'r') as f:
                raw = f.read()
            self.file.attrs['resolved_config'] = raw
            try:
                self.resolved = json.loads(raw)
            except ValueError as e:
                self.config_warning = f'resolved_config.json is not valid JSON ({e})'
                print(f'Logger: {self.config_warning}; '
                      'skipping the companion setup report')
        else:
            # Silence here used to mean no setup report and no explanation --
            # the run looked fine until someone went looking for the .txt.
            self.config_warning = ('no resolved_config.json on disk, so this log '
                                   'carries no record of the configuration it ran under')
            print(f'Logger: {self.config_warning}')

        if self.resolved is not None:
            self._check_config_is_this_session()

        # The session tare goes in its own attribute rather than into
        # resolved_config: that record is frozen at bring-up, before any tare
        # exists, and a runtime correction must never be mistaken for the
        # calibration constants alongside it. Absent attribute == no tare
        # applied, which is what every log written before this feature means.
        if self.live_tare:
            self.file.attrs['tare'] = json.dumps(self.live_tare, default=str)

        # makes the HDF5 file and the datasets within it that are needed
        self.dsets = {}
        self.dsets['behavior_ids'] = self.file.create_dataset('behavior_ids', shape=(10,), maxshape=(None,), dtype=h5py.string_dtype())
        self.dsets['behavior_indices'] = self.file.create_dataset('behavior_indices', shape=(10, 2), maxshape=(None, 2), dtype=np.int32, chunks=True)
        self.keys_written = 0
        for key in self.log_keys:
            self.dsets[key] = self.file.create_dataset(key,shape=(0,), chunks=(self.chunk_length,), maxshape=(None,), dtype='f4')


    def _check_config_is_this_session(self):
        """Warn when the config we just attached came from a DIFFERENT bring-up
        than the one currently feeding us telemetry.

        There is exactly one resolved_config.json and every bring-up overwrites
        it, so a Logger running without a fresh one -- a crashed bring-up, a
        controller restarted underneath us -- would otherwise attach a
        plausible-looking but wrong configuration and say nothing at all. Both
        ids originate in the same Master.run, so an honest pairing always
        agrees.

        Warn, never refuse: the data is still worth keeping, and the operator is
        the one who can judge whether the mismatch matters."""
        stamped = self.resolved.get('session_id')
        if stamped is None or self.live_session_id is None:
            return  # a config from before session ids, or telemetry without one
        if stamped != self.live_session_id:
            self.config_warning = (
                f'resolved_config.json was written by bring-up {stamped}, but the '
                f'running controller is {self.live_session_id} -- the setup '
                'recorded in this log may not be the setup that actually ran')
            print(f'Logger: WARNING - {self.config_warning}')

    def save_cache(self, data):
        # to save a chunk of data, just resize the dataset and write in the data
        for ui, key in enumerate(self.log_keys):
            self.dsets[key].resize((self.dsets[key].shape[0]+data.shape[1],))
            self.dsets[key][-data.shape[1]:] = data[ui, :]
                

    def stop_logging(self, stop_reason=None):
        """Close out the active file, recording how the run ended.

        `stop_reason` is the controller's dict (see DynoController._stop_test):
        a safety trip, a drive fault, an operator stop, or a clean finish. It
        is written both as the machine-readable `stop_reason` attribute and, via
        the report meta, into the setup overview an operator actually reads.

        None means the controller said nothing -- an older build, or a stop that
        bypassed _stop_test -- and is recorded as absence, never as success."""
        self.stop_reason = stop_reason
        if stop_reason:
            self.file.attrs['stop_reason'] = json.dumps(stop_reason, default=str)

        new_data = np.array(self.telemetry_samples).transpose()
        if not self.active_id == None:
            self.dsets['behavior_indices'][self.keys_written,1] = self.data_counter - 1

        # Get rid of null ID's in dataset
        setpoint_strings = [string for string in self.dsets['behavior_ids'][:] if len(string) > 0]
        self.dsets['behavior_ids'].resize(len(setpoint_strings), axis=0)
        self.dsets['behavior_indices'].resize(len(setpoint_strings), axis=0)

        # save any data in the buffer and close the file. The buffer can be
        # empty here: the flush in run() clears it outright when a chunk lands
        # exactly on the boundary, so a stop arriving on the very next sample
        # leaves nothing. np.array([]) has no second axis, and save_cache would
        # raise before the close below -- taking the whole log with it.
        if new_data.ndim == 2 and new_data.shape[1]:
            self.save_cache(new_data)
        self.telemetry_samples = []

        # Read the run's shape while the datasets are still open; the report
        # quotes duration and sample count.
        report_meta = self._report_meta()

        self._write_setup_group(report_meta)

        self.file.close()
        self.file = None
        self.data_dset = None
        self.data_counter = 0

        self._write_companion_report(report_meta)

    def _report_meta(self):
        meta = {'test_name': self.log_base,
                'log_dir': os.path.basename(self.log_folder or ''),
                'logged_at': time.strftime('%Y-%m-%d %H:%M:%S')}
        if self.config_warning:
            meta['config_warning'] = self.config_warning
        if self.live_tare:
            meta['tare'] = self.live_tare
        if self.stop_reason:
            meta['stop_reason'] = self.stop_reason
        time_dset = self.dsets.get('time')
        if time_dset is not None and time_dset.shape[0]:
            meta['samples'] = f'{time_dset.shape[0]:,}'
            meta['duration'] = f'{float(time_dset[-1]) - float(time_dset[0]):.1f} s'
        return meta

    def _write_setup_group(self, meta):
        """Render the same setup tables into /setup inside the log itself, so
        the record travels with the data: opened anywhere, by anyone, the file
        explains what ran without the companion .txt beside it.

        Runs before the close in stop_logging, while the file is still open.
        Same rule as the companion report -- a broken table must never cost a
        test -- so anything thrown here is reported and swallowed, leaving the
        close (and the data) untouched."""
        if self.resolved is None:
            return
        try:
            setup_summary.write_setup(self.file, self.resolved, meta)
        except Exception as e:
            print(f'Failed to write the /setup group: {e}')

    def _write_companion_report(self, meta):
        """Write the run's <test>.txt setup report as soon as the log closes,
        so a run still has a readable record when the operator skips the notes
        dialog or the app aborts before it ever appears.

        notes=None keeps anything already in the file. That matters because the
        GUI opens the notes dialog BEFORE we get here -- it forwards the stop
        sample to our queue and prompts in the same tick, while our run loop
        sleeps up to 100 ms before picking it up -- so a fast operator can save
        notes first. Neither writer has to win that race.

        A broken report is never allowed to cost a test: the hdf5 is closed and
        safe by this point, so anything thrown here is reported and swallowed."""
        if self.resolved is None:
            return
        try:
            path = setup_summary.report_path(self.log_folder, self.log_base)
            setup_summary.write_report(path, self.resolved, meta)
            print(f'Setup report written to {os.path.basename(path)}')
        except Exception as e:
            print(f'Failed to write setup report: {e}')