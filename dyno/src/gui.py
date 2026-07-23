import time
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import *
from PySide6.QtWidgets import *
from PySide6.QtGui import QDesktopServices
import sys
import os
import shutil
import signal
import subprocess
import multiprocessing
import yaml

def resolve_mode(argv):
    """Rig config selection: --config <name> (matching <name>_dyno_config.yaml),
    with the legacy flags (--gearbox etc.) kept as aliases."""
    mode = next((a.lstrip('-') for a in argv
                 if a in ('--gearbox', '--actuator', '--actuator_production')), None)
    if '--config' in argv:
        idx = argv.index('--config')
        mode = argv[idx + 1] if idx + 1 < len(argv) else None
    return mode


# Simulation sandbox: DYNO_SIM must be in the environment BEFORE the dyno
# imports below, because master.py chooses real-vs-fake pysoem at import time
# (and the forked Controller child inherits the parent's imported modules).
if '--sim' in sys.argv:
    _m = resolve_mode(sys.argv)
    if _m:
        os.environ['DYNO_SIM'] = _m
    print('#### SIMULATION MODE ####')

from dyno.src import logger
from dyno.src.logger import Logger
from dyno.src.dyno_controller import Controller
from dyno.src.config_utils import augment_log_keys, chown_to_invoking_user

from deployment import dyno_paths

class Window(QWidget):

    cli_command = Signal(object)

    def __init__(self, mode, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

        self.cli_command.connect(self._handle_cli_command)

        qApp = QApplication.instance()
        self.num_plots = 3 # The number of scopes can be increased or decreased here. Performance may be impared above 3
        self.mode = mode

        # Load parameters
        with open(f"{dyno_paths.dyno_config_directory}/{mode}_plot_config.yaml", 'r') as f:
            self.plot_params = yaml.safe_load(f)

        with open(f"{dyno_paths.dyno_config_directory}/{mode}_dyno_config.yaml", 'r') as f:
            self.dyno_params = yaml.safe_load(f)

        with open(f"{dyno_paths.dyno_config_directory}/master_config.yaml", 'r') as f:
            self.master_params = yaml.safe_load(f)

        # calculate decimation for gui buffer,
        self.gui_decimation = int(self.dyno_params['gui_params']['window_length_s']*1e6/self.master_params['cycle_time_us']/self.dyno_params['gui_params']['displayed_samples'])
        buffer_length = int(self.dyno_params['gui_params']['window_length_s'] * 1e6 / self.master_params['cycle_time_us'] / self.gui_decimation)

        # Real time between two decimated buffer samples (seconds). Used to seed the
        # time axis and to scroll it. NOTE: this is the control period * decimation,
        # NOT the 30 ms GUI timer period.
        self.window_length_s = self.dyno_params['gui_params']['window_length_s']
        self.gui_dt = self.gui_decimation * self.master_params['cycle_time_us'] / 1e6

        # add configured sensor keys (shared builder — must match Controller/Logger)
        self.log_keys = augment_log_keys(self.dyno_params, verbose=True)
        print("LOG_KEYS: item count = ", len(self.log_keys))
        for key in self.log_keys:
            print('\t',key)

        # make data buffer, one row for each item in the log keys
        self.gui_data = np.zeros((len(self.log_keys), buffer_length))
        # Seed the time row as one continuous window ending just before 0, spaced at
        # the true decimated-sample period. This makes the axis a full, correctly
        # scaled window at launch instead of ~64 s of mis-spaced fake history.
        self.gui_data[0,:] = np.arange(-buffer_length, 0)*self.gui_dt

        # make a buffer for incoming telementry
        self.telemetry_samples = []

        # log-flag transition tracking (for log folder naming + notes prompt)
        self._log_active = False
        self._active_log_dir = None
        self._active_log_test = None
        self._notes_prompt_pending = False

        # initialize ui
        self.__build_ui()

        self.telemetry_queue = multiprocessing.Queue(maxsize=0) # handles controller to gui data transfer
        self.logging_queue = multiprocessing.Queue(maxsize=0) # forwards telemetry to the logging class, this avoids having the control thread write to 2 queues
        self.control_command_queue = multiprocessing.Queue(maxsize=0) # handles gui to controller command transfer

        self.logging_process = multiprocessing.Process(target=Logger, args=[self.logging_queue, mode] ,name='LoggingProcess')
        self.logging_process.start()

        self.control_process = multiprocessing.Process(target=Controller, args=[self.telemetry_queue, self.control_command_queue, mode] ,name='ControlProcess')
        self.control_process.start()

        # set niceness slightly lower to improve performance
        # current_niceness = os.nice(0)
        # target_niceness = -2
        # os.nice(target_niceness - current_niceness)

        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(10))
            cpus = {7, 8, 9, 10}
            os.sched_setaffinity(0, cpus)
            print("GUI: Real-time scheduling enabled")
        except PermissionError:
            print("GUI: Real-time scheduling not permitted, running normally")

        self.timer = pg.Qt.QtCore.QTimer()
        self.timer.timeout.connect(self.update_data)
        self.timer.start(30)

        # when you close the gui it calls this method, which closes out all multiprocesses
        QApplication.instance().aboutToQuit.connect(self.close_processes)

        if '--load_test' in sys.argv:
            idx = sys.argv.index('--load_test')
            test_name = sys.argv[idx + 1]
            self.test_select.setCurrentText(test_name)

    def cli(self, *cmd):
        self.cli_command.emit(cmd)

    def _handle_cli_command(self, cmd: str):
        args = cmd[0].strip().split()

        if not args:
            return

        if args[0] == 'start_test':
            self.__start_test()

        elif args[0] == 'stop_test':
            self.__stop_test()

        elif args[0] == 'load_test' and len(args) > 1:
            self.test_select.setCurrentText(args[1])
            self.__load_test()

        elif args[0] == 'shutdown':
            self.close_processes()
            QApplication.quit()

    def close_processes(self):
        self.control_command_queue.put_nowait(['shutdown', 0])
        print('\nShutdown request sent to control thread')

        # joining is a good way to check that the thread terminated.
        if self.logging_process.is_alive():
            self.logging_process.terminate()
            self.logging_process.join()
            print('Logging process terminated')

        # Wait for the control loop to act on the shutdown command (disable the
        # drives, bring EtherCAT to INIT, exit its run loop) and exit on its own.
        # Only force-kill if it overruns, so a clean teardown is never cut short
        # mid-sequence.
        shutdown_timeout_s = 5
        self.control_process.join(timeout=shutdown_timeout_s)

        if self.control_process.is_alive():
            print(f'Control process did not exit within {shutdown_timeout_s}s, terminating')
            self.control_process.terminate()
            self.control_process.join()
            print('Control process terminated')
        else:
            print('Control process shut down cleanly')

    def __build_ui(self):
        self.main_layout = QHBoxLayout(self)
        self.controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_widget)

        self.plot_widget = pg.GraphicsLayoutWidget()

        title_label = QLabel('Scope Selection', alignment=Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet('font-size: 16px;')
        self.controls_layout.addWidget(title_label)

        # Make N many scopes, and their corresponding drop down selectors
        self.plots = []
        for i in range(self.num_plots):
            plot_key = list(self.plot_params.keys())[i]
            plot_params = self.plot_params[plot_key]

            # Create the drop down selector to change what the scope shows
            selection_box = QComboBox()
            selection_box.addItems(self.plot_params.keys())
            selection_box.setCurrentText(plot_key)
            selection_box.currentIndexChanged.connect(self.__change_scopes)

            # creat thge plot. the y range is defined from the plot_config file
            plot = self.plot_widget.addPlot(row=i, col=0, title=plot_params['title'])
            # plot.getViewBox().setYRange(plot_params['range'][0], plot_params['range'][1])
            plot.setLabel('left', plot_params['unit'])
            plot.showGrid(x=True, y=True, alpha=0.5)
            if i == 0:
                # We drive the x-range ourselves (rolling window in redraw), so keep
                # pyqtgraph from auto-ranging x and warping the axis during start-up.
                plot.getViewBox().enableAutoRange(x=False)
            if i > 0:
                plot.setXLink(self.plots[0]['plot']) # link x range of all other plots to the first
            legend = pg.LegendItem(offset=(80, 10))
            legend.setParentItem(plot)

            # for each curve that should be on the graph, load that trace from the data buffer and plot it
            curves = []
            for ui, data_key in enumerate(plot_params['data_keys']):               
                curve = plot.plot(self.gui_data[0,:], self.trace(data_key), pen=plot_params['pens'][ui], name=plot_params['legends'][ui])
                legend.addItem(curve, plot_params['legends'][ui])
                curves.append(curve)

            # create a dictionary of the objects associated with the scope in order to adjust them later
            plot = {
                'plot':plot,
                'selector':selection_box,
                'curves': curves,
                'data_keys':self.plot_params[plot_key]['data_keys']
            }

            # append that dictionary to a list of plots, and add the plot to the gui
            self.plots.append(plot)
            self.controls_layout.addWidget(selection_box)

        # --- Test Selection section: centered header, launcher button beneath
        # (same size as Start/Stop), then the Quick Select dropdown. ---
        test_select_title = QLabel('Test Selection', alignment=Qt.AlignmentFlag.AlignCenter)
        test_select_title.setStyleSheet('font-size: 16px;')
        self.controls_layout.addWidget(test_select_title)

        self.open_test_def_button = QPushButton('Open Test Builder')
        self.open_test_def_button.clicked.connect(self.__open_test_definition)
        self.controls_layout.addWidget(self.open_test_def_button)

        self.test_select = QComboBox()
        self.test_select.addItems([f for f in os.listdir(dyno_paths.dyno_test_directory) if f[-4:] == 'yaml'])
        self.test_select.setPlaceholderText('Quick Select')
        self.test_select.setCurrentIndex(-1)  # show the placeholder, arm no test
        self.test_select.currentIndexChanged.connect(self.__load_test)
        self.controls_layout.addWidget(self.test_select)

        title_label = QLabel('Test Controls', alignment=Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet('font-size: 16px;')
        self.controls_layout.addWidget(title_label)

        self.start_button = QPushButton('Start')
        self.start_button.setStyleSheet('background-color: green; color: white;')
        self.controls_layout.addWidget(self.start_button)
        self.start_button.clicked.connect(self.__start_test)

        self.stop_button = QPushButton('Stop')
        self.stop_button.setStyleSheet('background-color: red; color: white;')
        self.controls_layout.addWidget(self.stop_button)
        self.stop_button.clicked.connect(self.__stop_test)

        self.__build_safeties_panel()

        self.controls_layout.addStretch(1)
        self.main_layout.addWidget(self.plot_widget, stretch=5)
        self.main_layout.addWidget(self.controls_widget, stretch=1)

    def __build_safeties_panel(self):
        """Collapsible read-only view of the config's `safeties:` section, one
        row per check: name, live value vs limit (colored by margin), and an
        'edit' link that opens the rig config in VS Code at that entry's line.
        The panel is display-only on purpose — limits are changed by editing
        the config file itself, so the YAML stays the single source of truth."""
        self._config_path = f"{dyno_paths.dyno_config_directory}/{self.mode}_dyno_config.yaml"

        self.safeties_toggle = QPushButton(f'▸ Safeties')
        self.safeties_toggle.setCheckable(True)
        self.safeties_toggle.setStyleSheet('text-align: left; font-size: 16px; border: none;')
        self.controls_layout.addWidget(self.safeties_toggle)

        panel = QWidget()
        grid = QGridLayout(panel)
        grid.setContentsMargins(12, 0, 0, 0)
        panel.setVisible(False)
        self.controls_layout.addWidget(panel)

        def toggle(checked):
            panel.setVisible(checked)
            self.safeties_toggle.setText(('▾' if checked else '▸') + ' Safeties')
        self.safeties_toggle.toggled.connect(toggle)

        # Map each safety's telemetry source path back to its log_keys row so
        # live values come straight from the existing gui_data buffer (the
        # [name, path] pairs in dyno_params['log_keys'] include the
        # augment_log_keys sensor entries).
        path_to_row = {path: self.log_keys.index(name)
                       for name, path in self.dyno_params.get('log_keys', [])
                       if name in self.log_keys}

        self._safety_rows = []  # (value_label, buffer_row_or_None, limit)
        for i, (name, spec) in enumerate(self.dyno_params.get('safeties', {}).items()):
            if not isinstance(spec, dict) or 'limit' not in spec:
                continue
            limit = abs(spec['limit'])
            grid.addWidget(QLabel(name), i, 0)
            value_label = QLabel('—')
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            grid.addWidget(value_label, i, 1)
            edit_link = QLabel(f'<a href="{i}">edit</a>')
            edit_link.linkActivated.connect(
                lambda _, n=name: self.__open_config_at_safety(n))
            grid.addWidget(edit_link, i, 2)
            self._safety_rows.append(
                (value_label, path_to_row.get(spec.get('source')), limit))

    def __open_config_at_safety(self, safety_name):
        """Open the rig config in VS Code at the named safety's line
        (code -g file:line). Falls back to the OS default handler for yaml
        files if the `code` CLI isn't on PATH."""
        line = 1
        try:
            with open(self._config_path) as f:
                lines = f.readlines()
            in_safeties = False
            for ln, text in enumerate(lines, start=1):
                stripped = text.strip()
                if stripped.startswith('safeties:'):
                    in_safeties = True
                elif in_safeties and text[:1] not in (' ', '\t', '#', '\n'):
                    break  # left the safeties block
                elif in_safeties and stripped.startswith(f'{safety_name}:'):
                    line = ln
                    break
        except OSError:
            pass
        # Under sudo, PATH loses the user's `code` shim (in WSL it's appended
        # by Windows interop, so even a login shell can't recover it). The
        # launch script resolves it pre-sudo into DYNO_CODE_CLI.
        code = os.environ.get('DYNO_CODE_CLI') or shutil.which('code')
        if code:
            subprocess.Popen([code, '-g', f'{self._config_path}:{line}'])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._config_path))

    def __update_safeties_panel(self):
        for value_label, buffer_row, limit in self._safety_rows:
            if buffer_row is None:
                value_label.setText(f'— / {limit:g}')
                continue
            value = abs(self.gui_data[buffer_row, -1])
            frac = value / limit if limit else 1.0
            color = 'red' if frac >= 1.0 else ('orange' if frac >= 0.8 else 'green')
            value_label.setText(f'{value:.2f} / {limit:g}')
            value_label.setStyleSheet(f'color: {color};')

    def __start_test(self):
        self.control_command_queue.put_nowait(['start_test', 0])

    def __stop_test(self):
        self.control_command_queue.put_nowait(['stop_test', 0])

    def __prompt_experiment_notes(self):
        """Shown when a test stops gracefully (stop button, completion, or a
        safety stop). Non-empty notes are saved as <test>.txt next to the log
        hdf5; empty/cancelled leaves no file. An app abort never gets here, so
        no file is written in that case either. Typing just 'delete' (case
        insensitive) deletes the run's entire log folder instead.

        Shown non-modally via open() rather than exec(): exec() would suspend
        update_data() (this is called from the 30 ms timer), freezing the
        plots and telemetry drain until the dialog closes."""
        log_dir = self._active_log_dir
        if log_dir is None:
            return
        test_name = self._active_log_test
        dialog = QDialog(self)
        dialog.setWindowTitle('Experiment Notes')
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f'Test finished: {test_name}\n'
                                f'Log folder: {log_dir}\n\n'
                                'Anything worth remembering about this run?\n'
                                "(type just 'delete' to discard this run's log)"))
        text_edit = QTextEdit()
        layout.addWidget(text_edit)
        buttons = QHBoxLayout()
        save_btn = QPushButton('Save Notes')
        save_btn.clicked.connect(dialog.accept)
        skip_btn = QPushButton('Skip')
        skip_btn.clicked.connect(dialog.reject)
        buttons.addWidget(save_btn)
        buttons.addWidget(skip_btn)
        layout.addLayout(buttons)
        text_edit.setFocus()
        # Keep a reference so the dialog isn't GC'd while open.
        self._notes_dialog = dialog

        def on_finished(result):
            self._notes_dialog = None
            notes = text_edit.toPlainText().strip() if result else ''
            if notes:
                self.__save_experiment_notes(log_dir, test_name, notes)

        dialog.finished.connect(on_finished)
        dialog.open()
        # Center over the main window; some window managers ignore the
        # parent hint and throw the dialog onto another screen.
        dialog.adjustSize()
        geo = dialog.frameGeometry()
        geo.moveCenter(self.frameGeometry().center())
        dialog.move(geo.topLeft())

    def __save_experiment_notes(self, log_dir, test_name, notes):
        folder = f"{dyno_paths.dyno_logs_directory}/{log_dir}"
        if notes.lower() == 'delete':
            try:
                if os.path.isdir(folder):
                    shutil.rmtree(folder)
                    print(f'Deleted log folder {log_dir} (per experiment notes)')
                else:
                    print(f'Log folder {log_dir} not found, nothing to delete')
            except OSError as e:
                print(f'Failed to delete log folder {log_dir}: {e}')
            return
        base = os.path.splitext(os.path.basename(test_name or 'log'))[0] or 'log'
        try:
            os.makedirs(folder, exist_ok=True)
            with open(f'{folder}/{base}.txt', 'w') as f:
                f.write(notes + '\n')
            chown_to_invoking_user(folder, f'{folder}/{base}.txt')
            print(f'Experiment notes saved to {log_dir}/{base}.txt')
        except OSError as e:
            print(f'Failed to save experiment notes: {e}')

    def __load_test(self):
        self.control_command_queue.put_nowait(['test_def', (self.test_select.currentText(), self.mode)])
        # Tell the Logger which test is armed so it can name the log file
        # after the test yaml.
        self.logging_queue.put_nowait({'test_name': self.test_select.currentText()})

    def __open_test_definition(self):
        # Lazy import: keeps GUI start-up light and avoids a hard dependency
        # when running headless. Keep a reference so the window isn't GC'd.
        from dyno.src.test_builder_window import TestBuilderWindow
        if getattr(self, '_test_def_window', None) is None:
            self._test_def_window = TestBuilderWindow(self.mode)
            self._test_def_window.test_loaded.connect(self.__on_test_def_loaded)
        self._test_def_window.show()
        self._test_def_window.raise_()
        self._test_def_window.activateWindow()

    def __on_test_def_loaded(self, test_file):
        # A test previewed successfully in the definition window -> mirror it into
        # the Quick Select dropdown (which arms it on the controller).
        if self.test_select.findText(test_file) < 0:
            self.test_select.addItem(test_file)
        self.test_select.setCurrentText(test_file)

    # called if you change which scope is selected in the dropdown
    def __change_scopes(self):
        # Find the changed scope
        inds_changed = [i for i in range(self.num_plots) if not self.plots[i]['plot'].titleLabel.text == self.plots[i]['selector'].currentText()]
        
        for i in inds_changed:
            plot_key = self.plots[i]['selector'].currentText()
            plot_params = self.plot_params[plot_key]

            self.plots[i]['plot'].setTitle(plot_params['title'])
            self.plots[i]['data_keys'] = plot_params['data_keys']
            # self.plots[i]['plot'].getViewBox().setYRange(plot_params['range'][0], plot_params['range'][1])
            self.plots[i]['plot'].setLabel('left', plot_params['unit'])

            if not len(self.plots[i]['curves']) == len(self.plots[i]['data_keys']):
                legend = pg.LegendItem(offset=(80, 10)) # Adjust offset as needed
                legend.setParentItem(self.plots[i]['plot'])

                curves = []
                for ui, data_key in enumerate(plot_params['data_keys']):               
                    curve = self.plots[i]['plot'].plot(self.gui_data[0,:], self.trace(data_key), pen=plot_params['pens'][ui], name=plot_params['legends'][ui])
                    legend.addItem(curve, plot_params['legends'][ui])
                    curves.append(curve)
                
                self.plots[i]['curves'] = curves

    def trace(self, trace_key):
        if trace_key in self.log_keys:
            return self.gui_data[self.log_keys.index(trace_key), :]
        # fallback: NaNs so it doesn't draw misleading zeros
        return np.full(self.gui_data.shape[1], np.nan, dtype=float)

    def update_data(self):
        mode_offset = {4: 0, 3: 1, 7: 2} # lookup table to determine how to map a command for a given mode into the telemetry sample. This is done mainly so that the telemetry sample can return the control mode + command, vs. the control mode, + command + 2 more NAN commands

        # pull samples from telemetry queue
        read_queue = True
        while read_queue:
            try:
                sample = self.telemetry_queue.get_nowait()
                # Track log-flag transitions in the stream we forward. On
                # start, stamp the log folder name and send it AHEAD of the
                # first logged sample so the Logger and this GUI agree on the
                # folder (notes are written next to the hdf5). On stop, queue
                # the experiment-notes prompt (shown after the drain loop).
                log_flag = sample[-2].get('log', False)
                if log_flag and not self._log_active:
                    self._log_active = True
                    self._active_log_dir = logger.log_dir_name()
                    self._active_log_test = self.test_select.currentText()
                    self.logging_queue.put_nowait({'log_dir': self._active_log_dir})
                elif not log_flag and self._log_active:
                    self._log_active = False
                    self._notes_prompt_pending = True
                self.logging_queue.put_nowait(sample) #forward sample to the logging thread
                self.telemetry_samples.append(sample[:-2])
            except:
                read_queue = False
                pass

        if self._notes_prompt_pending:
            self._notes_prompt_pending = False
            self.__prompt_experiment_notes()


        new_gui_samples = int(len(self.telemetry_samples) / self.gui_decimation) # determine how many sample in the data buffer will get replaced
        if new_gui_samples >= 1:
            self.gui_data[:,:-new_gui_samples] = self.gui_data[:,new_gui_samples:] #shift over gui date array
            new_data = np.array(self.telemetry_samples[:self.gui_decimation*new_gui_samples]).transpose()
            # print(self.gui_data[:,-new_gui_samples:].shape, new_data[:,::self.gui_decimation].shape)
            self.gui_data[:,-new_gui_samples:] = new_data[:,::self.gui_decimation]

            # remove decimated samples from the telemetry_samples buffer
            if len(self.telemetry_samples) == new_gui_samples*self.gui_decimation:
                self.telemetry_samples = []
            else:
                self.telemetry_samples = self.telemetry_samples[new_gui_samples*self.gui_decimation:]

        self.__update_safeties_panel()
        self.redraw()

    def redraw(self):
        draw_start = time.time()

        # Scroll the x-axis as a fixed-width window ending at the latest sample. All
        # plots are x-linked to plots[0], so setting it here moves every scope. This
        # replaces auto-ranging, which warped the axis while the seeded start-up
        # samples flushed out of the buffer.
        t_latest = self.gui_data[0, -1]
        self.plots[0]['plot'].setXRange(t_latest - self.window_length_s, t_latest, padding=0)

        for plot in self.plots:
            times = []
            for curve, trace_id in zip(plot['curves'], plot['data_keys']):
                start = time.time()
                curve.setData(self.gui_data[0,:], self.trace(trace_id))
                times.append(np.round(time.time() - start, 5))
            plot['update_time'] = times


        draw_end = time.time()
        draw_time = draw_end - draw_start
        # if draw_time > self.longest_draw:
        if draw_time > 0.02:
            print('New longest draw time = ',draw_time)


if __name__=='__main__':
    automated = False

    mode = resolve_mode(sys.argv)
    if mode is None or not os.path.exists(
            f"{dyno_paths.dyno_config_directory}/{mode}_dyno_config.yaml"):
        available = sorted(f[:-len('_dyno_config.yaml')]
                           for f in os.listdir(dyno_paths.dyno_config_directory)
                           if f.endswith('_dyno_config.yaml'))
        print(f'Launch Aborted: no rig config selected (got {mode!r}).')
        print(f'Use --config <name> (or legacy flag). Available: {available}')
        sys.exit(1)

    if '--automated' in sys.argv:
        automated = True

    a = QApplication(sys.argv)
    g = Window(mode)
    
    if automated:
        from dyno.src import automation_test
        automation_test.handle_cli_test_commands(g)
        g.showFullScreen()
    else:
        g.show()

    # Route a terminal Ctrl-C through Qt's clean shutdown (aboutToQuit ->
    # close_processes), so the control child is told to safely disable the drives
    # instead of being orphaned. Qt's C++ event loop doesn't service Python
    # signals on its own, so a periodic no-op timer wakes the interpreter often
    # enough to deliver the pending SIGINT.
    signal.signal(signal.SIGINT, lambda *_: a.quit())
    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(200)

    sys.exit(a.exec())
