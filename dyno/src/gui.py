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


# Both of the settings below work around one root cause: setup.sh grants the
# venv interpreter file capabilities (cap_net_raw etc), so the kernel execs us
# with AT_SECURE=1 and dumpable=2, which leaves /proc/<pid>/* owned by root.
# xdg-desktop-portal probes /proc/<pid>/root to classify the caller, cannot open
# it, and denies every request:
#   "Portal operation not allowed: Unable to open /proc/<pid>/root"
# Dropping the capabilities is not an option - the GUI itself calls
# sched_setscheduler(SCHED_FIFO) and forks the EtherCAT Controller, which
# inherits its privileges. PR_SET_DUMPABLE(1) restores the /proc ownership but
# the portal still refuses, so both fixes below avoid the portal instead.

# On Wayland, GNOME does not decorate Qt windows; Qt draws its own title bar and
# reads the button layout from the (denied) Settings portal, so the window ends
# up with no close/minimize/maximize buttons. XWayland gets mutter's server-side
# decorations and never consults the portal. run_sim.sh already pins this for
# unrelated WSLg reasons; only default it so an explicit setting still wins.
if not os.environ.get('QT_QPA_PLATFORM') and os.environ.get('WAYLAND_DISPLAY'):
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Qt's native file dialog is portal-backed. When the portal denies the request
# the dialog window is created but never mapped, leaving getOpenFileName parked
# in its modal event loop - the GUI looks hung with nothing on screen. Qt's own
# widget dialog needs no portal. Must be set before the QApplication exists.
QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)

from dyno.src import logger, test_builder
from dyno.src.logger import Logger
from dyno.src.dyno_controller import Controller
from dyno.src.config_utils import augment_log_keys

from deployment import dyno_paths


class StatusLight(QWidget):
    """Stack-light-style state indicator: a colored lamp plus a caption.

    Colors follow the physical tower light `dyno_controller._write_led` drives
    on the production rig — green for executing, amber for ready, red for a
    fault or a refused action — so the screen and the bench read the same.
    """

    COLORS = {'idle': '#6e6e6e', 'checking': '#c8a200', 'armed': '#c8a200',
              'running': '#2e9e3e', 'error': '#c62828'}

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        self.lamp = QLabel()
        self.lamp.setFixedSize(14, 14)
        layout.addWidget(self.lamp, alignment=Qt.AlignmentFlag.AlignTop)
        self.caption = QLabel()
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption, stretch=1)
        self.state = None
        self.set_state('idle', 'No test armed')

    def set_state(self, state, text):
        # Re-derived every GUI tick, so bail on a no-op: setStyleSheet always
        # forces a style recompute even when nothing changed.
        if state == self.state and text == self.caption.text():
            return
        self.state = state
        color = self.COLORS.get(state, self.COLORS['idle'])
        self.lamp.setStyleSheet(f'background-color: {color}; '
                                'border: 1px solid #202020; border-radius: 7px;')
        self.caption.setStyleSheet(f'color: {color};')
        self.caption.setText(text)


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

        # Arming state, in the order a test moves through it:
        #   _pending_arm    being expanded here, to catch a bad plan early
        #   _requested_arm  queued to the controller, not yet confirmed
        #   _armed_test     the controller reports holding it — the only one
        #                   of the three that means "this is what Start runs"
        # _arm_error is a sticky failure message, cleared by the next arm.
        self._pending_arm = None
        self._requested_arm = None
        self._armed_test = None
        self._armed_caption = ''
        self._armed_stale = False
        self._arm_error = None
        self._controller_fault = False
        self._arm_threads = []  # QThreads must outlive their run()
        self._syncing_select = False

        # Progress readout. The plan half comes from the arm-time expansion
        # (per-segment durations, which are upper bounds — see
        # test_preview.expand_test); the live half comes from the controller in
        # telemetry slot -1. Wall clocks are stamped here rather than taken from
        # the controller because what the operator is timing is the run they
        # watched start, which is the log-flag transition this window sees.
        self._plan_segments = []
        self._plan_duration = 0.0
        self._plan_truncated = False
        self._run_started = None
        self._segment_started = None
        self._segment_seen = None
        self._progress = None

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
            self.arm_test(sys.argv[idx + 1])

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
            # Returns before the plan is verified; the status light reports the
            # outcome, and start_test says so if it is still checking.
            self.arm_test(args[1])

        elif args[0] == 'shutdown':
            self.close_processes()
            QApplication.quit()

    def close_processes(self):
        self.control_command_queue.put_nowait(['shutdown', 0])
        print('\nShutdown request sent to control thread')

        # Let any in-flight test-plan check finish first: a QThread destroyed
        # while it is still running aborts the process.
        for thread in self._arm_threads:
            thread.wait(5000)

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

        # Both readers are gone now, but this process still has to exit. Every
        # telemetry sample was forwarded to the Logger, so logging_queue owns a
        # feeder thread with a backlog of samples and a pipe nothing will drain
        # again; at interpreter exit multiprocessing joins that thread, it
        # blocks writing to the full pipe, and the GUI hangs after printing a
        # clean shutdown (Controller.__init__ drops its telemetry queue for the
        # same reason). The samples are worthless at this point - drop them.
        self.logging_queue.cancel_join_thread()
        self.control_command_queue.cancel_join_thread()

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

            # Create the drop down selector to change what the scope shows
            selection_box = QComboBox()
            selection_box.addItems(self.plot_params.keys())
            selection_box.setCurrentText(plot_key)
            selection_box.currentIndexChanged.connect(self.__change_scopes)

            # creat thge plot. the y range is defined from the plot_config file
            plot = self.plot_widget.addPlot(row=i, col=0)
            plot.showGrid(x=True, y=True, alpha=0.5)
            if i == 0:
                # We drive the x-range ourselves (rolling window in redraw), so keep
                # pyqtgraph from auto-ranging x and warping the axis during start-up.
                plot.getViewBox().enableAutoRange(x=False)
            if i > 0:
                plot.setXLink(self.plots[0]['plot']) # link x range of all other plots to the first

            # One legend per scope, parented once here. __load_scope refills it;
            # parenting a fresh LegendItem on every scope change stacks the old
            # ones on top of each other in the corner.
            legend = pg.LegendItem(offset=(80, 10))
            legend.setParentItem(plot)

            # a dictionary of the objects associated with the scope, so they can be adjusted later
            entry = {
                'plot': plot,
                'selector': selection_box,
                'legend': legend,
                'curves': [],
                'data_keys': [],
                'key': None,
            }
            self.__load_scope(entry, plot_key)

            # append that dictionary to a list of plots, and add the plot to the gui
            self.plots.append(entry)
            self.controls_layout.addWidget(selection_box)

        # --- Test Selection section: centered header, launcher button beneath
        # (same size as Start/Stop), then the Quick Select dropdown. ---
        test_select_title = QLabel('Test Selection', alignment=Qt.AlignmentFlag.AlignCenter)
        test_select_title.setStyleSheet('font-size: 16px;')
        self.controls_layout.addWidget(test_select_title)

        self.open_test_def_button = QPushButton('Open Test Builder')
        self.open_test_def_button.clicked.connect(self.__open_test_definition)
        self.controls_layout.addWidget(self.open_test_def_button)

        self.load_test_file_button = QPushButton('Load Test File…')
        self.load_test_file_button.clicked.connect(self.__load_test_file)
        self.controls_layout.addWidget(self.load_test_file_button)

        self.test_select = QComboBox()
        self.test_select.setPlaceholderText('Quick Select')
        self.test_select.currentIndexChanged.connect(self.__on_test_selected)
        self.__refresh_test_list()
        self.controls_layout.addWidget(self.test_select)

        # What is actually armed / running. The dropdown above only records
        # what was picked; this is the state the controller is in.
        self.status_light = StatusLight()
        self.controls_layout.addWidget(self.status_light)

        # Progress lives in its own label rather than the status light's
        # caption: the caption is guarded against redundant restyling by
        # comparing text (StatusLight.set_state), and a ticking clock in it
        # would defeat that guard on every update.
        self.progress_label = QLabel()
        self.progress_label.setWordWrap(True)
        self.progress_label.setStyleSheet('font-size: 11px; color: gray;')
        self.progress_label.setVisible(False)
        self.controls_layout.addWidget(self.progress_label)

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

        # Fault reset, with its outcome underneath. The status light says a
        # drive is faulted but not which one, and clearing it otherwise means
        # leaving the rig for Workbench or release_drives.sh. Only ever a
        # request: the reset drops the DS402 latch, it does not remove the
        # cause, so the label below is where the operator learns whether the
        # fault actually went away.
        self.clear_faults_button = QPushButton('Clear Drive Faults')
        self.clear_faults_button.setToolTip(
            'Send a DS402 fault reset to every faulted drive on the bus.\n'
            'No test may be running. This clears the latch, not the cause — a '
            'drive whose\nfault condition is still present will simply fault '
            'again.')
        self.clear_faults_button.clicked.connect(self.__clear_faults)
        self.controls_layout.addWidget(self.clear_faults_button)

        self.clear_faults_label = QLabel('No drive faults')
        self.clear_faults_label.setWordWrap(True)
        self.clear_faults_label.setStyleSheet('font-size: 11px; color: gray;')
        self.controls_layout.addWidget(self.clear_faults_label)

        # Tare, with its result underneath: the bias alone does not say whether
        # it is reasonable (0.5 Nm is noise on a 500 Nm cell and 2.5% on a 20 Nm
        # one), so the readout leads with percent of full scale and colours on
        # it. Right-click clears, for undoing a tare taken against a loaded cell.
        self.tare_button = QPushButton('Tare Torque')
        self.tare_button.setToolTip(
            'Average the torque cells at rest and apply the result as a session '
            'zero.\nThe rig must be idle and no test running.\n'
            'Right-click to clear the applied tare.')
        self.tare_button.clicked.connect(self.__tare)
        self.tare_button.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tare_button.customContextMenuRequested.connect(self.__clear_tare)
        self.controls_layout.addWidget(self.tare_button)

        self.tare_label = QLabel('No tare applied')
        self.tare_label.setWordWrap(True)
        self.tare_label.setStyleSheet('font-size: 11px; color: gray;')
        self.controls_layout.addWidget(self.tare_label)

        self.__build_safeties_panel()

        self.controls_layout.addStretch(1)
        self.main_layout.addWidget(self.plot_widget, stretch=5)
        self.main_layout.addWidget(self.controls_widget, stretch=1)

        # Open larger than the layout's natural hint, which packs the scopes
        # tighter than is readable at a glance across the bench. Taller than it
        # is wider: the extra height is what the stacked scopes actually need.
        hint = self.sizeHint()
        self.resize(int(hint.width() * 1.2), int(hint.height() * 1.4))

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
        code = shutil.which('code')
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
        # Never blocked on the arming state: starting with nothing armed
        # energizes the drives at zero command, which is a real bring-up
        # workflow. The indicator already says continuously whether a test is
        # armed, so there is nothing to add here.
        self.control_command_queue.put_nowait(['start_test', 0])

    def __stop_test(self):
        self.control_command_queue.put_nowait(['stop_test', 0])

    def __tare(self):
        self.control_command_queue.put_nowait(['tare', 0])

    def __clear_tare(self):
        self.control_command_queue.put_nowait(['clear_tare', 0])

    def __clear_faults(self):
        self.control_command_queue.put_nowait(['clear_faults', 0])

    def __refresh_fault_clear(self, state):
        """Render which drives are faulted, and the outcome of the last reset.

        Naming the drives is the part the status light cannot do, and it is what
        decides where the operator walks: a faulted DUT after a run is usually
        the test doing its job, a faulted LOAD is not."""
        faulted = state.get('faulted_drives') or []
        message = state.get('fault_clear_message')

        if state.get('fault_clear_active'):
            self.clear_faults_label.setText(message or 'Clearing faults…')
            colour = '#b58900'
        elif faulted:
            # The live fault leads: a 'cleared' message from a minute ago must
            # not sit above drives that are faulted right now.
            text = 'In fault: ' + ', '.join(faulted)
            if message:
                text += f'\n{message}'
            self.clear_faults_label.setText(text)
            colour = '#cb4b16'
        else:
            self.clear_faults_label.setText(message or 'No drive faults')
            colour = '#859900' if message and 'cleared' in message.lower() else 'gray'
        self.clear_faults_label.setStyleSheet(f'font-size: 11px; color: {colour};')

    def __refresh_tare(self, state):
        """Render the controller's tare state under the button.

        Leads with percent of full scale because the bias in Nm cannot be judged
        on its own, carries the spread because a mean over a noisy window is not
        a zero, and shows the age because a tare that was fine an hour ago has
        had an hour to drift."""
        if state.get('tare_active'):
            self.tare_label.setText('Taring… keep the rig still')
            self.tare_label.setStyleSheet('font-size: 11px; color: #b58900;')
            return

        tare = state.get('tare')
        if not tare:
            # A refusal ("a test is running", "the rig is moving") is the most
            # useful thing to show here -- it says what to fix.
            message = state.get('tare_message') or 'No tare applied'
            self.tare_label.setText(message)
            self.tare_label.setStyleSheet(
                'font-size: 11px; color: %s;'
                % ('#cb4b16' if 'refused' in message.lower()
                   or 'aborted' in message.lower() else 'gray'))
            return

        warn = float((self.dyno_params.get('tare') or {}).get('warn_frac_fs', 0.02))
        rows, worst, judgeable = [], 0.0, True
        for sensor in sorted(tare):
            record = tare[sensor] or {}
            bias = record.get('bias')
            frac = record.get('frac_fs')
            # Without a full scale there is nothing to judge the bias against,
            # and colouring it green would claim an all-clear we cannot support.
            if frac is None:
                judgeable = False
            worst = max(worst, frac or 0.0)
            parts = [f'{bias:+.4g}' if isinstance(bias, (int, float)) else str(bias)]
            parts.append(f'{100 * frac:.2f}% FS' if frac is not None else 'FS ?')
            if record.get('stddev') is not None:
                parts.append(f"sd {record['stddev']:.3g}")
            age = self.__tare_age(record.get('at'))
            if age:
                parts.append(age)
            rows.append(f'{sensor}: ' + '  '.join(parts))

        # Half the warn threshold is the amber band: still fine, but drifting
        # toward the point where it is worth asking why.
        colour = ('#cb4b16' if worst > warn
                  else 'gray' if not judgeable
                  else '#859900' if worst <= 0.5 * warn else '#b58900')
        self.tare_label.setText('\n'.join(rows))
        self.tare_label.setStyleSheet(f'font-size: 11px; color: {colour};')

    @staticmethod
    def __tare_age(stamp):
        """'4m ago' for a tare timestamp, or '' if it cannot be read."""
        if not stamp:
            return ''
        try:
            taken = time.mktime(time.strptime(stamp, '%Y-%m-%d %H:%M:%S'))
        except (ValueError, TypeError):
            return ''
        minutes = int(max(time.time() - taken, 0) // 60)
        if minutes < 1:
            return 'just now'
        return f'{minutes}m ago' if minutes < 60 else f'{minutes // 60}h{minutes % 60:02d}m ago'

    def __prompt_experiment_notes(self):
        """Shown when a test stops gracefully (stop button, completion, or a
        safety stop). Non-empty notes are saved as <test>.txt next to the log
        hdf5; empty/cancelled leaves no file. An app abort never gets here, so
        no file is written in that case either. The 'Delete test log' button
        deletes the run's entire log folder instead.

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
                                'Anything worth remembering about this run?'))
        text_edit = QTextEdit()
        layout.addWidget(text_edit)
        buttons = QHBoxLayout()
        save_btn = QPushButton('Save Notes')
        save_btn.clicked.connect(dialog.accept)
        skip_btn = QPushButton('Skip')
        skip_btn.clicked.connect(dialog.reject)
        DELETE_RESULT = 2  # distinct from QDialog Accepted (1) / Rejected (0)
        delete_btn = QPushButton('Delete test log')
        delete_btn.setStyleSheet('background-color: #c62828; color: white;')
        # Two-click confirm: first click arms the button, second click (within
        # 3 s) deletes. Reverts on timeout so a stray click can't linger armed.
        revert_timer = pg.Qt.QtCore.QTimer(dialog)
        revert_timer.setSingleShot(True)
        revert_timer.setInterval(3000)

        def disarm():
            delete_btn.setText('Delete test log')
            delete_btn.setStyleSheet('background-color: #c62828; color: white;')

        def on_delete_clicked():
            if revert_timer.isActive():
                revert_timer.stop()
                dialog.done(DELETE_RESULT)
            else:
                delete_btn.setText('Really delete? Click again')
                delete_btn.setStyleSheet('background-color: #7f0000; color: white; font-weight: bold;')
                revert_timer.start()

        revert_timer.timeout.connect(disarm)
        delete_btn.clicked.connect(on_delete_clicked)
        buttons.addWidget(save_btn)
        buttons.addWidget(skip_btn)
        buttons.addWidget(delete_btn)
        layout.addLayout(buttons)
        text_edit.setFocus()
        # Keep a reference so the dialog isn't GC'd while open.
        self._notes_dialog = dialog

        def on_finished(result):
            self._notes_dialog = None
            if result == DELETE_RESULT:
                self.__delete_test_log(log_dir)
                return
            notes = text_edit.toPlainText().strip() if result else ''
            if notes:
                self.__save_experiment_notes(log_dir, test_name, notes)

        dialog.finished.connect(on_finished)
        # Center over the main window; some window managers (WSLg
        # especially) ignore the parent hint and place the dialog
        # elsewhere. Set the geometry before showing, then re-assert it
        # after mapping since WSLg re-places windows post-show.
        def center():
            geo = dialog.frameGeometry()
            geo.moveCenter(self.frameGeometry().center())
            dialog.move(geo.topLeft())
        dialog.adjustSize()
        center()
        dialog.open()
        pg.Qt.QtCore.QTimer.singleShot(0, center)
        pg.Qt.QtCore.QTimer.singleShot(100, center)

    def __delete_test_log(self, log_dir):
        folder = f"{dyno_paths.dyno_logs_directory}/{log_dir}"
        try:
            if os.path.isdir(folder):
                shutil.rmtree(folder)
                print(f'Deleted log folder {log_dir}')
            else:
                print(f'Log folder {log_dir} not found, nothing to delete')
        except OSError as e:
            print(f'Failed to delete log folder {log_dir}: {e}')

    def __save_experiment_notes(self, log_dir, test_name, notes):
        """Re-render the run's companion report with the operator's notes on
        top. The Logger already wrote the setup half when the test stopped;
        this rewrites the whole file rather than overwriting it, so the two
        writers never clobber each other. Falls back to a notes-only file if
        the log has no resolved config to render (e.g. a pre-existing log)."""
        from dyno.src import setup_summary

        folder = f"{dyno_paths.dyno_logs_directory}/{log_dir}"
        base = os.path.splitext(os.path.basename(test_name or 'log'))[0] or 'log'
        path = setup_summary.report_path(folder, base)
        try:
            os.makedirs(folder, exist_ok=True)
            resolved, meta = self.__read_log_setup(folder, base)
            if resolved is None:
                with open(path, 'w') as f:
                    f.write(notes + '\n')
            else:
                setup_summary.write_report(path, resolved, meta, notes=notes)
            print(f'Experiment notes saved to {log_dir}/{base}.txt')
        except OSError as e:
            print(f'Failed to save experiment notes: {e}')

    def __read_log_setup(self, folder, base):
        """Read the just-closed log's resolved config so the notes rewrite can
        regenerate the setup section. Returns (None, {}) if unavailable."""
        import json
        import h5py

        log_path = f'{folder}/{base}.hdf5'
        try:
            with h5py.File(log_path, 'r') as f:
                if 'resolved_config' not in f.attrs:
                    return None, {}
                resolved = json.loads(f.attrs['resolved_config'])
                from dyno.src import setup_summary
                return resolved, setup_summary.meta_from_log(log_path, f)
        except (OSError, ValueError) as e:
            print(f'Could not read setup from {log_path}: {e}')
            return None, {}

    # --- arming -----------------------------------------------------------
    # Every way of choosing a test (Quick Select, Load Test File, the Test
    # Builder, --load_test / the CLI) funnels through arm_test.

    def arm_test(self, test_file):
        """Verify `test_file` loads, then ask the controller to arm it.

        Two checks, because neither alone is enough. Here: expand the plan
        through the same TestManager + validation path the rig uses, so a bad
        plan is caught before it is queued at all — but against the config's
        motor limits, not the limits the drives report over SDO. There: the
        controller re-validates against the real limits, and reports the
        outcome back through control_state. Nothing counts as armed until that
        report arrives (see __reconcile_controller).
        """
        if self._log_active:
            self._arm_error = f'Test running — stop it before arming {test_file}'
            self.__refresh_status()
            return

        # Disarm first. From here until the controller confirms, nothing is
        # armed, so a Start in that window can never run the previous test.
        self.__disarm()
        self.__select_in_dropdown(test_file)
        self._pending_arm = test_file
        self.__refresh_status()

        # Lazy import (see __open_test_definition) — same worker the builder
        # runs its previews on.
        from dyno.src.test_builder_window import ExpansionThread
        self._arm_threads = [t for t in self._arm_threads if t.isRunning()]
        thread = ExpansionThread(test_file, self.mode)
        thread.done.connect(
            lambda result, f=test_file: self.__on_arm_checked(f, result))
        thread.failed.connect(
            lambda msg, f=test_file: self.__on_arm_failed(f, msg))
        self._arm_threads.append(thread)
        thread.start()

    def __on_arm_checked(self, test_file, result):
        if test_file != self._pending_arm:
            return  # superseded by a later selection
        self._pending_arm = None
        self._requested_arm = test_file
        self.control_command_queue.put_nowait(['test_def', (test_file, self.mode)])
        # Tell the Logger which test is armed so it can name the log file
        # after the test yaml.
        self.logging_queue.put_nowait({'test_name': test_file})
        duration = result['t'][-1] if result['n_cycles'] else 0.0
        # Kept for the progress readout during the run.
        self._plan_segments = result.get('segments') or []
        self._plan_duration = duration
        self._plan_truncated = result['truncated']
        # `truncated` is a preview cap (MAX_CYCLES), not a problem with the
        # plan — the rig runs the whole thing, so say so and arm it anyway.
        note = ' — preview capped, full test is longer' if result['truncated'] else ''
        # Held until the controller confirms it holds the plan.
        self._armed_caption = (f'Armed: {test_file}\n'
                               f'{duration:.1f} s, {result["n_cycles"]:,} cycles{note}')
        self.__refresh_status()

    def __on_arm_failed(self, test_file, msg):
        if test_file != self._pending_arm:
            return
        self._pending_arm = None
        self.__arm_failed(f'{test_file} FAILED to load — {msg}')

    def __arm_failed(self, message):
        # arm_test disarms before requesting, so normally nothing is loaded on
        # the controller either — but report what is actually armed rather than
        # asserting it. Clear the dropdown too, so no part of the panel implies
        # the failed plan is ready.
        self._syncing_select = True
        self.test_select.setCurrentIndex(-1)
        self._syncing_select = False
        self._arm_error = message + ('\nNothing is armed.' if self._armed_test
                                     is None else
                                     f'\nStill armed: {self._armed_test}')
        self.__refresh_status()

    def __disarm(self):
        """Clear the armed test everywhere: this window, the controller, and
        the Logger's file naming.

        The controller owns test_definition, so a GUI-only reset would leave
        the previous plan loaded and runnable behind a light saying nothing is
        armed — the same confusion as before, pointing the more dangerous way.
        """
        self._pending_arm = None
        self._requested_arm = None
        self._armed_test = None
        self._armed_caption = ''
        self._armed_stale = False
        self._arm_error = None
        self._plan_segments = []
        self._plan_duration = 0.0
        self._plan_truncated = False
        self.control_command_queue.put_nowait(['clear_test', 0])
        self.logging_queue.put_nowait({'test_name': None})

    def __reconcile_controller(self, state):
        """Fold the controller's reported state (telemetry slot -1) into ours.

        This is what makes the indicator honest: until the controller says it
        holds the plan, the GUI has only asked for it.
        """
        self._controller_fault = bool(state.get('fault'))
        self._progress = state.get('progress')

        # Taring is only legal with the rig idle and no test running, so the
        # button follows the controller's own precondition rather than letting
        # the operator queue a command that will just be refused.
        self.tare_button.setEnabled(not state.get('test_active')
                                    and not state.get('tare_active'))
        self.__refresh_tare(state)

        # Same reasoning as the tare button: the controller refuses a reset with
        # a test running, and a reset with nothing faulted is a no-op, so the
        # button offers itself only when it would do something.
        self.clear_faults_button.setEnabled(
            bool(state.get('faulted_drives'))
            and not state.get('test_active')
            and not state.get('fault_clear_active'))
        self.__refresh_fault_clear(state)

        armed = state.get('armed')
        if self._requested_arm is not None:
            # TestManager names itself test_file.split('.')[0] — derive the same.
            if armed == self._requested_arm.split('.')[0]:
                self._armed_test = self._requested_arm
                self._requested_arm = None
            elif state.get('load_error'):
                # The controller's own validation rejected it — limit asserts
                # against the drives' real limits, which our pre-check cannot
                # see.
                failed, self._requested_arm = self._requested_arm, None
                self.__arm_failed(f'{failed} REJECTED by the controller — '
                                  f'{state["load_error"]}')
        if armed is None:
            # The controller holds nothing, whatever we believed. This only
            # ever narrows our claim to match it, which is the safe direction.
            self._armed_test = None

    def __set_status(self, state, text):
        self.status_light.set_state(state, text)

    def __refresh_status(self):
        """Derive the indicator from state, most urgent first. Called on every
        GUI tick, so it must stay a pure function of the fields it reads."""
        if self._controller_fault:
            self.__set_status('error', 'DRIVE FAULT — clear the fault before '
                                       'running')
        elif self._log_active:
            self.__set_status('running', f'Running: {self._armed_test}')
        elif self._arm_error:
            self.__set_status('error', self._arm_error)
        elif self._pending_arm:
            self.__set_status('checking', f'Checking {self._pending_arm}…')
        elif self._requested_arm:
            self.__set_status(
                'checking', f'Loading {self._requested_arm} on the controller…')
        elif self._armed_test and self._armed_stale:
            self.__set_status(
                'error', f'{self._armed_caption}\nEdited in the Test Builder — '
                         'save and re-arm, or this runs the file as it was.')
        elif self._armed_test:
            self.__set_status('armed', self._armed_caption)
        else:
            self.__set_status('idle', 'No test armed')

    @staticmethod
    def __fmt_duration(seconds):
        """'45s' / '6m12s' / '1h04m'. Whole seconds, so a label refreshed on
        every 30 ms tick only actually changes once a second."""
        seconds = int(max(seconds, 0))
        if seconds < 60:
            return f'{seconds}s'
        if seconds < 3600:
            return f'{seconds // 60}m{seconds % 60:02d}s'
        return f'{seconds // 3600}h{(seconds % 3600) // 60:02d}m'

    def __refresh_progress(self):
        """Where the run is: segment, repeat within it, and the clock.

        Every duration here is an UPPER bound, and is written with a `≤` for
        that reason. Three things make the expansion's timing a ceiling rather
        than a prediction: a ramp_break releases as soon as the shaft breaks
        free but expands as running every ramp to the amplitude ceiling, a
        preamble can abort a block on drift, and neither is knowable offline.
        Only the COUNTS (segment i/N, repeat j/m) are exact — which is why they
        lead, and why there is no progress bar or percentage here. Mode-switch
        stalls push the other way (the controller holds a command while a drive
        changes mode), so an overrun is possible and is labelled as one rather
        than clamped to a comfortable-looking zero.
        """
        if not self._log_active or self._run_started is None:
            self.progress_label.setVisible(False)
            self._segment_started = None
            self._segment_seen = None
            return

        progress = self._progress or {}
        index = progress.get('segment') or 0
        count = progress.get('segments') or 0

        # Stamp when the segment changed, for the in-segment estimate. The
        # controller publishes the number; only this window knows the wall
        # clock the operator is reading.
        if index != self._segment_seen:
            self._segment_seen = index
            self._segment_started = time.time()

        lines = []
        if index and count:
            line = f'Segment {index}/{count}'
            if progress.get('segment_type'):
                line += f' — {progress["segment_type"]}'
            lines.append(line)

        # A single-repeat behavior (a trace) has nothing to say here; showing
        # '1/1' would only be noise next to the segment line.
        repeat, repeats = progress.get('repeat') or 0, progress.get('repeats') or 0
        if repeat and repeats > 1:
            line = f'Repeat {repeat}/{repeats}'
            budget = self.__segment_budget(index)
            if budget is not None and self._segment_started is not None:
                left = budget - (time.time() - self._segment_started)
                # Floored at 1 s: a bound that counts down to '≤ 0s' and then
                # sits there reads as a stall, when all it means is that the
                # ceiling is about to be reached.
                line += ('  ·  over estimate' if left <= 0 else
                         f'  ·  ≤ {self.__fmt_duration(max(left, 1))} left here')
            lines.append(line)

        elapsed = self.__fmt_duration(time.time() - self._run_started)
        if self._plan_duration and not self._plan_truncated:
            lines.append(f'{elapsed} elapsed of ≤ '
                         f'{self.__fmt_duration(self._plan_duration)}')
        else:
            # Truncated expansions have no total worth quoting — the plan runs
            # past where the preview stopped looking.
            lines.append(f'{elapsed} elapsed')

        # Always at least the elapsed line: a running test with no progress
        # block yet (the controller has not published one) still gets a clock,
        # which is better than a panel that blinks in and out.
        self.progress_label.setText('\n'.join(lines))
        self.progress_label.setVisible(True)

    def __segment_budget(self, index):
        """The expansion's duration for segment `index` (1-based), or None if
        the plan does not reach it — a truncated expansion stops early, so the
        tail segments genuinely have no estimate to offer."""
        if 1 <= index <= len(self._plan_segments):
            return self._plan_segments[index - 1]['duration_s']
        return None

    def __refresh_test_list(self):
        current = self.test_select.currentText()
        self._syncing_select = True
        self.test_select.clear()
        self.test_select.addItems(
            test_builder.list_test_files(dyno_paths.dyno_test_directory))
        # findText -> -1 when the previous pick is gone, which shows the
        # placeholder. Selection here never re-arms; arm_test owns that.
        self.test_select.setCurrentIndex(self.test_select.findText(current))
        self._syncing_select = False

    def __select_in_dropdown(self, test_file):
        self._syncing_select = True
        if self.test_select.findText(test_file) < 0:
            self.test_select.addItem(test_file)
        self.test_select.setCurrentText(test_file)
        self._syncing_select = False

    def __on_test_selected(self, index):
        if self._syncing_select or index < 0:
            return
        self.arm_test(self.test_select.itemText(index))

    def __load_test_file(self):
        """Pick a test plan off disk and arm it.

        Constrained to the tests directory: TestManager resolves the plan and
        its trace csvs relative to that root, so a file from elsewhere cannot
        run as-is (see test_builder.rel_test_path)."""
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Test File', dyno_paths.dyno_test_directory,
            'Test plans (*.yaml *.yml)')
        if not path:
            return
        rel = test_builder.rel_test_path(path, dyno_paths.dyno_test_directory)
        if rel is None:
            self._arm_error = (
                f'{os.path.basename(path)} is outside '
                f'{dyno_paths.dyno_test_directory} — copy it, and any trace '
                'csvs it references, into the tests directory first.')
            self.__refresh_status()
            return
        self.__refresh_test_list()
        self.arm_test(rel)

    def __open_test_definition(self):
        # Lazy import: keeps GUI start-up light and avoids a hard dependency
        # when running headless. Keep a reference so the window isn't GC'd.
        from dyno.src.test_builder_window import TestBuilderWindow
        if getattr(self, '_test_def_window', None) is None:
            self._test_def_window = TestBuilderWindow(self.mode)
            self._test_def_window.test_loaded.connect(self.arm_test)
            self._test_def_window.test_saved.connect(
                lambda _: self.__refresh_test_list())
            self._test_def_window.test_dirty.connect(self.__on_test_def_dirty)
        self._test_def_window.show()
        self._test_def_window.raise_()
        self._test_def_window.activateWindow()

    def __on_test_def_dirty(self):
        # The builder's recipe no longer matches the file it handed us. What is
        # armed still runs the file on disk, so keep it armed and flag it.
        if self._armed_test is None or self._log_active:
            return
        self._armed_stale = True
        self.__refresh_status()

    def __load_scope(self, entry, plot_key):
        # Point one scope at plot_key: drop whatever it was drawing and rebuild
        # its curves, pens and legend from that key's config.
        #
        # The curves have to be torn down every time, not only when the number
        # of traces changes. A curve owns its pen and its legend label, so
        # reusing one for a different key draws the new data in the old color
        # under the old legend text; and any curve left behind in the plot keeps
        # showing its last frame forever, because redraw() only ever feeds the
        # curves listed in entry['curves'].
        plot = entry['plot']
        plot_params = self.plot_params[plot_key]

        for curve in entry['curves']:
            plot.removeItem(curve)
        entry['legend'].clear()

        plot.setTitle(plot_params['title'])
        plot.setLabel('left', plot_params['unit'])
        # plot.getViewBox().setYRange(plot_params['range'][0], plot_params['range'][1])

        # for each curve that should be on the graph, load that trace from the data buffer and plot it
        curves = []
        for ui, data_key in enumerate(plot_params['data_keys']):
            curve = plot.plot(self.gui_data[0,:], self.trace(data_key), pen=plot_params['pens'][ui], name=plot_params['legends'][ui])
            entry['legend'].addItem(curve, plot_params['legends'][ui])
            curves.append(curve)

        entry['curves'] = curves
        entry['data_keys'] = plot_params['data_keys']
        entry['key'] = plot_key

    # called if you change which scope is selected in the dropdown
    def __change_scopes(self):
        # Find the changed scope. This compares against the key we last loaded,
        # not the plot's title: the title is the config's 'title' field, which
        # is a different string from the dropdown entry (the config key), so the
        # old title comparison flagged every scope on every change.
        for entry in self.plots:
            plot_key = entry['selector'].currentText()
            if plot_key != entry['key']:
                self.__load_scope(entry, plot_key)

    def trace(self, trace_key):
        if trace_key in self.log_keys:
            return self.gui_data[self.log_keys.index(trace_key), :]
        # fallback: NaNs so it doesn't draw misleading zeros
        return np.full(self.gui_data.shape[1], np.nan, dtype=float)

    def update_data(self):
        mode_offset = {4: 0, 3: 1, 7: 2} # lookup table to determine how to map a command for a given mode into the telemetry sample. This is done mainly so that the telemetry sample can return the control mode + command, vs. the control mode, + command + 2 more NAN commands

        # pull samples from telemetry queue
        control_state = None
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
                    self._run_started = time.time()
                    self._segment_started = None
                    self._segment_seen = None
                    self._active_log_dir = logger.log_dir_name()
                    # The armed plan, not the dropdown text — the two differ
                    # whenever a pick failed to load.
                    self._active_log_test = self._armed_test
                    self.logging_queue.put_nowait({'log_dir': self._active_log_dir})
                elif not log_flag and self._log_active:
                    self._log_active = False
                    self._notes_prompt_pending = True
                # Slot -1: the controller's own view of what it holds. Keep the
                # newest; it is reconciled once, after the drain.
                control_state = sample[-1]
                self.logging_queue.put_nowait(sample) #forward sample to the logging thread
                self.telemetry_samples.append(sample[:-2])
            except:
                read_queue = False
                pass

        if control_state is not None:
            self.__reconcile_controller(control_state)
        self.__refresh_status()
        self.__refresh_progress()

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

    exit_code = a.exec()

    # The event loop is gone, so the handler above would turn a Ctrl-C into a
    # quit() on a dead application - silently swallowing it. Hand SIGINT back to
    # the interpreter so the terminal stays escapable if anything stalls during
    # interpreter teardown.
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    sys.exit(exit_code)
