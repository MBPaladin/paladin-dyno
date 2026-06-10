import time
import numpy as np
import pyqtgraph as pg
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import *
from PySide6.QtWidgets import *
import sys
import os   
import multiprocessing
import yaml

from dyno.src.logger import Logger
from dyno.src.dyno_controller import Controller

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

        # add configured sensor keys
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
        print("LOG_KEYS: item count = ", len(self.log_keys))
        for key in self.log_keys:
            print('\t',key)

        # make data buffer, one row for each item in the log keys
        self.gui_data = np.zeros((len(self.log_keys), buffer_length))
        self.gui_data[0,:] = np.arange(-buffer_length, 0)*0.03

        # make a buffer for incoming telementry
        self.telemetry_samples = []

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

        # wait for control thread shutdown to occur
        time.sleep(2)

        if self.control_process.is_alive():
            self.control_process.terminate()
            self.control_process.join()
            print('Control process terminated')

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

        # Add fixed buttons to the control layout
        title_label = QLabel('Test Selection', alignment=Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet('font-size: 16px;')
        self.controls_layout.addWidget(title_label)

        self.test_select = QComboBox()
        self.test_select.addItem("")

        self.test_select.addItems([f for f in os.listdir(dyno_paths.dyno_test_directory) if f[-4:] == 'yaml'])
        self.test_select.setCurrentText('Test Selection')
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

        self.controls_layout.addStretch(1)
        self.main_layout.addWidget(self.plot_widget, stretch=5)
        self.main_layout.addWidget(self.controls_widget, stretch=1)

    def __start_test(self):
        self.control_command_queue.put_nowait(['start_test', 0])

    def __stop_test(self):
        self.control_command_queue.put_nowait(['stop_test', 0])

    def __load_test(self):
        self.control_command_queue.put_nowait(['test_def', (self.test_select.currentText(), self.mode)])

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
                self.logging_queue.put_nowait(sample) #forward sample to the logging thread
                self.telemetry_samples.append(sample[:-2])
            except:
                read_queue = False
                pass


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

        self.redraw()

    def redraw(self):
        draw_start = time.time()
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

    if '--gearbox' in sys.argv:
        mode = 'gearbox'
    elif '--actuator' in sys.argv:
        mode = 'actuator'
    elif '--actuator_production' in sys.argv:
        mode = 'actuator_production'
    else:
        print('Launch Aborted, Errant mode key supplied')

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
    sys.exit(a.exec())
