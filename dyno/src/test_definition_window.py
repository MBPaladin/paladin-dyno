"""
Test Definition window (second window opened from the main GUI).

For now it visualizes a test plan's commanded input/output torque, velocity, and
position over time. Expansion runs offline in `test_preview` on a virtual clock,
so even a 20-minute plan previews in a couple of seconds. This is also the shell
where an interactive, safety-checked test *builder* will live later.
"""
import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from deployment import dyno_paths
from dyno.src import test_preview

# Command curves are piecewise-linear, so striding to this many points is
# visually lossless while keeping pyqtgraph responsive on long tests.
_MAX_DISPLAY_POINTS = 20000

# (title, y-axis unit, signal key matching test_preview.MODES)
_PLOTS = (
    ('Torque', 'Nm', 'torque'),
    ('Velocity', 'rad/s', 'velocity'),
    ('Position', 'rad', 'position'),
)
_INPUT_PEN = pg.mkPen('#1f77b4', width=2)
_OUTPUT_PEN = pg.mkPen('#d62728', width=2)


class _ExpansionThread(QThread):
    """Runs the (CPU-bound) expansion off the UI thread."""
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, test_file, mode):
        super().__init__()
        self.test_file = test_file
        self.mode = mode

    def run(self):
        try:
            self.done.emit(test_preview.expand_test(self.test_file, self.mode))
        except Exception as e:  # surface validation/load errors in the UI
            self.failed.emit(f"{type(e).__name__}: {e}")


class TestDefinitionWindow(QWidget):
    # Emitted (with the test filename) when a test expands successfully, so the
    # main GUI can mirror the selection into its Quick Select dropdown.
    test_loaded = Signal(str)

    def __init__(self, mode, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.setWindowTitle('Test Definition')
        self.resize(1000, 780)
        self._thread = None
        self._current_file = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        header = QLabel('Test Definition', alignment=Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet('font-size: 18px; font-weight: bold;')
        layout.addWidget(header)

        subtitle = QLabel(f'Commanded input/output vs. time  (rig: {self.mode})',
                          alignment=Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet('font-size: 12px; color: gray;')
        layout.addWidget(subtitle)

        self.plot_widget = pg.GraphicsLayoutWidget()
        self.curves = {}
        prev_plot = None
        for row, (title, unit, key) in enumerate(_PLOTS):
            plot = self.plot_widget.addPlot(row=row, col=0, title=title)
            plot.setLabel('left', unit)
            plot.setLabel('bottom', 'Time', units='s')
            plot.showGrid(x=True, y=True, alpha=0.3)
            legend = pg.LegendItem(offset=(60, 10))
            legend.setParentItem(plot)
            # connect='finite' breaks the line across NaNs, so each curve only
            # draws where that motor is actually in that control mode.
            input_curve = plot.plot([], [], pen=_INPUT_PEN, connect='finite')
            output_curve = plot.plot([], [], pen=_OUTPUT_PEN, connect='finite')
            legend.addItem(input_curve, 'input')
            legend.addItem(output_curve, 'output')
            if prev_plot is not None:
                plot.setXLink(prev_plot)  # shared, scrollable time axis
            prev_plot = plot
            self.curves[key] = (input_curve, output_curve)
        layout.addWidget(self.plot_widget, stretch=1)

        bottom = QHBoxLayout()
        self.choose_button = QPushButton('Choose Test File')
        self.choose_button.clicked.connect(self._choose_file)
        bottom.addWidget(self.choose_button)
        self.builder_button = QPushButton('Open Test Builder')
        self.builder_button.clicked.connect(self._open_builder)
        bottom.addWidget(self.builder_button)
        self.status = QLabel('No test loaded.')
        bottom.addWidget(self.status, stretch=1)
        layout.addLayout(bottom)

    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Choose Test File', dyno_paths.dyno_test_directory,
            'Test plans (*.yaml *.yml)')
        if path:
            # Keep tests in subfolders (e.g. ui_generated_tests/) loadable:
            # TestManager joins this onto the tests directory.
            rel = os.path.relpath(path, dyno_paths.dyno_test_directory)
            self.load_test(rel.replace(os.sep, '/'))

    def _open_builder(self):
        # Lazy import mirrors how the main GUI opens this window. Keep a
        # reference so the window isn't garbage-collected.
        from dyno.src.test_builder_window import TestBuilderWindow
        if getattr(self, '_builder_window', None) is None:
            self._builder_window = TestBuilderWindow(self.mode)
            # A saved test loads straight into this preview (which in turn
            # mirrors it to the main GUI's Quick Select via test_loaded).
            self._builder_window.test_saved.connect(self.load_test)
        self._builder_window.show()
        self._builder_window.raise_()
        self._builder_window.activateWindow()

    def load_test(self, test_file):
        if self._thread is not None and self._thread.isRunning():
            return  # ignore re-entrant loads while one is in flight
        self._current_file = test_file
        self.status.setText(f'Expanding {test_file}…')
        self.choose_button.setEnabled(False)
        self._thread = _ExpansionThread(test_file, self.mode)
        self._thread.done.connect(self._on_done)
        self._thread.failed.connect(self._on_failed)
        self._thread.finished.connect(lambda: self.choose_button.setEnabled(True))
        self._thread.start()

    def _clear_curves(self):
        for input_curve, output_curve in self.curves.values():
            input_curve.setData([], [])
            output_curve.setData([], [])

    def _on_failed(self, message):
        self._clear_curves()
        self.status.setText(f'{self._current_file}:  FAILED — {message}')

    def _on_done(self, result):
        t = result['t']
        n = result['n_cycles']
        if n == 0:
            self._clear_curves()
            self.status.setText(f'{self._current_file}:  no commands produced')
            return

        step = max(1, n // _MAX_DISPLAY_POINTS)
        t_ds = t[::step]
        for key, (input_curve, output_curve) in self.curves.items():
            input_curve.setData(t_ds, result[f'input_{key}'][::step])
            output_curve.setData(t_ds, result[f'output_{key}'][::step])

        msg = f'{self._current_file}:  {t[-1]:.1f}s test, {n:,} cycles'
        if result['truncated']:
            msg += '  (truncated at preview limit)'
        self.status.setText(msg)
        self.test_loaded.emit(self._current_file)
