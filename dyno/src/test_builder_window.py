"""
Test Builder window (opened from the main GUI).

Two roles in one window:

* Build: compose an experiment as a sequence of segments (sawtooth / step /
  ramp_release / dwell patterns with a secondary-motor level sweep), see the
  commanded curves live as parameters change, and save to
  tests/ui_generated_tests/<name>.yaml with the recipe embedded so the test
  can be reopened and edited. Saving re-validates through the real
  TestManager path (via test_preview.expand_test), so anything that saves
  green here will load on the rig.

* View: open any test yaml. Files with an embedded recipe come back editable;
  everything else (hand-written yamls, grid searches, loops/imports) is shown
  read-only as its authoritative per-cycle expansion, with the editing
  controls greyed out.
"""
import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QPushButton, QSpinBox,
                               QVBoxLayout, QWidget)

from deployment import dyno_paths
from dyno.src import test_builder, test_preview

# Command curves are piecewise-linear, so striding an expansion to this many
# points is visually lossless while keeping pyqtgraph responsive.
_MAX_DISPLAY_POINTS = 20000


class ExpansionThread(QThread):
    """Runs the (CPU-bound) test expansion off the UI thread.

    Shared with the main GUI, which uses the same expansion to verify a plan
    loads before arming it."""
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

_PLOTS = (('Torque', 'Nm', 'torque'),
          ('Velocity', 'rad/s', 'velocity'),
          ('Position', 'rad', 'position'))
_INPUT_PEN = pg.mkPen('#1f77b4', width=2)
_OUTPUT_PEN = pg.mkPen('#d62728', width=2)


def _parse_levels(text):
    values = [v for v in (s.strip() for s in text.split(',')) if v]
    return [float(v) for v in values] or [0.0]


class LevelsField(QWidget):
    """A comma-separated level list with an inline start/stop/n generator.

    The text is always the source of truth: Fill only writes into it, and
    typing in it afterwards drops the stored generator spec -- so a saved
    recipe can never advertise a generator that doesn't describe its levels.
    The spec is kept alongside the recipe only to restore the boxes on reopen
    (bump n, hit Fill again); nothing downstream reads it.
    """
    changed = Signal()
    failed = Signal(str)

    def __init__(self, label, tooltip=''):
        super().__init__()
        self._spec = None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(QLabel(label))
        self.edit = QLineEdit()
        self.edit.setToolTip(tooltip)
        top.addWidget(self.edit, stretch=1)
        outer.addLayout(top)

        gen = QHBoxLayout()
        gen.setContentsMargins(0, 0, 0, 0)
        gen.setSpacing(3)
        self.start = self._gen_spin(0.0)
        self.stop = self._gen_spin(10.0)
        self.count = QSpinBox()
        self.count.setRange(1, 1000)
        self.count.setValue(5)
        self.count.setFixedWidth(46)
        self.count.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.spacing = QComboBox()
        self.spacing.addItem('lin', 'linear')
        self.spacing.addItem('log', 'log')
        self.spacing.setFixedWidth(54)
        self.mirror = QCheckBox()
        self.mirror.setToolTip('Also sweep back down (…, 75, 50, 25, 0) '
                               'without repeating the turnaround, so '
                               'hysteresis shows up within one run.')
        self.fill = QPushButton('Fill')
        self.fill.setFixedWidth(38)
        self.fill.setToolTip('Overwrite the list above with the generated '
                             f'levels, rounded to '
                             f'{test_builder.LEVEL_DECIMALS} decimals. '
                             'Hand-edit them afterwards if you like.')
        for widget in (self.start, QLabel('→'), self.stop, QLabel('n'),
                       self.count, self.spacing, QLabel('↩'), self.mirror,
                       self.fill):
            gen.addWidget(widget)
        gen.addStretch(1)
        outer.addLayout(gen)

        self.fill.clicked.connect(self._fill)
        self.edit.textEdited.connect(self._on_manual_edit)
        self.edit.editingFinished.connect(self.changed)

    @staticmethod
    def _gen_spin(value):
        box = QDoubleSpinBox()
        box.setRange(-1e9, 1e9)
        box.setDecimals(test_builder.LEVEL_DECIMALS)
        box.setValue(value)
        box.setFixedWidth(68)
        box.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        return box

    def _fill(self):
        try:
            levels = test_builder.generate_levels(
                self.start.value(), self.stop.value(), self.count.value(),
                self.spacing.currentData(), self.mirror.isChecked())
        except ValueError as e:
            self.failed.emit(str(e))
            return
        self.edit.setText(', '.join(f'{v:g}' for v in levels))
        self._spec = {'start': self.start.value(), 'stop': self.stop.value(),
                      'n': self.count.value(),
                      'spacing': self.spacing.currentData(),
                      'mirror': self.mirror.isChecked()}
        self.changed.emit()

    def _on_manual_edit(self, _text):
        self._spec = None  # the boxes no longer describe the list

    # --- form <-> model ----------------------------------------------------

    def text(self):
        return self.edit.text()

    def spec(self):
        return dict(self._spec) if self._spec else None

    def set_values(self, values, spec=None):
        values = [float(v) for v in values]
        self.edit.setText(', '.join(f'{v:g}' for v in values))
        self._spec = dict(spec) if spec else None
        if spec:
            self.start.setValue(float(spec.get('start', 0.0)))
            self.stop.setValue(float(spec.get('stop', 0.0)))
            self.count.setValue(int(spec.get('n', 1)))
            index = self.spacing.findData(spec.get('spacing', 'linear'))
            self.spacing.setCurrentIndex(max(0, index))
            self.mirror.setChecked(bool(spec.get('mirror')))
        elif len(values) > 1:
            # Hand-written list: seed the boxes from it so Fill is immediately
            # useful for re-sampling the same span at a different resolution.
            self.start.setValue(values[0])
            self.stop.setValue(max(values, key=abs))
            self.count.setValue(len(values))


class TestBuilderWindow(QWidget):
    # Emitted with the saved test path (relative to the tests directory).
    test_saved = Signal(str)
    # Emitted when a test opens/saves successfully, so the main GUI can arm it.
    test_loaded = Signal(str)
    # Emitted when the recipe in this window diverges from the file the main
    # GUI armed. What runs is the file on disk, so the main window must stop
    # advertising the armed test as matching what is plotted here.
    test_dirty = Signal()

    def __init__(self, mode, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.setWindowTitle('Test Builder')
        self.resize(1280, 800)
        try:
            self.limits = test_preview.limits_from_config(mode)
        except Exception:
            self.limits = None  # config unavailable: skip limit checks live
        try:
            self._cont_torque_cfg = test_preview.continuous_torque_from_config(mode)
        except Exception:
            self._cont_torque_cfg = {'input': None, 'output': None}
        self.segments = [test_builder.default_segment()]
        self._param_widgets = {}
        self._loading = False  # guard: suppress form->model writes during load
        self._view_only = False
        self._expansion_thread = None
        # Hash of the recipe as last handed to the main GUI; None means nothing
        # of ours is armed (or what is armed has no editable recipe).
        self._armed_hash = None
        self._build_ui()
        self._select_segment(0)

    # --- UI construction ---------------------------------------------------

    def _build_ui(self):
        outer = QVBoxLayout(self)

        header = QLabel('Test Builder', alignment=Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet('font-size: 18px; font-weight: bold;')
        outer.addWidget(header)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel('Test name:'))
        self.name_edit = QLineEdit('my_test')
        name_row.addWidget(self.name_edit, stretch=1)
        self.new_button = QPushButton('New Test')
        self.new_button.clicked.connect(self._new_test)
        name_row.addWidget(self.new_button)
        self.open_button = QPushButton('Open Test…')
        self.open_button.clicked.connect(self._open_test)
        name_row.addWidget(self.open_button)
        self.save_button = QPushButton('Save Test')
        self.save_button.clicked.connect(self._save)
        name_row.addWidget(self.save_button)
        outer.addLayout(name_row)

        body = QHBoxLayout()
        outer.addLayout(body, stretch=1)

        # Left: segment list ------------------------------------------------
        left = QVBoxLayout()
        left.addWidget(QLabel('Segments (run in order):'))
        self.seg_list = QListWidget()
        self.seg_list.currentRowChanged.connect(self._on_row_changed)
        left.addWidget(self.seg_list, stretch=1)
        seg_buttons = QHBoxLayout()
        seg_buttons.setContentsMargins(0, 0, 0, 0)
        for label, slot in (('Add', self._add_segment),
                            ('Dup', self._dup_segment),
                            ('Del', self._del_segment),
                            ('Up', lambda: self._move_segment(-1)),
                            ('Down', lambda: self._move_segment(1))):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            seg_buttons.addWidget(btn)
        self.seg_buttons_wrap = QWidget()
        self.seg_buttons_wrap.setLayout(seg_buttons)
        left.addWidget(self.seg_buttons_wrap)
        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setFixedWidth(210)
        body.addWidget(left_wrap)

        # Middle: segment editor --------------------------------------------
        editor = QVBoxLayout()

        # Segment-level: repeats wrap the whole segment (both channels) in a
        # loop behavior, so they belong to neither drive.
        self.seg_box = QGroupBox('Segment')
        seg_form = QFormLayout(self.seg_box)
        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(1, 100000)
        self.repeats_spin.setToolTip('Run this whole segment (pattern and its '
                                     'level sweep) this many times back to '
                                     'back.')
        seg_form.addRow('Repeat segment', self.repeats_spin)
        editor.addWidget(self.seg_box)

        # Primary drive: the motor running the pattern, and the pattern itself.
        self.primary_box = QGroupBox('Primary drive')
        self.pattern_layout = QFormLayout(self.primary_box)
        self.primary_motor = QComboBox()
        self.primary_motor.addItems(test_builder.MOTORS)
        self.pattern_layout.addRow('Motor', self.primary_motor)
        self.primary_mode = QComboBox()
        self.primary_mode.addItems(test_builder.MODES)
        self.pattern_layout.addRow('Control mode', self.primary_mode)
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(list(test_builder.PATTERNS))
        self.pattern_layout.addRow('Pattern', self.pattern_combo)
        editor.addWidget(self.primary_box)

        # Secondary drive: the other motor's level sweep and how it moves
        # between levels.
        self.secondary_box = QGroupBox('Secondary drive')
        sec_form = QFormLayout(self.secondary_box)
        self.secondary_mode = QComboBox()
        self.secondary_mode.addItems(test_builder.MODES)
        sec_form.addRow('Control mode', self.secondary_mode)
        self.levels_field = LevelsField(
            'Levels', 'Setpoints for the other motor; the pattern runs once '
                      'per level, in the order listed. Use a single 0 to hold '
                      'it at zero.')
        sec_form.addRow(self.levels_field)
        self.sec_rate = self._spin(0.001, 1e6, 1.0)
        self.sec_rate.setToolTip(
            'How fast this motor walks from one level to the next (and back '
            'to 0 at the end) with the pattern motor held at 0. Too fast '
            'trips the rotatum limit in torque mode or the acceleration '
            'limit in velocity mode.')
        sec_form.addRow('Ramp rate [units/s]', self.sec_rate)
        self.sec_settle = self._spin(0.0, 1e6, 1.0)
        self.sec_settle.setToolTip(
            'Dead hold at each new level before the pattern starts, so the '
            'rig reaches steady state at that load before anything is '
            'measured. Also gives the first position corner room to blend — '
            "0 here is what triggers 'starts moving at t=0'.")
        sec_form.addRow('Settle at level [s]', self.sec_settle)
        editor.addWidget(self.secondary_box)

        # Position shaping: only ever does anything when a channel is in
        # position mode, so it stays folded away and greys itself out
        # otherwise (see _apply_field_states).
        self.shaping_toggle = QPushButton()
        self.shaping_toggle.setCheckable(True)
        self.shaping_toggle.setStyleSheet('text-align: left; border: none;')
        editor.addWidget(self.shaping_toggle)
        self.shaping_panel = QWidget()
        shaping_form = QFormLayout(self.shaping_panel)
        shaping_form.setContentsMargins(12, 0, 0, 0)
        self.primary_accel = self._spin(0.001, 1e6,
                                        test_builder.DEFAULT_POSITION_ACCEL)
        self.primary_accel.setToolTip(
            'A piecewise-linear position command steps the commanded velocity '
            'at every keyframe corner, which the drive chases with an '
            'inertial torque spike. Each corner is replaced by a '
            'constant-accel blend lasting |Δv| / accel, so a lower value '
            'means a longer, gentler blend. Applies to whichever channels are '
            'in position mode (both motors share this value). If a blend '
            "can't fit between two corners it is clamped, and validation "
            'flags the leftover step.')
        shaping_form.addRow('Blend accel [units/s²]', self.primary_accel)
        self.shaping_panel.setVisible(False)
        editor.addWidget(self.shaping_panel)
        self.shaping_toggle.toggled.connect(self._toggle_shaping)
        self._refresh_shaping_toggle(True)

        editor.addStretch(1)

        self.validation_label = QLabel('')
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet('color: #d62728;')
        editor.addWidget(self.validation_label)

        # Non-blocking warnings (e.g. a gridpoint's stored continuous-torque
        # rating differing from the current rig config's default).
        self.warn_label = QLabel('')
        self.warn_label.setWordWrap(True)
        self.warn_label.setStyleSheet('color: #e6a700;')
        editor.addWidget(self.warn_label)

        editor_wrap = QWidget()
        editor_wrap.setLayout(editor)
        editor_wrap.setFixedWidth(430)
        body.addWidget(editor_wrap)

        # Right: live preview ------------------------------------------------
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
            input_curve = plot.plot([], [], pen=_INPUT_PEN, connect='finite')
            output_curve = plot.plot([], [], pen=_OUTPUT_PEN, connect='finite')
            legend.addItem(input_curve, 'input')
            legend.addItem(output_curve, 'output')
            if prev_plot is not None:
                plot.setXLink(prev_plot)
            prev_plot = plot
            self.curves[key] = {'input': input_curve, 'output': output_curve}
        body.addWidget(self.plot_widget, stretch=1)

        self.status = QLabel('')
        outer.addWidget(self.status)

        # Wire model updates last so building the form doesn't fire them.
        self.pattern_combo.currentTextChanged.connect(self._on_pattern_changed)
        for widget in (self.primary_motor, self.primary_mode, self.secondary_mode):
            widget.currentTextChanged.connect(self._commit_form)
        self.levels_field.changed.connect(self._commit_form)
        self.levels_field.failed.connect(self.status.setText)
        self.primary_accel.valueChanged.connect(self._commit_form)
        self.sec_rate.valueChanged.connect(self._commit_form)
        self.sec_settle.valueChanged.connect(self._commit_form)
        self.repeats_spin.valueChanged.connect(self._commit_form)

    @staticmethod
    def _spin(lo, hi, value):
        box = QDoubleSpinBox()
        box.setRange(lo, hi)
        box.setDecimals(4)
        box.setValue(value)
        return box

    def _toggle_shaping(self, checked):
        self.shaping_panel.setVisible(checked)
        self._refresh_shaping_toggle(self.shaping_toggle.isEnabled())

    def _refresh_shaping_toggle(self, active):
        arrow = '▾' if self.shaping_toggle.isChecked() else '▸'
        hint = '' if active else '   (no position channel)'
        self.shaping_toggle.setText(f'{arrow} Position shaping{hint}')

    # --- segment list management -------------------------------------------

    def _next_seg_id(self):
        used = {s['id'] for s in self.segments}
        n = 1
        while f'SEG{n}' in used:
            n += 1
        return f'SEG{n}'

    def _refresh_list(self, keep_row=0):
        self.seg_list.blockSignals(True)
        self.seg_list.clear()
        for seg in self.segments:
            self.seg_list.addItem(f"{seg['id']}: {seg['pattern']}")
        self.seg_list.blockSignals(False)
        self.seg_list.setCurrentRow(max(0, min(keep_row, len(self.segments) - 1)))

    def _current_segment(self):
        row = self.seg_list.currentRow()
        return self.segments[row] if 0 <= row < len(self.segments) else None

    def _add_segment(self):
        self.segments.append(test_builder.default_segment(self._next_seg_id()))
        self._refresh_list(len(self.segments) - 1)

    def _dup_segment(self):
        seg = self._current_segment()
        if seg:
            copy = {**{k: v for k, v in seg.items()},
                    'primary': dict(seg['primary']),
                    'secondary': dict(seg['secondary']),
                    'params': dict(seg['params']),
                    'id': self._next_seg_id()}
            if 'levels_gen' in seg:
                copy['levels_gen'] = {k: dict(v)
                                      for k, v in seg['levels_gen'].items()}
            self.segments.insert(self.seg_list.currentRow() + 1, copy)
            self._refresh_list(self.seg_list.currentRow() + 1)

    def _del_segment(self):
        row = self.seg_list.currentRow()
        if 0 <= row < len(self.segments) and len(self.segments) > 1:
            del self.segments[row]
            self._refresh_list(row)

    def _move_segment(self, delta):
        row = self.seg_list.currentRow()
        new = row + delta
        if 0 <= row < len(self.segments) and 0 <= new < len(self.segments):
            self.segments[row], self.segments[new] = (self.segments[new],
                                                      self.segments[row])
            self._refresh_list(new)

    def _on_row_changed(self, row):
        if row >= 0:
            self._select_segment(row)

    # --- form <-> model ----------------------------------------------------

    def _select_segment(self, row):
        if self.seg_list.count() != len(self.segments):
            self._refresh_list(row)
        seg = self.segments[row]
        self._loading = True
        self.primary_motor.setCurrentText(seg['primary']['motor'])
        self.primary_mode.setCurrentText(seg['primary']['control_mode'])
        self.primary_accel.setValue(
            seg['primary'].get('accel', test_builder.DEFAULT_POSITION_ACCEL))
        self.secondary_mode.setCurrentText(seg['secondary']['control_mode'])
        self.levels_field.set_values(seg['secondary']['levels'],
                                     self._gen_spec(seg, 'secondary'))
        self.sec_rate.setValue(seg['secondary'].get('rate', 1.0))
        self.sec_settle.setValue(seg['secondary'].get('settle_s', 0.0))
        self.repeats_spin.setValue(int(seg.get('repeats', 1)))
        self.pattern_combo.setCurrentText(seg['pattern'])
        self._rebuild_param_widgets(seg)
        self._apply_field_states(seg)
        self._loading = False
        self._update_preview()

    @staticmethod
    def _gen_spec(segment, key):
        """The stored generator inputs for one level list, or None. Absent for
        every recipe saved before the generator existed, and for any list that
        was typed by hand."""
        return (segment.get('levels_gen') or {}).get(key)

    def _rebuild_param_widgets(self, seg):
        for widget in self._param_widgets.values():
            self.pattern_layout.removeRow(widget)  # deletes the label too
        self._param_widgets = {}
        for key, (label, default, typ) in test_builder.PATTERNS[seg['pattern']].items():
            value = seg['params'].get(key, default)
            if typ is bool:
                widget = QCheckBox()
                widget.setChecked(bool(value))
                widget.toggled.connect(self._commit_form)
            elif typ is int:
                widget = QSpinBox()
                widget.setRange(1, 100000)
                widget.setValue(int(value))
                widget.valueChanged.connect(self._commit_form)
            elif typ is list:
                # Same generator as the secondary levels; spans the form row
                # because it carries its own label.
                widget = LevelsField(label)
                widget.set_values(value, self._gen_spec(seg, key))
                widget.changed.connect(self._commit_form)
                widget.failed.connect(self.status.setText)
                self.pattern_layout.addRow(widget)
                self._param_widgets[key] = widget
                continue
            else:
                widget = self._spin(-1e9, 1e9, float(value))
                widget.valueChanged.connect(self._commit_form)
            self.pattern_layout.addRow(label, widget)
            self._param_widgets[key] = widget

    def _cont_torque_default(self, motor):
        """Config-sourced continuous-torque rating for `motor`, falling back to
        the GridSearch hardcodes when the config doesn't define one."""
        value = self._cont_torque_cfg.get(motor)
        return value if value is not None else \
            test_builder.GRID_CONT_TORQUE_FALLBACK[motor]

    def _apply_field_states(self, seg):
        """Enable only the fields the current segment actually uses.

        A gridpoint segment drives none of the trace-shaping knobs: grid_search
        has its own settle/transition settings, and grid segments don't support
        repeats. Position shaping only does anything when a channel is in
        position mode, so the drawer folds away and greys out otherwise."""
        grid = test_builder.is_gridpoint(seg)
        for widget in (self.sec_rate, self.sec_settle, self.repeats_spin):
            widget.setEnabled(not grid)
        shaping = not grid and 'position' in (self.primary_mode.currentText(),
                                              self.secondary_mode.currentText())
        if not shaping and self.shaping_toggle.isChecked():
            self.shaping_toggle.setChecked(False)  # fold, don't leave it grey
        self.shaping_toggle.setEnabled(shaping)
        self.primary_accel.setEnabled(shaping)
        self._refresh_shaping_toggle(shaping)

    def _on_pattern_changed(self, pattern):
        seg = self._current_segment()
        if seg is None or self._loading:
            return
        seg['pattern'] = pattern
        seg['params'] = {k: v[1] for k, v in test_builder.PATTERNS[pattern].items()}
        # Params reset to defaults, so any generator spec describing the old
        # pattern's level list is now a lie. The secondary's spec is unaffected.
        sec_spec = self._gen_spec(seg, 'secondary')
        if sec_spec:
            seg['levels_gen'] = {'secondary': sec_spec}
        else:
            seg.pop('levels_gen', None)
        if pattern == 'gridpoint':
            # Grid searches only run velocity x torque; snap the modes to that
            # pair and seed the continuous-torque rating from the rig config.
            self._loading = True
            if {self.primary_mode.currentText(),
                    self.secondary_mode.currentText()} != {'velocity', 'torque'}:
                self.primary_mode.setCurrentText('velocity')
                self.secondary_mode.setCurrentText('torque')
            self._loading = False
            torque_motor = (self.primary_motor.currentText()
                            if self.primary_mode.currentText() == 'torque'
                            else ('output' if self.primary_motor.currentText() == 'input'
                                  else 'input'))
            seg['params']['continuous_torque'] = self._cont_torque_default(torque_motor)
        self._apply_field_states(seg)
        self._rebuild_param_widgets(seg)
        item = self.seg_list.currentItem()
        if item:
            item.setText(f"{seg['id']}: {pattern}")
        self._commit_form()

    def _commit_form(self, *_):
        seg = self._current_segment()
        if seg is None or self._loading:
            return
        if test_builder.is_gridpoint(seg):
            # Keep the other motor on the complementary mode of the
            # velocity/torque pair as the pattern-motor mode changes.
            prim = self.primary_mode.currentText()
            if prim in ('velocity', 'torque'):
                other = 'torque' if prim == 'velocity' else 'velocity'
                if self.secondary_mode.currentText() != other:
                    self.secondary_mode.blockSignals(True)
                    self.secondary_mode.setCurrentText(other)
                    self.secondary_mode.blockSignals(False)
        seg['primary'] = {'motor': self.primary_motor.currentText(),
                          'control_mode': self.primary_mode.currentText(),
                          'accel': self.primary_accel.value()}
        # Generator inputs for every level list on the form, keyed the same way
        # _gen_spec reads them; only Fill-backed lists contribute an entry.
        gen = {'secondary': self.levels_field.spec()}
        try:
            levels = _parse_levels(self.levels_field.text())
        except ValueError:
            self.validation_label.setText('Levels must be comma-separated numbers')
            return
        seg['secondary'] = {'control_mode': self.secondary_mode.currentText(),
                            'levels': levels,
                            'rate': self.sec_rate.value(),
                            'settle_s': self.sec_settle.value()}
        seg['repeats'] = (1 if test_builder.is_gridpoint(seg)
                          else self.repeats_spin.value())
        params = {}
        for key, widget in self._param_widgets.items():
            typ = test_builder.PATTERNS[seg['pattern']][key][2]
            if typ is bool:
                params[key] = widget.isChecked()
            elif typ is int:
                params[key] = widget.value()
            elif typ is list:
                try:
                    params[key] = _parse_levels(widget.text())
                except ValueError:
                    self.validation_label.setText(
                        f'{key} must be comma-separated numbers')
                    return
                gen[key] = widget.spec()
            else:
                params[key] = widget.value()
        seg['params'] = params
        # Stay byte-identical to a pre-generator recipe when nothing on the
        # form was generated, so old tests don't re-hash on open.
        gen = {k: v for k, v in gen.items() if v}
        if gen:
            seg['levels_gen'] = gen
        else:
            seg.pop('levels_gen', None)
        self._apply_field_states(seg)
        self._update_preview()

    # --- preview & validation ----------------------------------------------

    def _recipe(self):
        return {'name': self.name_edit.text(), 'segments': self.segments}

    def _update_preview(self):
        # Route each segment's two channels onto the mode plots, repeats
        # unrolled, with a NaN break between segments so curves don't bridge.
        series = {(motor, mode): ([], []) for motor in test_builder.MOTORS
                  for mode in test_builder.MODES}
        t_offset = 0.0
        issues = []
        for seg in self.segments:
            issues.extend(f"[{seg['id']}] {m}"
                          for m in test_builder.validate_segment(seg, self.limits))
            try:
                if test_builder.is_gridpoint(seg):
                    # Approximate keyframes (cooldowns not modeled); the
                    # post-save expansion shows the exact command stream.
                    cols, rows = test_builder.gridpoint_preview_rows(
                        seg, self.limits)
                else:
                    cols, rows = test_builder.compile_segment(seg)
            except Exception as e:
                issues.append(f"[{seg['id']}] {type(e).__name__}: {e}")
                continue
            times = np.array([r[0] for r in rows])
            duration = float(times[-1]) if len(times) else 0.0
            for rep in range(int(seg.get('repeats', 1))):
                for idx, col in ((1, cols[1]), (2, cols[2])):
                    motor, mode = col.split('_motor_')
                    ts, vs = series[(motor, mode)]
                    ts.extend(times + t_offset)
                    vs.extend(r[idx] for r in rows)
                    ts.append(np.nan)
                    vs.append(np.nan)
                t_offset += duration
        for (motor, mode), (ts, vs) in series.items():
            self.curves[mode][motor].setData(np.array(ts), np.array(vs))
        self.validation_label.setText('\n'.join(issues))
        self._update_grid_warning()
        self.status.setText(f'Total duration: {t_offset:.1f} s '
                            f'({len(self.segments)} segment(s))')
        self._check_dirty()
        return issues

    def _update_grid_warning(self):
        """Yellow-flag the current gridpoint segment's continuous-torque field
        when its value differs from the rig config's default (e.g. a recipe
        saved under an older config, or a deliberate per-test override)."""
        seg = self._current_segment()
        warning = ''
        stale = False
        if seg is not None and test_builder.is_gridpoint(seg):
            motor = test_builder.gridpoint_torque_motor(seg)
            if motor is not None:
                default = self._cont_torque_default(motor)
                value = float(seg['params'].get('continuous_torque', default))
                stale = abs(value - default) > 1e-9
                if stale:
                    warning = (f'Continuous torque {value:g} Nm differs from '
                               f'the {motor} motor config default '
                               f'({default:g} Nm)')
        widget = self._param_widgets.get('continuous_torque')
        if widget is not None:
            widget.setStyleSheet('background-color: #fff3b0;' if stale else '')
        self.warn_label.setText(warning)

    def _check_dirty(self):
        """Announce (once) that the recipe no longer matches the armed file."""
        if self._armed_hash is None:
            return
        if test_builder.recipe_hash(self._recipe()) != self._armed_hash:
            self._armed_hash = None
            self.test_dirty.emit()

    def _announce_loaded(self, rel_path, editable):
        """Hand a test to the main GUI to arm, and take the baseline the
        dirty check compares against."""
        self._armed_hash = (test_builder.recipe_hash(self._recipe())
                            if editable else None)
        self.test_loaded.emit(rel_path)

    # --- save / open -------------------------------------------------------

    def _save(self):
        self._commit_form()
        issues = self._update_preview()
        if issues:
            self.status.setText('Fix validation issues before saving.')
            return
        recipe = self._recipe()
        try:
            rel_path = test_builder.save_test(recipe, dyno_paths.dyno_test_directory)
        except Exception as e:
            self.status.setText(f'Save FAILED — {type(e).__name__}: {e}')
            return
        # Authoritative check: run the saved file through the same expansion
        # (TestManager + TestTrace asserts) the preview window uses.
        try:
            result = test_preview.expand_test(rel_path, self.mode)
        except Exception as e:
            self.status.setText(f'Saved {rel_path}, but it fails to load: '
                                f'{type(e).__name__}: {e}')
            return
        dur = result['t'][-1] if result['n_cycles'] else 0.0
        # Show the verified expansion (the exact command stream the rig will
        # run) in place of the idealized keyframe preview; the next edit
        # reverts the plot to the live keyframes.
        self._plot_expansion(result)
        self.status.setText(f'Saved {rel_path}  ({dur:.1f}s, '
                            f'{result["n_cycles"]:,} cycles) — verified.')
        self.name_edit.setText(test_builder.sanitize_name(recipe['name']))
        self.test_saved.emit(rel_path)
        self._announce_loaded(rel_path, editable=True)

    def _open_test(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Test', dyno_paths.dyno_test_directory,
            'Test plans (*.yaml *.yml)')
        if not path:
            return
        rel = test_builder.rel_test_path(path, dyno_paths.dyno_test_directory)
        if rel is None:
            self.status.setText(
                f'{os.path.basename(path)} is outside '
                f'{dyno_paths.dyno_test_directory} — tests (and the trace csvs '
                'they reference) must live under the tests directory to run.')
            return
        try:
            recipe, stale = test_builder.load_recipe(
                path, dyno_paths.dyno_test_directory)
        except Exception as e:
            self.status.setText(f'Open FAILED — {type(e).__name__}: {e}')
            return
        if recipe is None:
            # No embedded recipe (hand-written yaml / grid search / loops):
            # show its authoritative expansion, read-only.
            self._load_expansion(rel)
            return
        self._set_view_only(False)
        self._armed_hash = None  # _select_segment must not fire a stale dirty
        self.segments = recipe['segments'] or [test_builder.default_segment()]
        self.name_edit.setText(recipe.get('name', 'untitled'))
        self._refresh_list(0)
        self._select_segment(0)
        # The file on disk is what the rig would run, so arm it even when the
        # recipe is stale — but say so, because the plot above is drawn from the
        # recipe and the run would use the (differing) csvs next to it.
        self.status.setText(f'Opened {rel}' if not stale else
                            f'Opened {rel} with warnings: ' + '; '.join(stale))
        self._announce_loaded(rel, editable=True)

    def _new_test(self):
        self._set_view_only(False)
        self._armed_hash = None
        self.segments = [test_builder.default_segment()]
        self.name_edit.setText('my_test')
        self._refresh_list(0)
        self._select_segment(0)
        self.status.setText('')

    # --- read-only expansion view ------------------------------------------

    def _set_view_only(self, view_only, label=None):
        self._view_only = view_only
        for widget in (self.name_edit, self.save_button, self.seg_list,
                       self.seg_buttons_wrap, self.seg_box, self.primary_box,
                       self.secondary_box, self.shaping_toggle,
                       self.shaping_panel):
            widget.setEnabled(not view_only)
        self.validation_label.setStyleSheet(
            'color: gray;' if view_only else 'color: #d62728;')
        self.validation_label.setText(
            f'Viewing {label} (read-only). Use New Test to start editing.'
            if view_only else '')
        if view_only:
            self.warn_label.setText('')

    def _load_expansion(self, rel_path):
        if self._expansion_thread is not None and self._expansion_thread.isRunning():
            return  # ignore re-entrant loads while one is in flight
        self._set_view_only(True, label=rel_path)
        self._clear_curves()
        self.status.setText(f'Expanding {rel_path}…')
        self.open_button.setEnabled(False)
        self._expansion_thread = ExpansionThread(rel_path, self.mode)
        self._expansion_thread.done.connect(
            lambda result: self._on_expansion_done(rel_path, result))
        self._expansion_thread.failed.connect(
            lambda msg: self.status.setText(f'{rel_path}:  FAILED — {msg}'))
        self._expansion_thread.finished.connect(
            lambda: self.open_button.setEnabled(True))
        self._expansion_thread.start()

    def _clear_curves(self):
        for by_motor in self.curves.values():
            for curve in by_motor.values():
                curve.setData([], [])

    def _plot_expansion(self, result):
        n = result['n_cycles']
        if n == 0:
            self._clear_curves()
            return
        step = max(1, n // _MAX_DISPLAY_POINTS)
        t_ds = result['t'][::step]
        for mode, by_motor in self.curves.items():
            for motor, curve in by_motor.items():
                curve.setData(t_ds, result[f'{motor}_{mode}'][::step])

    def _on_expansion_done(self, rel_path, result):
        self._plot_expansion(result)
        if result['n_cycles'] == 0:
            self.status.setText(f'{rel_path}:  no commands produced')
            return
        msg = (f"{rel_path}:  {result['t'][-1]:.1f}s test, "
               f"{result['n_cycles']:,} cycles")
        if result['truncated']:
            msg += '  (truncated at preview limit)'
        self.status.setText(msg)
        self._announce_loaded(rel_path, editable=False)
