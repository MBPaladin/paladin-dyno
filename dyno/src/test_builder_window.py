"""
Test Builder window (third window, opened from the Test Definition window).

Compose an experiment as a sequence of segments (sawtooth / step /
ramp_release / dwell patterns with a secondary-motor level sweep), see the
commanded curves live as parameters change, and save to
tests/ui_generated_tests/<name>.yaml with the recipe embedded so the test
can be reopened and edited. Saving re-validates through the real
TestManager path (via test_preview.expand_test), so anything that saves
green here will load on the rig.
"""
import os

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox,
                               QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from deployment import dyno_paths
from dyno.src import test_builder, test_preview

_PLOTS = (('Torque', 'Nm', 'torque'),
          ('Velocity', 'rad/s', 'velocity'),
          ('Position', 'rad', 'position'))
_INPUT_PEN = pg.mkPen('#1f77b4', width=2)
_OUTPUT_PEN = pg.mkPen('#d62728', width=2)


def _parse_levels(text):
    values = [v for v in (s.strip() for s in text.split(',')) if v]
    return [float(v) for v in values] or [0.0]


class TestBuilderWindow(QWidget):
    # Emitted with the saved test path (relative to the tests directory).
    test_saved = Signal(str)

    def __init__(self, mode, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.setWindowTitle('Test Builder')
        self.resize(1280, 800)
        try:
            self.limits = test_preview.limits_from_config(mode)
        except Exception:
            self.limits = None  # config unavailable: skip limit checks live
        self.segments = [test_builder.default_segment()]
        self._param_widgets = {}
        self._loading = False  # guard: suppress form->model writes during load
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
        self.open_button = QPushButton('Open…')
        self.open_button.clicked.connect(self._open_existing)
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
        for label, slot in (('Add', self._add_segment),
                            ('Dup', self._dup_segment),
                            ('Del', self._del_segment),
                            ('Up', lambda: self._move_segment(-1)),
                            ('Down', lambda: self._move_segment(1))):
            btn = QPushButton(label)
            btn.clicked.connect(slot)
            seg_buttons.addWidget(btn)
        left.addLayout(seg_buttons)
        left_wrap = QWidget()
        left_wrap.setLayout(left)
        left_wrap.setFixedWidth(210)
        body.addWidget(left_wrap)

        # Middle: segment editor --------------------------------------------
        editor = QVBoxLayout()

        roles = QGroupBox('Motors')
        roles_form = QFormLayout(roles)
        self.primary_motor = QComboBox()
        self.primary_motor.addItems(test_builder.MOTORS)
        roles_form.addRow('Pattern motor', self.primary_motor)
        self.primary_mode = QComboBox()
        self.primary_mode.addItems(test_builder.MODES)
        roles_form.addRow('Pattern control mode', self.primary_mode)
        self.secondary_mode = QComboBox()
        self.secondary_mode.addItems(test_builder.MODES)
        roles_form.addRow('Other motor control mode', self.secondary_mode)
        self.levels_edit = QLineEdit()
        self.levels_edit.setToolTip('Comma-separated setpoints for the other '
                                    'motor; the pattern runs once per level. '
                                    'Use a single 0 to hold it at zero.')
        roles_form.addRow('Other motor levels', self.levels_edit)
        self.sec_rate = self._spin(0.001, 1e6, 1.0)
        roles_form.addRow('Level ramp rate [units/s]', self.sec_rate)
        self.sec_settle = self._spin(0.0, 1e6, 1.0)
        roles_form.addRow('Settle at level [s]', self.sec_settle)
        editor.addWidget(roles)

        pattern_box = QGroupBox('Pattern')
        self.pattern_layout = QFormLayout(pattern_box)
        self.pattern_combo = QComboBox()
        self.pattern_combo.addItems(list(test_builder.PATTERNS))
        self.pattern_layout.addRow('Type', self.pattern_combo)
        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(1, 100000)
        self.pattern_layout.addRow('Repeat segment', self.repeats_spin)
        editor.addWidget(pattern_box)
        editor.addStretch(1)

        self.validation_label = QLabel('')
        self.validation_label.setWordWrap(True)
        self.validation_label.setStyleSheet('color: #d62728;')
        editor.addWidget(self.validation_label)

        editor_wrap = QWidget()
        editor_wrap.setLayout(editor)
        editor_wrap.setFixedWidth(360)
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
        self.levels_edit.editingFinished.connect(self._commit_form)
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
        self.secondary_mode.setCurrentText(seg['secondary']['control_mode'])
        self.levels_edit.setText(', '.join(f'{v:g}' for v in seg['secondary']['levels']))
        self.sec_rate.setValue(seg['secondary'].get('rate', 1.0))
        self.sec_settle.setValue(seg['secondary'].get('settle_s', 0.0))
        self.repeats_spin.setValue(int(seg.get('repeats', 1)))
        self.pattern_combo.setCurrentText(seg['pattern'])
        self._rebuild_param_widgets(seg)
        self._loading = False
        self._update_preview()

    def _rebuild_param_widgets(self, seg):
        for widget in self._param_widgets.values():
            label = self.pattern_layout.labelForField(widget)
            self.pattern_layout.removeRow(widget)
            del label
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
                widget = QLineEdit(', '.join(f'{float(v):g}' for v in value))
                widget.editingFinished.connect(self._commit_form)
            else:
                widget = self._spin(-1e9, 1e9, float(value))
                widget.valueChanged.connect(self._commit_form)
            self.pattern_layout.addRow(label, widget)
            self._param_widgets[key] = widget

    def _on_pattern_changed(self, pattern):
        seg = self._current_segment()
        if seg is None or self._loading:
            return
        seg['pattern'] = pattern
        seg['params'] = {k: v[1] for k, v in test_builder.PATTERNS[pattern].items()}
        self._rebuild_param_widgets(seg)
        item = self.seg_list.currentItem()
        if item:
            item.setText(f"{seg['id']}: {pattern}")
        self._commit_form()

    def _commit_form(self, *_):
        seg = self._current_segment()
        if seg is None or self._loading:
            return
        seg['primary'] = {'motor': self.primary_motor.currentText(),
                          'control_mode': self.primary_mode.currentText()}
        try:
            levels = _parse_levels(self.levels_edit.text())
        except ValueError:
            self.validation_label.setText('Levels must be comma-separated numbers')
            return
        seg['secondary'] = {'control_mode': self.secondary_mode.currentText(),
                            'levels': levels,
                            'rate': self.sec_rate.value(),
                            'settle_s': self.sec_settle.value()}
        seg['repeats'] = self.repeats_spin.value()
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
            else:
                params[key] = widget.value()
        seg['params'] = params
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
        self.status.setText(f'Total duration: {t_offset:.1f} s '
                            f'({len(self.segments)} segment(s))')
        return issues

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
        self.status.setText(f'Saved {rel_path}  ({dur:.1f}s, '
                            f'{result["n_cycles"]:,} cycles) — verified.')
        self.name_edit.setText(test_builder.sanitize_name(recipe['name']))
        self.test_saved.emit(rel_path)

    def _open_existing(self):
        start_dir = os.path.join(dyno_paths.dyno_test_directory,
                                 test_builder.GENERATED_TEST_DIR)
        if not os.path.isdir(start_dir):
            start_dir = dyno_paths.dyno_test_directory
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open Generated Test', start_dir, 'Test plans (*.yaml *.yml)')
        if not path:
            return
        try:
            recipe, stale = test_builder.load_recipe(
                path, dyno_paths.dyno_test_directory)
        except Exception as e:
            self.status.setText(f'Open FAILED — {type(e).__name__}: {e}')
            return
        if recipe is None:
            self.status.setText(stale[0])
            return
        self.segments = recipe['segments'] or [test_builder.default_segment()]
        self.name_edit.setText(recipe.get('name', 'untitled'))
        self._refresh_list(0)
        self._select_segment(0)
        if stale:
            self.status.setText('Opened with warnings: ' + '; '.join(stale))
