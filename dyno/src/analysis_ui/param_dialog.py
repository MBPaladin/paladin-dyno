"""Per-entry parameter form, generated from a processor's param spec.

The spec is `{key: (default, type, help)}`. The important case is a `None`
default, which means "auto-detect": each such field gets an `Auto` checkbox
that disables the widget and reverts the key to the entry's baseline -- the
value the predicate detected when the plan was built -- shown as the greyed
placeholder. Reverting to baseline (rather than pinning a literal null) is
what keeps the saved plan deltas-only: an Auto field writes nothing to the
file, so detection still runs fresh on every re-run.

Types are declared but not constrained; a bad value surfaces as a processor
failure downstream, which is already handled.
"""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QGridLayout, QLabel, QLineEdit,
                               QPushButton, QScrollArea, QSpinBox,
                               QVBoxLayout, QWidget)


def _make_widget(type_, value):
    if type_ is bool:
        w = QCheckBox()
        w.setChecked(bool(value))
        return w
    if type_ is int:
        w = QSpinBox()
        w.setRange(-2**31, 2**31 - 1)
        w.setValue(int(value or 0))
        return w
    if type_ is float:
        w = QDoubleSpinBox()
        w.setRange(-1e12, 1e12)
        w.setDecimals(6)
        w.setValue(float(value or 0.0))
        return w
    w = QLineEdit()
    if value is not None:
        w.setText(str(value))
    return w


def _read_widget(widget, type_):
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.value()
    text = widget.text()
    return type_(text) if text else None


class ParamDialog(QDialog):
    """Edit one plan entry's params. Call values() after exec() accepts."""

    def __init__(self, entry, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{entry['title']} — {entry['label']}")
        self._spec = entry['_param_spec']
        self._baseline = entry.get('_baseline', {})
        self._fields = {}            # key -> (widget, auto_checkbox_or_None)

        outer = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setColumnStretch(1, 1)

        for row2, (key, (default, type_, help_)) in \
                enumerate(self._spec.items()):
            row = row2 * 2
            current = entry['params'].get(key, default)
            widget = _make_widget(type_, current)
            auto = None
            if default is None:
                auto = QCheckBox('Auto')
                detected = self._baseline.get(key)
                placeholder = (f'Auto — detected: {detected}'
                               if detected is not None else 'Auto')
                if isinstance(widget, QLineEdit):
                    widget.setPlaceholderText(placeholder)
                else:
                    auto.setToolTip(placeholder)
                auto.toggled.connect(widget.setDisabled)
                # Manual only when the operator has moved off the detected
                # value; equal-to-detected is indistinguishable from Auto and
                # writes the same (empty) delta either way.
                auto.setChecked(current is None or
                                current == self._baseline.get(key))
            grid.addWidget(QLabel(key), row, 0)
            grid.addWidget(widget, row, 1)
            if auto:
                grid.addWidget(auto, row, 2)
            help_label = QLabel(help_)
            help_label.setWordWrap(True)
            help_label.setStyleSheet('color: #808080; font-size: 11px;')
            grid.addWidget(help_label, row + 1, 0, 1, 3)
            self._fields[key] = (widget, auto)
        grid.setRowStretch(len(self._spec) * 2, 1)

        scroll.setWidget(inner)
        outer.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                   QDialogButtonBox.StandardButton.Cancel)
        reset = QPushButton('Reset to defaults')
        reset.clicked.connect(self._reset)
        buttons.addButton(reset,
                          QDialogButtonBox.ButtonRole.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.resize(560, min(680, 90 + 64 * len(self._spec)))

    def _reset(self):
        for key, (widget, auto) in self._fields.items():
            default = self._spec[key][0]
            if auto:
                auto.setChecked(True)
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(default))
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.setValue(default if default is not None else 0)
            else:
                widget.setText('' if default is None else str(default))

    def values(self):
        """Full params dict for the entry: Auto fields revert to baseline."""
        out = {}
        for key, (widget, auto) in self._fields.items():
            if auto and auto.isChecked():
                out[key] = self._baseline.get(key)
            else:
                out[key] = _read_widget(widget, self._spec[key][1])
        return out
