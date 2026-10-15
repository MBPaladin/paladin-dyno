"""AnalysisWindow: pick a log, tick candidates, run analyze.sh, read findings.

Plan construction happens in-process on a worker thread (it is just opening
the HDF5 and running predicates). Execution shells out to analyze.sh --plan in
a subprocess: it imports matplotlib, holds dozens of figures, can run for
minutes, and this keeps the UI path and the CLI path literally the same code.
"""

import json
import os
import re

from PySide6.QtCore import QProcess, QSettings, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QFileDialog, QHBoxLayout, QLabel,
                               QMessageBox, QPlainTextEdit, QPushButton,
                               QSplitter, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from deployment import dyno_paths
from dyno.src.analysis import runner
from dyno.src.analysis.registry import get_processor
from dyno.src.analysis.segment import open_log
from dyno.src.analysis_ui import plan_model
from dyno.src.analysis_ui.param_dialog import ParamDialog

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))
_ANALYZE_SH = os.path.join(_REPO_ROOT, 'dyno', 'utilities', 'analyze.sh')

_VERDICT_COLORS = {'maybe': '#c8a200', 'no': '#9a9a9a', 'error': '#c62828'}
# Matches the per-entry header execute() prints: "<title>  [<label>]"
_HEADER_RE = re.compile(r'^(?P<title>\S.*\S)  \[(?P<label>.+)\]$')


class PlanBuildThread(QThread):
    """build_plan() off the UI thread -- opening the HDF5 and running every
    predicate is exactly the work behind the checkbox list."""
    done = Signal(str, list)         # resolved hdf5 path, display entries
    failed = Signal(str)

    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir

    def run(self):
        try:
            log_path = plan_model.resolve_log_dir(self.log_dir)
            with open_log(self.log_dir) as log:
                entries = plan_model.display_entries(runner.build_plan(log))
            self.done.emit(log_path, entries)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')


class RecheckThread(QThread):
    """Re-run one processor's predicate after a param edit (§8.2): a param
    change can change whether a test is applicable, so the row must reshade.

    build_plan installs overrides on the registered Processor singletons; in
    the CLI that process exits, but here they must be restored or they would
    leak into every later plan build in this window."""
    done = Signal(str, list)         # processor name, its rebuilt entries
    failed = Signal(str)

    def __init__(self, log_dir, processor, overrides):
        super().__init__()
        self.log_dir = log_dir
        self.processor = processor
        self.overrides = overrides

    def run(self):
        proc = get_processor(self.processor)
        saved = proc.overrides
        try:
            with open_log(self.log_dir) as log:
                plan = runner.build_plan(log, only=self.processor,
                                         overrides=self.overrides)
                entries = plan_model.display_entries(plan)
            self.done.emit(self.processor, entries)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}')
        finally:
            proc.overrides = saved


class AnalysisWindow(QWidget):

    def __init__(self, log_dir=None):
        super().__init__()
        self.setWindowTitle('Dyno Analysis')
        self.resize(1100, 750)
        self._log_dir = None
        self._entries = []
        self._items = {}             # (processor, label) -> QTreeWidgetItem
        self._build_thread = None
        self._proc = None
        self._no_override_confirmed = False
        self._settings = QSettings('paladin', 'dyno-analysis')

        layout = QVBoxLayout(self)

        picker = QHBoxLayout()
        self._pick_btn = QPushButton('Select log folder…')
        self._pick_btn.clicked.connect(self._pick_folder)
        self._log_label = QLabel('No log selected')
        self._log_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        picker.addWidget(self._pick_btn)
        picker.addWidget(self._log_label, stretch=1)
        layout.addLayout(picker)

        split = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(split, stretch=1)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(['Candidate', 'Status', ''])
        self._tree.setColumnWidth(0, 580)
        self._tree.setColumnWidth(1, 120)
        self._recheck = None
        self._tree.currentItemChanged.connect(self._show_reason)
        self._tree.itemChanged.connect(self._item_toggled)
        split.addWidget(self._tree)

        # The reason strings are dense and numeric -- the operator's main
        # input. A tooltip alone would leave them hover-only.
        self._reason = QLabel('')
        self._reason.setWordWrap(True)
        self._reason.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        split.addWidget(self._reason)

        self._output = QPlainTextEdit()
        self._output.setReadOnly(True)
        self._output.setMaximumBlockCount(5000)
        split.addWidget(self._output)
        split.setSizes([380, 60, 260])

        controls = QHBoxLayout()
        self._all_yes_btn = QPushButton('Select all "yes"')
        self._all_yes_btn.clicked.connect(lambda: self._select(verdict='yes'))
        self._none_btn = QPushButton('Select none')
        self._none_btn.clicked.connect(lambda: self._select(verdict=None))
        self._show_no = QCheckBox('Show "no" candidates')
        self._show_no.setToolTip(
            'Rows the predicate rejected are hidden by default. Tick to see '
            'them and their reasons -- a "no" can still be forced.')
        self._show_no.toggled.connect(self._apply_filter)
        # Off by default and not remembered between launches: re-rendering
        # rewrites every section of the report this log belongs to, and that
        # should be something the operator asks for each time rather than
        # something a stale tick box does behind them.
        self._make_report = QCheckBox('Update report after run')
        self._make_report.setToolTip(
            'After the analysis finishes, re-render the LaTeX report sections '
            'for whichever report.yaml this log sits under. Does nothing if it '
            'sits under none.')
        self._run_btn = QPushButton('Run')
        self._run_btn.clicked.connect(self._run)
        self._cancel_btn = QPushButton('Cancel')
        self._cancel_btn.clicked.connect(self._cancel)
        self._cancel_btn.setEnabled(False)
        self._open_btn = QPushButton('Open analysis folder')
        self._open_btn.clicked.connect(self._open_folder)
        for b in (self._all_yes_btn, self._none_btn, self._show_no,
                  self._make_report):
            controls.addWidget(b)
        controls.addStretch(1)
        for b in (self._open_btn, self._cancel_btn, self._run_btn):
            controls.addWidget(b)
        layout.addLayout(controls)

        self._set_have_plan(False)
        if log_dir:
            self._load_log(os.path.abspath(log_dir))

    # ------------------------------------------------------------ log loading

    def _pick_folder(self):
        # Runs no longer necessarily live under dyno/logs -- the rig GUI can
        # save one anywhere -- so remember where the last one was opened from
        # and start there. dyno/logs stays the first-launch default.
        start = (self._log_dir
                 or self._settings.value('last_log_parent')
                 or dyno_paths.dyno_logs_directory)
        folder = QFileDialog.getExistingDirectory(
            self, 'Select log folder', start)
        if folder:
            self._load_log(folder)

    def _load_log(self, folder):
        if self._build_thread and self._build_thread.isRunning():
            return
        self._log_dir = folder
        self._settings.setValue('last_log_parent', os.path.dirname(folder))
        self._log_label.setText(f'{folder}  (building plan…)')
        self._tree.clear()
        self._items.clear()
        self._reason.setText('')
        self._set_have_plan(False)
        self._build_thread = PlanBuildThread(folder)
        self._build_thread.done.connect(self._plan_ready)
        self._build_thread.failed.connect(self._plan_failed)
        self._build_thread.start()

    def _plan_failed(self, msg):
        self._log_label.setText('No log selected')
        self._log_dir = None
        QMessageBox.warning(self, 'Cannot open log', msg)

    def _plan_ready(self, log_path, entries):
        self._log_label.setText(log_path)
        self._entries = entries
        plan_model.restore_selection(
            entries, os.path.join(self._log_dir, runner.PLAN_FILENAME))
        self._tree.blockSignals(True)
        for label, group in plan_model.group_by_label(entries):
            parent = QTreeWidgetItem([label, ''])   # count set by _apply_filter
            parent.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self._tree.addTopLevelItem(parent)
            for e in group:
                item = QTreeWidgetItem([e['title'], e['verdict']])
                # Store the lookup key, not the dict: PySide converts a
                # dict stored in item data to a QVariantMap COPY, so edits
                # through item.data() would never reach self._entries.
                item.setData(0, Qt.ItemDataRole.UserRole,
                             f"{e['processor']}\x00{e['label']}")
                item.setToolTip(0, e['reason'])
                if e['verdict'] == 'error':
                    # A crashed predicate is shown, never runnable (§8.3).
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled |
                                  Qt.ItemFlag.ItemIsSelectable)
                else:
                    item.setFlags(item.flags() |
                                  Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(
                        0, Qt.CheckState.Checked if e.get('_selected')
                        else Qt.CheckState.Unchecked)
                self._shade(item, e)
                parent.addChild(item)
                if e['verdict'] != 'error':
                    gear = QPushButton('⚙')
                    gear.setFixedWidth(34)
                    gear.setFlat(True)
                    gear.setToolTip('Edit parameters')
                    gear.clicked.connect(
                        lambda _=False, it=item: self._edit_params(it))
                    self._tree.setItemWidget(item, 2, gear)
                self._items[(e['processor'], e['label'])] = item
            parent.setExpanded(True)
        self._tree.blockSignals(False)
        self._apply_filter()
        self._set_have_plan(True)

    def _entry_of(self, item):
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key is None:
            return None
        processor, label = key.split('\x00', 1)
        for e in self._entries:
            if e['processor'] == processor and e['label'] == label:
                return e
        return None

    def _shade(self, item, e):
        """Row text, tooltip, and verdict colour -- rerun after a recheck, so
        it must also reset styling a previous verdict applied."""
        modified = ' •' if plan_model.deltas(e) else ''
        item.setText(0, e['title'] + modified)
        item.setText(1, e['verdict'])
        item.setToolTip(0, e['reason'])
        color = _VERDICT_COLORS.get(e['verdict'])
        for col in (0, 1):
            item.setForeground(col, QBrush(QColor(color)) if color
                               else QBrush())

    # ------------------------------------------------------------- parameters

    def _edit_params(self, item):
        e = self._entry_of(item)
        if self._recheck and self._recheck.isRunning():
            QMessageBox.information(self, 'Busy',
                                    'Still rechecking a previous edit.')
            return
        dlg = ParamDialog(e, self)
        if not dlg.exec():
            return
        e['params'] = dlg.values()
        self._shade(item, e)
        # §8.2: a param change can change applicability -- rerun this
        # processor's predicate with the new values and reshade.
        overrides = {k: v for k, v in plan_model.deltas(e).items()
                     if v is not None}
        self._recheck = RecheckThread(self._log_dir, e['processor'], overrides)
        # The overrides are this entry's alone, so only this row may absorb
        # the rebuilt result -- other labels of the same processor keep theirs.
        self._recheck.done.connect(
            lambda proc, ents, it=item: self._recheck_done(it, ents))
        self._recheck.failed.connect(
            lambda msg: self._output.appendPlainText(f'! recheck failed: {msg}'))
        self._recheck.start()
        item.setText(1, 'checking…')

    def _recheck_done(self, item, new_entries):
        e = self._entry_of(item)
        n = next((n for n in new_entries if n['label'] == e['label']), None)
        if n is None:                # the group itself dissolved; keep as-was
            self._shade(item, e)
            return
        e.update({k: n[k] for k in
                  ('verdict', 'reason', 'params', '_baseline')})
        self._shade(item, e)
        if item is self._tree.currentItem():
            self._show_reason(item)

    # -------------------------------------------------------------- selection

    def _item_toggled(self, item, column):
        e = self._entry_of(item)
        if (e is None or column != 0 or
                item.checkState(0) != Qt.CheckState.Checked or
                e['verdict'] != 'no' or self._no_override_confirmed):
            return
        ok = QMessageBox.question(
            self, 'Predicate says no',
            f"The predicate objects:\n\n{e['reason']}\n\n"
            'Forced entries run anyway and their results carry a '
            '"predicate_objected" warning. Force it?')
        if ok == QMessageBox.StandardButton.Yes:
            self._no_override_confirmed = True   # prompt once per session
        else:
            item.setCheckState(0, Qt.CheckState.Unchecked)

    def _apply_filter(self):
        """Hide unchecked 'no' rows unless the operator asks to see them.

        A real log is mostly 'no': an 18-behaviour inertia log builds 109 rows
        of which 98 say no, and six processors repeat the same sentence once
        per behaviour. Hiding them is what makes the handful of candidates
        that matter findable.

        Two rows are never hidden. A *checked* 'no' is a force the operator
        (or a restored plan file) asked for, and it will run -- hiding it
        would mean running something invisible. An 'error' row is a crashed
        predicate (SOW 8.3); a processor vanishing with no trace is the exact
        failure that entry exists to prevent.
        """
        show_no = self._show_no.isChecked()
        for item in self._items.values():
            e = self._entry_of(item)
            item.setHidden(not show_no and e['verdict'] == 'no' and
                           item.checkState(0) != Qt.CheckState.Checked)
        for i in range(self._tree.topLevelItemCount()):
            parent = self._tree.topLevelItem(i)
            n = parent.childCount()
            shown = sum(not parent.child(j).isHidden() for j in range(n))
            parent.setHidden(shown == 0)
            parent.setText(1, f'{shown} candidate{"s" if shown != 1 else ""}'
                              + (f' ({n - shown} hidden)' if n > shown else ''))

    def _select(self, verdict):
        for item in self._items.values():
            e = self._entry_of(item)
            if e['verdict'] == 'error':
                continue
            checked = verdict is not None and e['verdict'] == verdict
            item.setCheckState(0, Qt.CheckState.Checked if checked
                               else Qt.CheckState.Unchecked)
        # These buttons restate the whole selection, so a 'no' row that was
        # only visible because it was forced has no reason to stay.
        self._apply_filter()

    def _show_reason(self, item, _prev=None):
        e = self._entry_of(item) if item else None
        self._reason.setText(f"[{e['verdict']}] {e['reason']}" if e else '')

    def _selected_keys(self):
        return {key for key, item in self._items.items()
                if item.checkState(0) == Qt.CheckState.Checked}

    # ---------------------------------------------------------------- running

    def _run(self):
        selected = self._selected_keys()
        if not selected:
            QMessageBox.information(self, 'Nothing selected',
                                    'Tick at least one candidate to run.')
            return
        plan_path = plan_model.write_plan(self._entries, selected,
                                          self._log_dir)
        self._output.clear()
        for key, item in self._items.items():
            if self._entry_of(item)['verdict'] != 'error':
                item.setText(1, 'queued' if key in selected else '')
        self._proc = QProcess(self)
        self._proc.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels)
        self._proc.readyReadStandardOutput.connect(self._read_output)
        self._proc.finished.connect(self._finished)
        self._run_btn.setEnabled(False)
        self._cancel_btn.setEnabled(True)
        self._running_item = None
        argv = [self._log_dir, '--plan', plan_path]
        if self._make_report.isChecked():
            argv.append('--report')
        self._proc.start(_ANALYZE_SH, argv)

    def _read_output(self):
        text = bytes(self._proc.readAllStandardOutput()).decode(
            'utf-8', errors='replace')
        self._output.moveCursor(self._output.textCursor().MoveOperation.End)
        self._output.insertPlainText(text)
        for line in text.splitlines():
            m = _HEADER_RE.match(line)
            if m:
                for key, item in self._items.items():
                    e = self._entry_of(item)
                    if (e['title'], e['label']) == (m['title'], m['label']):
                        self._mark_running(item)
                        break
            elif line.strip().startswith('! FAILED') and self._running_item:
                self._running_item.setText(1, 'failed')
                self._running_item.setForeground(
                    1, QBrush(QColor('#c62828')))
                self._running_item = None

    def _mark_running(self, item):
        if self._running_item is not None:       # previous one finished clean
            self._finish_row(self._running_item)
        item.setText(1, 'running…')
        self._running_item = item

    def _finish_row(self, item):
        item.setText(1, 'ok')
        item.setForeground(1, QBrush(QColor('#2e9e3e')))

    def _finished(self, code, _status):
        if self._running_item is not None:
            self._finish_row(self._running_item)
            self._running_item = None
        self._run_btn.setEnabled(True)
        self._cancel_btn.setEnabled(False)
        self._summarize(code)

    def _cancel(self):
        if self._proc and self._proc.state() != QProcess.ProcessState.NotRunning:
            self._proc.kill()
            self._output.appendPlainText(
                '\n! cancelled -- analysis/ may be partially written; '
                'run again to regenerate it')

    def _summarize(self, code):
        results_path = os.path.join(self._log_dir, runner.ANALYSIS_DIRNAME,
                                    'results.json')
        lines = [f'\n===== finished (exit {code}) =====']
        try:
            with open(results_path) as f:
                results = json.load(f)
            findings = [(f_['level'], r['processor'], r['label'],
                         f_['message'])
                        for r in results for f_ in r['findings']]
            rank = {'error': 0, 'warn': 1, 'info': 2}
            findings.sort(key=lambda x: rank.get(x[0], 3))
            for level, proc, label, msg in findings:
                lines.append(f'[{level}] {proc} {label}: {msg}')
            if not findings:
                lines.append('(no findings)')
        except Exception as exc:
            lines.append(f'could not read results.json: {exc}')
        self._output.appendPlainText('\n'.join(lines))

    def _open_folder(self):
        if self._log_dir:
            folder = os.path.join(self._log_dir, runner.ANALYSIS_DIRNAME)
            QDesktopServices.openUrl(QUrl.fromLocalFile(
                folder if os.path.isdir(folder) else self._log_dir))

    def _set_have_plan(self, have):
        for b in (self._all_yes_btn, self._none_btn, self._run_btn,
                  self._open_btn):
            b.setEnabled(have)

    def closeEvent(self, event):
        self._cancel()
        super().closeEvent(event)
