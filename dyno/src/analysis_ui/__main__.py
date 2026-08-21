"""Standalone entry point: python -m dyno.src.analysis_ui [log_dir]

Needs no rig hardware, no EtherCAT, and no config. An optional log directory
argument preselects that log -- that is how gui.py launches it.
"""

import sys

from dyno.src import qt_env                          # noqa: F401  before QApplication
from PySide6.QtWidgets import QApplication

from dyno.src.analysis_ui.window import AnalysisWindow


def main(argv=None):
    argv = sys.argv if argv is None else argv
    app = QApplication(argv)
    window = AnalysisWindow(log_dir=argv[1] if len(argv) > 1 else None)
    window.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
