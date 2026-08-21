"""Qt environment workarounds shared by every GUI entry point.

Import this module BEFORE creating the QApplication (and ideally before any
other Qt work). Both settings below work around one root cause: setup.sh grants
the venv interpreter file capabilities (cap_net_raw etc), so the kernel execs us
with AT_SECURE=1 and dumpable=2, which leaves /proc/<pid>/* owned by root.
xdg-desktop-portal probes /proc/<pid>/root to classify the caller, cannot open
it, and denies every request:
  "Portal operation not allowed: Unable to open /proc/<pid>/root"
Dropping the capabilities is not an option - the rig GUI itself calls
sched_setscheduler(SCHED_FIFO) and forks the EtherCAT Controller, which
inherits its privileges. PR_SET_DUMPABLE(1) restores the /proc ownership but
the portal still refuses, so both fixes below avoid the portal instead.

Kept in one module rather than copied per entry point: two copies of a
workaround this obscure would diverge.
"""

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

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
