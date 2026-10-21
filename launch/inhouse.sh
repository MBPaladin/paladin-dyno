#!/usr/bin/env bash
set -e
# Run from the repo root so the venv resolves and `dyno.src.*` imports work.
cd "$(dirname "$0")/.."

# Keep a copy of everything this run prints (the GUI, the control process and
# their shutdown messages) in a file of its own, named for the launch time, so
# relaunching never overwrites the output of an earlier run. Still echoed to the
# terminal. `tee -i` ignores the terminal's Ctrl-C so the log survives to catch
# the shutdown messages that follow it.
CONSOLE_LOG_DIR="dyno/logs/console"
mkdir -p "$CONSOLE_LOG_DIR"
CONSOLE_LOG="$CONSOLE_LOG_DIR/$(date +%Y-%m-%d_%H%M%S)_inhouse.log"
exec > >(tee -i "$CONSOLE_LOG") 2>&1
echo "Console log: $CONSOLE_LOG"
# The in-house rig's Rev B AKDs return statusword 0x0000 over PDO with the full
# 'extended' map; 'compact' is the largest profile they handle (see devices.py).
# Still overridable from the environment for bisecting.
exec env PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 \
  DYNO_AKD_PDO_PROFILE="${DYNO_AKD_PDO_PROFILE:-compact}" \
  .venv/bin/python dyno/src/gui.py --config inhouse "$@"
