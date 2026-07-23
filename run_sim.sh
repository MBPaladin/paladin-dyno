#!/usr/bin/env bash
# Launch the dyno GUI in simulation mode (no hardware needed).
# Usage: ./run_sim.sh [--gearbox|--actuator|--actuator_production] [gui args...]
# Defaults to --gearbox.
set -e
cd "$(dirname "$0")"
MODE_ARGS="$@"
if [[ "$MODE_ARGS" != *"--gearbox"* && "$MODE_ARGS" != *"--actuator"* && "$MODE_ARGS" != *"--actuator_production"* ]]; then
    MODE_ARGS="--actuator_production $MODE_ARGS"
fi

# QT_QPA_PLATFORM=xcb: force X11 (XWayland) instead of native Wayland.
# WSLg's Wayland compositor mishandles popup/grab windows (combobox dropdowns
# detach, stick to stale positions, or refuse to reopen).
# Resolve the VS Code CLI before sudo strips the interop PATH, so the GUI's
# safety "edit" links can open the config in the user's editor.
exec sudo env PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 \
  DYNO_CODE_CLI="$(command -v code || true)" \
  DISPLAY="$DISPLAY" WAYLAND_DISPLAY="$WAYLAND_DISPLAY" \
  XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" XAUTHORITY="$XAUTHORITY" \
  QT_QPA_PLATFORM=xcb \
  .venv/bin/python dyno/src/gui.py $MODE_ARGS --sim
