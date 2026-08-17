#!/usr/bin/env bash
set -e
# Run from the repo root so the venv resolves and `dyno.src.*` imports work.
cd "$(dirname "$0")/.."
# The in-house rig's Rev B AKDs return statusword 0x0000 over PDO with the full
# 'extended' map; 'compact' is the largest profile they handle (see devices.py).
# Still overridable from the environment for bisecting.
exec env PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 \
  DYNO_AKD_PDO_PROFILE="${DYNO_AKD_PDO_PROFILE:-compact}" \
  .venv/bin/python dyno/src/gui.py --config inhouse "$@"
