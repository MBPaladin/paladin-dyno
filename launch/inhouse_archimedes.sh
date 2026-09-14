#!/usr/bin/env bash
set -e
# Archimedes Drive setup on the in-house rig. Same as inhouse.sh but with its
# own config (dyno/config/inhouse_archimedes_dyno_config.yaml).
cd "$(dirname "$0")/.."
exec env PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 \
  DYNO_AKD_PDO_PROFILE="${DYNO_AKD_PDO_PROFILE:-compact}" \
  .venv/bin/python dyno/src/gui.py --config inhouse_archimedes "$@"
