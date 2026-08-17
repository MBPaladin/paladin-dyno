#!/usr/bin/env bash
set -e
# Run from the repo root (two levels up from dyno/utilities/) so the venv
# resolves and `dyno.src.*` imports work.
cd "$(dirname "$0")/../.."
exec env PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 \
  .venv/bin/python dyno/src/drive_status.py "$@"
