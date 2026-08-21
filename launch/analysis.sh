#!/usr/bin/env bash
set -e
# Launch the analysis UI. Needs no hardware, no EtherCAT, and no --config.
# Optional argument: a log folder to preselect.
# Resolve a path-like argument against the CALLER's directory before cd-ing to
# the repo root (same reasoning as dyno/utilities/analyze.sh).
args=()
for a in "$@"; do
  if [[ "$a" != -* && -e "$a" ]]; then
    args+=("$(realpath "$a")")
  else
    args+=("$a")
  fi
done

cd "$(dirname "$0")/.."
exec env PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 \
  .venv/bin/python -m dyno.src.analysis_ui "${args[@]}"
