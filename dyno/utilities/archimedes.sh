#!/usr/bin/env bash
set -e
# Run the Archimedes customer analysis pack for one unit.
#
#   ./dyno/utilities/archimedes.sh gbx_1p2p0            everything
#   ./dyno/utilities/archimedes.sh gbx_1p2p0 --list     inventory only
#   ./dyno/utilities/archimedes.sh gbx_1p2p0 --only efficiency
#
# Path-like arguments are resolved against the CALLER's directory before the cd
# below moves to the repo root, for the same reason analyze.sh does it: a
# relative log path would otherwise be reinterpreted against the wrong root.
args=()
for a in "$@"; do
  if [[ "$a" != -* && -e "$a" ]]; then
    args+=("$(realpath "$a")")
  else
    args+=("$a")
  fi
done

cd "$(dirname "$0")/../.."
exec env PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 \
  .venv/bin/python -m dyno.src.archimedes "${args[@]}"
