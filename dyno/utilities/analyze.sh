#!/usr/bin/env bash
set -e
# Resolve path-like arguments against the CALLER's directory before doing
# anything else. The cd below moves to the repo root so the venv and package
# imports resolve, which would otherwise silently reinterpret a relative log
# path against the wrong root -- `analyze.sh logs/2026-01-01_120000` run from
# dyno/ became <repo>/logs/... and failed with a bare NotADirectoryError.
# Flags and --set K=V never name an existing path, so they pass through.
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
  .venv/bin/python -m dyno.src.analysis "${args[@]}"
