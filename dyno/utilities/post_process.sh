#!/usr/bin/env bash
set -e
# Resolve path-like arguments against the CALLER's directory first; the cd below
# moves to the repo root (so the venv and `deployment.dyno_paths` imports
# resolve) and would otherwise reinterpret a relative path against it.
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
  .venv/bin/python dyno/src/post_processor.py "${args[@]}"
