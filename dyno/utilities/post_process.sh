#!/usr/bin/env bash
set -e
# Run from the repo root so the venv resolves and `deployment.dyno_paths` imports work.
cd "$(dirname "$0")/.."
exec env PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 \
  .venv/bin/python dyno/src/post_processor.py "$@"
