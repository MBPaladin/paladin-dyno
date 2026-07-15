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
PYTHONPATH=. PYTHONUNBUFFERED=1 exec .venv/bin/python dyno/src/gui.py $MODE_ARGS --sim
