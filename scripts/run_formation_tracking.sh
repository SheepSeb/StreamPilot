#!/usr/bin/env bash
# Formation tracking comparison: independent Stream AC, CTDE Stream AC, IPPO, MAPPO and PID,
# N in {1, 2, 3, 5}, 3 seeds, 2M team steps each. Trains, then evaluates the table, then plots the curves.
#
#   scripts/run_formation_tracking.sh                                   # the full experiment
#   scripts/run_formation_tracking.sh --drones 1 2 --steps 200000       # a quick smoke run
#
# Arguments are passed to compare_formation_tracking.py (see its --help). A method that fails stops the run.

set -euo pipefail

uv run python scripts/compare_formation_tracking.py train "$@"
uv run python scripts/compare_formation_tracking.py eval "$@"
uv run python scripts/compare_formation_tracking.py plot "$@"
