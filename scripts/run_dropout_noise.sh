#!/usr/bin/env bash
# Detection dropout x noise grid at N = 3, into runs_dropout_noise/.
# Waits for a running scripts/run_formation_tracking.sh to finish first, then trains, evaluates and plots.
#
#   scripts/run_dropout_noise.sh                                      # everything (a lot: see --help)
#   scripts/run_dropout_noise.sh --methods istream_ac ippo --seeds 1  # a cheaper cut
#
# Arguments are passed to compare_dropout_noise.py.

set -euo pipefail

while pgrep -f run_formation_tracking.sh >/dev/null; do sleep 60; done

uv run python scripts/compare_dropout_noise.py train "$@"
uv run python scripts/compare_dropout_noise.py eval "$@"
uv run python scripts/compare_dropout_noise.py plot "$@"
