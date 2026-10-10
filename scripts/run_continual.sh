#!/usr/bin/env bash
# Continual task sequence (tracking -> waypoint -> landing -> tracking, 500k team steps each), for
# independent StreamX, centralised-critic StreamX, MAPPO and IPPO, 3 seeds each, at most JOBS runs at a
# time. Each run writes to runs_continual/METHOD-continue_seedSEED/ (stdout.log there).
#
#   bash scripts/run_continual.sh            # all 12 runs, 4 at a time
#   JOBS=2 SEEDS="1 2" bash scripts/run_continual.sh
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${JOBS:-4}
SEEDS=${SEEDS:-"1 2 3"}
STEPS=${STEPS:-500000}
METHODS=${METHODS:-"istream cstream mappo ippo"}
COMMON="--mode continue --steps-per-task $STEPS --frames 8 --restore-best"
STREAM="$COMMON --compile --step-scale-end 0.1"
# MAPPO and IPPO run their environments in worker processes: cap them, they share the CPU with other runs.
PPO="$COMMON --num-workers 4"

for seed in $SEEDS; do
  for m in $METHODS; do
    case $m in
      istream) echo "istream-continue_seed$seed --method stream $STREAM --seed $seed" ;;
      cstream) echo "cstream-continue_seed$seed --method cstream $STREAM --seed $seed" ;;
      mappo | ippo) echo "$m-continue_seed$seed --method $m $PPO --seed $seed" ;;
    esac
  done
done | xargs -P "$JOBS" -L 1 bash -c '
  out=runs_continual/$0; mkdir -p "$out"
  echo "start $0 $(date +%T)"
  HF_HUB_OFFLINE=1 uv run streampilot-continual "$@" --out "$out" > "$out/stdout.log" 2>&1 \
    && echo "done  $0 $(date +%T)" || echo "FAILED $0 $(date +%T) (see $out/stdout.log)"
'
