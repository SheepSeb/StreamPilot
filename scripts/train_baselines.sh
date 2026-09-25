#!/usr/bin/env bash
# Train the PPO and SAC baselines over multiple seeds, in parallel.
#
#   scripts/train_baselines.sh                                  # waypoint, ppo+sac, seeds 0 1 2
#   scripts/train_baselines.sh landing                           # a different task
#   scripts/train_baselines.sh waypoint --algos ppo --seeds 0 1 2 3 4
#   scripts/train_baselines.sh waypoint --steps 500000 --jobs 4
#   scripts/train_baselines.sh waypoint -- --detection-noise 0.01 --detection-dropout 0.05
#
# Runs go to runs/ALGO_TASK_seedSEED/ (streampilot-train's default), and logs to
# runs/ALGO_TASK_seedSEED/train.log. Each algo is single-threaded (train.py calls
# torch.set_num_threads(1)), so --jobs runs are cheap to parallelize across cores.

set -euo pipefail

TASK="waypoint"
ALGOS=(ppo sac)
SEEDS=(1 2 3 4 5)
STEPS=2000000
JOBS=$(nproc)
EXTRA_ARGS=()

if [[ $# -gt 0 && "$1" != --* ]]; then
    TASK="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algos) IFS=' ' read -r -a ALGOS <<< "$2"; shift 2 ;;
        --seeds) SEEDS=(); shift; while [[ $# -gt 0 && "$1" != --* ]]; do SEEDS+=("$1"); shift; done ;;
        --steps) STEPS="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

echo "task=$TASK algos=${ALGOS[*]} seeds=${SEEDS[*]} steps=$STEPS jobs=$JOBS extra=${EXTRA_ARGS[*]-}"

pids=()
for algo in "${ALGOS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out="runs/${algo}_${TASK}_seed${seed}"
        mkdir -p "$out"
        echo "starting $out"
        uv run streampilot-train "$TASK" --algo "$algo" --seed "$seed" --steps "$STEPS" \
            "${EXTRA_ARGS[@]}" > "$out/train.log" 2>&1 &
        pids+=($!)
        while (( $(jobs -rp | wc -l) >= JOBS )); do
            wait -n
        done
    done
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

if [[ $status -eq 0 ]]; then
    echo "all runs finished"
else
    echo "one or more runs failed; check runs/*/train.log" >&2
fi
exit $status
