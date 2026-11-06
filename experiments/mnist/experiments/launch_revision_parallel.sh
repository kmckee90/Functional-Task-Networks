#!/usr/bin/env bash
# launch_revision_parallel.sh — launch run_revision.py across multiple GPUs.
#
# Strategy: shard by (task_family, seed) so each GPU writes a distinct
# per-seed JSON, no concurrent writes per file.
#   results/revision_<task>_s<seed>.json
#
# A merge step (run_revision_merge.py) collapses the per-seed shards into
# a single results/revision_<task>.json once everything is done.
#
# Usage:
#   bash experiments/launch_revision_parallel.sh
#
# Edit the lists below to control scope.

set -e
cd "$(dirname "$0")/.."

TASKS=(synthetic permuted rotated mnist_shared)
SEEDS=(0 1 2 3 4)
METHODS="FTN FTN+Router FTN_noreplay FixedMask FixedMask+Router NoMask NoMask+ER"
NUM_GPUS=8

mkdir -p logs

echo "Launching revision sweep across ${NUM_GPUS} GPUs"
echo "  Tasks:   ${TASKS[*]}"
echo "  Seeds:   ${SEEDS[*]}"
echo "  Methods: ${METHODS}"
echo

i=0
for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu=$(( i % NUM_GPUS ))
        suffix="_s${seed}"
        log="logs/revision_${task}_s${seed}.log"
        echo "  GPU${gpu}: ${task} seed ${seed} -> ${log}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
            python3 experiments/run_revision.py \
                --tasks "${task}" \
                --seeds "${seed}" \
                --methods ${METHODS} \
                --results-suffix "${suffix}" \
                > "${log}" 2>&1 &
        i=$(( i + 1 ))
        # small stagger to spread CUDA init load
        sleep 0.5
    done
done

echo
echo "Started ${i} jobs. Tail logs/revision_*.log for progress."
echo "Once all done, run: uv run python experiments/run_revision_merge.py"
wait
echo "All jobs complete."
