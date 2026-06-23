#!/usr/bin/env bash
# launch_baselines_parallel.sh — launch SupSup + Progressive baseline sweep
# across multiple GPUs.
#
# Strategy: shard by (task_family, seed) so each GPU writes a distinct
# per-seed JSON, no concurrent writes per file.
#   results/revision_<task>_baselines_s<seed>.json
#
# A merge step (run_revision_merge.py) collapses the per-seed shards into
# a single results/revision_<task>_baselines.json once everything is done.

set -e
cd "$(dirname "$0")/.."

TASKS=(synthetic mnist_shared permuted rotated)
SEEDS=(0 1 2 3 4)
METHODS="SupSup Progressive"
NUM_GPUS=8

mkdir -p logs

echo "Launching baseline sweep across ${NUM_GPUS} GPUs"
echo "  Tasks:   ${TASKS[*]}"
echo "  Seeds:   ${SEEDS[*]}"
echo "  Methods: ${METHODS}"
echo

i=0
for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu=$(( i % NUM_GPUS ))
        suffix="_baselines_s${seed}"
        log="logs/baselines_${task}_s${seed}.log"
        echo "  GPU${gpu}: ${task} seed ${seed} -> ${log}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
            uv run python experiments/run_revision.py \
                --tasks "${task}" \
                --seeds "${seed}" \
                --methods ${METHODS} \
                --results-suffix "${suffix}" \
                > "${log}" 2>&1 &
        i=$(( i + 1 ))
        sleep 0.5
    done
done

echo
echo "Started ${i} jobs. Tail logs/baselines_*.log for progress."
wait
echo "All jobs complete."
