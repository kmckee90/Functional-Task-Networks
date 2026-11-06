#!/usr/bin/env bash
# Support-set size sweep on Permuted/RotatedMNIST + synthetic.
# Run this AFTER the main sweep finishes (or alongside if GPUs are free).
# Each (task, seed) shard sweeps n in {1, 5, 10, 50, 256} for FTN only.

set -e
cd "$(dirname "$0")/.."

TASKS=(synthetic permuted rotated)
SEEDS=(0 1 2)
SUPPORT_SIZES="1 5 10 50 256"
NUM_GPUS=8

mkdir -p logs

i=0
for task in "${TASKS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        gpu=$(( i % NUM_GPUS ))
        log="logs/supportsize_${task}_s${seed}.log"
        echo "  GPU${gpu}: support-size sweep ${task} seed ${seed} -> ${log}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
            python3 experiments/run_revision.py \
                --tasks "${task}" \
                --seeds "${seed}" \
                --methods FTN \
                --ablation support_size \
                --support-sizes ${SUPPORT_SIZES} \
                --results-suffix "_support_size_s${seed}" \
                > "${log}" 2>&1 &
        i=$(( i + 1 ))
        sleep 0.5
    done
done

echo "Started ${i} jobs."
wait
echo "All support-size jobs complete."
