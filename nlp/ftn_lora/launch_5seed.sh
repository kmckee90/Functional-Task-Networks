#!/usr/bin/env bash
# Launch 5-seed sweep + FTN_RandMask ablation across GPUs 1-7 (skip GPU 0 — in use).
set -e
cd /data/users/kevin/misc_experiments/experiments/func_net_kt
mkdir -p logs/lora results/lora

run() {
  local gpu=$1; shift
  local method=$1; shift
  local seed=$1; shift
  local b_init=$1; shift
  local suffix=$1; shift

  local log="logs/lora/${method}_s${seed}${suffix}.log"
  CUDA_VISIBLE_DEVICES=$gpu \
    uv run python -m ftn_lora.run_experiment \
      --model "Qwen/Qwen2.5-1.5B-Instruct" \
      --methods "$method" --seeds "$seed" \
      --train-steps-per-task 100 --train-size 400 --eval-size 100 \
      --support-size 16 --batch-size 4 --grad-accum 1 --lr 2e-4 \
      --max-len 384 --head-start-eval-size 64 \
      --cfg-inner-steps 3 --cfg-lr 0.5 \
      --lora-H 128 --lora-k 16 --lora-rank 4 --lora-alpha 16 --lora-dropout 0.05 \
      --lora-b-init "$b_init" \
      --device cuda:0 \
      > "$log" 2>&1
}

# Phase 1 (parallel): FTN bnonzero seeds 2-4 on GPUs 1-3, FTN_RandMask bnonzero seeds 0-3 on GPUs 4-7
run 1 FTN 2 small "" &
run 2 FTN 3 small "" &
run 3 FTN 4 small "" &
run 4 FTN_RandMask 0 small "" &
run 5 FTN_RandMask 1 small "" &
run 6 FTN_RandMask 2 small "" &
run 7 FTN_RandMask 3 small "" &
wait

echo "Phase 1 done"

# Phase 2: FTN_RandMask s4 plus non-FTN seeds 2-4 across freed GPUs.
# FTN_RandMask s4 on GPU 1
run 1 FTN_RandMask 4 small "" &

# Non-FTN baselines seeds 2-4 distributed across GPUs 2-7 (B0=0, the standard init for these methods)
for seed in 2 3 4; do
  gpu=$((seed + 1))   # 3,4,5
  run $gpu SeqLoRA $seed zero "" &
done
for seed in 2 3 4; do
  gpu=$((seed + 4))   # 6,7,8 -> wrap to 6,7,2
  if [[ $gpu -gt 7 ]]; then gpu=$((gpu - 6)); fi
  run $gpu SeqLoRA_ER $seed zero "" &
done
wait
echo "Phase 2 done"

# Phase 3: FixedMask family seeds 2-4
for seed in 2 3 4; do
  gpu=$((seed))
  run $gpu FixedMask $seed zero "" &
done
for seed in 2 3 4; do
  gpu=$((seed + 3))
  run $gpu FixedMask_router $seed zero "" &
done
wait
echo "Phase 3 done"

# Phase 4: RandMask_router seeds 2-4
for seed in 2 3 4; do
  gpu=$((seed))
  run $gpu RandMask_router $seed zero "" &
done
wait
echo "Phase 4 done. All runs complete."
