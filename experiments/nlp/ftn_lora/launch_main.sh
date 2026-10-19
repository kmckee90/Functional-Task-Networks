#!/usr/bin/env bash
# Launch FTN-LoRA main experiment: 8 methods, parallel across GPUs.
set -e

cd /data/users/kevin/misc_experiments/experiments/func_net_kt
mkdir -p logs/lora results/lora

METHODS=(FTN FTN_router FTN_noreplay SeqLoRA SeqLoRA_ER FixedMask FixedMask_router RandMask_router)
SEED="${1:-0}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TRAIN_STEPS="${TRAIN_STEPS:-120}"
TRAIN_SIZE="${TRAIN_SIZE:-400}"
H_VAL="${H_VAL:-128}"
K_VAL="${K_VAL:-16}"

echo "Launching seed=${SEED} model=${MODEL} steps=${TRAIN_STEPS} H=${H_VAL} k=${K_VAL}"

for i in "${!METHODS[@]}"; do
  GPU=${i}
  METHOD=${METHODS[$i]}
  LOG="logs/lora/${METHOD}_s${SEED}.log"
  echo "  GPU $GPU: $METHOD -> $LOG"
  CUDA_VISIBLE_DEVICES=$GPU \
    uv run python -m ftn_lora.run_experiment \
      --model "$MODEL" \
      --methods "$METHOD" \
      --seeds "$SEED" \
      --train-steps-per-task "$TRAIN_STEPS" \
      --train-size "$TRAIN_SIZE" \
      --eval-size 100 \
      --support-size 16 \
      --batch-size 4 \
      --grad-accum 1 \
      --lr 2e-4 \
      --max-len 384 \
      --head-start-eval-size 64 \
      --cfg-inner-steps 3 \
      --cfg-lr 0.5 \
      --lora-H "$H_VAL" \
      --lora-k "$K_VAL" \
      --lora-rank 4 \
      --lora-alpha 16 \
      --lora-dropout 0.05 \
      --device cuda:0 \
      > "$LOG" 2>&1 &
done

wait
echo "All $((${#METHODS[@]})) jobs complete."
