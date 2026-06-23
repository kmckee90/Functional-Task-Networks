#!/bin/bash
# Launches exp2..exp5 sequentially, each with the same worker pool.
# Usage: CUDA_VISIBLE_DEVICES=3,4,5,6,7 bash experiments/run_all_exps.sh
set -e
cd "$(dirname "$0")/.."
mkdir -p logs
for exp in exp3 exp4 exp5; do
  echo "[run_all_exps] starting $exp"
  uv run python experiments/run_kt_exp.py --exp $exp --workers-per-gpu 4 > logs/$exp.log 2>&1
  echo "[run_all_exps] $exp done"
done
echo "[run_all_exps] all sweeps done."
