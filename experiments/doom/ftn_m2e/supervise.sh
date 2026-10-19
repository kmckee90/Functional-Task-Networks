#!/usr/bin/env bash
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$experiment_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$experiment_dir"
gpu="$1"
seed="$2"
runtime="${DOOM_RUNTIME:-/home/dev/vizdoom_state}"
mkdir -p "$runtime/run$gpu"
export CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTORCH_KERNEL_CACHE_PATH="$runtime/torch-kernels"
mkdir -p "$PYTORCH_KERNEL_CACHE_PATH"
while [[ ! -e "$runtime/STOP" ]]; do
    "$runtime/venv/bin/python" -u -m ftn_m2e.train --root "$runtime/run$gpu" --seed "$seed" >> "$runtime/run$gpu/runner.log" 2>&1 && break
    sleep 30
done
