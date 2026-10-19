#!/usr/bin/env bash
#
# setup_pod.sh — provision an actl pod to run globo_explorer_gpu.
#
# This project drives the warpcraft GPU globo engine (Warp/torch on CUDA) at
# num_worlds=1 for a single continuous lifetime, with the reservoir + branching
# DQN + RECODE all in JAX/Flax on-device. Requirements beyond `uv sync`:
#
#   1. git safe.directory — uv builds `warpcraft @ git+...` in a cache on the NFS
#      home PVC (owned by `nobody`); git-as-root rejects it as "dubious ownership"
#      unless every dir is trusted. One-time global git config.
#   2. A CUDA GPU — warpcraft exec_mode=1 hard-requires it; Warp JIT-compiles its
#      kernels on first run (slow once, then cached).
#
# Usage (on the pod, from this directory):
#   bash setup_pod.sh
#
set -euo pipefail
cd "$(dirname "$0")"

# The repo-root setup_pod.sh unsets UV_PYTHON; do it here too for standalone runs.
unset UV_PYTHON || true

# 1. Trust uv's git cache on the NFS PVC.
git config --global --add safe.directory '*'

# 2. Resolve deps (CUDA torch cu128 + warp-lang + jax[cuda12] + flax/optax + warpcraft).
uv sync

# 3. Prewarm Warp kernels so the first training/viewer launch isn't stalled by JIT.
uv run python -c "import warp as wp; wp.init(); print('[setup_pod] warp', wp.config.version, 'cuda:', wp.is_cuda_available())" || true

echo
echo "[setup_pod] done. Try:  uv run python train.py --config default.yaml"
echo "            Human viewer:  uv run python play.py"
