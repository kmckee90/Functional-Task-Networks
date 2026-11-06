#!/usr/bin/env bash
# Source on an actl benchmark pod. Bulk storage is never below workspace.
export BM_ROOT=/mnt/obelisk-shared/benchmarking/kmckee90/neurips_ftn_2026
export BM_DATA="$BM_ROOT/data"
export BM_CHECKPOINTS="$BM_ROOT/checkpoints"
export BM_RUNS="$BM_ROOT/runs"
export BM_CACHE="$BM_ROOT/cache"
export HF_HOME="$BM_CACHE/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$BM_CACHE/torch"
export PIP_CACHE_DIR="$BM_CACHE/pip"
export UV_CACHE_DIR="$BM_CACHE/uv"
export XDG_CACHE_HOME="$BM_CACHE/xdg"
export WANDB_DIR="$BM_ROOT/wandb"
export MPLCONFIGDIR="$BM_CACHE/matplotlib/${HOSTNAME:-unknown}"
# UNIX-domain sockets used by DataLoader need a short pathname (<108 bytes).
# Socket scratch uses node-local storage to avoid NFS socket-unlink errors.
export TMPDIR=/tmp/bm
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX="$BM_CACHE/pycache/${HOSTNAME:-unknown}"
export TRITON_CACHE_DIR="$BM_CACHE/triton/${HOSTNAME:-unknown}"
export TORCHINDUCTOR_CACHE_DIR="$BM_CACHE/torchinductor/${HOSTNAME:-unknown}"

if [[ ! -d /mnt/obelisk-shared ]]; then
    echo 'Shared benchmark PVC is not mounted; refusing local storage fallback.' >&2
    return 1 2>/dev/null || exit 1
fi
mkdir -p "$BM_DATA" "$BM_CHECKPOINTS" "$BM_RUNS" "$BM_CACHE" \
    "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$UV_CACHE_DIR" \
    "$XDG_CACHE_HOME" "$WANDB_DIR" "$MPLCONFIGDIR" "$TMPDIR" \
    "$PYTHONPYCACHEPREFIX" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
