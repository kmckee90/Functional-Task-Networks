#!/usr/bin/env bash
set -euo pipefail
source /home/dev/workspace/remote_env.sh
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export WANDB_MODE=disabled
cd /home/dev/workspace
exec "$BM_ROOT/envs/main/bin/python" -u "$@"
