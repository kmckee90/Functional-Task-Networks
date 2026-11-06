#!/usr/bin/env bash
set -euo pipefail
source /home/dev/workspace/remote_env.sh
# Make interactive benchmark shells inherit the remote-only cache paths too.
bm_env_line='[[ ! -f /home/dev/workspace/remote_env.sh ]] || source /home/dev/workspace/remote_env.sh'
if [[ ! -f "$HOME/.bashrc" ]] || ! grep -Fqx "$bm_env_line" "$HOME/.bashrc"; then
    printf '\n%s\n' "$bm_env_line" >> "$HOME/.bashrc"
fi
printf 'Bulk storage: %s\n' "$BM_ROOT"
df -h "$BM_ROOT"
printf '\nGPU inventory:\n'
nvidia-smi --query-gpu=index,name,memory.total --format=csv
printf '\nFramework:\n'
python -c 'import torch; print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPUs", torch.cuda.device_count())'
