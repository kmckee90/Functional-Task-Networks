#!/usr/bin/env bash
# Run inside the m2e pod. Keep installed packages on the persistent home volume.
# Setup: bash setup_recode.sh
# Train: CUDA_VISIBLE_DEVICES=0 uv run python experiment_recode.py
# Or activate: source "$HOME/.venvs/m2e-recode/bin/activate"
# --log uses the experiment's Databricks URI and needs Databricks credentials.
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
recode_env=${M2E_RECODE_ENV:-$HOME/.venvs/m2e-recode}
base_python=${PYTHON:-python}
if [[ ! -x "$recode_env/bin/python" ]]; then
    "$base_python" -m venv --system-site-packages "$recode_env"
fi
# The image's Python may itself be a venv (/app/.venv). Python's venv module
# inherits system packages, but does not inherit a parent venv's packages.
base_packages=$("$base_python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
env_packages=$("$recode_env/bin/python" -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
if [[ "$base_packages" != "$env_packages" ]]; then
    printf '%s\n' "$base_packages" > "$env_packages/m2e-pod-base.pth"
fi
cd -- "$source_dir"
if [[ ! -e .venv && ! -L .venv ]]; then
    ln -s "$recode_env" .venv
fi
uv sync --inexact
uv pip check --python .venv/bin/python
MPLBACKEND=Agg SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy \
    uv run python experiment_recode.py --help
printf '\nEnvironment ready: %s\nRun: CUDA_VISIBLE_DEVICES=0 uv run python experiment_recode.py\n' "$recode_env"
