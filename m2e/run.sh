#!/usr/bin/env bash
# Run on the pod with the retained environment's Python.
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
recode_env=${M2E_RECODE_ENV:-$HOME/.venvs/m2e-recode}
if [[ -z "${PYTHON:-}" && -x "$recode_env/bin/python" ]]; then
    PYTHON="$recode_env/bin/python"
fi
export MPLBACKEND=${MPLBACKEND:-Agg}
export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$source_dir"
exec "${PYTHON:-python}" "$@"
