#!/usr/bin/env bash
# Run on the pod with vizdoom installed in the selected Python environment.
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$experiment_dir${PYTHONPATH:+:$PYTHONPATH}"
cd "$experiment_dir"
exec "${PYTHON:-python}" "$@"
