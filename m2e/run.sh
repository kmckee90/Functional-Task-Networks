#!/usr/bin/env bash
# Run on the pod with the retained environment's Python.
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$source_dir${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$source_dir"
exec "${PYTHON:-python}" "$@"
