#!/usr/bin/env bash
# Run on the pod. Keep both qualified experiment imports and legacy modules usable.
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="$(dirname "$source_dir"):$source_dir${PYTHONPATH:+:$PYTHONPATH}"
cd -- "$(dirname "$source_dir")"
exec "${PYTHON:-python}" "$@"
