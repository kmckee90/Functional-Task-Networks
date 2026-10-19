#!/usr/bin/env bash
# Run on the pod. Keep upstream SiT unchanged and make both packages importable.
set -euo pipefail
experiment_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
sit_upstream="${SIT_UPSTREAM:-$experiment_dir/sit}"
if [[ ! -f "$sit_upstream/models.py" ]]; then
    echo "Missing upstream SiT models.py; set SIT_UPSTREAM to the clean SiT checkout." >&2
    exit 1
fi
export PYTHONPATH="$experiment_dir:$sit_upstream${PYTHONPATH:+:$PYTHONPATH}"
cd "$experiment_dir"
exec "${PYTHON:-python}" "$@"
