#!/usr/bin/env bash
set -euo pipefail
source /home/dev/workspace/remote_env.sh
until [[ -f "$BM_ROOT/manifests/environment.txt" ]]; do sleep 5; done
exec bash /home/dev/workspace/run_remote.sh prepare_data.py "$@"
