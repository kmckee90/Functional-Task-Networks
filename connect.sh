#!/usr/bin/env bash
# Repository-wide source sync; runtime data and the old venv stay outside it.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec actl pod up functional-task-networks -n obelisk --profile cpu-small --remote-path /home/dev/repo_source "$@"
