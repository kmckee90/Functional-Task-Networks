#!/usr/bin/env bash
# Run locally; explicit destination prevents syncing into the old source layout.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec actl pod up sit -n obelisk --sync --remote-path /home/dev/experiment "$@"
