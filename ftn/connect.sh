#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
exec actl pod up base -n obelisk --profile 2x --remote-path /home/dev/ftn_source/ftn "$@"
