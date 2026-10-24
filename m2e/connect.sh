#!/usr/bin/env bash
set -euo pipefail
# Connect this source tree to the dedicated m2e workspace.
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec actl pod up m2e --local-path "$source_dir" "$@"
