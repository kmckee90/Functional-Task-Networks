#!/usr/bin/env bash
# Run ON THE POD after sync. Repoint only the Python package source alias.
set -euo pipefail
source_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
state_root=${GLOBO_STATE:-/home/dev/globo_dreamer_state}
test -f "$source_root/craftium/game/game.conf"
mkdir -p "$state_root/packages"
ln -sfn "$source_root/experiments/globo_dreamer" "$state_root/packages/globo_dreamer"
printf 'Source ready. Use PYTHONPATH=%s/packages; runtime data unchanged.\n' "$state_root"
