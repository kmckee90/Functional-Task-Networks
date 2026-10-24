#!/usr/bin/env bash
# Sync the shared game and experiments together; preserve existing pod storage.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
project=${GLOBO_PROJECT:-globo1}
# A single sync root has one actl defaults record. Keep per-pod GPU sizes explicit
# so temporary CPU migration sessions do not change the next training allocation.
case "$project" in
  globo-dreamer|globo_dreamer) project=globo-dreamer; profile=8x ;;
  globo3|globo-actions) profile=4x ;;
  globo1|globo2|globo4) profile=2x ;;
  *) printf 'Unknown Globo workspace: %s\n' "$project" >&2; exit 1 ;;
esac
exec actl pod up "$project" -n obelisk --profile "$profile" --sync --remote-path /home/dev/globo_source "$@"
