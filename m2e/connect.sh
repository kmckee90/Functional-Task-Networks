#!/usr/bin/env bash
set -euo pipefail
# M2E shares the repository-wide workspace; never start a competing narrow sync.
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/../connect.sh" "$@"
