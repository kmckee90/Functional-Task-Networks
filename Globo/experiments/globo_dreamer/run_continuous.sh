#!/usr/bin/env bash
set -uo pipefail
STATE=${GLOBO_STATE:-/home/dev/globo_dreamer_state}
export GLOBO_STATE="$STATE" PYTHONPATH="$STATE/packages" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$STATE/continuous_runs"
cd "$STATE"
echo $$ > "$STATE/continuous_runs/launcher.pid"
child=
stop() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; fi; exit 0; }
trap stop TERM INT
while true; do
  "$STATE/venv/bin/python" -u -m globo_dreamer.continuous_runs &
  child=$!
  wait "$child"
  result=$?
  child=
  if [ "$result" -eq 0 ]; then exit 0; fi
  sleep 30
done
