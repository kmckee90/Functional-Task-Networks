#!/usr/bin/env bash
set -uo pipefail
STATE=${GLOBO_STATE:-/home/dev/globo_dreamer_state}
export GLOBO_STATE="$STATE" PYTHONPATH="$STATE/packages" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$STATE/model_study"
cd "$STATE"
echo $$ > "$STATE/model_study/launcher.pid"
child=
stop() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; fi; exit 0; }
trap stop TERM INT
while true; do
  "$STATE/venv/bin/python" -u -m globo_dreamer.model_study &
  child=$!
  wait "$child"
  result=$?
  child=
  if [ "$result" -eq 0 ]; then exit 0; fi
  echo "Model-study supervisor exited $result; restarting in 30 seconds"
  sleep 30
done
