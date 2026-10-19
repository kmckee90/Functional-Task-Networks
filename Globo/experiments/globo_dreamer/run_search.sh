#!/usr/bin/env bash
# Start ON POD with nohup; survives client disconnect and restarts on supervisor errors.
set -uo pipefail
STATE=${GLOBO_STATE:-/home/dev/globo_dreamer_state}
export GLOBO_STATE="$STATE" PYTHONPATH="$STATE/packages" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 LP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$STATE/search40k"
cd "$STATE"
echo $$ > "$STATE/search40k/launcher.pid"
child=
stop() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; fi; exit 0; }
trap stop TERM INT
while true; do
  "$STATE/venv/bin/python" -u -m globo_dreamer.search "$@" &
  child=$!
  wait "$child"
  result=$?
  child=
  if [ "$result" -eq 0 ]; then exit 0; fi
  echo "Supervisor exited $result; recovering durable state in 30 seconds"
  sleep 30
done
