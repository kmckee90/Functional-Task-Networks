#!/usr/bin/env bash
# ON POD ONLY. Independent launcher and state; never resumes/prunes search40k.
set -uo pipefail
STATE=${GLOBO_STATE:-/home/dev/globo_dreamer_state}
export GLOBO_STATE="$STATE" PYTHONPATH="$STATE/packages" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 LP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$STATE/search60k"
cd "$STATE"
echo $$ > "$STATE/search60k/launcher.pid"
child=
stop() { if [ -n "$child" ]; then kill -TERM "$child" 2>/dev/null || true; wait "$child" || true; fi; exit 0; }
trap stop TERM INT
while true; do
  "$STATE/venv/bin/python" -u -m globo_dreamer.search60 "$@" &
  child=$!
  wait "$child"
  result=$?
  child=
  if [ "$result" -eq 0 ]; then exit 0; fi
  echo "Supervisor exited $result; recovering durable state in 30 seconds"
  sleep 30
done
