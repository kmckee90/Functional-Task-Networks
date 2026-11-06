#!/usr/bin/env bash
set -euo pipefail
source /home/dev/workspace/remote_env.sh
bm_pod="${1:?Usage: start_workers.sh bm1|bm2|bm3}"
bm_suite="${BM_SUITE:-default}"
mkdir -p "$BM_RUNS/workers"
shift
if [[ "$#" -eq 0 ]]; then set -- 0 1 2 3; fi
for bm_gpu in "$@"; do
    bm_pidfile="$BM_RUNS/workers/${bm_pod}_gpu${bm_gpu}.pid"
    if [[ -f "$bm_pidfile" ]] && kill -0 "$(cat "$bm_pidfile")" 2>/dev/null; then
        echo "$bm_pod GPU $bm_gpu already has a worker"
        continue
    fi
    nohup bash /home/dev/workspace/run_remote.sh worker.py --pod "$bm_pod" --gpu "$bm_gpu" --suite "$bm_suite" \
        > "$BM_RUNS/workers/${bm_pod}_gpu${bm_gpu}.log" 2>&1 < /dev/null &
    echo "$!" > "$bm_pidfile"
    echo "$bm_pod GPU $bm_gpu worker PID $!"
done
