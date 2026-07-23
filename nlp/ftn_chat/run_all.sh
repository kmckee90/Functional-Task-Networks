#!/usr/bin/env bash
# Full v2 pipeline chain, run on the pod under nohup:
#     nohup bash run_all.sh > run_all.log 2>&1 &
# Waits for a live qa_synth.py to finish (if any), trains all registered
# tasks from scratch, then runs the routing crux gate. Touches run_all.done
# when finished so completion is visible at a glance.
set -u
cd "$(dirname "$0")"

echo "[chain] waiting for qa_synth.py (if running)..."
while pgrep -f "python3? qa_synth.py" >/dev/null 2>&1; do sleep 60; done

for t in base_ftn explorer globo rl_mt10; do
  if [ ! -s "data_cache/qa/$t.json" ]; then
    echo "[chain] ABORT: data_cache/qa/$t.json missing or empty — synthesis incomplete"
    exit 1
  fi
done

echo "[chain] training all tasks --fresh ..."
uv run python train.py --fresh 2>&1 | tee train_full.log
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "[chain] ABORT: train.py exited $rc"
  exit "$rc"
fi

echo "[chain] routing check ..."
uv run python routing_check.py 2>&1 | tee routing_check.log

touch run_all.done
echo "[chain] DONE"
