#!/bin/bash
# Progressive serving with union/usage-weighted coloring. As each training block
# checkpoints (starting at >=2 domains), snapshot the checkpoint to
# checkpoints_serve/, rebuild refs_usage.pt (union over reconfig outputs) against
# that snapshot, then (re)launch the app on it. The app serves from the snapshot
# (FTN_CKPT_DIR) so train.py can keep overwriting checkpoints/ for later blocks.
#   training -> GPUs 0-3 ; refs_usage rebuild -> GPUs 6,7 ; app -> GPUs 4,5
set -u
cd "$(dirname "$0")"
CKPT=checkpoints
SERVE=checkpoints_serve
APP_GPUS=4,5
REBUILD_GPUS=6,7
LOGW=serve_progressive.log
launched_for=0

ntrained() {
  python -c "import json;print(len(json.load(open('$CKPT/state.json'))['trained']))" 2>/dev/null || echo 0
}

relaunch() {
  local n="$1"
  echo "[$(date +%H:%M:%S)] block count=$n -> snapshot" >> "$LOGW"
  rm -rf "$SERVE"; mkdir -p "$SERVE"
  # state.json is written last by save_state, so when n increments bank.pt is
  # complete. Copy serving files; refs_usage is rebuilt below.
  cp "$CKPT/bank.pt" "$CKPT/refs.pt" "$CKPT/replay.pt" "$CKPT/state.json" "$SERVE"/
  echo "[$(date +%H:%M:%S)] rebuilding refs_usage on gpus $REBUILD_GPUS ..." >> "$LOGW"
  if FTN_CKPT_DIR="$PWD/$SERVE" CUDA_VISIBLE_DEVICES="$REBUILD_GPUS" \
       uv run python rebuild_refs_usage.py >> rebuild_serve.log 2>&1; then
    echo "[$(date +%H:%M:%S)] refs_usage rebuilt -> union coloring" >> "$LOGW"
  else
    echo "[$(date +%H:%M:%S)] refs_usage rebuild FAILED; app falls back to refs.pt" >> "$LOGW"
  fi
  pkill -f "python.*app.py" 2>/dev/null && sleep 5
  FTN_CKPT_DIR="$PWD/$SERVE" CUDA_VISIBLE_DEVICES="$APP_GPUS" \
    nohup uv run python app.py > app_serve.log 2>&1 &
  echo "[$(date +%H:%M:%S)] app relaunched for $n domains (gpus $APP_GPUS)" >> "$LOGW"
}

echo "[$(date +%H:%M:%S)] watcher started" >> "$LOGW"
while true; do
  n="$(ntrained)"
  if [ "$n" -ge 2 ] && [ "$n" -gt "$launched_for" ]; then
    relaunch "$n"; launched_for="$n"
  fi
  [ "$n" -ge 4 ] && { echo "[$(date +%H:%M:%S)] all 4 served; done" >> "$LOGW"; break; }
  if ! pgrep -f "train.py --fresh" >/dev/null; then
    n="$(ntrained)"
    if [ "$n" -ge 2 ] && [ "$n" -gt "$launched_for" ]; then relaunch "$n"; fi
    echo "[$(date +%H:%M:%S)] training gone; watcher exiting (n=$n)" >> "$LOGW"
    break
  fi
  sleep 30
done
