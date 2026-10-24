#!/usr/bin/env bash
# Run on the pod. Fresh Phase-4-style conv FTN-M2E with an actual live MJPEG view.
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export GLOBO_STATE=${GLOBO_STATE:-/home/dev/globo_dreamer_state}
name=${1:-live_ftnm2e_$(date +%Y%m%d_%H%M%S)}
[[ "$name" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid run name' >&2; exit 1; }
if [[ "${2:-}" == --resume ]]; then
  [[ -d "$GLOBO_STATE/runs/$name/ckpt" ]] || { echo 'Resume requires an existing checkpoint' >&2; exit 1; }
else
  [[ ! -e "$GLOBO_STATE/runs/$name" ]] || { echo 'Fresh run refused: directory already exists (use --resume)' >&2; exit 1; }
fi
export PYTHONPATH="$(dirname "$source_dir"):$GLOBO_STATE/packages"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4 LP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
export GLOBO_MF_REPLAY_DTYPE=float16 GLOBO_SOFTWARE_RENDERER=1 GLOBO_VISITATION=1
export GLOBO_FRONTIER_ROTATION=0
export GLOBO_ENGINE_ROOT="$GLOBO_STATE/live_engines/$name"
exec "$GLOBO_STATE/venv/bin/python" -u -m globo_dreamer.train \
  --name "$name" --seed 1901 --world-seed 19 --spawn 12.0,39.1,120.0 --steps 0 \
  --model-free --model-free-only --m2e-ftn conv --model-feedback \
  --render-size 256 --reservoir-channels 128 --reservoir-variant tuned_v1 \
  --mf-replay-capacity 2000000 --mf-goal-capacity 500000 --recode-capacity 100000 \
  --mf-gamma 0.9036665304295952 --mf-epsilon-end 0.015366900171141888 \
  --mf-online-coef 0.8728610896464566 --mf-frontier-scale 4.574998390699497 \
  --mf-lr 0.0001899492997635358 --mf-offline-coef 1 --dreamer-action-probability 0 \
  --achievements --continuous-monitor --checkpoint-seconds 600 --live-port "${LIVE_PORT:-8000}"
