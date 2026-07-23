#!/usr/bin/env bash
# Interim 3-domain bank for the applet while globo synthesis runs.
set -u
cd "$(dirname "$0")"
echo "[interim] waiting for rl_mt10 synthesis to save..."
until grep -q "distillation triples -> .*rl_mt10.json" qa_synth.log; do sleep 30; done
echo "[interim] training base_ftn explorer rl_mt10 -> checkpoints_interim"
FTN_CKPT_DIR=$HOME/ftn_chat_state/checkpoints_interim uv run python train.py --fresh --tasks base_ftn explorer rl_mt10 2>&1 | tee train_interim.log
echo "[interim] DONE"
