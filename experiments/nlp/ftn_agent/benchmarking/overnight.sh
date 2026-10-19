#!/usr/bin/env bash
# Overnight chain (pod side, nohup-safe). Waits for the baselines to finish,
# smoke-tests the ingestion pipeline, then runs the Phase 1 head-to-head on
# conv-26: FTN arm -> closed-book eval -> monolithic ablation -> eval.
# Everything resumes if re-run; logs to $FTN_AGENT_STATE/benchmarking/overnight.log.
set -uo pipefail
cd /home/dev/workspace/benchmarking
LOG_DIR="${FTN_AGENT_STATE:-/home/dev/ftn_agent_state}/benchmarking"
rm -f "$LOG_DIR/chain.done"

# Wait until no process holds the GPUs (robust against pgrep matching our own
# sibling watcher command lines — that bit us once).
echo "[chain] waiting for GPUs to be free..."
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; do sleep 60; done
echo "[chain] GPUs free. starting smoke."

# 1. Smoke: tiny ingest (2 sessions, few items) to validate the GPU pipeline.
#    Scratch WORK dir so its tiny datasets never pollute the shared synthesis
#    cache the real runs reuse.
FTN_AGENT_WORK=/home/dev/ftn_agent_work_smoke \
python ingest.py --run smoke --convs conv-26 --max-sessions 2 --n-items 12 \
    --epochs 1 --ckpt-every 1 --heldout-every 1
if [ $? -ne 0 ]; then
    echo "[chain] SMOKE FAILED — stopping so nothing burns GPU overnight."
    touch "$LOG_DIR/chain.done"
    exit 1
fi
echo "[chain] smoke passed. launching Phase 1 FTN arm (conv-26)."

# 2. Phase 1 FTN arm: full conv-26 (19 sessions), then closed-book eval.
python ingest.py --run p1-ftn --convs conv-26 && \
python eval_ftn.py --run p1-ftn
echo "[chain] p1-ftn done (exit $?). launching monolithic ablation."

# 3. Single-LoRA ablation on IDENTICAL cached data, then eval.
python ingest.py --run p1-mono --convs conv-26 --monolithic && \
python eval_ftn.py --run p1-mono
echo "[chain] all done."
touch "$LOG_DIR/chain.done"
