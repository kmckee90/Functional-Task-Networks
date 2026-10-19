#!/usr/bin/env bash
# Phase 1 chain, crash-resilient: every stage resumes from disk, so each is
# wrapped in a bounded retry loop (CUDA illegal-memory-access kills the
# process; state survives). Smoke already passed — goes straight to the runs.
set -u
cd /home/dev/workspace/benchmarking
LOG_DIR="${FTN_AGENT_STATE:-/home/dev/ftn_agent_state}/benchmarking"
rm -f "$LOG_DIR/chain.done"

retry() {
    local n=0
    until "$@"; do
        n=$((n + 1))
        echo "[chain] RETRY $n after crash: $*"
        [ "$n" -ge 10 ] && { echo "[chain] giving up on: $*"; return 1; }
        sleep 15
    done
}

retry python ingest.py --run p1-ftn --convs conv-26 --ckpt-every 2 && \
retry python eval_ftn.py --run p1-ftn
echo "[chain] p1-ftn finished (exit $?)"

retry python ingest.py --run p1-mono --convs conv-26 --monolithic --ckpt-every 2 && \
retry python eval_ftn.py --run p1-mono
echo "[chain] p1-mono finished (exit $?)"

retry python run_baselines.py --mode ceiling
echo "[chain] ceiling finished (exit $?)"
touch "$LOG_DIR/chain.done"
echo "[chain] all done."
