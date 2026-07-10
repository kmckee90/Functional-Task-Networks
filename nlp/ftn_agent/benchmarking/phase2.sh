#!/usr/bin/env bash
# Phase 2 chain: all 10 LoCoMo conversations sequentially into ONE shared bank
# per arm — p2-ftn first (synthesizes; cache shared), then p2-mono (single-LoRA
# ablation on the identical cached data). The FTN-vs-mono separation, retention,
# and routing-under-load story lives here.
#
# After each conversation: immediate-recall eval on the just-ingested conv
# (results/p2-*.imm.<conv>.jsonl) + retention probe on the first-ingested conv
# (results/p2-*.ret.after-<conv>.jsonl). Final full 1540-question eval per arm
# (results/p2-*.jsonl). Immediate-vs-final delta per conv = forgetting measure;
# the .ret files give the first-conv retention trajectory.
#
# Crash-resilient like phase1.sh: ingest resumes via state.json, evals resume
# via their output files, so every stage sits in a bounded retry loop.
set -u
cd /home/dev/workspace/benchmarking
LOG_DIR="${FTN_AGENT_STATE:-/home/dev/ftn_agent_state}/benchmarking"
rm -f "$LOG_DIR/phase2.done"

CONVS=(conv-26 conv-30 conv-41 conv-42 conv-43 conv-44 conv-47 conv-48 conv-49 conv-50)
FIRST=conv-26

retry() {
    local n=0
    until "$@"; do
        n=$((n + 1))
        echo "[chain] RETRY $n after crash: $*"
        [ "$n" -ge 10 ] && { echo "[chain] giving up on: $*"; return 1; }
        sleep 15
    done
}

arm() {
    local run=$1; shift
    local conv
    for conv in "${CONVS[@]}"; do
        retry python ingest.py --run "$run" --convs "$conv" ${1+"$@"} || return 1
        retry python eval_ftn.py --run "$run" --convs "$conv" \
            --out "results/$run.imm.$conv.jsonl" || return 1
        if [ "$conv" != "$FIRST" ]; then
            retry python eval_ftn.py --run "$run" --convs "$FIRST" \
                --out "results/$run.ret.after-$conv.jsonl" || return 1
        fi
        echo "[chain] $run: $conv ingested + evaluated"
    done
    retry python eval_ftn.py --run "$run" --out "results/$run.jsonl"
}

arm p2-ftn
echo "[chain] p2-ftn finished (exit $?)"
arm p2-mono --monolithic
echo "[chain] p2-mono finished (exit $?)"
touch "$LOG_DIR/phase2.done"
echo "[chain] all done."
