#!/usr/bin/env bash
# One-time (pod side): fetch LoCoMo + the audit's corrected answer key onto the
# PVC, outside the synced tree (design.md §7).
set -euo pipefail

ROOT="${FTN_AGENT_STATE:-/home/dev/ftn_agent_state}/benchmarking"
mkdir -p "$ROOT"
cd "$ROOT"

if [ ! -d locomo ]; then
  git clone --depth 1 https://github.com/snap-research/locomo.git
fi
if [ ! -d locomo-audit ]; then
  git clone --depth 1 https://github.com/dial481/locomo-audit.git || \
    echo "WARN: locomo-audit clone failed — will fall back to original answer key"
fi

python - <<'PY'
import json, os
root = os.path.join(os.environ.get("FTN_AGENT_STATE", "/home/dev/ftn_agent_state"), "benchmarking")
p = os.path.join(root, "locomo", "data", "locomo10.json")
data = json.load(open(p))
n_qa = sum(len(s["qa"]) for s in data)
cats = {}
for s in data:
    for q in s["qa"]:
        cats[q.get("category")] = cats.get(q.get("category"), 0) + 1
print(f"locomo10: {len(data)} conversations, {n_qa} questions, categories {cats}")
PY
echo "setup_data: done -> $ROOT"
