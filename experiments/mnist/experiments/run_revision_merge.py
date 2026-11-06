"""
run_revision_merge.py — merge per-seed revision shards into one JSON per task.

The parallel launcher writes results/revision_<task>_s<seed>.json shards;
this script collects them into results/revision_<task>.json with a single
runs dict and a fresh aggregate.
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_revision import aggregate, TASK_REGISTRY  # type: ignore


def main():
    for task in TASK_REGISTRY.keys():
        # Pick up the original per-seed shards (revision_<task>_s<seed>.json)
        # plus any FTN-RandMask ablation shards
        # (revision_<task>_randmask_s<seed>.json) so they merge into the
        # canonical results/revision_<task>.json that the table generator
        # reads.
        shards = sorted(
            glob.glob(f"results/revision_{task}_s*.json")
            + glob.glob(f"results/revision_{task}_randmask_s*.json")
        )
        if not shards:
            continue
        merged_runs: dict = {}
        for shard in shards:
            with open(shard) as f:
                d = json.load(f)
            for k, v in d.get("runs", {}).items():
                merged_runs[k] = v
        out = {"runs": merged_runs}
        aggregate(out)
        out_path = f"results/revision_{task}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  merged {len(shards):>2} shards -> {out_path} "
              f"({len(merged_runs)} total runs)")


if __name__ == "__main__":
    main()
