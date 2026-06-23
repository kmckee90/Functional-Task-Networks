"""
merge_baselines.py — fold `revision_<task>_baselines_s<seed>.json` shards
into the already-merged `revision_<task>.json` and re-aggregate.
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.run_revision import TASK_REGISTRY, aggregate  # type: ignore


def main() -> None:
    for task in TASK_REGISTRY.keys():
        merged_path = f"results/revision_{task}.json"
        shards = sorted(glob.glob(f"results/revision_{task}_baselines_s*.json"))
        if not shards:
            continue
        if os.path.exists(merged_path):
            with open(merged_path) as f:
                merged = json.load(f)
        else:
            merged = {"runs": {}}
        merged_runs = merged.setdefault("runs", {})
        added = 0
        for shard in shards:
            with open(shard) as f:
                d = json.load(f)
            for k, v in d.get("runs", {}).items():
                if k not in merged_runs:
                    added += 1
                merged_runs[k] = v
        aggregate(merged)
        with open(merged_path, "w") as f:
            json.dump(merged, f, indent=2)
        print(
            f"  {task}: merged {len(shards):>2} baseline shards -> {merged_path} "
            f"(+{added} new runs, total {len(merged_runs)})"
        )


if __name__ == "__main__":
    main()
