"""Snapshot the task source files from the functional_task_networks repo into
corpus/ so the project is self-contained (the actl pod syncs only nlp/ftn_chat,
so the surrounding repo is NOT available at runtime).

Run LOCALLY (plain stdlib, no torch) whenever the upstream sources change:

    python3 make_corpus.py

then retrain the affected block(s) on the pod. Globs are copied flat except
`globo`, which preserves the mods/ subtree so mod boundaries stay visible.
"""
from __future__ import annotations

import glob as _glob
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # functional_task_networks/ (above experiments/nlp/)
CORPUS = HERE / "corpus"

# task key -> list of (upstream glob, preserve_subtree_root or None)
SOURCES = {
    "base_ftn": [
        (REPO / "ftn/ftn.py", None),
        (REPO / "ftn/config.py", None),
    ],
    "explorer": [
        (REPO / "m2e/agents/agent_recode.py", None),
        (REPO / "m2e/recode/recode.py", None),
        (REPO / "m2e/recode/density_estimator.py", None),
        (REPO / "m2e/recode/observation_buffer.py", None),
        (REPO / "m2e/assets/reservoir_local_connectivity.py", None),
    ],
    "rl_mt10": [
        (REPO / "experiments/rl/robotics/train_mt10_bc.py", None),
        (REPO / "experiments/rl/robotics/train_mt10_ddpg.py", None),
        (REPO / "experiments/rl/robotics/ftn.py", None),
    ],
}


def main():
    for key, entries in SOURCES.items():
        dst_root = CORPUS / key
        if dst_root.exists():
            shutil.rmtree(dst_root)
        n = 0
        for pattern, subtree_root in entries:
            for fp in sorted(_glob.glob(str(pattern), recursive=True)):
                src = Path(fp)
                if not src.is_file():
                    continue
                dst = (dst_root / src.relative_to(subtree_root)
                       if subtree_root else dst_root / src.name)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                n += 1
        size = sum(f.stat().st_size for f in dst_root.rglob("*") if f.is_file())
        print(f"  {key:<10} {n:>4} files  {size/1024:>8.0f} KiB -> {dst_root}")


if __name__ == "__main__":
    main()
