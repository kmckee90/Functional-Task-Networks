"""Render xfer_*.png from saved experiment JSON files."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import viz  # noqa: E402


def render(path):
    path = Path(path)
    result = json.loads(path.read_text())
    blocks = result["blocks"]
    rows = [{int(t): scores["acc"] for t, scores in block["eval"].items()}
            for block in blocks]
    tasks = list(range(result["config"]["tasks"]))
    suffix = path.stem
    viz.save_forgetting_matrix(rows, [b["task"] for b in blocks], tasks,
                               suffix, out_dir=str(path.parent))
    print(path.parent / f"xfer_{suffix}.png")
    if "fixed_mask_eval" in blocks[0]:
        fixed_rows = [{int(t): acc for t, acc in block["fixed_mask_eval"].items()}
                      for block in blocks]
        viz.save_forgetting_matrix(fixed_rows, [b["task"] for b in blocks],
                                   tasks, f"fixed_{suffix}", out_dir=str(path.parent))
        print(path.parent / f"xfer_fixed_{suffix}.png")


if __name__ == "__main__":
    for filename in sys.argv[1:]:
        render(filename)
