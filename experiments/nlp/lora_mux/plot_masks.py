"""
plot_masks.py — RGB mask-tiling visualisation for the LoRA-Mux experiment.

Mirrors the project_c_4/figures/exp1_synthetic_clf/mask_tiling_*.pdf style:
each task is assigned a distinct colour; the binary mask of each task is
rendered with that colour into an HxW image; tasks are summed (with the
result clipped to [0, 1] per-channel) so overlap shows as mixed colour.

Multiple seeds are tiled into a grid so the topographic stability of the
mask layout is visible across initialisations.

Inputs:
    Per-seed JSONs at  experiments/lora/results/multiseed/seed_<k>.json
    each containing the standard run.py output keyed by method.

Outputs (saved to experiments/lora/figures/):
    mask_tiling_LoRA-Mux.pdf
    mask_tiling_LoRA-FixedMask.pdf  (if present)

Run from project_c_4/:
    uv run python experiments/lora/plot_masks.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MULTISEED_DIR = os.path.join(HERE, "results", "multiseed")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


# Distinct, high-saturation colours per task — same palette as the parent
# project's TASK_COLORS for visual continuity.
TASK_COLORS = np.array([
    [0.90, 0.10, 0.10],   # red       — task 0 (e.g. upper)
    [0.10, 0.72, 0.10],   # green     — task 1 (e.g. lower)
    [0.15, 0.35, 0.95],   # blue      — task 2 (e.g. reverse_words)
    [0.95, 0.75, 0.00],   # yellow    — task 3 (e.g. double_words)
    [0.85, 0.10, 0.85],   # magenta   — task 4 (e.g. starred_vowels)
    [0.00, 0.82, 0.82],   # cyan      — task 5 (e.g. pirate)
    [1.00, 0.50, 0.00],   # orange    — task 6
    [0.55, 0.00, 0.90],   # violet    — task 7
])


def _render_rgb(mask_store: Dict[str, list], grid_dim: int) -> np.ndarray:
    """Sum task colours weighted by per-task binary masks; clip to [0,1]."""
    n = grid_dim * grid_dim
    rgb = np.zeros((n, 3), dtype=np.float32)
    for task_id, m in mask_store.items():
        m_arr = np.asarray(m, dtype=np.float32)[:n]
        col = TASK_COLORS[int(task_id) % len(TASK_COLORS)]
        rgb += np.outer(m_arr, col)
    return rgb.clip(0.0, 1.0).reshape(grid_dim, grid_dim, 3)


def _load_per_seed(method: str) -> List[tuple]:
    """Return [(seed, mask_store, style_names), ...] sorted by seed."""
    out = []
    for path in sorted(glob.glob(os.path.join(MULTISEED_DIR, "seed_*.json"))):
        m = re.search(r"seed_(\d+)\.json$", path)
        if m is None:
            continue
        seed = int(m.group(1))
        with open(path) as f:
            d = json.load(f)
        if method not in d.get("methods", {}):
            continue
        ms = d["methods"][method]["mask_store"]
        names = d["methods"][method].get("style_names", d.get("styles"))
        out.append((seed, ms, names))
    return out


def tile_mask_panels(method: str, out_path: str, n_cols: int = 4):
    rows = _load_per_seed(method)
    if not rows:
        print(f"  (no seeds found for {method}, skipping)")
        return
    # Infer grid_dim from any mask vector length.
    H = len(next(iter(rows[0][1].values())))
    grid_dim = int(round(np.sqrt(H)))
    assert grid_dim * grid_dim == H, f"H={H} is not a perfect square"

    n = len(rows)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.6 * n_cols, 1.7 * n_rows),
        squeeze=False,
    )
    style_names = rows[0][2] or []
    for i, (seed, mask_store, _) in enumerate(rows):
        r, c = i // n_cols, i % n_cols
        ax = axes[r][c]
        img = _render_rgb(mask_store, grid_dim)
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"seed {seed}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    # Hide unused subplots
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    # Legend along the bottom: one swatch per task.
    legend_elems = []
    n_tasks = len(rows[0][1])
    for t in range(n_tasks):
        col = TASK_COLORS[t % len(TASK_COLORS)]
        label = style_names[t] if t < len(style_names) else f"task {t}"
        legend_elems.append(plt.Rectangle((0, 0), 1, 1, fc=col, ec="none", label=label))
    fig.legend(
        handles=legend_elems,
        loc="lower center",
        ncol=min(n_tasks, 8),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(f"{method}: per-seed mask layouts on the {grid_dim}×{grid_dim} component grid",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def main():
    if not os.path.isdir(MULTISEED_DIR):
        print(f"No multiseed results at {MULTISEED_DIR}.")
        print("Run the multiseed sweep first, e.g.:")
        print("  for s in 0 1 2 3 4 5 6 7; do "
              "uv run python experiments/lora/run.py --seed $s "
              "--methods LoRA-FTN LoRA-FixedMask "
              "--out experiments/lora/results/multiseed/seed_$s.json; done")
        return
    for method in ("LoRA-FTN", "LoRA-FixedMask"):
        tile_mask_panels(method, os.path.join(FIG_DIR, f"mask_tiling_{method}.pdf"))


if __name__ == "__main__":
    main()
