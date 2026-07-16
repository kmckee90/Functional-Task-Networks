"""Regenerate all figures for a finished run from its saved artifacts."""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftn_nar.config import Config  # noqa: E402
from ftn_nar.data import TaskGrid  # noqa: E402
from ftn_nar.model import FTN2  # noqa: E402
from ftn_nar.synthesis import MaskSynthesizer  # noqa: E402
from ftn_nar.viz import make_all_figures  # noqa: E402


def main(run="main2"):
    out = Path("results") / run
    ckpt = torch.load(out / "checkpoint.pt", weights_only=False)
    fm = torch.load(out / "final_masks.pt", weights_only=False)
    results = json.load(open(out / "results.json"))
    # results.json stringifies tuples; rebuild from checkpoint lists.
    results["train_cells"] = [tuple(c) for c in ckpt["train_cells"]]
    results["held_cells"] = [tuple(c) for c in ckpt["held_cells"]]
    for row in results["trained_rows"] + results["held_rows"]:
        row["cell"] = tuple(eval(row["cell"]) if isinstance(row["cell"], str)
                            else row["cell"])
    for row in results["joint_train"] + results["joint_held"]:
        row["cell"] = tuple(eval(row["cell"]) if isinstance(row["cell"], str)
                            else row["cell"])
    for row in results["consistency"]:
        row["cell"] = tuple(eval(row["cell"]) if isinstance(row["cell"], str)
                            else row["cell"])
        row["composite"] = tuple(eval(row["composite"])
                                 if isinstance(row["composite"], str)
                                 else row["composite"])

    cfg = Config(**ckpt["cfg"])
    grid = TaskGrid(cfg.p, cfg.L, cfg.input_train_frac, cfg.seed, cfg.device)
    model = FTN2(cfg.L * cfg.p, cfg.L * cfg.p, cfg.H1, cfg.H2, cfg.d_mid,
                 cfg.neuron_depth, cfg.neuron_width, cfg.dropout).to(cfg.device)
    model.load_state_dict(ckpt["model"])
    train_cells = results["train_cells"]
    synthesizer = MaskSynthesizer(fm["final_scores"], train_cells, cfg.H1,
                                  cfg.k1, cfg.k2, cfg.p, cfg.L)
    make_all_figures(out, model, grid, results, fm["final_masks"],
                     fm["final_scores"], synthesizer, cfg)
    print("figures regenerated")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "main2")
