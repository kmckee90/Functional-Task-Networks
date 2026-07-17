"""Re-run the evaluation battery for an already-trained variant (no
retraining). Used to iterate on the trace-search procedure — e.g. removing
the identity atom from the BFS action set — against frozen checkpoints.

    uv run python scripts/reeval.py <run> <variant> [<out_suffix>]
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ftn_temporal.config import full  # noqa: E402
from ftn_temporal.evals import eval_tasks, step_faithfulness  # noqa: E402
from ftn_temporal.model import RecurrentFTN  # noqa: E402
from ftn_temporal.tasks import AffineTasks  # noqa: E402


def main(run="main", variant="v1", suffix="noid"):
    cfg = full()
    torch.manual_seed(cfg.seed)
    out = Path("results") / run
    tasks_obj = AffineTasks(cfg)
    model = RecurrentFTN(cfg.L, cfg.p, cfg.H, cfg.neuron_depth,
                         cfg.neuron_width, cfg.dropout).to(cfg.device)
    model.load_state_dict(torch.load(out / f"model_{variant}.pt",
                                     weights_only=True))

    trained = eval_tasks(model, tasks_obj, cfg, tasks_obj.train_tasks)
    held = eval_tasks(model, tasks_obj, cfg, tasks_obj.held_tasks)
    faith = step_faithfulness(model, tasks_obj, cfg, held)
    with open(out / f"reeval_{variant}_{suffix}.json", "w") as f:
        json.dump(dict(trained=trained, held=held, faithfulness=faith),
                  f, indent=2, default=str)
    print("reeval written")


if __name__ == "__main__":
    main(*(sys.argv[1:4] or []))
