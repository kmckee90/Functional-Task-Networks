"""Compare ordinary replay, no replay, and current-input overlap preservation.

Run from anywhere: python projects/overlap/run.py --mode overlap --seed 123
The default model, task generator, mask search, overlap band, and evaluation
protocol come from ftn/config.py and ftn/run_experiments.py.
"""

import argparse
import copy
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
import metrics  # noqa: E402
import run_experiments as base  # noqa: E402
import tasks  # noqa: E402
from ftn import FTN  # noqa: E402


class MaskBuffer:
    """Masks only; no earlier inputs, labels, or output targets are retained."""

    def __init__(self, maxlen):
        self.masks = deque(maxlen=maxlen)

    def add(self, mask):
        self.masks.append(mask.detach().clone())


def overlap_indices(masks, current, low, high):
    if not masks:
        return torch.empty(0, dtype=torch.long, device=current.device)
    stored = torch.stack(tuple(masks))
    fraction = (stored * current).sum(-1) / current.sum()
    return torch.where((fraction > low) & (fraction <= high))[0]


def select_masks(masks, current, n, low, high):
    eligible = overlap_indices(masks, current, low, high)
    if eligible.numel() == 0 or n == 0:
        return [], int(eligible.numel())
    # Match base replay: sample n entries with replacement from eligible masks.
    chosen = eligible[torch.randint(eligible.numel(), (n,), device=current.device)]
    return [masks[int(i)] for i in chosen], int(eligible.numel())


def overlap_targets(model, x, selected):
    """Freeze every selected old-mask prediction before any weight update."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        targets = [model(x, m=mask).detach() for mask in selected]
    model.train(was_training)
    return targets


def run(args):
    cfg = config.Default()
    cfg.SEED = args.seed
    cfg.N_TASK = args.tasks
    cfg.TRAIN_DURATION = args.duration
    cfg.N_RETRO_EVAL_BATCHES = args.eval_batches
    cfg.EVAL_RECONFIG_STEPS = args.eval_reconfig_steps
    # The root evaluator reads this global, just as its own main() does.
    base.Config = cfg
    base.DataGen = tasks.DataGenSine

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    data_gen = base.make_data_gen(args.seed + 100)
    # Copy the task functions and local RNG without consuming model-init RNG.
    fixed_gen = copy.deepcopy(data_gen)
    model = FTN(data_gen.total_input_dim, data_gen.output_dim,
                cfg.HIDDEN_SIZE, **cfg.FTN_PARS)
    lr = args.lr if args.lr is not None else cfg.TRAIN_LR
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    replay = base.ReplayBuffer(cfg.REPLAY_MAXLEN, cfg.REPLAY_BATCH_SIZE)
    masks = MaskBuffer(cfg.REPLAY_MAXLEN)
    collector = metrics.MetricsCollector()
    records = []
    seen = []
    final_masks = {}
    selected_count = 0
    eligible_count = 0
    nonempty_count = 0
    tail_n = max(1, cfg.TRAIN_DURATION // 5)

    for task in range(cfg.N_TASK):
        optimizer.state = defaultdict(dict)  # base resets Adam per task
        tail_acc = []
        task_loss = []
        t0 = time.perf_counter()
        for batch in range(cfg.TRAIN_DURATION):
            x, y = data_gen(task)
            pred = model(x, y)  # same per-batch mask reconfiguration as base
            primary = model.loss(pred, y)
            if batch < cfg.N_SAMPLE_PER_TASK:
                if args.mode == "replay":
                    replay.add((x, y), model.m)
                elif args.mode == "overlap":
                    masks.add(model.m)

            if args.mode == "overlap":
                selected, eligible = select_masks(
                    masks.masks, model.m, args.overlap_samples,
                    args.overlap_low, args.overlap_high,
                )
                targets = overlap_targets(model, x, selected)
                selected_count += len(selected)
                eligible_count += eligible
                nonempty_count += bool(selected)
                # The initial old-mask loss is exactly zero with default dropout=0.
                # Recompute both terms after every update while retaining x, y,
                # the current mask, selected masks, and detached targets.
                for inner in range(args.steps):
                    if inner:
                        pred = model(x)
                        primary = model.loss(pred, y)
                    optimizer.zero_grad()
                    old_loss = sum(
                        (F.mse_loss(model(x, m=m), target) for m, target in zip(selected, targets)),
                        primary.new_zeros(()),
                    )
                    (primary + args.weight * old_loss).backward()
                    optimizer.step()
            elif args.mode == "none_multi":
                for inner in range(args.steps):
                    if inner:
                        pred = model(x)
                        primary = model.loss(pred, y)
                    optimizer.zero_grad()
                    primary.backward()
                    optimizer.step()
            else:
                optimizer.zero_grad()
                primary.backward()
                if args.mode == "replay":
                    base.replay_step(model, replay)
                optimizer.step()

            if batch >= cfg.TRAIN_DURATION - tail_n:
                with torch.no_grad():
                    tail_acc.append(base.accuracy(pred, y))
            task_loss.append(float(primary.detach()))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        seen.append(task)
        final_masks[task] = model.m.detach().clone()
        # Same holdout-based backwards/prospective evaluation as root runner.
        evals = base.evaluate_backwards(model, data_gen, list(range(cfg.N_TASK)))
        fixed_evals = {}
        with torch.no_grad():
            for old_task in seen:
                scores = []
                for _ in range(cfg.N_RETRO_EVAL_BATCHES):
                    x_fixed, y_fixed = fixed_gen(old_task)
                    scores.append(base.accuracy(model(x_fixed, m=final_masks[old_task]), y_fixed))
                fixed_evals[old_task] = float(np.mean(scores))
        collector.record_block(task, task, float(np.mean(tail_acc)),
                               {t: evals[t]["acc"] for t in evals}, seen,
                               block_wall_s=wall)
        row = {"task": task, "train_loss_tail": float(np.mean(task_loss[-tail_n:])),
               "train_acc_tail": float(np.mean(tail_acc)), "eval": evals,
               "fixed_mask_eval": fixed_evals,
               "train_wall_s": wall}
        records.append(row)
        print(json.dumps({"mode": args.mode, "seed": args.seed, **row}), flush=True)

    result = {
        "mode": args.mode, "seed": args.seed,
        "config": {"duration": cfg.TRAIN_DURATION, "tasks": cfg.N_TASK,
                   "batch_size": cfg.T, "hidden_size": cfg.HIDDEN_SIZE,
                   "k": cfg.K, "reconfig_steps": cfg.RECONFIG_STEPS,
                   "eval_reconfig_steps": cfg.EVAL_RECONFIG_STEPS,
                   "eval_batches": cfg.N_RETRO_EVAL_BATCHES,
                   "lr": lr, "overlap_samples": args.overlap_samples,
                   "overlap_low": args.overlap_low,
                   "overlap_high": args.overlap_high,
                   "steps": args.steps if args.mode in ("overlap", "none_multi") else 1,
                   "weight": args.weight},
        "blocks": records,
        "summary": collector.summary(),
        "selection": {"total_selected": selected_count,
                      "mean_eligible_per_batch": eligible_count / (cfg.N_TASK * cfg.TRAIN_DURATION),
                      "fraction_batches_with_overlap": nonempty_count / (cfg.N_TASK * cfg.TRAIN_DURATION)},
    }
    result["summary"]["final_mean_seen_acc"] = float(np.mean(
        [records[-1]["eval"][t]["acc"] for t in range(cfg.N_TASK)]))
    result["summary"]["final_mean_fixed_mask_acc"] = float(np.mean(
        list(records[-1]["fixed_mask_eval"].values())))
    result["summary"]["avg_fixed_mask_recall_acc"] = float(np.mean(
        [acc for row in records for t, acc in row["fixed_mask_eval"].items()
         if t != row["task"]]))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("RESULT", path, json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("replay", "none", "none_multi", "overlap"), required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--tasks", type=int, default=config.Default.N_TASK)
    p.add_argument("--duration", type=int, default=config.Default.TRAIN_DURATION)
    p.add_argument("--eval-batches", type=int, default=config.Default.N_RETRO_EVAL_BATCHES)
    p.add_argument("--eval-reconfig-steps", type=int, default=config.Default.EVAL_RECONFIG_STEPS)
    p.add_argument("--lr", type=float)
    p.add_argument("--steps", type=int, default=2,
                   help="Total optimizer steps on each current batch, at least 2.")
    p.add_argument("--weight", type=float, default=1.0)
    p.add_argument("--overlap-samples", type=int, default=config.Default.N_REPLAY_SAMPLES)
    p.add_argument("--overlap-low", type=float, default=0.2)
    p.add_argument("--overlap-high", type=float, default=0.8)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    if a.mode == "overlap" and a.steps < 2:
        p.error("overlap mode requires at least two optimizer steps")
    if a.out is None:
        a.out = str(Path(__file__).parent / "results" /
                    f"{a.mode}_s{a.seed}_n{a.steps}_lr{a.lr or config.Default.TRAIN_LR}.json")
    run(a)
