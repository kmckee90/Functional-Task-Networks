"""Orchestrate the temporal (recurrent FTN) experiment.

Usage (on the pod, from /home/dev/workspace/temporal):
    uv run python run_temporal.py --config small --run smoke --variant v1
    uv run python run_temporal.py --config full --run main --variant both
"""

import argparse
import json
import time
from pathlib import Path

import torch

from ftn_temporal import config as config_mod
from ftn_temporal.evals import eval_tasks, step_faithfulness, train_joint_baseline
from ftn_temporal.model import RecurrentFTN
from ftn_temporal.tasks import AffineTasks
from ftn_temporal.train import train_v1, train_v2


def build_model(cfg):
    return RecurrentFTN(cfg.L, cfg.p, cfg.H, cfg.neuron_depth,
                        cfg.neuron_width, cfg.dropout).to(cfg.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["small", "full"], default="small")
    ap.add_argument("--run", default="smoke")
    ap.add_argument("--variant", choices=["v1", "v2", "both"], default="both")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cycles", type=int, default=None)
    ap.add_argument("--steps-per-task", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--v2-s-train", type=int, default=None)
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    cfg = getattr(config_mod, args.config)()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.cycles is not None:
        cfg.cycles = args.cycles
    if args.steps_per_task is not None:
        cfg.steps_per_task = args.steps_per_task
    if args.lr is not None:
        cfg.lr = args.lr
    if args.v2_s_train is not None:
        cfg.v2_s_train = args.v2_s_train
    torch.manual_seed(cfg.seed)

    out = Path("results") / args.run
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(str(msg) + "\n")

    t0 = time.time()
    tasks_obj = AffineTasks(cfg)
    log(f"config={args.config} seed={cfg.seed} variant={args.variant}")
    log(json.dumps(cfg.to_dict(), default=str))
    log(f"group size={len(tasks_obj.length)} level sizes={tasks_obj.level_sizes()}")
    log(f"train tasks ({len(tasks_obj.train_tasks)}): {tasks_obj.train_tasks}")
    log(f"held-out tasks ({len(tasks_obj.held_tasks)}), levels "
        f"{sorted({tasks_obj.length[g] for g in tasks_obj.held_tasks})}")

    results = dict(cfg=cfg.to_dict(),
                   level_sizes=tasks_obj.level_sizes(),
                   train_tasks=tasks_obj.train_tasks,
                   held_tasks=tasks_obj.held_tasks,
                   words={str(g): tasks_obj.word[g] for g in
                          tasks_obj.train_tasks + tasks_obj.held_tasks})

    variants = ["v1", "v2"] if args.variant == "both" else [args.variant]
    for v in variants:
        torch.manual_seed(cfg.seed)
        model = build_model(cfg)
        log(f"=== training {v} ({sum(p.numel() for p in model.parameters()):,} params) ===")
        trainer = train_v1 if v == "v1" else train_v2
        history = trainer(model, tasks_obj, cfg, log_fn=log)
        torch.save(model.state_dict(), out / f"model_{v}.pt")
        log(f"{v} training done at {time.time() - t0:.0f}s")

        log(f"=== evaluating {v}: trained tasks ===")
        trained_rows = eval_tasks(model, tasks_obj, cfg,
                                  tasks_obj.train_tasks, log_fn=log)
        log(f"=== evaluating {v}: held-out tasks (length OOD) ===")
        held_rows = eval_tasks(model, tasks_obj, cfg,
                               tasks_obj.held_tasks, log_fn=log)
        faith = step_faithfulness(model, tasks_obj, cfg, held_rows)
        results[v] = dict(history=history, trained=trained_rows,
                          held=held_rows, faithfulness=faith)

        # random-bank control, same eval battery on an untrained twin
        log(f"=== {v}: random-bank control ===")
        torch.manual_seed(cfg.seed + 999)
        rand_model = build_model(cfg)
        rand_rows = eval_tasks(rand_model, tasks_obj, cfg,
                               tasks_obj.held_tasks[:12], log_fn=log)
        results[v]["random_bank"] = rand_rows

    log("=== joint task-conditioned MLP baseline ===")
    joint_train, joint_held = train_joint_baseline(tasks_obj, cfg, log_fn=log)
    results["joint_train"] = joint_train
    results["joint_held"] = joint_held
    results["elapsed_s"] = time.time() - t0

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"results written ({time.time() - t0:.0f}s total)")

    if not args.skip_figures:
        from ftn_temporal.viz import make_all_figures
        make_all_figures(out, results, tasks_obj, cfg)
        log("figures written")


if __name__ == "__main__":
    main()
