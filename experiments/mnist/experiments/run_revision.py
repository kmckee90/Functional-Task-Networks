"""
run_revision.py — NeurIPS-revision experiments.

Adds the reviewer-requested comparators on Permuted/Rotated MNIST plus
the existing synthetic and shared-shuffle MNIST families:

  * ER (Experience Replay) baseline: NoMask configurer + random
    (uniform) replay, identical memory budget as FTN. Isolates
    whether retention is from the mask mechanism or from replay.
  * RetrievalRouter eval: at evaluation time, instead of running the
    inner-loop mask optimisation, scan the bank of stored training
    masks and pick the argmin-support-loss mask. The training-time
    configurer can be FTN (per-batch reconfig) or FixedMask
    (PackNet/SupSup-style disjoint blocks); the latter is the
    closest-mechanism parameter-isolation comparator.
  * No-replay ablation: FTN with use_replay=False.
  * Support-set size sweep: reconfig_batch_size in {1,5,10,50,256}.

Conditions per benchmark (5 seeds default):
  - FTN (default: AnnealSTE, S=30, K=512, per-batch reconfig)
  - FTN + Router-eval (training same as FTN; eval via retrieval)
  - FixedMask + Router-eval (SupSup-like)
  - NoMask + ER replay (random replay buffer)
  - NoMask + no replay (naive sequential)
  - FTN + no replay (mask-only)

Usage:
    uv run python experiments/run_revision.py --tasks permuted rotated --seeds 0 1 2 3 4
    uv run python experiments/run_revision.py --ablation no_replay --tasks synthetic permuted
    uv run python experiments/run_revision.py --ablation support_size \
                  --tasks permuted rotated --seeds 0 1 2

Outputs a flat dict keyed by run-id at results/revision_<task>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import masks as M
import metrics
import models
import numpy as np
import torch
from baselines import progressive_train, supsup_train
from tasks import (
    MNISTSharedShuffle,
    PermutedMNIST,
    RotatedMNIST,
    SyntheticTasks,
)
from trainer import TrainConfig, block_sequential_train


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES_DIR = "results"

# --- Default model + mask hyperparameters (match exp1-6 defaults) ---
HIDDEN_SIZE = 8192
K = 512
MODEL_LAYERS = 8
MODEL_INNER_SIZE = 8
MODEL_DROPOUT_P = 0.2
RECONFIG_LR = 0.3
RECONFIG_N = 256
RECONFIG_STEPS = 30  # AnnealSTE S=30


def synthetic_taskset(seed, device):
    return SyntheticTasks(
        input_size=2, output_size=2, n_tasks=12, n_components=6,
        task_size=1, frequency=8.0, classification=True,
        seed=seed, device=device, test_size=2048,
    )

def mnist_shared_taskset(seed, _device):
    return MNISTSharedShuffle(
        n_tasks=6, n_components=4, component_size=5, seed=seed, data_dir="./data",
    )

def permuted_taskset(seed, _device):
    return PermutedMNIST(n_tasks=10, seed=seed, data_dir="./data")

def rotated_taskset(seed, _device):
    return RotatedMNIST(n_tasks=10, max_angle=180.0, seed=seed, data_dir="./data")


TASK_REGISTRY = {
    "synthetic": {
        "build": synthetic_taskset,
        "input_size": 2,
        "output_size": 2,
        "train_cfg_kwargs": dict(
            n_epochs=1, train_steps=1000, batch_size=256,
            eval_batch_size=2048, reconfig_batch_size=256,
            lr=3e-4, replay_n_per_step=2, replay_buf_size=200,
            reconfig_freq="batch",
        ),
    },
    "mnist_shared": {
        "build": mnist_shared_taskset,
        "input_size": 784,
        "output_size": 10,
        "train_cfg_kwargs": dict(
            n_epochs=1, train_steps=1000, batch_size=256,
            eval_batch_size=2048, reconfig_batch_size=256,
            lr=3e-4, replay_n_per_step=2, replay_buf_size=200,
            reconfig_freq="batch",
        ),
    },
    "permuted": {
        "build": permuted_taskset,
        "input_size": 784,
        "output_size": 10,
        "train_cfg_kwargs": dict(
            n_epochs=1, train_steps=1000, batch_size=256,
            eval_batch_size=2048, reconfig_batch_size=256,
            lr=3e-4, replay_n_per_step=2, replay_buf_size=200,
            reconfig_freq="batch",
        ),
    },
    "rotated": {
        "build": rotated_taskset,
        "input_size": 784,
        "output_size": 10,
        "train_cfg_kwargs": dict(
            n_epochs=1, train_steps=1000, batch_size=256,
            eval_batch_size=2048, reconfig_batch_size=256,
            lr=3e-4, replay_n_per_step=2, replay_buf_size=200,
            reconfig_freq="batch",
        ),
    },
}


# ---------------------------------------------------------------------------
# Method specs:
#   "train"       -> mask configurer used during training
#   "eval"        -> mask configurer used at evaluation (None => same as train)
#   "use_replay"  -> bool
#   "replay_mode" -> "mask_overlap" | "random"
# ---------------------------------------------------------------------------


def make_train_cfg(method: str, is_clf: bool, seed: int = 0):
    if method == "FTN":
        return M.KWTA_AnnealedSTE(
            hidden_size=HIDDEN_SIZE, classification=is_clf,
            k=K, reconfig_steps=RECONFIG_STEPS,
            reconfig_lr=RECONFIG_LR, reconfig_n=RECONFIG_N,
        )
    if method == "FixedMask":
        return M.FixedMask(hidden_size=HIDDEN_SIZE, k=K)
    if method == "NoMask":
        return M.NoMask(hidden_size=HIDDEN_SIZE, k=K)
    if method == "RandMask":
        return M.RandMask(hidden_size=HIDDEN_SIZE, k=K, per_call_resample=True, seed=seed)
    raise ValueError(f"Unknown training method {method}")


def make_eval_cfg(eval_kind: str | None, is_clf: bool):
    if eval_kind is None:
        return None
    if eval_kind == "router":
        return M.RetrievalRouter(hidden_size=HIDDEN_SIZE, classification=is_clf)
    if eval_kind == "ftn":
        # Use the FTN configurer at eval (inner-loop support-set optimisation),
        # regardless of the training-time mask source. Used by FTN_RandMask to
        # mirror the LoRA-side ablation protocol: only the training-time mask
        # source differs from FTN, eval-time configurer is the same.
        return M.KWTA_AnnealedSTE(
            hidden_size=HIDDEN_SIZE, classification=is_clf,
            k=K, reconfig_steps=RECONFIG_STEPS,
            reconfig_lr=RECONFIG_LR, reconfig_n=RECONFIG_N,
        )
    raise ValueError(f"Unknown eval_kind {eval_kind}")


METHOD_SPECS = {
    "FTN":               dict(train="FTN", eval=None, use_replay=True, replay_mode="mask_overlap"),
    "FTN+Router":        dict(train="FTN", eval="router", use_replay=True, replay_mode="mask_overlap"),
    "FTN_noreplay":      dict(train="FTN", eval=None, use_replay=False, replay_mode="mask_overlap"),
    "FixedMask":         dict(train="FixedMask", eval=None, use_replay=True, replay_mode="mask_overlap"),
    "FixedMask+Router":  dict(train="FixedMask", eval="router", use_replay=False, replay_mode="mask_overlap"),
    "NoMask":            dict(train="NoMask", eval=None, use_replay=False, replay_mode="mask_overlap"),
    "NoMask+ER":         dict(train="NoMask", eval=None, use_replay=True, replay_mode="random"),
    # Decisive routing-vs-random-mask ablation: same training pipeline as FTN
    # (filtered replay, parameter budget, schedule), random per-batch top-k
    # mask in place of the support-conditioned configurer; eval uses the FTN
    # configurer to match FTN's eval protocol.
    "FTN_RandMask":      dict(train="RandMask", eval="ftn", use_replay=True, replay_mode="mask_overlap"),
    # Reviewer-requested baselines: separate training paths in baselines.py.
    "SupSup":            dict(train="SupSup", eval=None, use_replay=False, replay_mode="mask_overlap"),
    "Progressive":       dict(train="Progressive", eval=None, use_replay=False, replay_mode="mask_overlap"),
}


def build_model(input_size: int, output_size: int, seed: int):
    torch.manual_seed(seed)
    return models.NeuroModel(
        input_size, output_size, HIDDEN_SIZE,
        layers=MODEL_LAYERS, inner_size=MODEL_INNER_SIZE, dropout_p=MODEL_DROPOUT_P,
    )


def run_one(method: str, task_family: str, seed: int,
            reconfig_batch_size: int | None = None,
            device: str = DEVICE) -> dict:
    cfg = TASK_REGISTRY[task_family]
    spec = METHOD_SPECS[method]
    task_set = cfg["build"](seed, device)
    is_clf = getattr(task_set, "is_classification", True)

    train_kwargs = dict(cfg["train_cfg_kwargs"])
    if reconfig_batch_size is not None:
        train_kwargs["reconfig_batch_size"] = reconfig_batch_size
    train_cfg = TrainConfig(
        device=device,
        use_replay=spec["use_replay"],
        replay_mode=spec["replay_mode"],
        **train_kwargs,
    )

    model = build_model(cfg["input_size"], cfg["output_size"], seed)

    t0 = time.time()
    if spec["train"] == "SupSup":
        out = supsup_train(
            model, task_set, train_cfg,
            hidden_size=HIDDEN_SIZE, k=K, output_size=cfg["output_size"],
            verbose=False,
        )
    elif spec["train"] == "Progressive":
        out = progressive_train(
            model, task_set, train_cfg,
            hidden_size=HIDDEN_SIZE, k=K, output_size=cfg["output_size"],
            verbose=False,
        )
    else:
        train_mask_cfg = make_train_cfg(spec["train"], is_clf, seed=seed)
        eval_mask_cfg = make_eval_cfg(spec["eval"], is_clf)
        out = block_sequential_train(
            model, train_mask_cfg, task_set, train_cfg,
            verbose=False, eval_mask_cfg=eval_mask_cfg,
        )
    elapsed = time.time() - t0

    return {
        "method": method,
        "task_family": task_family,
        "seed": seed,
        "reconfig_batch_size": train_kwargs["reconfig_batch_size"],
        "elapsed_sec": elapsed,
        "summary_stored": metrics.summarise(out["evals"]["stored"], label=method),
        "summary_reconfig": metrics.summarise(out["evals"]["reconfig"], label=method),
        "perf_matrix_reconfig": out["evals"]["reconfig"],
        "head_start": out["head_start"],
    }


def load_results(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": {}}


def save_results(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def aggregate(data):
    by_key: dict = {}
    for r in data["runs"].values():
        key = (r["method"], int(r.get("reconfig_batch_size", 256)))
        by_key.setdefault(key, []).append(r)
    agg = {}
    for (method, n), runs in by_key.items():
        e = {"n_seeds": len(runs)}
        for proto in ("summary_stored", "summary_reconfig"):
            for field in ("ACC", "FM", "BWT"):
                vals = [r[proto][field] for r in runs if r[proto].get(field) is not None]
                if vals:
                    e[f"{proto}.{field}"] = {"mean": float(np.mean(vals)),
                                              "std": float(np.std(vals))}
        # Mean head-start across tasks 1..N-1
        hs_vals = []
        for r in runs:
            hs = [hs_t["task_loss_with_mask"] for hs_t in r["head_start"][1:]
                  if hs_t["task_loss_with_mask"] is not None]
            if hs:
                hs_vals.append(float(np.mean(hs)))
        if hs_vals:
            e["head_start_mean"] = {"mean": float(np.mean(hs_vals)),
                                    "std": float(np.std(hs_vals))}
        agg.setdefault(method, {})[str(n)] = e
    data["aggregated"] = agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["permuted"],
                        choices=list(TASK_REGISTRY.keys()))
    parser.add_argument("--methods", nargs="+", default=list(METHOD_SPECS.keys()),
                        choices=list(METHOD_SPECS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--ablation", choices=["main", "support_size"], default="main")
    parser.add_argument(
        "--support-sizes", nargs="+", type=int, default=[1, 5, 10, 50, 256],
        help="reconfig_batch_size sweep when --ablation=support_size",
    )
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--results-suffix", default="",
                        help="appended to results filename: results/revision_<task><suffix>.json")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Working directory: {os.getcwd()}")
    print(f"Device   : {args.device}")
    print(f"Tasks    : {args.tasks}")
    print(f"Methods  : {args.methods}")
    print(f"Seeds    : {args.seeds}")
    print(f"Ablation : {args.ablation}")

    for task_family in args.tasks:
        suffix = args.results_suffix or f"_{args.ablation}"
        res_path = os.path.join(RES_DIR, f"revision_{task_family}{suffix}.json")
        data = load_results(res_path)
        runs = data.setdefault("runs", {})

        if args.ablation == "support_size":
            # Only sweep n on FTN by default (most informative for the
            # support-size sensitivity story).
            methods_to_run = ["FTN"] if "FTN" in args.methods else args.methods[:1]
            n_values = args.support_sizes
        else:
            methods_to_run = args.methods
            n_values = [None]

        for method in methods_to_run:
            for n in n_values:
                for seed in args.seeds:
                    if n is None:
                        rkey = f"{method}__s{seed}"
                    else:
                        rkey = f"{method}__n{n}__s{seed}"
                    if rkey in runs:
                        print(f"  [skip] {task_family} {rkey}")
                        continue
                    print(f"  [run]  {task_family} {rkey}")
                    res = run_one(
                        method, task_family, seed,
                        reconfig_batch_size=n, device=args.device,
                    )
                    print(f"          ACC={res['summary_reconfig'].get('ACC',0):.3f} "
                          f"FM={res['summary_reconfig'].get('FM',0):.3f} "
                          f"BWT={res['summary_reconfig'].get('BWT',0):+.3f} "
                          f"({res['elapsed_sec']:.1f}s)")
                    runs[rkey] = res
                    aggregate(data)
                    save_results(data, res_path)

        aggregate(data)
        save_results(data, res_path)
        print(f"  Saved: {res_path}")


if __name__ == "__main__":
    main()
