"""
run_kt.py — Knowledge-transfer experiments for func_net_kt.

Compares three KWTA-only mask-optimisation strategies on two task
families with structured concept sharing:

    Methods:        Sigmoid, AnnealSTE, Gumbel
    Compute axis:   reconfig_steps in {50, 500, 5000, 50000}
    Tasks:          SyntheticTasks (synthetic correlated sinusoids)
                    MNISTSharedShuffle (MNIST with structured digit-label
                                        permutations sharing components
                                        across tasks)

Tracked per (method, budget, seed):

    1. Retrospective forgetting evals (stored + reconfigured masks)
    2. Pre-training "head start" loss after the per-task mask
       optimisation (before any neuron training)
    3. Backward transfer (BWT) on prior tasks at the end of training
    4. Neuron-sharing fidelity = correlation(model_overlap, gt_overlap)

Output:
    results/{task_family}.json       multi-seed flat dict
    figures/{task_family}/...        sharing matrices, head-start curves

Run:
    uv run python experiments/run_kt.py --tasks synthetic
    uv run python experiments/run_kt.py --tasks mnist_shared
    uv run python experiments/run_kt.py --tasks synthetic mnist_shared \
        --budgets 50 500 --methods Sigmoid AnnealSTE --seeds 0 1
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
import plotting
import torch
from tasks import MNISTSharedShuffle, SyntheticTasks
from trainer import TrainConfig, block_sequential_train

# =============================================================================
# STATIC CONFIG -- edit these to tune model / task / mask / experiment
# =============================================================================
# Everything below this section reads from these constants; nothing is
# duplicated downstream.  CLI flags can override sweep axes (--methods,
# --budgets, --seeds, --tasks, --device, --include-baselines, --no-figs)
# but the per-task / per-method / per-model hyperparameters are tuned only
# here.

# --- I/O & device ----------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RES_DIR = "results"
FIG_DIR = "figures"

# --- Sweep axes ------------------------------------------------------------

DEFAULT_SEEDS = [0]
DEFAULT_BUDGETS = [30]
DEFAULT_METHODS = ["AnnealSTE"]  # ["Sigmoid", "AnnealSTE", "Gumbel"]
BASELINE_METHODS = ["FixedMask", "NoMask"]

# --- Model (NeuroModel = ParallelMLP bank + linear readout) ----------------

HIDDEN_SIZE = (
    512 * 16
)  # number of parallel neuron slots. Should be a little higher than K*N_TASKS
K = 512  # KWTA budget (active neurons per task)
MODEL_LAYERS = 8  # hidden layers inside each per-neuron MLP
MODEL_INNER_SIZE = 8  # hidden width inside each per-neuron MLP
MODEL_DROPOUT_P = 0.2  # dropout on the per-neuron output vector

# --- Mask configurers ------------------------------------------------------
# Reconfig LRs tuned per method by their objective scale.

RECONFIG_LR = {
    "Sigmoid": 0.3,
    "AnnealSTE": 0.3,
    "Gumbel": 0.2,
}
RECONFIG_N = 256  # max samples used by the configurer per call

# Gumbel-Sigmoid (Binary Concrete) specifics
GUMBEL_TAU_START = 1.0  # initial relaxation temperature
GUMBEL_TAU_END = 0.1  # final relaxation temperature
GUMBEL_REG_STRENGTH = 1.0  # weight on (E[#active]-k)^2/k^2 budget penalty

# --- Synthetic task family (correlated sinusoidal mixture) -----------------

SYNTH_INPUT_SIZE = 2
SYNTH_OUTPUT_SIZE = 2
SYNTH_N_TASKS = 12
SYNTH_N_COMPONENTS = 6
SYNTH_TASK_SIZE = 1
SYNTH_FREQUENCY = 8
SYNTH_CLASSIFICATION = True
SYNTH_TEST_SIZE = 2048

# Synthetic training config
SYNTH_N_EPOCHS = 1
SYNTH_TRAIN_STEPS = 5000
SYNTH_BATCH_SIZE = 256
SYNTH_EVAL_BATCH_SIZE = 2048
SYNTH_RECONFIG_BATCH_SIZE = 256
SYNTH_LR = 3e-4
SYNTH_USE_REPLAY = True
SYNTH_REPLAY_BUF_SIZE = 200
SYNTH_REPLAY_N_PER_STEP = 2
SYNTH_RECONFIG_FREQ = "batch"  # "task" | "epoch" | "batch"

# --- MNIST shared-shuffle task family --------------------------------------

MNIST_INPUT_SIZE = 784
MNIST_OUTPUT_SIZE = 10
MNIST_N_TASKS = 6
MNIST_N_COMPONENTS = 4
MNIST_COMPONENT_SIZE = 5
MNIST_DATA_DIR = "./data"

# MNIST training config
MNIST_N_EPOCHS = 1
MNIST_TRAIN_STEPS = 3000
MNIST_BATCH_SIZE = 256
MNIST_EVAL_BATCH_SIZE = 2048
MNIST_RECONFIG_BATCH_SIZE = 256
MNIST_LR = 3e-4
MNIST_USE_REPLAY = True
MNIST_REPLAY_BUF_SIZE = 200
MNIST_REPLAY_N_PER_STEP = 2
MNIST_RECONFIG_FREQ = "task"

# =============================================================================
# END STATIC CONFIG -- nothing below this line is intended to be tuned.
# =============================================================================


MODEL_KWARGS = {
    "layers": MODEL_LAYERS,
    "inner_size": MODEL_INNER_SIZE,
    "dropout_p": MODEL_DROPOUT_P,
}


# ---------------------------------------------------------------------------
# Task-family configurations
# ---------------------------------------------------------------------------


def synthetic_taskset(seed: int, device: str):
    """
    Correlated synthetic-sinusoid tasks built as pairwise unions of
    concept components on a low-dim input space.
    """
    return SyntheticTasks(
        input_size=SYNTH_INPUT_SIZE,
        output_size=SYNTH_OUTPUT_SIZE,
        n_tasks=SYNTH_N_TASKS,
        n_components=SYNTH_N_COMPONENTS,
        task_size=SYNTH_TASK_SIZE,
        frequency=SYNTH_FREQUENCY,
        classification=SYNTH_CLASSIFICATION,
        seed=seed,
        device=device,
        test_size=SYNTH_TEST_SIZE,
    )


def mnist_taskset(seed: int, _device: str):
    return MNISTSharedShuffle(
        n_tasks=MNIST_N_TASKS,
        n_components=MNIST_N_COMPONENTS,
        component_size=MNIST_COMPONENT_SIZE,
        seed=seed,
        data_dir=MNIST_DATA_DIR,
    )


TASK_REGISTRY = {
    "synthetic": {
        "build": synthetic_taskset,
        "input_size": SYNTH_INPUT_SIZE,
        "output_size": SYNTH_OUTPUT_SIZE,
        "train_cfg": TrainConfig(
            n_epochs=SYNTH_N_EPOCHS,
            train_steps=SYNTH_TRAIN_STEPS,
            batch_size=SYNTH_BATCH_SIZE,
            eval_batch_size=SYNTH_EVAL_BATCH_SIZE,
            reconfig_batch_size=SYNTH_RECONFIG_BATCH_SIZE,
            lr=SYNTH_LR,
            use_replay=SYNTH_USE_REPLAY,
            replay_n_per_step=SYNTH_REPLAY_N_PER_STEP,
            replay_buf_size=SYNTH_REPLAY_BUF_SIZE,
            reconfig_freq=SYNTH_RECONFIG_FREQ,
            device=DEVICE,
        ),
    },
    "mnist_shared": {
        "build": mnist_taskset,
        "input_size": MNIST_INPUT_SIZE,
        "output_size": MNIST_OUTPUT_SIZE,
        "train_cfg": TrainConfig(
            n_epochs=MNIST_N_EPOCHS,
            train_steps=MNIST_TRAIN_STEPS,
            batch_size=MNIST_BATCH_SIZE,
            eval_batch_size=MNIST_EVAL_BATCH_SIZE,
            reconfig_batch_size=MNIST_RECONFIG_BATCH_SIZE,
            lr=MNIST_LR,
            use_replay=MNIST_USE_REPLAY,
            replay_n_per_step=MNIST_REPLAY_N_PER_STEP,
            replay_buf_size=MNIST_REPLAY_BUF_SIZE,
            reconfig_freq=MNIST_RECONFIG_FREQ,
            device=DEVICE,
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_mask_cfg(method: str, is_clf: bool, reconfig_steps: int):
    if method == "FixedMask":
        return M.FixedMask(hidden_size=HIDDEN_SIZE, k=K)
    if method == "NoMask":
        return M.NoMask(hidden_size=HIDDEN_SIZE, k=K)
    if method == "Sigmoid":
        return M.KWTA_Sigmoid(
            hidden_size=HIDDEN_SIZE,
            classification=is_clf,
            k=K,
            reconfig_steps=reconfig_steps,
            reconfig_lr=RECONFIG_LR["Sigmoid"],
            reconfig_n=RECONFIG_N,
        )
    if method == "AnnealSTE":
        return M.KWTA_AnnealedSTE(
            hidden_size=HIDDEN_SIZE,
            classification=is_clf,
            k=K,
            reconfig_steps=reconfig_steps,
            reconfig_lr=RECONFIG_LR["AnnealSTE"],
            reconfig_n=RECONFIG_N,
        )
    if method == "Gumbel":
        return M.KWTA_Gumbel(
            hidden_size=HIDDEN_SIZE,
            classification=is_clf,
            k=K,
            reconfig_steps=reconfig_steps,
            reconfig_lr=RECONFIG_LR["Gumbel"],
            reconfig_n=RECONFIG_N,
            tau_start=GUMBEL_TAU_START,
            tau_end=GUMBEL_TAU_END,
            reg_strength=GUMBEL_REG_STRENGTH,
        )
    raise ValueError(f"Unknown method {method}")


def build_model(input_size: int, output_size: int, seed: int) -> models.NeuroModel:
    torch.manual_seed(seed)
    return models.NeuroModel(input_size, output_size, HIDDEN_SIZE, **MODEL_KWARGS)


def load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"runs": {}}


def save_results(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def run_one(
    method: str,
    budget: int,
    seed: int,
    task_family: str,
    cfg: dict,
    device: str,
) -> dict:
    """Train one (method, budget, seed) and return everything we need."""
    task_set = cfg["build"](seed, device)
    is_clf = getattr(task_set, "is_classification", True)
    train_cfg = cfg["train_cfg"]
    train_cfg.device = device

    mask_cfg = build_mask_cfg(method, is_clf, budget)
    model = build_model(cfg["input_size"], cfg["output_size"], seed)

    t0 = time.time()
    out = block_sequential_train(model, mask_cfg, task_set, train_cfg, verbose=False)
    elapsed = time.time() - t0

    # Sharing matrices
    masks = [out["mask_store"][i].cpu().numpy() for i in range(task_set.n_tasks)]
    model_S = plotting.build_neuron_sharing_matrix(masks)
    gt_S = plotting.build_concept_sharing_matrix(task_set.concept_table())
    sharing_corr = plotting.sharing_correlation(gt_S, model_S)

    return {
        "method": method,
        "budget": budget,
        "seed": seed,
        "task_family": task_family,
        "elapsed_sec": elapsed,
        "summary_stored": metrics.summarise(out["evals"]["stored"], label=method),
        "summary_reconfig": metrics.summarise(out["evals"]["reconfig"], label=method),
        "perf_matrix_stored": out["evals"]["stored"],
        "perf_matrix_reconfig": out["evals"]["reconfig"],
        "head_start": out["head_start"],
        "first_epoch_loss": out["first_epoch_loss"],
        "mask_store": {str(k): v.cpu().tolist() for k, v in out["mask_store"].items()},
        "concept_table": task_set.concept_table().tolist(),
        "model_sharing": model_S.tolist(),
        "gt_sharing": gt_S.tolist(),
        "sharing_corr": sharing_corr,
    }


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------


def _agg_field(seed_runs: list[dict], path: list) -> tuple[float, float]:
    """Mean ± std of a scalar field across a list of runs (path = list of keys)."""
    vals = []
    for r in seed_runs:
        cur = r
        for k in path:
            cur = cur[k]
        if cur is None:
            continue
        vals.append(float(cur))
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _agg_per_task(seed_runs: list[dict], path: list) -> tuple[list, list]:
    """Mean / std per-task vector across seeds."""
    vecs = []
    for r in seed_runs:
        cur = r
        for k in path:
            cur = cur[k]
        if isinstance(cur, list) and cur and isinstance(cur[0], dict):
            # head_start: list of dicts, extract scalar field (last key in path?)
            # Caller should have given us the right path.
            return [], []
        vecs.append(np.array(cur, dtype=float))
    if not vecs:
        return [], []
    n = min(len(v) for v in vecs)
    arr = np.stack([v[:n] for v in vecs])
    return arr.mean(0).tolist(), arr.std(0).tolist()


def aggregate(data: dict) -> None:
    """Recompute data['aggregated'][method][budget] from data['runs']."""
    by_mb: dict[tuple[str, int], list[dict]] = {}
    for r in data["runs"].values():
        key = (r["method"], int(r["budget"]))
        by_mb.setdefault(key, []).append(r)

    agg: dict = {}
    for (method, budget), runs in by_mb.items():
        m_agg = agg.setdefault(method, {})
        # Scalar metrics, mean over seeds
        e = {}
        for proto in ("summary_stored", "summary_reconfig"):
            for field in ("ACC", "FM", "BWT"):
                mean, std = _agg_field(runs, [proto, field])
                e[f"{proto}.{field}"] = {"mean": mean, "std": std}
        # head start (per-task task_loss_with_mask) -- per-task mean/std
        n_tasks = max(len(r["head_start"]) for r in runs)
        head_means, head_stds = [], []
        for ti in range(n_tasks):
            vals = [
                r["head_start"][ti]["task_loss_with_mask"]
                for r in runs
                if ti < len(r["head_start"])
            ]
            head_means.append(float(np.mean(vals)) if vals else float("nan"))
            head_stds.append(float(np.std(vals)) if vals else float("nan"))
        e["head_start_per_task_mean"] = head_means
        e["head_start_per_task_std"] = head_stds
        # First-epoch loss per task
        fe_mean, fe_std = _agg_per_task(runs, ["first_epoch_loss"])
        e["first_epoch_loss_per_task_mean"] = fe_mean
        e["first_epoch_loss_per_task_std"] = fe_std
        # Sharing correlation
        sc_mean, sc_std = _agg_field(runs, ["sharing_corr"])
        e["sharing_corr"] = {"mean": sc_mean, "std": sc_std}
        e["n_seeds"] = len(runs)
        m_agg[str(budget)] = e

    data["aggregated"] = agg


# ---------------------------------------------------------------------------
# Visualisation: per (method, budget) pick a representative seed
# ---------------------------------------------------------------------------


def render_sharing_figs(
    data: dict, task_family: str, fig_root: str, methods: list[str]
) -> None:
    """For each method, pick the largest budget run from seed 0 and dump
    the sharing matrices."""
    out_dir = os.path.join(fig_root, task_family)
    os.makedirs(out_dir, exist_ok=True)
    for r in data["runs"].values():
        if r["task_family"] != task_family:
            continue
        if r["seed"] != 0:
            continue
        if r["method"] not in methods:
            continue
        # take largest budget
    # Group runs
    best: dict[str, dict] = {}
    for r in data["runs"].values():
        if r["task_family"] != task_family or r["seed"] != 0:
            continue
        if r["method"] not in methods:
            continue
        cur = best.get(r["method"])
        if cur is None or int(r["budget"]) > int(cur["budget"]):
            best[r["method"]] = r

    if not best:
        return

    # GT once (taken from any run)
    any_run = next(iter(best.values()))
    gt = np.array(any_run["gt_sharing"])
    plotting.plot_sharing_matrix(
        gt,
        title=f"Ground-truth concept sharing — {task_family}",
        out_path=os.path.join(out_dir, "sharing_gt.pdf"),
    )

    for method, r in sorted(best.items()):
        ms = np.array(r["model_sharing"])
        plotting.plot_sharing_matrix(
            ms,
            title=f"Model neuron sharing — {method} (budget={r['budget']})",
            out_path=os.path.join(out_dir, f"sharing_model_{method}.pdf"),
        )
        plotting.plot_sharing_comparison(
            gt,
            ms,
            title=f"Concept vs neuron sharing — {method} ({task_family})",
            out_path=os.path.join(out_dir, f"sharing_compare_{method}.pdf"),
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=list(TASK_REGISTRY.keys()),
        default=["synthetic"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS + BASELINE_METHODS,
    )
    parser.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGETS,
        help="Reconfig steps to evaluate per method (ignored for baselines).",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument(
        "--include-baselines",
        action="store_true",
        help="Also run FixedMask and NoMask at budget=1 for comparison.",
    )
    parser.add_argument("--no-figs", action="store_true")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Working directory: {os.getcwd()}")
    print(f"Device   : {args.device}")
    print(f"Tasks    : {args.tasks}")
    print(f"Methods  : {args.methods}")
    print(f"Budgets  : {args.budgets}")
    print(f"Seeds    : {args.seeds}")

    for task_family in args.tasks:
        print("TASK:", task_family)
        cfg = TASK_REGISTRY[task_family]
        res_path = os.path.join(RES_DIR, f"{task_family}.json")
        data = load_results(res_path)
        runs = data.setdefault("runs", {})

        all_methods = list(args.methods)
        if args.include_baselines:
            all_methods += BASELINE_METHODS

        for method in all_methods:
            method_budgets = [1] if method in BASELINE_METHODS else args.budgets
            for budget in method_budgets:
                print("BUDGET:", budget)
                for seed in args.seeds:
                    print("SEED:", seed)
                    rkey = f"{method}__b{budget}__s{seed}"
                    if rkey in runs:
                        print(f"  [skip] {task_family} {rkey}")
                        continue
                    print(f"  [run]  {task_family} {rkey}")
                    res = run_one(method, budget, seed, task_family, cfg, args.device)
                    runs[rkey] = res
                    aggregate(data)
                    save_results(data, res_path)

        aggregate(data)
        save_results(data, res_path)
        print(f"  Saved: {res_path}")

        if not args.no_figs:
            render_sharing_figs(data, task_family, FIG_DIR, args.methods)

        # Quick console summary
        print(f"\n  Aggregated for {task_family}:")
        if "aggregated" in data:
            for method, by_budget in sorted(data["aggregated"].items()):
                for bkey, e in sorted(by_budget.items(), key=lambda kv: int(kv[0])):
                    print(
                        f"    {method:<10} budget={bkey:<6} n_seeds={e['n_seeds']}  "
                        f"ACC(stored)={e['summary_stored.ACC']['mean']:.3f}±"
                        f"{e['summary_stored.ACC']['std']:.3f}  "
                        f"BWT(stored)={e['summary_stored.BWT']['mean']:+.3f}  "
                        f"sharing_corr={e['sharing_corr']['mean']:.3f}"
                    )

    print("\nDone.")


if __name__ == "__main__":
    main()
