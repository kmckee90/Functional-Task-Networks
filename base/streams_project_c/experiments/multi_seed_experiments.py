"""
multi_seed_experiments.py — Primary trainer for project_c_3.

Run from project_c_3/:
    uv run python experiments/multi_seed_experiments.py [options]

Options:
    --experiments 1 2 3 4   Which experiments to run (default: all)
    --models KWTA NoMask ... Which models to run per experiment (default: all)
    --seeds 0 1 2 ...        Which seeds (default: 0..7)
    --device cuda            Device (default: auto-detect)
    --no-reconfig-eval       Disable mask-recovery evaluation; use stored masks instead

Key design:
  - Forgetting is evaluated by re-running the mask reconfiguration algorithm on a
    fresh data sample from each previous task (eval_via_reconfig=True).  This tests
    whether the subnetwork can be *recovered* via optimization — the relevant question
    since task identity is unknown at inference time.  The stored training masks are
    saved for visualization only.
  - Results are stored in a seed-keyed JSON dict.  Running a subset of models or
    seeds only updates those entries; existing entries for other models/seeds are
    preserved.
  - All data needed to regenerate figures is stored in the JSON.  Run
    generate_figures.py to regenerate without re-training.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import masks as M
import matplotlib.pyplot as plt
import metrics
import models
import numpy as np
import torch
from tasks import ElectricityELEC2, MNISTShuffled, PermutedMNIST, SyntheticTasks
from trainer import TrainConfig, block_sequential_train, block_sequential_train_ewc
import supsup

# ---------------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = list(range(8))
RES_DIR = "results"
FIG_BASE = "figures"

HIDDEN_SIZE = 1024
GRID_DIM = 32
K = 128

# NeuroModel init kwargs — match the parent project's config.Default values.
# See func_net_spatial/config.py::Default: LAYERS=8, DROPOUT=0.2, INNER_DIM=8.
MODEL_KWARGS = {
    "layers": 8,  # n_layers inside each ParallelMLP neuron
    "inner_size": 8,  # hidden width inside each ParallelMLP neuron
    "dropout_p": 0.2,  # dropout on the per-neuron output vector
}

# Per-task RGB colors for mask tiling (up to 10 tasks)
TASK_COLORS = np.array(
    [
        [0.90, 0.10, 0.10],
        [0.10, 0.72, 0.10],
        [0.15, 0.35, 0.95],
        [0.95, 0.75, 0.00],
        [0.85, 0.10, 0.85],
        [0.00, 0.82, 0.82],
        [1.00, 0.50, 0.00],
        [0.55, 0.00, 0.90],
        [0.00, 0.50, 0.25],
        [0.55, 0.30, 0.00],
    ]
)

# ---------------------------------------------------------------------------
# Per-experiment hyperparameters
# ---------------------------------------------------------------------------

# Exp 1 — Synthetic (aligned to parent project's config.Default)
#   X_DIM=2, Y_DIM=2, N_TASK=3, TASK_SIZE=4, FREQUENCY=8,
#   TRAIN_DURATION=1000, N_EPOCHS=10, TRAIN_LR=3e-4,
#   MASK_PARS: reconfig_steps=1, reconfig_lr=1
EXP1_INPUT_SIZE = 2
EXP1_N_TASKS = 3
EXP1_TASK_SIZE = 4
EXP1_FREQUENCY = 8
EXP1_N_EPOCHS = 1
EXP1_TRAIN_STEPS = 3000
EXP1_BATCH_SIZE = 256
EXP1_EVAL_BATCH_SIZE = 4096
EXP1_RECONFIG_BATCH_SIZE = 256
EXP1_LR = 3e-4
EXP1_RECONFIG_LR = 1.0  # classification (CE-gradient breaks symmetry in 1 step)
EXP1_RECONFIG_STEPS = 1
EXP1_RECONFIG_LR_REG = 0.2
EXP1_RECONFIG_STEPS_REG = 10
EXP1_RECONFIG_FREQ = "task"
EXP1_EWC_LAMBDA = 400.0

# Exp 2 — MNIST Shuffled
EXP2_N_TASKS = 5
EXP2_N_EPOCHS = 5
EXP2_TRAIN_STEPS = 400
EXP2_BATCH_SIZE = 256
EXP2_EVAL_BATCH_SIZE = 1000
EXP2_RECONFIG_BATCH_SIZE = 256
EXP2_LR = 3e-4
EXP2_RECONFIG_LR = 0.2
EXP2_RECONFIG_STEPS = 20
EXP2_RECONFIG_FREQ = "epoch"
EXP2_EWC_LAMBDA = 400.0

# Exp 3 — Permuted MNIST
EXP3_N_TASKS = 10
EXP3_N_EPOCHS = 3
EXP3_TRAIN_STEPS = 400
EXP3_BATCH_SIZE = 256
EXP3_EVAL_BATCH_SIZE = 1000
EXP3_RECONFIG_BATCH_SIZE = 256
EXP3_LR = 3e-4
EXP3_RECONFIG_LR = 0.3
EXP3_RECONFIG_STEPS = 10
EXP3_RECONFIG_FREQ = "epoch"
EXP3_EWC_LAMBDA = 400.0

# Exp 4 — Electricity (ELEC2) real tabular concept drift
# 45,312 half-hourly records split into EXP4_N_TASKS contiguous time
# regimes.  Temporal features dropped per Žliobaitė 2013.
EXP4_N_TASKS = 4
EXP4_N_EPOCHS = 20
EXP4_TRAIN_STEPS = 100
EXP4_BATCH_SIZE = 256
EXP4_EVAL_BATCH_SIZE = 1024
EXP4_RECONFIG_BATCH_SIZE = 256
EXP4_LR = 3e-4
EXP4_RECONFIG_LR = 0.2
EXP4_RECONFIG_STEPS = 10
EXP4_RECONFIG_FREQ = "epoch"
EXP4_EWC_LAMBDA = 400.0

# Valid model names per experiment
EXP1_VALID_MODELS = ["KWTA", "FTN-Fast", "FTN-Slow", "FixedMask", "NoMask", "EWC", "SupSup", "SupSup-Wide"]
EXP2_VALID_MODELS = ["KWTA", "FTN-Fast", "FTN-Slow", "NoMask", "EWC"]
EXP3_VALID_MODELS = ["KWTA", "FTN-Fast", "FTN-Slow", "NoMask", "EWC"]
EXP4_VALID_MODELS = ["KWTA", "FTN-Fast", "FTN-Slow", "FixedMask", "NoMask", "EWC"]


# ---------------------------------------------------------------------------
# Helpers: configurers and models
# ---------------------------------------------------------------------------


def _kwta(is_clf: bool, reconfig_steps: int, reconfig_lr: float) -> M.KWTAOnly:
    return M.KWTAOnly(
        hidden_size=HIDDEN_SIZE,
        grid_dim=GRID_DIM,
        classification=is_clf,
        k=K,
        reconfig_steps=reconfig_steps,
        reconfig_lr=reconfig_lr,
    )


def _ftn_fast(
    is_clf: bool, reconfig_steps: int, reconfig_lr: float
) -> M.SmoothKWTAFast:
    return M.SmoothKWTAFast(
        hidden_size=HIDDEN_SIZE,
        grid_dim=GRID_DIM,
        classification=is_clf,
        k=K,
        reconfig_steps=reconfig_steps,
        reconfig_lr=reconfig_lr,
    )


def _ftn_slow(
    is_clf: bool, reconfig_steps: int, reconfig_lr: float
) -> M.SmoothKWTASlow:
    return M.SmoothKWTASlow(
        hidden_size=HIDDEN_SIZE,
        grid_dim=GRID_DIM,
        classification=is_clf,
        k=K,
        reconfig_steps=reconfig_steps,
        reconfig_lr=reconfig_lr,
    )


def _nomask() -> M.NoMask:
    return M.NoMask(hidden_size=HIDDEN_SIZE, grid_dim=GRID_DIM, k=K)


def _fixedmask() -> M.FixedMask:
    return M.FixedMask(hidden_size=HIDDEN_SIZE, grid_dim=GRID_DIM, k=K)


def _masked_model(input_size: int, output_size: int, seed: int) -> models.MaskedMLP:
    torch.manual_seed(seed)
    return models.MaskedMLP(input_size, output_size, HIDDEN_SIZE, **MODEL_KWARGS)


def _nomask_model(input_size: int, output_size: int, seed: int) -> models.NoMaskMLP:
    torch.manual_seed(seed)
    return models.NoMaskMLP(input_size, output_size, HIDDEN_SIZE, **MODEL_KWARGS)


# ---------------------------------------------------------------------------
# Results: non-destructive JSON merge
# ---------------------------------------------------------------------------


def load_results(path: str) -> dict:
    """Load existing results JSON, or return empty structure."""
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"seeds": {}}


def merge_seed_results(existing: dict, seed: int, new_method_results: dict) -> None:
    """Merge new_method_results into existing["seeds"][seed] non-destructively."""
    key = str(seed)
    if key not in existing["seeds"]:
        existing["seeds"][key] = {}
    existing["seeds"][key].update(new_method_results)


def _agg_summary(seed_list, seeds_data, method, key) -> dict:
    summaries = [seeds_data[s][method][key] for s in seed_list]
    return {
        "label": method,
        "ACC": float(np.mean([s["ACC"] for s in summaries])),
        "ACC_std": float(np.std([s["ACC"] for s in summaries])),
        "FM": float(np.mean([s["FM"] for s in summaries])),
        "FM_std": float(np.std([s["FM"] for s in summaries])),
        "BWT": float(np.mean([s["BWT"] for s in summaries])),
        "BWT_std": float(np.std([s["BWT"] for s in summaries])),
    }


def _agg_pm(seed_list, seeds_data, method, key) -> tuple[list, list]:
    pms = [seeds_data[s][method][key] for s in seed_list]
    n_rows = len(pms[0])
    mean_pm, std_pm = [], []
    for i in range(n_rows):
        n_cols = len(pms[0][i])
        mean_pm.append(
            [float(np.mean([pm[i][j] for pm in pms])) for j in range(n_cols)]
        )
        std_pm.append([float(np.std([pm[i][j] for pm in pms])) for j in range(n_cols)])
    return mean_pm, std_pm


def recompute_aggregates(data: dict) -> None:
    """
    Recompute 'aggregated', 'pm_mean', 'pm_std' in-place from data["seeds"].

    Each entry is keyed by eval protocol: {"stored": ..., "reconfig": ...}.
    Only includes methods present in ALL seeds (to avoid partial-seed bias).
    """
    seeds_data = data["seeds"]
    if not seeds_data:
        return

    all_methods: set[str] | None = None
    for seed_result in seeds_data.values():
        if all_methods is None:
            all_methods = set(seed_result.keys())
        else:
            all_methods &= set(seed_result.keys())
    if not all_methods:
        return

    seed_list = sorted(seeds_data.keys(), key=int)
    agg, pm_mean, pm_std = {}, {}, {}

    for method in sorted(all_methods):
        agg[method] = {
            "stored": _agg_summary(seed_list, seeds_data, method, "summary_stored"),
            "reconfig": _agg_summary(seed_list, seeds_data, method, "summary_reconfig"),
        }
        m_s, s_s = _agg_pm(seed_list, seeds_data, method, "perf_matrix_stored")
        m_r, s_r = _agg_pm(seed_list, seeds_data, method, "perf_matrix_reconfig")
        pm_mean[method] = {"stored": m_s, "reconfig": m_r}
        pm_std[method] = {"stored": s_s, "reconfig": s_r}

    data["aggregated"] = agg
    data["pm_mean"] = pm_mean
    data["pm_std"] = pm_std


def save_results(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Mask tiling visualization
# ---------------------------------------------------------------------------


def _render_mask_rgb(mask_store: dict, grid_dim: int) -> np.ndarray:
    n = grid_dim * grid_dim
    rgb = np.zeros((n, 3), dtype=np.float32)
    for task_id, mask in mask_store.items():
        m = np.array(mask, dtype=np.float32)[:n]
        color = TASK_COLORS[int(task_id) % len(TASK_COLORS)]
        rgb += np.outer(m, color)
    return rgb.clip(0.0, 1.0).reshape(grid_dim, grid_dim, 3)


def save_mask_tiling(
    mask_stores: list[dict],
    grid_dim: int,
    out_path: str,
    n_cols: int = 4,
    title: str = "",
) -> None:
    tiles = [_render_mask_rgb(ms, grid_dim) for ms in mask_stores]
    n = len(tiles)
    n_rows = (n + n_cols - 1) // n_cols
    h, w = grid_dim, grid_dim
    delim = 1

    canvas_h = n_rows * h + (n_rows - 1) * delim
    canvas_w = n_cols * w + (n_cols - 1) * delim
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.float32)

    for i, tile in enumerate(tiles):
        r = i // n_cols
        c = i % n_cols
        y0 = r * (h + delim)
        x0 = c * (w + delim)
        canvas[y0 : y0 + h, x0 : x0 + w] = tile

    scale = 2
    fig, ax = plt.subplots(
        figsize=(n_cols * w * scale / 72, n_rows * h * scale / 72), dpi=72
    )
    ax.imshow(canvas, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8)
    fig.tight_layout(pad=0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved tiling: {out_path}")


# ---------------------------------------------------------------------------
# Experiment 1 — Synthetic
# ---------------------------------------------------------------------------


def run_exp1_seed(
    seed: int,
    is_clf: bool,
    run_models: list[str],
    device: str = DEVICE,
    eval_via_reconfig: bool = True,
) -> dict:
    out_size = 2 if is_clf else 1
    task_set = SyntheticTasks(
        input_size=EXP1_INPUT_SIZE,
        output_size=out_size,
        n_tasks=EXP1_N_TASKS,
        task_size=EXP1_TASK_SIZE,
        frequency=EXP1_FREQUENCY,
        classification=is_clf,
        seed=seed,
        device=device,
        test_size=EXP1_EVAL_BATCH_SIZE,
    )
    train_cfg = TrainConfig(
        n_epochs=EXP1_N_EPOCHS,
        train_steps=EXP1_TRAIN_STEPS,
        batch_size=EXP1_BATCH_SIZE,
        eval_batch_size=EXP1_EVAL_BATCH_SIZE,
        reconfig_batch_size=EXP1_RECONFIG_BATCH_SIZE,
        lr=EXP1_LR,
        device=device,
        reconfig_freq=EXP1_RECONFIG_FREQ,
    )
    if is_clf:
        rs, rl = int(EXP1_RECONFIG_STEPS), EXP1_RECONFIG_LR
    else:
        rs, rl = int(EXP1_RECONFIG_STEPS_REG), EXP1_RECONFIG_LR_REG
    results = {}

    cfg_map = {
        "KWTA": lambda: _kwta(is_clf, rs, rl),
        "FTN-Fast": lambda: _ftn_fast(is_clf, rs, rl),
        "FTN-Slow": lambda: _ftn_slow(is_clf, rs, rl),
        "FixedMask": _fixedmask,
        "NoMask": _nomask,
    }
    for name in run_models:
        if name in cfg_map:
            model = _masked_model(EXP1_INPUT_SIZE, out_size, seed)
            evals, mask_store, _ = block_sequential_train(
                model,
                cfg_map[name](),
                task_set,
                train_cfg,
                verbose=False,
                eval_via_reconfig=eval_via_reconfig,
            )
            results[name] = {
                "summary_stored": metrics.summarise(evals["stored"], label=name),
                "summary_reconfig": metrics.summarise(evals["reconfig"], label=name),
                "perf_matrix_stored": evals["stored"],
                "perf_matrix_reconfig": evals["reconfig"],
                "mask_store": {str(k): v.cpu().tolist() for k, v in mask_store.items()},
            }
        elif name == "EWC":
            model = _nomask_model(EXP1_INPUT_SIZE, out_size, seed)
            evals, _ = block_sequential_train_ewc(
                model,
                task_set,
                train_cfg,
                ewc_lambda=EXP1_EWC_LAMBDA,
                k=K,
                verbose=False,
            )
            results["EWC"] = {
                "summary_stored": metrics.summarise(evals["stored"], label="EWC"),
                "summary_reconfig": metrics.summarise(evals["reconfig"], label="EWC"),
                "perf_matrix_stored": evals["stored"],
                "perf_matrix_reconfig": evals["reconfig"],
                "mask_store": {},
            }
        elif name in ("SupSup", "SupSup-Wide"):
            if name == "SupSup-Wide":
                ss_kwargs = {**MODEL_KWARGS, "inner_size": MODEL_KWARGS["inner_size"] * 2}
                signed = True
            else:
                ss_kwargs = MODEL_KWARGS
                signed = False
            ss_model = supsup.make_supsup_model(
                EXP1_INPUT_SIZE, out_size, HIDDEN_SIZE, seed, ss_kwargs,
                signed_init=signed,
            )
            evals, mask_store = supsup.block_sequential_supsup(
                ss_model, task_set,
                is_clf=is_clf, k=K,
                train_steps=EXP1_TRAIN_STEPS,
                batch_size=EXP1_BATCH_SIZE,
                reconfig_batch_size=EXP1_RECONFIG_BATCH_SIZE,
                eval_batch_size=EXP1_EVAL_BATCH_SIZE,
                lr=EXP1_LR,
                device=device,
                eval_via_reconfig=eval_via_reconfig,
            )
            results[name] = {
                "summary_stored": metrics.summarise(evals["stored"], label=name),
                "summary_reconfig": metrics.summarise(evals["reconfig"], label=name),
                "perf_matrix_stored": evals["stored"],
                "perf_matrix_reconfig": evals["reconfig"],
                "mask_store": {str(k_): v.tolist() for k_, v in mask_store.items()},
            }
    return results


def _exp1_worker(args):
    seed, is_clf, gpu_id, run_models, eval_via_reconfig = args
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    return seed, run_exp1_seed(
        seed, is_clf, run_models, device=device, eval_via_reconfig=eval_via_reconfig
    )


def exp1(
    seeds: list[int],
    run_models: list[str],
    eval_via_reconfig: bool = True,
    variants: list[str] | None = None,
) -> None:
    print(
        f"\n{'#' * 60}\n  Experiment 1 — Synthetic ({len(seeds)} seeds, parallel)\n{'#' * 60}"
    )

    all_variants = [("clf", True), ("reg", False)]
    if variants:
        all_variants = [(v, c) for v, c in all_variants if v in variants]

    for variant, is_clf in all_variants:
        res_path = f"{RES_DIR}/exp1_synthetic_{variant}.json"
        data = load_results(res_path)

        print(f"  Launching {len(seeds)} workers ({variant}) ...")
        worker_args = [
            (
                seed,
                is_clf,
                i % max(1, torch.cuda.device_count()),
                run_models,
                eval_via_reconfig,
            )
            for i, seed in enumerate(seeds)
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(seeds)) as pool:
            results_list = pool.map(_exp1_worker, worker_args)

        for seed, seed_results in results_list:
            merge_seed_results(data, seed, seed_results)

        recompute_aggregates(data)
        save_results(data, res_path)

        # Print aggregated summary
        if "aggregated" in data:
            variant_name = "Classification" if is_clf else "Regression"
            print(
                f"\n  Exp1-{variant} ({variant_name}) aggregated (over {len(data['seeds'])} seeds):"
            )
            print(
                f"  {'Method':<14} {'protocol':<10} {'ACC':>12} {'FM':>12} {'BWT':>12}"
            )
            for m, protocols in data["aggregated"].items():
                for proto, a in protocols.items():
                    print(
                        f"  {m:<14} {proto:<10} "
                        f"{a['ACC']:>6.3f}±{a['ACC_std']:.3f}  "
                        f"{a['FM']:>6.3f}±{a['FM_std']:.3f}  "
                        f"{a['BWT']:>6.3f}±{a['BWT_std']:.3f}"
                    )


# ---------------------------------------------------------------------------
# Experiment 2 — MNIST Shuffled Labels
# ---------------------------------------------------------------------------


def run_exp2_seed(
    seed: int,
    task_set,
    run_models: list[str],
    eval_via_reconfig: bool = True,
) -> dict:
    train_cfg = TrainConfig(
        n_epochs=EXP2_N_EPOCHS,
        train_steps=EXP2_TRAIN_STEPS,
        batch_size=EXP2_BATCH_SIZE,
        eval_batch_size=EXP2_EVAL_BATCH_SIZE,
        reconfig_batch_size=EXP2_RECONFIG_BATCH_SIZE,
        lr=EXP2_LR,
        device=DEVICE,
        reconfig_freq=EXP2_RECONFIG_FREQ,
    )
    rs, rl = int(EXP2_RECONFIG_STEPS), EXP2_RECONFIG_LR
    results = {}

    cfg_map = {
        "KWTA": lambda: _kwta(True, rs, rl),
        "FTN-Fast": lambda: _ftn_fast(True, rs, rl),
        "FTN-Slow": lambda: _ftn_slow(True, rs, rl),
        "NoMask": _nomask,
    }
    for name in run_models:
        if name in cfg_map:
            model = _masked_model(784, 10, seed)
            evals, mask_store, _ = block_sequential_train(
                model,
                cfg_map[name](),
                task_set,
                train_cfg,
                verbose=False,
                eval_via_reconfig=eval_via_reconfig,
            )
            results[name] = {
                "summary_stored": metrics.summarise(evals["stored"], label=name),
                "summary_reconfig": metrics.summarise(evals["reconfig"], label=name),
                "perf_matrix_stored": evals["stored"],
                "perf_matrix_reconfig": evals["reconfig"],
                "mask_store": {str(k): v.cpu().tolist() for k, v in mask_store.items()},
            }
        elif name == "EWC":
            model = _nomask_model(784, 10, seed)
            evals, _ = block_sequential_train_ewc(
                model,
                task_set,
                train_cfg,
                ewc_lambda=EXP2_EWC_LAMBDA,
                k=K,
                verbose=False,
            )
            results["EWC"] = {
                "summary_stored": metrics.summarise(evals["stored"], label="EWC"),
                "summary_reconfig": metrics.summarise(evals["reconfig"], label="EWC"),
                "perf_matrix_stored": evals["stored"],
                "perf_matrix_reconfig": evals["reconfig"],
                "mask_store": {},
            }
    return results


def exp2(
    seeds: list[int], run_models: list[str], eval_via_reconfig: bool = True
) -> None:
    print(
        f"\n{'#' * 60}\n  Experiment 2 — MNIST Shuffled ({len(seeds)} seeds)\n{'#' * 60}"
    )
    res_path = f"{RES_DIR}/exp2_mnist_shuffled.json"
    data = load_results(res_path)

    task_set = MNISTShuffled(n_tasks=EXP2_N_TASKS, seed=0, data_dir="./data")
    task_set._load()

    for seed in seeds:
        print(f"  seed={seed}")
        seed_results = run_exp2_seed(seed, task_set, run_models, eval_via_reconfig)
        merge_seed_results(data, seed, seed_results)

    recompute_aggregates(data)
    save_results(data, res_path)

    if "aggregated" in data:
        print("\n  Exp2 aggregated:")
        for m, protocols in data["aggregated"].items():
            for proto, a in protocols.items():
                print(
                    f"  {m:<14} {proto:<10} ACC={a['ACC']:.3f}±{a['ACC_std']:.3f}  FM={a['FM']:.3f}±{a['FM_std']:.3f}"
                )


# ---------------------------------------------------------------------------
# Experiment 3 — Permuted MNIST
# ---------------------------------------------------------------------------


def run_exp3_seed(
    seed: int,
    task_set,
    run_models: list[str],
    eval_via_reconfig: bool = True,
) -> dict:
    train_cfg = TrainConfig(
        n_epochs=EXP3_N_EPOCHS,
        train_steps=EXP3_TRAIN_STEPS,
        batch_size=EXP3_BATCH_SIZE,
        eval_batch_size=EXP3_EVAL_BATCH_SIZE,
        reconfig_batch_size=EXP3_RECONFIG_BATCH_SIZE,
        lr=EXP3_LR,
        device=DEVICE,
        reconfig_freq=EXP3_RECONFIG_FREQ,
    )
    rs, rl = int(EXP3_RECONFIG_STEPS), EXP3_RECONFIG_LR
    results = {}

    cfg_map = {
        "KWTA": lambda: _kwta(True, rs, rl),
        "FTN-Fast": lambda: _ftn_fast(True, rs, rl),
        "FTN-Slow": lambda: _ftn_slow(True, rs, rl),
        "NoMask": _nomask,
    }
    for name in run_models:
        if name in cfg_map:
            model = _masked_model(784, 10, seed)
            evals, mask_store, _ = block_sequential_train(
                model,
                cfg_map[name](),
                task_set,
                train_cfg,
                verbose=False,
                eval_via_reconfig=eval_via_reconfig,
            )
            results[name] = {
                "summary_stored": metrics.summarise(evals["stored"], label=name),
                "summary_reconfig": metrics.summarise(evals["reconfig"], label=name),
                "perf_matrix_stored": evals["stored"],
                "perf_matrix_reconfig": evals["reconfig"],
                "mask_store": {str(k): v.cpu().tolist() for k, v in mask_store.items()},
            }
        elif name == "EWC":
            model = _nomask_model(784, 10, seed)
            evals, _ = block_sequential_train_ewc(
                model,
                task_set,
                train_cfg,
                ewc_lambda=EXP3_EWC_LAMBDA,
                k=K,
                verbose=False,
            )
            results["EWC"] = {
                "summary_stored": metrics.summarise(evals["stored"], label="EWC"),
                "summary_reconfig": metrics.summarise(evals["reconfig"], label="EWC"),
                "perf_matrix_stored": evals["stored"],
                "perf_matrix_reconfig": evals["reconfig"],
                "mask_store": {},
            }
    return results


def exp3(
    seeds: list[int], run_models: list[str], eval_via_reconfig: bool = True
) -> None:
    print(
        f"\n{'#' * 60}\n  Experiment 3 — Permuted MNIST ({len(seeds)} seeds)\n{'#' * 60}"
    )
    res_path = f"{RES_DIR}/exp3_permuted_mnist.json"
    data = load_results(res_path)

    task_set = PermutedMNIST(n_tasks=EXP3_N_TASKS, seed=0, data_dir="./data")
    task_set._load()

    for seed in seeds:
        print(f"  seed={seed}")
        seed_results = run_exp3_seed(seed, task_set, run_models, eval_via_reconfig)
        merge_seed_results(data, seed, seed_results)

    recompute_aggregates(data)
    save_results(data, res_path)

    if "aggregated" in data:
        print("\n  Exp3 aggregated:")
        for m, protocols in data["aggregated"].items():
            for proto, a in protocols.items():
                print(
                    f"  {m:<14} {proto:<10} ACC={a['ACC']:.3f}±{a['ACC_std']:.3f}  FM={a['FM']:.3f}±{a['FM_std']:.3f}"
                )


# ---------------------------------------------------------------------------
# Experiment 4 — Seasonal Energy Demand
# ---------------------------------------------------------------------------


def run_exp4_seed(
    seed: int,
    run_models: list[str],
    eval_via_reconfig: bool = True,
) -> dict:
    task_set = ElectricityELEC2(n_tasks=EXP4_N_TASKS, seed=seed)
    train_cfg = TrainConfig(
        n_epochs=EXP4_N_EPOCHS,
        train_steps=EXP4_TRAIN_STEPS,
        batch_size=EXP4_BATCH_SIZE,
        eval_batch_size=EXP4_EVAL_BATCH_SIZE,
        reconfig_batch_size=EXP4_RECONFIG_BATCH_SIZE,
        lr=EXP4_LR,
        device=DEVICE,
        reconfig_freq=EXP4_RECONFIG_FREQ,
    )
    rs, rl = int(EXP4_RECONFIG_STEPS), EXP4_RECONFIG_LR
    results = {}

    cfg_map = {
        "KWTA": lambda: _kwta(True, rs, rl),
        "FTN-Fast": lambda: _ftn_fast(True, rs, rl),
        "FTN-Slow": lambda: _ftn_slow(True, rs, rl),
        "FixedMask": _fixedmask,
        "NoMask": _nomask,
    }
    for name in run_models:
        if name in cfg_map:
            model = _masked_model(5, 2, seed)
            evals, mask_store, _ = block_sequential_train(
                model,
                cfg_map[name](),
                task_set,
                train_cfg,
                verbose=False,
                eval_via_reconfig=eval_via_reconfig,
            )
            results[name] = {
                "summary_stored": metrics.summarise(evals["stored"], label=name),
                "summary_reconfig": metrics.summarise(evals["reconfig"], label=name),
                "perf_matrix_stored": evals["stored"],
                "perf_matrix_reconfig": evals["reconfig"],
                "mask_store": {str(k): v.cpu().tolist() for k, v in mask_store.items()},
            }
        elif name == "EWC":
            model = _nomask_model(5, 2, seed)
            evals, _ = block_sequential_train_ewc(
                model,
                task_set,
                train_cfg,
                ewc_lambda=EXP4_EWC_LAMBDA,
                k=K,
                verbose=False,
            )
            results["EWC"] = {
                "summary_stored": metrics.summarise(evals["stored"], label="EWC"),
                "summary_reconfig": metrics.summarise(evals["reconfig"], label="EWC"),
                "perf_matrix_stored": evals["stored"],
                "perf_matrix_reconfig": evals["reconfig"],
                "mask_store": {},
            }
    return results


def exp4(
    seeds: list[int], run_models: list[str], eval_via_reconfig: bool = True
) -> None:
    print(
        f"\n{'#' * 60}\n  Experiment 4 — Electricity ELEC2 ({len(seeds)} seeds)\n{'#' * 60}"
    )
    res_path = f"{RES_DIR}/exp4_elec2.json"
    data = load_results(res_path)

    for seed in seeds:
        print(f"  seed={seed}")
        seed_results = run_exp4_seed(seed, run_models, eval_via_reconfig)
        merge_seed_results(data, seed, seed_results)

    recompute_aggregates(data)
    save_results(data, res_path)

    if "aggregated" in data:
        print("\n  Exp4 aggregated:")
        for m, protocols in data["aggregated"].items():
            for proto, a in protocols.items():
                print(
                    f"  {m:<14} {proto:<10} ACC={a['ACC']:.3f}±{a['ACC_std']:.3f}  FM={a['FM']:.3f}±{a['FM_std']:.3f}"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="project_c_3 multi-seed continual learning experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all experiments, all models, all seeds:
  python experiments/multi_seed_experiments.py

  # Run only experiments 1 and 2 with KWTA and EWC:
  python experiments/multi_seed_experiments.py --experiments 1 2 --models KWTA EWC

  # Run only seeds 0-3 for experiment 3:
  python experiments/multi_seed_experiments.py --experiments 3 --seeds 0 1 2 3

  # Evaluate using stored masks instead of mask recovery:
  python experiments/multi_seed_experiments.py --no-reconfig-eval
""",
    )
    parser.add_argument(
        "--experiments",
        "-e",
        nargs="+",
        choices=["1", "2", "3", "4"],
        default=["1", "2", "3", "4"],
        help="Which experiments to run (default: all)",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        choices=["KWTA", "FTN-Fast", "FTN-Slow", "NoMask", "FixedMask", "EWC", "SupSup", "SupSup-Wide"],
        default=None,
        help="Which models to run (default: all valid for each experiment)",
    )
    parser.add_argument(
        "--seeds",
        "-s",
        nargs="+",
        type=int,
        default=SEEDS,
        help=f"Which seeds to run (default: {SEEDS})",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help=f"Compute device (default: {DEVICE})",
    )
    parser.add_argument(
        "--exp1-variants",
        nargs="+",
        choices=["clf", "reg"],
        default=None,
        help="Restrict Exp1 to a subset of variants (default: both clf and reg)",
    )
    parser.add_argument(
        "--no-reconfig-eval",
        action="store_true",
        help=(
            "Use stored training masks for evaluation instead of re-running "
            "mask reconfiguration.  Disables the mask-recovery forgetting test."
        ),
    )
    args = parser.parse_args()

    DEVICE = args.device
    eval_via_reconfig = not args.no_reconfig_eval

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(RES_DIR, exist_ok=True)
    print(f"Working directory: {os.getcwd()}")
    print(f"Device:  {DEVICE}")
    print(f"Seeds:   {args.seeds}")
    print(f"Experiments: {args.experiments}")
    print(f"Models:  {args.models or '(all valid per experiment)'}")
    print(f"Eval via mask recovery: {eval_via_reconfig}")

    def _models_for(valid: list[str]) -> list[str]:
        if args.models is None:
            return valid
        return [m for m in args.models if m in valid]

    if "1" in args.experiments:
        exp1(
            args.seeds,
            _models_for(EXP1_VALID_MODELS),
            eval_via_reconfig,
            variants=args.exp1_variants,
        )

    if "2" in args.experiments:
        exp2(args.seeds, _models_for(EXP2_VALID_MODELS), eval_via_reconfig)

    if "3" in args.experiments:
        exp3(args.seeds, _models_for(EXP3_VALID_MODELS), eval_via_reconfig)

    if "4" in args.experiments:
        exp4(args.seeds, _models_for(EXP4_VALID_MODELS), eval_via_reconfig)

    print("\nDone.")
