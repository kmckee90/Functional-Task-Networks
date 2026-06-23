"""
run_kt_exp.py — Generic parallel runner for the post-exp1 sweeps
(orthogonality, reconfig frequency, K, reconfig batch size).

All sweeps use the *winner* of the exp1 head-start search as the fixed
baseline and vary a single axis:

    AnnealSTE, budget=30, K=512, reconfig_n=256, reconfig_freq=batch,
    correlated tasks (synthetic), 8 seeds.

Each (axis_value, seed) cell is one independent CUDA job; jobs are
distributed across workers (one process per worker, pinned to a CUDA
device).  Results are written to ``results/<exp_tag>.json`` and figures
to ``figures/<exp_tag>/``.

Run:

    CUDA_VISIBLE_DEVICES=2,3,4,5 uv run python experiments/run_kt_exp.py \\
        --exp exp2 --workers-per-gpu 4

Each --exp value selects one axis:

    exp2 — task orthogonality {correlated, orthogonal}
    exp3 — reconfig frequency {batch, task}
    exp4 — KWTA winners K {512, 128, 32, 8}
    exp5 — reconfig batch size {256, 64, 16, 4}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.multiprocessing as mp

import masks as M
import metrics
import models
import plotting
from tasks import SyntheticTasks
from trainer import TrainConfig, block_sequential_train

import run_kt as rk  # type: ignore  # sibling module; reuse load_results/save_results

# ---------------------------------------------------------------------------
# Fixed condition ("winner of exp1")
# ---------------------------------------------------------------------------

DEFAULT_METHOD = "AnnealSTE"
DEFAULT_BUDGET = 30
DEFAULT_K = 512
DEFAULT_RECONFIG_N = 256
DEFAULT_RECONFIG_FREQ = "batch"
DEFAULT_ORTHOGONAL = False
DEFAULT_HIDDEN_SIZE = 512 * 16  # 8192 — fixed across all axis values

DEFAULT_SEEDS = list(range(8))

# ---------------------------------------------------------------------------
# Axis configs
# ---------------------------------------------------------------------------

EXP_CONFIGS: dict[str, dict] = {
    "exp2": {
        "axis": "orthogonal",
        "values": [False, True],
        "label": lambda v: ("correlated" if not v else "orthogonal"),
    },
    "exp3": {
        "axis": "reconfig_freq",
        "values": ["batch", "task"],
        "label": lambda v: v,
    },
    "exp4": {
        "axis": "k",
        "values": [512, 128, 32, 8],
        "label": lambda v: f"k{int(v)}",
    },
    "exp5": {
        "axis": "reconfig_n",
        "values": [256, 64, 16, 4],
        "label": lambda v: f"rn{int(v)}",
    },
}

# ---------------------------------------------------------------------------
# Synthetic-task config (matches run_kt.py defaults)
# ---------------------------------------------------------------------------

SYNTH_INPUT_SIZE = 2
SYNTH_OUTPUT_SIZE = 2
SYNTH_N_TASKS = 12
SYNTH_N_COMPONENTS = 6  # used when correlated=True
SYNTH_TASK_SIZE = 1
SYNTH_FREQUENCY = 8
SYNTH_CLASSIFICATION = True
SYNTH_TEST_SIZE = 2048

# Default train_steps is now 1000 (matched to the t=1000 exp1 rerun).
SYNTH_TRAIN_CFG = TrainConfig(
    n_epochs=1,
    train_steps=1000,
    batch_size=256,
    eval_batch_size=2048,
    reconfig_batch_size=DEFAULT_RECONFIG_N,  # default; overridden when axis=reconfig_n
    lr=3e-4,
    use_replay=True,
    replay_n_per_step=2,
    replay_buf_size=200,
    reconfig_freq=DEFAULT_RECONFIG_FREQ,
    device="cuda",
)

MODEL_LAYERS = 8
MODEL_INNER_SIZE = 8
MODEL_DROPOUT_P = 0.2

# Gumbel hyperparams (used only if method ever switches; here we stay on AnnealSTE)
GUMBEL_TAU_START = 1.0
GUMBEL_TAU_END = 0.1
GUMBEL_REG_STRENGTH = 1.0


# ---------------------------------------------------------------------------
# Job runner — single (axis_value, seed) cell
# ---------------------------------------------------------------------------


def _build_synth(
    seed: int, device: str, *, orthogonal: bool
) -> SyntheticTasks:
    if orthogonal:
        n_components = 2 * SYNTH_N_TASKS
    else:
        n_components = SYNTH_N_COMPONENTS
    return SyntheticTasks(
        input_size=SYNTH_INPUT_SIZE,
        output_size=SYNTH_OUTPUT_SIZE,
        n_tasks=SYNTH_N_TASKS,
        n_components=n_components,
        task_size=SYNTH_TASK_SIZE,
        frequency=SYNTH_FREQUENCY,
        classification=SYNTH_CLASSIFICATION,
        seed=seed,
        device=device,
        test_size=SYNTH_TEST_SIZE,
        orthogonal=orthogonal,
    )


def _build_mask(method: str, k: int, reconfig_n: int, budget: int):
    if method == "AnnealSTE":
        return M.KWTA_AnnealedSTE(
            hidden_size=DEFAULT_HIDDEN_SIZE,
            classification=True,
            k=k,
            reconfig_steps=budget,
            reconfig_lr=0.3,
            reconfig_n=reconfig_n,
        )
    if method == "Sigmoid":
        return M.KWTA_Sigmoid(
            hidden_size=DEFAULT_HIDDEN_SIZE,
            classification=True,
            k=k,
            reconfig_steps=budget,
            reconfig_lr=0.3,
            reconfig_n=reconfig_n,
        )
    if method == "Gumbel":
        return M.KWTA_Gumbel(
            hidden_size=DEFAULT_HIDDEN_SIZE,
            classification=True,
            k=k,
            reconfig_steps=budget,
            reconfig_lr=0.2,
            reconfig_n=reconfig_n,
            tau_start=GUMBEL_TAU_START,
            tau_end=GUMBEL_TAU_END,
            reg_strength=GUMBEL_REG_STRENGTH,
        )
    raise ValueError(method)


def _build_model(seed: int) -> models.NeuroModel:
    torch.manual_seed(seed)
    return models.NeuroModel(
        SYNTH_INPUT_SIZE,
        SYNTH_OUTPUT_SIZE,
        DEFAULT_HIDDEN_SIZE,
        layers=MODEL_LAYERS,
        inner_size=MODEL_INNER_SIZE,
        dropout_p=MODEL_DROPOUT_P,
    )


def run_one(
    *,
    seed: int,
    axis: str,
    axis_value,
    device: str,
) -> dict:
    """Run one (axis_value, seed) on the synthetic family."""
    # Derive overrides from the axis
    orthogonal = DEFAULT_ORTHOGONAL
    reconfig_freq = DEFAULT_RECONFIG_FREQ
    k = DEFAULT_K
    reconfig_n = DEFAULT_RECONFIG_N

    if axis == "orthogonal":
        orthogonal = bool(axis_value)
    elif axis == "reconfig_freq":
        reconfig_freq = str(axis_value)
    elif axis == "k":
        k = int(axis_value)
    elif axis == "reconfig_n":
        reconfig_n = int(axis_value)
    else:
        raise ValueError(axis)

    train_cfg = replace(
        SYNTH_TRAIN_CFG,
        reconfig_freq=reconfig_freq,
        reconfig_batch_size=reconfig_n,
        device=device,
    )

    task_set = _build_synth(seed, device, orthogonal=orthogonal)
    mask_cfg = _build_mask(DEFAULT_METHOD, k, reconfig_n, DEFAULT_BUDGET)
    model = _build_model(seed)

    t0 = time.time()
    out = block_sequential_train(model, mask_cfg, task_set, train_cfg, verbose=False)
    elapsed = time.time() - t0

    masks_arr = [out["mask_store"][i].cpu().numpy() for i in range(task_set.n_tasks)]
    model_S = plotting.build_neuron_sharing_matrix(masks_arr)
    gt_S = plotting.build_concept_sharing_matrix(task_set.concept_table())
    sharing_corr = plotting.sharing_correlation(gt_S, model_S)

    return {
        "method": DEFAULT_METHOD,
        "budget": DEFAULT_BUDGET,
        "seed": seed,
        "task_family": "synthetic",
        "axis": axis,
        "axis_value": axis_value,
        "k": k,
        "reconfig_n": reconfig_n,
        "reconfig_freq": reconfig_freq,
        "orthogonal": orthogonal,
        "elapsed_sec": elapsed,
        "summary_stored": metrics.summarise(out["evals"]["stored"], label=DEFAULT_METHOD),
        "summary_reconfig": metrics.summarise(out["evals"]["reconfig"], label=DEFAULT_METHOD),
        "perf_matrix_stored": out["evals"]["stored"],
        "perf_matrix_reconfig": out["evals"]["reconfig"],
        "head_start": out["head_start"],
        "first_epoch_loss": out["first_epoch_loss"],
        "concept_table": task_set.concept_table().tolist(),
        "model_sharing": model_S.tolist(),
        "gt_sharing": gt_S.tolist(),
        "sharing_corr": sharing_corr,
    }


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------


def _worker(worker_id: int, device: str, job_q, res_q) -> None:
    if device.startswith("cuda"):
        try:
            torch.cuda.set_device(device)
        except Exception as e:
            print(f"[worker {worker_id}] set_device {device} failed: {e}", flush=True)
    while True:
        job = job_q.get()
        if job is None:
            return
        seed, axis, axis_value, rkey = job
        t0 = time.time()
        try:
            res = run_one(
                seed=seed, axis=axis, axis_value=axis_value, device=device,
            )
            res_q.put((rkey, res, None))
            print(
                f"[worker {worker_id} {device}] done {rkey} "
                f"in {time.time()-t0:.1f}s",
                flush=True,
            )
        except Exception:
            err = traceback.format_exc()
            res_q.put((rkey, None, err))
            print(
                f"[worker {worker_id} {device}] FAILED {rkey}\n{err}", flush=True,
            )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _agg_field(runs, path):
    vals = []
    for r in runs:
        cur = r
        for k in path:
            cur = cur[k]
        if cur is None:
            continue
        vals.append(float(cur))
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def aggregate(data: dict, exp_cfg: dict) -> None:
    by_val: dict[str, list[dict]] = {}
    for r in data["runs"].values():
        v = r["axis_value"]
        # JSON keys must be strings; use the human-friendly label.
        key = exp_cfg["label"](v)
        by_val.setdefault(key, []).append(r)
    agg = {}
    for label, runs in by_val.items():
        e = {}
        for proto in ("summary_stored", "summary_reconfig"):
            for field in ("ACC", "FM", "BWT"):
                m, s = _agg_field(runs, [proto, field])
                e[f"{proto}.{field}"] = {"mean": m, "std": s}
        # head start: per-task task_loss_with_mask
        n_tasks = max(len(r["head_start"]) for r in runs)
        head_means, head_stds = [], []
        for ti in range(n_tasks):
            vals = [
                r["head_start"][ti]["task_loss_with_mask"]
                for r in runs
                if ti < len(r["head_start"])
                and r["head_start"][ti]["task_loss_with_mask"] is not None
            ]
            head_means.append(float(np.mean(vals)) if vals else float("nan"))
            head_stds.append(float(np.std(vals)) if vals else float("nan"))
        e["head_start_per_task_mean"] = head_means
        e["head_start_per_task_std"] = head_stds
        # mean head start over tasks 1..N
        per_run = []
        for r in runs:
            vals = [
                h["task_loss_with_mask"] for h in r["head_start"][1:]
                if h["task_loss_with_mask"] is not None
            ]
            per_run.append(np.mean(vals) if vals else np.nan)
        e["head_start_mean_over_tasks"] = {
            "mean": float(np.nanmean(per_run)),
            "std": float(np.nanstd(per_run)),
        }
        sc_m, sc_s = _agg_field(runs, ["sharing_corr"])
        e["sharing_corr"] = {"mean": sc_m, "std": sc_s}
        e["n_seeds"] = len(runs)
        agg[label] = e
    data["aggregated"] = agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_gpus(spec: str | None) -> list[int]:
    if spec is None or spec == "":
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", required=True, choices=list(EXP_CONFIGS.keys()))
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Working directory: {os.getcwd()}", flush=True)

    exp_tag = args.exp
    exp_cfg = EXP_CONFIGS[exp_tag]
    axis = exp_cfg["axis"]
    values = exp_cfg["values"]

    print(f"Exp     : {exp_tag} (axis={axis}, values={values})", flush=True)
    print(f"Seeds   : {args.seeds}", flush=True)

    if args.cpu or not torch.cuda.is_available():
        devices = ["cpu"] * max(1, args.workers_per_gpu)
    else:
        gpu_ids = _parse_gpus(args.gpus)
        if not gpu_ids:
            raise SystemExit("No GPUs.")
        devices = [f"cuda:{g}" for g in gpu_ids for _ in range(args.workers_per_gpu)]
    print(f"Workers : {len(devices)} -> {devices}", flush=True)

    res_path = os.path.join(rk.RES_DIR, f"{exp_tag}_synthetic.json")
    data = rk.load_results(res_path)
    runs = data.setdefault("runs", {})

    jobs = []
    for v in values:
        label = exp_cfg["label"](v)
        for seed in args.seeds:
            rkey = f"{label}__s{seed}"
            if rkey in runs:
                continue
            jobs.append((seed, axis, v, rkey))

    if not jobs:
        print("Nothing to do.", flush=True)
        aggregate(data, exp_cfg)
        rk.save_results(data, res_path)
        return

    print(f"Total jobs: {len(jobs)}", flush=True)

    ctx = mp.get_context("spawn")
    job_q = ctx.Queue()
    res_q = ctx.Queue()
    for j in jobs:
        job_q.put(j)
    for _ in devices:
        job_q.put(None)

    procs = []
    for wid, dev in enumerate(devices):
        p = ctx.Process(target=_worker, args=(wid, dev, job_q, res_q), daemon=False)
        p.start()
        procs.append(p)

    completed = failed = 0
    t_start = time.time()
    while completed + failed < len(jobs):
        rkey, res, err = res_q.get()
        if err is not None:
            failed += 1
            print(f"  [fail] {rkey}", flush=True)
            continue
        runs[rkey] = res
        aggregate(data, exp_cfg)
        rk.save_results(data, res_path)
        completed += 1
        elapsed = time.time() - t_start
        rate = completed / max(elapsed, 1e-6)
        eta = (len(jobs) - completed - failed) / rate if rate > 0 else float("inf")
        print(
            f"  [save] {rkey} ({completed}/{len(jobs)} failed={failed} "
            f"eta={eta/60:.1f}m)",
            flush=True,
        )

    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    aggregate(data, exp_cfg)
    rk.save_results(data, res_path)
    print(
        f"\nDone. completed={completed} failed={failed} "
        f"elapsed={(time.time()-t_start)/60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
