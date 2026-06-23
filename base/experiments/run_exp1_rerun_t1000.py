"""
run_exp1_rerun_t1000.py — Rerun of exp1 (method × compute budget) on
the correlated synthetic family with the corrected fresh-batch
head-start metric and train_steps=1000 (down from the original 5000).

Sweep: 3 methods (Sigmoid, AnnealSTE, Gumbel) × 6 budgets (10, 20,
30, 40, 50, 60) × 1 seed = 18 jobs. All other hyperparameters match
the original exp1 (k=512, reconfig_n=256, per-batch reconfig,
correlated tasks, H=8192, L=8, inner=8, dropout=0.2).

Writes to results/exp1_rerun_t1000.json (does NOT overwrite
results/synthetic.json) and prints a summary.

Run:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run python \\
        experiments/run_exp1_rerun_t1000.py --workers-per-gpu 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from dataclasses import replace

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

import run_kt as rk

DEFAULT_K = 512
DEFAULT_HIDDEN_SIZE = 512 * 16
DEFAULT_RECONFIG_N = 256
DEFAULT_TRAIN_STEPS = 1000
DEFAULT_SEEDS = [0]
DEFAULT_METHODS = ["Sigmoid", "AnnealSTE", "Gumbel"]
DEFAULT_BUDGETS = [10, 20, 30, 40, 50, 60]

SYNTH_INPUT_SIZE = 2
SYNTH_OUTPUT_SIZE = 2
SYNTH_N_TASKS = 12
SYNTH_N_COMPONENTS = 6
SYNTH_TASK_SIZE = 1
SYNTH_FREQUENCY = 8
SYNTH_TEST_SIZE = 2048

TRAIN_CFG = TrainConfig(
    n_epochs=1,
    train_steps=DEFAULT_TRAIN_STEPS,
    batch_size=256,
    eval_batch_size=2048,
    reconfig_batch_size=DEFAULT_RECONFIG_N,
    lr=3e-4,
    use_replay=True,
    replay_n_per_step=2,
    replay_buf_size=200,
    reconfig_freq="batch",
    device="cuda",
)

GUMBEL_TAU_START = 1.0
GUMBEL_TAU_END = 0.1
GUMBEL_REG_STRENGTH = 1.0


def _build_synth(seed: int, device: str):
    return SyntheticTasks(
        input_size=SYNTH_INPUT_SIZE,
        output_size=SYNTH_OUTPUT_SIZE,
        n_tasks=SYNTH_N_TASKS,
        n_components=SYNTH_N_COMPONENTS,
        task_size=SYNTH_TASK_SIZE,
        frequency=SYNTH_FREQUENCY,
        classification=True,
        seed=seed,
        device=device,
        test_size=SYNTH_TEST_SIZE,
        orthogonal=False,
    )


def _build_mask(method: str, budget: int):
    if method == "Sigmoid":
        return M.KWTA_Sigmoid(
            hidden_size=DEFAULT_HIDDEN_SIZE,
            classification=True,
            k=DEFAULT_K,
            reconfig_steps=budget,
            reconfig_lr=0.3,
            reconfig_n=DEFAULT_RECONFIG_N,
        )
    if method == "AnnealSTE":
        return M.KWTA_AnnealedSTE(
            hidden_size=DEFAULT_HIDDEN_SIZE,
            classification=True,
            k=DEFAULT_K,
            reconfig_steps=budget,
            reconfig_lr=0.3,
            reconfig_n=DEFAULT_RECONFIG_N,
        )
    if method == "Gumbel":
        return M.KWTA_Gumbel(
            hidden_size=DEFAULT_HIDDEN_SIZE,
            classification=True,
            k=DEFAULT_K,
            reconfig_steps=budget,
            reconfig_lr=0.2,
            reconfig_n=DEFAULT_RECONFIG_N,
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
        layers=8,
        inner_size=8,
        dropout_p=0.2,
    )


def run_one(*, method: str, budget: int, seed: int, device: str) -> dict:
    train_cfg = replace(TRAIN_CFG, device=device)
    task_set = _build_synth(seed, device)
    mask_cfg = _build_mask(method, budget)
    model = _build_model(seed)

    t0 = time.time()
    out = block_sequential_train(model, mask_cfg, task_set, train_cfg, verbose=False)
    elapsed = time.time() - t0

    masks_arr = [out["mask_store"][i].cpu().numpy() for i in range(task_set.n_tasks)]
    model_S = plotting.build_neuron_sharing_matrix(masks_arr)
    gt_S = plotting.build_concept_sharing_matrix(task_set.concept_table())
    sharing_corr = plotting.sharing_correlation(gt_S, model_S)

    return {
        "method": method,
        "budget": budget,
        "seed": seed,
        "task_family": "synthetic",
        "k": DEFAULT_K,
        "reconfig_n": DEFAULT_RECONFIG_N,
        "train_steps": DEFAULT_TRAIN_STEPS,
        "elapsed_sec": elapsed,
        "summary_stored": metrics.summarise(out["evals"]["stored"], label=method),
        "summary_reconfig": metrics.summarise(out["evals"]["reconfig"], label=method),
        "perf_matrix_stored": out["evals"]["stored"],
        "perf_matrix_reconfig": out["evals"]["reconfig"],
        "head_start": out["head_start"],
        "first_epoch_loss": out["first_epoch_loss"],
        "concept_table": task_set.concept_table().tolist(),
        "model_sharing": model_S.tolist(),
        "gt_sharing": gt_S.tolist(),
        "sharing_corr": sharing_corr,
    }


def _worker(worker_id: int, device: str, job_q, res_q) -> None:
    if device.startswith("cuda"):
        try:
            torch.cuda.set_device(device)
        except Exception as e:
            print(f"[worker {worker_id}] set_device failed: {e}", flush=True)
    while True:
        job = job_q.get()
        if job is None:
            return
        method, budget, seed, rkey = job
        t0 = time.time()
        try:
            res = run_one(method=method, budget=budget, seed=seed, device=device)
            res_q.put((rkey, res, None))
            print(
                f"[worker {worker_id} {device}] done {rkey} in {time.time()-t0:.1f}s",
                flush=True,
            )
        except Exception:
            err = traceback.format_exc()
            res_q.put((rkey, None, err))
            print(f"[worker {worker_id} {device}] FAILED {rkey}\n{err}", flush=True)


def aggregate_and_summary(data: dict) -> None:
    runs = list(data["runs"].values())
    if not runs:
        return
    by = {}
    for r in runs:
        by.setdefault((r["method"], r["budget"]), []).append(r)
    print("=" * 78)
    print("exp1_rerun_t1000  (3 methods × 6 budgets × 1 seed, t=1000, fresh-batch HS)")
    print("=" * 78)
    print(f"{'method':<12}{'budget':>7}  {'ACC':>7} {'FM':>7} {'BWT':>8} {'HS':>7} {'share':>7}")
    for (method, budget), rs in sorted(by.items()):
        r = rs[0]
        hs_vals = [
            h["task_loss_with_mask"]
            for h in r["head_start"][1:]
            if h["task_loss_with_mask"] is not None
        ]
        hs = float(np.mean(hs_vals)) if hs_vals else float("nan")
        print(
            f"{method:<12}{budget:>7}  "
            f"{r['summary_stored']['ACC']:>7.3f} "
            f"{r['summary_stored']['FM']:>7.3f} "
            f"{r['summary_stored']['BWT']:>+8.3f} "
            f"{hs:>7.3f} "
            f"{r['sharing_corr']:>7.3f}"
        )


def _parse_gpus(spec):
    if spec is None or spec == "":
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Working directory: {os.getcwd()}", flush=True)

    if args.cpu or not torch.cuda.is_available():
        devices = ["cpu"] * max(1, args.workers_per_gpu)
    else:
        gpu_ids = _parse_gpus(args.gpus)
        devices = [f"cuda:{g}" for g in gpu_ids for _ in range(args.workers_per_gpu)]
    print(f"Workers : {len(devices)}", flush=True)

    res_path = os.path.join(rk.RES_DIR, "exp1_rerun_t1000.json")
    data = rk.load_results(res_path)
    runs = data.setdefault("runs", {})

    jobs = []
    for method in args.methods:
        for budget in args.budgets:
            for seed in args.seeds:
                rkey = f"{method}__b{budget}__s{seed}"
                if rkey in runs:
                    continue
                jobs.append((method, budget, seed, rkey))

    if not jobs:
        aggregate_and_summary(data)
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
        rk.save_results(data, res_path)
        completed += 1
        rate = completed / max(time.time() - t_start, 1e-6)
        eta = (len(jobs) - completed - failed) / rate if rate > 0 else float("inf")
        print(
            f"  [save] {rkey} ({completed}/{len(jobs)} eta={eta/60:.1f}m)",
            flush=True,
        )
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    aggregate_and_summary(data)
    print(
        f"\nDone. completed={completed} failed={failed} "
        f"elapsed={(time.time()-t_start)/60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
