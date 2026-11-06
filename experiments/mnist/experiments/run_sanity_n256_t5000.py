"""
run_sanity_n256_t5000.py — Sanity rerun of exp5's n=256 condition with
train_steps=5000 (matching exp1) and the fixed (fresh-batch)
head-start metric. 8 seeds in parallel. Writes to a separate file
(does NOT overwrite exp5_synthetic.json) and produces:
    results/sanity_n256_t5000.json
    figures/synthetic/sanity_n256_t5000_headstart_curve.pdf
    + console summary (mean ± std).

Run:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 uv run python \\
        experiments/run_sanity_n256_t5000.py --workers-per-gpu 1
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import masks as M
import metrics
import models
import plotting
from tasks import SyntheticTasks
from trainer import TrainConfig, block_sequential_train

import run_kt as rk

DEFAULT_K = 512
DEFAULT_HIDDEN_SIZE = 512 * 16
DEFAULT_BUDGET = 30
DEFAULT_RECONFIG_N = 256
DEFAULT_TRAIN_STEPS = 5000
DEFAULT_SEEDS = list(range(8))

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


def _build_mask():
    return M.KWTA_AnnealedSTE(
        hidden_size=DEFAULT_HIDDEN_SIZE,
        classification=True,
        k=DEFAULT_K,
        reconfig_steps=DEFAULT_BUDGET,
        reconfig_lr=0.3,
        reconfig_n=DEFAULT_RECONFIG_N,
    )


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


def run_one(*, seed: int, device: str) -> dict:
    train_cfg = replace(TRAIN_CFG, device=device)
    task_set = _build_synth(seed, device)
    mask_cfg = _build_mask()
    model = _build_model(seed)

    t0 = time.time()
    out = block_sequential_train(model, mask_cfg, task_set, train_cfg, verbose=False)
    elapsed = time.time() - t0

    masks_arr = [out["mask_store"][i].cpu().numpy() for i in range(task_set.n_tasks)]
    model_S = plotting.build_neuron_sharing_matrix(masks_arr)
    gt_S = plotting.build_concept_sharing_matrix(task_set.concept_table())
    sharing_corr = plotting.sharing_correlation(gt_S, model_S)

    return {
        "method": "AnnealSTE",
        "budget": DEFAULT_BUDGET,
        "seed": seed,
        "task_family": "synthetic",
        "k": DEFAULT_K,
        "reconfig_n": DEFAULT_RECONFIG_N,
        "train_steps": DEFAULT_TRAIN_STEPS,
        "elapsed_sec": elapsed,
        "summary_stored": metrics.summarise(out["evals"]["stored"], label="AnnealSTE"),
        "summary_reconfig": metrics.summarise(out["evals"]["reconfig"], label="AnnealSTE"),
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
        seed, rkey = job
        t0 = time.time()
        try:
            res = run_one(seed=seed, device=device)
            res_q.put((rkey, res, None))
            print(
                f"[worker {worker_id} {device}] done {rkey} in {time.time()-t0:.1f}s",
                flush=True,
            )
        except Exception:
            err = traceback.format_exc()
            res_q.put((rkey, None, err))
            print(f"[worker {worker_id} {device}] FAILED {rkey}\n{err}", flush=True)


def aggregate_and_plot(data: dict) -> None:
    runs = list(data["runs"].values())
    if not runs:
        return

    head_starts_per_run = []
    n_tasks_max = max(len(r["head_start"]) for r in runs)
    per_task = np.full((len(runs), n_tasks_max), np.nan)
    for i, r in enumerate(runs):
        for t, h in enumerate(r["head_start"]):
            v = h.get("task_loss_with_mask")
            if v is not None:
                per_task[i, t] = float(v)
        vals = [
            h["task_loss_with_mask"]
            for h in r["head_start"][1:]
            if h["task_loss_with_mask"] is not None
        ]
        head_starts_per_run.append(np.mean(vals) if vals else np.nan)

    hs_m = float(np.nanmean(head_starts_per_run))
    hs_s = float(np.nanstd(head_starts_per_run))
    fm = [r["summary_stored"]["FM"] for r in runs]
    bwt = [r["summary_stored"]["BWT"] for r in runs]
    sc = [r["sharing_corr"] for r in runs]
    acc = [r["summary_stored"]["ACC"] for r in runs]

    print("=" * 60)
    print(f"sanity_n256_t5000  (8 seeds, AnnealSTE, k=512, "
          f"reconfig_n=256, train_steps=5000, fresh-batch HS)")
    print("=" * 60)
    print(f"  ACC      = {np.mean(acc):.3f}±{np.std(acc):.3f}")
    print(f"  FM       = {np.mean(fm):.3f}±{np.std(fm):.3f}")
    print(f"  BWT      = {np.mean(bwt):+.3f}±{np.std(bwt):.3f}")
    print(f"  HS       = {hs_m:.3f}±{hs_s:.3f}  (mean over tasks 1..N)")
    print(f"  sharing  = {np.mean(sc):.3f}±{np.std(sc):.3f}")
    print()
    print(f"For comparison:")
    print(f"  exp5 rn=256 (train_steps=500, fresh-batch HS): "
          f"HS=0.525±0.147 ACC=0.818±0.114 FM=0.066±0.046 share=0.300±0.228")
    print(f"  exp5 rn=256 OLD (train_steps=500, configurer-batch HS): "
          f"HS=0.412±0.157 (HS only, other metrics same as above)")

    # Per-task head-start curve
    n_real = max(len(r["head_start"]) for r in runs)
    mean = np.nanmean(per_task[:, :n_real], axis=0)
    std = np.nanstd(per_task[:, :n_real], axis=0)
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    xs = np.arange(1, n_real)
    ax.errorbar(
        xs, mean[1:], yerr=std[1:],
        marker="o", capsize=3, linewidth=1.4,
        color="#377eb8", label="train_steps=5000, n=256",
    )
    ax.set_xlabel("Task index")
    ax.set_ylabel("Head-start loss (held-out batch)")
    ax.set_title("Sanity: n=256, t=5000, fresh-batch HS (8 seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    out_path = "figures/synthetic/sanity_n256_t5000_headstart_curve.pdf"
    plotting.save(fig, out_path)


def _parse_gpus(spec):
    if spec is None or spec == "":
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
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

    res_path = os.path.join(rk.RES_DIR, "sanity_n256_t5000.json")
    data = rk.load_results(res_path)
    runs = data.setdefault("runs", {})

    jobs = []
    for seed in args.seeds:
        rkey = f"AnnealSTE__b30__rn256__t5000__s{seed}"
        if rkey in runs:
            continue
        jobs.append((seed, rkey))

    if not jobs:
        aggregate_and_plot(data)
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

    aggregate_and_plot(data)
    print(
        f"\nDone. completed={completed} failed={failed} "
        f"elapsed={(time.time()-t_start)/60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
