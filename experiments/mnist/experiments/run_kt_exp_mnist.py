"""
run_kt_exp_mnist.py — exp6: shared-shuffle MNIST with the best config
from exp1-5.

Default config matches exp2-5 winner (AnnealSTE, S=30, K=512,
reconfig_n=256, reconfig_freq=batch). Runs 8 seeds in parallel and
writes results to results/exp6_mnist_shared.json plus figures to
figures/mnist_shared/.

Run:
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 uv run python \\
        experiments/run_kt_exp_mnist.py --workers-per-gpu 4
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
from tasks import MNISTSharedShuffle
from trainer import TrainConfig, block_sequential_train

import run_kt as rk

# ---------------------------------------------------------------------------
# Config — matches synthetic default but tuned for MNIST scale
# ---------------------------------------------------------------------------

DEFAULT_METHOD = "AnnealSTE"
DEFAULT_BUDGET = 30
DEFAULT_K = 512
DEFAULT_RECONFIG_N = 256
DEFAULT_HIDDEN_SIZE = 512 * 16  # 8192
DEFAULT_RECONFIG_FREQ = "batch"

DEFAULT_SEEDS = list(range(8))

MNIST_INPUT_SIZE = 784
MNIST_OUTPUT_SIZE = 10
MNIST_N_TASKS = 6
MNIST_N_COMPONENTS = 4
MNIST_COMPONENT_SIZE = 5
MNIST_DATA_DIR = "./data"

MODEL_LAYERS = 8
MODEL_INNER_SIZE = 8
MODEL_DROPOUT_P = 0.2

MNIST_TRAIN_CFG = TrainConfig(
    n_epochs=1,
    train_steps=1000,
    batch_size=256,
    eval_batch_size=2048,
    reconfig_batch_size=DEFAULT_RECONFIG_N,
    lr=3e-4,
    use_replay=True,
    replay_n_per_step=2,
    replay_buf_size=200,
    reconfig_freq=DEFAULT_RECONFIG_FREQ,
    device="cuda",
)


def _build_mnist(seed: int):
    return MNISTSharedShuffle(
        n_tasks=MNIST_N_TASKS,
        n_components=MNIST_N_COMPONENTS,
        component_size=MNIST_COMPONENT_SIZE,
        seed=seed,
        data_dir=MNIST_DATA_DIR,
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
        MNIST_INPUT_SIZE,
        MNIST_OUTPUT_SIZE,
        DEFAULT_HIDDEN_SIZE,
        layers=MODEL_LAYERS,
        inner_size=MODEL_INNER_SIZE,
        dropout_p=MODEL_DROPOUT_P,
    )


def run_one(*, seed: int, device: str) -> dict:
    train_cfg = replace(MNIST_TRAIN_CFG, device=device)
    task_set = _build_mnist(seed)
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
        "method": DEFAULT_METHOD,
        "budget": DEFAULT_BUDGET,
        "seed": seed,
        "task_family": "mnist_shared",
        "k": DEFAULT_K,
        "reconfig_n": DEFAULT_RECONFIG_N,
        "reconfig_freq": DEFAULT_RECONFIG_FREQ,
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
            print(
                f"[worker {worker_id} {device}] FAILED {rkey}\n{err}",
                flush=True,
            )


def aggregate(data: dict) -> None:
    runs = list(data["runs"].values())
    if not runs:
        return
    e: dict = {}
    for proto in ("summary_stored", "summary_reconfig"):
        for field in ("ACC", "FM", "BWT"):
            vals = [r[proto][field] for r in runs if r[proto].get(field) is not None]
            e[f"{proto}.{field}"] = {
                "mean": float(np.mean(vals)) if vals else float("nan"),
                "std": float(np.std(vals)) if vals else float("nan"),
            }
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
    sc_vals = [r["sharing_corr"] for r in runs if r["sharing_corr"] is not None]
    e["sharing_corr"] = {
        "mean": float(np.mean(sc_vals)) if sc_vals else float("nan"),
        "std": float(np.std(sc_vals)) if sc_vals else float("nan"),
    }
    e["n_seeds"] = len(runs)
    data["aggregated"] = e


def _parse_gpus(spec: str | None) -> list[int]:
    if spec is None or spec == "":
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Working directory: {os.getcwd()}", flush=True)

    if args.cpu or not torch.cuda.is_available():
        devices = ["cpu"] * max(1, args.workers_per_gpu)
    else:
        gpu_ids = _parse_gpus(args.gpus)
        if not gpu_ids:
            raise SystemExit("No GPUs.")
        devices = [f"cuda:{g}" for g in gpu_ids for _ in range(args.workers_per_gpu)]
    print(f"Workers : {len(devices)}", flush=True)

    res_path = os.path.join(rk.RES_DIR, "exp6_mnist_shared.json")
    data = rk.load_results(res_path)
    runs = data.setdefault("runs", {})

    jobs = []
    for seed in args.seeds:
        rkey = f"AnnealSTE__b30__s{seed}"
        if rkey in runs:
            continue
        jobs.append((seed, rkey))

    if not jobs:
        aggregate(data)
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
        aggregate(data)
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

    aggregate(data)
    rk.save_results(data, res_path)
    print(
        f"Done. completed={completed} failed={failed} elapsed="
        f"{(time.time()-t_start)/60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
