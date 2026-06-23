"""
run_baselines.py — Run NoMask and FixedMask baselines on synthetic
and MNIST shared-shuffle, 8 seeds each, at the same hyperparameters
as the exp2-5 / exp6 sweeps. Saves to:
    results/baselines_synthetic.json
    results/baselines_mnist_shared.json

Run:
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 uv run python \\
        experiments/run_baselines.py --workers-per-gpu 4
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
from tasks import MNISTSharedShuffle, SyntheticTasks
from trainer import TrainConfig, block_sequential_train

import run_kt as rk

DEFAULT_K = 512
DEFAULT_HIDDEN_SIZE = 512 * 16
DEFAULT_SEEDS = list(range(8))

# Synthetic config (matches run_kt_exp.py)
SYNTH_INPUT_SIZE = 2
SYNTH_OUTPUT_SIZE = 2
SYNTH_N_TASKS = 12
SYNTH_N_COMPONENTS = 6
SYNTH_TASK_SIZE = 1
SYNTH_FREQUENCY = 8
SYNTH_TEST_SIZE = 2048
SYNTH_TRAIN_CFG = TrainConfig(
    n_epochs=1, train_steps=1000, batch_size=256,
    eval_batch_size=2048, reconfig_batch_size=256,
    lr=3e-4, use_replay=True, replay_n_per_step=2, replay_buf_size=200,
    reconfig_freq="batch", device="cuda",
)

# MNIST config (matches run_kt_exp_mnist.py)
MNIST_TRAIN_CFG = TrainConfig(
    n_epochs=1, train_steps=1000, batch_size=256,
    eval_batch_size=2048, reconfig_batch_size=256,
    lr=3e-4, use_replay=True, replay_n_per_step=2, replay_buf_size=200,
    reconfig_freq="batch", device="cuda",
)

MODEL_LAYERS = 8
MODEL_INNER_SIZE = 8
MODEL_DROPOUT_P = 0.2


def _build_synth(seed: int, device: str):
    return SyntheticTasks(
        input_size=SYNTH_INPUT_SIZE, output_size=SYNTH_OUTPUT_SIZE,
        n_tasks=SYNTH_N_TASKS, n_components=SYNTH_N_COMPONENTS,
        task_size=SYNTH_TASK_SIZE, frequency=SYNTH_FREQUENCY,
        classification=True, seed=seed, device=device,
        test_size=SYNTH_TEST_SIZE, orthogonal=False,
    )


def _build_mnist(seed: int):
    return MNISTSharedShuffle(
        n_tasks=6, n_components=4, component_size=5,
        seed=seed, data_dir="./data",
    )


def _build_model(in_size: int, out_size: int, seed: int) -> models.NeuroModel:
    torch.manual_seed(seed)
    return models.NeuroModel(
        in_size, out_size, DEFAULT_HIDDEN_SIZE,
        layers=MODEL_LAYERS, inner_size=MODEL_INNER_SIZE,
        dropout_p=MODEL_DROPOUT_P,
    )


def _build_mask(method: str):
    if method == "FixedMask":
        return M.FixedMask(hidden_size=DEFAULT_HIDDEN_SIZE, k=DEFAULT_K)
    if method == "NoMask":
        return M.NoMask(hidden_size=DEFAULT_HIDDEN_SIZE, k=DEFAULT_K)
    raise ValueError(method)


def run_one(*, family: str, method: str, seed: int, device: str) -> dict:
    if family == "synthetic":
        train_cfg = replace(SYNTH_TRAIN_CFG, device=device)
        task_set = _build_synth(seed, device)
        in_size, out_size = SYNTH_INPUT_SIZE, SYNTH_OUTPUT_SIZE
    elif family == "mnist_shared":
        train_cfg = replace(MNIST_TRAIN_CFG, device=device)
        task_set = _build_mnist(seed)
        in_size, out_size = 784, 10
    else:
        raise ValueError(family)

    mask_cfg = _build_mask(method)
    model = _build_model(in_size, out_size, seed)

    t0 = time.time()
    out = block_sequential_train(model, mask_cfg, task_set, train_cfg, verbose=False)
    elapsed = time.time() - t0

    masks_arr = [out["mask_store"][i].cpu().numpy() for i in range(task_set.n_tasks)]
    model_S = plotting.build_neuron_sharing_matrix(masks_arr)
    gt_S = plotting.build_concept_sharing_matrix(task_set.concept_table())
    sharing_corr = plotting.sharing_correlation(gt_S, model_S)

    return {
        "method": method,
        "budget": 1,
        "seed": seed,
        "task_family": family,
        "k": DEFAULT_K,
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
        family, method, seed, rkey = job
        t0 = time.time()
        try:
            res = run_one(family=family, method=method, seed=seed, device=device)
            res_q.put((family, rkey, res, None))
            print(
                f"[worker {worker_id} {device}] done {family} {rkey} "
                f"in {time.time()-t0:.1f}s",
                flush=True,
            )
        except Exception:
            err = traceback.format_exc()
            res_q.put((family, rkey, None, err))
            print(
                f"[worker {worker_id} {device}] FAILED {family} {rkey}\n{err}",
                flush=True,
            )


def aggregate(data: dict) -> None:
    runs = list(data["runs"].values())
    if not runs:
        return
    by_method: dict[str, list[dict]] = {}
    for r in runs:
        by_method.setdefault(r["method"], []).append(r)
    agg = {}
    for method, rs in by_method.items():
        e = {}
        for proto in ("summary_stored", "summary_reconfig"):
            for field in ("ACC", "FM", "BWT"):
                vals = [
                    r[proto][field] for r in rs if r[proto].get(field) is not None
                ]
                e[f"{proto}.{field}"] = {
                    "mean": float(np.mean(vals)) if vals else float("nan"),
                    "std": float(np.std(vals)) if vals else float("nan"),
                }
        sc_vals = [r["sharing_corr"] for r in rs if r["sharing_corr"] is not None]
        e["sharing_corr"] = {
            "mean": float(np.mean(sc_vals)) if sc_vals else float("nan"),
            "std": float(np.std(sc_vals)) if sc_vals else float("nan"),
        }
        e["n_seeds"] = len(rs)
        agg[method] = e
    data["aggregated"] = agg


def _parse_gpus(spec):
    if spec is None or spec == "":
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument(
        "--methods", nargs="+", default=["NoMask", "FixedMask"],
        choices=["NoMask", "FixedMask"],
    )
    parser.add_argument(
        "--families", nargs="+", default=["synthetic", "mnist_shared"],
        choices=["synthetic", "mnist_shared"],
    )
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
        devices = [f"cuda:{g}" for g in gpu_ids for _ in range(args.workers_per_gpu)]
    print(f"Workers : {len(devices)}", flush=True)

    res_paths: dict[str, str] = {}
    data_per_family: dict[str, dict] = {}
    for fam in args.families:
        path = os.path.join(rk.RES_DIR, f"baselines_{fam}.json")
        res_paths[fam] = path
        data_per_family[fam] = rk.load_results(path)

    jobs = []
    for fam in args.families:
        runs = data_per_family[fam].setdefault("runs", {})
        for method in args.methods:
            for seed in args.seeds:
                rkey = f"{method}__s{seed}"
                if rkey in runs:
                    continue
                jobs.append((fam, method, seed, rkey))

    if not jobs:
        for fam in args.families:
            aggregate(data_per_family[fam])
            rk.save_results(data_per_family[fam], res_paths[fam])
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
        fam, rkey, res, err = res_q.get()
        if err is not None:
            failed += 1
            print(f"  [fail] {fam} {rkey}", flush=True)
            continue
        data_per_family[fam]["runs"][rkey] = res
        aggregate(data_per_family[fam])
        rk.save_results(data_per_family[fam], res_paths[fam])
        completed += 1
        rate = completed / max(time.time() - t_start, 1e-6)
        eta = (len(jobs) - completed - failed) / rate if rate > 0 else float("inf")
        print(
            f"  [save] {fam} {rkey} ({completed}/{len(jobs)} eta={eta/60:.1f}m)",
            flush=True,
        )
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    for fam in args.families:
        aggregate(data_per_family[fam])
        rk.save_results(data_per_family[fam], res_paths[fam])
    print(
        f"Done. completed={completed} failed={failed} "
        f"elapsed={(time.time()-t_start)/60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
