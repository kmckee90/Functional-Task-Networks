"""
run_kt_parallel.py — Parallel sweep runner for run_kt.py.

Distributes (task_family, method, budget, seed) jobs across the available
GPUs.  Each worker process is pinned to one CUDA device and pulls jobs
from a shared queue until drained.  Results are written back to the
same per-task JSON files that ``run_kt.py`` produces, so the two
scripts are interchangeable for resuming a sweep.

Run:
    uv run python experiments/run_kt_parallel.py --tasks synthetic mnist_shared
    uv run python experiments/run_kt_parallel.py --tasks synthetic \\
        --budgets 50 500 5000 --methods Sigmoid AnnealSTE Gumbel \\
        --seeds 0 1 2 --workers-per-gpu 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.multiprocessing as mp

import run_kt as rk  # type: ignore  # sibling module in experiments/

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _worker(
    worker_id: int,
    device: str,
    job_q: "mp.Queue",
    res_q: "mp.Queue",
) -> None:
    """Pull jobs from job_q, run them on `device`, push results to res_q."""
    # Make this process see only the assigned device for any libraries that
    # default to cuda:0.  We still pass the explicit device string downstream.
    try:
        torch.cuda.set_device(device) if device.startswith("cuda") else None
    except Exception as e:  # pragma: no cover - defensive
        print(f"[worker {worker_id}] failed to set device {device}: {e}",
              flush=True)

    while True:
        job = job_q.get()
        if job is None:
            return
        task_family, method, budget, seed, rkey = job
        cfg = rk.TASK_REGISTRY[task_family]
        t0 = time.time()
        try:
            res = rk.run_one(method, budget, seed, task_family, cfg, device)
            res_q.put((task_family, rkey, res, None))
            print(
                f"[worker {worker_id} {device}] done "
                f"{task_family} {rkey} in {time.time()-t0:.1f}s",
                flush=True,
            )
        except Exception:
            err = traceback.format_exc()
            res_q.put((task_family, rkey, None, err))
            print(
                f"[worker {worker_id} {device}] FAILED {task_family} {rkey}\n{err}",
                flush=True,
            )


# ---------------------------------------------------------------------------
# Job-list construction
# ---------------------------------------------------------------------------


def _enumerate_jobs(
    task_families: list[str],
    methods: list[str],
    budgets: list[int],
    seeds: list[int],
    include_baselines: bool,
    existing: dict[str, dict],
) -> list[tuple]:
    """Build (task_family, method, budget, seed, rkey) tuples for everything
    not already present in the loaded results dicts."""
    jobs: list[tuple] = []
    all_methods = list(methods)
    if include_baselines:
        all_methods += rk.BASELINE_METHODS

    for tf in task_families:
        runs = existing[tf].setdefault("runs", {})
        for method in all_methods:
            method_budgets = (
                [1] if method in rk.BASELINE_METHODS else list(budgets)
            )
            for budget in method_budgets:
                for seed in seeds:
                    rkey = f"{method}__b{budget}__s{seed}"
                    if rkey in runs:
                        continue
                    jobs.append((tf, method, budget, seed, rkey))
    return jobs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_gpu_list(spec: str | None) -> list[int]:
    if spec is None or spec == "":
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks", nargs="+",
        choices=list(rk.TASK_REGISTRY.keys()),
        default=["synthetic"],
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=rk.DEFAULT_METHODS,
        choices=rk.DEFAULT_METHODS + rk.BASELINE_METHODS,
    )
    parser.add_argument(
        "--budgets", nargs="+", type=int, default=rk.DEFAULT_BUDGETS,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=rk.DEFAULT_SEEDS)
    parser.add_argument(
        "--include-baselines", action="store_true",
        help="Also run FixedMask and NoMask at budget=1.",
    )
    parser.add_argument(
        "--gpus", default=None,
        help="Comma-separated GPU indices, e.g. '0,2,3'. "
             "Default: every visible GPU.",
    )
    parser.add_argument(
        "--workers-per-gpu", type=int, default=1,
        help="Concurrent jobs per GPU (raise if memory headroom permits).",
    )
    parser.add_argument(
        "--cpu", action="store_true",
        help="Force CPU execution with `--workers-per-gpu` workers total.",
    )
    parser.add_argument("--no-figs", action="store_true")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"Working directory: {os.getcwd()}", flush=True)

    # Decide on devices
    if args.cpu or not torch.cuda.is_available():
        gpu_ids: list[int] = []
        devices = ["cpu"] * max(1, args.workers_per_gpu)
    else:
        gpu_ids = _parse_gpu_list(args.gpus)
        if not gpu_ids:
            raise SystemExit("No GPUs selected and --cpu not set.")
        devices = [
            f"cuda:{g}" for g in gpu_ids for _ in range(args.workers_per_gpu)
        ]

    print(f"Tasks   : {args.tasks}", flush=True)
    print(f"Methods : {args.methods}", flush=True)
    print(f"Budgets : {args.budgets}", flush=True)
    print(f"Seeds   : {args.seeds}", flush=True)
    print(f"Workers : {len(devices)}  -> {devices}", flush=True)

    # Load existing results so we can skip and so the writer has somewhere
    # to merge into.
    existing: dict[str, dict] = {}
    res_paths: dict[str, str] = {}
    for tf in args.tasks:
        path = os.path.join(rk.RES_DIR, f"{tf}.json")
        res_paths[tf] = path
        existing[tf] = rk.load_results(path)

    jobs = _enumerate_jobs(
        args.tasks, args.methods, args.budgets, args.seeds,
        args.include_baselines, existing,
    )
    if not jobs:
        print("Nothing to do — all (method, budget, seed) cells already "
              "present in the result files.", flush=True)
        # Still re-aggregate and (optionally) regenerate figures.
        for tf in args.tasks:
            rk.aggregate(existing[tf])
            rk.save_results(existing[tf], res_paths[tf])
            if not args.no_figs:
                rk.render_sharing_figs(
                    existing[tf], tf, rk.FIG_DIR, args.methods,
                )
        return

    print(f"Total jobs: {len(jobs)}", flush=True)

    ctx = mp.get_context("spawn")
    job_q: mp.Queue = ctx.Queue()
    res_q: mp.Queue = ctx.Queue()

    for j in jobs:
        job_q.put(j)
    # Sentinels — one per worker.
    for _ in devices:
        job_q.put(None)

    procs = []
    for wid, dev in enumerate(devices):
        p = ctx.Process(
            target=_worker, args=(wid, dev, job_q, res_q), daemon=False,
        )
        p.start()
        procs.append(p)

    completed = 0
    failed = 0
    t_start = time.time()
    while completed + failed < len(jobs):
        tf, rkey, res, err = res_q.get()
        if err is not None:
            failed += 1
            print(f"  [fail] {tf} {rkey}", flush=True)
            continue
        existing[tf]["runs"][rkey] = res
        rk.aggregate(existing[tf])
        rk.save_results(existing[tf], res_paths[tf])
        completed += 1
        elapsed = time.time() - t_start
        remaining = len(jobs) - completed - failed
        rate = completed / max(elapsed, 1e-6)
        eta = remaining / rate if rate > 0 else float("inf")
        print(
            f"  [save] {tf} {rkey}  ({completed}/{len(jobs)}, "
            f"failed={failed}, eta={eta/60:.1f}m)",
            flush=True,
        )

    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()

    # Final aggregate + figs
    for tf in args.tasks:
        rk.aggregate(existing[tf])
        rk.save_results(existing[tf], res_paths[tf])
        if not args.no_figs:
            rk.render_sharing_figs(
                existing[tf], tf, rk.FIG_DIR, args.methods,
            )

    print(
        f"\nDone. completed={completed} failed={failed} "
        f"elapsed={(time.time()-t_start)/60:.1f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
