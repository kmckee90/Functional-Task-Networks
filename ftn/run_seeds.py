#!/usr/bin/env python
"""
run_seeds.py - Multi-seed sweep driver for run_experiments.py.

Orchestration only: this script owns nothing about the experiment itself. For
each CONDITION (a set of sub-config names - exactly the args you'd hand to
gpurun.sh) it runs `--seeds` runs, each with a distinct --seed and each in
--metrics-only mode, so every run drops a single parseable metrics_<name>.txt
and no visualizations. A later pass (analyze_sweep.py) pools those files.

Dispatch is a GPU-queue scheduler (default): every (condition, seed) run goes
into one queue, and the scheduler works through it, keeping a pinned run on each
GPU in the pool and launching the next queued run as soon as a GPU frees up. So
the sweep can be much larger than the GPU count without ever failing on GPU
exhaustion - extra runs just wait their turn. The pool defaults to the GPUs that
are currently free (used memory below --mem-threshold); override with --gpus.
This process stays alive until the whole queue drains.

  --serial       run inline via `uv`, one at a time, blocking, live output
                 (single GPU / whatever CUDA_VISIBLE_DEVICES is set).
  --gpus 0,1,2   force the GPU pool instead of auto-detecting free ones.
  --jobs-per-gpu pack more than one run onto each GPU (default 1).
  --max-parallel cap the total number of concurrent runs.

Layout of outputs (one metrics file per run):

    <out-root>/<sweep-name>/<config1_config2_...>/seed<k>/metrics_seed<k>.txt

Per-run stdout/stderr goes to <out-root>/<sweep-name>/logs/.

Examples:
  # 5 seeds each of two conditions, worked through as GPUs free up
  uv run python run_seeds.py --sweep-name reconf --seeds 5 \
      --condition "Hopfield0 RNN_Resid_1x8 Reactive_Low NoReplay" \
      --condition "Fast RNN_Resid_1x8 Reactive_Low NoReplay"

  # restrict to GPUs 0-3, two runs packed per GPU
  uv run python run_seeds.py --seeds 8 --gpus 0,1,2,3 --jobs-per-gpu 2 \
      --condition "Default NoReplay"

  # per-condition passthrough args go after a literal -- inside the condition
  uv run python run_seeds.py --seeds 3 \
      --condition "Default NoReplay -- --reconfig-every 5"

  # single-GPU, blocking, live output
  uv run python run_seeds.py --serial --seeds 3 --condition "Default"

With no --condition, DEFAULT_CONDITIONS below is used.
"""

import argparse
import datetime as _dt
import os
import shutil
import subprocess
import sys
import time
from collections import deque

# Edit this to hard-code a sweep (used when no --condition is passed). Each
# entry is one condition: the sub-config names you'd give gpurun.sh, optionally
# followed by "--" and extra run_experiments.py args.
DEFAULT_CONDITIONS = [
    "Default NoReplay",
    # "Hopfield0 RNN_Resid_1x8 Reactive_Low NoReplay",
]


def parse_condition(spec):
    """'A B -- --flag v' -> (['A','B'], ['--flag','v']). Configs before the
    first literal '--', run_experiments passthrough args after it."""
    toks = spec.split()
    if "--" in toks:
        i = toks.index("--")
        return toks[:i], toks[i + 1:]
    return toks, []


def condition_tag(configs):
    """Folder-safe tag for a condition: its sub-configs joined with '_'."""
    return "_".join(configs)


def build_run_cmd(configs, extra, name, seed, cond_out_root):
    """The argv for one (condition, seed) run: run_experiments.py with --seed /
    --metrics-only / --out-root / --name, so outputs land at
    <cond_out_root>/<name>/. The GPU is chosen by the scheduler via
    CUDA_VISIBLE_DEVICES in the child env, not baked into the command."""
    return [
        "uv", "run", "python", "run_experiments.py",
        "--config", *configs,
        "--name", name,
        "--seed", str(seed),
        "--metrics-only",
        "--out-root", cond_out_root,
        *extra,
    ]


# =============================================================================
# GPU pool
# =============================================================================


def detect_free_gpus(mem_threshold):
    """GPU indices whose used memory is below mem_threshold MB, via nvidia-smi.
    Returns None if nvidia-smi is unavailable / errors."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.check_output(
            [exe, "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return None
    free = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx, used = int(parts[0]), float(parts[1])
        except ValueError:
            continue
        if used < mem_threshold:
            free.append(idx)
    return free


def build_slots(pool, jobs_per_gpu, max_parallel):
    """Concurrency slots as a list of GPU ids. Each GPU appears jobs_per_gpu
    times (round-robin ordering), truncated to max_parallel if set."""
    slots = []
    for _ in range(jobs_per_gpu):
        slots.extend(pool)
    if max_parallel is not None:
        slots = slots[:max_parallel]
    return slots


# =============================================================================
# Scheduler
# =============================================================================


def _ts():
    return _dt.datetime.now().strftime("%H:%M:%S")


def _label(run):
    return f"{condition_tag(run['configs'])}/{run['name']}"


def run_scheduler(runs, slots, log_dir, poll=1.0):
    """Work through `runs` keeping one pinned job per slot (a GPU id). Launches
    the next queued run whenever a slot frees. Blocks until the queue drains.
    Returns (completed, failed). Per-run output goes to a log file in log_dir."""
    os.makedirs(log_dir, exist_ok=True)
    queue = deque(runs)
    free = deque(slots)
    active = {}  # Popen -> {run, gpu, fh, log}
    total = len(runs)
    done = failed = 0

    try:
        while queue or active:
            # Fill every free slot from the queue.
            while queue and free:
                gpu = free.popleft()
                run = queue.popleft()
                log = os.path.join(
                    log_dir,
                    f"{condition_tag(run['configs'])}__{run['name']}__gpu{gpu}.log",
                )
                fh = open(log, "w")
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                proc = subprocess.Popen(
                    run["cmd"], stdout=fh, stderr=subprocess.STDOUT, env=env,
                )
                active[proc] = {"run": run, "gpu": gpu, "fh": fh, "log": log}
                print(f"[{_ts()}] launch  gpu{gpu}  {_label(run)}   "
                      f"(running {len(active)}, queued {len(queue)})")

            if not active:
                break
            time.sleep(poll)

            # Reap finished jobs and free their slots.
            for proc in list(active):
                rc = proc.poll()
                if rc is None:
                    continue
                info = active.pop(proc)
                info["fh"].close()
                free.append(info["gpu"])
                done += 1
                if rc == 0:
                    print(f"[{_ts()}] done    gpu{info['gpu']}  "
                          f"{_label(info['run'])}   ({done}/{total})")
                else:
                    failed += 1
                    print(f"[{_ts()}] FAIL    gpu{info['gpu']}  "
                          f"{_label(info['run'])}  rc={rc}  log={info['log']}",
                          file=sys.stderr)
    except KeyboardInterrupt:
        print("\nInterrupted - terminating running jobs...", file=sys.stderr)
        for proc in active:
            proc.terminate()
        for proc in active:
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            active[proc]["fh"].close()
        raise

    return done, failed


def run_serial(runs):
    """Inline, one at a time, blocking, inheriting stdout (single GPU)."""
    done = failed = 0
    for r in runs:
        print(f"\n>>> [{_label(r)}] {' '.join(r['cmd'])}")
        rc = subprocess.call(r["cmd"])
        done += 1
        if rc != 0:
            failed += 1
            print(f"!!! [{_label(r)}] exited rc={rc}", file=sys.stderr)
    return done, failed


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--condition", action="append", default=[], metavar="SPEC",
        help="A condition: sub-config names (as given to gpurun.sh), optionally "
        "followed by '-- <extra run_experiments args>'. Repeatable. Defaults to "
        "DEFAULT_CONDITIONS in this file when omitted.",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of seeds per condition (default 3).",
    )
    parser.add_argument(
        "--seed-start", type=int, default=0,
        help="First seed value; seeds run over [start, start+seeds) (default 0).",
    )
    parser.add_argument(
        "--sweep-name", default=None,
        help="Sweep folder name under <out-root> (default: sweep_<timestamp>).",
    )
    parser.add_argument(
        "--out-root", default="artifacts/sweeps",
        help="Base directory for sweeps (default artifacts/sweeps).",
    )
    parser.add_argument(
        "--gpus", default=None,
        help="Comma-separated GPU pool (e.g. 0,1,2,3). Default: auto-detect the "
        "GPUs currently free (used mem < --mem-threshold).",
    )
    parser.add_argument(
        "--jobs-per-gpu", type=int, default=1,
        help="Concurrent runs packed onto each GPU (default 1). Raise only for "
        "small models that comfortably share a card.",
    )
    parser.add_argument(
        "--max-parallel", type=int, default=None,
        help="Cap the total number of concurrent runs (default: pool size x "
        "jobs-per-gpu).",
    )
    parser.add_argument(
        "--mem-threshold", type=float, default=1000.0,
        help="A GPU counts as free when its used memory is below this many MB "
        "(default 1000). Used only for auto-detection.",
    )
    parser.add_argument(
        "--serial", action="store_true",
        help="Run inline via uv, one at a time, blocking, live output (single "
        "GPU) instead of the GPU-queue scheduler.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the planned runs (and GPU pool) and exit.",
    )
    args = parser.parse_args()

    specs = args.condition or DEFAULT_CONDITIONS
    if not specs:
        parser.error("no conditions given and DEFAULT_CONDITIONS is empty")
    conditions = [parse_condition(s) for s in specs]
    for configs, _ in conditions:
        if not configs:
            parser.error("a condition has no sub-configs")

    sweep = args.sweep_name or f"sweep_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    sweep_root = os.path.join(args.out_root, sweep)
    log_dir = os.path.join(sweep_root, "logs")
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    # Plan the full grid up front so the manifest is complete regardless of how
    # long the queue takes to drain.
    runs = []
    for configs, extra in conditions:
        cond_out_root = os.path.join(sweep_root, condition_tag(configs))
        for k in seeds:
            name = f"seed{k}"
            runs.append({
                "configs": configs,
                "extra": extra,
                "seed": k,
                "name": name,
                "cond_out_root": cond_out_root,
                "metrics_path": os.path.join(
                    cond_out_root, name, f"metrics_{name}.txt"
                ),
                "cmd": build_run_cmd(configs, extra, name, k, cond_out_root),
            })

    # Resolve the GPU pool / slots (scheduler mode only).
    slots = None
    if not args.serial:
        if args.gpus:
            pool = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
        else:
            pool = detect_free_gpus(args.mem_threshold)
            if pool is None:
                parser.error(
                    "could not query GPUs (nvidia-smi missing). Pass --gpus "
                    "explicitly, or --serial."
                )
            if not pool:
                parser.error(
                    f"no free GPUs (all >= {args.mem_threshold:g} MB used). "
                    "Pass --gpus to force specific ones, or --serial."
                )
        slots = build_slots(pool, args.jobs_per_gpu, args.max_parallel)

    mode = "serial (blocking)" if args.serial else "GPU-queue scheduler"
    print(f"Sweep      : {sweep}")
    print(f"Root       : {sweep_root}")
    print(f"Conditions : {len(conditions)}  |  seeds/condition: {len(seeds)} "
          f"{seeds}  |  total runs: {len(runs)}")
    print(f"Dispatch   : {mode}")
    if slots is not None:
        print(f"GPU pool   : {pool}  |  concurrency: {len(slots)} slots "
              f"({args.jobs_per_gpu}/gpu)")
    print("=" * 70)

    if args.dry_run:
        for r in runs:
            print(f"[{_label(r)}] {' '.join(r['cmd'])}")
        print("\n(dry run - nothing launched)")
        return

    os.makedirs(sweep_root, exist_ok=True)
    _write_manifest(sweep_root, sweep, conditions, seeds, runs, mode, slots)

    if args.serial:
        done, failed = run_serial(runs)
    else:
        done, failed = run_scheduler(runs, slots, log_dir)

    print("\n" + "=" * 70)
    print(f"Completed  : {done - failed}/{len(runs)} runs "
          f"({failed} failed).")
    if failed and not args.serial:
        print(f"Failed run logs are in {log_dir}/ (see FAIL lines above).")
    print(f"Manifest   : {os.path.join(sweep_root, 'manifest.txt')}")
    print(f"Metrics    : {sweep_root}/<condition>/seed<k>/metrics_seed<k>.txt")
    print(f"Analyze with:  uv run python analyze_sweep.py {sweep_root}")
    sys.exit(1 if failed else 0)


def _write_manifest(sweep_root, sweep, conditions, seeds, runs, mode, slots):
    """Record the planned grid so aggregation can find every metrics file."""
    path = os.path.join(sweep_root, "manifest.txt")
    with open(path, "w") as f:
        f.write(f"# sweep: {sweep}\n")
        f.write(f"# created: {_dt.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# argv: {' '.join(sys.argv)}\n")
        f.write(f"# dispatch: {mode}\n")
        if slots is not None:
            f.write(f"# gpu slots: {slots}\n")
        f.write(f"# seeds: {seeds}\n")
        f.write(f"# conditions: {len(conditions)}  runs: {len(runs)}\n\n")
        for r in runs:
            f.write(
                f"{condition_tag(r['configs'])}\tseed{r['seed']}\t"
                f"{r['metrics_path']}\t{' '.join(r['cmd'])}\n"
            )
    print(f"Manifest written to {path}")


if __name__ == "__main__":
    main()
