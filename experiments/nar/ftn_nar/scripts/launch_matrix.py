#!/usr/bin/env python
"""Schedule fold x seed x method runs across GPUs (one run per GPU slot, design 20.3).

Usage (on the pod, from /home/dev/workspace):
  uv run python scripts/launch_matrix.py --methods FTN_FULL ORACLE_COMPOSITION \
      --seeds 0 1 2 --folds all --out /home/dev/runs --gpus 0 1 2 3 --slots-per-gpu 1

Runs are skipped when the run directory already contains eval/eval.json (resumable
matrix). Each run's stdout goes to <out>/launcher_logs/.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ALGOS = ["MIN_ADD", "MAX_ADD", "MIN_SUB", "MAX_SUB",
         "MIN_MAX_EDGE", "MAX_MAX_EDGE", "MIN_MIN_EDGE", "MAX_MIN_EDGE"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", nargs="+", default=["all"])
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--slots-per-gpu", type=int, default=1)
    ap.add_argument("--out", default="/home/dev/runs")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--config", default=None, help="experiment yaml passed to every run")
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--extra", default="", help="extra args string, shlex-split")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    folds = ALGOS if args.folds == ["all"] else args.folds
    jobs = []
    for method, fold, seed in itertools.product(args.methods, folds, args.seeds):
        jobs.append((method, fold, seed))

    out = Path(args.out)
    logdir = out / "launcher_logs"
    logdir.mkdir(parents=True, exist_ok=True)

    def already_done(method, fold, seed):
        for d in out.glob(f"{method}__heldout-{fold}__seed-{seed}__confighash-*"):
            if (d / "eval" / "eval.json").exists():
                return True
        return False

    def resumable(method, fold, seed):
        for d in out.glob(f"{method}__heldout-{fold}__seed-{seed}__confighash-*"):
            if (d / "checkpoints" / "latest.txt").exists():
                return True
        return False

    queue = [(m, f, s) for m, f, s in jobs if not already_done(m, f, s)]
    print(f"{len(jobs)} jobs, {len(jobs) - len(queue)} already done, {len(queue)} to run")
    if args.dry_run:
        for j in queue:
            print("  ", j)
        return

    slots = [(g, s) for g in args.gpus for s in range(args.slots_per_gpu)]
    running: dict[tuple, tuple] = {}  # slot -> (proc, job, logfile)
    results = []
    qi = 0
    while qi < len(queue) or running:
        for slot in list(slots):
            if slot in running or qi >= len(queue):
                continue
            method, fold, seed = queue[qi]
            qi += 1
            cmd = [sys.executable, "-m", "ftn_nar.cli", "train", "--method", method,
                   "--held-out", fold, "--seed", str(seed), "--out", str(out)]
            if resumable(method, fold, seed):
                cmd.append("--resume")
            if args.steps:
                cmd += ["--steps", str(args.steps)]
            if args.config:
                cmd += ["--config", args.config]
            if args.tiny:
                cmd.append("--tiny")
            if args.extra:
                import shlex
                cmd += shlex.split(args.extra)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(slot[0]))
            lf = open(logdir / f"{method}__{fold}__s{seed}.log", "a")
            lf.write(f"\n==== launch {time.ctime()} slot={slot} ====\n{' '.join(cmd)}\n")
            lf.flush()
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
            running[slot] = (proc, (method, fold, seed), lf)
            print(f"[{time.strftime('%H:%M:%S')}] start {method} {fold} s{seed} "
                  f"on gpu{slot[0]} ({qi}/{len(queue)})", flush=True)
        time.sleep(10)
        for slot, (proc, job, lf) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            lf.close()
            del running[slot]
            results.append({"job": job, "rc": rc, "t": time.time()})
            print(f"[{time.strftime('%H:%M:%S')}] done {job} rc={rc}", flush=True)
            (out / "launcher_status.json").write_text(json.dumps(
                {"done": results, "queued": len(queue) - qi,
                 "running": [str(j) for _, j, _ in running.values()]}, indent=1, default=str))

    fails = [r for r in results if r["rc"] != 0]
    print(f"ALL DONE: {len(results) - len(fails)} ok, {len(fails)} failed")
    for f in fails:
        print("  FAILED:", f["job"])


if __name__ == "__main__":
    main()
