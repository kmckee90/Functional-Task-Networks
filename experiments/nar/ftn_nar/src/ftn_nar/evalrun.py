"""Final evaluation of a completed run directory.

Produces eval/eval.json containing:
  retention: support-conditioned metrics for every seen task (ID val + test)
  held_out/<mode>: induction results on identifiable and random support sets
FTN-bank runs additionally get the 'retrieval' mode from their stored medoid
prototypes -- that is the FTN_RETRIEVAL baseline (same checkpoint, no training).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import algorithms as A
from .config import load_config
from .evaluator import EvalCache, evaluate_program, held_out_evaluation
from .trainer import Trainer

PRIMARY_MODE = {
    "FTN_FULL": "ftn", "FTN_NO_REPLAY": "ftn", "FTN_RANDOM_MASK_TRAIN": "ftn",
    "JOINT_MULTITASK_FTN": "ftn", "FTN_PER_TASK_MASK": "ftn", "SOFT_MIXTURE": "soft",
    "DENSE_ACTIVE_MATCHED": "dense", "ORACLE_COMPOSITION": "oracle",
    "DISJOINT_TASK_MODULES": "retrieval", "SINGLE_TASK": "oracle",
}


def _medoid_prototypes(tr: Trainer) -> list[dict] | None:
    protos = []
    for task, entry in tr.prototypes.items():
        if "medoid_ext" not in entry:
            return None
        protos.append({"mask_ext": entry["medoid_ext"], "mask_red": entry["medoid_red"],
                       "task": entry["task"]})
    return protos or None


def _disjoint_prototypes(tr: Trainer) -> list[dict]:
    return [{"mask_ext": tr._disjoint_mask(b)[0].cpu().numpy().tolist(),
             "mask_red": tr._disjoint_mask(b)[1].cpu().numpy().tolist(),
             "task": A.ALGORITHM_NAMES[tr.tasks[b]]}
            for b in range(len(tr.tasks))]


def evaluate_run(run_dir: str, do_local_swap: bool = False, n_random_sets: int | None = None,
                 skip_retention: bool = False) -> dict:
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    tr = Trainer(cfg, resume=True, run_dir=str(run_dir))
    model = tr.model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = tr.device
    cache = tr.eval_cache
    method = cfg.method
    t0 = time.time()
    out: dict = {"run": run_dir.name, "method": method,
                 "held_out": A.ALGORITHM_NAMES[cfg.held_out], "seed": cfg.seed}

    # ---- retention on seen tasks (support-conditioned; no task identity) ----
    if not skip_retention:
        retention = {}
        for task in tr.tasks:
            prog = tr._infer_for_task(task, support_id=0)
            entry = {}
            for cond in ("id_val", "id_test"):
                entry[cond] = evaluate_program(model, prog, cache.batches(task, cond),
                                               cfg, device)
            retention[A.ALGORITHM_NAMES[task]] = entry
        out["retention"] = retention

    # ---- held-out induction ----
    mode = PRIMARY_MODE[method]
    protos = None
    if mode == "retrieval":  # DISJOINT_TASK_MODULES
        protos = _disjoint_prototypes(tr)
    out["held_out_eval"] = {}
    # oracle inference ignores the support set (uses true modules) -> deterministic,
    # so one support set is sufficient and 10 would be identical.
    n_sets = 1 if mode == "oracle" else None
    out["held_out_eval"][mode] = held_out_evaluation(
        model, cfg, cfg.seed, cfg.held_out, mode, protos, device, cache,
        identifiable=True, do_local_swap=do_local_swap, n_support_sets=n_sets)
    n_rand = cfg.n_eval_support_sets if n_random_sets is None else n_random_sets
    if n_rand:
        out["held_out_eval"][mode + "_random_support"] = held_out_evaluation(
            model, cfg, cfg.seed, cfg.held_out, mode, protos, device, cache,
            identifiable=False, n_support_sets=n_rand)

    # FTN_RETRIEVAL baseline from the same checkpoint (design 11.3)
    if mode in ("ftn",):
        med = _medoid_prototypes(tr)
        if med:
            out["held_out_eval"]["retrieval"] = held_out_evaluation(
                model, cfg, cfg.seed, cfg.held_out, "retrieval", med, device, cache,
                identifiable=True)

    out["boundary_records"] = tr.boundary_records
    out["wall_s"] = time.time() - t0
    ev_dir = run_dir / "eval"
    ev_dir.mkdir(exist_ok=True)
    (ev_dir / "eval.json").write_text(json.dumps(out, indent=1))
    return out
