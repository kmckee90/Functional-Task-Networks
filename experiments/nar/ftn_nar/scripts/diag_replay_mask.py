"""Discriminate mask-selection drift vs subnetwork overwrite for a finished FTN run.

Dev-fold FTN_FULL kept replay loss low all the way through training yet reload
retention (fresh support-conditioned masks) is ~0. Evaluate each seen task on the
same fresh id_val graphs under three programs:

  fresh  - freshly configured mask, identical to the reload-eval retention path
  proto  - the boundary medoid prototype stored at the end of the task's own block
  replay - the newest replay-item mask for the task (what replay training ran under)

proto/replay high while fresh low -> mask-selection drift (task weights intact, the
eval-time configurer just lands on the wrong subnetwork). All three low -> the task
subnetwork itself was overwritten (interference / replay-buffer memorization).

Usage: uv run python scripts/diag_replay_mask.py <run_dir>
"""
import sys

import numpy as np
import torch

from ftn_nar import algorithms as A
from ftn_nar.config import load_config
from ftn_nar.evaluator import evaluate_program
from ftn_nar.model import Program
from ftn_nar.trainer import Trainer

run_dir = sys.argv[1].rstrip("/")
cfg = load_config(run_dir + "/config.yaml")
tr = Trainer(cfg, resume=True, run_dir=run_dir)
model = tr.model.eval()
for p in model.parameters():
    p.requires_grad_(False)
dev = tr.device

newest_replay: dict[int, object] = {}
for it in tr.replay.items:
    prev = newest_replay.get(it.algorithm_id)
    if prev is None or it.insertion_step > prev.insertion_step:
        newest_replay[it.algorithm_id] = it


def prog_from(me, mr) -> Program:
    return Program(
        gate_ext=torch.as_tensor(np.asarray(me), dtype=torch.float32, device=dev),
        gate_red=torch.as_tensor(np.asarray(mr), dtype=torch.float32, device=dev))


def iou(a, b) -> float:
    a = np.asarray(a).ravel() > 0.5
    b = np.asarray(b).ravel() > 0.5
    return float((a & b).sum() / max(1, (a | b).sum()))


print(f"{run_dir.split('/')[-1][:50]}  block order: "
      f"{[A.ALGORITHM_NAMES[t] for t in tr.tasks]}")
for task in tr.tasks:
    name = A.ALGORITHM_NAMES[task]
    proto = tr.prototypes.get(task)
    arms: dict[str, Program] = {"fresh": tr._infer_for_task(task, support_id=0)}
    if proto is not None:
        arms["proto"] = prog_from(proto["medoid_ext"], proto["medoid_red"])
    rit = newest_replay.get(task)
    if rit is not None:
        arms["replay"] = prog_from(rit.mask_ext, rit.mask_red)

    batches = tr.eval_cache.batches(task, "id_val")
    cells = []
    for label, prog in arms.items():
        r = evaluate_program(model, prog, batches, cfg, dev)
        cells.append(f"{label} node={r['node_acc']:.3f} "
                     f"exact={r['exact_graph_value_and_pointer_acc']:.3f}")
    agree = []
    fe = arms["fresh"].gate_ext.detach().cpu().numpy()
    fr = arms["fresh"].gate_red.detach().cpu().numpy()
    if proto is not None:
        agree.append(f"IoU(fresh,proto)={iou(fe, proto['medoid_ext']):.2f}/"
                     f"{iou(fr, proto['medoid_red']):.2f}")
    if rit is not None and proto is not None:
        agree.append(f"IoU(proto,replay)={iou(proto['medoid_ext'], rit.mask_ext):.2f}/"
                     f"{iou(proto['medoid_red'], rit.mask_red):.2f}")
    print(f"{name:14} " + " | ".join(cells) + "   " + " ".join(agree))
