"""Apples-to-apples: recompute the training-step loss with final checkpoint params."""
import sys

import torch

from ftn_nar import algorithms as A
from ftn_nar import generator as G
from ftn_nar.batch import build_batch
from ftn_nar.config import load_config
from ftn_nar.losses import model_training_loss
from ftn_nar.model import Program, oracle_masks
from ftn_nar.trainer import Trainer

run_dir = sys.argv[1]
cfg = load_config(run_dir + "/config.yaml")
tr = Trainer(cfg, resume=True)
model = tr.model
dev = tr.device
task = cfg.held_out
prog = Program(*oracle_masks(task, dev))

for gstep in (1800, 1900, 1990):
    qry = [G.generate("train", cfg.seed, "train_qry", cfg.held_out, gstep, i)
           for i in range(cfg.data.query_batch_size)]
    labels = [A.execute(g, task) for g in qry]
    b = build_batch(qry, task, with_trace=True, labels=labels).to(dev)
    model.train()
    out = model(b, prog, edge_mode="ste", return_trace=True)
    parts = model_training_loss(out, b, cfg.loss)
    mae = float((out["final_values"] - b.final_values).abs()[~b.source_mask].mean())
    print(f"gstep={gstep} fv_loss={float(parts['final_value']):.6f} mae={mae:.4f} "
          f"loss={float(parts['loss']):.5f}")
