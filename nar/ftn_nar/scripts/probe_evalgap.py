"""Diagnose train/eval accuracy gap for a SINGLE_TASK run."""
import sys

import numpy as np
import torch

from ftn_nar import generator as G
from ftn_nar.batch import build_batch
from ftn_nar.config import load_config
from ftn_nar.evaluator import EvalCache, evaluate_program
from ftn_nar.model import Program, oracle_masks
from ftn_nar.trainer import Trainer

run_dir = sys.argv[1]
cfg = load_config(run_dir + "/config.yaml")
tr = Trainer(cfg, resume=True)
model = tr.model.eval()
dev = tr.device
task = cfg.held_out
me, mr = oracle_masks(task, dev)
prog = Program(gate_ext=me, gate_red=mr)

# (a) id_val big chunks
cache = EvalCache(cfg.seed, cfg)
m = evaluate_program(model, prog, cache.batches(task, "id_val"), cfg, dev)
print("id_val chunked:", {k: round(v, 4) for k, v in m.items() if "acc" in k or "mae" in k})

# (b) same graphs, batches of 32
graphs = [g for b in cache.batches(task, "id_val") for g in []]  # placeholder
from ftn_nar.evaluator import eval_graphs
gs = eval_graphs(cfg.seed, task, "id_val", 256)
batches32 = [build_batch(gs[i:i+32], task, with_trace=False) for i in range(0, 256, 32)]
m = evaluate_program(model, prog, batches32, cfg, dev)
print("id_val batch32:", {k: round(v, 4) for k, v in m.items() if "acc" in k or "mae" in k})

# (c) fresh train-stream graphs, batches of 32
gs = [G.generate("train", cfg.seed, "train_qry", cfg.held_out, 1900 + i // 32, i % 32)
      for i in range(256)]
batches = [build_batch(gs[i:i+32], task, with_trace=False) for i in range(0, 256, 32)]
m = evaluate_program(model, prog, batches, cfg, dev)
print("train-stream batch32:", {k: round(v, 4) for k, v in m.items() if "acc" in k or "mae" in k})

# per-depth breakdown on id_val
b = build_batch(gs, task, with_trace=False).to(dev)
with torch.no_grad():
    out = model(b, prog, edge_mode="hard")
err = (out["final_values"] - b.final_values).abs()
ok = (torch.round(out["final_values"]) == b.final_values) | b.source_mask
depth_n = b.depth_per_graph[b.graph_index_per_node]
for d in range(3, 7):
    sel = (depth_n == d) & ~b.source_mask
    if sel.any():
        print(f"depth {d}: node_acc={float(ok[sel].float().mean()):.4f} "
              f"mae={float(err[sel].mean()):.4f} n={int(sel.sum())}")
# layer breakdown
lay = torch.tensor(np.concatenate([g.layer_of for g in gs]), device=dev)
for l in range(0, 7):
    sel = (lay == l) & ~b.source_mask
    if sel.any():
        print(f"layer {l}: node_acc={float(ok[sel].float().mean()):.4f} "
              f"mae={float(err[sel].mean()):.4f} n={int(sel.sum())}")
