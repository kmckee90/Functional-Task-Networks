"""Break down held-out ORACLE failure: value vs pointer, register vs readout, by layer."""
import glob
import json
import sys

import torch

from ftn_nar import algorithms as A
from ftn_nar.config import load_config
from ftn_nar.evaluator import eval_graphs
from ftn_nar.batch import build_batch
from ftn_nar.model import Program, oracle_masks
from ftn_nar.trainer import Trainer

run = sys.argv[1]
cfg = load_config(run + "/config.yaml")
tr = Trainer(cfg, resume=True)
model = tr.model.eval()
dev = tr.device
task = cfg.held_out
prog = Program(*oracle_masks(task, dev))

gs = eval_graphs(cfg.seed, task, "id_val", 256)
b = build_batch(gs, task, with_trace=False).to(dev)
with torch.no_grad():
    out = model(b, prog, edge_mode="hard")
nonsrc = ~b.source_mask
val_ok = (torch.round(out["final_values"]) == b.final_values) | b.source_mask
mae = float((out["final_values"] - b.final_values).abs()[nonsrc].mean())
# pointer accuracy
from ftn_nar import ops
ph = ops.segment_argmax_onehot(out["final_edge_scores"], b.edge_index[1], b.num_nodes,
                               out["final_edge_valid"])
ok_e = (ph > 0.5) & b.valid_pred_edge_mask
node_ptr_ok = ops.segment_any(ok_e, b.edge_index[1], b.num_nodes) | b.source_mask
print(f"held-out {A.ALGORITHM_NAMES[task]}: node_val_acc={float(val_ok[nonsrc].float().mean()):.3f} "
      f"mae={mae:.3f} node_ptr_acc={float(node_ptr_ok[nonsrc].float().mean()):.3f}")
lay = torch.tensor([l for g in gs for l in g.layer_of], device=dev)
for L in range(1, 7):
    sel = (lay == L) & nonsrc
    if sel.any():
        print(f"  layer {L}: val_acc={float(val_ok[sel].float().mean()):.3f} "
              f"ptr_acc={float(node_ptr_ok[sel].float().mean()):.3f} "
              f"mae={float((out['final_values']-b.final_values).abs()[sel].mean()):.3f} n={int(sel.sum())}")
