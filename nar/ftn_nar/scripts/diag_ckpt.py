"""Load a checkpoint's weights directly (bypass Trainer hash) and eval held-out with
current model code. round_register override via env RR=0/1."""
import glob
import os
import sys

import torch

from ftn_nar import algorithms as A
from ftn_nar import ops
from ftn_nar.batch import build_batch
from ftn_nar.config import load_config
from ftn_nar.evaluator import eval_graphs
from ftn_nar.model import DPModel, Program, oracle_masks, oracle_model_cfg

run = sys.argv[1].rstrip("/")
cfg = load_config(run + "/config.yaml")
ck_path = sorted(glob.glob(run + "/checkpoints/checkpoint_final.pt")
                 or glob.glob(run + "/checkpoints/checkpoint_task*.pt"))[-1]
ck = torch.load(ck_path, map_location="cuda", weights_only=False)
mcfg = oracle_model_cfg(cfg.model)
if "RR" in os.environ:
    mcfg.round_register = os.environ["RR"] == "1"
dev = torch.device("cuda")
model = DPModel(mcfg).to(dev).eval()
model.load_state_dict(ck["model"])
task = cfg.held_out
prog = Program(*oracle_masks(task, dev))
print(f"{run.split('/')[-1][:40]}  round_register={model.round_register}")
for cond, kw in [("id_val", {}), ("ood_width_64", {}), ("ood_depth_16", {})]:
    gs = eval_graphs(cfg.seed, task, cond, 256)
    b = build_batch(gs, task, with_trace=False).to(dev)
    with torch.no_grad():
        out = model(b, prog, edge_mode="hard")
    nonsrc = ~b.source_mask
    vok = (torch.round(out["final_values"]) == b.final_values) | b.source_mask
    gidx = b.graph_index_per_node
    gok = torch.ones(b.num_graphs, dtype=torch.long, device=dev)
    gok.scatter_reduce_(0, gidx, vok.long(), reduce="amin", include_self=True)
    print(f"  {cond:14} node_acc={float(vok[nonsrc].float().mean()):.3f} "
          f"graph_exact={float((gok>0).float().mean()):.3f}")
