"""Isolate composition-gap from depth-extrapolation-gap: evaluate a SEEN task and the
HELD-OUT task, both with true oracle modules, at ID and OOD depth/width."""
import glob
import sys

import torch

from ftn_nar import algorithms as A
from ftn_nar.batch import build_batch
from ftn_nar.config import load_config
from ftn_nar.evaluator import eval_graphs
from ftn_nar.model import DPModel, Program, oracle_masks, oracle_model_cfg

run = sys.argv[1].rstrip("/")
cfg = load_config(run + "/config.yaml")
ck = torch.load(sorted(glob.glob(run + "/checkpoints/checkpoint_final.pt"))[0],
                map_location="cuda", weights_only=False)
mcfg = oracle_model_cfg(cfg.model)
dev = torch.device("cuda")
model = DPModel(mcfg).to(dev).eval()
model.load_state_dict(ck["model"])
tasks = ck["tasks"]  # seen task ids
held = cfg.held_out
seen = tasks[0]


@torch.no_grad()
def ev(task, cond):
    gs = eval_graphs(cfg.seed, task, cond, 128)
    b = build_batch(gs, task, with_trace=False).to(dev)
    out = model(b, Program(*oracle_masks(task, dev)), edge_mode="hard")
    nonsrc = ~b.source_mask
    vok = (torch.round(out["final_values"]) == b.final_values) | b.source_mask
    gidx = b.graph_index_per_node
    gok = torch.ones(b.num_graphs, dtype=torch.long, device=dev)
    gok.scatter_reduce_(0, gidx, vok.long(), reduce="amin", include_self=True)
    mae = float((out["final_values"] - b.final_values).abs()[nonsrc].mean())
    return float(vok[nonsrc].float().mean()), float((gok > 0).float().mean()), mae


for label, task in [("SEEN " + A.ALGORITHM_NAMES[seen], seen),
                    ("HELD " + A.ALGORITHM_NAMES[held], held)]:
    print(label)
    for cond in ["id_val", "ood_width_64", "ood_depth_8", "ood_depth_16"]:
        na, ga, mae = ev(task, cond)
        print(f"  {cond:14} node={na:.3f} graph={ga:.3f} mae={mae:.3f}")
