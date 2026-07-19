"""Scratch probe: can the executor overfit 4 graphs to exactness? (stage-B risk check)"""
import sys
import time

import torch

from ftn_nar import generator as G
from ftn_nar.batch import build_batch
from ftn_nar.config import ExperimentCfg
from ftn_nar.losses import model_training_loss
from ftn_nar.model import DPModel, Program, oracle_masks, oracle_model_cfg

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 3e-3
n_graphs = int(sys.argv[3]) if len(sys.argv) > 3 else 4

cfg = ExperimentCfg()
cfg.model = oracle_model_cfg(cfg.model)  # full-size oracle model
dev = torch.device("cuda")
torch.manual_seed(0)
model = DPModel(cfg.model).to(dev)
graphs = [G.generate("train", 0, "probe", i) for i in range(n_graphs)]
b = build_batch(graphs, 0, with_trace=True).to(dev)
me, mr = oracle_masks(0, dev)
prog = Program(gate_ext=me, gate_red=mr)
opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

t0 = time.time()
for i in range(steps):
    out = model(b, prog, edge_mode="ste", return_trace=True)
    parts = model_training_loss(out, b, cfg.loss)
    opt.zero_grad()
    parts["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if i % 200 == 0 or i == steps - 1:
        with torch.no_grad():
            o = model(b, prog, edge_mode="hard")
            ok = (torch.round(o["final_values"]) == b.final_values) | b.source_mask
            acc = float(ok.float().mean())
            nonsrc = ~b.source_mask
            mae = float((o["final_values"] - b.final_values).abs()[nonsrc].mean())
            rmae = float((o["final_register"] - b.final_values).abs()[nonsrc].mean())
            rok = (torch.round(o["final_register"]) == b.final_values) | b.source_mask
            racc = float(rok.float().mean())
        print(f"{i:5d} loss={float(parts['loss']):.5f} fv={float(parts['final_value']):.6f} "
              f"tv={float(parts['trace_value']):.6f} node_acc={acc:.3f} mae={mae:.3f} "
              f"reg_acc={racc:.3f} reg_mae={rmae:.3f} ({(time.time()-t0):.0f}s)", flush=True)
