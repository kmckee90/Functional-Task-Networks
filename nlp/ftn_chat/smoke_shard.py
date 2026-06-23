"""Smoke test for the sharded 7B + large LoRA bank (no checkpoint save).

Validates: multi-GPU backbone sharding, bank atoms co-located per layer, the
cross-device mask fix, one reconfig inner loop, and one causal-LM train step +
optimizer step. Prints per-GPU peak memory. Run with the GPUs you intend to
train on, e.g.:  CUDA_VISIBLE_DEVICES=0,2,4,5 uv run python smoke_shard.py
"""
from __future__ import annotations

import time

import torch

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

from config import CFG
from data import build_examples, build_task
from model import build, lora_params
from perplexity_configurer import PerplexityConfigurer

print(f"visible GPUs: {torch.cuda.device_count()}")
print(f"model={CFG.model_name}  H={CFG.H} ({CFG.grid_dim}x{CFG.grid_dim}) "
      f"k={CFG.k} rank={CFG.rank}")

t0 = time.time()
model, tok, ctrl, wrappers, lora_cfg = build(CFG)
print(f"  build done in {time.time()-t0:.0f}s")
for p in lora_params(wrappers):
    p.requires_grad = True
opt = torch.optim.Adam(lora_params(wrappers), lr=CFG.lr, foreach=False)
cfgr = PerplexityConfigurer(CFG, ctrl)

# Tiny real data from one of the two-file domains (exercises code + doc globs).
data = build_task(CFG, tok, "genetic_programming")
ex = build_examples(CFG, tok, "genetic_programming", data)
raw = ex["raw"]
print(f"  genetic_programming: {len(raw)} raw chunks")

dev = torch.device("cuda")
model.train()
for step in range(2):
    e = raw[step % len(raw)]
    m, _, _ = cfgr(model, e["recfg"].to(dev).unsqueeze(0), steps=3)
    ctrl.set_mask(m)
    opt.zero_grad()
    loss = model(input_ids=e["ids"].to(dev).unsqueeze(0),
                 labels=e["lbl"].to(dev).unsqueeze(0)).loss
    loss.backward()
    opt.step()
    print(f"  step {step}: nll={float(loss):.3f}  active_atoms={int((m>0).sum())}")

print("peak memory per visible GPU:")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i}  {torch.cuda.max_memory_allocated(i)/1e9:.1f} GB")
print("SMOKE OK")
