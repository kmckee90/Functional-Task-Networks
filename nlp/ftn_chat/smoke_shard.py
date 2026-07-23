"""Smoke test for the sharded 7B + large LoRA bank (no checkpoint save).

Validates: multi-GPU backbone sharding with banks on the last 25% of layers
(one banked layer per GPU), the cross-device mask fix, one reconfig inner
loop, the teacher (zero-mask, context-in-prompt) pass, and one context-
distillation train step + optimizer step. Prints per-GPU peak memory.

    uv run python smoke_shard.py
"""
from __future__ import annotations

import time

import torch

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

from config import CFG
from data import build_task, format_distill
from model import build, lora_params
from perplexity_configurer import PerplexityConfigurer
from train import _collate, distill_loss, teacher_answer_logits

print(f"visible GPUs: {torch.cuda.device_count()}")
print(f"model={CFG.model_name}  H={CFG.H} ({CFG.grid_dim}x{CFG.grid_dim}) "
      f"k={CFG.k} rank={CFG.rank} layer_frac={CFG.lora_layer_frac}")

t0 = time.time()
model, tok, ctrl, wrappers, lora_cfg = build(CFG)
print(f"  build done in {time.time()-t0:.0f}s")
for p in lora_params(wrappers):
    p.requires_grad = True
opt = torch.optim.Adam(lora_params(wrappers), lr=CFG.lr, foreach=False)
cfgr = PerplexityConfigurer(CFG, ctrl)

# One real source chunk as distillation context (no qa cache needed).
data = build_task(CFG, tok, "base_ftn")
context = tok.decode(data.train[0].tolist())
ex = format_distill(
    "What does the K parameter control in the base FTN configuration?",
    "K is the kWTA budget: the number of active mask units selected per "
    "reconfiguration, i.e. how many hidden units the task mask switches on.",
    context, tok, CFG,
)
print(f"  example: student {len(ex['ids'])} tok, teacher {len(ex['t_ids'])} tok")

dev = torch.device("cuda")
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
model.train()
for step in range(2):
    m, _, _ = cfgr(model, ex["recfg"].to(dev).unsqueeze(0), steps=3)
    s_batch = _collate([(ex["ids"], ex["lbl"])], pad_id, dev)
    t_batch = _collate([(ex["t_ids"], ex["t_lbl"])], pad_id, dev)
    t_sel = teacher_answer_logits(model, ctrl, t_batch)
    opt.zero_grad()
    loss = distill_loss(model, ctrl, m, s_batch, t_sel, CFG)
    loss.backward()
    opt.step()
    print(f"  step {step}: kl={float(loss):.3f}  active_atoms={int((m>0).sum())}")

print("peak memory per visible GPU:")
for i in range(torch.cuda.device_count()):
    print(f"  cuda:{i}  {torch.cuda.max_memory_allocated(i)/1e9:.1f} GB")
print("SMOKE OK")
