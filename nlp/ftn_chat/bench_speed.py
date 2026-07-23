"""Micro-benchmark for the Qwen3.6 backbone: teacher-answer generation (the
synthesis bottleneck) and one distillation train step. Run before/after
installing flash-linear-attention to quantify the DeltaNet fast-path speedup.

    uv run python bench_speed.py
"""
from __future__ import annotations

import time

import torch

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

from config import CFG
from data import build_task, format_distill, teacher_user_msg
from model import build, lora_params
from perplexity_configurer import PerplexityConfigurer
from train import _collate, distill_loss, teacher_answer_logits

try:
    import fla  # noqa: F401
    fast = "flash-linear-attention PRESENT"
except Exception:
    fast = "flash-linear-attention ABSENT (torch fallback)"
print(f"fast path: {fast}")

model, tok, ctrl, wrappers, _ = build(CFG)
for p in lora_params(wrappers):
    p.requires_grad = True
opt = torch.optim.Adam(lora_params(wrappers), lr=CFG.lr, foreach=False)
cfgr = PerplexityConfigurer(CFG, ctrl)
dev = torch.device("cuda")
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

data = build_task(CFG, tok, "base_ftn")
context = tok.decode(data.train[0].tolist())
q = "What does the K parameter control in the base FTN configuration?"

# --- Generation (synthesis bottleneck): thinking answer WITH context ---
msg = teacher_user_msg(context, q)
text = tok.apply_chat_template([{"role": "user", "content": msg}],
                               add_generation_prompt=True, tokenize=False,
                               enable_thinking=True)
enc = tok(text, return_tensors="pt", add_special_tokens=False).to(dev)
model.eval()
with torch.no_grad():  # warmup (triton JIT compiles on first call)
    model.generate(**enc, max_new_tokens=32, do_sample=False,
                   pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()
t0 = time.time()
with torch.no_grad():
    out = model.generate(**enc, max_new_tokens=512, do_sample=True,
                         temperature=0.6, pad_token_id=tok.eos_token_id)
torch.cuda.synchronize()
gen_tok = out.shape[1] - enc["input_ids"].shape[1]
dt = time.time() - t0
print(f"GEN: {gen_tok} tok in {dt:.1f}s -> {gen_tok/dt:.1f} tok/s")

# --- One distillation train step ---
ex = format_distill(q, "K is the kWTA budget: the number of active mask units "
                    "selected per reconfiguration.", context, tok, CFG)
s_batch = _collate([(ex["ids"], ex["lbl"])], pad_id, dev)
t_batch = _collate([(ex["t_ids"], ex["t_lbl"])], pad_id, dev)
model.train()
for i in range(2):  # 1 warmup + 1 timed
    torch.cuda.synchronize(); t0 = time.time()
    m, _, _ = cfgr(model, ex["recfg"].to(dev).unsqueeze(0), steps=CFG.train_reconfig_steps)
    t_sel = teacher_answer_logits(model, ctrl, t_batch)
    opt.zero_grad()
    loss = distill_loss(model, ctrl, m, s_batch, t_sel, CFG)
    loss.backward(); opt.step()
    torch.cuda.synchronize()
    tag = "warmup" if i == 0 else "TRAIN STEP"
    print(f"{tag}: {time.time()-t0:.1f}s  (reconfig {CFG.train_reconfig_steps} + teacher + student)")
print("BENCH OK")
