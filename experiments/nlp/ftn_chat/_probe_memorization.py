"""Throwaway diagnostic: does the bank actually memorize the cscg source under
the EXACT training reconfiguration params? Imports app.py to reuse its loaded
model/bank/refs (Gradio UI only launches under __main__, so import is safe)."""
from __future__ import annotations

import torch

import app  # loads MODEL/TOK/CTRL/CFGR/REFS, sets default device cuda
from config import CFG
from data import build_task
from spatial import kwta

MODEL, TOK, CTRL, CFGR, REFS, DEV = app.MODEL, app.TOK, app.CTRL, app.CFGR, app.REFS, app.DEV

# --- EXACT training reconfig params (train.py -> cfgr(model, recfg, steps=30) ->
#     iterate() with all other args defaulting to config) -----------------------
RP = dict(
    steps=CFG.train_reconfig_steps,   # 30
    reconfig_lr=CFG.reconfig_lr,      # 0.3
    lateral_steps=CFG.lateral_steps,  # 1
    k=CFG.k,                          # 128
    init_a=0.0,
    init_b=1e-3,                      # iterate() default used by training
)
print(f"== reconfig params (exact training) == {RP}", flush=True)


def reconfigure(ids):
    """Drive iterate() to completion (mirrors PerplexityConfigurer.__call__) and
    return the final mask, using the exact training params above."""
    last = None
    for m in CFGR.iterate(MODEL, ids, **RP, init_mask=None):
        last = m
    return last.detach()


def overlap(a, b, k):
    ka, kb = kwta(a.float(), k).bool(), kwta(b.float(), k).bool()
    return float((ka & kb).sum()) / k


@torch.no_grad()
def teacher_forced(ids):
    out = MODEL(input_ids=ids, labels=ids)
    logits = out.logits[:, :-1, :]
    tgt = ids[:, 1:]
    acc = (logits.argmax(-1) == tgt).float().mean().item()
    return float(out.loss), acc


@torch.no_grad()
def free_continue(prefix_ids, n_new):
    # Plain greedy, NO repetition penalty -> best case for verbatim recall
    out = MODEL.generate(
        input_ids=prefix_ids,
        attention_mask=torch.ones_like(prefix_ids),
        max_new_tokens=n_new,
        do_sample=False,
        pad_token_id=TOK.eos_token_id,
    )
    return out[:, prefix_ids.shape[1]:]


key = "cscg"
data = build_task(CFG, TOK, key)
ref = REFS.get(key)
print(f"\n==== {key}: {len(data.train)} train chunks, chunk_tokens={CFG.chunk_tokens} ====", flush=True)

for ci in [0, 1, 2]:
    chunk = data.train[ci]
    ids = chunk.unsqueeze(0).to(DEV)

    # Reconfigure exactly as training did: on the FULL raw chunk.
    m_src = reconfigure(ids)
    CTRL.set_mask(m_src)

    nll, acc = teacher_forced(ids)
    ov_ref = overlap(m_src, ref.to(DEV), CFG.k) if ref is not None else float("nan")

    # Free-running verbatim continuation from the first 48 tokens.
    P = min(48, ids.shape[1] // 2)
    gen_ids = free_continue(ids[:, :P], n_new=ids.shape[1] - P)
    true_cont = ids[:, P:]
    g, t = gen_ids[0], true_cont[0]
    n = min(len(g), len(t))
    cont_acc = (g[:n] == t[:n]).float().mean().item() if n else float("nan")

    print(f"\n--- chunk {ci} (len={ids.shape[1]}) ---", flush=True)
    print(f"  teacher-forced  nll={nll:.4f}  top1_acc={acc:.3f}", flush=True)
    print(f"  mask overlap with cscg ref = {ov_ref:.2f}", flush=True)
    print(f"  free-continue verbatim top1_acc (greedy, {P}->{ids.shape[1]}) = {cont_acc:.3f}", flush=True)
    print(f"  PROMPT (first {P} tok):\n    {TOK.decode(ids[0,:P])!r}", flush=True)
    print(f"  TRUE  continuation:\n    {TOK.decode(t[:80])!r}", flush=True)
    print(f"  MODEL continuation:\n    {TOK.decode(g[:80])!r}", flush=True)

# ---- Contrast: reconfigure on a natural-language question (what app.py does) ----
print("\n==== ROUTING CONTRAST: reconfigure on an NL question ====", flush=True)
q = "Explain how the CSCG model handles hard evidence."
qids = TOK(q, return_tensors="pt")["input_ids"].to(DEV)
m_q = reconfigure(qids)
m_src0 = reconfigure(data.train[0].unsqueeze(0).to(DEV))
print(f"  question: {q!r}", flush=True)
print(f"  overlap(NL-question mask, cscg-source mask) = {overlap(m_q, m_src0, CFG.k):.2f}", flush=True)
if ref is not None:
    print(f"  overlap(NL-question mask, cscg ref)         = {overlap(m_q, ref.to(DEV), CFG.k):.2f}", flush=True)
    print(f"  overlap(cscg-source mask, cscg ref)         = {overlap(m_src0, ref.to(DEV), CFG.k):.2f}", flush=True)
print("\n[done]", flush=True)
