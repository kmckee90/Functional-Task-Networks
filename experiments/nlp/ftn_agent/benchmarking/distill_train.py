"""Context-distillation block trainer for the LoCoMo experiment.

One block = one conversation session's self-study dataset (selfstudy.py).
Per step:
  1. perplexity-reconfigure the atom mask on one batch member's bare
     instruction (matches closed-book QA time),
  2. TEACHER pass (no grad, mask ZEROED = frozen backbone) on
     [session transcript + instruction + response],
  3. STUDENT pass (masked bank, NO transcript) on [instruction + response],
  4. loss = KL(teacher || student) on the response tokens (+ small CE),
  5. overlap-filtered CE replay over prior blocks (retention).

The `monolithic` mode is the single-LoRA ablation: identical parameterization
and data, but the mask is fixed all-ones and never reconfigured — isolating
what sparse routing adds.

Run state (bank/refs/replay/state.json) lives in a per-run dir on the PVC and
reuses train.py's load_state/save_state by pointing cfg.state_dir at it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # core ftn_agent

import torch
import torch.nn.functional as F

from selfstudy import TEACHER_SYS, TEACHER_USER


# --- example construction ------------------------------------------------------

def _prompt_ids(tok, sys_msg: str, user: str) -> list[int]:
    ids = tok.apply_chat_template(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}],
        add_generation_prompt=True, tokenize=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def make_examples(tok, data: dict, max_resp_tokens: int = 512) -> list[dict]:
    """data = selfstudy load_session_data() dict. The response token ids are
    IDENTICAL in the teacher and student sequences (appended to both prompts),
    so teacher logits align with student targets position-by-position."""
    conv_sys = TEACHER_SYS.format(a=data["meta"].get("a", ""), b=data["meta"].get("b", ""))
    sess_text = data["session_text"]
    out = []
    for it in data["items"]:
        resp = tok(it["response"], add_special_tokens=False)["input_ids"][:max_resp_tokens]
        resp = resp + [tok.eos_token_id]
        s_prompt = _prompt_ids(tok, it["student_sys"], it["instruction"])
        t_prompt = _prompt_ids(tok, conv_sys,
                               TEACHER_USER.format(session=sess_text,
                                                   instruction=it["instruction"]))
        recfg = tok(it["instruction"], add_special_tokens=False)["input_ids"][:512]
        t = lambda x: torch.tensor(x, dtype=torch.long, device="cpu")
        out.append({"student_ids": t(s_prompt + resp), "teacher_ids": t(t_prompt + resp),
                    "s_start": len(s_prompt), "t_start": len(t_prompt),
                    "n_resp": len(resp), "recfg": t(recfg)})
    return out


def _collate(seqs: list[torch.Tensor], pad_id: int, dev):
    L = max(int(s.size(0)) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=dev)
    attn = torch.zeros((len(seqs), L), dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        ids[i, : s.size(0)] = s.to(dev)
        attn[i, : s.size(0)] = 1
    return ids, attn


def _resp_logits(logits: torch.Tensor, batch, start_key: str):
    """Gather the logits that PREDICT each response token: positions
    [start-1, start+n_resp-1) per example. Returns (N_total_resp, vocab)."""
    picks = []
    for i, e in enumerate(batch):
        s, n = e[start_key], e["n_resp"]
        picks.append(logits[i, s - 1: s + n - 1])
    return torch.cat(picks, 0)


def _resp_targets(batch, dev):
    return torch.cat([e["student_ids"][e["s_start"]: e["s_start"] + e["n_resp"]]
                      for e in batch]).to(dev)


# --- the trainer ----------------------------------------------------------------

def distill_step(h, batch, opt, mask, ce_weight: float = 0.1,
                 kl_temp: float = 1.0) -> float:
    model, ctrl, tok = h.model, h.ctrl, h.tok
    dev = h.dev
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # teacher: frozen backbone (mask zeroed), sees the transcript
    t_ids, t_attn = _collate([e["teacher_ids"] for e in batch], pad, dev)
    ctrl.set_mask(torch.zeros_like(mask))
    with torch.no_grad():
        t_out = model(input_ids=t_ids, attention_mask=t_attn).logits
        t_logp = F.log_softmax(_resp_logits(t_out, batch, "t_start").float() / kl_temp, -1)
        del t_out

    # student: masked bank, closed-book
    ctrl.set_mask(mask)
    s_ids, s_attn = _collate([e["student_ids"] for e in batch], pad, dev)
    s_out = model(input_ids=s_ids, attention_mask=s_attn).logits
    s_resp = _resp_logits(s_out, batch, "s_start").float()
    del s_out
    s_logp = F.log_softmax(s_resp / kl_temp, -1)

    kl = F.kl_div(s_logp, t_logp.to(s_logp.device), log_target=True,
                  reduction="batchmean") * (kl_temp ** 2)
    loss = kl
    if ce_weight > 0:
        tgt = _resp_targets(batch, s_resp.device)
        loss = loss + ce_weight * F.cross_entropy(s_resp, tgt)

    opt.zero_grad()
    loss.backward()
    return float(kl.detach())


def train_session_block(cfg, h, opt, key: str, examples: list[dict],
                        replay_store: dict, epochs: int = 2, batch_size: int = 8,
                        rsteps: int = None, monolithic: bool = False,
                        replay_frac: float = 0.3, ce_weight: float = 0.1,
                        gen: torch.Generator = None, log=print):
    """Distill one session into the bank. Mutates bank in place; returns the
    per-step KL history."""
    model, ctrl, cfgr, dev = h.model, h.ctrl, h.cfgr, h.dev
    gen = gen or torch.Generator(device="cpu").manual_seed(0)
    rsteps = rsteps if rsteps is not None else cfg.train_reconfig_steps
    prior = [(ids, lbl, m) for tk, buf in replay_store.items() if tk != key
             for ids, lbl, m in buf]
    ones = torch.ones(cfg.H, device=dev)

    model.train()
    prev_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        pass

    steps = max(1, (len(examples) * epochs) // batch_size)
    hist = []
    try:
        for step in range(steps):
            # device="cpu" everywhere a cpu Generator is used: the entry point
            # sets torch default device to cuda (design.md CS-style gotcha).
            idx = torch.randint(0, len(examples), (batch_size,), generator=gen, device="cpu")
            batch = [examples[int(i)] for i in idx]

            if monolithic:
                mask = ones
            else:
                r = batch[int(torch.randint(0, len(batch), (1,), generator=gen, device="cpu"))]
                mask, _, _ = cfgr(model, r["recfg"].to(dev).unsqueeze(0), steps=rsteps)

            kl = distill_step(h, batch, opt, mask, ce_weight=ce_weight)

            # retention: CE replay over prior blocks, overlap-filtered
            if prior and float(torch.rand(1, generator=gen, device="cpu")) < replay_frac:
                if monolithic:
                    cand = prior
                else:
                    cand = [(i_, l_, m_) for i_, l_, m_ in prior
                            if 0.1 < float((m_.to(dev) * mask).sum() / cfg.k) <= 0.8]
                if cand:
                    ids_r, lbl_r, m_r = cand[int(torch.randint(0, len(cand), (1,), generator=gen, device="cpu"))]
                    ctrl.set_mask(ones if monolithic else m_r.to(dev))
                    model(input_ids=ids_r.to(dev).unsqueeze(0),
                          labels=lbl_r.to(dev).unsqueeze(0)).loss.backward()
                    ctrl.set_mask(mask)
            opt.step()
            hist.append(kl)
            if step % 10 == 0:
                log(f"    {key} step {step:>3}/{steps} kl={kl:.4f}")
    finally:
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
        model.config.use_cache = prev_cache
        model.eval()
    return hist


def build_block_ref(cfg, h, examples: list[dict], n_support: int = 6,
                    replay_cap: int = 16, monolithic: bool = False):
    """Reference mask for routing diagnostics + a bounded CE replay buffer
    (labels over the response span)."""
    from spatial import kwta
    model, ctrl, cfgr, dev = h.model, h.ctrl, h.cfgr, h.dev
    model.eval()
    buf, acc = [], torch.zeros(cfg.H, device=dev)
    for e in examples[:max(n_support, replay_cap)]:
        ids = e["student_ids"]
        lbl = torch.full_like(ids, -100)
        lbl[e["s_start"]: e["s_start"] + e["n_resp"]] = ids[e["s_start"]: e["s_start"] + e["n_resp"]]
        if monolithic:
            mm = torch.ones(cfg.H, device=dev)
        else:
            mm, _, _ = cfgr(model, e["recfg"].to(dev).unsqueeze(0))
        if len(buf) < replay_cap:
            buf.append((ids, lbl, mm.detach().cpu()))
        acc += mm.float()
    ref = kwta(acc, cfg.k) if float(acc.sum()) > 0 else torch.zeros(cfg.H)
    return ref, buf
