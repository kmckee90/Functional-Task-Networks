"""Synthesize post-training data from episodes (design.md §5.2).

Two record types, both grounded in REAL captured artifacts (the anti-drift
guard, design.md §5.5):
  (a) grounded Q&A  — facts about the workspace/code/session
  (b) insight/postmortem — for episodes with a failure: "X was tried, it failed
      because Y, do Z instead" — the records that make the continuity demos land.

Reuses the SERVING model (design.md §5.2): no second load. We zero the atom mask
so synthesis is the frozen backbone (clean base-model writing, uncolored by the
bank), then restore the caller's mask.
"""
from __future__ import annotations

import json
import re

import torch

from synth.segment import Episode, render

QA_PROMPT = (
    "Below is a log of a real work episode in a software project (user + AI "
    "assistant turns, shell/tool commands, outputs, git diffs).\n\n"
    "Write {n} self-contained question/answer pairs a future engineer or AI "
    "assistant would find useful when returning to this project.\n"
    "Requirements:\n"
    "- Questions are specific and natural; do NOT refer to 'the log', 'the "
    "episode', or line numbers.\n"
    "- Answers are 2-4 complete sentences of plain prose explaining the what and "
    "the why. You may mention an identifier inline (like `foo`) but do not dump "
    "code.\n"
    'Return ONLY a JSON array of objects with keys "question" and "answer".\n\n'
    "EPISODE LOG:\n{log}\n\nJSON:"
)

INSIGHT_PROMPT = (
    "Below is a log of a real work episode that hit at least one FAILURE and "
    "(possibly) resolved it.\n\n"
    "Extract {n} durable LESSONS as question/answer pairs, so a future engineer "
    "or AI assistant who is about to repeat the mistake gets warned.\n"
    "Requirements:\n"
    "- The question is what someone would naturally ask BEFORE trying the thing "
    '(e.g. "Can I use X for Y here?", "Why isn\'t my Z taking effect?").\n'
    "- The answer states plainly: what was tried, WHY it failed, and what to do "
    "instead. Be concrete and specific to this project.\n"
    "- Only include lessons actually supported by the log; do not invent.\n"
    'Return ONLY a JSON array of objects with keys "question" and "answer".\n\n'
    "EPISODE LOG:\n{log}\n\nJSON:"
)

_CODEY = set("{}[]()=;<>|\\")


def _is_good(q: str, a: str) -> bool:
    if len(a) < 40 or len(a) > 1600:
        return False
    if "?" not in q:
        return False
    if sum(c in _CODEY for c in a) / max(1, len(a)) > 0.10:
        return False
    if len(a.split()) < 8:
        return False
    return True


def _extract(text: str, kind: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out = []
    for d in arr:
        if not (isinstance(d, dict) and d.get("question") and d.get("answer")):
            continue
        q, a = str(d["question"]).strip(), str(d["answer"]).strip()
        if _is_good(q, a):
            out.append({"question": q, "answer": a, "kind": kind})
    return out


def _generate(model, tok, cfg, prompt: str) -> str:
    msgs = [{"role": "user", "content": prompt}]
    try:
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, enable_thinking=cfg.synth_use_thinking,
        )
    except TypeError:  # tokenizer without enable_thinking kwarg
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=cfg.synth_max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def synthesize_episode(model, tok, ctrl, cfg, ep: Episode) -> list[dict]:
    """Return a list of {question, answer, kind} for one episode. Zeros the mask
    (frozen backbone) during synthesis, then restores the caller's mask."""
    log = render(ep)
    if len(log.strip()) < 80:
        return []
    saved_mask = ctrl.mask.clone()
    ctrl.set_mask(torch.zeros(cfg.H, device=saved_mask.device))
    try:
        pairs: list[dict] = []
        qa_txt = _generate(model, tok, cfg,
                           QA_PROMPT.format(n=cfg.qa_per_episode, log=log))
        pairs += _extract(qa_txt, "qa")
        if ep.has_failure:
            ins_txt = _generate(model, tok, cfg,
                               INSIGHT_PROMPT.format(n=cfg.insights_per_episode, log=log))
            pairs += _extract(ins_txt, "insight")
    finally:
        ctrl.set_mask(saved_mask)
    return pairs


def synthesize_blocks(model, tok, ctrl, cfg, episodes: list[Episode], log=print) -> list[dict]:
    """Synthesize over many episodes; returns the pooled example list."""
    all_pairs: list[dict] = []
    n = len(episodes)
    for i, ep in enumerate(episodes):
        # Log BEFORE the (slow, ~1-2 min) generation so the GUI shows progress
        # immediately instead of going silent until episode 1 finishes.
        log(f"  synthesizing episode {i + 1}/{n} "
            f"({'FAIL' if ep.has_failure else 'ok'}, {len(ep.records)} records)…")
        pairs = synthesize_episode(model, tok, ctrl, cfg, ep)
        all_pairs += pairs
        log(f"    -> {len(pairs)} pairs "
            f"[{sum(p['kind'] == 'insight' for p in all_pairs)} insights total so far]")
    log(f"  synthesis complete: {len(all_pairs)} pairs from {n} episodes")
    return all_pairs
