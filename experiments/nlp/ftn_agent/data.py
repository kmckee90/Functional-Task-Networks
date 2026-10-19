"""Turn synthesized pairs + raw activity text into training examples.

Reuses ftn_chat's answer-masked-SFT format (design.md §5.2): the reconfiguration
sees the BARE question (matching chat-time inference), the loss is applied only to
the answer span. Also handles per-block dataset persistence to work_dir (written
by synthesis, read by train, PRUNED after the block trains — design.md §7.3).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

from config import Config


# --- answer-masked SFT formatting (from ftn_chat/data.py) --------------------

def format_qa(question: str, answer: str, tokenizer, sys_msg: str, max_len: int):
    """Return (recfg_ids, ids, labels) as cpu LongTensors.

    recfg_ids = raw question tokens (what reconfiguration sees at chat time)
    ids       = full chat (system + user + assistant answer)
    labels    = -100 on prompt/pad, answer token ids on the answer span
    """
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                               tokenize=True)
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
    if len(prompt_ids) and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    prompt_ids = list(prompt_ids)
    answer_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    answer_ids = answer_ids + [tokenizer.eos_token_id]
    ids = (prompt_ids + answer_ids)[:max_len]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_len]
    recfg = tokenizer(question, add_special_tokens=False)["input_ids"][:max_len]
    t = lambda x: torch.tensor(x, dtype=torch.long, device="cpu")
    return t(recfg), t(ids), t(labels)


def _chunk(token_ids, chunk_tokens: int):
    chunks = []
    for i in range(0, len(token_ids) - 1, chunk_tokens):
        window = token_ids[i:i + chunk_tokens]
        if len(window) < 16:
            continue
        chunks.append(torch.tensor(window, dtype=torch.long))
    return chunks


# --- per-block dataset persistence (work_dir; pruned after training) ---------

def _block_dir(cfg: Config, key: str) -> Path:
    d = Path(cfg.work_dir) / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_block_dataset(cfg: Config, key: str, pairs: list[dict], raw_text: str) -> Path:
    d = _block_dir(cfg, key)
    (d / "pairs.json").write_text(json.dumps(pairs, indent=2))
    (d / "raw.txt").write_text(raw_text or "")
    return d


def load_block_dataset(cfg: Config, key: str) -> tuple[list[dict], str]:
    d = Path(cfg.work_dir) / key
    pairs, raw = [], ""
    if (d / "pairs.json").exists():
        try:
            pairs = json.loads((d / "pairs.json").read_text())
        except Exception:
            pairs = []
    if (d / "raw.txt").exists():
        raw = (d / "raw.txt").read_text()
    return pairs, raw


def prune_block_dataset(cfg: Config, key: str) -> None:
    """Delete a block's dataset from work_dir — it's in the weights now (§7.3)."""
    d = Path(cfg.work_dir) / key
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


# --- build training pools for one block --------------------------------------

def build_examples(cfg: Config, tokenizer, pairs: list[dict], raw_text: str) -> dict:
    """Returns {qa, raw, support}:
      qa      - answer-masked examples from synthesized {question, answer} pairs
      raw     - plain-LM chunks of the raw activity narrative (all tokens labeled)
      support - a few raw chunks reserved for the reference mask + replay seeding
    """
    qa = []
    for pair in pairs:
        q, a = pair.get("question", ""), pair.get("answer", "")
        if not q or not a:
            continue
        recfg, ids, lbl = format_qa(q, a, tokenizer, cfg.qa_sys, cfg.qa_max_len)
        if len(recfg) >= 2 and (lbl != -100).any():
            qa.append({"recfg": recfg, "ids": ids, "lbl": lbl})

    raw = []
    if raw_text.strip():
        ids = tokenizer(raw_text, add_special_tokens=False)["input_ids"][: cfg.token_ceiling]
        for ch in _chunk(ids, cfg.chunk_tokens):
            c = ch.detach().cpu()
            raw.append({"recfg": c, "ids": c, "lbl": c})

    support = raw[:8] if raw else qa[:8]
    return {"qa": qa, "raw": raw, "support": support}


# --- general-instruction rehearsal (anti-forgetting; from ftn_chat) ----------

def load_rehearsal(cfg: Config, tokenizer):
    if cfg.rehearsal_frac <= 0 or cfg.rehearsal_n <= 0:
        return []
    from datasets import load_dataset
    ds = load_dataset(cfg.rehearsal_dataset, split="train")
    sys_msg = "You are a helpful assistant."
    out = []
    for row in ds:
        instr = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        ans = (row.get("output") or "").strip()
        if not instr or len(ans) < 20:
            continue
        q = instr + (("\n\n" + inp) if inp else "")
        recfg, ids, lbl = format_qa(q, ans, tokenizer, sys_msg, cfg.qa_max_len)
        if len(recfg) >= 2 and (lbl != -100).any():
            out.append({"recfg": recfg, "ids": ids, "lbl": lbl})
        if len(out) >= cfg.rehearsal_n:
            break
    return out
