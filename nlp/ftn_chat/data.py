"""Domain ingestion for FTN-Chat — registry-driven, one task at a time.

Each task is defined in registry.py by a set of file globs into corpus/. We
concatenate the matching files, tokenize, cut into non-overlapping windows of
`chunk_tokens`, cap at the per-task token ceiling, and split into
train / support / heldout.

Chunks supply context for distillation: qa_synth.py turns them into (question,
answer, context) triples, and train.py distills the teacher-with-context onto
the closed-book student. Optional source CE also trains on train-split chunks.
Raw chunks serve reconfiguration support and routing_check.

`build_task` builds ONE task (so adding a dataset doesn't rebuild the others);
`build_tasks` builds several. Run standalone to see per-task counts:
    cd ftn_chat && uv run python data.py [task ...]
"""
from __future__ import annotations

import glob as _glob
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from config import CFG, Config
from registry import TASKS


@dataclass
class DomainData:
    key: str
    train: list           # list[Tensor] of token-id chunks
    support: list
    heldout: list
    n_raw_tokens: int
    n_used_tokens: int


def _ipynb_to_text(path: Path) -> str:
    try:
        nb = json.loads(path.read_text(errors="ignore"))
    except Exception:
        return ""
    parts = []
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if isinstance(src, list):
            src = "".join(src)
        parts.append(src)
    return "\n\n".join(parts)


def gather_task_text(key: str) -> str:
    """Concatenate all files matching the task's globs (.ipynb handled)."""
    task = TASKS[key]
    texts, seen = [], set()
    for pattern in task.globs:
        for fp in sorted(_glob.glob(pattern, recursive=True)):
            if fp in seen:
                continue
            seen.add(fp)
            p = Path(fp)
            if not p.is_file():
                continue
            if p.suffix == ".ipynb":
                texts.append(f"# FILE: {p.name}\n{_ipynb_to_text(p)}")
            else:
                try:
                    texts.append(f"# FILE: {p.name}\n{p.read_text(errors='ignore')}")
                except Exception:
                    continue
    return "\n\n".join(texts)


def _chunk(token_ids, chunk_tokens: int):
    chunks = []
    for i in range(0, len(token_ids) - 1, chunk_tokens):
        window = token_ids[i:i + chunk_tokens]
        if len(window) < 16:
            continue
        chunks.append(torch.tensor(window, dtype=torch.long))
    return chunks


def build_task(cfg: Config, tokenizer, key: str) -> DomainData:
    text = gather_task_text(key)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n_raw = len(ids)
    ids = ids[: cfg.token_ceiling]
    chunks = _chunk(ids, cfg.chunk_tokens)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    perm = torch.randperm(len(chunks), generator=gen, device="cpu").tolist()
    chunks = [chunks[i] for i in perm]
    h, s = cfg.heldout_per_domain, cfg.support_per_domain
    return DomainData(key=key, train=chunks[h + s:], support=chunks[h:h + s],
                      heldout=chunks[:h], n_raw_tokens=n_raw, n_used_tokens=len(ids))


def build_tasks(cfg: Config, tokenizer, keys) -> dict[str, DomainData]:
    return {k: build_task(cfg, tokenizer, k) for k in keys}


# Back-compat alias used by routing_check.
def build_domains(cfg: Config, tokenizer) -> dict[str, DomainData]:
    return build_tasks(cfg, tokenizer, cfg.domains)


# --- Context-distillation examples -----------------------------------------

def load_qa(cfg: Config, key: str):
    """Load synthesized (question, answer, context) triples for a task (see
    qa_synth.py). [] if none."""
    p = cfg.data_cache / "qa" / f"{key}.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def _chat_prompt_ids(tokenizer, sys_msg: str, user_msg: str) -> list:
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg}]
    ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                        tokenize=True)
    if hasattr(ids, "input_ids"):              # some versions return a dict
        ids = ids["input_ids"]
    if len(ids) and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def teacher_user_msg(context: str, question: str) -> str:
    """The teacher sees the source excerpt in context; the student does not."""
    return (
        "Consult the following source material:\n\n"
        f"{context}\n\n"
        f"Using it, answer this question:\n{question}"
    )


def format_distill(question: str, answer: str, context: str | None,
                   tokenizer, cfg: Config):
    """Build one context-distillation example. Returns a dict of cpu LongTensors:

      recfg   raw question tokens (what reconfiguration sees at chat time)
      ids/lbl     STUDENT chat (sys + question + answer); lbl = -100 on the
                  prompt, answer token ids on the answer span
      t_ids/t_lbl TEACHER chat (sys + context + question + same answer tokens);
                  lbl marks the identical answer span

    The answer token ids are IDENTICAL in both sequences (same text appended
    after each generation prompt), so per-token KL aligns position-for-position
    over the answer span. The context is token-capped so the answer never gets
    truncated differently between the two views. context=None (e.g. rehearsal)
    makes the teacher input identical to the student input: pure self-
    distillation toward base behavior.
    """
    s_prompt = _chat_prompt_ids(tokenizer, cfg.qa_sys, question)
    ans = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    ans = ans[: max(1, cfg.qa_max_len - len(s_prompt) - 1)] + [tokenizer.eos_token_id]

    if context:
        ctx_budget = max(64, cfg.qa_ctx_max_len - cfg.qa_max_len)
        ctx_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        if len(ctx_ids) > ctx_budget:
            context = tokenizer.decode(ctx_ids[:ctx_budget])
        t_prompt = _chat_prompt_ids(
            tokenizer, cfg.qa_sys, teacher_user_msg(context, question))
    else:
        t_prompt = s_prompt

    def seq(prompt):
        ids = prompt + ans
        lbl = [-100] * len(prompt) + ans
        t = lambda x: torch.tensor(x, dtype=torch.long, device="cpu")
        return t(ids), t(lbl)

    ids, lbl = seq(s_prompt)
    t_ids, t_lbl = seq(t_prompt)
    recfg = torch.tensor(
        tokenizer(question, add_special_tokens=False)["input_ids"][: cfg.qa_max_len],
        dtype=torch.long, device="cpu")
    return {"recfg": recfg, "ids": ids, "lbl": lbl, "t_ids": t_ids, "t_lbl": t_lbl}


def load_rehearsal(cfg: Config, tokenizer):
    """A slice of a general instruction dataset as context-free distillation
    examples (teacher == student input; KL pulls the masked model back toward
    base behavior on general questions). Off unless rehearsal_frac > 0."""
    if cfg.rehearsal_frac <= 0 or cfg.rehearsal_n <= 0:
        return []
    from datasets import load_dataset
    ds = load_dataset(cfg.rehearsal_dataset, split="train")
    out = []
    for row in ds:
        instr = (row.get("instruction") or "").strip()
        inp = (row.get("input") or "").strip()
        ans = (row.get("output") or "").strip()
        if not instr or len(ans) < 20:
            continue
        q = instr + (("\n\n" + inp) if inp else "")
        e = format_distill(q, ans, None, tokenizer, cfg)
        if len(e["recfg"]) >= 2 and (e["lbl"] != -100).any():
            out.append(e)
        if len(out) >= cfg.rehearsal_n:
            break
    return out


def build_examples(cfg: Config, tokenizer, key: str, data: DomainData):
    """Build the training pools for one task block.

    Returns dict with:
      qa       - list of distillation examples (see format_distill)
      support  - raw support chunks as {recfg} for reference-mask building
    """
    qa = []
    for triple in load_qa(cfg, key):
        e = format_distill(triple["question"], triple["answer"],
                           triple.get("context"), tokenizer, cfg)
        if len(e["recfg"]) >= 2 and (e["lbl"] != -100).any():
            qa.append(e)
    support = [{"recfg": ch.detach().cpu()} for ch in data.support]
    return {"qa": qa, "support": support}


def report(domains: dict[str, DomainData]):
    print(f"{'task':<12} {'raw_tok':>10} {'used_tok':>10} "
          f"{'train':>7} {'support':>8} {'heldout':>8}")
    for k, d in domains.items():
        print(f"{k:<12} {d.n_raw_tokens:>10,} {d.n_used_tokens:>10,} "
              f"{len(d.train):>7} {len(d.support):>8} {len(d.heldout):>8}")


if __name__ == "__main__":
    import sys
    from transformers import AutoTokenizer
    keys = sys.argv[1:] or list(CFG.domains)
    tok = AutoTokenizer.from_pretrained(CFG.model_name)
    report(build_tasks(CFG, tok, keys))
