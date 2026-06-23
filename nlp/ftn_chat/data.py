"""Domain ingestion for FTN-Chat — registry-driven, one task at a time.

Each task is defined in registry.py by a set of file globs. We concatenate the
matching files, tokenize, cut into non-overlapping windows of `chunk_tokens`,
cap at the per-task token ceiling, and split into train / support / heldout.

`build_task` builds ONE task (so adding a dataset doesn't rebuild the others);
`build_tasks` builds several. Run standalone to see per-task counts:
    cd ftn_chat && uv run python data.py [task ...]
"""
from __future__ import annotations

import glob as _glob
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from config import CFG, Config
from registry import TASKS, SANDBOX_CACHE

SANDBOX_URL = "https://github.com/astera-org/sandbox"


@dataclass
class DomainData:
    key: str
    train: list           # list[Tensor] of token-id chunks
    support: list
    heldout: list
    n_raw_tokens: int
    n_used_tokens: int


def ensure_sandbox() -> Path:
    """Clone astera-org/sandbox into the cache if not already present."""
    if (SANDBOX_CACHE / ".git").exists():
        return SANDBOX_CACHE
    print(f"  cloning {SANDBOX_URL} -> {SANDBOX_CACHE}")
    SANDBOX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", SANDBOX_URL, str(SANDBOX_CACHE)],
                   check=True, capture_output=True, text=True)
    return SANDBOX_CACHE


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
    if task.needs_sandbox:
        ensure_sandbox()
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


# --- Q&A examples (answer-masked SFT) -------------------------------------

def load_qa(cfg: Config, key: str):
    """Load synthesized Q&A pairs for a task (see qa_synth.py). [] if none."""
    p = cfg.data_cache / "qa" / f"{key}.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def format_qa(question: str, answer: str, tokenizer, sys_msg: str, max_len: int):
    """Return (recfg_ids, ids, labels) as cpu LongTensors.

    recfg_ids  = raw question tokens (what reconfiguration sees at chat time)
    ids        = full chat (system + user + assistant answer)
    labels     = -100 on prompt/pad, answer token ids on the answer span
    """
    msgs = [{"role": "system", "content": sys_msg},
            {"role": "user", "content": question}]
    prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                               tokenize=True)
    if hasattr(prompt_ids, "input_ids"):          # some versions return a dict
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


def load_rehearsal(cfg: Config, tokenizer):
    """A slice of a general instruction dataset, formatted as answer-masked
    examples, mixed into every block so the bank can't erode general fluency."""
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


def build_examples(cfg: Config, tokenizer, key: str, data: DomainData):
    """Build the training pools for one task block.

    Returns dict with:
      raw      - list of {recfg, ids, lbl} from train chunks (LM, all tokens)
      qa       - list of {recfg, ids, lbl} from synthesized Q&A (answer-masked)
      support  - list of {recfg, ids, lbl} (raw, from support chunks) for the
                 reference mask + replay seeding
    """
    def raw_ex(ch):
        c = ch.detach().cpu()
        return {"recfg": c, "ids": c, "lbl": c}

    raw = [raw_ex(ch) for ch in data.train]
    support = [raw_ex(ch) for ch in data.support]
    qa = []
    for pair in load_qa(cfg, key):
        recfg, ids, lbl = format_qa(pair["question"], pair["answer"],
                                    tokenizer, cfg.qa_sys, cfg.qa_max_len)
        if len(recfg) >= 2 and (lbl != -100).any():
            qa.append({"recfg": recfg, "ids": ids, "lbl": lbl})
    return {"raw": raw, "qa": qa, "support": support}


def report(domains: dict[str, DomainData]):
    print(f"{'task':<12} {'raw_tok':>10} {'used_tok':>10} "
          f"{'train':>7} {'support':>8} {'heldout':>8}")
    for k, d in domains.items():
        print(f"{k:<12} {d.n_raw_tokens:>10,} {d.n_used_tokens:>10,} "
              f"{len(d.train):>7} {len(d.support):>8} {len(d.heldout):>8}")


if __name__ == "__main__":
    import sys
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    keys = sys.argv[1:] or list(CFG.domains)
    tok = AutoTokenizer.from_pretrained(CFG.model_name)
    report(build_tasks(CFG, tok, keys))
