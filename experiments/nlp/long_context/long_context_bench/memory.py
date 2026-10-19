from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import artifact_path, get_path, run_dir, seed
from .data import (
    _extract_qa_pairs,
    _flatten_dialogue,
    _load_locomo_rows,
    _official_locomo_sessions,
)
from .models import FTNLoRAAdapter, build_model


@dataclass
class MemoryBlock:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryQuestion:
    id: str
    question: str
    answer: str
    oracle_block_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEpisode:
    id: str
    dataset: str
    split: str
    blocks: list[MemoryBlock]
    questions: list[MemoryQuestion]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, obj: dict[str, Any]) -> "MemoryEpisode":
        return cls(
            id=str(obj["id"]),
            dataset=str(obj.get("dataset", "")),
            split=str(obj.get("split", "")),
            blocks=[MemoryBlock(**b) for b in obj.get("blocks", [])],
            questions=[MemoryQuestion(**q) for q in obj.get("questions", [])],
            metadata=dict(obj.get("metadata", {})),
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def memory_processed_dir(cfg: dict[str, Any]) -> Path:
    return artifact_path(cfg, get_path(cfg, "memory", "processed_dir", default="data/processed/memory"))


def memory_manifest_path(cfg: dict[str, Any], split: str) -> Path:
    return memory_processed_dir(cfg) / f"{split}_episodes.jsonl"


def memory_run_dir(cfg: dict[str, Any]) -> Path:
    return run_dir(cfg, "ftn_lora_memory")


def write_episodes(path: str | Path, episodes: Iterable[MemoryEpisode]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w") as f:
        for ep in episodes:
            f.write(json.dumps(ep.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_episodes(path: str | Path) -> Iterator[MemoryEpisode]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield MemoryEpisode.from_json(json.loads(line))


def prepare_memory_episodes(cfg: dict[str, Any]) -> dict[str, int]:
    """Prepare LoCoMo-style memory episodes for write/recall experiments."""
    out_dir = memory_processed_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed(cfg))
    counts = {}
    for split in ("train", "eval"):
        episodes: list[MemoryEpisode] = []
        for spec in get_path(cfg, "memory", "datasets", default=[]):
            if not spec.get("enabled", True):
                continue
            typ = str(spec.get("type", "locomo")).lower()
            if typ != "locomo":
                raise ValueError(f"memory datasets currently support type='locomo', got {typ!r}")
            episodes.extend(_build_locomo_memory_episodes(spec, split, rng.randrange(2**31)))
        rng.shuffle(episodes)
        counts[split] = write_episodes(memory_manifest_path(cfg, split), episodes)
    return counts


def _build_locomo_memory_episodes(
    spec: dict[str, Any], split: str, base_seed: int
) -> list[MemoryEpisode]:
    rows = list(_load_locomo_rows(spec, split))
    rng = random.Random(base_seed + (0 if split == "train" else 10_000))
    rng.shuffle(rows)
    limit = spec.get(f"{split}_episodes", spec.get("episodes"))
    if limit is not None:
        rows = rows[: int(limit)]

    name = str(spec.get("name", "locomo"))
    block_mode = str(spec.get("block_mode", "conversation")).lower()
    max_questions = spec.get(f"{split}_questions_per_episode", spec.get("questions_per_episode"))
    episodes = []
    for row_idx, row in enumerate(rows):
        blocks = _memory_blocks_from_row(row, block_mode)
        if not blocks:
            continue
        qas = list(_extract_qa_pairs(row))
        if max_questions is not None:
            qas = qas[: int(max_questions)]
        questions = []
        default_block = blocks[0].id
        for qa_idx, qa in enumerate(qas):
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            if not question or not answer:
                continue
            oracle = _oracle_block_id(qa, blocks, default_block)
            questions.append(
                MemoryQuestion(
                    id=f"{name}-{split}-{row_idx:05d}-q{qa_idx:03d}",
                    question=question,
                    answer=answer,
                    oracle_block_id=oracle,
                    metadata={
                        "row_index": row_idx,
                        "qa_index": qa_idx,
                        "category": qa.get("category") or qa.get("type"),
                    },
                )
            )
        if not questions:
            continue
        episodes.append(
            MemoryEpisode(
                id=f"{name}-{split}-{row_idx:05d}",
                dataset=name,
                split=split,
                blocks=blocks,
                questions=questions,
                metadata={"block_mode": block_mode, "source": spec.get("path") or spec.get("dataset_name")},
            )
        )
    return episodes


def _memory_blocks_from_row(row: dict[str, Any], block_mode: str) -> list[MemoryBlock]:
    conversation = row.get("conversation")
    if block_mode == "conversation":
        text = _flatten_dialogue(conversation if conversation is not None else row)
        return [MemoryBlock(id="conversation", text=text, metadata={"mode": block_mode})] if text else []

    if block_mode != "session":
        raise ValueError(f"unknown memory block_mode: {block_mode}")

    if isinstance(conversation, dict):
        official = _official_locomo_sessions(conversation)
        if official:
            return [
                MemoryBlock(
                    id=f"session_{idx:03d}",
                    text=(f"[Session {idx}" + (f" | {dt}" if dt else "") + f"]\n{text}"),
                    metadata={"session_index": idx, "date_time": dt, "mode": "official_locomo"},
                )
                for idx, dt, text in official
            ]

    raw_sessions = None
    for key in ("sessions", "session", "conversation_sessions"):
        if isinstance(row.get(key), list):
            raw_sessions = row[key]
            break
    if raw_sessions is None:
        text = _flatten_dialogue(row)
        return [MemoryBlock(id="session_000", text=text, metadata={"mode": "fallback_conversation"})] if text else []

    blocks = []
    for i, session in enumerate(raw_sessions):
        text = _flatten_dialogue(session)
        if text.strip():
            blocks.append(MemoryBlock(id=f"session_{i:03d}", text=text, metadata={"session_index": i}))
    return blocks


def _oracle_block_id(qa: dict[str, Any], blocks: list[MemoryBlock], default: str) -> str:
    block_ids = {b.id for b in blocks}
    evidence = qa.get("evidence")
    if isinstance(evidence, list) and evidence:
        first = str(evidence[0])
        if first.startswith("D") and ":" in first:
            sess = first[1:].split(":", 1)[0]
            if sess.isdigit():
                candidate = f"session_{int(sess):03d}"
                if candidate in block_ids:
                    return candidate
    for key in ("oracle_block_id", "block_id", "session_id", "evidence_session_id"):
        val = qa.get(key)
        if val is None:
            continue
        sval = str(val)
        if sval in block_ids:
            return sval
        if sval.isdigit():
            candidate = f"session_{int(sval):03d}"
            if candidate in block_ids:
                return candidate
    for key in ("evidence_sessions", "sessions", "evidence"):
        val = qa.get(key)
        if isinstance(val, list) and val:
            first = str(val[0])
            if first in block_ids:
                return first
            if first.isdigit():
                candidate = f"session_{int(first):03d}"
                if candidate in block_ids:
                    return candidate
    return default


def write_memory(cfg: dict[str, Any], limit_episodes: int | None = None) -> dict[str, Any]:
    """Write memory blocks into FTN-LoRA masked subspaces.

    The default split is `eval`: the benchmark writes the conversation context
    for the same held-out questions it later asks, while never training on the
    QA answers.
    """
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    adapter = build_model("ftn_lora", cfg)
    if not isinstance(adapter, FTNLoRAAdapter):
        raise TypeError("memory writing requires the ftn_lora adapter")

    write_cfg = get_path(cfg, "memory", "write", default={})
    write_split = str(write_cfg.get("split", "eval"))
    write_path = memory_manifest_path(cfg, write_split)
    if not write_path.exists():
        raise FileNotFoundError(f"missing {write_path}; run memory-prepare first")
    episodes = list(read_episodes(write_path))
    if limit_episodes is not None:
        episodes = episodes[:limit_episodes]

    max_length = int(write_cfg.get("max_length", 1024))
    stride = int(write_cfg.get("stride", max_length))
    steps_per_block = int(write_cfg.get("steps_per_block", 10))
    batch_size = int(write_cfg.get("batch_size", 1))
    lr = float(write_cfg.get("lr", get_path(cfg, "train", "lr", default=2e-4)))
    max_blocks = write_cfg.get("max_blocks")
    max_blocks = int(max_blocks) if max_blocks is not None else None
    reconfig_steps = int(write_cfg.get("reconfig_steps", adapter.train_reconfig_steps))
    log_every = max(1, int(write_cfg.get("log_every", 10)))

    out_dir = memory_run_dir(cfg)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "memory_write_metrics.jsonl"
    opt = torch.optim.AdamW(adapter.trainable_parameters(), lr=lr, foreach=False)

    mask_index = []
    global_step = 0
    t0 = time.time()
    adapter.model.train()
    with metrics_path.open("w") as mf:
        for episode in episodes:
            for block in episode.blocks:
                if max_blocks is not None and len(mask_index) >= max_blocks:
                    break
                chunks = _token_chunks(
                    adapter.tokenizer,
                    block.text,
                    max_length=max_length,
                    stride=stride,
                    eos=True,
                )
                if not chunks:
                    continue
                recfg = _tensorize([chunks[0]], adapter.device, pad_id=_pad_id(adapter))
                adapter.configurer(
                    adapter.model,
                    recfg["input_ids"],
                    recfg["attention_mask"],
                    steps=reconfig_steps,
                )
                mask = adapter.controller.mask.detach().clone()
                loader = DataLoader(chunks, batch_size=batch_size, shuffle=True, collate_fn=lambda b: _tensorize(b, adapter.device, _pad_id(adapter)))
                losses = []
                for _ in tqdm(range(steps_per_block), desc=f"write {episode.id}/{block.id}", leave=False):
                    for batch in loader:
                        opt.zero_grad(set_to_none=True)
                        out = adapter.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                        out.loss.backward()
                        torch.nn.utils.clip_grad_norm_(adapter.trainable_parameters(), float(write_cfg.get("max_grad_norm", 1.0)))
                        opt.step()
                        global_step += 1
                        lv = float(out.loss.detach())
                        losses.append(lv)
                        if global_step % log_every == 0:
                            mf.write(json.dumps({
                                "phase": "memory_write",
                                "step": global_step,
                                "episode_id": episode.id,
                                "block_id": block.id,
                                "loss": lv,
                                "wall_seconds": time.time() - t0,
                            }) + "\n")
                            mf.flush()
                        break
                mask_index.append({
                    "episode_id": episode.id,
                    "dataset": episode.dataset,
                    "split": episode.split,
                    "block_id": block.id,
                    "active_indices": _active_indices(mask),
                    "k": adapter.controller.k,
                    "mean_write_nll": sum(losses) / max(1, len(losses)),
                    "n_chunks": len(chunks),
                    "metadata": block.metadata,
                })
            if max_blocks is not None and len(mask_index) >= max_blocks:
                break

    bank_path = ckpt_dir / "memory_bank.pt"
    adapter.save_checkpoint(bank_path)
    index_path = out_dir / "mask_index.json"
    index_path.write_text(json.dumps(mask_index, indent=2))
    summary = {
        "phase": "memory_write",
        "split": write_split,
        "episodes": len(episodes),
        "blocks_written": len(mask_index),
        "steps": global_step,
        "checkpoint": str(bank_path),
        "mask_index": str(index_path),
        "wall_seconds": time.time() - t0,
    }
    (out_dir / "memory_write_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def eval_memory(
    cfg: dict[str, Any],
    checkpoint: str | Path | None = None,
    limit_questions: int | None = None,
) -> dict[str, Any]:
    """Evaluate question-only recall against written FTN-LoRA memory masks."""
    import torch
    from tqdm import tqdm

    adapter = build_model("ftn_lora", cfg)
    if not isinstance(adapter, FTNLoRAAdapter):
        raise TypeError("memory eval requires the ftn_lora adapter")
    out_dir = memory_run_dir(cfg)
    if checkpoint is None:
        checkpoint = out_dir / "checkpoints" / "memory_bank.pt"
    adapter.load_checkpoint(checkpoint)

    eval_cfg = get_path(cfg, "memory", "eval", default={})
    eval_split = str(eval_cfg.get("split", "eval"))
    eval_path = memory_manifest_path(cfg, eval_split)
    if not eval_path.exists():
        raise FileNotFoundError(f"missing {eval_path}; run memory-prepare first")
    episodes = list(read_episodes(eval_path))
    index = _load_mask_index(out_dir / "mask_index.json", adapter.controller.H, adapter.device)
    modes = list(eval_cfg.get("modes", ["zero", "oracle", "query", "wrong", "full_context"]))
    max_length = int(eval_cfg.get("max_length", 1024))
    overflow = str(eval_cfg.get("overflow", "skip"))
    query_steps = int(eval_cfg.get("query_reconfig_steps", adapter.eval_reconfig_steps))
    results_path = out_dir / "memory_eval_results.jsonl"
    summaries = {mode: _MetricAccumulator() for mode in modes}
    rows_written = 0

    adapter.model.eval()
    with results_path.open("w") as rf:
        for episode in tqdm(episodes, desc="memory eval episodes", leave=False):
            block_text = {b.id: b.text for b in episode.blocks}
            for question in episode.questions:
                if limit_questions is not None and rows_written >= limit_questions * len(modes):
                    break
                oracle_mask = _lookup_mask(index, episode.id, question.oracle_block_id)
                for mode in modes:
                    prompt = _question_prompt(question.question)
                    if mode == "full_context":
                        prompt = _full_context_prompt(block_text.get(question.oracle_block_id, ""), question.question)
                        adapter.controller.zero()
                    elif mode == "zero":
                        adapter.controller.zero()
                    elif mode == "oracle":
                        if oracle_mask is None:
                            continue
                        adapter.controller.set_mask(oracle_mask.clone())
                    elif mode == "wrong":
                        wrong = _wrong_mask(index, episode.id, question.oracle_block_id)
                        if wrong is None:
                            adapter.controller.zero()
                        else:
                            adapter.controller.set_mask(wrong.clone())
                    elif mode == "query":
                        recfg = _encode_prompt(adapter.tokenizer, prompt, adapter.device, _pad_id(adapter), max_length=max_length, overflow=overflow)
                        if recfg is None:
                            continue
                        adapter.configurer(adapter.model, recfg["input_ids"], recfg["attention_mask"], steps=query_steps)
                    else:
                        raise ValueError(f"unknown memory eval mode: {mode}")

                    batch = _encode_qa(
                        adapter.tokenizer,
                        prompt,
                        question.answer,
                        adapter.device,
                        _pad_id(adapter),
                        max_length=max_length,
                        overflow=overflow,
                    )
                    if batch is None:
                        continue
                    with torch.no_grad():
                        out = adapter.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                    label_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                    correct, total, exact = _teacher_forced_accuracy(out.logits, batch["labels"])
                    query_mask = adapter.controller.mask.detach().clone()
                    overlap = _mask_overlap(query_mask, oracle_mask, adapter.controller.k) if oracle_mask is not None else None
                    nll = float(out.loss.detach())
                    rec = {
                        "mode": mode,
                        "episode_id": episode.id,
                        "question_id": question.id,
                        "oracle_block_id": question.oracle_block_id,
                        "nll": nll,
                        "label_tokens": label_tokens,
                        "token_correct": correct,
                        "token_total": total,
                        "exact": exact,
                        "mask_overlap": overlap,
                        "category": question.metadata.get("category"),
                    }
                    rf.write(json.dumps(rec) + "\n")
                    summaries[mode].add(nll, label_tokens, correct, total, exact, overlap)
                    rows_written += 1
            if limit_questions is not None and rows_written >= limit_questions * len(modes):
                break

    summary = {
        "phase": "memory_eval",
        "split": eval_split,
        "checkpoint": str(checkpoint),
        "results": str(results_path),
        "modes": {mode: acc.summary() for mode, acc in summaries.items()},
    }
    (out_dir / "memory_eval_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


class _MetricAccumulator:
    def __init__(self):
        self.nll_x_tokens = 0.0
        self.tokens = 0
        self.examples = 0
        self.correct = 0
        self.total = 0
        self.exact = 0
        self.overlaps = []

    def add(self, nll: float, tokens: int, correct: int, total: int, exact: int, overlap: float | None):
        self.nll_x_tokens += nll * max(1, tokens)
        self.tokens += tokens
        self.examples += 1
        self.correct += correct
        self.total += total
        self.exact += exact
        if overlap is not None:
            self.overlaps.append(float(overlap))

    def summary(self) -> dict[str, Any]:
        nll = self.nll_x_tokens / max(1, self.tokens)
        return {
            "examples": self.examples,
            "label_tokens": self.tokens,
            "nll": nll,
            "perplexity": math.exp(nll) if nll < 50 else float("inf"),
            "teacher_forced_token_accuracy": self.correct / max(1, self.total),
            "teacher_forced_exact": self.exact / max(1, self.examples),
            "mean_mask_overlap": sum(self.overlaps) / max(1, len(self.overlaps)) if self.overlaps else None,
        }


def _token_chunks(tokenizer: Any, text: str, max_length: int, stride: int, eos: bool) -> list[list[int]]:
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos and eos_id is not None:
        ids.append(int(eos_id))
    chunks = []
    stride = max(1, stride)
    for start in range(0, max(1, len(ids)), stride):
        chunk = ids[start:start + max_length]
        if len(chunk) >= 8:
            chunks.append(chunk)
        if start + max_length >= len(ids):
            break
    return chunks


def _tensorize(chunks: list[list[int]], device: Any, pad_id: int) -> dict[str, Any]:
    import torch

    max_len = max(len(x) for x in chunks)
    input_ids = torch.full((len(chunks), max_len), pad_id, dtype=torch.long, device=device)
    labels = torch.full((len(chunks), max_len), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(chunks), max_len), dtype=torch.long, device=device)
    for i, chunk in enumerate(chunks):
        n = len(chunk)
        vals = torch.tensor(chunk, dtype=torch.long, device=device)
        input_ids[i, :n] = vals
        labels[i, :n] = vals
        attention_mask[i, :n] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def _question_prompt(question: str) -> str:
    return f"Answer from memory. Do not assume facts not stored in memory.\n\nQuestion: {question}\nAnswer:"


def _full_context_prompt(context: str, question: str) -> str:
    return (
        "Answer the question from the conversation. Use the conversation as the only source of truth.\n\n"
        f"Conversation:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def _encode_prompt(tokenizer: Any, prompt: str, device: Any, pad_id: int, max_length: int, overflow: str):
    ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    ids = _apply_overflow(ids, max_length, overflow)
    if ids is None:
        return None
    return _tensorize([ids], device, pad_id)


def _encode_qa(tokenizer: Any, prompt: str, answer: str, device: Any, pad_id: int, max_length: int, overflow: str):
    prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    target_ids = list(tokenizer(f" {answer}", add_special_tokens=False)["input_ids"])
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        target_ids.append(int(eos_id))
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if len(input_ids) > max_length:
        pair = _apply_pair_overflow(input_ids, labels, max_length, overflow)
        if pair is None:
            return None
        input_ids, labels = pair
    if not any(x != -100 for x in labels):
        return None

    import torch

    batch = _tensorize([input_ids], device, pad_id)
    batch["labels"] = torch.tensor([labels], dtype=torch.long, device=device)
    return batch


def _apply_overflow(ids: list[int], max_length: int, overflow: str) -> list[int] | None:
    if len(ids) <= max_length:
        return ids
    if overflow == "skip":
        return None
    if overflow == "truncate_left":
        return ids[-max_length:]
    if overflow == "truncate_right":
        return ids[:max_length]
    raise ValueError(f"unknown overflow policy: {overflow}")


def _apply_pair_overflow(input_ids: list[int], labels: list[int], max_length: int, overflow: str):
    if len(input_ids) <= max_length:
        return input_ids, labels
    if overflow == "skip":
        return None
    if overflow == "truncate_left":
        return input_ids[-max_length:], labels[-max_length:]
    if overflow == "truncate_right":
        return input_ids[:max_length], labels[:max_length]
    raise ValueError(f"unknown overflow policy: {overflow}")


def _pad_id(adapter: FTNLoRAAdapter) -> int:
    return int(adapter.tokenizer.pad_token_id or adapter.tokenizer.eos_token_id or 0)


def _active_indices(mask: Any) -> list[int]:
    return [int(i) for i in (mask > 0).nonzero(as_tuple=True)[0].detach().cpu().tolist()]


def _load_mask_index(path: Path, H: int, device: Any) -> dict[tuple[str, str], Any]:
    import torch

    raw = json.loads(path.read_text()) if path.exists() else []
    out = {}
    for row in raw:
        mask = torch.zeros(H, device=device)
        mask[torch.tensor(row["active_indices"], dtype=torch.long, device=device)] = 1.0
        out[(row["episode_id"], row["block_id"])] = mask
    return out


def _lookup_mask(index: dict[tuple[str, str], Any], episode_id: str, block_id: str):
    return index.get((episode_id, block_id))


def _wrong_mask(index: dict[tuple[str, str], Any], episode_id: str, block_id: str):
    for (eid, bid), mask in index.items():
        if eid != episode_id or bid != block_id:
            return mask
    return None


def _mask_overlap(a: Any, b: Any, k: int) -> float:
    if a is None or b is None:
        return 0.0
    return float(((a > 0) & (b > 0)).sum().item() / max(1, k))


def _teacher_forced_accuracy(logits: Any, labels: Any) -> tuple[int, int, int]:
    import torch

    preds = logits[:, :-1, :].argmax(dim=-1)
    gold = labels[:, 1:]
    mask = gold != -100
    correct = int(((preds == gold) & mask).sum().item())
    total = int(mask.sum().item())
    exact = int(total > 0 and bool(torch.equal(preds[0][mask[0]], gold[0][mask[0]])))
    return correct, total, exact
