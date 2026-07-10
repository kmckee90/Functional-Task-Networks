from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import processed_dir, seed
from .records import DataRecord, read_jsonl, write_jsonl


FILLER_WORDS = (
    "archive", "signal", "market", "lantern", "vector", "memo", "harbor",
    "delta", "orbit", "ledger", "ceramic", "notebook", "garden", "policy",
    "render", "silver", "thread", "station", "matrix", "canvas", "packet",
    "bridge", "forest", "copper", "module", "syntax", "valley", "window",
    "kernel", "mirror", "circuit", "planet", "schema", "island", "report",
    "anchor", "fabric", "summit", "engine", "signal", "folder", "battery",
)


def prepare_datasets(cfg: dict[str, Any]) -> dict[str, int]:
    """Build configured datasets and write train/eval JSONL manifests."""
    out_dir = processed_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed(cfg))
    by_split: dict[str, list[DataRecord]] = {"train": [], "eval": []}

    for spec in cfg.get("data", {}).get("datasets", []):
        if not spec.get("enabled", True):
            continue
        ds_seed = rng.randrange(2**31)
        for split in ("train", "eval"):
            records = list(_build_records_for_split(spec, split, ds_seed))
            by_split[split].extend(records)

    counts = {}
    for split, records in by_split.items():
        rng.shuffle(records)
        counts[split] = write_jsonl(out_dir / f"{split}.jsonl", records)
    return counts


def manifest_path(cfg: dict[str, Any], split: str) -> Path:
    return processed_dir(cfg) / f"{split}.jsonl"


def _build_records_for_split(
    spec: dict[str, Any], split: str, base_seed: int
) -> Iterator[DataRecord]:
    typ = str(spec.get("type", "")).lower()
    if typ == "niah":
        yield from build_niah_records(spec, split, base_seed)
    elif typ == "locomo":
        yield from build_locomo_records(spec, split, base_seed)
    else:
        raise ValueError(f"unknown dataset type: {typ!r}")


def build_niah_records(
    spec: dict[str, Any], split: str, base_seed: int
) -> Iterator[DataRecord]:
    """Generate deterministic needle-in-a-haystack QA examples."""
    n = int(spec.get(f"{split}_size", spec.get("size", 0)))
    name = str(spec.get("name", "niah"))
    context_lengths = [int(x) for x in spec.get("context_lengths", [4096])]
    depths = [float(x) for x in spec.get("needle_depths", [0.5])]
    num_needles = max(1, int(spec.get("num_needles", 1)))
    rng = random.Random(base_seed + (0 if split == "train" else 10_000))

    for i in range(n):
        context_len = context_lengths[i % len(context_lengths)]
        depth = depths[(i // max(1, len(context_lengths))) % len(depths)]
        local = random.Random(rng.randrange(2**31))
        key = f"{split.upper()}-{i:05d}-{local.randrange(10_000, 99_999)}"
        answer = f"CODE-{local.randrange(100_000, 999_999)}"
        decoys = [
            (f"DECOY-{j}-{local.randrange(10_000, 99_999)}",
             f"CODE-{local.randrange(100_000, 999_999)}")
            for j in range(max(0, num_needles - 1))
        ]
        context = _make_haystack(local, context_len, key, answer, depth, decoys)
        prompt = (
            "Read the complete context and answer using the exact hidden code.\n\n"
            f"{context}\n\n"
            f"Question: What is the hidden code for record {key}?\n"
            "Answer:"
        )
        yield DataRecord(
            id=f"{name}-{split}-{i:06d}",
            dataset=name,
            split=split,
            prompt=prompt,
            target=f" {answer}",
            answers=[answer],
            metadata={
                "type": "needle_in_a_haystack",
                "context_words": context_len,
                "needle_depth": depth,
                "needle_key": key,
                "num_needles": num_needles,
            },
        )


def _make_haystack(
    rng: random.Random,
    n_words: int,
    key: str,
    answer: str,
    depth: float,
    decoys: list[tuple[str, str]],
) -> str:
    words = [rng.choice(FILLER_WORDS) for _ in range(max(32, n_words))]
    needle = (
        f"\n\nHidden memory record {key}: the answer code is {answer}. "
        "Use this exact code if asked about this record.\n\n"
    )
    insert_at = max(0, min(len(words), int(len(words) * depth)))
    pieces = [" ".join(words[:insert_at]), needle, " ".join(words[insert_at:])]
    if decoys:
        for j, (dkey, dans) in enumerate(decoys):
            pos = max(0, min(len(pieces), 1 + 2 * j))
            pieces.insert(
                pos,
                (
                    f"\n\nDistractor memory record {dkey}: the answer code is "
                    f"{dans}. This is not the requested record.\n\n"
                ),
            )
    return "".join(pieces)


def build_locomo_records(
    spec: dict[str, Any], split: str, base_seed: int
) -> Iterator[DataRecord]:
    """Load LoCoMo-like long dialogue QA rows from local JSON/JSONL or HF.

    The loader is deliberately permissive because public LoCoMo exports differ
    in field names. It looks for a dialogue/session/messages field and QA pairs
    with question/answer-like keys.
    """
    rows = list(_load_locomo_rows(spec, split))
    rng = random.Random(base_seed + (1 if split == "train" else 10_001))
    rng.shuffle(rows)
    limit = spec.get(f"{split}_size", spec.get("size"))
    if limit is not None:
        rows = rows[: int(limit)]

    name = str(spec.get("name", "locomo"))
    emitted = 0
    for row_idx, row in enumerate(rows):
        context = _flatten_dialogue(row)
        if not context.strip():
            continue
        for qa_idx, qa in enumerate(_extract_qa_pairs(row)):
            question = qa.get("question", "").strip()
            answer = qa.get("answer", "").strip()
            if not question or not answer:
                continue
            prompt = (
                "Answer the question from the long conversation. "
                "Use the conversation as the only source of truth.\n\n"
                f"Conversation:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
            yield DataRecord(
                id=f"{name}-{split}-{row_idx:05d}-{qa_idx:03d}",
                dataset=name,
                split=split,
                prompt=prompt,
                target=f" {answer}",
                answers=[answer],
                metadata={
                    "type": "locomo",
                    "row_index": row_idx,
                    "qa_index": qa_idx,
                    "source": spec.get("path") or spec.get("dataset_name"),
                    "question_type": qa.get("category") or qa.get("type"),
                },
            )
            emitted += 1
    if emitted == 0:
        print(f"warning: LoCoMo spec {name!r} emitted no {split} records")


def _load_locomo_rows(spec: dict[str, Any], split: str) -> Iterable[dict[str, Any]]:
    if spec.get("path"):
        return _load_json_rows(Path(spec["path"]))
    if spec.get("dataset_name"):
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise RuntimeError("LoCoMo Hugging Face loading requires `datasets`.") from exc
        hf_split = spec.get(f"{split}_split", split)
        subset = spec.get("subset")
        if subset:
            ds = load_dataset(spec["dataset_name"], subset, split=hf_split)
        else:
            ds = load_dataset(spec["dataset_name"], split=hf_split)
        return [dict(x) for x in ds]
    print("warning: enabled LoCoMo dataset has neither `path` nor `dataset_name`; skipping")
    return []


def _load_json_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    obj = json.loads(path.read_text())
    if isinstance(obj, list):
        return [dict(x) for x in obj]
    for key in ("data", "examples", "conversations", "rows"):
        if isinstance(obj, dict) and isinstance(obj.get(key), list):
            return [dict(x) for x in obj[key]]
    if isinstance(obj, dict):
        return [obj]
    raise ValueError(f"unsupported JSON structure: {path}")


def _flatten_dialogue(row: Any) -> str:
    if isinstance(row, str):
        return row
    if isinstance(row, list):
        parts = [_flatten_dialogue(x) for x in row]
        return "\n".join(p for p in parts if p.strip())
    if not isinstance(row, dict):
        return ""

    official_sessions = _official_locomo_sessions(row)
    if official_sessions:
        return "\n\n".join(
            f"[Session {idx}" + (f" | {dt}" if dt else "") + f"]\n{text}"
            for idx, dt, text in official_sessions
            if text.strip()
        )

    for key in ("conversation", "dialogue", "messages", "sessions", "turns", "chat"):
        if key in row:
            return _flatten_dialogue(row[key])

    text = row.get("text") or row.get("content") or row.get("utterance") or row.get("message")
    if text:
        speaker = row.get("speaker") or row.get("role") or row.get("agent") or row.get("name")
        dia_id = row.get("dia_id")
        prefix = f"{speaker}: " if speaker else ""
        if dia_id:
            prefix = f"[{dia_id}] " + prefix
        if row.get("blip_caption"):
            return f"{prefix}{text}\nImage caption: {row.get('blip_caption')}"
        return f"{prefix}{text}"
    return ""


def _official_locomo_sessions(row: dict[str, Any]) -> list[tuple[int, str, str]]:
    """Return LoCoMo official session text from {session_N: [...]} dicts."""
    session_nums = []
    for key, val in row.items():
        if key.startswith("session_") and key[len("session_"):].isdigit() and isinstance(val, list):
            session_nums.append(int(key[len("session_"):]))
    out = []
    for idx in sorted(session_nums):
        turns = row.get(f"session_{idx}") or []
        dt = str(row.get(f"session_{idx}_date_time") or "")
        text = _flatten_dialogue(turns)
        if text.strip():
            out.append((idx, dt, text))
    return out


def _extract_qa_pairs(row: dict[str, Any]) -> Iterator[dict[str, str]]:
    containers = []
    for key in ("qa", "qas", "question_answers", "questions", "qa_pairs"):
        val = row.get(key)
        if isinstance(val, list):
            containers.extend(val)
    if not containers and any(k in row for k in ("question", "query", "answer", "answers")):
        containers = [row]

    for item in containers:
        if not isinstance(item, dict):
            continue
        q = item.get("question") or item.get("query") or item.get("q")
        a = item.get("answer") or item.get("gold_answer") or item.get("target") or item.get("a")
        if isinstance(a, list):
            a = a[0] if a else ""
        if q and a:
            out = dict(item)
            out["question"] = str(q)
            out["answer"] = str(a)
            yield out


class JsonlCausalDataset:
    """Tokenized causal examples backed by a JSONL manifest."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_length: int,
        overflow: str = "skip",
        add_eos: bool = True,
        reconfig_max_length: int | None = None,
        limit: int | None = None,
    ):
        self.path = Path(path)
        self.records = list(read_jsonl(self.path))
        if limit is not None:
            self.records = self.records[:limit]
        self.examples = []
        self.skipped = 0
        for rec in self.records:
            ex = encode_record(
                rec,
                tokenizer=tokenizer,
                max_length=max_length,
                overflow=overflow,
                add_eos=add_eos,
                reconfig_max_length=reconfig_max_length,
            )
            if ex is None:
                self.skipped += 1
            else:
                self.examples.append(ex)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def encode_record(
    rec: DataRecord,
    tokenizer: Any,
    max_length: int,
    overflow: str = "skip",
    add_eos: bool = True,
    reconfig_max_length: int | None = None,
) -> dict[str, Any] | None:
    prompt_ids = list(tokenizer(rec.prompt, add_special_tokens=False)["input_ids"])
    target_ids = list(tokenizer(rec.target, add_special_tokens=False)["input_ids"])
    eos = getattr(tokenizer, "eos_token_id", None)
    if add_eos and eos is not None:
        target_ids.append(int(eos))
    if not target_ids:
        return None

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if len(input_ids) > max_length:
        input_ids, labels = _handle_overflow(input_ids, labels, max_length, overflow)
        if input_ids is None or labels is None:
            return None
    if not any(x != -100 for x in labels):
        return None

    recfg = prompt_ids
    if reconfig_max_length is not None and len(recfg) > reconfig_max_length:
        recfg = recfg[-reconfig_max_length:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
        "recfg_input_ids": recfg or input_ids,
        "id": rec.id,
        "dataset": rec.dataset,
        "answers": rec.answers,
        "metadata": rec.metadata,
    }


def _handle_overflow(
    input_ids: list[int], labels: list[int], max_length: int, policy: str
) -> tuple[list[int] | None, list[int] | None]:
    policy = policy.lower()
    if policy == "skip":
        return None, None
    if policy == "truncate_left":
        return input_ids[-max_length:], labels[-max_length:]
    if policy == "truncate_right":
        return input_ids[:max_length], labels[:max_length]
    raise ValueError(f"unknown overflow policy: {policy}")


def collate_causal(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    import torch

    bsz = len(batch)
    max_len = max(len(x["input_ids"]) for x in batch)
    max_recfg = max(len(x["recfg_input_ids"]) for x in batch)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    recfg_input_ids = torch.full((bsz, max_recfg), pad_token_id, dtype=torch.long)
    recfg_attention_mask = torch.zeros((bsz, max_recfg), dtype=torch.long)

    for i, ex in enumerate(batch):
        n = len(ex["input_ids"])
        input_ids[i, :n] = torch.tensor(ex["input_ids"], dtype=torch.long)
        labels[i, :n] = torch.tensor(ex["labels"], dtype=torch.long)
        attention_mask[i, :n] = 1
        r = len(ex["recfg_input_ids"])
        recfg_input_ids[i, :r] = torch.tensor(ex["recfg_input_ids"], dtype=torch.long)
        recfg_attention_mask[i, :r] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "recfg_input_ids": recfg_input_ids,
        "recfg_attention_mask": recfg_attention_mask,
        "ids": [x["id"] for x in batch],
        "datasets": [x["dataset"] for x in batch],
        "answers": [x["answers"] for x in batch],
        "metadata": [x["metadata"] for x in batch],
    }
