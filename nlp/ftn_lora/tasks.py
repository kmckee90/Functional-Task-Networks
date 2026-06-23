"""TRACE-aligned task suite for FTN-LoRA.

The original TRACE benchmark uses 8 tasks: C-STANCE, FOMC, MeetingBank,
Py150, ScienceQA, NumGLUE-cm, NumGLUE-ds, 20Minuten. Several of those
datasets are gated, multimodal, or require dataset-builder code execution
that we cannot enable in this environment, so we substitute reliably
loadable analogues that preserve TRACE's diversity goals (multilingual,
multi-domain, multi-format).

Task suite:
    1. AG News         — English topic classification (4-class)         [analogue: news/topic]
    2. DBpedia 14      — English ontology classification (14-class)     [analogue: knowledge categorisation]
    3. TREC            — English question-type classification (6-class) [analogue: ScienceQA in textual form]
    4. dair-ai/emotion — English emotion classification (6-class)
    5. Financial PB    — English finance sentiment (3-class)            [analogue: FOMC]
    6. C-STANCE-A      — Chinese stance detection (3-class)             [identical to TRACE]

All tasks are formatted as instruction → single-token answer in the Qwen2.5
chat template, and the loss is masked to the answer tokens only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import torch
from datasets import load_dataset


@dataclass
class TaskSpec:
    name: str
    instruction: str
    label_strs: list[str]
    formatter: Callable               # row -> (prompt_text, answer_text)
    train_size: int = 500
    eval_size: int = 200
    support_size: int = 16


def _ag_news(row):
    labels = ["World", "Sports", "Business", "Technology"]
    return f"Headline: {row['text'].strip()}\nAnswer with one of: World, Sports, Business, Technology.", \
           labels[int(row["label"])]


def _dbpedia(row):
    labels = [
        "Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
        "Mean of Transportation", "Building", "Natural Place", "Village", "Animal",
        "Plant", "Album", "Film", "Written Work",
    ]
    text = f"{row['title'].strip()}: {row['content'].strip()}"
    return f"Text: {text[:400]}\nWhat is this entity? Answer with one of the 14 categories.", \
           labels[int(row["label"])]


def _trec(row):
    labels = ["Abbreviation", "Entity", "Description", "Person", "Location", "Number"]
    return f"Question: {row['text'].strip()}\nWhat is the question type? Choose from: Abbreviation, Entity, Description, Person, Location, Number.", \
           labels[int(row["coarse_label"])]


def _emotion(row):
    labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    return f"Tweet: {row['text'].strip()}\nWhat emotion does this express? Choose from: sadness, joy, love, anger, fear, surprise.", \
           labels[int(row["label"])]


def _financial(row):
    labels = ["negative", "neutral", "positive"]
    return f"Financial news: {row['sentence'].strip()}\nClassify the sentiment as negative, neutral, or positive.", \
           labels[int(row["label"])]


def _cstance(row):
    # TRACE-original: Chinese stance detection (Support / Against / None)
    # Original labels: 支持 / 反对 / 中立
    label_map = {"支持": "Support", "反对": "Against", "中立": "Neutral"}
    target = row["Target"].strip()
    text = row["Text"].strip()
    stance_zh = row["Stance"].strip()
    answer = label_map.get(stance_zh, "Neutral")
    return f"Statement (Chinese): {text[:400]}\nTarget: {target}\nDoes the author Support, Oppose, or take a Neutral stance toward the target? Answer with one of: Support, Against, Neutral.", answer


TASK_REGISTRY = {
    "ag_news": TaskSpec(
        name="ag_news",
        instruction="Classify the news topic.",
        label_strs=["World", "Sports", "Business", "Technology"],
        formatter=_ag_news,
    ),
    "dbpedia": TaskSpec(
        name="dbpedia",
        instruction="Classify the DBpedia entity type.",
        label_strs=["Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
                    "Mean of Transportation", "Building", "Natural Place", "Village", "Animal",
                    "Plant", "Album", "Film", "Written Work"],
        formatter=_dbpedia,
    ),
    "trec": TaskSpec(
        name="trec",
        instruction="Classify the question type.",
        label_strs=["Abbreviation", "Entity", "Description", "Person", "Location", "Number"],
        formatter=_trec,
    ),
    "emotion": TaskSpec(
        name="emotion",
        instruction="Classify the emotion.",
        label_strs=["sadness", "joy", "love", "anger", "fear", "surprise"],
        formatter=_emotion,
    ),
    "financial": TaskSpec(
        name="financial",
        instruction="Classify the financial sentiment.",
        label_strs=["negative", "neutral", "positive"],
        formatter=_financial,
    ),
    "cstance": TaskSpec(
        name="cstance",
        instruction="Detect the stance.",
        label_strs=["Support", "Against", "Neutral"],
        formatter=_cstance,
    ),
}

DATASET_LOADERS = {
    "ag_news": lambda: load_dataset("ag_news"),
    "dbpedia": lambda: load_dataset("dbpedia_14"),
    "trec": lambda: load_dataset("trec"),
    "emotion": lambda: load_dataset("dair-ai/emotion", "split"),
    "financial": lambda: load_dataset("financial_phrasebank", "sentences_50agree"),
    "cstance": lambda: load_dataset("yfhe/C-STANCE-A"),
}

# Canonical task order for the TRACE-aligned sequence.
DEFAULT_ORDER = ["cstance", "financial", "trec", "emotion", "ag_news", "dbpedia"]


def _stratified_take(rows: list[dict], n: int, label_key: str, seed: int) -> list[dict]:
    """Take n examples while balancing across labels where possible."""
    g = torch.Generator().manual_seed(seed)
    by_label: dict = {}
    for r in rows:
        by_label.setdefault(r[label_key], []).append(r)
    classes = sorted(by_label.keys(), key=lambda x: str(x))
    per = max(1, n // len(classes))
    out: list[dict] = []
    for c in classes:
        bucket = by_label[c]
        if not bucket:
            continue
        idx = torch.randperm(len(bucket), generator=g).tolist()
        out.extend(bucket[i] for i in idx[:per])
        if len(out) >= n:
            break
    if len(out) < n:
        # top up uniformly
        idx = torch.randperm(len(rows), generator=g).tolist()
        existing = set(id(x) for x in out)
        for i in idx:
            if id(rows[i]) not in existing:
                out.append(rows[i])
                if len(out) >= n:
                    break
    return out[:n]


def build_task_dataset(name: str, seed: int = 0,
                       train_size: int = 500, eval_size: int = 200,
                       support_size: int = 16) -> dict:
    """Return {'train': [...], 'eval': [...], 'support': [...]} of (prompt, answer) tuples."""
    spec = TASK_REGISTRY[name]
    raw = DATASET_LOADERS[name]()
    train_split = raw["train"] if "train" in raw else list(raw.values())[0]
    test_split = (raw.get("test") or raw.get("validation")
                  or raw.get("eval") or train_split)
    if test_split is train_split:
        # Use the tail of train as eval
        all_rows = list(train_split)
        test_rows = all_rows[-2 * eval_size:]
        train_rows = all_rows[:-2 * eval_size]
    else:
        train_rows = list(train_split)
        test_rows = list(test_split)

    label_key = "label"
    if name == "trec":
        label_key = "coarse_label"
    elif name == "cstance":
        label_key = "Stance"

    train_rows = _stratified_take(train_rows, train_size + support_size, label_key, seed)
    eval_rows = _stratified_take(test_rows, eval_size, label_key, seed + 7)

    formatter = spec.formatter
    train_examples = [formatter(r) for r in train_rows[support_size:]]
    support_examples = [formatter(r) for r in train_rows[:support_size]]
    eval_examples = [formatter(r) for r in eval_rows]

    return {
        "train": train_examples,
        "support": support_examples,
        "eval": eval_examples,
        "spec": spec,
    }


def build_task_suite(order: Iterable[str], seed: int = 0,
                     train_size: int = 500, eval_size: int = 200,
                     support_size: int = 16) -> list[dict]:
    return [build_task_dataset(n, seed, train_size, eval_size, support_size) for n in order]


def format_chat_qwen(prompt: str, answer: str | None, tokenizer, max_len: int = 1024,
                     sys_msg: str = "You are a helpful assistant. Answer concisely with the requested label."):
    """Apply Qwen2.5 chat template. Returns (input_ids, labels) where labels is
    -100 on prompt tokens, 0..V on answer tokens.

    If `answer` is None we only encode the prompt (for generation/eval).
    """
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": prompt},
    ]
    prompt_ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
    if hasattr(prompt_ids, "input_ids"):
        prompt_ids = prompt_ids["input_ids"]
        if isinstance(prompt_ids[0], list):
            prompt_ids = prompt_ids[0]
    prompt_ids = list(prompt_ids)
    if answer is None:
        return torch.tensor(prompt_ids[:max_len], dtype=torch.long), None
    answer_ids = list(tokenizer(answer, add_special_tokens=False)["input_ids"])
    eos = tokenizer.eos_token_id
    answer_ids = answer_ids + [eos]
    full_ids = (prompt_ids + answer_ids)[:max_len]
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    labels = labels[:max_len]
    return torch.tensor(full_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_for_train(batch_examples, tokenizer, max_len: int = 1024) -> dict:
    """Pack a list of (prompt, answer) pairs into a left-padded batch."""
    ids_list, lbl_list = [], []
    for prompt, answer in batch_examples:
        ids, lbl = format_chat_qwen(prompt, answer, tokenizer, max_len=max_len)
        ids_list.append(ids)
        lbl_list.append(lbl)
    L = max(t.shape[0] for t in ids_list)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    out_ids = torch.full((len(ids_list), L), pad, dtype=torch.long)
    out_lbl = torch.full((len(ids_list), L), -100, dtype=torch.long)
    out_mask = torch.zeros((len(ids_list), L), dtype=torch.long)
    for i, (ids, lbl) in enumerate(zip(ids_list, lbl_list)):
        n = ids.shape[0]
        out_ids[i, :n] = ids
        out_lbl[i, :n] = lbl
        out_mask[i, :n] = 1
    return {"input_ids": out_ids, "labels": out_lbl, "attention_mask": out_mask}


def collate_for_eval(batch_examples, tokenizer, max_len: int = 1024) -> dict:
    """Eval batch with answer tokens for NLL/accuracy computation."""
    return collate_for_train(batch_examples, tokenizer, max_len)


def first_answer_token(example, tokenizer) -> int:
    """Return the first answer-token id for a (prompt, answer) example."""
    _, answer = example
    ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    return ids[0]
