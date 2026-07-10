from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import get_path, run_dir
from .data import JsonlCausalDataset, collate_causal, manifest_path
from .models import ModelAdapter, build_model


def build_loader(
    cfg: dict[str, Any],
    adapter: ModelAdapter,
    split: str,
    limit: int | None = None,
) -> tuple[DataLoader, JsonlCausalDataset]:
    section = cfg.get("train" if split == "train" else "eval", {})
    max_length = int(section.get("max_length", 2048))
    overflow = str(section.get("overflow", "skip"))
    reconfig_max_length = section.get("reconfig_max_length")
    if reconfig_max_length is not None:
        reconfig_max_length = int(reconfig_max_length)
    path = manifest_path(cfg, split)
    if not path.exists():
        raise FileNotFoundError(f"missing manifest {path}; run `prepare` first")
    pad = adapter.tokenizer.pad_token_id
    if pad is None:
        pad = adapter.tokenizer.eos_token_id or 0
    dataset = JsonlCausalDataset(
        path,
        adapter.tokenizer,
        max_length=max_length,
        overflow=overflow,
        reconfig_max_length=reconfig_max_length,
        limit=limit,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            f"{split} dataset is empty after tokenization; skipped={dataset.skipped}. "
            "Increase max_length or change overflow policy."
        )
    loader = DataLoader(
        dataset,
        batch_size=int(section.get("batch_size", 1)),
        shuffle=(split == "train"),
        num_workers=int(section.get("num_workers", 0)),
        collate_fn=lambda batch: collate_causal(batch, int(pad)),
    )
    if dataset.skipped:
        print(f"{split}: skipped {dataset.skipped} examples that exceeded max_length")
    return loader, dataset


def train_model(
    cfg: dict[str, Any],
    kind: str,
    checkpoint: str | Path | None = None,
    limit_train: int | None = None,
    limit_eval: int | None = None,
) -> dict[str, Any]:
    adapter = build_model(kind, cfg)
    if checkpoint:
        adapter.load_checkpoint(checkpoint)
    params = adapter.trainable_parameters()
    if not params:
        print(f"{kind} has no trainable parameters; running evaluation only")
        return evaluate_adapter(cfg, adapter, split="eval", limit=limit_eval)

    out_dir = run_dir(cfg, kind)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    loader, dataset = build_loader(cfg, adapter, "train", limit=limit_train)
    train_cfg = cfg.get("train", {})
    opt = torch.optim.AdamW(params, lr=float(train_cfg.get("lr", 2e-4)), foreach=False)
    grad_accum = max(1, int(train_cfg.get("grad_accum", 1)))
    epochs = int(train_cfg.get("epochs", 1))
    max_steps = train_cfg.get("max_steps")
    max_steps = int(max_steps) if max_steps is not None else None
    log_every = max(1, int(train_cfg.get("log_every", 10)))
    save_every = int(train_cfg.get("save_every", 0) or 0)

    step = 0
    running = None
    t0 = time.time()
    adapter.model.train()
    opt.zero_grad(set_to_none=True)
    pending_grads = 0
    with metrics_path.open("a") as mf:
        for epoch in range(epochs):
            pbar = tqdm(loader, desc=f"train {kind} e{epoch}", leave=False)
            for batch_idx, batch in enumerate(pbar):
                step += 1
                adapter.before_batch(_move_batch(batch, adapter.device), train=True)
                batch = _move_batch(batch, adapter.device)
                out = adapter.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss / grad_accum
                loss.backward()
                pending_grads += 1
                if step % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(params, float(train_cfg.get("max_grad_norm", 1.0)))
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    pending_grads = 0

                lv = float(out.loss.detach())
                running = lv if running is None else 0.9 * running + 0.1 * lv
                pbar.set_postfix(loss=f"{lv:.3f}", ema=f"{running:.3f}")
                if step % log_every == 0:
                    rec = {
                        "phase": "train",
                        "model": kind,
                        "step": step,
                        "epoch": epoch,
                        "loss": lv,
                        "loss_ema": running,
                        "examples": len(dataset),
                        "wall_seconds": time.time() - t0,
                    }
                    mf.write(json.dumps(rec) + "\n")
                    mf.flush()
                if save_every and step % save_every == 0:
                    adapter.save_checkpoint(ckpt_dir / f"step_{step}.pt")
                if max_steps is not None and step >= max_steps:
                    break
            if max_steps is not None and step >= max_steps:
                break

    if pending_grads:
        torch.nn.utils.clip_grad_norm_(params, float(train_cfg.get("max_grad_norm", 1.0)))
        opt.step()
        opt.zero_grad(set_to_none=True)

    latest = ckpt_dir / "latest.pt"
    adapter.save_checkpoint(latest)
    print(f"saved checkpoint: {latest}")
    summary = evaluate_adapter(cfg, adapter, split="eval", limit=limit_eval)
    summary["checkpoint"] = str(latest)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def evaluate_model(
    cfg: dict[str, Any],
    kind: str,
    checkpoint: str | Path | None = None,
    split: str = "eval",
    limit: int | None = None,
) -> dict[str, Any]:
    adapter = build_model(kind, cfg)
    if checkpoint:
        adapter.load_checkpoint(checkpoint)
    return evaluate_adapter(cfg, adapter, split=split, limit=limit)


def evaluate_adapter(
    cfg: dict[str, Any],
    adapter: ModelAdapter,
    split: str = "eval",
    limit: int | None = None,
) -> dict[str, Any]:
    kind = adapter.kind
    out_dir = run_dir(cfg, kind)
    out_dir.mkdir(parents=True, exist_ok=True)
    loader, dataset = build_loader(cfg, adapter, split, limit=limit)
    compute_accuracy = bool(get_path(cfg, "eval", "compute_accuracy", default=False))
    adapter.model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_examples = 0
    token_correct = 0
    exact_correct = 0
    t0 = time.time()

    for batch in tqdm(loader, desc=f"eval {kind}", leave=False):
        moved = _move_batch(batch, adapter.device)
        adapter.before_batch(moved, train=False)
        with torch.no_grad():
            out = adapter.model(
                input_ids=moved["input_ids"],
                attention_mask=moved["attention_mask"],
                labels=moved["labels"],
            )
        label_tokens = _label_token_count(moved["labels"])
        total_nll += float(out.loss.detach()) * max(1, label_tokens)
        total_tokens += label_tokens
        total_examples += moved["input_ids"].size(0)
        if compute_accuracy:
            correct, total, exact = _teacher_forced_accuracy(out.logits, moved["labels"])
            token_correct += correct
            exact_correct += exact

    mean_nll = total_nll / max(1, total_tokens)
    summary = {
        "phase": "eval",
        "model": kind,
        "split": split,
        "examples": total_examples,
        "manifest_examples": len(dataset.records),
        "tokenized_examples": len(dataset),
        "skipped_examples": dataset.skipped,
        "label_tokens": total_tokens,
        "nll": mean_nll,
        "perplexity": math.exp(mean_nll) if mean_nll < 50 else float("inf"),
        "wall_seconds": time.time() - t0,
    }
    if compute_accuracy:
        summary["teacher_forced_token_accuracy"] = token_correct / max(1, total_tokens)
        summary["teacher_forced_exact"] = exact_correct / max(1, total_examples)
    summary_path = out_dir / f"{split}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    with (out_dir / "metrics.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {}
    for k, v in batch.items():
        moved[k] = v.to(device) if torch.is_tensor(v) else v
    return moved


def _label_token_count(labels: torch.Tensor) -> int:
    if labels.size(1) <= 1:
        return 0
    return int((labels[:, 1:] != -100).sum().item())


def _teacher_forced_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> tuple[int, int, int]:
    preds = logits[:, :-1, :].argmax(dim=-1)
    gold = labels[:, 1:]
    mask = gold != -100
    correct = int(((preds == gold) & mask).sum().item())
    total = int(mask.sum().item())
    exact = 0
    for i in range(gold.size(0)):
        m = mask[i]
        if int(m.sum().item()) > 0 and bool(torch.equal(preds[i][m], gold[i][m])):
            exact += 1
    return correct, total, exact
