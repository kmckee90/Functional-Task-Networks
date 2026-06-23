"""
trainer.py — Block-sequential trainer for the LoRA-Mux LLM experiment.

For each task, the trainer:
  (1) reconfigures the LoRA-component mask using a fresh support batch,
  (2) trains LoRA (A, B) parameters on that task with the mask frozen,
  (3) evaluates all tasks seen so far under both
      - "stored": use the mask saved at training time (oracle protocol)
      - "reconfig": cold-start the configurer on a fresh support batch
                    (realistic, task-identity-hidden protocol).

CE loss is computed only on target tokens; the source-prefix tokens are
masked with -100 in the labels so that the LM is supervised exclusively
on style-transformed continuations.
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


SEP = "\n=>\n"  # source / target separator in the encoded sequence


# ---------------------------------------------------------------------------
# Tokenisation utilities
# ---------------------------------------------------------------------------


def encode_pairs(
    tokenizer,
    pairs: List[Tuple[str, str]],
    max_len: int,
    device: torch.device,
):
    """
    Encode (source, target) pairs as causal-LM training examples:
        sequence = source + SEP + target
        labels   = -100 on (source + SEP); target tokens kept.

    Returns input_ids, attention_mask, labels — all (B, L).
    """
    full = [s + SEP + t for s, t in pairs]
    enc = tokenizer(
        full,
        padding=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    labels = input_ids.clone()
    labels[attention_mask == 0] = -100

    # Mask the source-prefix tokens so we only supervise the target.
    for i, (src, _) in enumerate(pairs):
        prefix_text = src + SEP
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        cut = min(len(prefix_ids), input_ids.size(1))
        labels[i, :cut] = -100

    return input_ids, attention_mask, labels


def lm_loss(model, input_ids, attention_mask, labels) -> torch.Tensor:
    """Cross-entropy on `labels != -100` positions only.  HF's GPT-2 head
    auto-shifts inputs/labels and applies CE with ignore_index=-100."""
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    return out.loss


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    n_steps_per_task: int = 200
    train_batch_size: int = 8
    eval_batch_size: int = 16
    reconfig_batch_size: int = 8
    lr: float = 5e-4
    max_seq_len: int = 96
    device: str = "cuda"
    log_every: int = 50


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def block_sequential_train(
    model: nn.Module,
    mask_state,
    configurer,
    task_set,
    tokenizer,
    cfg: TrainConfig,
    seed: int = 0,
    verbose: bool = True,
) -> Dict:
    """
    Run block-sequential continual learning over the task suite.  Returns
    a dict with per-task perf matrices for both protocols, the stored masks,
    and the loss history.
    """
    device = torch.device(cfg.device)
    model.to(device)

    rng = random.Random(seed)
    n_tasks = task_set.n_tasks
    H = mask_state.n_components

    # Train only parameters that require grad (= LoRA A/B).
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=cfg.lr, weight_decay=0.0)

    perf_stored: List[List[float]] = []
    perf_reconfig: List[List[float]] = []
    mask_store: Dict[int, torch.Tensor] = {}
    loss_history: List[Tuple[int, int, float]] = []

    def make_loss_fn(pairs):
        def _loss_fn():
            ids, am, lbl = encode_pairs(tokenizer, pairs, cfg.max_seq_len, device)
            return lm_loss(model, ids, am, lbl)
        return _loss_fn

    @torch.no_grad()
    def eval_task(task_idx: int) -> float:
        pairs = task_set.all_pairs(task_idx, split="test")
        # Chunk through eval set for memory.
        was_training = model.training
        model.eval()
        total_loss, total_tokens = 0.0, 0
        for start in range(0, len(pairs), cfg.eval_batch_size):
            chunk = pairs[start : start + cfg.eval_batch_size]
            ids, am, lbl = encode_pairs(tokenizer, chunk, cfg.max_seq_len, device)
            out = model(input_ids=ids, attention_mask=am, labels=lbl)
            n = (lbl != -100).sum().item()
            total_loss += out.loss.item() * n
            total_tokens += n
        if was_training:
            model.train()
        return total_loss / max(total_tokens, 1)

    for task_idx in range(n_tasks):
        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  Task {task_idx + 1}/{n_tasks}: '{task_set.style_names()[task_idx]}'")
            print(f"{'─' * 60}")

        # Reset Adam (m, v) at task boundary.
        optimizer.state = defaultdict(dict)

        # Configure mask for this task on a fresh support batch.
        support = task_set.sample_pairs(task_idx, cfg.reconfig_batch_size, rng, split="train")
        mask = configurer(make_loss_fn(support), mask_state, device, task_idx=task_idx)
        mask_state.set(mask)
        mask_store[task_idx] = mask.detach().clone()

        # Train
        model.train()
        for step in range(cfg.n_steps_per_task):
            pairs = task_set.sample_pairs(task_idx, cfg.train_batch_size, rng, split="train")
            ids, am, lbl = encode_pairs(tokenizer, pairs, cfg.max_seq_len, device)
            optimizer.zero_grad()
            loss = lm_loss(model, ids, am, lbl)
            loss.backward()
            optimizer.step()
            loss_history.append((task_idx, step, loss.item()))
            if verbose and (step + 1) % cfg.log_every == 0:
                print(f"    step {step + 1:>4d}/{cfg.n_steps_per_task}  loss={loss.item():.4f}")

        # Evaluate every task seen so far under both protocols
        row_stored: List[float] = []
        row_reconfig: List[float] = []
        for j in range(task_idx + 1):
            # Stored: use the saved training-time mask
            mask_state.set(mask_store[j].to(device))
            row_stored.append(eval_task(j))

            # Reconfig: cold-start mask recovery on a fresh support batch
            support_j = task_set.sample_pairs(j, cfg.reconfig_batch_size, rng, split="train")
            recovered = configurer(make_loss_fn(support_j), mask_state, device, task_idx=j)
            mask_state.set(recovered.to(device))
            row_reconfig.append(eval_task(j))

        perf_stored.append(row_stored)
        perf_reconfig.append(row_reconfig)
        if verbose:
            stored_str = "  ".join(f"T{j}={v:.4f}" for j, v in enumerate(row_stored))
            recfg_str = "  ".join(f"T{j}={v:.4f}" for j, v in enumerate(row_reconfig))
            print(f"  stored:   {stored_str}")
            print(f"  reconfig: {recfg_str}")

    return {
        "perf_matrix_stored": perf_stored,
        "perf_matrix_reconfig": perf_reconfig,
        "mask_store": {k: v.cpu().tolist() for k, v in mask_store.items()},
        "loss_history": loss_history,
        "style_names": task_set.style_names(),
    }


# ---------------------------------------------------------------------------
# CL metrics — computed in this file so the experiment is self-contained
# ---------------------------------------------------------------------------


def cl_metrics(perf_matrix: List[List[float]]) -> Dict[str, float]:
    """
    Compute Average Loss (ACC analogue, lower-is-better for LM CE), Forgetting
    Measure (rise in CE on prior tasks from when they were freshly trained to
    the final state), and Backward Transfer.

    For LM CE (lower-is-better), we define
        FM  = mean over j<N-1 of (R[N-1][j] - min_{i>=j} R[i][j])
            i.e. how much each prior task's loss has *risen* above the best
            level it ever reached (so a positive FM means forgetting).
        BWT = mean over j<N-1 of (R[N-1][j] - R[j][j])
            negative = improvement, positive = forgetting.
    """
    N = len(perf_matrix)
    final = perf_matrix[-1]
    avg_loss = sum(final) / len(final)
    if N < 2:
        return {"avg_loss": float(avg_loss), "FM": 0.0, "BWT": 0.0}

    fm_terms, bwt_terms = [], []
    for j in range(N - 1):
        col = [perf_matrix[i][j] for i in range(j, N)]
        best = min(col)
        fm_terms.append(perf_matrix[-1][j] - best)
        bwt_terms.append(perf_matrix[-1][j] - perf_matrix[j][j])
    return {
        "avg_loss": float(avg_loss),
        "FM": float(sum(fm_terms) / len(fm_terms)),
        "BWT": float(sum(bwt_terms) / len(bwt_terms)),
    }
