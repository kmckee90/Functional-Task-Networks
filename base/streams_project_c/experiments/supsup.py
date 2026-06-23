"""
supsup.py — SupSup baseline (Wortsman et al. 2020) on the synthetic
benchmark of project_c.  Adapted to support both classification and
regression by using supervised loss minimisation on a labeled support
batch as the task-inference signal at "reconfig" time, instead of the
classification-specific entropy minimisation from the original paper.

Per-task supermasks are trained over a frozen, randomly-initialised
NeuroModel via top-k straight-through estimator on the mask logits.
Base weights stay frozen for the entire run.
"""
from __future__ import annotations

import sys
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import models  # noqa: E402


def _topk_ste_mask(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Binary top-k mask with straight-through gradient (identity in backward)."""
    topk = scores.topk(k).indices
    hard = torch.zeros_like(scores)
    hard[topk] = 1.0
    return hard.detach() + scores - scores.detach()


def _task_loss(y_pred: torch.Tensor, y: torch.Tensor, is_clf: bool) -> torch.Tensor:
    if is_clf:
        target = y.argmax(-1) if y.dim() == 2 else y.long()
        return F.cross_entropy(y_pred, target)
    return F.mse_loss(y_pred.squeeze(), y.squeeze())


def train_supermask(
    model: nn.Module,
    task_set,
    task_idx: int,
    is_clf: bool,
    k: int,
    n_steps: int,
    batch_size: int,
    lr: float,
    device: str,
) -> torch.Tensor:
    """Train one binary supermask for a single task on a frozen base model."""
    H = model.hidden_size
    scores = nn.Parameter(torch.randn(H, device=device) * 0.01)
    opt = torch.optim.Adam([scores], lr=lr)
    for _ in range(n_steps):
        x, y = task_set.sample_batch(task_idx, batch_size, device=device)
        mask = _topk_ste_mask(scores, k)
        y_pred = model(x, mask)
        loss = _task_loss(y_pred, y, is_clf)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        topk = scores.topk(k).indices
        out = torch.zeros(H, device=device)
        out[topk] = 1.0
    return out


def select_best_supermask(
    model: nn.Module,
    supermasks: dict,
    eval_data: tuple,
    is_clf: bool,
    device: str,
) -> torch.Tensor:
    """Pick the stored supermask with lowest loss on a labeled support batch."""
    x, y = eval_data
    best_idx, best_loss = None, float("inf")
    with torch.no_grad():
        for j, mask in supermasks.items():
            y_pred = model(x, mask.to(device))
            loss = _task_loss(y_pred, y, is_clf).item()
            if loss < best_loss:
                best_loss = loss
                best_idx = j
    return supermasks[best_idx]


def freeze_base(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def block_sequential_supsup(
    model: nn.Module,
    task_set,
    is_clf: bool,
    k: int,
    train_steps: int,
    batch_size: int,
    reconfig_batch_size: int,
    eval_batch_size: int,
    lr: float,
    device: str,
    eval_via_reconfig: bool = True,
) -> tuple[dict, dict]:
    """
    Run SupSup block-sequentially across task_set.n_tasks tasks.

    Returns:
        evals: {"stored": [[...]], "reconfig": [[...]]}  (lower-triangular)
        supermasks: {task_idx: tensor(H,) on cpu}
    """
    model = model.to(device)
    freeze_base(model)
    n_tasks = task_set.n_tasks
    supermasks: dict[int, torch.Tensor] = {}
    perf_stored: list[list[float]] = []
    perf_reconfig: list[list[float]] = []

    for task_idx in range(n_tasks):
        m = train_supermask(
            model, task_set, task_idx, is_clf,
            k=k, n_steps=train_steps,
            batch_size=batch_size, lr=lr, device=device,
        )
        supermasks[task_idx] = m

        row_stored: list[float] = []
        row_reconfig: list[float] = []
        for eval_idx in range(task_idx + 1):
            stored_mask = supermasks[eval_idx].to(device)
            row_stored.append(task_set.evaluate(
                model, stored_mask, eval_idx,
                split="test", batch_size=eval_batch_size, device=device,
            ))
            if eval_via_reconfig:
                eval_data = task_set.sample_batch(
                    eval_idx, reconfig_batch_size, device=device
                )
                picked = select_best_supermask(
                    model,
                    {j: supermasks[j] for j in range(task_idx + 1)},
                    eval_data, is_clf, device,
                )
                row_reconfig.append(task_set.evaluate(
                    model, picked.to(device), eval_idx,
                    split="test", batch_size=eval_batch_size, device=device,
                ))
            else:
                row_reconfig.append(row_stored[-1])
        perf_stored.append(row_stored)
        perf_reconfig.append(row_reconfig)

    return (
        {"stored": perf_stored, "reconfig": perf_reconfig},
        {i: m.cpu() for i, m in supermasks.items()},
    )


def signed_kaiming_init_(model: nn.Module) -> None:
    """
    Replace every weight tensor in `model` with a signed Kaiming-He constant:
    each entry is ±sqrt(2/fan_in) with random sign.  This is the SupSup /
    edge-popup init: random signed subnetworks of such an init are known to
    be more expressive than under standard uniform/Gaussian init, because
    every weight has the same magnitude and the supermask selects only sign
    patterns rather than magnitudes.
    """
    import math
    for module in model.modules():
        if isinstance(module, models.ParallelMLP):
            for i, (w, b) in enumerate(zip(module.weights, module.biases)):
                # ParallelMLP weight shape: (K, fan_in, fan_out)
                fan_in = w.shape[1]
                fan_out = w.shape[2]
                if i < module.n_layers:
                    std = math.sqrt(2.0 / fan_in)              # Kaiming He
                else:
                    std = math.sqrt(2.0 / (fan_in + fan_out))  # Xavier
                signs = torch.randint(0, 2, w.shape, device=w.device).float() * 2 - 1
                w.data = signs * std
                # Biases: small zero-mean normal (signed init usually keeps b small).
                b.data.normal_(0, 0.01)
        elif isinstance(module, nn.Linear):
            fan_in, fan_out = module.in_features, module.out_features
            std = math.sqrt(2.0 / (fan_in + fan_out))
            signs = torch.randint(
                0, 2, module.weight.shape, device=module.weight.device,
            ).float() * 2 - 1
            module.weight.data = signs * std
            if module.bias is not None:
                module.bias.data.zero_()


def make_supsup_model(
    input_size: int,
    output_size: int,
    hidden_size: int,
    seed: int,
    model_kwargs: dict,
    signed_init: bool = False,
) -> nn.Module:
    """Construct a frozen-base NeuroModel for SupSup."""
    torch.manual_seed(seed)
    m = models.NeuroModel(input_size, output_size, hidden_size, **model_kwargs)
    if signed_init:
        signed_kaiming_init_(m)
    freeze_base(m)
    return m
