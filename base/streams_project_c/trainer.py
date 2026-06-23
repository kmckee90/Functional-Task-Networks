"""
trainer.py — Block-sequential continual learning trainer for project_c_3.

Key change vs project_c_2: the evaluation of previous tasks after training a
new task can optionally re-derive the mask via the mask reconfiguration
algorithm (eval_via_reconfig=True) rather than using the stored mask.

This addresses the proper forgetting question: "Can the model recover the
subnetwork for a previous task via mask optimization, given that we don't
know a priori which task we are in?"  The stored masks are kept for
visualization, but performance is measured using the recovered mask.
"""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """
    Fixed-capacity buffer storing (x, y, mask) triplets for selective replay.

    Overlap is measured as:
        ρ = (m · m_j) / sum(m)

    Only entries with ρ ∈ [rho_low, rho_high] are replayed.
    """

    def __init__(
        self,
        max_size: int = 200,
        rho_low: float = 0.2,
        rho_high: float = 0.8,
    ) -> None:
        self.max_size = max_size
        self.rho_low = rho_low
        self.rho_high = rho_high
        self._buf: deque = deque(maxlen=max_size)

    def add(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
        self._buf.append((x.cpu(), y.cpu(), mask.cpu()))

    def sample(self, current_mask: torch.Tensor, n: int = 1) -> list[tuple]:
        if len(self._buf) == 0:
            return []
        denom = current_mask.sum().clamp(min=1).item()
        candidates = []
        for x, y, m in self._buf:
            rho = (current_mask.cpu() * m).sum().item() / denom
            if self.rho_low < rho <= self.rho_high:
                candidates.append((x, y, m))
        if not candidates:
            return []
        return random.sample(candidates, min(n, len(candidates)))

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """Hyperparameters for the block-sequential trainer."""

    n_epochs: int = 10
    train_steps: int = 200
    batch_size: int = 256
    eval_batch_size: int = 1024
    reconfig_batch_size: int = 256

    lr: float = 3e-4
    train_lr: float = 3e-4

    use_replay: bool = False
    replay_n_per_step: int = 1
    replay_buf_size: int = 200

    reconfig_freq: str = "epoch"  # "epoch" | "task" | "batch"

    device: str = "cuda"


# ---------------------------------------------------------------------------
# Core block-sequential trainer
# ---------------------------------------------------------------------------


def block_sequential_train(
    model: nn.Module,
    mask_cfg,
    task_set,
    config: TrainConfig,
    verbose: bool = True,
    eval_via_reconfig: bool = True,
) -> tuple[dict[str, list[list[float]]], dict, list]:
    """
    Run block-sequential continual learning.

    Args:
        model:             MaskedMLP on config.device.
        mask_cfg:          Mask configurer callable (model, data, m) -> mask.
        task_set:          Task set with .sample_batch() and .evaluate().
        config:            Training hyperparameters.
        verbose:           Print progress.
        eval_via_reconfig: If True (default), run both evaluation protocols:
                           - "stored":   use the mask saved during training
                             (oracle — knows which subnetwork belongs to which task).
                           - "reconfig": re-run mask_cfg on a fresh data sample
                             (realistic — task identity unknown at inference time).
                           If False, both protocols return the stored-mask result.

    Returns:
        evals:        {"stored": R, "reconfig": R} where R[i][j] is performance
                      on task j after training through task i.
        mask_store:   {task_idx: mask tensor} — training masks (for visualization).
        loss_history: List of (task_idx, epoch, step, loss) tuples.
    """
    device = config.device
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0)
    replay_buf = (
        ReplayBuffer(max_size=config.replay_buf_size) if config.use_replay else None
    )

    perf_matrix_stored: list[list[float]] = []
    perf_matrix_reconfig: list[list[float]] = []
    mask_store: dict[int, torch.Tensor] = {}
    loss_history: list = []

    n_tasks = task_set.n_tasks
    is_clf = getattr(task_set, "is_classification", True)

    hidden_size = model.hidden_size
    m = torch.zeros(hidden_size, device=device)

    for task_idx in range(n_tasks):
        if verbose:
            print(f"\n{'─' * 50}")
            print(f"  Task {task_idx + 1}/{n_tasks}")
            print(f"{'─' * 50}")

        # Reset optimizer state at task boundary so Adam's (m, v) from the
        # previous task can't drift parameters whose gradient is zero under
        # the new mask (this is what makes FixedMask FM == 0 exactly).
        optimizer.state = defaultdict(dict)

        if config.reconfig_freq == "task":
            data = task_set.sample_batch(
                task_idx, config.reconfig_batch_size, device=device
            )
            m = mask_cfg(model, data, m, task_idx=task_idx)
            mask_store[task_idx] = m.clone()

        epoch_bar = range(config.n_epochs)
        if verbose:
            epoch_bar = tqdm(epoch_bar, desc=f"  T{task_idx}", leave=False, ncols=72)

        for epoch in epoch_bar:
            if config.reconfig_freq == "epoch":
                data = task_set.sample_batch(
                    task_idx, config.reconfig_batch_size, device=device
                )
                m = mask_cfg(model, data, m, task_idx=task_idx)

            epoch_loss = 0.0
            for step in range(config.train_steps):
                if config.reconfig_freq == "batch":
                    data = task_set.sample_batch(
                        task_idx, config.reconfig_batch_size, device=device
                    )
                    m = mask_cfg(model, data, m, task_idx=task_idx)

                x, y = task_set.sample_batch(task_idx, config.batch_size, device=device)

                optimizer.zero_grad()

                y_pred = model(x, m)
                if is_clf:
                    target = y.argmax(-1)
                    loss = F.cross_entropy(y_pred, target)
                else:
                    loss = F.mse_loss(y_pred.squeeze(), y.squeeze())

                loss.backward()

                if replay_buf is not None:
                    for rx, ry, rm in replay_buf.sample(m, n=config.replay_n_per_step):
                        rx, ry, rm = rx.to(device), ry.to(device), rm.to(device)
                        rp = model(rx, rm)
                        if is_clf:
                            rl = F.cross_entropy(rp, ry.argmax(-1))
                        else:
                            rl = F.mse_loss(rp.squeeze(), ry.squeeze())
                        rl.backward()

                optimizer.step()
                epoch_loss += loss.item()
                loss_history.append((task_idx, epoch, step, loss.item()))

            if verbose and hasattr(epoch_bar, "set_postfix"):
                avg_loss = epoch_loss / config.train_steps
                epoch_bar.set_postfix(loss=f"{avg_loss:.4f}")

        # Store final training mask for this task (used for visualization)
        if config.reconfig_freq != "task":
            mask_store[task_idx] = m.clone()

        if replay_buf is not None:
            for _ in range(5):
                rx, ry = task_set.sample_batch(
                    task_idx, config.batch_size // 4, device="cpu"
                )
                replay_buf.add(rx, ry, m.cpu())

        # Evaluate all tasks trained so far with two protocols:
        #   stored:   use the mask saved during training (oracle — knows which task)
        #   reconfig: re-run mask_cfg on fresh data (realistic — task identity unknown)
        row_stored: list[float] = []
        row_reconfig: list[float] = []

        for eval_idx in range(task_idx + 1):
            stored_mask = mask_store[eval_idx].to(device)
            perf_stored = task_set.evaluate(
                model,
                stored_mask,
                eval_idx,
                split="test",
                batch_size=config.eval_batch_size,
                device=device,
            )
            row_stored.append(perf_stored)

            if eval_via_reconfig:
                eval_data = task_set.sample_batch(
                    eval_idx, config.reconfig_batch_size, device=device
                )
                reconfig_mask = mask_cfg(
                    model,
                    eval_data,
                    torch.zeros(hidden_size, device=device),
                    task_idx=eval_idx,
                )
                perf_reconfig = task_set.evaluate(
                    model,
                    reconfig_mask,
                    eval_idx,
                    split="test",
                    batch_size=config.eval_batch_size,
                    device=device,
                )
                row_reconfig.append(perf_reconfig)
            else:
                row_reconfig.append(perf_stored)

        perf_matrix_stored.append(row_stored)
        perf_matrix_reconfig.append(row_reconfig)

        if verbose:
            parts_s = [f"T{j}={v:.4f}" for j, v in enumerate(row_stored)]
            parts_r = [f"T{j}={v:.4f}" for j, v in enumerate(row_reconfig)]
            print(f"  Eval stored:   {' | '.join(parts_s)}")
            if eval_via_reconfig:
                print(f"  Eval reconfig: {' | '.join(parts_r)}")

    return (
        {"stored": perf_matrix_stored, "reconfig": perf_matrix_reconfig},
        mask_store,
        loss_history,
    )


# ---------------------------------------------------------------------------
# EWC baseline
# ---------------------------------------------------------------------------


class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al. 2017).

    L_ewc = λ/2 * Σ_i F_i * (θ_i - θ*_i)^2
    """

    def __init__(self, model: nn.Module, lam: float = 400.0, n_fish: int = 200) -> None:
        self.model = model
        self.lam = lam
        self.n_fish = n_fish
        self._tasks: list[dict] = []

    def register_task(
        self,
        task_set,
        task_idx: int,
        device: str = "cuda",
        batch_size: int = 64,
        k: int = None,
    ) -> None:
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        self.model.eval()
        hidden_size = self.model.hidden_size
        if k is None or k >= hidden_size:
            m = torch.ones(hidden_size, device=device)
        else:
            m = torch.zeros(hidden_size, device=device)
            m[:k] = 1.0
        for _ in range(math.ceil(self.n_fish / batch_size)):
            x, y = task_set.sample_batch(task_idx, batch_size, device=device)
            self.model.zero_grad()
            out = self.model(x, m)
            if getattr(task_set, "is_classification", True):
                loss = F.cross_entropy(out, y.argmax(-1))
            else:
                loss = F.mse_loss(out.squeeze(), y.squeeze())
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

        for n in fisher:
            fisher[n] /= math.ceil(self.n_fish / batch_size)

        params = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self._tasks.append({"fisher": fisher, "params": params})
        self.model.train()

    def penalty(self) -> torch.Tensor:
        if not self._tasks:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for task in self._tasks:
            for n, p in self.model.named_parameters():
                f = task["fisher"][n].to(p.device)
                p0 = task["params"][n].to(p.device)
                loss = loss + (f * (p - p0).pow(2)).sum()
        return self.lam / 2 * loss


def block_sequential_train_ewc(
    model: nn.Module,
    task_set,
    config: TrainConfig,
    ewc_lambda: float = 400.0,
    k: int = None,
    verbose: bool = True,
) -> tuple[dict[str, list[list[float]]], list]:
    """Block-sequential CL with EWC regularisation (no masking)."""
    device = config.device
    model = model.to(device)
    ewc = EWC(model, lam=ewc_lambda)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    hidden_size = model.hidden_size
    if k is None or k >= hidden_size:
        m = torch.ones(hidden_size, device=device)
    else:
        m = torch.zeros(hidden_size, device=device)
        m[:k] = 1.0

    perf_matrix: list[list[float]] = []
    loss_history: list = []
    is_clf = getattr(task_set, "is_classification", True)

    for task_idx in range(task_set.n_tasks):
        if verbose:
            print(f"\n  [EWC] Task {task_idx + 1}/{task_set.n_tasks}")

        # Match block_sequential_train: reset Adam moments at task boundary so
        # carried-over (m, v) can't move params orthogonal to the EWC penalty.
        optimizer.state = defaultdict(dict)

        epoch_bar = range(config.n_epochs)
        if verbose:
            epoch_bar = tqdm(epoch_bar, desc=f"  T{task_idx}", leave=False, ncols=72)

        for epoch in epoch_bar:
            epoch_loss = 0.0
            for step in range(config.train_steps):
                x, y = task_set.sample_batch(task_idx, config.batch_size, device=device)
                optimizer.zero_grad()

                y_pred = model(x, m)
                if is_clf:
                    loss = F.cross_entropy(y_pred, y.argmax(-1))
                else:
                    loss = F.mse_loss(y_pred.squeeze(), y.squeeze())

                loss = loss + ewc.penalty()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                loss_history.append((task_idx, epoch, step, loss.item()))

        ewc.register_task(task_set, task_idx, device=device, k=k)

        row = []
        for eval_idx in range(task_idx + 1):
            perf = task_set.evaluate(
                model,
                m,
                eval_idx,
                split="test",
                batch_size=config.eval_batch_size,
                device=device,
            )
            row.append(perf)
        perf_matrix.append(row)

        if verbose:
            parts = [f"T{j}={v:.4f}" for j, v in enumerate(row)]
            print(f"  Eval: {' | '.join(parts)}")

    # EWC has no mask reconfiguration — stored and reconfig are identical
    return {"stored": perf_matrix, "reconfig": perf_matrix}, loss_history
