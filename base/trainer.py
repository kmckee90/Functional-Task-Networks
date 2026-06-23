"""
trainer.py — Block-sequential continual learning trainer for func_net_kt.

Differences vs func_net_spatial/project_c_4/trainer.py:

  * No EWC scaffolding (this project compares mask-optimisation
    strategies, not classical CL regularisers).
  * Mask-filtered replay buffer is enabled by default.
  * Logs the *initial* loss on each new task -- the loss obtained by
    the freshly optimised mask BEFORE any neuron training -- as a
    head-start metric for knowledge transfer.
  * Logs the per-task initial reconfig loss (already tracked by mask
    configurers via ``last_initial_loss`` / ``last_final_loss``).
"""

from __future__ import annotations

import random
from collections import defaultdict, deque
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Replay buffer (mask-correlation filtered)
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """
    Fixed-capacity buffer storing (x, y, mask) triplets for replay.

    Two replay modes:
      * ``mode="mask_overlap"`` (FTN default): overlap-filtered.  ρ =
        (m · m_j) / sum(m); only entries with ρ ∈ (rho_low, rho_high]
        are replayed.  This targets shared-neuron regions at risk of
        interference, leaving disjoint or fully-overlapping memories
        alone.
      * ``mode="random"`` (ER baseline): uniform-random sampling over
        the entire buffer, ignoring the mask.  Identical memory budget;
        isolates whether retention comes from the mask mechanism or
        from replay alone.
    """

    def __init__(
        self,
        max_size: int = 200,
        rho_low: float = 0.2,
        rho_high: float = 0.8,
        mode: str = "mask_overlap",
    ) -> None:
        if mode not in {"mask_overlap", "random"}:
            raise ValueError(f"Unknown replay mode: {mode}")
        self.max_size = max_size
        self.rho_low = rho_low
        self.rho_high = rho_high
        self.mode = mode
        self._buf: deque = deque(maxlen=max_size)

    def add(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
        self._buf.append((x.cpu(), y.cpu(), mask.cpu()))

    def sample(self, current_mask: torch.Tensor, n: int = 1) -> list[tuple]:
        if len(self._buf) == 0:
            return []
        if self.mode == "random":
            return random.sample(list(self._buf), min(n, len(self._buf)))
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

    use_replay: bool = True
    replay_n_per_step: int = 1
    replay_buf_size: int = 200
    # "mask_overlap" (FTN default, rho-filtered) or "random" (ER baseline).
    replay_mode: str = "mask_overlap"

    # Reconfigure mask "task" (once at the start of each task) is the
    # default for KT experiments because we want a single, well-defined
    # mask-optimisation budget per task.  "epoch"/"batch" are also valid.
    reconfig_freq: str = "task"

    device: str = "cuda"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task_loss(
    model: nn.Module,
    mask: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    is_clf: bool,
) -> float:
    """Compute task loss with the given mask (no_grad)."""
    model.eval()
    with torch.no_grad():
        y_pred = model(x, mask)
        if is_clf:
            target = y.argmax(-1) if y.dim() == 2 else y.long()
            loss = F.cross_entropy(y_pred, target)
        else:
            loss = F.mse_loss(y_pred.squeeze(), y.squeeze())
    model.train()
    return float(loss.item())


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
    eval_mask_cfg=None,
) -> dict:
    """
    Run block-sequential continual learning with a mask-only knowledge-
    transfer protocol.

    Returns dict with:
        evals:            {"stored": R, "reconfig": R} performance matrices
        mask_store:       {task_idx: mask tensor (cpu)}
        loss_history:     list of (task, epoch, step, loss)
        head_start:       per task -- {reconfig_initial_loss,
                          reconfig_final_loss, task_loss_with_mask}
                          where task_loss_with_mask is the model's loss
                          on a fresh batch from the new task using ONLY
                          the freshly optimised mask (no neuron training).
        first_epoch_loss: per task -- mean training loss across the first
                          epoch (sanity check on the head-start).
    """
    device = config.device
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0)
    replay_buf = (
        ReplayBuffer(
            max_size=config.replay_buf_size,
            mode=config.replay_mode,
        )
        if config.use_replay
        else None
    )

    perf_matrix_stored: list[list[float]] = []
    perf_matrix_reconfig: list[list[float]] = []
    mask_store: dict[int, torch.Tensor] = {}
    loss_history: list = []
    head_start: list[dict] = []
    first_epoch_losses: list[float] = []

    n_tasks = task_set.n_tasks
    is_clf = getattr(task_set, "is_classification", True)
    hidden_size = model.hidden_size
    m = torch.zeros(hidden_size, device=device)

    for task_idx in range(n_tasks):
        if verbose:
            print("TASK_IDX", task_idx)
        if verbose:
            print(f"\n{'─' * 50}")
            print(f"  Task {task_idx + 1}/{n_tasks}")
            print(f"{'─' * 50}")

        optimizer.state = defaultdict(dict)

        # Initial mask reconfiguration on a single batch -- ALWAYS run
        # at the task boundary so we can log the head-start, even if
        # config.reconfig_freq != "task". The configurer is given a
        # `reconfig_batch_size` batch to optimise on, but the head-start
        # is then measured on a *fresh* held-out batch of size
        # `eval_batch_size` so the metric reports generalisation to the
        # new task rather than fit to the configurer's training batch.
        head_data = task_set.sample_batch(
            task_idx, config.reconfig_batch_size, device=device
        )
        m = mask_cfg(model, head_data, m, task_idx=task_idx)
        mask_store[task_idx] = m.clone()

        eval_x, eval_y = task_set.sample_batch(
            task_idx, config.eval_batch_size, split="train", device=device,
        )
        head_start.append(
            {
                "reconfig_initial_loss": getattr(mask_cfg, "last_initial_loss", None),
                "reconfig_final_loss": getattr(mask_cfg, "last_final_loss", None),
                "task_loss_with_mask": _task_loss(
                    model, m, eval_x, eval_y, is_clf
                ),
            }
        )

        epoch_bar = range(config.n_epochs)
        if verbose:
            epoch_bar = tqdm(epoch_bar, desc=f"  T{task_idx}", leave=False, ncols=72)

        first_epoch_loss_acc = 0.0
        first_epoch_count = 0
        for epoch in epoch_bar:
            if verbose:
                print("EPOCH", epoch)
            if config.reconfig_freq == "epoch" and epoch > 0:
                # The first epoch reuses the task-boundary mask we just
                # logged the head-start for; only refresh thereafter.
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
                if epoch == 0:
                    first_epoch_loss_acc += loss.item()
                    first_epoch_count += 1

            if verbose and hasattr(epoch_bar, "set_postfix"):
                avg_loss = epoch_loss / config.train_steps
                epoch_bar.set_postfix(loss=f"{avg_loss:.4f}")

        first_epoch_losses.append(first_epoch_loss_acc / max(1, first_epoch_count))
        mask_store[task_idx] = m.clone()

        if replay_buf is not None:
            for _ in range(5):
                rx, ry = task_set.sample_batch(
                    task_idx, config.batch_size // 4, device="cpu"
                )
                replay_buf.add(rx, ry, m.cpu())

        # Eval all tasks seen so far.
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
                cfg_for_eval = eval_mask_cfg if eval_mask_cfg is not None else mask_cfg
                reconfig_mask = cfg_for_eval(
                    model,
                    eval_data,
                    torch.zeros(hidden_size, device=device),
                    task_idx=eval_idx,
                    mask_store=mask_store,
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
            print(f"  Eval stored:   {' | '.join(parts_s)}")
            if eval_via_reconfig:
                parts_r = [f"T{j}={v:.4f}" for j, v in enumerate(row_reconfig)]
                print(f"  Eval reconfig: {' | '.join(parts_r)}")

    return {
        "evals": {
            "stored": perf_matrix_stored,
            "reconfig": perf_matrix_reconfig,
        },
        "mask_store": mask_store,
        "loss_history": loss_history,
        "head_start": head_start,
        "first_epoch_loss": first_epoch_losses,
    }
