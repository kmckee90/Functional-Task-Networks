"""Mask-overlap-filtered replay buffer for FTN-LoRA training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ReplayBufferConfig:
    capacity: int = 200            # examples per completed task
    rho_low: float = 0.1           # mask-overlap lower bound
    rho_high: float = 0.7          # mask-overlap upper bound
    n_per_step: int = 1


class MaskFilteredReplayBuffer:
    """Stores (prompt, answer, mask) triples; samples by mask-overlap band."""

    def __init__(self, cfg: ReplayBufferConfig):
        self.cfg = cfg
        self._items: list[tuple[tuple, torch.Tensor]] = []
        # Keep per-task counts so the buffer is roughly balanced
        self._task_counts: dict[str, int] = {}

    def add(self, example: tuple, mask: torch.Tensor, task: str):
        # Per-task cap: cfg.capacity // num_tasks_seen heuristic. Simpler: keep
        # at most cfg.capacity items total via reservoir-like replacement.
        if len(self._items) < self.cfg.capacity:
            self._items.append((example, mask.detach().cpu().clone()))
            self._task_counts[task] = self._task_counts.get(task, 0) + 1
        else:
            # Replace a random item
            i = torch.randint(0, self.cfg.capacity, (1,)).item()
            self._items[i] = (example, mask.detach().cpu().clone())

    def sample(self, current_mask: torch.Tensor, n: Optional[int] = None) -> list[tuple]:
        if not self._items:
            return []
        n = n or self.cfg.n_per_step
        cur = current_mask.detach().cpu().float()
        budget = float(cur.sum().clamp(min=1.0))
        # Filter by mask-overlap fraction in (rho_low, rho_high]
        candidates = []
        for ex, m in self._items:
            overlap = float((cur * m.float()).sum() / budget)
            if self.cfg.rho_low < overlap <= self.cfg.rho_high:
                candidates.append(ex)
        if not candidates:
            # If no candidates in band, fall back to uniform — better than no
            # cross-task supervision at all.
            candidates = [ex for ex, _ in self._items]
        idx = torch.randperm(len(candidates))[:n].tolist()
        return [candidates[i] for i in idx]


class UniformReplayBuffer:
    """Non-mask-aware uniform-random replay (matched-memory baseline)."""

    def __init__(self, cfg: ReplayBufferConfig):
        self.cfg = cfg
        self._items: list[tuple] = []

    def add(self, example: tuple, mask=None, task: str = ""):
        if len(self._items) < self.cfg.capacity:
            self._items.append(example)
        else:
            i = torch.randint(0, self.cfg.capacity, (1,)).item()
            self._items[i] = example

    def sample(self, current_mask=None, n: Optional[int] = None) -> list[tuple]:
        if not self._items:
            return []
        n = n or self.cfg.n_per_step
        idx = torch.randperm(len(self._items))[:n].tolist()
        return [self._items[i] for i in idx]
