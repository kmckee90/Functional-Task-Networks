"""Reservoir replay buffer with mask-overlap-filtered sampling (design 10.8)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .masks import combined_overlap


@dataclass
class ReplayItem:
    graph: object
    labels: dict  # reference-A label dict (final/trace/valid preds)
    mask_ext: np.ndarray | None  # stored hard mask (or soft gates for SOFT_MIXTURE)
    mask_red: np.ndarray | None
    z: np.ndarray | None  # DENSE task vector
    algorithm_id: int  # diagnostics ONLY; never reaches model or configurer
    graph_seed: int
    insertion_step: int


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.items: list[ReplayItem] = []
        self.n_seen = 0

    def __len__(self):
        return len(self.items)

    def insert(self, item: ReplayItem):
        self.n_seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            j = int(self.rng.integers(0, self.n_seen))
            if j < self.capacity:
                self.items[j] = item

    def eligible(self, cur_ext: np.ndarray, cur_red: np.ndarray,
                 k_ext: int, k_red: int, low: float, high: float) -> list[ReplayItem]:
        out = []
        for it in self.items:
            ov = combined_overlap(cur_ext, it.mask_ext, cur_red, it.mask_red, k_ext, k_red)
            if low < ov <= high:
                out.append(it)
        return out

    def sample(self, cur_ext, cur_red, k_ext, k_red, low, high, max_batch: int,
               uniform: bool = False) -> list[ReplayItem]:
        """Overlap-filtered uniform sampling; `uniform=True` skips the filter
        (ORACLE / DISJOINT / DENSE baselines). No fallback to arbitrary replay."""
        pool = self.items if uniform else self.eligible(cur_ext, cur_red, k_ext, k_red, low, high)
        if not pool:
            return []
        if len(pool) <= max_batch:
            return list(pool)
        idx = self.rng.choice(len(pool), size=max_batch, replace=False)
        return [pool[i] for i in idx]

    def task_histogram(self) -> dict[int, int]:
        h: dict[int, int] = {}
        for it in self.items:
            h[it.algorithm_id] = h.get(it.algorithm_id, 0) + 1
        return h
