"""Sparse, task-balanced mask-filtered replay for context distillation.

One random Q&A example can be proposed per training step. Bernoulli sampling
and a minimum step gap spread proposals over the block; per-task reservoir
replacement avoids favoring its beginning. Only token IDs and a hard mask are
stored. The frozen teacher is recomputed when an example is replayed.
"""
from __future__ import annotations

import torch


_FIELDS = ("ids", "lbl", "t_ids", "t_lbl")


class MaskReplayBuffer:
    def __init__(self, cfg):
        if not (0 <= cfg.replay_store_prob <= 1 and 0 <= cfg.replay_use_prob <= 1):
            raise ValueError("replay probabilities must be in [0, 1]")
        if min(cfg.replay_min_gap, cfg.replay_per_task, cfg.replay_max_entries) < 1:
            raise ValueError("replay gap and capacities must be positive")
        if not 0 <= cfg.replay_overlap_min < cfg.replay_overlap_max <= 1:
            raise ValueError("replay overlap range must lie in [0, 1]")
        if cfg.replay_weight < 0:
            raise ValueError("replay weight must be nonnegative")
        self.cfg = cfg
        self.entries = {}  # task -> [{example: CPU tensors, mask: CPU k-hot}]
        self.steps_seen = {}
        self.last_candidate_step = {}
        self.candidates_seen = {}

    def __len__(self):
        return sum(map(len, self.entries.values()))

    def state_dict(self):
        return {
            "entries": self.entries,
            "steps_seen": self.steps_seen,
            "last_candidate_step": self.last_candidate_step,
            "candidates_seen": self.candidates_seen,
        }

    def load_state_dict(self, state):
        entries = state.get("entries", {})
        for task_entries in entries.values():
            for item in task_entries:
                if item["mask"].numel() != self.cfg.H:
                    raise ValueError("replay mask size differs from configured atom bank")
        self.entries = entries
        self.steps_seen = state.get("steps_seen", {})
        self.last_candidate_step = state.get("last_candidate_step", {})
        self.candidates_seen = state.get("candidates_seen", {})

    def consider(self, task, examples, mask, gen):
        """Possibly store one uniformly chosen task example; return True if kept."""
        step = self.steps_seen.get(task, 0)
        self.steps_seen[task] = step + 1
        if not examples or self.cfg.replay_store_prob <= 0:
            return False
        if step - self.last_candidate_step.get(task, -self.cfg.replay_min_gap) < self.cfg.replay_min_gap:
            return False
        if float(torch.rand((), generator=gen, device="cpu")) >= self.cfg.replay_store_prob:
            return False
        self.last_candidate_step[task] = step
        seen = self.candidates_seen.get(task, 0) + 1
        self.candidates_seen[task] = seen
        bucket = self.entries.setdefault(task, [])
        if len(bucket) >= self.cfg.replay_per_task:
            slot = int(torch.randint(seen, (), generator=gen, device="cpu"))
            if slot >= self.cfg.replay_per_task:
                return False
        else:
            if len(self) >= self.cfg.replay_max_entries:
                # First trim a task with multiple entries. When every task
                # has one, replace within this task if possible, preserving
                # at least one memory for each represented task.
                surplus = [name for name, items in self.entries.items() if len(items) > 1]
                if surplus:
                    largest = max(surplus, key=lambda name: len(self.entries[name]))
                    old = self.entries[largest]
                    old.pop(int(torch.randint(len(old), (), generator=gen, device="cpu")))
                elif bucket:
                    slot = int(torch.randint(seen, (), generator=gen, device="cpu"))
                    if slot >= len(bucket):
                        return False
                else:
                    # More distinct tasks than global slots: one must leave.
                    names = [name for name, items in self.entries.items() if items]
                    victim = names[int(torch.randint(len(names), (), generator=gen, device="cpu"))]
                    self.entries[victim].pop()
            if len(self) < self.cfg.replay_max_entries:
                slot = len(bucket)
        example = examples[int(torch.randint(len(examples), (), generator=gen, device="cpu"))]
        item = {
            "example": {name: example[name].detach().cpu().clone() for name in _FIELDS},
            "mask": mask.detach().float().cpu().clone(),
            "step": step,
        }
        if slot == len(bucket):
            bucket.append(item)
        else:
            bucket[slot] = item
        return True

    def eligible(self, mask):
        """Stored entries with partial overlap, following base FTN replay."""
        current = mask.detach().float().cpu()
        active = max(float(current.sum()), 1.0)
        out = []
        for bucket in self.entries.values():
            for item in bucket:
                overlap = float((current * item["mask"]).sum()) / active
                if self.cfg.replay_overlap_min < overlap <= self.cfg.replay_overlap_max:
                    out.append(item)
        return out

    def sample(self, mask, gen):
        choices = self.eligible(mask)
        if not choices:
            return None
        index = int(torch.randint(len(choices), (), generator=gen, device="cpu"))
        return choices[index]
