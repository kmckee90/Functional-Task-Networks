"""Retrieve a saved Bayesian mask-logit snapshot from the existing replay data.

The replay buffer still serves exactly the same gradient updates. This module
only reads its saved inputs and logits at evaluation; support labels are used
only by the optional refinement and zero-prior control.
"""

from collections import defaultdict

import numpy as np
import torch

from ftn import kwta


class ReplayLogitPrior:
    def __init__(self, config, replay):
        if config.RP_NEIGHBORS < 1:
            raise ValueError("RP_NEIGHBORS must be at least one")
        self.config = config
        self.replay = replay
        self.match_errors = []
        self.match_tasks = []
        self.evals = 0
        self.misses = 0
        self.by_task = defaultdict(list)

    def train_step(self, x, y):
        pass

    def train_block_end(self, predictor, task):
        pass

    def _topk(self, predictor, logits):
        if hasattr(predictor, "_topk_mask"):
            return predictor._topk_mask(logits)
        return kwta(logits, predictor.k)

    @torch.no_grad()
    def _match(self, xs):
        if not self.replay.logits:
            return None, float("inf")
        if self.config.RP_MATCH == "nearest":
            rows = torch.stack([data[0] for data in self.replay.data])
            distance = torch.cdist(xs.expand(len(rows), -1, -1), rows)
            errors = distance.min(dim=-1).values.mean(dim=-1)
        elif self.config.RP_MATCH == "subspace":
            bases = torch.stack(list(self.replay.bases))
            projection = torch.einsum("sd,ndr->nsr", xs, bases)
            reconstruction = torch.einsum("nsr,ndr->nsd", projection, bases)
            power = xs.square().sum().clamp_min(1e-12)
            errors = (xs.unsqueeze(0) - reconstruction).square().sum(dim=(1, 2)) / power
        else:
            raise ValueError("RP_MATCH must be nearest or subspace")
        # Most recent entry wins a numerical tie; useful when an old domain
        # has multiple posterior snapshots in the replay buffer.
        best = errors.min()
        candidates = torch.where(errors <= best + self.config.RP_TIE_TOLERANCE)[0]
        index = int(candidates[-1].item())
        if self.config.RP_NEIGHBORS == 1:
            indices = [index]
        else:
            count = min(self.config.RP_NEIGHBORS, len(errors))
            indices = torch.argsort(errors, stable=True)[:count].tolist()
        return indices, float(errors[index].item())

    def eval_reconfigure(self, predictor, x, y, task=None):
        c = self.config
        xs, ys = x[:c.IC_SUPPORT_SIZE], y[:c.IC_SUPPORT_SIZE]
        if c.RP_EVAL_STRATEGY == "zero":
            return predictor.cold_reconfigure(xs, ys, **c.EVAL_FTN_PARS)
        indices, error = self._match(xs)
        if indices is None:
            self.misses += 1
            return predictor.cold_reconfigure(xs, ys, **c.EVAL_FTN_PARS)
        self.evals += 1
        self.match_errors.append(error)
        if task is not None:
            correct = sum(self.replay.tasks[index] == task for index in indices) / len(indices)
            self.match_tasks.append(correct)
            self.by_task[task].append(correct)
        logits = torch.stack([self.replay.logits[index] for index in indices]).mean(0)
        if c.RP_EVAL_STRATEGY == "direct":
            predictor.m.copy_(self._topk(predictor, logits))
            return predictor.m
        if c.RP_EVAL_STRATEGY != "refine":
            raise ValueError("RP_EVAL_STRATEGY must be zero, direct, or refine")
        saved_logits = predictor.mask_logits.detach().clone()
        try:
            predictor.mask_logits.copy_(logits)
            predictor._posterior_update(
                xs, ys, steps=c.RP_PRIOR_STEPS,
                lr=c.EVAL_RECONFIG_LR, sample_size=None,
                kl_weight=c.RP_PRIOR_KL,
                temperature=c.RP_PRIOR_TEMPERATURE,
                prior_decay=1.0, stochastic_mask=False,
                changepoint=False)
        finally:
            predictor.mask_logits.copy_(saved_logits)
        return predictor.m

    def record_query(self, predictor, x, y, task):
        pass

    def metrics(self):
        result = {
            "rp_evals": self.evals,
            "rp_misses": self.misses,
            "rp_match_error": float(np.mean(self.match_errors)) if self.match_errors else None,
            "rp_match_task_accuracy": float(np.mean(self.match_tasks)) if self.match_tasks else None,
        }
        for task, values in self.by_task.items():
            result[f"rp_t{task}_match_task_accuracy"] = float(np.mean(values))
        return result
