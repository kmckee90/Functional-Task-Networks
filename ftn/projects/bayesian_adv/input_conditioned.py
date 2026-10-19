"""Input-reconstruction routing with replay-repaired Bayesian mask memory.

The router is an append-only collection of rank-r linear autoencoders (PCA
projectors). It selects a mask from a few unlabeled inputs. Old projectors never
receive gradient updates; labeled *training replay* may replace a stale mask
value by appending a newer entry for the same projector.
"""

from dataclasses import dataclass

import torch

from ftn import kwta


@dataclass
class RouteEntry:
    projector: torch.Tensor
    logits: torch.Tensor
    samples: int


class InputConditionedRouting:
    """Mixin for BayesFTN, BayesLoRAFTN, and BayesLoRARNNFTN."""

    def configure_input_routing(self, config):
        self.ic_support_size = config.IC_SUPPORT_SIZE
        self.ic_rank = config.IC_AE_RANK
        self.ic_threshold = config.IC_RECON_THRESHOLD
        self.ic_match_tolerance = config.IC_MATCH_TOLERANCE
        self.ic_same_domain_distance = config.IC_SAME_DOMAIN_DISTANCE
        self.ic_novelty_trigger = config.IC_NOVELTY_TRIGGER
        self.ic_train_prior = config.IC_TRAIN_PRIOR
        self.ic_prior_steps = config.IC_PRIOR_STEPS
        self.ic_prior_kl = config.IC_PRIOR_KL
        self.ic_eval_strategy = config.IC_EVAL_STRATEGY
        self.ic_repair = config.IC_REPLAY_REPAIR
        self.ic_repair_fit_rows = config.IC_REPAIR_FIT_ROWS
        self.ic_repair_validation_rows = config.IC_REPAIR_VALIDATION_ROWS
        self.ic_repair_steps = config.IC_REPAIR_STEPS
        self.ic_repair_lr = config.IC_REPAIR_LR
        self.ic_repair_temperature = config.IC_REPAIR_TEMPERATURE
        self.ic_repair_restarts = config.IC_REPAIR_RESTARTS
        self.ic_repair_take_best = config.IC_REPAIR_TAKE_BEST
        self.ic_repair_min_improvement = config.IC_REPAIR_MIN_IMPROVEMENT
        self.ic_repair_logit_magnitude = config.IC_REPAIR_LOGIT_MAGNITUDE
        self.ic_repair_every_blocks = config.IC_REPAIR_EVERY_BLOCKS
        self.ic_blocks = 0
        self.ic_entries = []
        self.ic_cov = None
        self.ic_samples = 0
        self.ic_stats = dict(eval_hits=0, eval_misses=0, train_hits=0,
                             train_misses=0, novelty_triggers=0,
                             repair_attempts=0, repair_accepted=0)
        self.ic_repair_before = []
        self.ic_repair_after = []

        if self.ic_support_size < 1 or self.ic_rank < 1:
            raise ValueError("IC_SUPPORT_SIZE and IC_AE_RANK must be positive")
        if self.ic_rank > self.ic_support_size:
            raise ValueError("IC_AE_RANK cannot exceed IC_SUPPORT_SIZE")
        if self.ic_threshold <= 0 or self.ic_match_tolerance < 0 or self.ic_same_domain_distance <= 0:
            raise ValueError("IC reconstruction threshold must be positive")
        if self.ic_eval_strategy not in ("direct", "refine"):
            raise ValueError("IC_EVAL_STRATEGY must be 'direct' or 'refine'")
        if self.ic_prior_steps < 1 or self.ic_prior_kl < 0:
            raise ValueError("IC_PRIOR_STEPS must be positive and IC_PRIOR_KL nonnegative")
        if min(self.ic_repair_fit_rows, self.ic_repair_validation_rows,
               self.ic_repair_steps, self.ic_repair_restarts,
               self.ic_repair_every_blocks) < 1:
            raise ValueError("IC replay repair sizes, steps, restarts, and interval must be positive")
        if self.ic_repair_lr <= 0 or self.ic_repair_temperature <= 0:
            raise ValueError("IC repair learning rate and temperature must be positive")
        if self.ic_repair_min_improvement < 0 or self.ic_repair_logit_magnitude <= 0:
            raise ValueError("IC repair improvement must be nonnegative and logit magnitude positive")

    def ic_observe(self, x):
        """Accumulate unlabeled training inputs for the active domain."""
        with torch.no_grad():
            scatter = x.detach().T @ x.detach()
            self.ic_cov = scatter if self.ic_cov is None else self.ic_cov + scatter
            self.ic_samples += x.shape[0]

    def _ic_projector(self):
        if self.ic_cov is None or self.ic_samples < self.ic_rank:
            return None
        _, vecs = torch.linalg.eigh(self.ic_cov)
        basis = vecs[:, -self.ic_rank:]
        return basis @ basis.T

    def _ic_archive(self):
        projector = self._ic_projector()
        if projector is not None:
            self.ic_entries.append(RouteEntry(projector.detach().clone(),
                                self.mask_logits.detach().clone(), self.ic_samples))
        self.ic_cov, self.ic_samples = None, 0

    def _ic_candidates(self, include_live=True):
        entries = self.ic_entries.copy()
        if include_live:
            projector = self._ic_projector()
            if projector is not None:
                entries.append(RouteEntry(projector, self.mask_logits, self.ic_samples))
        return entries

    def _ic_match(self, x, include_live=True):
        entries = self._ic_candidates(include_live)
        if not entries:
            return None, float("inf")
        xs = x[:self.ic_support_size].detach()
        power = xs.square().sum().clamp_min(1e-12)
        scored = [(((xs - xs @ e.projector).square().sum() / power).item(), e)
                  for e in entries]
        best = min(error for error, _ in scored)
        error, entry = next((error, entry) for error, entry in reversed(scored)
                            if error <= best + self.ic_match_tolerance)
        return (entry if error < self.ic_threshold else None), error

    def ic_novelty(self, x):
        if not self.ic_novelty_trigger:
            return False
        projector = self._ic_projector()
        if projector is None:
            return False
        xs = x[:self.ic_support_size].detach()
        error = ((xs - xs @ projector).square().sum() /
                 xs.square().sum().clamp_min(1e-12)).item()
        novel = error >= self.ic_threshold
        self.ic_stats["novelty_triggers"] += int(novel)
        return novel

    def _ic_topk(self, logits):
        # Recurrent LoRA has a separate k-of-H budget at each recurrent step.
        return self._topk_mask(logits) if hasattr(self, "_topk_mask") else kwta(logits, self.k)

    def reconfigure_changepoint(self, x, y):
        self._ic_archive()
        entry, _ = self._ic_match(x, include_live=False)
        use_prior = self.ic_train_prior and entry is not None
        self.ic_stats["train_hits" if use_prior else "train_misses"] += 1
        if not use_prior:
            return super().reconfigure_changepoint(x, y)

        saved = (self.bayes_changepoint_prior_decay,
                 self.bayes_changepoint_kl_weight,
                 self.bayes_changepoint_reconfig_steps)
        self.mask_logits.copy_(entry.logits)
        self.bayes_changepoint_prior_decay = 1.0
        self.bayes_changepoint_kl_weight = self.ic_prior_kl
        self.bayes_changepoint_reconfig_steps = self.ic_prior_steps
        try:
            return super().reconfigure_changepoint(x, y)
        finally:
            (self.bayes_changepoint_prior_decay,
             self.bayes_changepoint_kl_weight,
             self.bayes_changepoint_reconfig_steps) = saved

    def ic_repair_from_replay(self, replay):
        self.ic_blocks += 1
        if not self.ic_repair or self.ic_blocks % self.ic_repair_every_blocks:
            return
        live = self._ic_projector()
        domains = []
        for entry in reversed(self.ic_entries):
            if live is not None and torch.linalg.norm(entry.projector - live).item() < self.ic_same_domain_distance:
                continue
            if all(torch.linalg.norm(entry.projector - old.projector).item() >= self.ic_same_domain_distance
                   for old in domains):
                domains.append(entry)
        if not domains:
            return

        grouped = [[] for _ in domains]
        for xr, yr in replay.data:
            xs = xr[:self.ic_support_size]
            power = xs.square().sum().clamp_min(1e-12)
            errors = [((xs - xs @ e.projector).square().sum() / power).item()
                      for e in domains]
            closest = min(range(len(errors)), key=errors.__getitem__)
            if errors[closest] < self.ic_threshold:
                grouped[closest].append((xr, yr))

        saved = self.get_reconfig_state()
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        needed = self.ic_repair_fit_rows + self.ic_repair_validation_rows
        try:
            with torch.random.fork_rng(devices=devices):
                for entry, batches in zip(domains, grouped):
                    if sum(x.shape[0] for x, _ in batches) < needed:
                        continue
                    x = torch.cat([pair[0] for pair in batches])
                    y = torch.cat([pair[1] for pair in batches])
                    order = torch.randperm(x.shape[0], device=x.device)[:needed]
                    x, y = x[order], y[order]
                    n = self.ic_repair_fit_rows
                    xf, yf, xv, yv = x[:n], y[:n], x[n:], y[n:]
                    old_mask = self._ic_topk(entry.logits)
                    with torch.no_grad():
                        old_loss = self.loss(self(xv, m=old_mask), yv).item()
                    self.ic_repair_before.append(old_loss)
                    self.ic_stats["repair_attempts"] += 1

                    super().cold_reconfigure(
                        xf, yf, steps=self.ic_repair_steps,
                        lr=self.ic_repair_lr, sample_size=None,
                        temperature=self.ic_repair_temperature,
                        restarts=self.ic_repair_restarts,
                        take_best=self.ic_repair_take_best)
                    candidate = self.m.detach().clone()
                    with torch.no_grad():
                        candidate_loss = self.loss(self(xv, m=candidate), yv).item()
                    accepted = (candidate_loss + self.ic_repair_min_improvement
                                < old_loss)
                    if accepted:
                        magnitude = self.ic_repair_logit_magnitude
                        logits = torch.where(candidate > 0.5,
                                             torch.full_like(candidate, magnitude),
                                             torch.full_like(candidate, -magnitude))
                        self.ic_entries.append(RouteEntry(entry.projector.clone(),
                                               logits, entry.samples))
                        self.ic_stats["repair_accepted"] += 1
                    self.ic_repair_after.append(candidate_loss if accepted else old_loss)
        finally:
            self.set_reconfig_state(saved)

    def cold_reconfigure(self, x, y, *, steps=30, lr=0.3, sample_size=None,
                         temperature=1.0, restarts=1, take_best=None):
        entry, error = self._ic_match(x)
        if entry is None:
            self.ic_stats["eval_misses"] += 1
            return super().cold_reconfigure(x, y, steps=steps, lr=lr,
                sample_size=sample_size, temperature=temperature,
                restarts=restarts, take_best=take_best)
        self.ic_stats["eval_hits"] += 1
        if self.ic_eval_strategy == "direct":
            self.m.copy_(self._ic_topk(entry.logits))
            self.last_cold_reconfig_stats = {"routed": True, "direct": True,
                                             "reconstruction_error": error}
            return self.m
        saved = self.mask_logits.detach().clone()
        try:
            self.mask_logits.copy_(entry.logits)
            self._posterior_update(
                x, y, steps=self.ic_prior_steps, lr=lr,
                sample_size=sample_size, kl_weight=self.ic_prior_kl,
                temperature=temperature, prior_decay=1.0,
                stochastic_mask=False, changepoint=False)
        finally:
            self.mask_logits.copy_(saved)
        self.last_cold_reconfig_stats = {"routed": True, "direct": False,
                                         "reconstruction_error": error}
        return self.m
