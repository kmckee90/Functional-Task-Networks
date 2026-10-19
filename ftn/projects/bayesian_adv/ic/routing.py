"""Append-only input reconstruction memory paired with Bayesian mask logits.

Each observed domain is represented by the optimal rank-r linear autoencoder
(the leading PCA subspace of its input covariance). Its reconstruction error
selects a previously discovered mask. Old entries are never trained or edited.
This deliberately tests the *easy x -> x task as a prior* without introducing
a shared router that could itself forget.
"""

from dataclasses import dataclass

import torch

from bayes_lora_ftn import BayesLoRAFTN as _BayesLoRA
from bayes_lora_rnn_ftn import BayesLoRARNNFTN as _BayesLoRARNN


@dataclass
class Entry:
    projector: torch.Tensor
    logits: torch.Tensor
    samples: int
    support_x: torch.Tensor | None = None
    support_y: torch.Tensor | None = None


class ReconstructionPrior:
    def configure_ic(self, config):
        self.ic_mode = config.IC_MODE
        self.ic_train_prior = config.IC_TRAIN_PRIOR
        self.ic_eval_prior = config.IC_EVAL_PRIOR
        self.ic_support_size = config.IC_SUPPORT_SIZE
        self.ic_rank = config.IC_AE_RANK
        self.ic_threshold = config.IC_RECON_THRESHOLD
        self.ic_trigger = config.IC_TRIGGER
        self.ic_prior_steps = config.IC_PRIOR_STEPS
        self.ic_prior_kl = config.IC_PRIOR_KL
        self.ic_eval_strategy = config.IC_EVAL_STRATEGY
        self.ic_refresh = config.IC_REFRESH
        self.ic_refresh_rows = config.IC_REFRESH_ROWS
        self.ic_refresh_steps = config.IC_REFRESH_STEPS
        self.ic_refresh_kl = config.IC_REFRESH_KL
        self.ic_refresh_method = config.IC_REFRESH_METHOD
        self.ic_replay_repair = config.IC_REPLAY_REPAIR
        self.ic_repair_fit_rows = config.IC_REPAIR_FIT_ROWS
        self.ic_repair_steps = config.IC_REPAIR_STEPS
        self.ic_repair_restarts = config.IC_REPAIR_RESTARTS
        self.ic_entries = []
        self.ic_cov = None
        self.ic_samples = 0
        self.ic_support_x = None
        self.ic_support_y = None
        self.ic_stats = {"train_hits": 0, "train_misses": 0,
                         "eval_hits": 0, "eval_misses": 0,
                         "novelty_triggers": 0}
        self.ic_cp_holdout_acc = []
        self.ic_cp_holdout_loss = []
        self.ic_refresh_before = []
        self.ic_refresh_after = []
        self.ic_repair_before = []
        self.ic_repair_after = []
        self.ic_repair_accepted = 0

    def _ic_observe(self, x, y=None):
        if not hasattr(self, "ic_mode"):
            return
        # Use unlabeled training inputs only. Full batches stabilize the
        # reconstruction model; routing itself always sees just support rows.
        with torch.no_grad():
            x = x.detach()
            scatter = x.T @ x
            self.ic_cov = scatter if self.ic_cov is None else self.ic_cov + scatter
            self.ic_samples += x.shape[0]
            if y is not None:
                n_saved = 0 if self.ic_support_x is None else self.ic_support_x.shape[0]
                n_new = min(self.ic_refresh_rows - n_saved, x.shape[0])
                if n_new > 0:
                    xx, yy = x[:n_new].clone(), y[:n_new].detach().clone()
                    self.ic_support_x = (xx if self.ic_support_x is None else
                                         torch.cat((self.ic_support_x, xx)))
                    self.ic_support_y = (yy if self.ic_support_y is None else
                                         torch.cat((self.ic_support_y, yy)))

    def _ic_projector(self):
        if self.ic_cov is None or self.ic_samples < self.ic_rank:
            return None
        _, vecs = torch.linalg.eigh(self.ic_cov)
        basis = vecs[:, -self.ic_rank:]
        return basis @ basis.T

    def _ic_archive(self):
        projector = self._ic_projector()
        if projector is not None:
            self.ic_entries.append(Entry(projector.detach().clone(),
                                         self.mask_logits.detach().clone(),
                                         self.ic_samples,
                                         self.ic_support_x, self.ic_support_y))
        self.ic_cov = None
        self.ic_samples = 0
        self.ic_support_x = None
        self.ic_support_y = None

    def _ic_candidates(self, include_live=True):
        entries = self.ic_entries.copy()
        if include_live:
            projector = self._ic_projector()
            if projector is not None:
                entries.append(Entry(projector, self.mask_logits, self.ic_samples))
        return entries

    def _ic_match(self, x, include_live=True):
        entries = self._ic_candidates(include_live)
        if not entries:
            return None, float("inf")
        x = x[:self.ic_support_size].detach()
        power = x.square().sum().clamp_min(1e-12)
        errors = [(((x - x @ entry.projector).square().sum() / power).item(), entry)
                  for entry in entries]
        best_error = min(error for error, _ in errors)
        # PCA reconstruction is effectively exact on these rank-two domains.
        # Prefer a recent mask when several archived entries fit equally well.
        error, entry = next((error, entry) for error, entry in reversed(errors)
                            if error <= best_error + 1e-3)
        return (entry if error < self.ic_threshold else None), error

    def ic_novelty(self, x):
        if not getattr(self, "ic_trigger", False):
            return False
        # The current segment determines change detection. An archived domain
        # can still be recognized as the prior after a change is detected.
        if self._ic_projector() is None:
            return False
        current = self._ic_candidates(include_live=True)[-1]
        xs = x[:self.ic_support_size].detach()
        power = xs.square().sum().clamp_min(1e-12)
        current_error = ((xs - xs @ current.projector).square().sum() / power).item()
        novel = current_error >= self.ic_threshold
        if novel:
            self.ic_stats["novelty_triggers"] += 1
        return novel

    def reconfigure(self, x, y):
        result = super().reconfigure(x, y)
        self._ic_observe(x, y)
        return result

    def reconfigure_changepoint(self, x, y):
        if not hasattr(self, "ic_mode"):
            return super().reconfigure_changepoint(x, y)
        self._ic_archive()
        entry, _ = self._ic_match(x, include_live=False)
        routed = (self.ic_mode == "reconstruction" and self.ic_train_prior
                  and entry is not None)
        self.ic_stats["train_hits" if routed else "train_misses"] += 1
        if routed:
            self.mask_logits.copy_(entry.logits)
            old_decay = self.bayes_changepoint_prior_decay
            old_kl = self.bayes_changepoint_kl_weight
            old_steps = self.bayes_changepoint_reconfig_steps
            self.bayes_changepoint_prior_decay = 1.0
            self.bayes_changepoint_kl_weight = self.ic_prior_kl
            self.bayes_changepoint_reconfig_steps = self.ic_prior_steps
            try:
                result = super().reconfigure_changepoint(x, y)
            finally:
                self.bayes_changepoint_prior_decay = old_decay
                self.bayes_changepoint_kl_weight = old_kl
                self.bayes_changepoint_reconfig_steps = old_steps
        else:
            result = super().reconfigure_changepoint(x, y)
        self._ic_observe(x, y)
        return result

    def ic_refresh_memory(self):
        """Append updated mask values for old domains using their train cache.

        The reconstruction projectors and prior entries stay immutable. This
        maintenance never routes the current training batch or changes model
        weights; it only tracks their drift with cached historical labels.
        """
        if not self.ic_refresh:
            return
        recent = []
        for entry in reversed(self.ic_entries):
            if entry.support_x is None:
                continue
            if all(torch.linalg.norm(entry.projector - other.projector).item() > 0.05
                   for other in recent):
                recent.append(entry)
        saved = self.get_reconfig_state()
        try:
            for entry in recent:
                with torch.no_grad():
                    before_mask = self._topk_mask(entry.logits)
                    before = self.loss(
                        self._run(entry.support_x, before_mask, detach=True),
                        entry.support_y).item()
                self.mask_logits.copy_(entry.logits)
                if self.ic_refresh_method == "cold":
                    devices = ([torch.cuda.current_device()]
                               if torch.cuda.is_available() else [])
                    with torch.random.fork_rng(devices=devices):
                        super().cold_reconfigure(
                            entry.support_x, entry.support_y,
                            steps=self.ic_refresh_steps,
                            lr=self.bayes_changepoint_reconfig_lr,
                            sample_size=None, temperature=1.0,
                            restarts=2, take_best=True)
                    self.mask_logits.copy_(torch.where(
                        self.m > 0.5, torch.full_like(self.m, 4.0),
                        torch.full_like(self.m, -4.0)))
                else:
                    self._posterior_update(
                        entry.support_x, entry.support_y,
                        steps=self.ic_refresh_steps,
                        lr=self.bayes_changepoint_reconfig_lr, sample_size=None,
                        kl_weight=self.ic_refresh_kl, temperature=1.0,
                        prior_decay=1.0, stochastic_mask=False, changepoint=False)
                with torch.no_grad():
                    after_mask = self._topk_mask(self.mask_logits)
                    after = self.loss(
                        self._run(entry.support_x, after_mask, detach=True),
                        entry.support_y).item()
                self.ic_entries.append(Entry(entry.projector.clone(),
                                             self.mask_logits.detach().clone(),
                                             entry.samples, entry.support_x,
                                             entry.support_y))
                self.ic_refresh_before.append(before)
                self.ic_refresh_after.append(after)
        finally:
            self.set_reconfig_state(saved)

    def ic_repair_from_replay(self, replay):
        """Refresh old mask values from replay grouped by immutable input keys.

        No task identifier is used to assign replay. A fit/validation split
        selects between the old mask and a fresh full-support mask search.
        Only the mask value is appended; old entries and projectors stay frozen.
        """
        if not self.ic_replay_repair:
            return
        live_projector = self._ic_projector()
        domains = []
        for entry in reversed(self.ic_entries):
            if live_projector is not None and torch.linalg.norm(
                    entry.projector - live_projector).item() < 0.05:
                continue
            if all(torch.linalg.norm(entry.projector - prev.projector).item() > 0.05
                   for prev in domains):
                domains.append(entry)
        if not domains:
            return

        grouped = [[] for _ in domains]
        for xr, yr in replay.data:
            xs = xr[:min(self.ic_support_size, xr.shape[0])]
            power = xs.square().sum().clamp_min(1e-12)
            errors = [((xs - xs @ entry.projector).square().sum() / power).item()
                      for entry in domains]
            closest = min(range(len(errors)), key=errors.__getitem__)
            if errors[closest] < self.ic_threshold:
                grouped[closest].append((xr, yr))

        saved = self.get_reconfig_state()
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=devices):
                for entry, batches in zip(domains, grouped):
                    if not batches:
                        continue
                    x = torch.cat([batch[0] for batch in batches])
                    y = torch.cat([batch[1] for batch in batches])
                    if x.shape[0] < 2 * self.ic_repair_fit_rows:
                        continue
                    order = torch.randperm(x.shape[0], device=x.device)
                    x, y = x[order], y[order]
                    xf, yf = x[:self.ic_repair_fit_rows], y[:self.ic_repair_fit_rows]
                    xv, yv = x[self.ic_repair_fit_rows:], y[self.ic_repair_fit_rows:]

                    old_mask = self._topk_mask(entry.logits)
                    with torch.no_grad():
                        old_loss = self.loss(self._run(xv, old_mask, detach=True), yv).item()
                    self.ic_repair_before.append(old_loss)

                    super().cold_reconfigure(
                        xf, yf, steps=self.ic_repair_steps,
                        lr=self.bayes_changepoint_reconfig_lr, sample_size=None,
                        temperature=1.0, restarts=self.ic_repair_restarts,
                        take_best=True)
                    candidate = self.m.detach().clone()
                    with torch.no_grad():
                        candidate_loss = self.loss(
                            self._run(xv, candidate, detach=True), yv).item()
                    if candidate_loss < old_loss:
                        logits = torch.where(candidate > 0.5,
                                             torch.full_like(candidate, 4.0),
                                             torch.full_like(candidate, -4.0))
                        self.ic_entries.append(Entry(
                            entry.projector.clone(), logits, entry.samples,
                            entry.support_x, entry.support_y))
                        self.ic_repair_accepted += 1
                    self.ic_repair_after.append(min(old_loss, candidate_loss))
        finally:
            self.set_reconfig_state(saved)

    def cold_reconfigure(self, x, y, *, steps=30, lr=0.3, sample_size=None,
                         temperature=1.0, restarts=1, take_best=None):
        if (not hasattr(self, "ic_mode") or self.ic_mode != "reconstruction"
                or not self.ic_eval_prior):
            return super().cold_reconfigure(x, y, steps=steps, lr=lr,
                sample_size=sample_size, temperature=temperature,
                restarts=restarts, take_best=take_best)
        entry, error = self._ic_match(x)
        if entry is None:
            self.ic_stats["eval_misses"] += 1
            return super().cold_reconfigure(x, y, steps=steps, lr=lr,
                sample_size=sample_size, temperature=temperature,
                restarts=restarts, take_best=take_best)
        self.ic_stats["eval_hits"] += 1
        if self.ic_eval_strategy == "direct":
            self.m.copy_(self._topk_mask(entry.logits))
            self.last_cold_reconfig_stats = {"reconstruction_error": error,
                                             "routed": True, "direct": True}
            return self.m
        saved = self.mask_logits.detach().clone()
        try:
            self.mask_logits.copy_(entry.logits)
            self._posterior_update(x, y, steps=self.ic_prior_steps, lr=lr,
                sample_size=sample_size, kl_weight=self.ic_prior_kl,
                temperature=temperature, prior_decay=1.0,
                stochastic_mask=False, changepoint=False)
        finally:
            self.mask_logits.copy_(saved)
        self.last_cold_reconfig_stats = {"reconstruction_error": error,
                                         "routed": True}
        return self.m


class BayesLoRAFTN(ReconstructionPrior, _BayesLoRA):
    pass


class BayesLoRARNNFTN(ReconstructionPrior, _BayesLoRARNN):
    pass
