"""Persistent BayesLoRA mask inference on a transformer's prompt NLL.

The support objective is the same bare prompt used by the old configurer.
Stable prompts make a short KL-regularized posterior update; detected changes
reset/refit from the configured prior. Every forward uses an exact-k mask.
"""
from __future__ import annotations

import math

import torch

from config import Config
from lora_bank import MaskController
from spatial import lateral_kwta


class ReactiveState:
    """EMA change detector on prompt NLL under the *preceding* hard mask."""

    def __init__(self, state=None):
        state = state or {}
        self.mean = state.get("mean")
        self.var = state.get("var", 0.0)
        self.since = state.get("since", 0)

    def sd(self):
        return max(self.var, 0.0) ** 0.5

    def triggered(self, value, threshold, warmup):
        return (self.mean is None or self.since >= warmup and
                value > self.mean + threshold * self.sd())

    def update(self, value, ema, changepoint):
        if changepoint or self.mean is None:
            self.mean, self.var, self.since = value, 0.0, 0
        else:
            diff = value - self.mean
            self.mean = ema * self.mean + (1 - ema) * value
            self.var = ema * self.var + (1 - ema) * diff * diff
            self.since += 1

    def to_dict(self):
        return {"mean": self.mean, "var": self.var, "since": self.since}


class BayesianPerplexityConfigurer:
    """One shared posterior and one detector for a shared transformer LoRA mask."""

    def __init__(self, cfg: Config, controller: MaskController, detector=None):
        self.cfg = cfg
        self.controller = controller
        self.mask_logits = torch.zeros(controller.H, device=controller.mask.device)
        self.detector = ReactiveState(detector)
        self.last_stats = {}

    def state_dict(self):
        return {"mask_logits": self.mask_logits.detach().cpu(),
                "mask": self.controller.mask.detach().cpu()}

    def load_state_dict(self, state):
        logits = state["mask_logits"].float().to(self.mask_logits.device)
        if logits.shape != self.mask_logits.shape:
            raise ValueError(f"posterior shape {logits.shape} != {self.mask_logits.shape}")
        self.mask_logits.copy_(logits)
        self.controller.set_mask(state["mask"].float())

    def _nll(self, model, ids, attn):
        return model(input_ids=ids, attention_mask=attn, labels=ids).loss

    @staticmethod
    def _kl(logits, prior, temperature):
        eps = torch.finfo(logits.dtype).eps
        q = torch.sigmoid(logits / temperature).clamp(eps, 1 - eps)
        p = torch.sigmoid(prior / temperature).clamp(eps, 1 - eps)
        return (q * (q.log() - p.log()) +
                (1 - q) * ((1 - q).log() - (1 - p).log())).mean()

    def _sample(self, logits, k, temperature, stochastic):
        if stochastic:
            u = torch.rand_like(logits).clamp_(1e-7, 1 - 1e-7)
            logits = logits / temperature - torch.log(-torch.log(u))
        return lateral_kwta(logits, k, self.cfg)

    def iterate(self, model, input_ids, attention_mask=None, *, k=None,
                reactive_sd=None, reactive_ema=None, steps=None, lr=None,
                temperature=None, kl_weight=None, prior_decay=None,
                changepoint_steps=None, changepoint_lr=None,
                changepoint_temperature=None, changepoint_kl_weight=None,
                changepoint_prior_decay=None, stochastic=True,
                routed_prior=None, routed_steps=None, routed_kl_weight=None):
        """Yield candidate masks; commit the posterior only on completion.

        Overrides are per call, so the GUI can adjust the search live. The
        caller must exhaust or close this generator before another model call.
        """
        cfg, ctrl = self.cfg, self.controller
        k = ctrl.k if k is None else int(k)
        if not 1 <= k <= ctrl.H:
            raise ValueError("k must be in [1, H]")
        sd = cfg.reactive_sd if reactive_sd is None else float(reactive_sd)
        ema = cfg.reactive_ema if reactive_ema is None else float(reactive_ema)
        if not 0 <= ema < 1:
            raise ValueError("reactive EMA must be in [0, 1)")
        if sd < 0:
            raise ValueError("reactive threshold must be nonnegative")
        stable_steps = cfg.bayes_reconfig_steps if steps is None else int(steps)
        stable_lr = cfg.bayes_reconfig_lr if lr is None else float(lr)
        stable_temp = cfg.bayes_temperature if temperature is None else float(temperature)
        stable_kl = cfg.bayes_kl_weight if kl_weight is None else float(kl_weight)
        stable_decay = cfg.bayes_prior_decay if prior_decay is None else float(prior_decay)
        cp_steps = cfg.bayes_changepoint_reconfig_steps if changepoint_steps is None else int(changepoint_steps)
        cp_lr = cfg.bayes_changepoint_reconfig_lr if changepoint_lr is None else float(changepoint_lr)
        cp_temp = cfg.bayes_changepoint_temperature if changepoint_temperature is None else float(changepoint_temperature)
        cp_kl = cfg.bayes_changepoint_kl_weight if changepoint_kl_weight is None else float(changepoint_kl_weight)
        cp_decay = cfg.bayes_changepoint_prior_decay if changepoint_prior_decay is None else float(changepoint_prior_decay)
        if min(stable_steps, cp_steps) < 1 or min(stable_lr, cp_lr, stable_temp, cp_temp) <= 0:
            raise ValueError("steps, learning rates, and temperatures must be positive")
        if min(stable_kl, cp_kl) < 0 or not 0 <= stable_decay <= 1 or not 0 <= cp_decay <= 1:
            raise ValueError("KL must be nonnegative and prior decay in [0, 1]")
        if routed_prior is not None:
            routed_steps = (cfg.input_route_refine_steps if routed_steps is None
                            else int(routed_steps))
            routed_kl_weight = (cfg.input_route_refine_kl if routed_kl_weight is None
                                else float(routed_kl_weight))
            if routed_prior.shape != self.mask_logits.shape:
                raise ValueError("retrieved prior shape differs from mask logits")
            if routed_steps < 1 or routed_kl_weight < 0:
                raise ValueError("route refinement steps must be positive and KL nonnegative")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if input_ids.shape[-1] < 2:
            raise ValueError("prompt needs at least two tokens for NLL")

        old_mask, old_k = ctrl.mask.detach().clone(), ctrl.k
        saved = [(p, p.requires_grad) for p in model.parameters()]
        ckpt_on = getattr(model, "is_gradient_checkpointing", False)
        committed = False
        try:
            for p, _ in saved:
                p.requires_grad_(False)
            if ckpt_on:
                model.gradient_checkpointing_disable()
            ctrl.k = k
            if routed_prior is None:
                # Observe the incoming prompt under the previous mask. On a k
                # change or an empty mask the old observation is not comparable.
                with torch.no_grad():
                    probe = float(self._nll(model, input_ids, attention_mask)) if (
                        old_k == k and old_mask.sum().item() > 0
                    ) else float("nan")
                warmup = math.ceil(1.0 / max(1.0 - ema, 1e-8))
                cp = (old_k != k or old_mask.sum().item() == 0 or
                      self.detector.triggered(probe, sd, warmup))
                nsteps, rate, temp, weight, decay = (
                    (cp_steps, cp_lr, cp_temp, cp_kl, cp_decay) if cp else
                    (stable_steps, stable_lr, stable_temp, stable_kl, stable_decay)
                )
                prior = (decay * self.mask_logits).detach().clone()
            else:
                # A matched archive entry is the prior for this prompt. Bypass
                # change detection and preserve it even when CP decay is zero.
                probe, cp = float("nan"), True
                nsteps, rate, temp, weight = (routed_steps, stable_lr,
                                               stable_temp, routed_kl_weight)
                prior = routed_prior.detach().to(
                    device=self.mask_logits.device,
                    dtype=self.mask_logits.dtype).clone()
                stochastic = False
            logits = prior.clone().requires_grad_(True)
            opt = torch.optim.Adam([logits], lr=rate)
            task_value = kl_value = float("nan")
            with torch.enable_grad():
                for i in range(nsteps):
                    soft = torch.sigmoid(logits / temp)
                    anneal = math.exp(-i / max((nsteps - 1) / 3, 10))
                    hard = self._sample(logits, k, temp * anneal, stochastic)
                    ctrl.set_mask(hard + soft - soft.detach())
                    task = self._nll(model, input_ids, attention_mask)
                    kl = self._kl(logits, prior, temp)
                    objective = task + weight * kl
                    opt.zero_grad()
                    objective.backward()
                    opt.step()
                    task_value, kl_value = float(task.detach()), float(kl.detach())
                    yield hard.detach()
            final = self._sample(logits.detach(), k, temp, stochastic)
            ctrl.set_mask(final)
            # A new episode's baseline is measured with its new mask. Stable
            # updates likewise track post-update NLL, avoiding false spikes.
            with torch.no_grad():
                fitted = float(self._nll(model, input_ids, attention_mask))
            self.mask_logits.copy_(logits.detach())
            self.detector.update(fitted, ema, cp)
            self.last_stats = {"changepoint": cp, "probe_nll": probe,
                               "nll": fitted, "search_nll": task_value,
                               "kl": kl_value, "steps": nsteps,
                               "routed_refine": routed_prior is not None}
            committed = True
            yield final.detach()
        finally:
            if not committed:
                ctrl.k = old_k
                ctrl.set_mask(old_mask)
            for p, rg in saved:
                p.requires_grad_(rg)
            if ckpt_on:
                model.gradient_checkpointing_enable()

    def __call__(self, model, input_ids, attention_mask=None, record=False, **kwargs):
        frames, last = [], None
        for mask in self.iterate(model, input_ids, attention_mask, **kwargs):
            last = mask
            if record:
                frames.append(mask.clone())
        return last, self.last_stats["nll"], frames
