"""
bayes_reconfig.py - Persistent approximate-posterior mask reconfiguration,
factored out of the bayesian subproject's BayesFTN (ftn.BayesFTN) so the
FTN-LoRA and recurrent-residual FTN backbones can reuse the *exact* same
variational mechanism instead of the plain per-batch FTN search.

A backbone that mixes this in (BayesLoRAFTN, BayesRNNResidualFTN) gets:

  - a persistent ``mask_logits`` buffer - the variational state carried across
    batches (a buffer, never a Parameter: the weight optimizer must not touch
    it; reconfiguration updates it explicitly);
  - a cheap posterior update for ordinary stable batches (``reconfigure``) that
    starts from a detached copy of the preceding logits (the prior), takes a few
    support-loss steps, and penalizes movement with a factorized-Bernoulli KL;
  - a reset/refit update for detected change points (``reconfigure_changepoint``)
    under a separate, boundary-free parameter set (typically prior_decay=0,
    kl_weight=0 - a fresh fit that then becomes the new persistent posterior);
  - Gumbel top-k (exact-k) mask sampling with a sigmoid straight-through
    surrogate (the forward pass is always a hard k-of-H mask; the sigmoid is
    only a gradient path);
  - the architecture-independent stateless zero-logit evaluation search
    (``cold_reconfigure``), shared unchanged with BayesFTN's semantics.

The Bernoulli KL is a tractable proxy: conditioning independent Bernoulli
variables on exactly k active entries has an expensive normalizer, while Gumbel
top-k enforces the budget exactly in every actual forward pass.

Backbone-specific pieces the subclass MUST provide
--------------------------------------------------
    _topk_mask(scores) -> hard exact-k mask (a flat (hidden_size,) 0/1 tensor)
        A single kWTA over a flat atom bank (LoRA), or a *per-layer* kWTA over
        each square_dim**2 block of a stacked/recurrent mask. Called on raw
        logits, Gumbel-perturbed logits, and annealed noisy logits alike.

    _run(x, mask, detach) -> predictions under a (differentiable) flat mask
        Runs the full backbone. detach=True detaches every trainable weight so
        the search leaves gradient only on the mask logits (used by every
        reconfiguration); detach=False is the ordinary trainable pass.

plus the attributes ``self.k``, ``self.hidden_size``, ``self.loss`` and the
search knobs (``reconfig_steps`` / ``reconfig_sample_size`` /
``reconfig_noise_scale``) that _init_bayes leaves in place.
"""

import math

import torch
import torch.nn as nn
from overlap_regularizer import OverlapRegularizer


class BayesMaskReconfig(OverlapRegularizer):
    """Mixin implementing BayesFTN's persistent-posterior mask inference.

    Mix in *before* nn.Module in the MRO. The subclass registers ``self.m``
    (the hard task mask) and sets ``self.hidden_size`` / ``self.k`` and the
    search knobs, then calls ``self._init_bayes(**bayes_pars)`` from its own
    __init__ (after nn.Module.__init__).
    """

    def _init_bayes(
        self,
        *,
        bayes_reconfig_steps: int = 5,
        bayes_reconfig_lr: float = 0.3,
        bayes_kl_weight: float = 1.0,
        bayes_temperature: float = 1.0,
        bayes_prior_decay: float = 1.0,
        bayes_stochastic_mask: bool = True,
        bayes_reconfig_take_best: bool = False,
        bayes_changepoint_reconfig_steps: int = 30,
        bayes_changepoint_reconfig_lr: float = 0.3,
        bayes_changepoint_reconfig_sample_size=None,
        bayes_changepoint_kl_weight: float = 0.0,
        bayes_changepoint_temperature: float = 1.0,
        bayes_changepoint_prior_decay: float = 0.0,
        bayes_changepoint_stochastic_mask: bool = True,
        sample_type: str = "gumbel",
        overlap_mode: str = "bump",
        overlap_lambda: float = 0.0,
        overlap_threshold: float = 0.5,
        overlap_temperature: float = 0.2,
    ):
        """Validate and store the variational hyperparameters and register the
        persistent ``mask_logits`` buffer. Mirrors BayesFTN.__init__."""
        self.configure_overlap(overlap_mode, overlap_lambda, overlap_threshold, overlap_temperature)
        if bayes_reconfig_steps < 1:
            raise ValueError("bayes_reconfig_steps must be at least 1")
        if bayes_temperature <= 0:
            raise ValueError("bayes_temperature must be positive")
        if bayes_kl_weight < 0:
            raise ValueError("bayes_kl_weight must be non-negative")
        if not 0 <= bayes_prior_decay <= 1:
            raise ValueError("bayes_prior_decay must be in [0, 1]")
        if bayes_changepoint_reconfig_steps < 1:
            raise ValueError("bayes_changepoint_reconfig_steps must be at least 1")
        if bayes_changepoint_reconfig_lr <= 0:
            raise ValueError("bayes_changepoint_reconfig_lr must be positive")
        if (
            bayes_changepoint_reconfig_sample_size is not None
            and bayes_changepoint_reconfig_sample_size < 1
        ):
            raise ValueError(
                "bayes_changepoint_reconfig_sample_size must be positive or None"
            )
        if bayes_changepoint_kl_weight < 0:
            raise ValueError("bayes_changepoint_kl_weight must be non-negative")
        if bayes_changepoint_temperature <= 0:
            raise ValueError("bayes_changepoint_temperature must be positive")
        if not 0 <= bayes_changepoint_prior_decay <= 1:
            raise ValueError("bayes_changepoint_prior_decay must be in [0, 1]")
        if sample_type not in ("gumbel", "annealste"):
            raise ValueError("sample_type must be 'gumbel' or 'annealste'")

        self.bayes_reconfig_steps = bayes_reconfig_steps
        self.bayes_reconfig_lr = bayes_reconfig_lr
        self.bayes_kl_weight = bayes_kl_weight
        self.bayes_temperature = bayes_temperature
        self.bayes_prior_decay = bayes_prior_decay
        self.bayes_stochastic_mask = bayes_stochastic_mask
        self.bayes_reconfig_take_best = bayes_reconfig_take_best
        self.bayes_changepoint_reconfig_steps = bayes_changepoint_reconfig_steps
        self.bayes_changepoint_reconfig_lr = bayes_changepoint_reconfig_lr
        self.bayes_changepoint_reconfig_sample_size = (
            bayes_changepoint_reconfig_sample_size
        )
        self.bayes_changepoint_kl_weight = bayes_changepoint_kl_weight
        self.bayes_changepoint_temperature = bayes_changepoint_temperature
        self.bayes_changepoint_prior_decay = bayes_changepoint_prior_decay
        self.bayes_changepoint_stochastic_mask = bayes_changepoint_stochastic_mask

        # A buffer, not a Parameter: the weight optimizer must never update the
        # variational state. Reconfiguration updates it explicitly.
        self.register_buffer("mask_logits", torch.zeros(self.hidden_size))
        self._reconfig = self.reconfigure
        self.last_reconfig_stats = {}
        self.sample_type = sample_type

    # ------------------------------------------------------------------ hooks
    # Subclasses MUST override these two; everything else is backbone-agnostic.

    def _topk_mask(self, scores: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _run(self, x: torch.Tensor, mask: torch.Tensor, detach: bool) -> torch.Tensor:
        raise NotImplementedError

    # ---------------------------------------------------------- reactive / state

    @property
    def continuous_reactive_updates(self):
        """Stable batches keep making cheap posterior updates in reactive mode."""
        return True

    def get_reconfig_state(self):
        """Clone mutable mask-inference state for side-effect-free evaluation."""
        return {"m": self.m.clone(), "mask_logits": self.mask_logits.clone()}

    def set_reconfig_state(self, state):
        """Restore a state returned by get_reconfig_state()."""
        self.m.copy_(state["m"])
        self.mask_logits.copy_(state["mask_logits"])

    # -------------------------------------------------------------- sampling / KL

    def _sample_mask(self, logits, temperature, stochastic):
        """Sample exactly k entries (per block) from the logit-induced ranking.

        stochastic=True perturbs the logits with Gumbel noise before the top-k
        (an exact-k sample); False is the deterministic top-k (an ablation).
        """
        if stochastic:
            u = torch.rand_like(logits).clamp_(1e-7, 1 - 1e-7)
            return self._topk_mask(logits / temperature - torch.log(-torch.log(u)))
        return self._topk_mask(logits)

    def _kl_to_prior(self, logits, prior_logits, temperature):
        """Mean factorized-Bernoulli KL(q_logits || q_prior)."""
        eps = torch.finfo(logits.dtype).eps
        q = torch.sigmoid(logits / temperature).clamp(eps, 1 - eps)
        p = torch.sigmoid(prior_logits / temperature).clamp(eps, 1 - eps)
        kl = q * (q.log() - p.log())
        kl = kl + (1 - q) * ((1 - q).log() - (1 - p).log())
        return kl.mean()

    # ----------------------------------------------------------- posterior update

    @torch.enable_grad()
    def _posterior_update(
        self,
        x,
        y,
        *,
        steps,
        lr,
        sample_size,
        kl_weight,
        temperature,
        prior_decay,
        stochastic_mask,
        changepoint,
    ):
        """Run one persistent posterior update under an explicit parameter set.

        The prior is a detached copy of ``prior_decay * mask_logits`` (rho=1 is
        strict persistence; rho=0 diffuses toward zero logits - the change-point
        reset). Every candidate mask is scored through the detached backbone, so
        gradients reach only the mask logits. The final persistent logits are
        installed and a hard mask is sampled into ``self.m``.
        """
        n = x.shape[0] if sample_size is None else min(sample_size, x.shape[0])
        x, y = x[:n], y[:n]

        prior_logits = (prior_decay * self.mask_logits).detach().clone()
        reference = self._overlap_reference()
        logits = nn.Parameter(prior_logits.clone())
        opt = torch.optim.Adam([logits], lr=lr)

        task_value = kl_value = objective_value = float("nan")
        best_loss, best_mask, best_logits = float("inf"), None, None
        best_kl = best_objective = float("nan")
        best_selection = float("inf")
        overlap_value = best_overlap = 0.0
        for i in range(steps):
            if self.sample_type == "annealste":
                sm = torch.sigmoid(logits)
                tau = logits.detach().std() * math.exp(-i / (self.reconfig_steps / 4))
                noise = (
                    temperature
                    * torch.randn_like(logits)
                    * tau
                    * self.reconfig_noise_scale
                )
                mask = self._topk_mask(logits + noise) + sm - sm.detach()
            else:
                soft = torch.sigmoid(logits / temperature)
                anneal = math.exp(-i / max((steps - 1) / 3, 10))
                hard = self._sample_mask(logits, temperature * anneal, stochastic_mask)
                mask = hard + soft - soft.detach()

            pred = self._run(x, mask, detach=True)
            task = self.loss(pred, y)
            kl = self._kl_to_prior(logits, prior_logits, temperature)
            objective = task + kl_weight * kl
            penalty = self._overlap_penalty(mask, reference)
            objective = self._with_overlap(objective, mask, reference)
            selection = self._with_overlap(task, mask, reference)

            if self.bayes_reconfig_take_best and selection.item() < best_selection:
                best_selection = selection.item()
                best_overlap = penalty.detach().item()
                best_loss = task.item()
                best_mask = mask.detach().clone()
                best_logits = logits.detach().clone()
                best_kl = kl.item()
                best_objective = objective.item()

            opt.zero_grad()
            objective.backward()
            opt.step()
            task_value = task.detach().item()
            kl_value = kl.detach().item()
            objective_value = objective.detach().item()
            overlap_value = penalty.detach().item()

        if self.bayes_reconfig_take_best:
            self.mask_logits.copy_(best_logits)
            self.m.copy_(best_mask)
            task_value = best_loss
            kl_value = best_kl
            objective_value = best_objective
            overlap_value = best_overlap
        else:
            self.mask_logits.copy_(logits.detach())
            self.m.copy_(
                self._sample_mask(self.mask_logits, temperature, stochastic_mask)
            )
        self.last_reconfig_stats = {
            "changepoint": changepoint,
            "task_loss": task_value,
            "kl": kl_value,
            "objective": objective_value,
            "overlap_penalty": overlap_value,
            "prior_decay": prior_decay,
            "steps": steps,
        }
        return self.m

    def reconfigure(self, x, y):
        """Cheap persistent posterior update for an ordinary stable batch."""
        return self._posterior_update(
            x,
            y,
            steps=self.bayes_reconfig_steps,
            lr=self.bayes_reconfig_lr,
            sample_size=self.reconfig_sample_size,
            kl_weight=self.bayes_kl_weight,
            temperature=self.bayes_temperature,
            prior_decay=self.bayes_prior_decay,
            stochastic_mask=self.bayes_stochastic_mask,
            changepoint=False,
        )

    def reconfigure_changepoint(self, x, y):
        """Reset/refit the posterior using change-point-specific parameters."""
        return self._posterior_update(
            x,
            y,
            steps=self.bayes_changepoint_reconfig_steps,
            lr=self.bayes_changepoint_reconfig_lr,
            sample_size=self.bayes_changepoint_reconfig_sample_size,
            kl_weight=self.bayes_changepoint_kl_weight,
            temperature=self.bayes_changepoint_temperature,
            prior_decay=self.bayes_changepoint_prior_decay,
            stochastic_mask=self.bayes_changepoint_stochastic_mask,
            changepoint=True,
        )

    # ------------------------------------------------------- stateless evaluation

    @torch.enable_grad()
    def cold_reconfigure(
        self,
        x,
        y,
        steps: int = 30,
        lr: float = 0.3,
        sample_size=None,
        temperature: float = 1.0,
        restarts: int = 1,
        take_best: bool | None = None,
    ):
        """Stateless support-conditioned mask inference for evaluation.

        Every restart creates zero logits and a fresh optimizer, uses no KL or
        stored posterior state, and returns the mask with the lowest support
        loss. With take_best, sampled masks compete with the final deterministic
        top-k mask. Never reads or writes ``self.mask_logits``.
        """
        if steps < 1:
            raise ValueError("evaluation reconfiguration steps must be at least 1")
        if lr <= 0:
            raise ValueError("evaluation reconfiguration lr must be positive")
        if temperature <= 0:
            raise ValueError("evaluation temperature must be positive")
        if restarts < 1:
            raise ValueError("evaluation restarts must be at least 1")
        if sample_size is not None and sample_size < 1:
            raise ValueError("evaluation sample_size must be positive or None")
        if take_best is None:
            take_best = self.bayes_reconfig_take_best

        n = x.shape[0] if sample_size is None else min(sample_size, x.shape[0])
        x, y = x[:n], y[:n]

        best_mask, best_loss = None, float("inf")
        for _ in range(restarts):
            logits = nn.Parameter(torch.zeros_like(self.mask_logits))
            opt = torch.optim.Adam([logits], lr=lr)
            for i in range(steps):
                soft = torch.sigmoid(logits / temperature)
                anneal = math.exp(-i / max((steps - 1) / 3, 10))
                hard = self._sample_mask(
                    logits, temperature * anneal, stochastic=True
                )
                mask = hard + soft - soft.detach()
                loss = self.loss(self._run(x, mask, detach=True), y)
                if take_best and loss.item() < best_loss:
                    best_mask = hard.detach().clone()
                    best_loss = loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()

            candidate = self._topk_mask(logits.detach())
            with torch.no_grad():
                candidate_loss = self.loss(
                    self._run(x, candidate, detach=True), y
                ).item()
            if candidate_loss < best_loss:
                best_mask = candidate.clone()
                best_loss = candidate_loss

        self.m.copy_(best_mask)
        self.last_cold_reconfig_stats = {
            "support_loss": best_loss,
            "restarts": restarts,
        }
        return self.m
