"""
masks.py — KWTA-only mask configurers for func_net_kt (knowledge transfer).

This project compares three optimisation strategies for selecting a binary
top-k mask over hidden units, using the same NeuroModel backbone as the
parent func_net_spatial project but **without** lateral spatial dynamics.
The mask is the only adaptive variable during reconfiguration; all neurons
in the mask see large amounts of optimisation budget at low LR so the mask
search itself can serve as a knowledge-transfer mechanism.

A mask configurer is a callable that takes (model, data_batch, prev_mask)
and returns a binary mask (top-k) over the hidden units.

Three configurers:

  * KWTA_Sigmoid     -- continuous relaxation: optimise sigmoid(m) directly,
                        return KWTA(m) at the end.  Simplest baseline.
  * KWTA_AnnealedSTE -- straight-through KWTA with simulated annealing of
                        Gaussian noise on the logits, taken from
                        func_net_kwta/masks.py::KWTA_dyn.  Each step
                        evaluates a stochastic *binary* mask and keeps the
                        best hard mask seen.
  * KWTA_Gumbel      -- Gumbel-Sigmoid (Binary Concrete) relaxation with
                        annealed temperature.  A quadratic budget penalty
                        ``(sum sigmoid(m) - k)**2`` keeps the expected
                        active count near k; the final return is the hard
                        top-k mask of the learned logits.

All three accept the same kwargs:
    hidden_size, classification, k, reconfig_steps, reconfig_lr,
    reconfig_n  (max samples used during reconfig).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return binary mask with 1 at the top-k positions of x."""
    topk = torch.topk(x, k).indices
    out = torch.zeros_like(x)
    out[topk] = 1.0
    return out


# ---------------------------------------------------------------------------
# Common base
# ---------------------------------------------------------------------------


class _MaskConfigBase:
    """Shared bookkeeping for all three KWTA-only mask configurers."""

    name: str = "base"

    def __init__(
        self,
        hidden_size: int,
        classification: bool = True,
        k: int = 128,
        reconfig_steps: int = 50,
        reconfig_lr: float = 0.05,
        reconfig_n: int = 2048,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.classification = classification
        self.k = k
        self.reconfig_steps = max(1, int(reconfig_steps))
        self.reconfig_lr = reconfig_lr
        self.reconfig_n = reconfig_n
        # Tracks the loss obtained by the *first* mask the configurer
        # commits to on the most recent call, and the final lowest loss
        # the configurer ended with.  Used by the trainer to log the
        # "head start" obtained from mask optimisation prior to any
        # neuron training on the new task.
        self.last_initial_loss: float | None = None
        self.last_final_loss: float | None = None

    def __call__(
        self,
        model: nn.Module,
        data: tuple,
        m: torch.Tensor,
        task_idx: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _loss(
        self,
        model: nn.Module,
        z: torch.Tensor,
        mask: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        n: int,
    ) -> torch.Tensor:
        if hasattr(model, "linear"):
            y_pred = model.linear(z * mask.reshape(1, -1))
        else:
            y_pred = model(x[:n], mask)
        if self.classification:
            target = y[:n].argmax(-1) if y.dim() == 2 else y[:n].long()
            return F.cross_entropy(y_pred[:n], target)
        return F.mse_loss(y_pred.squeeze()[:n], y[:n].squeeze())


# ---------------------------------------------------------------------------
# 1) Sigmoid relaxation: simplest baseline
# ---------------------------------------------------------------------------


class KWTA_Sigmoid(_MaskConfigBase):
    """
    Continuous-relaxation KWTA: parameterise the mask logits ``m`` and
    optimise the soft mask ``sigmoid(m)`` directly through the readout.
    No straight-through, no noise, no budget penalty -- the binary
    constraint is enforced only at the end via ``kwta(m, k)``.
    """

    name = "sigmoid"

    def __call__(
        self,
        model: nn.Module,
        data: tuple,
        m: torch.Tensor,
        task_idx: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x, y = data
        device = x.device
        n = min(self.reconfig_n, x.shape[0])

        m_param = nn.Parameter(torch.zeros(self.hidden_size, device=device))
        opt = torch.optim.RMSprop([m_param], lr=self.reconfig_lr)
        z = model.parallel(x[:n]).detach()

        first_loss: float | None = None
        for _step in range(self.reconfig_steps):
            soft_mask = torch.sigmoid(m_param)
            loss = self._loss(model, z, soft_mask, x, y, n)
            if first_loss is None:
                first_loss = loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            best = kwta(m_param.detach(), k=self.k)
            final_loss = self._loss(model, z, best, x, y, n).item()

        self.last_initial_loss = float(first_loss) if first_loss is not None else None
        self.last_final_loss = float(final_loss)
        return best


# ---------------------------------------------------------------------------
# 2) Annealed STE: simulated annealing with straight-through KWTA
# ---------------------------------------------------------------------------


class KWTA_AnnealedSTE(_MaskConfigBase):
    """
    Simulated annealing with straight-through KWTA, after
    ``func_net_kwta/masks.py::KWTA_dyn``::

        tau   = m.std() * exp(-step / (reconfig_steps / 4))
        noise = randn_like(m) * tau
        mask  = kwta(m + noise, k) + m - m.detach()      # STE

    Each step evaluates a stochastic *binary* mask and keeps the best one
    seen so far.  Returns the lowest-loss hard mask.
    """

    name = "anneal_ste"

    def __call__(
        self,
        model: nn.Module,
        data: tuple,
        m: torch.Tensor,
        task_idx: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x, y = data
        device = x.device
        n = min(self.reconfig_n, x.shape[0])

        m_param = nn.Parameter(0.00 * torch.randn(self.hidden_size, device=device))
        opt = torch.optim.SGD([m_param], lr=self.reconfig_lr)
        z = model.parallel(x[:n]).detach()

        best: torch.Tensor | None = None
        lowest_loss = float("inf")
        first_loss: float | None = None
        denom = max(1.0, self.reconfig_steps / 4)

        for step in range(self.reconfig_steps):
            std = m_param.detach().std().item()
            # Use a positive default scale early on when m is still ~0.
            scale = std if std > 1e-6 else 0.1
            tau = scale * float(torch.exp(torch.tensor(-step / denom)).item())
            noise = torch.randn_like(m_param) * tau
            hard = kwta((m_param + noise).detach(), k=self.k)
            mask_ste = hard + m_param - m_param.detach()  # STE through topk

            loss = self._loss(model, z, mask_ste, x, y, n)

            if first_loss is None:
                first_loss = loss.item()
            if loss.item() < lowest_loss:
                lowest_loss = loss.item()
                best = hard.detach().clone()

            opt.zero_grad()
            loss.backward()
            opt.step()

        if best is None:
            best = kwta(m_param.detach(), k=self.k)
            with torch.no_grad():
                lowest_loss = self._loss(model, z, best, x, y, n).item()

        self.last_initial_loss = float(first_loss) if first_loss is not None else None
        self.last_final_loss = float(lowest_loss)
        return best


# ---------------------------------------------------------------------------
# 3) Gumbel-Sigmoid (Binary Concrete) relaxation, annealed
# ---------------------------------------------------------------------------


class KWTA_Gumbel(_MaskConfigBase):
    """
    Binary Concrete (Gumbel-Sigmoid) relaxation with temperature annealing.

    Each unit's activation is a binary decision so we use the Gumbel-Sigmoid
    (Maddison et al. 2017, Jang et al. 2017) rather than Gumbel-Softmax::

        u   ~ Uniform(0,1)
        g   = log(u) - log(1 - u)             # logistic noise = Gumbel(0)-Gumbel(0)
        z   = sigmoid((m + g) / tau)
        E[#active] ≈ sum sigmoid(m)
        loss = task_loss(z) + reg_strength * (sum sigmoid(m) - k)**2

    The temperature ``tau`` is annealed from ``tau_start`` to ``tau_end``
    on a log schedule across ``reconfig_steps`` steps so the soft mask
    sharpens into a binary sample.  The budget penalty pulls
    ``E[#active]`` toward ``k`` so the post-hoc top-k extraction is well
    defined.  At the end we return ``kwta(m, k)``.
    """

    name = "gumbel"

    def __init__(
        self,
        hidden_size: int,
        classification: bool = True,
        k: int = 128,
        reconfig_steps: int = 50,
        reconfig_lr: float = 0.05,
        reconfig_n: int = 2048,
        tau_start: float = 1.0,
        tau_end: float = 0.1,
        reg_strength: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            classification=classification,
            k=k,
            reconfig_steps=reconfig_steps,
            reconfig_lr=reconfig_lr,
            reconfig_n=reconfig_n,
            **kwargs,
        )
        self.tau_start = tau_start
        self.tau_end = tau_end
        # Budget penalty weight; the (E[#active]-k)^2 term is divided by
        # k**2 so the penalty stays of order 1 across hidden_size scales.
        self.reg_strength = reg_strength

    def __call__(
        self,
        model: nn.Module,
        data: tuple,
        m: torch.Tensor,
        task_idx: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x, y = data
        device = x.device
        n = min(self.reconfig_n, x.shape[0])

        # Initialise logits so E[#active] = sigmoid(m_init)*hidden_size ≈ k.
        # logit(k/hidden_size) is a much better starting point than zeros
        # (which makes E[#active] = hidden_size/2).
        p0 = self.k / max(1, self.hidden_size)
        logit0 = float(torch.logit(torch.tensor(p0, dtype=torch.float32)).item())
        m_param = nn.Parameter(torch.full((self.hidden_size,), logit0, device=device))
        opt = torch.optim.Adam([m_param], lr=self.reconfig_lr)
        z = model.parallel(x[:n]).detach()

        best: torch.Tensor | None = None
        lowest_loss = float("inf")
        first_loss: float | None = None
        total = max(1, self.reconfig_steps - 1)
        budget_scale = max(1.0, float(self.k * self.k))

        for step in range(self.reconfig_steps):
            frac = step / total if total > 0 else 0.0
            tau = self.tau_start * (self.tau_end / self.tau_start) ** frac

            # Logistic noise is the difference of two Gumbels.
            u = torch.rand_like(m_param).clamp(1e-6, 1.0 - 1e-6)
            logistic = torch.log(u) - torch.log1p(-u)
            soft = torch.sigmoid((m_param + logistic) / tau)

            task_loss = self._loss(model, z, soft, x, y, n)
            expected_active = torch.sigmoid(m_param).sum()
            reg = (expected_active - self.k) ** 2 / budget_scale
            loss = task_loss + self.reg_strength * reg

            if first_loss is None:
                first_loss = task_loss.item()

            with torch.no_grad():
                hard = kwta(m_param.detach(), k=self.k)
                hard_loss = self._loss(model, z, hard, x, y, n).item()
                if hard_loss < lowest_loss:
                    lowest_loss = hard_loss
                    best = hard.clone()

            opt.zero_grad()
            loss.backward()
            opt.step()

        if best is None:
            best = kwta(m_param.detach(), k=self.k)

        self.last_initial_loss = float(first_loss) if first_loss is not None else None
        self.last_final_loss = float(lowest_loss)
        return best


# ---------------------------------------------------------------------------
# Baselines (interface-compat with the trainer)
# ---------------------------------------------------------------------------


class FixedMask:
    """Deterministic disjoint blocks of size k assigned per task index."""

    name = "fixed"

    def __init__(self, hidden_size: int, k: int = 128, **kwargs) -> None:
        self.hidden_size = hidden_size
        self.k = k
        self._task_masks: dict[int, torch.Tensor] = {}
        self.last_initial_loss = None
        self.last_final_loss = None

    def __call__(
        self,
        model: nn.Module,
        data: tuple,
        m: torch.Tensor,
        task_idx: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        device = data[0].device
        if task_idx is None:
            task_idx = 0
        if task_idx not in self._task_masks:
            start = (task_idx * self.k) % self.hidden_size
            mask = torch.zeros(self.hidden_size, device=device)
            end = min(start + self.k, self.hidden_size)
            mask[start:end] = 1.0
            if end - start < self.k:
                mask[: self.k - (end - start)] = 1.0
            self._task_masks[task_idx] = mask
        return self._task_masks[task_idx].to(device)


class NoMask:
    """All-active fixed mask (first k slots) — naive fine-tuning baseline."""

    name = "nomask"

    def __init__(self, hidden_size: int, k: int = 128, **kwargs) -> None:
        self.hidden_size = hidden_size
        self.k = k
        _m = torch.zeros(hidden_size)
        _m[:k] = 1.0
        self._mask = _m
        self.last_initial_loss = None
        self.last_final_loss = None

    def __call__(
        self,
        model: nn.Module,
        data: tuple,
        m: torch.Tensor,
        task_idx: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._mask.to(data[0].device)


class RandMask:
    """Random per-batch top-k mask. Same signature as the configurer, but each
    call returns a fresh uniformly random k-of-H mask irrespective of the
    support batch. Used as the decisive ablation that holds the FTN training
    pipeline (filtered replay, schedule, parameter budget) fixed and replaces
    only the support-conditioned mask selection. The first time a given
    `task_idx` is seen we cache one mask per task so the eval-time pass
    (under the "stored" protocol) sees the same mask the training step did;
    a fresh sample is produced for every call when `per_call_resample=True`.
    """

    name = "randmask"

    def __init__(
        self,
        hidden_size: int,
        k: int = 128,
        per_call_resample: bool = True,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        self.hidden_size = hidden_size
        self.k = k
        self.per_call_resample = per_call_resample
        self._task_masks: dict[int, torch.Tensor] = {}
        self._gen = torch.Generator()
        if seed is not None:
            self._gen.manual_seed(int(seed))
        self.last_initial_loss = None
        self.last_final_loss = None

    def _sample(self, device) -> torch.Tensor:
        idx = torch.randperm(self.hidden_size, generator=self._gen)[: self.k]
        m = torch.zeros(self.hidden_size)
        m[idx] = 1.0
        return m.to(device)

    def __call__(
        self,
        model: nn.Module,
        data: tuple,
        m: torch.Tensor,
        task_idx: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        device = data[0].device
        if self.per_call_resample:
            return self._sample(device)
        if task_idx is None:
            task_idx = 0
        if task_idx not in self._task_masks:
            self._task_masks[task_idx] = self._sample(device)
        return self._task_masks[task_idx].to(device)


# ---------------------------------------------------------------------------
# Eval-time retrieval router (support-set conditioned)
# ---------------------------------------------------------------------------


class RetrievalRouter:
    """
    Eval-time configurer that picks a mask from a bank of stored
    training masks by minimising the loss on a labeled support set.

    Use this as the trainer's ``eval_mask_cfg``: at evaluation time the
    trainer hands it a small support set drawn from the target task,
    plus the dict of per-task masks accumulated during training, and
    the router scans the bank and returns the argmin-support-loss
    mask.  This tests whether the FTN inner-loop optimisation beats
    a simple support-set-conditioned retrieval over the same stored
    masks.

    Note: the task identity that produced the support set is NEVER
    revealed to the router; only the labeled support pairs are used.
    """

    name = "router"

    def __init__(self, hidden_size: int, classification: bool = True, **kwargs) -> None:
        self.hidden_size = hidden_size
        self.classification = classification
        self.last_initial_loss: float | None = None
        self.last_final_loss: float | None = None

    def __call__(
        self,
        model: nn.Module,
        data: tuple,
        m: torch.Tensor,
        task_idx: int | None = None,
        mask_store: dict | None = None,
        **kwargs,
    ) -> torch.Tensor:
        x, y = data
        device = x.device
        if mask_store is None or len(mask_store) == 0:
            # No stored masks yet — fall back to passing the input mask.
            return m if m is not None else torch.zeros(self.hidden_size, device=device)

        model.eval()
        best_mask: torch.Tensor | None = None
        best_loss = float("inf")
        first_loss: float | None = None
        with torch.no_grad():
            z = model.parallel(x).detach() if hasattr(model, "parallel") else None
            for _idx, stored in mask_store.items():
                stored = stored.to(device)
                if hasattr(model, "linear") and z is not None:
                    y_pred = model.linear(z * stored.reshape(1, -1))
                else:
                    y_pred = model(x, stored)
                if self.classification:
                    target = y.argmax(-1) if y.dim() == 2 else y.long()
                    loss = F.cross_entropy(y_pred, target).item()
                else:
                    loss = F.mse_loss(y_pred.squeeze(), y.squeeze()).item()
                if first_loss is None:
                    first_loss = loss
                if loss < best_loss:
                    best_loss = loss
                    best_mask = stored.clone()
        model.train()
        self.last_initial_loss = first_loss
        self.last_final_loss = best_loss
        return best_mask if best_mask is not None else next(iter(mask_store.values())).to(device)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


CONFIGURERS = {
    "Sigmoid": KWTA_Sigmoid,
    "AnnealSTE": KWTA_AnnealedSTE,
    "Gumbel": KWTA_Gumbel,
    "FixedMask": FixedMask,
    "NoMask": NoMask,
    "Router": RetrievalRouter,
}
