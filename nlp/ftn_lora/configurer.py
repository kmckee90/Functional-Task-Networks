"""Support-conditioned mask configurer for the LoRA bank.

The configurer takes a labeled support batch and returns a binary top-k mask.
Model weights are frozen during configuration; only mask logits receive
gradients. We implement AnnealSTE (the chosen default in the FTN paper) and
a simple Random retrieval router for ablations.

The support loss is the cross-entropy on answer tokens, masked by labels==-100
to exclude prompt and padding from the loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora_bank import MaskController


@dataclass
class ConfigurerConfig:
    type: str = "AnnealSTE"      # AnnealSTE | RandomFixed | Stored
    inner_steps: int = 10
    lr: float = 0.1
    init: str = "zero"           # logits init (AnnealSTE only)


def _topk_mask(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Return a hard binary mask with exactly k ones at the top-k logit positions."""
    H = logits.shape[-1]
    idx = logits.topk(k).indices
    m = torch.zeros_like(logits)
    m.scatter_(-1, idx, 1.0)
    return m


def _ste_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k mask in forward, identity gradient on logits in backward."""
    hard = _topk_mask(logits, k)
    return (hard - logits).detach() + logits


class AnnealSTEConfigurer:
    """Optimize mask logits for S inner steps on a support batch with weights frozen.

    Each inner step:
      1. apply hard top-k via STE → mask m
      2. compute support loss with mask m active
      3. backward to logits and SGD step
    Returns the lowest-loss hard mask found across inner steps.

    `model` is optional; if provided, gradient_checkpointing is toggled off
    during the inner loop (we don't need bank-weight grads, only mask grads,
    and checkpointing's memory savings come at the cost of recompute that
    isn't worth it for the small graph we actually need to retain). It is
    re-enabled before the function returns.
    """

    def __init__(self, cfg: ConfigurerConfig, controller: MaskController):
        self.cfg = cfg
        self.controller = controller

    def __call__(
        self,
        support_loss_fn: Callable[[], torch.Tensor],
        device: torch.device,
        model=None,
        verbose: bool = False,
    ) -> torch.Tensor:
        H, k = self.controller.H, self.controller.k
        logits = torch.zeros(H, device=device, requires_grad=True) \
            if self.cfg.init == "zero" \
            else (torch.randn(H, device=device) * 0.1).requires_grad_(True)

        opt = torch.optim.SGD([logits], lr=self.cfg.lr)

        # Freeze ALL params for the configurer; only the logits should receive
        # gradient. Otherwise PyTorch builds a graph allocating buffers for
        # every leaf with requires_grad=True.
        was_requires_grad = []
        if model is not None:
            for p in model.parameters():
                was_requires_grad.append((p, p.requires_grad))
                p.requires_grad = False
            ckpt_was_on = getattr(model, "is_gradient_checkpointing", False)
            if ckpt_was_on:
                try:
                    model.gradient_checkpointing_disable()
                except Exception:
                    ckpt_was_on = False

        best_loss = float("inf")
        best_mask: Optional[torch.Tensor] = None
        try:
            for step in range(self.cfg.inner_steps + 1):
                with torch.enable_grad():
                    m = _ste_topk(logits, k)
                    self.controller.set_mask(m)
                    loss = support_loss_fn()
                with torch.no_grad():
                    loss_val = float(loss.detach())
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_mask = _topk_mask(logits.detach(), k)
                    if verbose:
                        print(f"  cfg step {step}: loss={loss_val:.4f}")
                if step < self.cfg.inner_steps:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                # Free the loss/graph promptly
                del loss
        finally:
            if model is not None:
                for p, rg in was_requires_grad:
                    p.requires_grad = rg
                if ckpt_was_on:
                    try:
                        model.gradient_checkpointing_enable()
                    except Exception:
                        pass

        assert best_mask is not None
        self.controller.set_mask(best_mask)
        return best_mask.detach()


class RandomFixedConfigurer:
    """Fixed-random per-task mask. The first time we see a (task) key we sample
    a random k-of-H mask and cache it; subsequent calls reuse the same mask.
    Used as a baseline ('shared LoRA-bank with random fixed top-k')."""

    def __init__(self, controller: MaskController, seed: int = 0):
        self.controller = controller
        self.cache: dict[str, torch.Tensor] = {}
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def get_or_sample(self, key: str, device: torch.device) -> torch.Tensor:
        if key not in self.cache:
            idx = torch.randperm(self.controller.H, generator=self.gen)[: self.controller.k]
            m = torch.zeros(self.controller.H)
            m[idx] = 1.0
            self.cache[key] = m
        m = self.cache[key].to(device)
        self.controller.set_mask(m)
        return m


class RetrievalRouter:
    """At eval time, scan stored masks and pick the one with lowest support loss.

    This is the support-set-conditioned retrieval baseline (analogue of SupSup
    retrieval and L2P/DualPrompt support-set selection).
    """

    def __init__(self, controller: MaskController):
        self.controller = controller
        self.store: list[tuple[str, torch.Tensor]] = []  # (label, mask)

    def add(self, key: str, mask: torch.Tensor):
        self.store.append((key, mask.detach().cpu().clone()))

    def __call__(
        self,
        support_loss_fn: Callable[[], torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        assert self.store, "RetrievalRouter has no stored masks"
        best_loss = float("inf")
        best_mask: Optional[torch.Tensor] = None
        for _, m in self.store:
            m_dev = m.to(device)
            self.controller.set_mask(m_dev)
            with torch.no_grad():
                loss = float(support_loss_fn().detach())
            if loss < best_loss:
                best_loss = loss
                best_mask = m_dev
        assert best_mask is not None
        self.controller.set_mask(best_mask)
        return best_mask.detach()
