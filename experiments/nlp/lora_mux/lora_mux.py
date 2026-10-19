"""
lora_mux.py — LoRA-Mux: a structured pool of LoRA components routed by a
binary mask, applied to HuggingFace GPT-2-style Conv1D projections.

Forward of a single LoRA-Mux module:
    y = (W_frozen + sum_{k=1}^{H} m_k * B_k A_k) x

with m in {0,1}^H. Gradients to inactive components (m_k = 0) are exactly
zero, so disjoint masks across tasks give exactly disjoint gradient updates
(matches Theorem 1 in the ICML draft).

Self-contained: no imports from project_c_4. Only depends on torch and
HuggingFace `transformers` (for the Conv1D class used by GPT-2).
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.pytorch_utils import Conv1D


# ---------------------------------------------------------------------------
# Mask state — a tiny mutable container threaded through the wrapped modules
# ---------------------------------------------------------------------------


class MaskState:
    """
    Global current-mask holder. The trainer/configurer sets `mask` before
    every forward; LoRAMuxConv1D reads it. Letting `mask` be a tensor with
    `requires_grad=True` makes the forward differentiable w.r.t. the mask
    (used by the SmoothKWTA configurer to compute mask-logit gradients).
    """

    def __init__(self, n_components: int, device: str = "cpu") -> None:
        self.n_components = n_components
        self.mask = torch.ones(n_components, device=device)

    def set(self, m: torch.Tensor) -> None:
        self.mask = m


# ---------------------------------------------------------------------------
# LoRA-Mux wrapper for HF GPT-2 Conv1D
# ---------------------------------------------------------------------------


class LoRAMuxConv1D(nn.Module):
    """
    Wrap a HuggingFace Conv1D (used in GPT-2 attention/MLP) with a pool of
    LoRA components.  HF Conv1D stores weight as (in, out) and computes
    `x @ weight + bias`; we mirror that and add the LoRA delta.

    Args:
        base:           the original Conv1D (frozen).
        n_components:   pool size H.
        rank:           per-component rank r0.
        mask_state:     MaskState whose `.mask` is read at every forward.
        scaling:        LoRA alpha/r0 scale (default 1.0).
    """

    def __init__(
        self,
        base: Conv1D,
        n_components: int,
        rank: int,
        mask_state: MaskState,
        scaling: float = 1.0,
        random_B_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        # HF Conv1D weight shape: (nx, nf)  i.e. (in_features, out_features).
        nx, nf = base.weight.shape
        self.in_features = nx
        self.out_features = nf
        self.n_components = n_components
        self.rank = rank
        self.scaling = scaling
        self.mask_state = mask_state

        # Standard LoRA init: A small uniform, B zero (so initial delta is 0).
        # For the LoRA-Bank baseline we want each component to do something
        # different from the start; pass random_B_init>0 to break symmetry.
        bound = 1.0 / math.sqrt(nx)
        self.A = nn.Parameter(torch.empty(n_components, rank, nx).uniform_(-bound, bound))
        if random_B_init > 0:
            self.B = nn.Parameter(torch.randn(n_components, nf, rank) * random_B_init)
        else:
            self.B = nn.Parameter(torch.zeros(n_components, nf, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate HF Conv1D's reshape behaviour.
        size_out = x.size()[:-1] + (self.out_features,)
        x_flat = x.view(-1, self.in_features)
        base = torch.addmm(self.base.bias, x_flat, self.base.weight)  # (N, nf)

        # LoRA delta over the pool: delta_k = (B_k A_k) x ; mask weights it.
        # Compute A x first to avoid materialising (H, nf, nx) products.
        Ax = torch.einsum("hri,ni->nhr", self.A, x_flat)              # (N, H, r)
        delta = torch.einsum("hor,nhr->nho", self.B, Ax)              # (N, H, nf)
        m = self.mask_state.mask
        out = base + self.scaling * (m.view(1, -1, 1) * delta).sum(dim=1)
        return out.view(*size_out)

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield self.A
        yield self.B


# ---------------------------------------------------------------------------
# Apply LoRA-Mux to a HF model
# ---------------------------------------------------------------------------


def apply_lora_mux(
    model: nn.Module,
    n_components: int,
    rank: int,
    mask_state: MaskState,
    target_names: tuple = ("c_attn",),
    random_B_init: float = 0.0,
) -> List[LoRAMuxConv1D]:
    """
    Walk the model and replace every attribute named in `target_names` whose
    value is a Conv1D with a LoRAMuxConv1D wrapper.  Returns the list of
    wrappers (handy for grouping LoRA-only parameters).
    """
    wrappers: List[LoRAMuxConv1D] = []
    for parent in model.modules():
        for tn in target_names:
            attr = getattr(parent, tn, None)
            if attr is not None and isinstance(attr, Conv1D):
                wrap = LoRAMuxConv1D(
                    attr, n_components, rank, mask_state,
                    random_B_init=random_B_init,
                )
                setattr(parent, tn, wrap)
                wrappers.append(wrap)
    return wrappers


def lora_mux_parameters(wrappers: List[LoRAMuxConv1D]) -> List[nn.Parameter]:
    out: List[nn.Parameter] = []
    for w in wrappers:
        out.extend([w.A, w.B])
    return out


# ---------------------------------------------------------------------------
# Mask configurers
# ---------------------------------------------------------------------------


@dataclass
class SmoothKWTA:
    """
    Three-stage configurer over the LoRA-component pool:
        1. Adam steps on sigmoid-relaxed mask logits to minimise the loss
           on a support batch (with the live LoRA-Mux model).
        2. Lateral smoothing on a 2-D grid (torus padding).
        3. Top-k binarisation.

    Args:
        n_components, grid_dim, k:    pool/grid/winners (D*D == H).
        kernel_size, lateral_steps:   smoothing s and T.
        reconfig_steps, reconfig_lr:  Stage-1 inner-loop steps S and lr.
    """

    n_components: int
    grid_dim: int
    k: int
    kernel_size: int
    lateral_steps: int
    reconfig_steps: int
    reconfig_lr: float

    def __post_init__(self) -> None:
        assert self.grid_dim * self.grid_dim == self.n_components, (
            f"grid_dim^2 ({self.grid_dim**2}) must equal n_components ({self.n_components})"
        )

    def _smooth(self, m_2d: torch.Tensor) -> torch.Tensor:
        kernel = torch.ones(
            1, 1, self.kernel_size, self.kernel_size, device=m_2d.device
        ) / (self.kernel_size**2)
        m = m_2d.unsqueeze(0).unsqueeze(0)
        pad = self.kernel_size // 2
        for _ in range(self.lateral_steps):
            padded = F.pad(m, (pad, pad, pad, pad), mode="circular")
            m = F.conv2d(padded, kernel)
        return m.squeeze(0).squeeze(0)

    def __call__(
        self,
        loss_fn,
        mask_state: MaskState,
        device: torch.device,
        task_idx: int = 0,
    ) -> torch.Tensor:
        """
        loss_fn(): closure that runs a forward pass and returns a scalar loss
                   using whatever mask is currently in mask_state.mask.
                   The configurer sets a relaxed (sigmoid-of-logits) mask in
                   mask_state.mask before each call.
        """
        old_mask = mask_state.mask
        m_logits = torch.zeros(self.n_components, device=device, requires_grad=True)
        opt = torch.optim.Adam([m_logits], lr=self.reconfig_lr)
        for _ in range(self.reconfig_steps):
            opt.zero_grad()
            m_relaxed = torch.sigmoid(m_logits)
            mask_state.mask = m_relaxed
            loss = loss_fn()
            loss.backward()
            opt.step()
        with torch.no_grad():
            m2d = m_logits.detach().view(self.grid_dim, self.grid_dim)
            m2d = self._smooth(m2d)
            m_flat = m2d.flatten()
            top = m_flat.topk(self.k).indices
            mask = torch.zeros_like(m_flat)
            mask[top] = 1.0
        mask_state.mask = old_mask
        return mask


@dataclass
class FixedMask:
    """Disjoint blocks of size k per task index.  Structural oracle baseline."""

    n_components: int
    k: int
    _cache: dict = field(default_factory=dict)

    def install(self, task_idx: int, device: torch.device) -> torch.Tensor:
        if task_idx not in self._cache:
            mask = torch.zeros(self.n_components, device=device)
            start = (task_idx * self.k) % self.n_components
            end = min(start + self.k, self.n_components)
            mask[start:end] = 1.0
            if end - start < self.k:
                mask[: self.k - (end - start)] = 1.0
            self._cache[task_idx] = mask
        return self._cache[task_idx].clone()

    def __call__(self, loss_fn, mask_state, device, task_idx=0):
        return self.install(task_idx, device)


@dataclass
class LoRABank:
    """
    Best-of-N LoRA bank: at every (re)configuration we evaluate the support
    batch under each one-hot mask m=e_k for k=0..H-1 and select the
    component with the lowest loss.  This is the simplest version of the
    LoRA-FTN idea (no mask machinery, just argmin over independently
    initialised LoRAs); requires non-zero $B$ initialisation to break
    initial symmetry — see `LoRAMuxConv1D(random_B_init=...)`.

    Args:
        n_components: pool size H.
        is_clf:       Cross-entropy when True (dummy here; LM loss is set by
                      the loss_fn passed at call time).  Reserved for parity
                      with the synthetic experiment's interface.
    """

    n_components: int
    is_clf: bool = False  # unused for LM but kept for API parity

    def __call__(self, loss_fn, mask_state, device, task_idx=0):
        H = self.n_components
        old_mask = mask_state.mask
        best_idx, best_loss = 0, float("inf")
        with torch.no_grad():
            for k in range(H):
                m = torch.zeros(H, device=device)
                m[k] = 1.0
                mask_state.mask = m
                loss = loss_fn().item()
                if loss < best_loss:
                    best_loss = loss
                    best_idx = k
        mask_state.mask = old_mask
        out = torch.zeros(H, device=device)
        out[best_idx] = 1.0
        return out


@dataclass
class FullMask:
    """
    All-ones mask: every LoRA component is active for every task.  This is
    the LoRA-Naive baseline — a single shared LoRA (effective rank H * r0)
    sequentially fine-tuned across tasks; the catastrophic-forgetting
    reference point.
    """

    n_components: int

    def __call__(self, loss_fn, mask_state, device, task_idx=0):
        return torch.ones(self.n_components, device=device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def reset_optimizer_state(opt: torch.optim.Optimizer) -> None:
    """Reset Adam (m, v) at task boundaries.  Required for the disjoint-mask
    structural guarantee to hold empirically when the active mask changes
    between tasks (see ICML draft, Section 3 / Implementation Notes)."""
    opt.state = defaultdict(dict)
