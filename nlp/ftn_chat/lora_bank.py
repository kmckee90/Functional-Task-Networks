"""LoRA bank with mask-gated atom selection (tailored for FTN-Chat).

A bank of H LoRA atoms is attached to each target linear of a frozen Qwen
backbone. A single shared binary mask of size H selects k active atoms; the
effective delta is the sum of selected atoms scaled by alpha/k.

This is a trimmed copy of ftn_lora/lora_bank.py with one deliberate change:
`init_b_zero` defaults to False (B ~ N(0, 0.01)), because the perplexity
configurer must be able to *select* untrained atoms, which a B=0 init makes
impossible (zero gradient through untrained atoms). See CONTINUATION.md §6.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn

# Empty tuple = wrap EVERY nn.Linear inside the module handed to
# install_lora_bank. This is the default: hybrid architectures (e.g. Qwen3.6's
# Gated DeltaNet blocks) use projection names that a fixed q/k/v/o-style list
# would silently miss. Pass explicit names to restrict.
TARGET_MODULES_DEFAULT: tuple = ()


@dataclass
class LoRABankConfig:
    H: int = 256
    k: int = 48
    rank: int = 4
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: tuple = TARGET_MODULES_DEFAULT
    init_b_zero: bool = False               # FTN-Chat default: small random B


class MaskController:
    """Holds the active mask; shared by every BankedLinear so one selection
    drives the whole model."""

    def __init__(self, H: int, k: int, device=None):
        self.H = H
        self.k = k
        self._mask = torch.zeros(H, device=device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    def set_mask(self, m: torch.Tensor):
        assert m.shape == (self.H,), f"mask shape {m.shape} != ({self.H},)"
        self._mask = m.to(self._mask.device, dtype=self._mask.dtype)

    def random_topk(self, generator: torch.Generator | None = None) -> torch.Tensor:
        idx = torch.randperm(self.H, generator=generator, device=self._mask.device)[: self.k]
        m = torch.zeros(self.H, device=self._mask.device)
        m[idx] = 1.0
        return m


class BankedLinear(nn.Module):
    """Frozen base linear + bank of H LoRA atoms.

    Forward: y = base(x) + (alpha/k) * sum_j m_j (x @ A_j @ B_j)
    Memory-safe: contract atom and rank dims in one einsum, never materialising
    the (..., k, out) intermediate.
    """

    def __init__(self, base: nn.Linear, cfg: LoRABankConfig, controller: MaskController):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.cfg = cfg
        self.controller = controller
        in_f, out_f = base.in_features, base.out_features
        device = base.weight.device
        dtype = base.weight.dtype

        A = torch.empty(cfg.H, in_f, cfg.rank, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        if cfg.init_b_zero:
            B = torch.zeros(cfg.H, cfg.rank, out_f, device=device, dtype=dtype)
        else:
            B = torch.empty(cfg.H, cfg.rank, out_f, device=device, dtype=dtype).normal_(0, 0.01)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    @property
    def scaling(self) -> float:
        # alpha/k normalization tracks the LIVE active-atom count so the GUI can
        # change k at inference without blowing up (more atoms) or fading out
        # (fewer) the delta magnitude. controller.k is the source of truth.
        return self.cfg.alpha / max(1, self.controller.k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        # Under multi-GPU sharding this layer's atoms (A/B) live on the same
        # device as its base weight; the shared mask is held on one device by
        # the controller, so move it here. .to() stays differentiable, so the
        # reconfigurer's straight-through gradient still reaches the mask logits.
        dev = self.A.device
        m = self.controller.mask
        if m.device != dev:
            m = m.to(dev)
        active = (m > 0).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return y
        w = m[active].to(x.dtype)
        A_act = self.A[active].to(x.dtype)
        B_act = self.B[active].to(x.dtype)
        xd = self.dropout(x)
        proj = torch.einsum("...i,kir->...kr", xd, A_act)
        proj = proj * w.view(*([1] * (proj.dim() - 2)), -1, 1)
        delta = torch.einsum("...kr,kro->...o", proj, B_act)
        return y + self.scaling * delta

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield self.A
        yield self.B


def install_lora_bank(model, cfg: LoRABankConfig, controller: MaskController,
                      target_modules: Iterable[str] | None = None) -> list[BankedLinear]:
    """Wrap linears under `model` with BankedLinear. With an empty target list
    (the default) EVERY nn.Linear is wrapped — robust to architectures whose
    projection names differ from the classic q/k/v/o convention."""
    targets = tuple(target_modules if target_modules is not None else cfg.target_modules)
    wrappers: list[BankedLinear] = []
    for p in model.parameters():
        p.requires_grad = False
    # Materialize the hit list BEFORE mutating the tree: setattr during a lazy
    # named_modules() walk would re-visit each fresh BankedLinear and wrap its
    # own `base` Linear, nesting without end. Also never descend into an
    # existing BankedLinear (idempotence).
    hits = []
    for _, parent in list(model.named_modules()):
        if isinstance(parent, BankedLinear):
            continue
        for child_name, child in parent.named_children():
            if isinstance(child, nn.Linear) and not isinstance(child, BankedLinear) and (
                not targets or any(child_name.endswith(t) for t in targets)
            ):
                hits.append((parent, child_name, child))
    for parent, child_name, child in hits:
        new = BankedLinear(child, cfg, controller)
        setattr(parent, child_name, new)
        wrappers.append(new)
    return wrappers


def lora_bank_state_dict(wrappers: Iterable[BankedLinear]) -> dict:
    sd = {}
    for i, w in enumerate(wrappers):
        sd[f"layer{i}.A"] = w.A.detach().clone().cpu()
        sd[f"layer{i}.B"] = w.B.detach().clone().cpu()
    return sd


def load_lora_bank_state_dict(wrappers: list[BankedLinear], sd: dict):
    for i, w in enumerate(wrappers):
        w.A.data.copy_(sd[f"layer{i}.A"].to(w.A.device, dtype=w.A.dtype))
        w.B.data.copy_(sd[f"layer{i}.B"].to(w.B.device, dtype=w.B.dtype))
