"""LoRA bank with mask-gated atom selection.

Implements a bank of H LoRA atoms attached to a frozen transformer linear
layer. A binary mask of size H selects k active atoms per forward pass; the
effective LoRA delta is the sum of selected atoms scaled by alpha/k.

The bank is wired into Qwen attention/MLP projections (q/k/v/o, up/gate/down).
A single shared mask drives every banked layer in the model — masks live on
the model and are read by every wrapper at forward time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


TARGET_MODULES_DEFAULT = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "gate_proj", "down_proj",
)


@dataclass
class LoRABankConfig:
    H: int = 256                    # number of atoms in the bank
    k: int = 16                     # active atoms per forward pass
    rank: int = 4                   # rank of each atom
    alpha: float = 16.0             # LoRA scaling, effective scale = alpha / k
    dropout: float = 0.05
    target_modules: tuple = TARGET_MODULES_DEFAULT
    init_b_zero: bool = True        # standard LoRA init: A random, B zero


class MaskController:
    """Holds the active mask and lets the model swap between training/eval masks.

    A wrapper module reads `controller.mask` at forward time. The same controller
    instance is shared by every BankedLinear in the model so a single mask
    selection drives the whole bank.
    """

    def __init__(self, H: int, k: int, device=None):
        self.H = H
        self.k = k
        self._mask = torch.zeros(H, device=device)
        # Default: no atoms active. Caller is responsible for setting before fwd.

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    def set_mask(self, m: torch.Tensor):
        assert m.shape == (self.H,)
        self._mask = m.to(self._mask.device, dtype=self._mask.dtype)

    def random_topk(self, generator: torch.Generator | None = None) -> torch.Tensor:
        idx = torch.randperm(self.H, generator=generator, device=self._mask.device)[: self.k]
        m = torch.zeros(self.H, device=self._mask.device)
        m[idx] = 1.0
        return m


class BankedLinear(nn.Module):
    """Wraps a frozen base linear with a bank of H LoRA atoms.

    Forward: y = base(x) + scale * sum_j m_j (x @ A_j @ B_j)
    A: (H, in, r), B: (H, r, out). For numerical stability we batch over H
    and gather only the k active atoms (typical case).
    """

    def __init__(self, base: nn.Linear, cfg: LoRABankConfig, controller: MaskController):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.cfg = cfg
        self.controller = controller
        in_f, out_f = base.in_features, base.out_features
        # Inherit device/dtype from the wrapped linear so install can be done
        # after the model is on GPU.
        device = base.weight.device
        dtype = base.weight.dtype

        # Bank parameters: (H, in, r) and (H, r, out).
        A = torch.empty(cfg.H, in_f, cfg.rank, device=device, dtype=dtype)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        if cfg.init_b_zero:
            B = torch.zeros(cfg.H, cfg.rank, out_f, device=device, dtype=dtype)
        else:
            B = torch.empty(cfg.H, cfg.rank, out_f, device=device, dtype=dtype).normal_(0, 0.01)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.scaling = cfg.alpha / cfg.k
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        m = self.controller.mask  # (H,)
        # Gather active atoms
        active = (m > 0).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return y
        w = m[active].to(x.dtype)               # (k_active,)
        A_act = self.A[active].to(x.dtype)      # (k_active, in, r)
        B_act = self.B[active].to(x.dtype)      # (k_active, r, out)
        xd = self.dropout(x)
        # Atom-wise low-rank projection without materialising per-atom output.
        # proj = x @ A : (..., k, r). Apply mask weights here (per atom). Then
        # contract over both k and r: (..., k, r) x (k, r, out) -> (..., out).
        # This avoids the (..., k, out) intermediate, which dominates memory
        # when the output dim is large (e.g. MLP up/gate at intermediate size).
        proj = torch.einsum("...i,kir->...kr", xd, A_act)
        proj = proj * w.view(*([1] * (proj.dim() - 2)), -1, 1)
        delta = torch.einsum("...kr,kro->...o", proj, B_act)
        return y + self.scaling * delta

    def lora_parameters(self) -> Iterable[nn.Parameter]:
        yield self.A
        yield self.B


def install_lora_bank(
    model: nn.Module,
    cfg: LoRABankConfig,
    controller: MaskController,
    target_modules: Iterable[str] | None = None,
) -> list[BankedLinear]:
    """Walk the model and replace every Linear whose name endswith a target
    module with a BankedLinear. Returns the list of wrappers."""
    targets = tuple(target_modules or cfg.target_modules)
    wrappers: list[BankedLinear] = []

    def _replace(parent: nn.Module, name: str):
        child = getattr(parent, name)
        new = BankedLinear(child, cfg, controller)
        setattr(parent, name, new)
        wrappers.append(new)

    # Freeze the entire model first
    for p in model.parameters():
        p.requires_grad = False

    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and any(child_name.endswith(t) for t in targets):
                _replace(parent, child_name)
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
