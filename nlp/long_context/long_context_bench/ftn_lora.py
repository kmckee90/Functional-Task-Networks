from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn


TARGET_MODULES_DEFAULT = (
    "q_proj",
    "o_proj",
    "up_proj",
    "gate_proj",
    "down_proj",
)


@dataclass
class LoRABankConfig:
    H: int = 256
    k: int = 32
    rank: int = 2
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: tuple[str, ...] = TARGET_MODULES_DEFAULT
    init_b_zero: bool = False


class MaskController:
    def __init__(self, H: int, k: int, device=None):
        self.H = H
        self.k = k
        self._mask = torch.zeros(H, device=device)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    def set_mask(self, mask: torch.Tensor) -> None:
        if tuple(mask.shape) != (self.H,):
            raise ValueError(f"mask shape {tuple(mask.shape)} != ({self.H},)")
        self._mask = mask.to(self._mask.device, dtype=self._mask.dtype)

    def zero(self) -> None:
        self._mask = torch.zeros_like(self._mask)

    def random_topk(self, generator: torch.Generator | None = None) -> torch.Tensor:
        idx = torch.randperm(self.H, generator=generator, device=self._mask.device)[: self.k]
        mask = torch.zeros(self.H, device=self._mask.device)
        mask[idx] = 1.0
        return mask


class BankedLinear(nn.Module):
    """Frozen linear layer plus a mask-gated bank of LoRA atoms."""

    def __init__(self, base: nn.Linear, cfg: LoRABankConfig, controller: MaskController):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.cfg = cfg
        self.controller = controller
        in_f, out_f = base.in_features, base.out_features
        device, dtype = base.weight.device, base.weight.dtype

        self.A = nn.Parameter(torch.empty(cfg.H, in_f, cfg.rank, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        if cfg.init_b_zero:
            self.B = nn.Parameter(torch.zeros(cfg.H, cfg.rank, out_f, device=device, dtype=dtype))
        else:
            self.B = nn.Parameter(
                torch.empty(cfg.H, cfg.rank, out_f, device=device, dtype=dtype).normal_(0, 0.01)
            )
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

    @property
    def scaling(self) -> float:
        return self.cfg.alpha / max(1, self.controller.k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        mask = self.controller.mask
        if mask.device != self.A.device:
            mask = mask.to(self.A.device)
        active = (mask > 0).nonzero(as_tuple=True)[0]
        if active.numel() == 0:
            return y

        weights = mask[active].to(dtype=x.dtype)
        a = self.A[active].to(dtype=x.dtype)
        b = self.B[active].to(dtype=x.dtype)
        xd = self.dropout(x)
        proj = torch.einsum("...i,kir->...kr", xd, a)
        proj = proj * weights.view(*([1] * (proj.dim() - 2)), -1, 1)
        delta = torch.einsum("...kr,kro->...o", proj, b)
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
    targets = tuple(target_modules or cfg.target_modules)
    wrappers: list[BankedLinear] = []
    for p in model.parameters():
        p.requires_grad = False
    for _, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and any(child_name.endswith(t) for t in targets):
                wrapped = BankedLinear(child, cfg, controller)
                setattr(parent, child_name, wrapped)
                wrappers.append(wrapped)
    if not wrappers:
        raise ValueError(f"no target Linear modules matched {targets}")
    return wrappers


def lora_parameters(wrappers: Iterable[BankedLinear]) -> list[nn.Parameter]:
    return [p for wrapper in wrappers for p in wrapper.lora_parameters()]


def lora_bank_state_dict(wrappers: Iterable[BankedLinear]) -> dict[str, torch.Tensor]:
    state = {}
    for i, wrapper in enumerate(wrappers):
        state[f"layer{i}.A"] = wrapper.A.detach().cpu()
        state[f"layer{i}.B"] = wrapper.B.detach().cpu()
    return state


def load_lora_bank_state_dict(wrappers: list[BankedLinear], state: dict[str, torch.Tensor]) -> None:
    for i, wrapper in enumerate(wrappers):
        wrapper.A.data.copy_(state[f"layer{i}.A"].to(wrapper.A.device, dtype=wrapper.A.dtype))
        wrapper.B.data.copy_(state[f"layer{i}.B"].to(wrapper.B.device, dtype=wrapper.B.dtype))


def kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    k = max(0, min(int(k), x.numel()))
    out = torch.zeros_like(x)
    if k == 0:
        return out
    idx = torch.topk(x.flatten(), k).indices
    out.flatten()[idx] = 1.0
    return out


class LateralSmoother(nn.Module):
    def __init__(self, grid_dim: int, kernel_size: int, torus: bool = False):
        super().__init__()
        self.grid_dim = grid_dim
        self.conv = nn.Conv2d(
            1,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode="circular" if torus else "zeros",
            bias=False,
        )
        self.conv.requires_grad_(False)
        self.conv.weight.data.fill_(1.0 / (kernel_size * kernel_size))

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        x = mask.view(1, 1, self.grid_dim, self.grid_dim)
        return self.conv(x).view(-1)


@dataclass
class ReconfigConfig:
    grid_dim: int
    reconfig_steps: int = 1
    train_reconfig_steps: int = 1
    reconfig_lr: float = 1.0
    lateral_steps: int = 0
    lateral_kernel: int = 3
    lateral_polyak: float = 0.0
    torus: bool = False


class PerplexityConfigurer:
    """Select an FTN-LoRA mask by minimizing prompt NLL."""

    def __init__(self, cfg: ReconfigConfig, controller: MaskController):
        self.cfg = cfg
        self.controller = controller
        self.smoother = LateralSmoother(cfg.grid_dim, cfg.lateral_kernel, cfg.torus).to(
            controller.mask.device
        )
        self._last_nll = float("inf")

    def __call__(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        steps: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        cfg, ctrl = self.cfg, self.controller
        n_steps = cfg.reconfig_steps if steps is None else int(steps)
        if n_steps <= 0:
            mask = kwta(torch.zeros(ctrl.H, device=ctrl.mask.device), ctrl.k)
            ctrl.set_mask(mask)
            return mask, float("nan")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        saved = [(p, p.requires_grad) for p in model.parameters()]
        for p, _ in saved:
            p.requires_grad = False
        ckpt_on = bool(getattr(model, "is_gradient_checkpointing", False))
        if ckpt_on and hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

        logits = (torch.randn(ctrl.H, device=ctrl.mask.device) * 1e-3).requires_grad_(True)
        opt = torch.optim.Adam([logits], lr=cfg.reconfig_lr)
        best_mask, best_nll = None, float("inf")
        try:
            with torch.enable_grad():
                for step in range(n_steps):
                    sm = torch.sigmoid(logits)
                    tau = logits.detach().std() * math.exp(-step / max(1.0, n_steps / 4))
                    hard = kwta(logits + torch.randn_like(logits) * tau, ctrl.k)
                    mask = hard + sm - sm.detach()
                    ctrl.set_mask(mask)
                    loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=input_ids,
                    ).loss
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    nll = float(loss.detach())
                    if nll < best_nll:
                        best_nll = nll
                        best_mask = kwta(logits.detach(), ctrl.k)
        finally:
            for p, requires_grad in saved:
                p.requires_grad = requires_grad
            if ckpt_on and hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )

        with torch.no_grad():
            mask = best_mask if best_mask is not None else kwta(logits.detach(), ctrl.k)
            for _ in range(cfg.lateral_steps):
                smooth = self.smoother(mask.float())
                mask = cfg.lateral_polyak * mask + (1.0 - cfg.lateral_polyak) * smooth
                mask = kwta(mask, ctrl.k)
            ctrl.set_mask(mask)
        self._last_nll = best_nll
        return mask.detach(), best_nll
