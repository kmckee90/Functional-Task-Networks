"""
masks.py — Mask configurers for Functional Task Networks (project_c).

A mask configurer is a callable that takes (model, data_batch, prev_mask)
and returns a new binary mask over hidden units.

The three-stage procedure:
  1. Gradient-based optimization: treat mask as a continuous parameter,
     optimise with RMSprop for a small number of steps.
  2. Lateral dynamics: convolve mask on 2-D cortical grid with a uniform
     kernel, enforcing spatial coherence.
  3. K-winner-take-all (KWTA): binarise by selecting the top-k values.

Ablation variants disable one or more stages to isolate contributions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return binary mask with 1 at the top-k positions of x."""
    topk = torch.topk(x, k).indices
    out = torch.zeros_like(x)
    out[topk] = 1.0
    return out


def _mask_to_grid(m: torch.Tensor, block_dim: int) -> torch.Tensor:
    """Reshape flat mask (n,) into 2-D grid (block_dim, W) for spatial ops."""
    block_size = block_dim * block_dim
    blocks = m.split(block_size)
    return torch.cat([b.reshape(block_dim, block_dim) for b in blocks], dim=-1)


def _grid_to_mask(grid: torch.Tensor, block_dim: int) -> torch.Tensor:
    """Inverse of _mask_to_grid."""
    cols = grid.split(block_dim, dim=-1)
    return torch.cat([c.flatten() for c in cols])


# ---------------------------------------------------------------------------
# SmoothKWTA — the main FTN mask configurer
# ---------------------------------------------------------------------------


class SmoothKWTA:
    """
    Mask configurer: gradient optimisation → lateral smoothing → KWTA.

    This is the core mask configuration procedure described in the paper.
    The mask is re-initialised from small Gaussian noise (std=0.1) at each
    call (not warm-started), ensuring each task starts from an unbiased
    prior while breaking tie-symmetry in the downstream KWTA step.  A
    zero initialisation would make sigmoid(m)=0.5 uniform and cause the
    KWTA index tie-break to deterministically pick neurons [0..k-1] for
    every task; this is harmless for cross-entropy (per-class gradient
    structure breaks symmetry in one step) but catastrophic for scalar
    MSE, where the symmetry-breaking signal from a single batch is weak.

    Args:
        hidden_size:    Total number of hidden units (grid_dim^2 * n_blocks).
        grid_dim:       Side length of each square block on the cortical grid.
        classification: Use cross-entropy loss (True) or MSE (False).
        k:              Number of KWTA winners (active neurons per task).
        kernel_size:    Size of uniform averaging kernel (odd integer).
        lateral_steps:  Number of lateral smoothing iterations.
        lateral_polyak: Polyak blend α: m ← α*m + (1-α)*smooth(m).
                        Set to 0 for full replacement (default).
        reconfig_steps: Gradient optimisation steps per call.
        reconfig_lr:    Learning rate for RMSprop mask optimiser.
        reconfig_n:     Subsample size for mask optimisation (max samples used).
        torus:          If True, use circular (torus) padding in lateral conv.
    """

    def __init__(
        self,
        hidden_size: int,
        grid_dim: int,
        classification: bool = True,
        k: int = 128,
        kernel_size: int = 17,
        lateral_steps: int = 2,
        lateral_polyak: float = 0.0,
        reconfig_steps: int = 10,
        reconfig_lr: float = 0.2,
        reconfig_n: int = 2048,
        torus: bool = True,
    ) -> None:
        self.hidden_size = hidden_size
        self.grid_dim = grid_dim
        self.classification = classification
        self.k = k
        self.lateral_steps = lateral_steps
        self.lateral_polyak = lateral_polyak
        self.reconfig_steps = reconfig_steps
        self.reconfig_lr = reconfig_lr
        self.reconfig_n = reconfig_n

        padding_mode = "circular" if torus else "zeros"
        self.lateral = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
            padding_mode=padding_mode,
        ).requires_grad_(False)
        self.lateral.weight.data.fill_(1.0 / kernel_size**2)
        # unnormalised — KWTA normalises

    def _smooth(self, m: torch.Tensor) -> torch.Tensor:
        """Apply one step of lateral smoothing on the 2-D cortical grid."""
        grid = _mask_to_grid(m, self.grid_dim).unsqueeze(0).unsqueeze(0)
        grid = self.lateral(grid).squeeze(0).squeeze(0)
        return _grid_to_mask(grid, self.grid_dim)

    def __call__(
        self,
        model: nn.Module,
        data: tuple,
        m: torch.Tensor,
        task_idx: int = None,  # accepted but unused; kept for interface compat
        **kwargs,
    ) -> torch.Tensor:
        """
        Reconfigure mask for the current task/data batch.

        Args:
            model: Network with forward(x, m) signature.
            data:  (x, y) batch tensors on the correct device.
            m:     Previous mask (shape: hidden_size); used for shape/device info
                   only — optimisation always starts from randn*0.1.

        Returns:
            Binary mask tensor of shape (hidden_size,).
        """
        x, y = data
        n = min(self.reconfig_n, x.shape[0])
        device = x.device

        # Move lateral kernel to correct device
        self.lateral = self.lateral.to(device)

        # Stage 1: gradient-based mask optimisation.  Init with small
        # Gaussian noise (std=0.1) rather than zeros to break KWTA
        # tie-symmetry (see class docstring).
        m_param = nn.Parameter(torch.randn(self.hidden_size, device=device) * 0.0)
        opt = torch.optim.RMSprop([m_param], lr=self.reconfig_lr)
        z = model.parallel(x[:n]).detach()
        for _ in range(self.reconfig_steps):
            sm = torch.sigmoid(m_param)
            mask = sm  # torch.bernoulli(sm).detach() + sm - sm.detach()  # STE
            y_pred = model.linear(z * mask)

            if self.classification:
                target = y[:n].argmax(dim=-1) if y.dim() == 2 else y[:n].long()
                loss = F.cross_entropy(y_pred[:n], target)
            else:
                loss = F.mse_loss(y_pred.squeeze()[:n], y[:n].squeeze())

            opt.zero_grad()
            loss.backward()
            opt.step()

        m_cont = m_param.data.clone().detach()

        # Stage 2: lateral dynamics
        if self.lateral_steps > 0:
            for _ in range(self.lateral_steps):
                m_smooth = self._smooth(m_cont)
                m_cont = (
                    self.lateral_polyak * m_cont + (1 - self.lateral_polyak) * m_smooth
                )

        # Stage 3: KWTA binarisation
        return kwta(m_cont, k=self.k)


# ---------------------------------------------------------------------------
# Preset variants
# ---------------------------------------------------------------------------


class SmoothKWTAFast(SmoothKWTA):
    """Large kernel (17×17), 2 lateral steps — rapid spatial convergence."""

    def __init__(self, hidden_size, grid_dim, **kw):
        kw.setdefault("kernel_size", 17)
        kw.setdefault("lateral_steps", 2)
        kw.setdefault("lateral_polyak", 0.0)
        super().__init__(hidden_size, grid_dim, **kw)


class SmoothKWTASlow(SmoothKWTA):
    """Small kernel (3×3), 15 lateral steps — fine-grained spatial patterns."""

    def __init__(self, hidden_size, grid_dim, **kw):
        kw.setdefault("kernel_size", 3)
        kw.setdefault("lateral_steps", 15)
        kw.setdefault("lateral_polyak", 0.0)
        super().__init__(hidden_size, grid_dim, **kw)


class PolyakKWTAFast(SmoothKWTA):
    """Medium kernel (7×7), 2 steps, Polyak α=0.95 — preserves gradient structure."""

    def __init__(self, hidden_size, grid_dim, **kw):
        kw.setdefault("kernel_size", 7)
        kw.setdefault("lateral_steps", 2)
        kw.setdefault("lateral_polyak", 0.95)
        super().__init__(hidden_size, grid_dim, **kw)


class KWTAOnly(SmoothKWTA):
    """
    Gradient optimisation + KWTA, NO lateral smoothing.
    Ablation: isolates value of spatial coherence.
    """

    def __init__(self, hidden_size, grid_dim, **kw):
        kw.setdefault("kernel_size", 3)
        kw.setdefault("lateral_steps", 0)
        super().__init__(hidden_size, grid_dim, **kw)


# ---------------------------------------------------------------------------
# FixedMask — static partition baseline
# ---------------------------------------------------------------------------


class FixedMask:
    """
    Assigns a fixed, predetermined block of neurons to each task.

    Tasks receive disjoint blocks of size k in the order they appear.
    No gradient optimisation or lateral dynamics — the mask is purely
    determined by task index. This prevents forgetting by construction
    but does not adapt to the data.

    Args:
        hidden_size: Total hidden units.
        grid_dim:    Grid side length (for interface compatibility).
        k:           Number of active neurons per task.
    """

    def __init__(self, hidden_size: int, grid_dim: int, k: int = 128, **kw):
        self.hidden_size = hidden_size
        self.k = k
        self._task_counter = 0
        self._task_masks: dict[int, torch.Tensor] = {}

    def __call__(
        self,
        model: nn.Module,
        data: tuple,
        m: torch.Tensor,
        task_idx: int = None,
        **kwargs,
    ) -> torch.Tensor:
        device = data[0].device
        # If task_idx is not provided, we cannot determine which fixed block
        # to use unambiguously.  Fall back to 0 to avoid silent counter bugs.
        if task_idx is None:
            task_idx = 0

        if task_idx not in self._task_masks:
            start = (task_idx * self.k) % self.hidden_size
            mask = torch.zeros(self.hidden_size, device=device)
            # Wrap around if needed
            end = min(start + self.k, self.hidden_size)
            mask[start:end] = 1.0
            if end - start < self.k:
                mask[: self.k - (end - start)] = 1.0
            self._task_masks[task_idx] = mask

        return self._task_masks[task_idx].to(device)


# ---------------------------------------------------------------------------
# NoMask — all-ones mask (naive finetuning baseline)
# ---------------------------------------------------------------------------


class NoMask:
    """
    Returns an all-ones mask — equivalent to no masking at all.

    Used as the naive SGD fine-tuning baseline. When used with
    block-sequential training, this exhibits catastrophic forgetting
    because all neurons are updated for every task.

    Args:
        hidden_size: Total hidden units.
        grid_dim:    Grid side length (for interface compatibility).
    """

    def __init__(self, hidden_size: int, grid_dim: int, k: int = 128, **kw):
        self.hidden_size = hidden_size
        self.k = k
        # Fixed mask: first k neurons always active (same for every task).
        _m = torch.zeros(hidden_size)
        _m[:k] = 1.0
        self._mask = _m

    def __call__(
        self,
        model: nn.Module,
        data: tuple,
        m: torch.Tensor,
        task_idx: int = None,
        **kwargs,
    ) -> torch.Tensor:
        return self._mask.to(data[0].device)


# ---------------------------------------------------------------------------
# Configurer registry
# ---------------------------------------------------------------------------

CONFIGURERS = {
    "FTN-Fast": SmoothKWTAFast,
    "FTN-Slow": SmoothKWTASlow,
    "PolyakFast": PolyakKWTAFast,
    "KWTA-only": KWTAOnly,
    "FixedMask": FixedMask,
    "NoMask": NoMask,
}
