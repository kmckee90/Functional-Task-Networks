"""2-D grid helpers for the atom bank + lateral smoothing.

The atom bank is laid out as a grid_dim x grid_dim square so the mask renders
as a clean image and active atoms can be encouraged to form spatially coherent
blobs (purely for demo legibility). Ported / trimmed from
func_net_spatial/masks.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    """Binary mask with 1s at the top-k entries of x, 0s elsewhere."""
    idx = torch.topk(x, k).indices
    out = torch.zeros_like(x)
    out[idx] = 1.0
    return out


def mask_to_grid(m: torch.Tensor, grid_dim: int) -> torch.Tensor:
    """Flat (grid_dim**2,) -> (grid_dim, grid_dim)."""
    return m.reshape(grid_dim, grid_dim)


def mask_to_vec(grid: torch.Tensor) -> torch.Tensor:
    """(grid_dim, grid_dim) -> flat (grid_dim**2,)."""
    return grid.flatten()


class LateralSmoother:
    """Uniform conv smoothing on the grid, used between reconfig and final kwta
    to pull active atoms into contiguous blobs.

    Not an nn.Module we train — the kernel is fixed to a box filter. We keep it
    as a plain object so it can sit on the configurer without showing up in any
    optimizer.
    """

    def __init__(self, grid_dim: int, kernel_size: int = 3, torus: bool = True):
        self.grid_dim = grid_dim
        self.conv = nn.Conv2d(
            1, 1, kernel_size=kernel_size, stride=1,
            padding=kernel_size // 2, bias=False,
            padding_mode="circular" if torus else "zeros",
        ).requires_grad_(False)
        self.conv.weight.data.fill_(1.0)

    def step(self, m: torch.Tensor) -> torch.Tensor:
        """One smoothing pass on a flat mask -> flat smoothed values."""
        g = mask_to_grid(m, self.grid_dim).unsqueeze(0).unsqueeze(0)
        g = self.conv(g).squeeze(0).squeeze(0)
        return mask_to_vec(g)
