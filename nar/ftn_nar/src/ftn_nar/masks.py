"""Program masks: hard top-k, straight-through relaxation, overlap utilities."""

from __future__ import annotations

import numpy as np
import torch


def hard_topk(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Binary top-k mask (f32, no grad)."""
    idx = torch.topk(logits.detach(), k).indices
    m = torch.zeros_like(logits, requires_grad=False)
    m = m.detach().clone()
    m[idx] = 1.0
    return m


def ste_topk_gate(logits: torch.Tensor, k: int, tau: float) -> torch.Tensor:
    """Forward value is EXACTLY the hard top-k mask; gradient flows through
    k * softmax(logits / tau). Both sum to k, keeping atom mixture scale consistent."""
    soft = k * torch.softmax(logits / tau, dim=-1)
    hard = hard_topk(logits, k)
    return hard + (soft - soft.detach())  # parens: forward is bitwise-exactly hard


def soft_gate(logits: torch.Tensor, k: int) -> torch.Tensor:
    """SOFT_MIXTURE gate: dense k * softmax, no hard projection."""
    return k * torch.softmax(logits, dim=-1)


def mask_from_indices(h: int, idx, device=None) -> torch.Tensor:
    m = torch.zeros(h, dtype=torch.float32, device=device)
    m[list(idx)] = 1.0
    return m


def random_topk_mask(h: int, k: int, rng: np.random.Generator, device=None) -> torch.Tensor:
    return mask_from_indices(h, rng.choice(h, size=k, replace=False), device)


def active_indices(mask: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(mask > 0.5).flatten()


def overlap(m_a, m_b, k: int) -> float:
    a = np.asarray(m_a, dtype=np.float64)
    b = np.asarray(m_b, dtype=np.float64)
    return float(a @ b) / k


def combined_overlap(ext_a, ext_b, red_a, red_b, k_ext: int, k_red: int) -> float:
    return 0.5 * overlap(ext_a, ext_b, k_ext) + 0.5 * overlap(red_a, red_b, k_red)


def hamming(m_a, m_b) -> int:
    return int(np.count_nonzero(np.asarray(m_a) != np.asarray(m_b)))


def medoid(masks: list[np.ndarray]) -> np.ndarray:
    """Mask minimizing total Hamming distance to the others (first on ties)."""
    dists = [sum(hamming(m, o) for o in masks) for m in masks]
    return masks[int(np.argmin(dists))]
