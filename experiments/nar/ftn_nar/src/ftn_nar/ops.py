"""Segment (per-destination-node) operations over edge sets."""

from __future__ import annotations

import torch

_NEG = -1e30


def segment_amax(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = torch.full((size,), _NEG, dtype=src.dtype, device=src.device)
    out.scatter_reduce_(0, index, src, reduce="amax", include_self=True)
    return out


def segment_sum(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = torch.zeros(size, dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def segment_sum_vec(src: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = torch.zeros(size, src.shape[1], dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def segment_softmax(scores: torch.Tensor, index: torch.Tensor, size: int,
                    valid: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """Softmax over each destination node's valid incoming edges. Invalid edges get 0.
    Empty segments produce all-zero probabilities."""
    s = (scores / tau).masked_fill(~valid, _NEG)
    mx = segment_amax(s.detach(), index, size)
    z = torch.exp(s - mx.clamp_min(_NEG)[index]) * valid.to(s.dtype)
    den = segment_sum(z, index, size)
    return z / den.clamp_min(1e-30)[index]


def segment_argmax_onehot(scores: torch.Tensor, index: torch.Tensor, size: int,
                          valid: torch.Tensor) -> torch.Tensor:
    """Hard one-hot over each segment's valid edges; ties broken by lowest edge index.
    Empty segments produce all zeros. No gradients."""
    with torch.no_grad():
        s = scores.masked_fill(~valid, _NEG)
        mx = segment_amax(s, index, size)
        e = scores.shape[0]
        ar = torch.arange(e, device=scores.device)
        is_max = valid & (s >= mx[index])
        cand = torch.where(is_max, ar, torch.full_like(ar, e))
        first = torch.full((size,), e, dtype=torch.long, device=scores.device)
        first.scatter_reduce_(0, index, cand, reduce="amin", include_self=True)
        return (ar == first[index]).to(scores.dtype)


def segment_any(src_bool: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = torch.zeros(size, dtype=torch.long, device=src_bool.device)
    out.index_add_(0, index, src_bool.long())
    return out > 0
