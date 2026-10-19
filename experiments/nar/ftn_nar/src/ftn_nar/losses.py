"""Training and support losses (design sections 9.6, 10.4)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .algorithms import CLIP_BOUND
from .config import LossCfg
from . import ops

_B = float(CLIP_BOUND)
# SmoothL1 transition at 1/16 of a value unit. With beta = 1/1024 (one full unit on
# the /1024 scale) the quadratic region's vanishing gradients equilibrate training at
# MAE ~0.3/node -- just under the rounding threshold -- capping node accuracy at ~86%.
_BETA = 1.0 / (_B * 16.0)


def _value_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    el = F.smooth_l1_loss(pred / _B, target / _B, beta=_BETA, reduction="none")
    m = mask.to(el.dtype)
    return (el * m).sum() / m.sum().clamp_min(1.0)


def pointer_loss(scores: torch.Tensor, edge_valid: torch.Tensor, positives: torch.Tensor,
                 dst: torch.Tensor, n: int, node_mask: torch.Tensor) -> torch.Tensor:
    """Multi-positive NLL: -log sum_{u in P_v} softmax(scores_incoming(v))[u]."""
    p = ops.segment_softmax(scores, dst, n, edge_valid, tau=1.0)
    pos = ops.segment_sum(p * (positives & edge_valid).to(p.dtype), dst, n)
    has_valid = ops.segment_any(edge_valid, dst, n)
    m = (node_mask & has_valid).to(p.dtype)
    nll = -torch.log(pos + 1e-12)
    return (nll * m).sum() / m.sum().clamp_min(1.0)


def support_loss(out: dict, batch, w: LossCfg,
                 logits: list[torch.Tensor] | None = None,
                 extra: torch.Tensor | None = None) -> torch.Tensor:
    """Final outputs only; no trace supervision reaches the configurer."""
    nonsrc = ~batch.source_mask
    n = batch.num_nodes
    loss = w.support_value * _value_loss(out["final_values"], batch.final_values, nonsrc)
    loss = loss + w.support_pointer * pointer_loss(
        out["final_edge_scores"], out["final_edge_valid"], batch.valid_pred_edge_mask,
        batch.edge_index[1], n, nonsrc)
    if logits is not None:
        loss = loss + w.support_logit_l2 * sum(a.square().mean() for a in logits)
    if extra is not None:
        loss = loss + extra
    return loss


def model_training_loss(out: dict, batch, w: LossCfg, trace: bool = True) -> dict:
    """Query/replay loss with trace supervision. Source excluded from value and
    pointer losses; included in reached loss."""
    nonsrc = ~batch.source_mask
    n = batch.num_nodes
    dst = batch.edge_index[1]
    gin = batch.graph_index_per_node
    gie = batch.graph_index_per_edge

    total = {}
    total["final_value"] = _value_loss(out["final_values"], batch.final_values, nonsrc)
    total["final_pointer"] = pointer_loss(
        out["final_edge_scores"], out["final_edge_valid"], batch.valid_pred_edge_mask,
        dst, n, nonsrc)

    if trace:
        t_max = out["trace_values"].shape[0]
        steps = torch.arange(1, t_max + 1, device=out["trace_values"].device)
        step_live_n = steps.unsqueeze(1) <= batch.depth_per_graph[gin].unsqueeze(0)  # [T,N]
        tgt_v = batch.trace_values[1:]
        tgt_r = batch.trace_reached[1:]
        vmask = tgt_r & nonsrc.unsqueeze(0) & step_live_n
        total["trace_value"] = _value_loss(out["trace_values"], tgt_v, vmask)

        rmask = step_live_n.to(torch.float32)
        bce = F.binary_cross_entropy_with_logits(
            out["trace_reached_logits"], tgt_r.to(torch.float32), reduction="none")
        total["reached"] = (bce * rmask).sum() / rmask.sum().clamp_min(1.0)

        step_live_e = steps.unsqueeze(1) <= batch.depth_per_graph[gie].unsqueeze(0)  # [T,E]
        ptr_losses = []
        for t in range(t_max):
            node_mask = tgt_r[t] & nonsrc & step_live_n[t]
            if not bool(node_mask.any()):
                continue
            ptr_losses.append(pointer_loss(
                out["trace_edge_scores"][t], out["trace_edge_valid"][t] & step_live_e[t],
                batch.valid_pred_edge_mask, dst, n, node_mask))
        total["trace_pointer"] = (torch.stack(ptr_losses).mean() if ptr_losses
                                  else torch.zeros((), device=out["final_values"].device))
    else:
        z = torch.zeros((), device=out["final_values"].device)
        total["trace_value"] = total["reached"] = total["trace_pointer"] = z

    total["loss"] = (w.final_value * total["final_value"]
                     + w.trace_value * total["trace_value"]
                     + w.reached * total["reached"]
                     + w.final_pointer * total["final_pointer"]
                     + w.trace_pointer * total["trace_pointer"])
    return total
