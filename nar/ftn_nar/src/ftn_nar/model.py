"""Recurrent layered-DAG processors: FTN atom-bank model, oracle variant, dense baseline.

Precision note: the design doc suggests bf16 autocast for neural ops. We run fp32
throughout: the model is ~2-5M parameters, activations are tiny, and exact rounded
integer readout at OOD depth (values up to +-256 of a +-1024 range) leaves less than
bf16's 2^-9 relative precision to spare. fp32 registers/masks/losses are as specified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ModelCfg
from . import ops
from .algorithms import CLIP_BOUND


@dataclass
class Program:
    """Task-conditioning for one forward pass.

    FTN/oracle: gate_ext/gate_red are f32 [H] or [G, H] (per-graph, e.g. replay).
    Dense: z is f32 [latent] or [G, latent].
    """
    gate_ext: torch.Tensor | None = None
    gate_red: torch.Tensor | None = None
    z: torch.Tensor | None = None


class AtomBank(nn.Module):
    """H independent MLPs Linear(in,w)+SiLU+Linear(w,out), evaluated jointly."""

    def __init__(self, h: int, in_dim: int, width: int, out_dim: int, scalar_out_dims: int = 1):
        super().__init__()
        self.h, self.in_dim, self.width, self.out_dim = h, in_dim, width, out_dim
        self.w1 = nn.Parameter(torch.empty(h, in_dim, width))
        self.b1 = nn.Parameter(torch.zeros(h, width))
        self.w2 = nn.Parameter(torch.empty(h, width, out_dim))
        self.b2 = nn.Parameter(torch.zeros(h, out_dim))
        with torch.no_grad():
            b1 = 1.0 / math.sqrt(in_dim)
            self.w1.uniform_(-b1, b1)
            b2 = 1.0 / math.sqrt(width)
            self.w2.uniform_(-b2, b2)
            # Final scalar output layers: N(0, 0.02); never exactly zero so the mask
            # optimizer receives gradient signal for every atom (design 7.9).
            self.w2[:, :, :scalar_out_dims].normal_(0.0, 0.02)

    def forward_full(self, x: torch.Tensor, gate: torch.Tensor, k: int,
                     gie: torch.Tensor | None = None) -> torch.Tensor:
        """x [E,in]; gate [H] or [G,H] (with gie [E]); returns (1/k) sum_j gate_j out_j."""
        h1 = torch.einsum("ei,hiw->ehw", x, self.w1) + self.b1
        h1 = torch.nn.functional.silu(h1)
        out = torch.einsum("ehw,hwo->eho", h1, self.w2) + self.b2  # [E,H,out]
        if gate.dim() == 1:
            comb = torch.einsum("eho,h->eo", out, gate)
        else:
            comb = torch.einsum("eho,eh->eo", out, gate[gie])
        return comb / k

    def forward_active(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Hard-mask fast path: only the k active atoms are evaluated."""
        w1, b1, w2, b2 = self.w1[idx], self.b1[idx], self.w2[idx], self.b2[idx]
        h1 = torch.nn.functional.silu(torch.einsum("ei,kiw->ekw", x, w1) + b1)
        out = torch.einsum("ekw,kwo->eko", h1, w2) + b2
        return out.mean(dim=1)


class DenseFiLM(nn.Module):
    """DENSE_ACTIVE_MATCHED replacement for a bank: one MLP, FiLM-modulated by z."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, z_dim: int, scalar_out_dims: int = 1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)
        self.film = nn.Linear(z_dim, 2 * hidden)
        with torch.no_grad():
            self.fc1.bias.zero_()
            self.fc2.bias.zero_()
            self.fc2.weight[:scalar_out_dims].normal_(0.0, 0.02)
            self.film.weight.normal_(0.0, 0.01)
            self.film.bias.zero_()

    def forward(self, x: torch.Tensor, z: torch.Tensor, gie: torch.Tensor | None) -> torch.Tensor:
        h1 = self.fc1(x)
        f = self.film(z)
        if z.dim() == 2:
            f = f[gie]
        s, b = f.chunk(2, dim=-1)
        h1 = h1 * (1.0 + s) + b
        return self.fc2(torch.nn.functional.silu(h1))


class DPModel(nn.Module):
    """Shared encode-process-decode recurrent executor (design section 7).

    kind='ftn': typed sparse extender/reducer atom banks gated by Program masks.
    kind='dense': dense FiLM-modulated MLPs conditioned on Program.z.
    """

    def __init__(self, cfg: ModelCfg, kind: str = "ftn"):
        super().__init__()
        self.cfg = cfg
        self.kind = kind
        self.round_register = getattr(cfg, "round_register", True)
        self.op_latent = getattr(cfg, "op_latent", True)
        self.extender_residual = getattr(cfg, "extender_residual", False)
        d, de = cfg.latent_dim, cfg.edge_embed_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(3, cfg.encoder_hidden), nn.SiLU(),
            nn.Linear(cfg.encoder_hidden, d), nn.LayerNorm(d))
        self.edge_embedding = nn.Embedding(16, de)
        ext_in = 3 + 2 * d + de
        red_in = 2 + 2 * d + de
        if kind == "ftn":
            self.ext_bank = AtomBank(cfg.extender_atoms, ext_in, cfg.atom_width, 1 + d)
            self.red_bank = AtomBank(cfg.reducer_atoms, red_in, cfg.atom_width, 1)
        elif kind == "dense":
            self.ext_dense = DenseFiLM(ext_in, cfg.dense_hidden, 1 + d, d)
            self.red_dense = DenseFiLM(red_in, cfg.dense_hidden, 1, d)
        else:
            raise ValueError(kind)
        self.in_proj = nn.Linear(d + 1, d)
        self.gru = nn.GRUCell(d, d)
        self.ln_h = nn.LayerNorm(d)
        # Value readout IS the scalar register (algorithmically aligned; a latent-decoded
        # tanh head cannot extrapolate to OOD value ranges and lets the optimizer bypass
        # the register entirely -- see README deviations).
        self.reached_head = nn.Linear(d, 1)
        with torch.no_grad():
            nn.init.orthogonal_(self.gru.weight_hh)
            for m in self.modules():
                if isinstance(m, (nn.Linear, nn.GRUCell)):
                    for name, p in m.named_parameters(recurse=False):
                        if "bias" in name:
                            p.zero_()

    # ----- bank dispatch -----

    def _extender(self, x, prog: Program, gie):
        if self.kind == "dense":
            return self.ext_dense(x, prog.z, gie)
        g = prog.gate_ext
        k = self.cfg.active_extender_atoms
        if g.dim() == 1 and not g.requires_grad and bool(((g == 0) | (g == 1)).all()):
            return self.ext_bank.forward_active(x, torch.nonzero(g > 0.5).flatten())
        return self.ext_bank.forward_full(x, g, k, gie)

    def _reducer(self, x, prog: Program, gie):
        if self.kind == "dense":
            return self.red_dense(x, prog.z, gie)
        g = prog.gate_red
        k = self.cfg.active_reducer_atoms
        if g.dim() == 1 and not g.requires_grad and bool(((g == 0) | (g == 1)).all()):
            return self.red_bank.forward_active(x, torch.nonzero(g > 0.5).flatten())
        return self.red_bank.forward_full(x, g, k, gie)

    # ----- forward -----

    def forward(self, batch, prog: Program, edge_mode: str = "ste",
                return_trace: bool = False) -> dict:
        """edge_mode: 'soft' (warmup), 'ste' (hard fwd / soft bwd), 'hard' (inference).
        The same Program is used at every recurrent step (masks are episode-fixed)."""
        cfg = self.cfg
        dev = batch.node_features.device
        n, e = batch.num_nodes, batch.num_edges
        u, v = batch.edge_index[0], batch.edge_index[1]
        gie = batch.graph_index_per_edge
        gin = batch.graph_index_per_node
        e_emb = self.edge_embedding(batch.edge_weight_category)
        ws = batch.edge_weight_scalar.unsqueeze(1)

        h = self.node_encoder(batch.node_features)
        r = torch.zeros(n, device=dev)
        reached = batch.source_mask.clone()

        t_max = int(batch.max_depth)
        depth_n = batch.depth_per_graph[gin]
        depth_e = batch.depth_per_graph[gie]

        h_final = torch.zeros_like(h)
        r_final = torch.zeros(n, device=dev)
        final_scores = torch.zeros(e, device=dev)
        final_edge_valid = torch.zeros(e, dtype=torch.bool, device=dev)
        trace_values, trace_reached_logits, trace_scores, trace_edge_valid = [], [], [], []

        # Lean operator: the DP scalar (candidate/score) is a function of (register,
        # weight) alone -- a+w, max(a,w), argmin over candidates -- so feeding h_u,h_v
        # into the scalar path only invites task/position shortcuts that don't
        # extrapolate OOD. When op_latent is False the latent slots are zeroed for the
        # banks (dims unchanged); latent still drives the GRU/pointer downstream.
        # NOTE: hu must be recomputed each step -- h is reassigned by the GRU update
        # below, so binding hu once before the loop would freeze the banks at the
        # initial encoder state.
        for t in range(1, t_max + 1):
            hu = h if self.op_latent else torch.zeros_like(h)
            src_reached = reached[u]
            ext_in = torch.cat([
                (r[u] / CLIP_BOUND).unsqueeze(1), ws, src_reached.float().unsqueeze(1),
                hu[u], hu[v], e_emb], dim=1)
            ext_out = self._extender(ext_in, prog, gie)
            q, msg = ext_out[:, 0], ext_out[:, 1:]
            # Residual candidate: c = r_u + delta. Every operator is r_u plus a
            # (register,weight)-local increment -- ADD:+w, SUB:-w, MAX_EDGE:max(0,w-a),
            # MIN_EDGE:min(0,w-a). For the additive ops delta is INDEPENDENT of the
            # register magnitude, so the atom extrapolates to any depth for free instead
            # of having to reproduce an ever-growing absolute value with fixed MLP
            # precision (the depth-extrapolation wall). Design 25 Case F: algorithmic
            # alignment of the recurrence.
            if self.extender_residual:
                c = r[u] + CLIP_BOUND * q
            else:
                c = CLIP_BOUND * q
            c_hard = c.clamp(-CLIP_BOUND, CLIP_BOUND)
            c_used = c_hard + (c - c.detach()) if c.requires_grad else c_hard

            red_in = torch.cat([
                (c_used / CLIP_BOUND).unsqueeze(1), (r[v] / CLIP_BOUND).unsqueeze(1),
                msg, hu[v], e_emb], dim=1)
            scores = self._reducer(red_in, prog, gie).squeeze(1)

            if edge_mode == "soft":
                p = ops.segment_softmax(scores, v, n, src_reached, tau=1.0)
            elif edge_mode == "ste":
                p_soft = ops.segment_softmax(scores, v, n, src_reached, tau=cfg.edge_temperature)
                p_hard = ops.segment_argmax_onehot(scores, v, n, src_reached)
                p = p_hard + (p_soft - p_soft.detach())
            elif edge_mode == "hard":
                p = ops.segment_argmax_onehot(scores, v, n, src_reached)
            else:
                raise ValueError(edge_mode)

            r_new = ops.segment_sum(p * c_used, v, n)
            if self.round_register:
                # DP node values are exact integers. Without quantization the continuous
                # register drifts sub-integer each step and the error compounds over
                # depth (layer 1 exact -> layer 6 ~0.8; OOD depth -> 0). Straight-through
                # rounding snaps each step back to the exact integer (fwd=round, bwd=id),
                # so a per-step error < 0.5 never accumulates. This is the executor
                # tightening the design's decision tree (Case B/F) calls for.
                r_new = r_new + (torch.round(r_new) - r_new).detach()
            a_new = ops.segment_sum_vec(p.unsqueeze(1) * msg, v, n)
            has_pred = ops.segment_any(src_reached, v, n)
            upd = has_pred & ~batch.source_mask

            x_in = self.in_proj(torch.cat([a_new, (r_new / CLIP_BOUND).unsqueeze(1)], dim=1))
            h_upd = self.ln_h(self.gru(x_in, h))
            h = torch.where(upd.unsqueeze(1), h_upd, h)
            r = torch.where(upd, r_new, r)
            reached = reached | has_pred
            r = r.masked_fill(batch.source_mask, 0.0)  # clamp source register

            fin_n = depth_n == t
            if fin_n.any():
                h_final = torch.where(fin_n.unsqueeze(1), h, h_final)
                r_final = torch.where(fin_n, r, r_final)
            fin_e = depth_e == t
            if fin_e.any():
                final_scores = torch.where(fin_e, scores, final_scores)
                final_edge_valid = final_edge_valid | (fin_e & src_reached)

            if return_trace:
                trace_values.append(r)
                trace_reached_logits.append(self.reached_head(h).squeeze(1))
                trace_scores.append(scores)
                trace_edge_valid.append(src_reached)

        final_values = r_final
        final_reached_logits = self.reached_head(h_final).squeeze(1)

        out = {
            "final_values": final_values,
            "final_reached_logits": final_reached_logits,
            "final_edge_scores": final_scores,
            "final_edge_valid": final_edge_valid,
            "final_register": r_final,
        }
        if return_trace:
            out["trace_values"] = torch.stack(trace_values)  # [T, N] steps 1..T
            out["trace_reached_logits"] = torch.stack(trace_reached_logits)
            out["trace_edge_scores"] = torch.stack(trace_scores)  # [T, E]
            out["trace_edge_valid"] = torch.stack(trace_edge_valid)
        return out


# ----- oracle composition -----

ORACLE_GROUP = 8  # atoms per labeled module == active path size


def oracle_model_cfg(cfg: ModelCfg) -> ModelCfg:
    """Explicitly labeled modules: 4 extenders x 8 atoms, 2 reducers x 8 atoms.
    Same per-module architecture and active parameter count as the FTN active path."""
    import dataclasses
    return dataclasses.replace(cfg, extender_atoms=4 * ORACLE_GROUP, reducer_atoms=2 * ORACLE_GROUP,
                               active_extender_atoms=ORACLE_GROUP, active_reducer_atoms=ORACLE_GROUP)


def oracle_masks(algorithm: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    from .algorithms import EXTENDER_OF, REDUCER_OF
    from .masks import mask_from_indices
    ext = EXTENDER_OF[algorithm]
    red = REDUCER_OF[algorithm]
    m_ext = mask_from_indices(4 * ORACLE_GROUP, range(ext * ORACLE_GROUP, (ext + 1) * ORACLE_GROUP), device)
    m_red = mask_from_indices(2 * ORACLE_GROUP, range(red * ORACLE_GROUP, (red + 1) * ORACLE_GROUP), device)
    return m_ext, m_red


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
