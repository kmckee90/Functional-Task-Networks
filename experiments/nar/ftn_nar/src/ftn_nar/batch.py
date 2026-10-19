"""GraphBatch: flat-tensor batching of layered DAGs with exact labels."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import algorithms as A
from .generator import WEIGHT_TO_CATEGORY, Graph

_DEG_NORM = math.log1p(6.0)


@dataclass
class GraphBatch:
    # inputs
    node_features: torch.Tensor  # f32 [N, 3] (source flag, norm in-degree, norm out-degree)
    edge_index: torch.Tensor  # i64 [2, E] (u, v)
    edge_weight_scalar: torch.Tensor  # f32 [E] (w / 8)
    edge_weight_category: torch.Tensor  # i64 [E] (0..15)
    source_mask: torch.Tensor  # bool [N]
    graph_index_per_node: torch.Tensor  # i64 [N]
    graph_index_per_edge: torch.Tensor  # i64 [E]
    depth_per_graph: torch.Tensor  # i64 [G]
    # labels (present when with_labels)
    final_values: torch.Tensor | None  # f32 [N]
    trace_values: torch.Tensor | None  # f32 [T+1, N]
    trace_reached: torch.Tensor | None  # bool [T+1, N]
    valid_pred_edge_mask: torch.Tensor | None  # bool [E]
    # bookkeeping
    graph_seeds: list  # diagnostics + disjointness assertions
    algorithm_id: int | None  # diagnostics ONLY; never an input to model/configurer
    num_graphs: int
    max_depth: int

    @property
    def num_nodes(self) -> int:
        return self.node_features.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def to(self, device) -> "GraphBatch":
        def mv(x):
            return x.to(device, non_blocking=True) if torch.is_tensor(x) else x
        return GraphBatch(
            node_features=mv(self.node_features), edge_index=mv(self.edge_index),
            edge_weight_scalar=mv(self.edge_weight_scalar),
            edge_weight_category=mv(self.edge_weight_category),
            source_mask=mv(self.source_mask),
            graph_index_per_node=mv(self.graph_index_per_node),
            graph_index_per_edge=mv(self.graph_index_per_edge),
            depth_per_graph=mv(self.depth_per_graph),
            final_values=mv(self.final_values), trace_values=mv(self.trace_values),
            trace_reached=mv(self.trace_reached),
            valid_pred_edge_mask=mv(self.valid_pred_edge_mask),
            graph_seeds=self.graph_seeds, algorithm_id=self.algorithm_id,
            num_graphs=self.num_graphs, max_depth=self.max_depth,
        )


def build_batch(graphs: list[Graph], algorithm: int | None, with_labels: bool = True,
                with_trace: bool = True, labels: list[dict] | None = None) -> GraphBatch:
    """Concatenate graphs into one flat batch; labels from reference A unless given."""
    n_tot = sum(g.n for g in graphs)
    e_tot = sum(len(g.edges) for g in graphs)
    t_max = max(g.depth for g in graphs)

    x = np.zeros((n_tot, 3), dtype=np.float32)
    ei = np.zeros((2, e_tot), dtype=np.int64)
    ws = np.zeros(e_tot, dtype=np.float32)
    wc = np.zeros(e_tot, dtype=np.int64)
    src_mask = np.zeros(n_tot, dtype=bool)
    gin = np.zeros(n_tot, dtype=np.int64)
    gie = np.zeros(e_tot, dtype=np.int64)
    depths = np.array([g.depth for g in graphs], dtype=np.int64)

    fv = np.zeros(n_tot, dtype=np.float32) if with_labels else None
    tv = np.zeros((t_max + 1, n_tot), dtype=np.float32) if with_labels and with_trace else None
    tr = np.zeros((t_max + 1, n_tot), dtype=bool) if with_labels and with_trace else None
    vp = np.zeros(e_tot, dtype=bool) if with_labels else None

    n_off = e_off = 0
    for gi, g in enumerate(graphs):
        nn, ne = g.n, len(g.edges)
        in_deg = np.bincount(g.edges[:, 1], minlength=nn)
        out_deg = np.bincount(g.edges[:, 0], minlength=nn)
        x[n_off:n_off + nn, 0] = 0.0
        x[n_off + g.source, 0] = 1.0
        x[n_off:n_off + nn, 1] = np.log1p(in_deg) / _DEG_NORM
        x[n_off:n_off + nn, 2] = np.log1p(out_deg) / _DEG_NORM
        ei[0, e_off:e_off + ne] = g.edges[:, 0] + n_off
        ei[1, e_off:e_off + ne] = g.edges[:, 1] + n_off
        ws[e_off:e_off + ne] = g.w / 8.0
        wc[e_off:e_off + ne] = [WEIGHT_TO_CATEGORY[int(w)] for w in g.w]
        src_mask[n_off + g.source] = True
        gin[n_off:n_off + nn] = gi
        gie[e_off:e_off + ne] = gi

        if with_labels:
            lab = labels[gi] if labels is not None else A.execute(g, algorithm)
            fv[n_off:n_off + nn] = lab["final_values"]
            vp[e_off:e_off + ne] = lab["valid_pred_edge_mask"]
            if with_trace:
                sv, sr = lab["step_values"], lab["step_reached"]
                d = g.depth
                tv[:d + 1, n_off:n_off + nn] = sv
                tr[:d + 1, n_off:n_off + nn] = sr
                if d < t_max:  # hold final state; loss masks out t > depth anyway
                    tv[d + 1:, n_off:n_off + nn] = sv[d]
                    tr[d + 1:, n_off:n_off + nn] = sr[d]
        n_off += nn
        e_off += ne

    t = torch.from_numpy
    return GraphBatch(
        node_features=t(x), edge_index=t(ei), edge_weight_scalar=t(ws),
        edge_weight_category=t(wc), source_mask=t(src_mask),
        graph_index_per_node=t(gin), graph_index_per_edge=t(gie),
        depth_per_graph=t(depths),
        final_values=t(fv) if fv is not None else None,
        trace_values=t(tv) if tv is not None else None,
        trace_reached=t(tr) if tr is not None else None,
        valid_pred_edge_mask=t(vp) if vp is not None else None,
        graph_seeds=[g.seed for g in graphs], algorithm_id=algorithm,
        num_graphs=len(graphs), max_depth=t_max,
    )
