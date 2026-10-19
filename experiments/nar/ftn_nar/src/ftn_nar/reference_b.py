"""Independent exact reference implementation B (vectorized numpy).

Written independently of algorithms.py on purpose: layer-parallel updates using
ufunc.at scatter reductions with +/- inf sentinels instead of per-node candidate
lists. Only shares the constant tables (operator ids / clip bound), not logic.
"""

from __future__ import annotations

import numpy as np

from .algorithms import CLIP_BOUND, EXTENDER_OF, REDUCER_OF

_POS = np.int64(2**62)
_NEG = np.int64(-(2**62))


def _extend_vec(ext: int, a: np.ndarray, w: np.ndarray) -> np.ndarray:
    if ext == 0:
        out = a + w
    elif ext == 1:
        out = a - w
    elif ext == 2:
        out = np.maximum(a, w)
    elif ext == 3:
        out = np.minimum(a, w)
    else:
        raise ValueError(ext)
    return np.clip(out, -CLIP_BOUND, CLIP_BOUND)


def execute_b(graph, algorithm: int) -> dict:
    red = REDUCER_OF[algorithm]
    ext = EXTENDER_OF[algorithm]
    n, depth = graph.n, graph.depth
    layer_of = np.asarray(graph.layer_of)
    u = graph.edges[:, 0].astype(np.int64)
    v = graph.edges[:, 1].astype(np.int64)
    w = graph.w.astype(np.int64)

    d = np.zeros(n, dtype=np.int64)
    step_values = np.zeros((depth + 1, n), dtype=np.int64)
    step_reached = layer_of[None, :] <= np.arange(depth + 1)[:, None]

    dst_layer = layer_of[v]
    for t in range(1, depth + 1):
        sel = dst_layer == t
        cand = _extend_vec(ext, d[u[sel]], w[sel])
        acc = np.full(n, _POS if red == 0 else _NEG, dtype=np.int64)
        if red == 0:
            np.minimum.at(acc, v[sel], cand)
        else:
            np.maximum.at(acc, v[sel], cand)
        this_layer = layer_of == t
        d = np.where(this_layer, acc, d)
        step_values[t] = np.where(step_reached[t], d, 0)

    valid = _extend_vec(ext, d[u], w) == d[v]
    return {
        "final_values": d,
        "step_values": step_values,
        "step_reached": step_reached,
        "valid_pred_edge_mask": valid,
    }
