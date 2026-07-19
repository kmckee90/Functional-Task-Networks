"""The eight graph dynamic programs and exact reference implementation A.

Reference A is deliberately plain: python loops over layers, int64 arithmetic.
An independently written implementation lives in reference_b.py; the test suite
cross-checks them on >=10,000 random graphs.
"""

from __future__ import annotations

import numpy as np

CLIP_BOUND = 1024

EXTENDER_NAMES = ["ADD", "SUB", "MAX_EDGE", "MIN_EDGE"]
REDUCER_NAMES = ["MIN", "MAX"]

# Design doc section 3.6 ordering (ids 0..7).
ALGORITHM_NAMES = [
    "MIN_ADD",
    "MAX_ADD",
    "MIN_SUB",
    "MAX_SUB",
    "MIN_MAX_EDGE",
    "MAX_MAX_EDGE",
    "MIN_MIN_EDGE",
    "MAX_MIN_EDGE",
]
REDUCER_OF = [0, 1, 0, 1, 0, 1, 0, 1]  # 0=MIN, 1=MAX
EXTENDER_OF = [0, 0, 1, 1, 2, 2, 3, 3]  # 0=ADD, 1=SUB, 2=MAX_EDGE, 3=MIN_EDGE

ALL_ALGORITHMS = list(range(8))

EDGE_WEIGHT_VALUES = [-8, -7, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8]


def alg_id(name: str) -> int:
    return ALGORITHM_NAMES.index(name)


def clip(x: int) -> int:
    return max(-CLIP_BOUND, min(CLIP_BOUND, x))


def extend(ext: int, a: int, w: int) -> int:
    if ext == 0:  # ADD
        v = a + w
    elif ext == 1:  # SUB
        v = a - w
    elif ext == 2:  # MAX_EDGE
        v = max(a, w)
    elif ext == 3:  # MIN_EDGE
        v = min(a, w)
    else:
        raise ValueError(ext)
    return clip(v)


def reduce_(red: int, candidates: list[int]) -> int:
    return min(candidates) if red == 0 else max(candidates)


def execute(graph, algorithm: int) -> dict:
    """Exact synchronous execution with full trace.

    Returns dict with:
      final_values:  int64 [N]
      step_values:   int64 [D+1, N]  (unreached nodes hold 0)
      step_reached:  bool  [D+1, N]
      valid_pred_edge_mask: bool [E]  (final valid predecessor sets, per edge)
    Per-step valid predecessor sets are step_reached[t] & valid_pred_edge_mask
    restricted to edges into reached nodes (layering makes them constant once
    the destination is reached).
    """
    red = REDUCER_OF[algorithm]
    ext = EXTENDER_OF[algorithm]
    n, depth = graph.n, graph.depth
    layer_of = graph.layer_of
    edges = graph.edges
    w = graph.w

    incoming: list[list[int]] = [[] for _ in range(n)]
    for e in range(len(edges)):
        incoming[edges[e, 1]].append(e)

    d = [0] * n
    reached = [False] * n
    reached[graph.source] = True

    step_values = np.zeros((depth + 1, n), dtype=np.int64)
    step_reached = np.zeros((depth + 1, n), dtype=bool)
    step_reached[0, graph.source] = True

    nodes_in_layer: list[list[int]] = [[] for _ in range(depth + 1)]
    for v in range(n):
        nodes_in_layer[layer_of[v]].append(v)

    for t in range(1, depth + 1):
        for v in nodes_in_layer[t]:
            cands = []
            for e in incoming[v]:
                u = int(edges[e, 0])
                assert reached[u], "layered DAG invariant broken"
                cands.append(extend(ext, d[u], int(w[e])))
            d[v] = reduce_(red, cands)
            reached[v] = True
        step_reached[t] = step_reached[t - 1]
        step_reached[t, nodes_in_layer[t]] = True
        step_values[t] = step_values[t - 1]
        for v in nodes_in_layer[t]:
            step_values[t, v] = d[v]

    final_values = np.array(d, dtype=np.int64)
    valid = np.zeros(len(edges), dtype=bool)
    for e in range(len(edges)):
        u, v = int(edges[e, 0]), int(edges[e, 1])
        valid[e] = extend(ext, d[u], int(w[e])) == final_values[v]
    return {
        "final_values": final_values,
        "step_values": step_values,
        "step_reached": step_reached,
        "valid_pred_edge_mask": valid,
    }


def final_values(graph, algorithm: int) -> np.ndarray:
    return execute(graph, algorithm)["final_values"]
