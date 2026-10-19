"""Layered-DAG generator (design doc section 4).

Each instance is generated deterministically from a seed-key tuple; the
resulting Graph carries a 64-bit seed identity used for support/query
disjointness assertions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .algorithms import EDGE_WEIGHT_VALUES
from . import seeds as S

WEIGHT_TO_CATEGORY = {w: i for i, w in enumerate(EDGE_WEIGHT_VALUES)}


@dataclass
class Graph:
    n: int
    depth: int
    source: int
    layer_of: np.ndarray  # int64 [N]
    edges: np.ndarray  # int64 [E, 2] (u, v), u in layer i, v in layer i+1
    w: np.ndarray  # int64 [E]
    seed: int  # 64-bit identity of the generating seed-key
    meta: dict = field(default_factory=dict)


def _alloc_widths(rng: np.random.Generator, n: int, depth: int, prefer_max: int = 4,
                  prefer_min_two: bool = False) -> np.ndarray:
    """Allocate n-1 nodes across layers 1..depth, >=1 each, preferring widths in
    [2, prefer_max]; overflow beyond the cap is spread uniformly."""
    widths = np.ones(depth, dtype=np.int64)
    rem = n - 1 - depth
    assert rem >= 0, f"n={n} too small for depth={depth}"
    if prefer_min_two:
        order = rng.permutation(depth)
        for i in order:
            if rem <= 0:
                break
            widths[i] += 1
            rem -= 1
    while rem > 0:
        under = np.flatnonzero(widths < prefer_max)
        i = rng.choice(under) if len(under) > 0 else rng.integers(0, depth)
        widths[i] += 1
        rem -= 1
    return widths


def _build_graph(rng: np.random.Generator, widths: np.ndarray, seed: int, meta: dict) -> Graph:
    depth = len(widths)
    n = int(1 + widths.sum())
    layer_of = np.zeros(n, dtype=np.int64)
    layers: list[np.ndarray] = [np.array([0])]
    nxt = 1
    for i, wd in enumerate(widths, start=1):
        ids = np.arange(nxt, nxt + wd)
        layers.append(ids)
        layer_of[ids] = i
        nxt += wd

    src_list: list[int] = []
    dst_list: list[int] = []
    for i in range(1, depth + 1):
        prev = layers[i - 1]
        for v in layers[i]:
            q = int(rng.integers(1, min(6, len(prev)) + 1))
            preds = rng.choice(prev, size=q, replace=False)
            src_list.extend(int(u) for u in preds)
            dst_list.extend([int(v)] * q)

    # Ensure every node in layers 0..depth-1 has an outgoing edge. Prefer targets
    # whose in-degree is still below 6; a wide layer feeding a narrow one can
    # nevertheless exceed 6 (inherent to the repair rule).
    has_out = np.zeros(n, dtype=bool)
    has_out[src_list] = True
    in_deg = np.zeros(n, dtype=np.int64)
    np.add.at(in_deg, dst_list, 1)
    for i in range(depth):
        for u in layers[i]:
            if not has_out[u]:
                nxt = layers[i + 1]
                under = nxt[in_deg[nxt] < 6]
                v = int(rng.choice(under if len(under) else nxt))
                src_list.append(int(u))
                dst_list.append(v)
                in_deg[v] += 1
                has_out[u] = True

    edges = np.stack([np.array(src_list, dtype=np.int64), np.array(dst_list, dtype=np.int64)], axis=1)
    w = rng.choice(np.array(EDGE_WEIGHT_VALUES, dtype=np.int64), size=len(edges), replace=True)

    # Random node relabeling; retain layer membership, source identity, edges.
    perm = rng.permutation(n)  # perm[old] = new
    new_layer = np.empty(n, dtype=np.int64)
    new_layer[perm] = layer_of
    new_edges = perm[edges]
    return Graph(n=n, depth=depth, source=int(perm[0]), layer_of=new_layer,
                 edges=new_edges, w=w.astype(np.int64), seed=seed, meta=meta)


def generate(dist: str, *key, n_target: int | None = None, d_target: int | None = None) -> Graph:
    """Generate one graph. `dist` in {train, ood_width, ood_depth, ood_both}.

    key: seed-key tuple (run_seed, stream, index, ...). For ood distributions
    n_target / d_target select the condition, e.g. ood_width with n_target=64.
    """
    rng = S.rng(*key)
    sid = S.seed_id(*key)
    if dist == "train":
        n = int(rng.integers(8, 17))
        d = int(rng.integers(3, min(6, n - 1) + 1))
        widths = _alloc_widths(rng, n, d)
    elif dist == "ood_width":
        assert n_target in (32, 64, 128)
        n = n_target
        d = int(rng.integers(3, 7))
        widths = _alloc_widths(rng, n, d)
    elif dist == "ood_depth":
        assert d_target in (8, 16, 32)
        d = d_target
        widths = rng.integers(2, 5, size=d).astype(np.int64)
        n = int(1 + widths.sum())
    elif dist == "ood_both":
        assert n_target in (32, 64, 128)
        n = n_target
        lo, hi = {32: (7, 9), 64: (14, 18), 128: (28, 32)}[n_target]
        d = int(rng.integers(lo, hi + 1))
        widths = _alloc_widths(rng, n, d, prefer_min_two=True)
    else:
        raise ValueError(dist)
    meta = {"dist": dist, "n_target": n_target, "d_target": d_target}
    return _build_graph(rng, widths, sid, meta)


def generate_many(dist: str, run_seed: int, stream: str, count: int, start: int = 0, **kw) -> list[Graph]:
    return [generate(dist, run_seed, stream, i, **kw) for i in range(start, start + count)]
