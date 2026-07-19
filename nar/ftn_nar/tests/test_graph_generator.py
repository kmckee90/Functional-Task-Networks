"""Generator structural invariants + determinism (design 4, 22.1)."""

import numpy as np

from ftn_nar import generator as G


def _check_structure(g, n_min=None, n_max=None, d_min=None, d_max=None):
    assert g.layer_of[g.source] == 0
    assert (g.layer_of == 0).sum() == 1
    # edges strictly adjacent-layer
    lu, lv = g.layer_of[g.edges[:, 0]], g.layer_of[g.edges[:, 1]]
    assert (lv == lu + 1).all()
    # every non-source layer nonempty; every non-source node has an incoming edge
    for t in range(g.depth + 1):
        assert (g.layer_of == t).any()
    in_deg = np.bincount(g.edges[:, 1], minlength=g.n)
    assert (in_deg[g.layer_of > 0] >= 1).all()
    # sampled in-degree is capped at 6; the outgoing-edge repair may rarely exceed it
    assert (in_deg > 6).mean() < 0.02
    # every node before final layer has an outgoing edge
    out_deg = np.bincount(g.edges[:, 0], minlength=g.n)
    assert (out_deg[g.layer_of < g.depth] >= 1).all()
    # weights nonzero integers in [-8, 8]
    assert set(np.unique(g.w)).issubset(set(range(-8, 9)) - {0})
    if n_min:
        assert n_min <= g.n <= n_max
    if d_min:
        assert d_min <= g.depth <= d_max
    # reachability from source (layered + in-degree>=1 implies it, verify anyway)
    reach = {g.source}
    for t in range(1, g.depth + 1):
        for e in range(len(g.edges)):
            if int(g.edges[e, 0]) in reach and g.layer_of[g.edges[e, 1]] == t:
                reach.add(int(g.edges[e, 1]))
    assert len(reach) == g.n


def test_train_distribution():
    for i in range(300):
        g = G.generate("train", 0, "gtest", i)
        _check_structure(g, 8, 16, 3, 6)


def test_ood_distributions():
    for i in range(30):
        for n in (32, 64, 128):
            _check_structure(G.generate("ood_width", 0, "gtest_w", n, i, n_target=n),
                             n, n, 3, 6)
            g = G.generate("ood_both", 0, "gtest_b", n, i, n_target=n)
            _check_structure(g, n, n)
            assert g.depth >= 7
        for d in (8, 16, 32):
            g = G.generate("ood_depth", 0, "gtest_d", d, i, d_target=d)
            _check_structure(g)
            assert g.depth == d
            widths = np.bincount(g.layer_of)
            assert (widths[1:] >= 2).all() and (widths[1:] <= 4).all()


def test_determinism():
    a = G.generate("train", 5, "det", 42)
    b = G.generate("train", 5, "det", 42)
    assert a.seed == b.seed and a.n == b.n and a.depth == b.depth
    np.testing.assert_array_equal(a.edges, b.edges)
    np.testing.assert_array_equal(a.w, b.w)
    np.testing.assert_array_equal(a.layer_of, b.layer_of)
    c = G.generate("train", 5, "det", 43)
    assert c.seed != a.seed
