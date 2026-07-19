"""Reference cross-checks (design 3.8, 22.1): two independent implementations must
agree on >=10,000 random small graphs, all eight algorithms, full traces."""

import numpy as np
import pytest

from ftn_nar import algorithms as A
from ftn_nar import generator as G
from ftn_nar.reference_b import execute_b


def test_extenders_and_clip():
    assert A.extend(0, 5, 3) == 8
    assert A.extend(0, 1020, 8) == 1024  # clip
    assert A.extend(0, -1020, -8) == -1024
    assert A.extend(1, 5, 3) == 2
    assert A.extend(1, -1020, 8) == -1024
    assert A.extend(2, 5, 3) == 5
    assert A.extend(2, 2, 7) == 7
    assert A.extend(3, 5, 3) == 3
    assert A.extend(3, -4, 3) == -4


def test_reducers():
    assert A.reduce_(0, [3, -1, 7]) == -1
    assert A.reduce_(1, [3, -1, 7]) == 7


def test_hand_example():
    # source -> a (w=2), source -> b (w=-3), a -> c (w=1), b -> c (w=5)
    g = G.Graph(n=4, depth=2, source=0,
                layer_of=np.array([0, 1, 1, 2]),
                edges=np.array([[0, 1], [0, 2], [1, 3], [2, 3]]),
                w=np.array([2, -3, 1, 5]), seed=0)
    # MIN_ADD: d(a)=2, d(b)=-3, d(c)=min(2+1, -3+5)=2
    out = A.execute(g, A.alg_id("MIN_ADD"))
    assert out["final_values"].tolist() == [0, 2, -3, 2]
    # both predecessors of c are valid (tie 3 vs 2? no: 3 vs 2 -> only b path=2)
    assert out["valid_pred_edge_mask"].tolist() == [True, True, False, True]
    # MAX_ADD: d(c)=max(3, 2)=3
    out = A.execute(g, A.alg_id("MAX_ADD"))
    assert out["final_values"].tolist() == [0, 2, -3, 3]
    # MIN_MAX_EDGE: d(a)=max(0,2)=2, d(b)=max(0,-3)=0, d(c)=min(max(2,1),max(0,5))=2
    out = A.execute(g, A.alg_id("MIN_MAX_EDGE"))
    assert out["final_values"].tolist() == [0, 2, 0, 2]


def test_trace_invariants():
    for i in range(50):
        g = G.generate("train", 7, "trace_test", i)
        for alg in A.ALL_ALGORITHMS:
            out = A.execute(g, alg)
            sr, sv = out["step_reached"], out["step_values"]
            assert sr[0].sum() == 1 and sr[0][g.source]
            assert sr[g.depth].all()
            for t in range(g.depth + 1):
                np.testing.assert_array_equal(sr[t], g.layer_of <= t)
                # reached nodes hold final value; unreached hold zero
                np.testing.assert_array_equal(sv[t][sr[t]], out["final_values"][sr[t]])
                assert (sv[t][~sr[t]] == 0).all()
            # every non-source node has at least one valid predecessor
            has_valid = np.zeros(g.n, dtype=bool)
            has_valid[g.edges[out["valid_pred_edge_mask"], 1]] = True
            assert has_valid[g.layer_of > 0].all()


@pytest.mark.parametrize("chunk", range(10))
def test_cross_check_10k(chunk):
    """10,000 graphs x 8 algorithms against the independent implementation."""
    for i in range(1000):
        g = G.generate("train", 1234, "xcheck", chunk * 1000 + i)
        for alg in A.ALL_ALGORITHMS:
            a = A.execute(g, alg)
            b = execute_b(g, alg)
            np.testing.assert_array_equal(a["final_values"], b["final_values"])
            np.testing.assert_array_equal(a["step_values"], b["step_values"])
            np.testing.assert_array_equal(a["step_reached"], b["step_reached"])
            np.testing.assert_array_equal(a["valid_pred_edge_mask"], b["valid_pred_edge_mask"])


def test_cross_check_ood():
    """The two references must also agree on OOD-sized graphs."""
    cases = ([("ood_width", {"n_target": n}) for n in (32, 64, 128)]
             + [("ood_depth", {"d_target": d}) for d in (8, 16, 32)]
             + [("ood_both", {"n_target": n}) for n in (32, 64, 128)])
    for dist, kw in cases:
        for i in range(30):
            g = G.generate(dist, 99, "xcheck_ood", dist, str(kw), i, **kw)
            for alg in A.ALL_ALGORITHMS:
                np.testing.assert_array_equal(A.execute(g, alg)["final_values"],
                                              execute_b(g, alg)["final_values"])
