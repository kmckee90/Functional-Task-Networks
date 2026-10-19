"""Configurer freeze/reproducibility, support selection, replay filtering (design 22)."""

import hashlib

import numpy as np
import torch

from ftn_nar import algorithms as A
from ftn_nar import generator as G
from ftn_nar import support as SUP
from ftn_nar.batch import build_batch
from ftn_nar.config import ExperimentCfg, tiny_model
from ftn_nar.configurer import configure, configure_with_restarts
from ftn_nar.model import DPModel
from ftn_nar.replay import ReplayBuffer, ReplayItem

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _param_hash(model) -> str:
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _setup():
    cfg = ExperimentCfg()
    cfg.model = tiny_model()
    torch.manual_seed(7)
    model = DPModel(cfg.model).to(DEV)
    graphs = [G.generate("train", 0, "cfg_test", i) for i in range(8)]
    batch = build_batch(graphs, 2, with_trace=False).to(DEV)
    return cfg, model, batch


def test_configuration_freezes_weights():
    cfg, model, batch = _setup()
    before = _param_hash(model)
    res = configure(model, batch, cfg, (0, "freeze"))
    assert _param_hash(model) == before, "model parameters changed during configuration"
    assert all(p.requires_grad for p in model.parameters()), "requires_grad not restored"
    me, mr = res.masks_np()
    assert int(me.sum()) == cfg.model.active_extender_atoms
    assert int(mr.sum()) == cfg.model.active_reducer_atoms


def test_configurer_reproducible_restarts():
    cfg, model, batch = _setup()
    r1 = configure(model, batch, cfg, (0, "rep", 3))
    r2 = configure(model, batch, cfg, (0, "rep", 3))
    np.testing.assert_array_equal(r1.masks_np()[0], r2.masks_np()[0])
    np.testing.assert_array_equal(r1.masks_np()[1], r2.masks_np()[1])
    assert abs(r1.selection_loss - r2.selection_loss) < 1e-4


def test_configurer_improves_support_loss():
    cfg, model, batch = _setup()
    best, allr = configure_with_restarts(model, batch, cfg, (0, "improve"), 2)
    assert best.selection_loss == min(r.selection_loss for r in allr)


def test_support_identifiability():
    fam = A.ALL_ALGORITHMS
    for target in fam:
        graphs = SUP.greedy_identifiable_support(0, target, target, fam, support_id=0,
                                                 size=16, pool_size=128)
        assert len(graphs) == 16
        assert SUP.is_identifiable(graphs, target, fam)
        assert SUP.consistent_algorithms(graphs, target, fam) == [target]


def test_replay_overlap_filter_and_reservoir():
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(capacity=64, rng=rng)
    h, k = 16, 4

    def mask(idx):
        m = np.zeros(h, dtype=np.float32)
        m[list(idx)] = 1
        return m

    cur_e, cur_r = mask(range(4)), mask(range(4))
    # overlap 1.0 (same), 0.5 (2 shared), 0.0 (disjoint)
    for i, idx in enumerate([range(4), (0, 1, 4, 5), range(8, 12)]):
        buf.insert(ReplayItem(graph=None, labels={}, mask_ext=mask(idx), mask_red=mask(idx),
                              z=None, algorithm_id=i, graph_seed=i, insertion_step=0))
    elig = buf.eligible(cur_e, cur_r, k, k, 0.20, 0.80)
    assert len(elig) == 1 and elig[0].algorithm_id == 1  # only the 0.5-overlap item
    assert len(buf.sample(cur_e, cur_r, k, k, 0.20, 0.80, 16)) == 1
    assert len(buf.sample(cur_e, cur_r, k, k, 0.20, 0.80, 16, uniform=True)) == 3

    # reservoir: capacity respected, all items have a chance
    buf2 = ReplayBuffer(capacity=10, rng=np.random.default_rng(1))
    for i in range(1000):
        buf2.insert(ReplayItem(None, {}, mask(range(4)), mask(range(4)), None, 0, i, i))
    assert len(buf2) == 10
    assert buf2.n_seen == 1000
    steps = [it.insertion_step for it in buf2.items]
    assert max(steps) > 500  # late items represented


def test_support_query_disjoint_streams():
    sup = [G.generate("train", 0, "train_sup", 0, 5, i) for i in range(16)]
    qry = [G.generate("train", 0, "train_qry", 0, 5, i) for i in range(32)]
    assert not ({g.seed for g in sup} & {g.seed for g in qry})
