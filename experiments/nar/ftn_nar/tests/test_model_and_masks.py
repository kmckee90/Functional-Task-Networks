"""Model + mask mechanics: cardinality, STE equality, permutation invariance,
mask-logit gradients, single-batch overfit (design 22.1, 22.2)."""

import numpy as np
import pytest
import torch

from ftn_nar import generator as G
from ftn_nar.batch import build_batch
from ftn_nar.config import ExperimentCfg, tiny_model
from ftn_nar.losses import model_training_loss, support_loss
from ftn_nar.masks import hard_topk, mask_from_indices, random_topk_mask, ste_topk_gate
from ftn_nar.model import DPModel, Program, oracle_masks, oracle_model_cfg

torch.manual_seed(0)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _batch(n=6, alg=0, seed=0, stream="mtest", trace=True):
    graphs = [G.generate("train", seed, stream, i) for i in range(n)]
    return build_batch(graphs, alg, with_trace=trace).to(DEV)


def _model(cfg=None):
    torch.manual_seed(1234)
    return DPModel(cfg or tiny_model()).to(DEV)


def _rand_prog(mcfg, seed=0):
    rng = np.random.default_rng(seed)
    return Program(
        gate_ext=random_topk_mask(mcfg.extender_atoms, mcfg.active_extender_atoms, rng, DEV),
        gate_red=random_topk_mask(mcfg.reducer_atoms, mcfg.active_reducer_atoms, rng, DEV))


def test_topk_cardinality_and_ste_forward_equality():
    for k in (1, 2, 4, 8, 16):
        logits = torch.randn(64, requires_grad=True)
        hard = hard_topk(logits, k)
        assert int(hard.sum()) == k
        for tau in (2.0, 0.7, 0.2):
            gate = ste_topk_gate(logits, k, tau)
            assert torch.equal(gate.detach(), hard), "STE forward must equal hard mask"
            gate.sum().backward()
            assert logits.grad is not None
            logits.grad = None


def test_mask_logit_gradients_nonzero_at_init():
    """Every atom's logit must receive gradient signal at initialization (design 7.9)."""
    model = _model()
    b = _batch(trace=False)
    mcfg = model.cfg
    a_ext = torch.zeros(mcfg.extender_atoms, requires_grad=True, device=DEV)
    a_red = torch.zeros(mcfg.reducer_atoms, requires_grad=True, device=DEV)
    prog = Program(gate_ext=ste_topk_gate(a_ext, mcfg.active_extender_atoms, 2.0),
                   gate_red=ste_topk_gate(a_red, mcfg.active_reducer_atoms, 2.0))
    out = model(b, prog, edge_mode="ste", return_trace=False)
    loss = support_loss(out, b, ExperimentCfg().loss)
    loss.backward()
    assert (a_ext.grad != 0).all(), "some extender logit got zero gradient"
    assert (a_red.grad != 0).all(), "some reducer logit got zero gradient"


def test_hard_gate_full_vs_active_paths_agree():
    model = _model()
    b = _batch(trace=False)
    prog = _rand_prog(model.cfg)
    out_active = model(b, prog, edge_mode="hard")
    # force the full-compute path by making the gates per-graph
    gext = prog.gate_ext.unsqueeze(0).expand(b.num_graphs, -1).contiguous()
    gred = prog.gate_red.unsqueeze(0).expand(b.num_graphs, -1).contiguous()
    out_full = model(b, Program(gate_ext=gext, gate_red=gred), edge_mode="hard")
    torch.testing.assert_close(out_active["final_values"], out_full["final_values"],
                               rtol=1e-4, atol=1e-3)


def test_permutation_invariance():
    g = G.generate("train", 3, "perm", 0)
    rng = np.random.default_rng(0)
    perm = rng.permutation(g.n)
    g2 = G.Graph(n=g.n, depth=g.depth, source=int(perm[g.source]),
                 layer_of=np.empty_like(g.layer_of), edges=perm[g.edges],
                 w=g.w.copy(), seed=g.seed)
    g2.layer_of[perm] = g.layer_of
    model = _model()
    prog = _rand_prog(model.cfg)
    o1 = model(build_batch([g], 0).to(DEV), prog, edge_mode="hard")
    o2 = model(build_batch([g2], 0).to(DEV), prog, edge_mode="hard")
    v1 = o1["final_values"].detach().cpu().numpy()
    v2 = o2["final_values"].detach().cpu().numpy()
    np.testing.assert_allclose(v2[perm], v1, rtol=1e-4, atol=1e-3)


def test_masks_fixed_across_steps_and_no_nans():
    model = _model()
    for dist, kw in (("ood_depth", {"d_target": 32}), ("ood_width", {"n_target": 128})):
        graphs = [G.generate(dist, 11, "nan_test", dist, i, **kw) for i in range(4)]
        b = build_batch(graphs, 4, with_trace=False).to(DEV)
        out = model(b, _rand_prog(model.cfg), edge_mode="hard")
        assert torch.isfinite(out["final_values"]).all()
        assert out["final_values"].abs().max() <= 1024.0


def test_multi_positive_pointer_loss():
    from ftn_nar.losses import pointer_loss
    # 1 node with 3 incoming edges, two of them valid predecessors
    scores = torch.tensor([1.0, 1.0, 1.0])
    dst = torch.tensor([0, 0, 0])
    valid = torch.ones(3, dtype=torch.bool)
    pos = torch.tensor([True, True, False])
    l = pointer_loss(scores, valid, pos, dst, 1, torch.ones(1, dtype=torch.bool))
    assert abs(float(l) + np.log(2.0 / 3.0)) < 1e-5


def test_single_batch_overfit():
    """4 graphs, one algorithm, fixed oracle mask -> near-zero loss (design 22.2)."""
    cfg = ExperimentCfg()
    cfg.model = oracle_model_cfg(tiny_model())
    model = _model(cfg.model)
    b = _batch(n=4, alg=0, stream="overfit")
    me, mr = oracle_masks(0, DEV)
    prog = Program(gate_ext=me, gate_red=mr)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    first = last = None
    for i in range(600):
        out = model(b, prog, edge_mode="ste", return_trace=True)
        parts = model_training_loss(out, b, cfg.loss)
        loss = parts["loss"]
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if first is None:
            first = float(parts["final_value"])
        last = float(parts["final_value"])
    assert last < 0.02 * max(first, 1e-9) or last < 2e-4, \
        f"single-batch overfit failed: first={first} last={last}"
