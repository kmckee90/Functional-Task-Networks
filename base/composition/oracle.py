"""oracle.py - exact Bayes ceilings and floors for the GF(2) composition task.

Every model in the sweep is judged against a computed ceiling and floor at the
SAME support size B, so "is the model hitting the cap?" is answered quantitatively
rather than by eyeballing a raw accuracy.

Given a task's support set (B pairs) and a fresh query set (Q pairs):

CEILING
  catalog_oracle : knows the operator set + S-box; enumerates the length-``steps``
                   word catalog, keeps the words consistent with the support set,
                   and predicts the query by per-cell majority vote over the
                   surviving words (uniform posterior). This is the strongest
                   prior -- "perfect compositional learning" -- and is what a model
                   that truly composes the primitives should approach. Exactly
                   computable (catalog is n_ops**steps words, deduped).

FLOORS
  identity       : predict y = x.
  marginal       : predict each output cell's majority bit from the support set.
  memorization   : 1-NN over the support set by Hamming distance (provably ~chance
                   on fresh queries -- makes concrete that memorization cannot
                   generalize here, so any FTN gain is compositional).

All metrics: per-cell accuracy and exact-match (whole-vector correct) rate.
"""

import numpy as np
import torch


def _percell_and_exact(pred_bits, y_bits):
    """pred_bits, y_bits: (Q, d) in {0,1}. Returns (per_cell_acc, exact_match)."""
    correct = (pred_bits == y_bits)
    per_cell = correct.float().mean().item()
    exact = correct.all(dim=1).float().mean().item()
    return per_cell, exact


def catalog_oracle(gen, task_idx, xs, ys, xq, yq):
    """Per-cell majority vote over the catalog words consistent with the support.

    xs/ys: support (B, d); xq/yq: query (Q, d). Returns (per_cell, exact,
    n_consistent) where n_consistent is how many catalog words survived the
    support set (1 == the task is uniquely identified)."""
    words = gen.enumerate_words()
    consistent = []
    for w in words:
        if torch.equal(gen.apply_word(xs, w), ys):
            consistent.append(w)
    if not consistent:                       # cannot happen (true word matches)
        pred = torch.zeros_like(yq)
        pc, ex = _percell_and_exact(pred, yq)
        return pc, ex, 0
    # Per-cell majority vote over surviving words (uniform posterior).
    votes = torch.zeros_like(yq)
    for w in consistent:
        votes = votes + gen.apply_word(xq, w)
    pred = (votes >= (len(consistent) / 2.0)).float()
    pc, ex = _percell_and_exact(pred, yq)
    return pc, ex, len(consistent)


def gf2_rank(X):
    """Rank over GF(2) of a {0,1} matrix X (B, d) via Gaussian elimination."""
    A = (X.detach().to(torch.int64) % 2).clone()
    rows, cols = A.shape
    r = 0
    for c in range(cols):
        nz = torch.nonzero(A[r:, c], as_tuple=False)
        if nz.numel() == 0:
            continue
        piv = r + int(nz[0])
        A[[r, piv]] = A[[piv, r]]
        m = A[:, c].bool().clone()
        m[r] = False
        A[m] = (A[m] + A[r]) % 2
        r += 1
        if r == rows:
            break
    return r


def linear_agnostic_cap(xs):
    """Exact Bayes cap for a learner that knows only that the task is GF(2)-LINEAR
    (uniform prior over linear maps), given support inputs xs (B, d).

    Each output bit is <m_j, x_q>; from the support we can determine it exactly for
    queries in the row space of xs (dimension r = rank), and it is 50/50 for the
    orthogonal component. For a uniform-random query,
        per-cell cap = 0.5 + 0.5 * 2**(r - d),
        exact-match cap ~= 2**(r - d).
    Sharp phase transition to 1.0 at support size B ~= d. This is the "linear
    regression" reference ceiling; the catalog oracle (which knows the primitive
    set) is stronger and needs only ~1 sample."""
    d = xs.shape[1]
    r = gf2_rank(xs)
    frac = 2.0 ** (r - d)
    return 0.5 + 0.5 * frac, frac, r


def identity_floor(xq, yq):
    return _percell_and_exact(xq, yq)


def marginal_floor(ys, yq):
    """Predict each output cell's majority bit estimated from the support set."""
    maj = (ys.mean(dim=0, keepdim=True) >= 0.5).float().expand_as(yq)
    return _percell_and_exact(maj, yq)


def memorization_floor(xs, ys, xq, yq):
    """1-nearest-neighbour over the support by Hamming distance."""
    # dist[q, b] = # differing bits between xq[q] and xs[b]
    dist = (xq.unsqueeze(1) != xs.unsqueeze(0)).sum(dim=2)   # (Q, B)
    nn = dist.argmin(dim=1)                                   # (Q,)
    pred = ys[nn]
    return _percell_and_exact(pred, yq)


def evaluate_task(gen, task_idx, support_B, query_Q):
    """All ceilings + floors for one task at support size support_B."""
    xs, ys = gen.sample(task_idx, support_B)
    xq, yq = gen.sample(task_idx, query_Q)
    cap_pc, cap_ex, n_cons = catalog_oracle(gen, task_idx, xs, ys, xq, yq)
    lin_pc, lin_ex, rank = linear_agnostic_cap(xs)
    id_pc, id_ex = identity_floor(xq, yq)
    mg_pc, mg_ex = marginal_floor(ys, yq)
    mem_pc, mem_ex = memorization_floor(xs, ys, xq, yq)
    return {
        "catalog_acc": cap_pc, "catalog_exact": cap_ex, "n_consistent": n_cons,
        "linear_acc": lin_pc, "linear_exact": lin_ex, "support_rank": rank,
        "identity_acc": id_pc, "identity_exact": id_ex,
        "marginal_acc": mg_pc, "marginal_exact": mg_ex,
        "memorization_acc": mem_pc, "memorization_exact": mem_ex,
    }


def evaluate_split(gen, task_indices, support_B, query_Q):
    """Mean over tasks of every ceiling/floor metric at support size support_B."""
    rows = [evaluate_task(gen, t, support_B, query_Q) for t in task_indices]
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}
