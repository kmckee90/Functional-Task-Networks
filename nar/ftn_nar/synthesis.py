"""Zero-shot mask synthesis at held-out grid cells.

The factorization hypothesis: trained masks organize as
score(unit; a, r) ~ mu + alpha_a + beta_r  (per bank), i.e. an additive
factor model over the grid. If true, a mask for a never-trained cell can be
completed from the trained cells' masks alone -- no data from the cell, no
gradient steps, pure mask algebra.

Primary synthesizer: additive ANOVA completion on the stored masks (or mask
EMAs), followed by per-bank KWTA. Controls: mu-only (grand-mean mask, tests
whether factor terms add anything), majority-vote union (simpler algebra),
nearest trained neighbor, random k-mask.
"""

import torch

from .configurer import kwta


def _fit_additive(mask_matrix: torch.Tensor, cells):
    """Least-squares fit of m[cell] ~ mu + alpha[a] + beta[r] per unit.

    With sum-to-zero factor encoding, on a (possibly incomplete) grid the
    row/column-mean estimator is biased; solve the small normal equations
    exactly instead. mask_matrix: (n_cells, U). Returns (mu, alpha, beta) with
    alpha: (n_a, U), beta: (n_r, U).
    """
    a_levels = sorted({a for a, _ in cells})
    r_levels = sorted({r for _, r in cells})
    n, U = mask_matrix.shape
    n_a, n_r = len(a_levels), len(r_levels)

    # Design matrix with explicit dummy coding + intercept; pinv handles the
    # rank deficiency of dummy coding gracefully.
    X = torch.zeros(n, 1 + n_a + n_r)
    for i, (a, r) in enumerate(cells):
        X[i, 0] = 1.0
        X[i, 1 + a_levels.index(a)] = 1.0
        X[i, 1 + n_a + r_levels.index(r)] = 1.0
    coef = torch.linalg.pinv(X) @ mask_matrix  # (1 + n_a + n_r, U)

    mu = coef[0]
    alpha = {a: coef[1 + i] for i, a in enumerate(a_levels)}
    beta = {r: coef[1 + n_a + i] for i, r in enumerate(r_levels)}
    return mu, alpha, beta


def _topk_per_bank(score: torch.Tensor, H1: int, k1: int, k2: int):
    return torch.cat([kwta(score[:H1], k1), kwta(score[H1:], k2)])


class MaskSynthesizer:
    def __init__(self, masks: dict, cells, H1: int, k1: int, k2: int,
                 p: int, L: int):
        """masks: {cell: (H1+H2,) float tensor} from trained cells only."""
        self.cells = list(cells)
        self.M = torch.stack([masks[c].float() for c in self.cells])
        self.H1, self.k1, self.k2 = H1, k1, k2
        self.p, self.L = p, L
        self.mu, self.alpha, self.beta = _fit_additive(self.M, self.cells)

    def covers(self, a: int, r: int) -> bool:
        """A held-out cell is identifiable only if both its factor levels
        appear in at least one trained cell."""
        return a in self.alpha and r in self.beta

    def additive(self, a: int, r: int) -> torch.Tensor:
        """Primary zero-shot synthesis: mu + alpha_a + beta_r -> per-bank KWTA."""
        score = self.mu + self.alpha[a] + self.beta[r]
        return _topk_per_bank(score, self.H1, self.k1, self.k2)

    def mu_only(self) -> torch.Tensor:
        """Control: grand-mean mask. If this matches additive(), the factor
        structure contributes nothing."""
        return _topk_per_bank(self.mu.clone(), self.H1, self.k1, self.k2)

    def assigned(self, a: int, r: int, swap: bool = False) -> torch.Tensor:
        """Depth-assignment probe: bank 1 synthesized purely from same-a cells,
        bank 2 purely from same-r cells (swap reverses the assignment). Tests
        whether the value atom lives in one bank and the rotation atom in the
        other, which additive completion over both banks would dilute."""
        same_a = self.M[[i for i, c in enumerate(self.cells) if c[0] == a]].mean(0)
        same_r = self.M[[i for i, c in enumerate(self.cells) if c[1] == r]].mean(0)
        first, second = (same_r, same_a) if swap else (same_a, same_r)
        return torch.cat([kwta(first[: self.H1], self.k1),
                          kwta(second[self.H1:], self.k2)])

    def majority_union(self, a: int, r: int) -> torch.Tensor:
        """Secondary synthesis: mean mask over same-a cells + mean over
        same-r cells (union-style vote), per-bank KWTA."""
        same_a = self.M[[i for i, c in enumerate(self.cells) if c[0] == a]]
        same_r = self.M[[i for i, c in enumerate(self.cells) if c[1] == r]]
        score = same_a.mean(0) + same_r.mean(0)
        return _topk_per_bank(score, self.H1, self.k1, self.k2)

    def nearest_neighbor(self, a: int, r: int) -> torch.Tensor:
        """Floor: reuse the mask of the closest trained cell (torus distance)."""
        def dist(c):
            da = min((c[0] - a) % self.p, (a - c[0]) % self.p)
            dr = min((c[1] - r) % self.L, (r - c[1]) % self.L)
            return da + dr
        i = min(range(len(self.cells)), key=lambda i: dist(self.cells[i]))
        return _topk_per_bank(self.M[i].clone(), self.H1, self.k1, self.k2)

    def random_mask(self, generator: torch.Generator) -> torch.Tensor:
        return _topk_per_bank(torch.rand(self.M.shape[1], generator=generator),
                              self.H1, self.k1, self.k2)


def anova_decomposition(masks: dict, cells, H1: int):
    """Variance decomposition of mask bits across grid cells, per bank.

    Returns fractions of cross-cell variance explained by the a main effect,
    the r main effect, and the interaction/residual, averaged over units.
    """
    M = torch.stack([masks[c].float() for c in cells])
    mu, alpha, beta = _fit_additive(M, cells)
    fitted = torch.stack(
        [mu + alpha[a] + beta[r] for a, r in cells])
    fitted_a = torch.stack([mu + alpha[a] for a, _ in cells])
    fitted_r = torch.stack([mu + beta[r] for _, r in cells])

    out = {}
    for name, sl in (("bank1", slice(0, H1)), ("bank2", slice(H1, None))):
        Mb, Fb = M[:, sl], fitted[:, sl]
        total = Mb.var(0, unbiased=False).sum().item()
        if total == 0:
            out[name] = dict(a=0.0, r=0.0, additive=0.0, residual=1.0)
            continue
        ss_a = (fitted_a[:, sl] - Mb.mean(0)).pow(2).mean(0).sum().item()
        ss_r = (fitted_r[:, sl] - Mb.mean(0)).pow(2).mean(0).sum().item()
        ss_add = (Fb - Mb.mean(0)).pow(2).mean(0).sum().item()
        out[name] = dict(
            a=ss_a / total, r=ss_r / total, additive=ss_add / total,
            residual=1.0 - ss_add / total)
    return out


def overlap_matrix(masks: dict, cells, sl: slice, k: int):
    """Pairwise mask overlap |m_i & m_j| / k over cells, ordered as given."""
    M = torch.stack([masks[c].float()[sl] for c in cells])
    return (M @ M.T) / k
