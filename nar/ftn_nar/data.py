"""Abelian task grid: value shifts T_a (mod p) x cyclic rotations R_r on
fixed-length token sequences, one-hot encoded.

A task cell (a, r) maps sequence x to y with y[i] = x[(i - r) mod L] + a (mod p).
Every cell is a fixed permutation of the L*p input bits. Task identity is
never part of the input; it lives only in the mask.
"""

import itertools

import torch


class TaskGrid:
    def __init__(self, p: int, L: int, input_train_frac: float, seed: int, device: str):
        self.p, self.L, self.device = p, L, device
        g = torch.Generator().manual_seed(seed)

        # Full input space: all p^L sequences, as (N, L) int tensor.
        seqs = torch.tensor(
            list(itertools.product(range(p), repeat=L)), dtype=torch.long
        )
        perm = torch.randperm(seqs.shape[0], generator=g)
        n_train = int(input_train_frac * seqs.shape[0])
        self.train_seqs = seqs[perm[:n_train]].to(device)
        self.test_seqs = seqs[perm[n_train:]].to(device)

    def apply(self, seqs: torch.Tensor, a: int, r: int) -> torch.Tensor:
        """Ground-truth data-generating algorithm for cell (a, r)."""
        return torch.roll(seqs, shifts=r, dims=1).add(a).remainder(self.p)

    def encode(self, seqs: torch.Tensor) -> torch.Tensor:
        """(N, L) ints -> (N, L*p) flattened one-hot floats."""
        oh = torch.nn.functional.one_hot(seqs, num_classes=self.p).float()
        return oh.reshape(seqs.shape[0], self.L * self.p)

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        """(N, L*p) logits -> (N, L) argmax predictions."""
        return logits.reshape(-1, self.L, self.p).argmax(-1)

    def batch(self, cell, n: int, split: str = "train", generator=None):
        """Sample n labeled examples (x one-hot, y int targets) from a cell."""
        pool = self.train_seqs if split == "train" else self.test_seqs
        idx = torch.randint(0, pool.shape[0], (n,), generator=generator,
                            device=pool.device if generator is None else "cpu")
        seqs = pool[idx.to(pool.device)]
        a, r = cell
        return self.encode(seqs), self.apply(seqs, a, r)

    def eval_tensors(self, cell, split: str = "test"):
        """The full evaluation split of a cell: (x one-hot, y int targets)."""
        pool = self.train_seqs if split == "train" else self.test_seqs
        a, r = cell
        return self.encode(pool), self.apply(pool, a, r)


def split_grid(p: int, L: int, n_train_cells: int, min_per_a: int, min_per_r: int,
               seed: int):
    """Choose training cells under coverage constraints; the rest are held out.

    The identity cell (0, 0) is always held out: it is the purest 'never
    trained, must be synthesized' probe and keeps a degenerate task out of
    the training stream.
    """
    g = torch.Generator().manual_seed(seed + 1)
    cells = [(a, r) for a in range(p) for r in range(L) if (a, r) != (0, 0)]
    for _ in range(10_000):
        idx = torch.randperm(len(cells), generator=g)[:n_train_cells]
        train = [cells[i] for i in idx.tolist()]
        a_counts = [sum(1 for c in train if c[0] == a) for a in range(p)]
        r_counts = [sum(1 for c in train if c[1] == r) for r in range(L)]
        if min(a_counts) >= min_per_a and min(r_counts) >= min_per_r:
            held = [c for c in cells if c not in train] + [(0, 0)]
            return sorted(train), sorted(held)
    raise RuntimeError("could not satisfy grid coverage constraints")
