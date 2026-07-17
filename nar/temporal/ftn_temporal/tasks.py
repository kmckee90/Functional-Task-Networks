"""The non-commutative task family: the affine group Aff(Z_p).

A task is one affine map g = (c, a): x -> c*x + a (mod p), applied elementwise
to a sequence of L tokens. Tasks are reached as WORDS over a tiny generator
set (default T1: x+1 and M2: 2x). Composition is non-commutative
(T1∘M2 ≠ M2∘T1), and every map has a shortest word length l(g) — the
program-length axis. Training tasks are all maps with l <= train_max_len
(plus the identity, the halt atom); everything longer is held out, so the
scale-OOD axis is literally "programs longer than any ever trained".

Task identity is never in the input: the model sees only (x, g(x)) pairs.
"""

import itertools
from collections import deque

import torch


def compose_after(g2, g1, p):
    """Apply g1 first, then g2: returns g2 ∘ g1 as (c, a)."""
    c1, a1 = g1
    c2, a2 = g2
    return ((c2 * c1) % p, (c2 * a1 + a2) % p)


def bfs_words(generators, p):
    """Shortest word length and one shortest word (list of generator names)
    for every reachable group element. Word [n1, n2, ...] means: apply
    generator n1 first, then n2, etc."""
    ident = (1, 0)
    info = {ident: (0, [])}
    frontier = deque([ident])
    while frontier:
        g = frontier.popleft()
        depth, word = info[g]
        for name, gen in generators:
            h = compose_after(gen, g, p)   # append gen as the NEXT step
            if h not in info:
                info[h] = (depth + 1, word + [name])
                frontier.append(h)
    return info  # {(c, a): (length, word)}


class AffineTasks:
    def __init__(self, cfg):
        self.p, self.L = cfg.p, cfg.L
        self.device = cfg.device
        self.generators = list(cfg.generators)
        self.gen_map = dict(self.generators)

        self.info = bfs_words(self.generators, self.p)
        self.length = {g: l for g, (l, _) in self.info.items()}
        self.word = {g: w for g, (_, w) in self.info.items()}

        self.train_tasks = sorted(
            [g for g, l in self.length.items() if l <= cfg.train_max_len],
            key=lambda g: (self.length[g], g))          # includes identity (l=0)
        self.held_tasks = sorted(
            [g for g, l in self.length.items()
             if cfg.train_max_len < l <= cfg.eval_max_len],
            key=lambda g: (self.length[g], g))

        # Full input space, split once, shared across all tasks.
        gseed = torch.Generator().manual_seed(cfg.seed)
        seqs = torch.tensor(list(itertools.product(range(self.p), repeat=self.L)),
                            dtype=torch.long)
        perm = torch.randperm(seqs.shape[0], generator=gseed)
        n_train = int(cfg.input_train_frac * seqs.shape[0])
        self.train_seqs = seqs[perm[:n_train]].to(self.device)
        self.test_seqs = seqs[perm[n_train:]].to(self.device)

    # ---- data plumbing ----
    def apply(self, seqs, g):
        c, a = g
        return (c * seqs + a) % self.p

    def encode(self, seqs):
        oh = torch.nn.functional.one_hot(seqs, num_classes=self.p).float()
        return oh.reshape(seqs.shape[0], self.L * self.p)

    def decode(self, logits):
        return logits.reshape(-1, self.L, self.p).argmax(-1)

    def batch(self, g, n, split="train"):
        pool = self.train_seqs if split == "train" else self.test_seqs
        idx = torch.randint(0, pool.shape[0], (n,), device=pool.device)
        seqs = pool[idx]
        return self.encode(seqs), self.apply(seqs, g)

    def eval_tensors(self, g, split="test"):
        pool = self.train_seqs if split == "train" else self.test_seqs
        return self.encode(pool), self.apply(pool, g)

    def level_sizes(self):
        out = {}
        for g, l in self.length.items():
            out[l] = out.get(l, 0) + 1
        return dict(sorted(out.items()))
