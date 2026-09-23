"""task_gf2.py - GF(2) linear + interleaved reversible S-box composition task.

The rigorous, lookup-proof, solver-proof composition testbed (see composition/
README.md). Each of ``n_ops`` primitive operators is a bijection of the
``d = matrix_dim**2``-bit state:

    f_i(v) = S( M_i @ v )                 (all arithmetic over GF(2))

where each ``M_i`` is a random INVERTIBLE GF(2) matrix (built from elementary
row additions on the identity, so invertibility is guaranteed and the number of
additions -- hence the density / sparsity -- is a tunable knob), and ``S`` is a
single fixed reversible block S-box (a random permutation of every b-bit block,
b = ``sbox_bits``). Setting ``nonlinear=False`` drops ``S`` (pure GF(2)-linear).

A *task* is a length-``steps`` word over the operators; its function is the
composition applied left to right::

    g(v) = f_{w[steps-1]}( ... f_{w[1]}( f_{w[0]}(v) ) ... )

Because ``S`` is interleaved after every linear step, the composition is NOT a
single matrix (no closed-form linear solver) and no output bit is a copy of one
input bit (the per-cell lookup shortcut is defeated). Distinct words are distinct
bijections (verified on a probe batch); with the nonlinearity on, a structureless
bijection admits NO generalization to fresh inputs except through the known
compositional structure -- so any above-chance held-out / length-generalized
accuracy is direct evidence of compositional learning.

Interface parity with task_comp.DataGenComposition so the run_compose machinery
drives it unchanged:
    - __call__(task_idx, seed=None, temp=None) -> (x, y)   (n_samples, d) floats
    - .total_input_dim / .input_dim / .output_dim   (all d)
    - .n_tasks / .task_description(task_idx)
Plus, for the oracle / sweep:
    - .sample(task_idx, n) -> (x, y) at an arbitrary batch size
    - .apply_word(v, word) / .enumerate_words()

Determinism: the OPERATORS (M_i, S) and the task-word selection are seeded from
the STRUCTURAL ``seed`` (== Config.SEED) only, NOT from the per-batch sampling
``rng``. So every generator built with the same Config.SEED shares identical
primitives and tasks (train / eval / reload all agree); only the sampled inputs
vary with ``rng``.
"""

import itertools

import torch

import _paths  # noqa: F401  (adds base/ to sys.path; kept for standalone use)


def _cur_device():
    return torch.empty(0).device


def random_invertible_gf2(d, density, gen, device, min_row_weight=2, max_factor=200):
    """Random invertible GF(2) d x d matrix, built from elementary row additions
    (transvections) applied to the identity -- always invertible, and the number
    of additions controls the density.

    Row additions accumulate until the fraction of nonzero entries reaches
    ``density`` (dense default ~0.5) or a cap (max_factor * d) is hit. Lower
    density => sparser matrix => fewer XOR terms per output bit => easier to
    represent (the sparsity difficulty knob).

    ``min_row_weight`` then guarantees every OUTPUT bit is an XOR of at least that
    many input bits (each is a transvection, so invertibility is preserved). With
    min_row_weight >= 2 no output bit is a copy of a single input bit, so the task
    is strictly lookup-proof and the identity predictor is no better than chance.
    """
    M = torch.eye(d, dtype=torch.float32, device=device)
    target = float(density)
    max_ops = int(max_factor * d)
    for _ in range(max_ops):
        if M.mean().item() >= target:
            break
        ij = torch.randint(d, (2,), generator=gen, device=device)
        i, j = int(ij[0]), int(ij[1])
        if i == j:
            continue
        M[i] = torch.remainder(M[i] + M[j], 2.0)
    # Lift any row below the minimum weight by adding another (distinct) row --
    # still an elementary op, so M stays invertible; only the target row changes.
    w = int(max(1, min_row_weight))
    for i in range(d):
        tries = 0
        while int(M[i].sum().item()) < w and tries < 8 * d:
            j = int(torch.randint(d, (1,), generator=gen, device=device))
            if j != i:
                M[i] = torch.remainder(M[i] + M[j], 2.0)
            tries += 1
    return M


class _SBox:
    """Fixed reversible S-box over consecutive blocks of the d-bit state.

    The state is split into blocks of width ``bits`` (a final short block takes
    the remainder); each distinct width gets its own random permutation of its
    2**w values, so the whole map is a bijection on exactly d bits.
    """

    def __init__(self, d, bits, gen, device):
        self.d = d
        self.device = device
        self.blocks = []          # (start, width)
        s = 0
        while s < d:
            w = min(int(bits), d - s)
            self.blocks.append((s, w))
            s += w
        self.tables = {}          # width -> (perm[2**w], bit_table[2**w, w])
        for _, w in self.blocks:
            if w not in self.tables:
                n = 1 << w
                perm = torch.randperm(n, generator=gen, device=device)
                vals = torch.arange(n, device=device)
                shifts = torch.arange(w - 1, -1, -1, device=device)
                bit_table = ((vals.unsqueeze(1) >> shifts) & 1).float()
                self.tables[w] = (perm, bit_table)

    def __call__(self, v):
        out = v.clone()
        for start, w in self.blocks:
            perm, bit_table = self.tables[w]
            block = v[:, start:start + w].long()
            shifts = torch.arange(w - 1, -1, -1, device=self.device)
            idx = (block << shifts).sum(dim=1)          # (B,)
            out[:, start:start + w] = bit_table[perm[idx]]
        return out


class DataGenGF2:
    def __init__(
        self,
        n_tasks,
        n_samples=256,
        matrix_dim=5,
        steps=4,
        n_ops=3,
        p=0.5,
        m_density=0.5,
        m_min_row_weight=2,
        nonlinear=True,
        sbox_bits=4,
        reject_identity=True,
        reject_reducible=True,
        dedupe_unique=True,
        n_probe=256,
        classify=False,          # signature parity; targets are {0,1} either way
        rng=None,
        seed=None,
    ):
        self.n_tasks = int(n_tasks)
        self.n_samples = int(n_samples)
        self.matrix_dim = int(matrix_dim)
        self.d = self.matrix_dim ** 2
        self.steps = int(steps)
        self.n_ops = int(n_ops)
        self.p = float(p)
        self.nonlinear = bool(nonlinear)
        self.device = _cur_device()
        self.input_dim = self.d
        self.output_dim = self.d
        self.total_input_dim = self.d          # no domain-shift widening
        self.rng = rng                          # per-batch sampling generator

        # Structural RNG: fixes operators + task selection from Config.SEED only.
        sgen = torch.Generator(device=self.device)
        sgen.manual_seed(int(seed) if seed is not None else 0)

        self.Ms = [
            random_invertible_gf2(self.d, m_density, sgen, self.device,
                                  min_row_weight=m_min_row_weight)
            for _ in range(self.n_ops)
        ]
        self.sbox = _SBox(self.d, sbox_bits, sgen, self.device) if self.nonlinear else None

        self._build_catalog(reject_identity, reject_reducible, dedupe_unique,
                            int(n_probe), sgen)

    # ------------------------------------------------------------- operators

    def _apply_op(self, v, op):
        v = torch.remainder(v @ self.Ms[op].T, 2.0)
        if self.sbox is not None:
            v = self.sbox(v)
        return v

    def apply_word(self, v, word):
        """Apply a word (sequence of op indices) left to right to v (B, d)."""
        for op in word:
            v = self._apply_op(v, int(op))
        return v

    def enumerate_words(self):
        """The valid deduped length-``steps`` word catalog (oracle hypothesis
        space); the per-task functions are a subset of this."""
        return [tuple(w.tolist()) for w in self.catalog_words]

    # ------------------------------------------------------------- catalog

    def _build_catalog(self, reject_identity, reject_reducible, dedupe_unique,
                       n_probe, sgen):
        probe = (torch.rand(n_probe, self.d, generator=sgen, device=self.device)
                 < 0.5).float()
        pow2 = (2.0 ** torch.arange(self.d, device=self.device)).unsqueeze(0)

        def key(out):
            # Distinct-transform fingerprint: hash the flattened probe outputs.
            bits = out.reshape(-1)
            return hash(tuple(bits.to(torch.int8).tolist()))

        probe_key = key(probe)

        shorter_keys = set()
        if reject_reducible:
            for length in range(1, self.steps):
                for w in itertools.product(range(self.n_ops), repeat=length):
                    shorter_keys.add(key(self.apply_word(probe, w)))

        seen = set()
        valid = []
        for w in itertools.product(range(self.n_ops), repeat=self.steps):
            out = self.apply_word(probe, w)
            k = key(out)
            if reject_identity and k == probe_key:
                continue
            if reject_reducible and k in shorter_keys:
                continue
            if dedupe_unique and k in seen:
                continue
            seen.add(k)
            valid.append(w)

        if len(valid) < self.n_tasks:
            raise ValueError(
                f"Only {len(valid)} distinct length-{self.steps} GF(2) "
                f"compositions available (n_ops={self.n_ops}); requested "
                f"n_tasks={self.n_tasks}. Increase GF2_STEPS / GF2_N_OPS or "
                f"reduce N_TASK."
            )

        self.catalog_words = torch.tensor(valid, dtype=torch.long,
                                          device=self.device)
        order = torch.randperm(len(valid), generator=sgen, device=self.device)
        pick = order[:self.n_tasks]
        self.task_words = self.catalog_words[pick].clone()

    # ------------------------------------------------------------- sampling

    def sample(self, task_idx, n):
        """(x, y) at an arbitrary batch size n (used by the oracle / sweep)."""
        x = (torch.rand(int(n), self.d, generator=self.rng, device=self.device)
             < self.p).float()
        y = self.apply_word(x, self.task_words[task_idx])
        return x, y

    def __call__(self, task_idx, seed=None, temp=None):
        if seed is not None and self.rng is not None:
            self.rng.manual_seed(seed)
        return self.sample(task_idx, self.n_samples)

    def task_description(self, task_idx):
        ops = self.task_words[task_idx].tolist()
        return " -> ".join(f"f{o}" for o in ops)
