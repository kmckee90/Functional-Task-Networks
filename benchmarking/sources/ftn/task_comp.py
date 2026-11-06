import itertools
from collections import Counter

import torch


class BinaryCompositionGenerator:
    """
    Synthetic function-composition task on binary N x N matrices.

    Operators:
        0: rotate 90 degrees clockwise
        1: flip left-right
        2: cyclic shift right by `shift` columns

    The generator can reject:
        - identity compositions
        - compositions equivalent to a shorter sequence
        - exact-length sequences algebraically equivalent to another sequence
        - individual binary matrices for which different transformations happen
          to produce the same output

    This is intended to generate batches directly on CPU or GPU rather than
    being wrapped in a PyTorch DataLoader.
    """

    OP_NAMES = ("rot90_cw", "flip_lr", "shift_right")

    def __init__(
        self,
        matrix_dim=8,
        steps=4,
        p=0.25,
        device=None,
        dtype=torch.float32,
        shift=1,
        reject_identity=True,
        reject_reducible=True,
        require_unique_transform=True,
        require_unique_output=True,
        min_changed_cells=1,
        oversample_factor=2.0,
        ambiguity_chunk_size=64,
        max_resample_rounds=100,
        seed=None,
    ):
        """
        Args:
            matrix_dim:
                Matrix side length N.

            steps:
                Number of operators in each composition.

            p:
                Bernoulli probability for each matrix entry being 1.

            device:
                e.g. "cuda", "cuda:0", or "cpu".
                Defaults to CUDA if available.

            dtype:
                Output matrix dtype.

            shift:
                Number of columns used by shift_right.

            reject_identity:
                Don't use sequences whose total transformation is identity.

            reject_reducible:
                Don't use sequences whose total transformation can be
                represented using fewer than `steps` operators.

            require_unique_transform:
                Reject a sequence if another sequence of exactly `steps`
                operators implements the identical transformation globally.

                This is important if the target is the exact operator sequence,
                rather than merely the resulting transformation.

            require_unique_output:
                For each sampled binary matrix, verify that no other distinct
                exact-length transformation produces the same output.

                This removes accidental ambiguities caused by symmetric or
                periodic input matrices.

            min_changed_cells:
                Require at least this many cells to differ between x and y.

            oversample_factor:
                Generate extra candidates per rejection-sampling round.

            ambiguity_chunk_size:
                Chunk size for uniqueness checking. Lower this to reduce peak
                GPU memory.

            max_resample_rounds:
                Maximum rejection-sampling rounds.

            seed:
                Optional RNG seed.
        """
        if matrix_dim < 2:
            raise ValueError("matrix_dim must be >= 2")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        if min_changed_cells < 0:
            raise ValueError("min_changed_cells must be >= 0")
        if oversample_factor < 1.0:
            raise ValueError("oversample_factor must be >= 1")

        self.matrix_dim = int(matrix_dim)
        self.steps = int(steps)
        self.p = float(p)

        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.dtype = dtype
        self.shift = int(shift) % self.matrix_dim

        self.reject_identity = reject_identity
        self.reject_reducible = reject_reducible
        self.require_unique_transform = require_unique_transform
        self.require_unique_output = require_unique_output

        self.min_changed_cells = int(min_changed_cells)
        self.oversample_factor = float(oversample_factor)
        self.ambiguity_chunk_size = int(ambiguity_chunk_size)
        self.max_resample_rounds = int(max_resample_rounds)

        self.generator = torch.Generator(device=self.device)

        if seed is not None:
            self.generator.manual_seed(seed)

        # Enumerate the small operator language once and determine which
        # sequences are actually usable labels.
        self._build_sequence_catalog()

    @property
    def num_valid_sequences(self):
        return self._valid_words.shape[0]

    @property
    def op_names(self):
        return self.OP_NAMES

    def _apply_op(self, x, op_id):
        """Apply one operator to the final two dimensions."""
        if op_id == 0:
            return torch.rot90(x, k=-1, dims=(-2, -1))

        if op_id == 1:
            return torch.flip(x, dims=(-1,))

        if op_id == 2:
            return torch.roll(x, shifts=self.shift, dims=-1)

        raise ValueError(f"Unknown op_id {op_id}")

    def _permutation_for_word(self, word):
        """
        Represent a composition as a permutation of matrix locations.

        Using unique integer values rather than binary values lets us detect
        whether two operator sequences are globally identical.
        """
        n = self.matrix_dim

        x = torch.arange(
            n * n,
            dtype=torch.long,
        ).reshape(n, n)

        for op_id in word:
            x = self._apply_op(x, op_id)

        return tuple(x.reshape(-1).tolist())

    def _build_sequence_catalog(self):
        n_ops = len(self.OP_NAMES)

        words = list(
            itertools.product(
                range(n_ops),
                repeat=self.steps,
            )
        )

        perms = [
            self._permutation_for_word(word)
            for word in words
        ]

        # Number of syntactically different words implementing each
        # transformation.
        transform_counts = Counter(perms)

        identity = tuple(
            range(self.matrix_dim * self.matrix_dim)
        )

        # Transformations achievable in fewer than `steps` operations.
        shorter_perms = set()

        if self.reject_reducible:
            # Zero-step composition.
            shorter_perms.add(identity)

            for length in range(1, self.steps):
                for word in itertools.product(
                    range(n_ops),
                    repeat=length,
                ):
                    shorter_perms.add(
                        self._permutation_for_word(word)
                    )

        valid_words = []
        valid_perms = []

        for word, perm in zip(words, perms):

            if self.reject_identity and perm == identity:
                continue

            if (
                self.reject_reducible
                and perm in shorter_perms
            ):
                continue

            if (
                self.require_unique_transform
                and transform_counts[perm] != 1
            ):
                continue

            valid_words.append(word)
            valid_perms.append(perm)

        if not valid_words:
            raise ValueError(
                "No valid operator sequences remain. "
                "Try increasing matrix_dim, reducing steps, "
                "or relaxing the rejection constraints."
            )

        # Keep one representative for every distinct exact-length
        # transformation. These are used to detect accidental ambiguities
        # on particular random binary matrices.
        distinct_exact_perms = list(
            dict.fromkeys(perms)
        )

        self._valid_words = torch.tensor(
            valid_words,
            dtype=torch.long,
            device=self.device,
        )

        self._valid_perms = torch.tensor(
            valid_perms,
            dtype=torch.long,
            device=self.device,
        )

        self._exact_perms = torch.tensor(
            distinct_exact_perms,
            dtype=torch.long,
            device=self.device,
        )

    def _outputs_are_unique(self, x_flat, y_flat):
        """
        Return [B] boolean mask.

        True means exactly one distinct exact-length transformation maps
        that input matrix to that output matrix.
        """
        batch_size = x_flat.shape[0]

        match_count = torch.zeros(
            batch_size,
            dtype=torch.int32,
            device=self.device,
        )

        for start in range(
            0,
            self._exact_perms.shape[0],
            self.ambiguity_chunk_size,
        ):
            perms = self._exact_perms[
                start : start + self.ambiguity_chunk_size
            ]

            # x_flat: [B, N*N]
            # perms:  [K, N*N]
            #
            # result: [B, K, N*N]
            candidate_outputs = x_flat[:, perms]

            matches = (
                candidate_outputs
                == y_flat[:, None, :]
            ).all(dim=-1)

            match_count += matches.sum(
                dim=-1
            ).to(torch.int32)

        return match_count == 1

    def apply_sequence(
        self,
        x,
        ops,
        return_intermediates=False,
    ):
        """
        Explicitly apply a batch of operator sequences.

        Args:
            x:
                [B, C, N, N]

            ops:
                [B, steps]

            return_intermediates:
                If True, also return all states.

        Returns:
            y:
                [B, C, N, N]

            states, if requested:
                [B, steps + 1, C, N, N]
        """
        if (
            x.ndim != 4
            or x.shape[-2:]
            != (self.matrix_dim, self.matrix_dim)
        ):
            raise ValueError(
                "x must have shape [B, C, N, N]"
            )

        if ops.shape != (
            x.shape[0],
            self.steps,
        ):
            raise ValueError(
                f"ops must have shape [B, {self.steps}]"
            )

        z = x

        states = (
            [z]
            if return_intermediates
            else None
        )

        for t in range(self.steps):
            # Compute all three possibilities in parallel.
            r = torch.rot90(
                z,
                k=-1,
                dims=(-2, -1),
            )

            f = torch.flip(
                z,
                dims=(-1,),
            )

            s = torch.roll(
                z,
                shifts=self.shift,
                dims=-1,
            )

            op = ops[:, t]

            z = torch.where(
                (op == 0)[:, None, None, None],
                r,
                torch.where(
                    (op == 1)[:, None, None, None],
                    f,
                    s,
                ),
            )

            if return_intermediates:
                states.append(z)

        if return_intermediates:
            return z, torch.stack(
                states,
                dim=1,
            )

        return z

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_intermediates=False,
    ):
        """
        Generate one batch.

        Returns:
            {
                "x":   [B, 1, N, N],
                "y":   [B, 1, N, N],
                "ops": [B, steps],

                # optionally
                "states": [B, steps+1, 1, N, N]
            }

        ops are in application order:

            x
              --ops[:, 0]-->
              --ops[:, 1]-->
              ...
              --ops[:, -1]-->
            y
        """
        if batch_size < 1:
            raise ValueError(
                "batch_size must be >= 1"
            )

        xs = []
        ys = []
        op_batches = []

        accepted = 0

        for _ in range(
            self.max_resample_rounds
        ):
            if accepted >= batch_size:
                break

            needed = batch_size - accepted

            n_candidates = max(
                needed,
                int(
                    self.oversample_factor
                    * needed
                ),
            )

            # Bernoulli random binary matrices.
            x = (
                torch.rand(
                    (
                        n_candidates,
                        1,
                        self.matrix_dim,
                        self.matrix_dim,
                    ),
                    device=self.device,
                    generator=self.generator,
                )
                < self.p
            ).to(self.dtype)

            # Uniformly sample from valid operator words.
            word_ids = torch.randint(
                self.num_valid_sequences,
                (n_candidates,),
                device=self.device,
                generator=self.generator,
            )

            ops = self._valid_words[word_ids]

            # Each permutation tells us which source cell appears at each
            # output location.
            perms = self._valid_perms[word_ids]

            x_flat = x[:, 0].reshape(
                n_candidates,
                -1,
            )

            # Efficiently execute the complete composition as a gather.
            y_flat = torch.gather(
                x_flat,
                dim=1,
                index=perms,
            )

            y = y_flat.reshape(
                n_candidates,
                1,
                self.matrix_dim,
                self.matrix_dim,
            )

            keep = torch.ones(
                n_candidates,
                dtype=torch.bool,
                device=self.device,
            )

            # Reject nearly/no-op examples.
            if self.min_changed_cells > 0:
                changed = (
                    x_flat != y_flat
                ).sum(dim=1)

                keep &= (
                    changed
                    >= self.min_changed_cells
                )

            # Reject input-specific ambiguities.
            if self.require_unique_output:
                keep &= self._outputs_are_unique(
                    x_flat,
                    y_flat,
                )

            keep_idx = torch.nonzero(
                keep,
                as_tuple=False,
            ).squeeze(1)

            if keep_idx.numel() == 0:
                continue

            take = min(
                needed,
                keep_idx.numel(),
            )

            keep_idx = keep_idx[:take]

            xs.append(x[keep_idx])
            ys.append(y[keep_idx])
            op_batches.append(
                ops[keep_idx]
            )

            accepted += take

        if accepted < batch_size:
            raise RuntimeError(
                f"Could only generate "
                f"{accepted}/{batch_size} "
                f"valid examples after "
                f"{self.max_resample_rounds} rounds. "
                "Try increasing p or matrix_dim, "
                "or relaxing the uniqueness constraints."
            )

        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)
        ops = torch.cat(
            op_batches,
            dim=0,
        )

        result = {
            "x": x,
            "y": y,
            "ops": ops,
        }

        if return_intermediates:
            y_check, states = (
                self.apply_sequence(
                    x,
                    ops,
                    return_intermediates=True,
                )
            )

            if not torch.equal(
                y,
                y_check,
            ):
                raise RuntimeError(
                    "Internal consistency error."
                )

            result["states"] = states

        return result


class DataGenComposition:
    """Continual-learning function-composition task, following the DataGenSine
    convention so run_experiments can drive it unchanged.

    Each of ``n_tasks`` tasks is a *distinct, fixed* operator composition of
    length ``steps`` (e.g. task 0 = rot90_cw -> flip_lr -> shift_right, task 1 =
    shift_right -> rot90_cw -> flip_lr). Calling the generator with a task index
    returns a batch of ``n_samples`` random binary N x N matrices and their
    images under that task's composition::

        x, y = data_gen(task_idx)     # x, y: (n_samples, N*N) float in {0, 1}
                                      # y = x transformed by task_idx's function

    The task functions are drawn from BinaryCompositionGenerator's valid-word
    catalog (non-identity, optionally non-reducible / unique-transform) and
    deduped by global transform, so every task is a genuinely different function.

    Interface parity with tasks.DataGenSine:
        - __call__(task_idx, seed=None, temp=None) -> (x, y)
        - .total_input_dim / .input_dim / .output_dim  (all N*N here)
        - .n_tasks

    The inputs and targets are binary matrices in {0, 1}, so the target is the
    transformed matrix trained as per-cell binary classification (one sigmoid
    per cell; config.Compose sets BINARY=True). ``classify`` is accepted for
    signature parity and does not change the emitted data (targets are {0, 1}
    floats either way).
    """

    OP_NAMES = BinaryCompositionGenerator.OP_NAMES

    def __init__(
        self,
        n_tasks,
        n_samples=256,
        matrix_dim=8,
        steps=3,
        p=0.25,
        shift=1,
        reject_identity=True,
        reject_reducible=True,
        require_unique_transform=True,
        classify=False,
        rng=None,
        seed=None,
    ):
        self.n_tasks = int(n_tasks)
        self.n_samples = int(n_samples)
        self.matrix_dim = int(matrix_dim)
        self.steps = int(steps)
        self.p = float(p)
        self.classify = bool(classify)
        self.rng = rng

        N = self.matrix_dim
        self.input_dim = N * N
        self.output_dim = N * N
        # No domain-shift widening for compose: the model input width is N*N.
        self.total_input_dim = N * N

        device = torch.empty(0).device

        # Reuse the operator-language engine for the catalog / validity logic.
        self._engine = BinaryCompositionGenerator(
            matrix_dim=matrix_dim,
            steps=steps,
            p=p,
            shift=shift,
            device=device,
            reject_identity=reject_identity,
            reject_reducible=reject_reducible,
            require_unique_transform=require_unique_transform,
            require_unique_output=False,
            seed=seed,
        )

        # Dedupe valid words by their global transform so every task is a
        # genuinely distinct function even if require_unique_transform is off.
        words = self._engine._valid_words
        perms = self._engine._valid_perms
        seen = set()
        keep = []
        for i in range(words.shape[0]):
            key = tuple(perms[i].tolist())
            if key not in seen:
                seen.add(key)
                keep.append(i)
        keep = torch.tensor(keep, dtype=torch.long, device=device)

        if keep.numel() < self.n_tasks:
            raise ValueError(
                f"Only {keep.numel()} distinct length-{steps} compositions "
                f"exist for matrix_dim={matrix_dim} (ops={self.OP_NAMES}), but "
                f"N_TASK={n_tasks} were requested. Increase COMP_STEPS or "
                f"COMP_MATRIX_DIM, or reduce N_TASK."
            )

        # Shuffle (deterministic under the provided RNG) and take the first
        # N_TASK words as the fixed per-task functions.
        if rng is not None:
            order = torch.randperm(keep.numel(), generator=rng, device=device)
            keep = keep[order]
        pick = keep[: self.n_tasks]

        self.task_words = words[pick].clone()  # (n_tasks, steps) operator ids
        self.task_perms = perms[pick].clone()  # (n_tasks, N*N)   gather index

    @property
    def op_names(self):
        return self.OP_NAMES

    def task_description(self, task_idx):
        """Human-readable operator sequence for a task, e.g.
        'rot90_cw -> flip_lr -> shift_right'."""
        ops = self.task_words[task_idx].tolist()
        return " -> ".join(self.OP_NAMES[o] for o in ops)

    def __call__(self, task_idx, seed=None, temp=None):
        """One batch for a task: (x, y), each (n_samples, N*N) float in {0, 1}.
        y is x transformed by task_idx's fixed operator composition."""
        if seed is not None and self.rng is not None:
            self.rng.manual_seed(seed)
        N = self.matrix_dim
        with torch.no_grad():
            x = (
                torch.rand(self.n_samples, N * N, generator=self.rng) < self.p
            ).float()
            idx = self.task_perms[task_idx].unsqueeze(0).expand(self.n_samples, -1)
            y = torch.gather(x, 1, idx)
        return x, y


if __name__ == "__main__":
    gen = BinaryCompositionGenerator(
        matrix_dim=8,
        steps=4,
        p=0.2,
        device="cuda",
        seed=123,
    )

    print(gen.op_names)
    # ('rot90_cw', 'flip_lr', 'shift_right')

    print(gen.num_valid_sequences)

    batch = gen.sample(
        batch_size=256,
        return_intermediates=True,
    )

    x = batch["x"]          # [256, 1, 8, 8], already on GPU
    y = batch["y"]          # [256, 1, 8, 8]
    ops = batch["ops"]      # [256, 4]
    states = batch["states"]  # [256, 5, 1, 8, 8]

    breakpoint()