"""
tasks.py — Task families for func_net_kt knowledge-transfer experiments.

All task families expose:

    task.sample_batch(task_idx, batch_size, split, device) -> (x, y)
    task.evaluate(model, mask, task_idx, split, device)    -> scalar metric
    task.concept_table() -> (n_tasks, n_components) binary numpy array
        which abstract concept components each task uses; used as the
        ground-truth reference when comparing the model's empirical
        neuron-sharing matrix.
    task.n_tasks, task.input_size, task.output_size, task.is_classification

Two families:

  1. SyntheticTasks        -- DataGenSine sinusoidal mixtures whose
                              latent space is partitioned into C concept
                              components.  Each task is the union of a
                              pair of components, so any two tasks share
                              exactly one component.
  2. MNISTSharedShuffle    -- MNIST images with structured digit->label
                              permutations: each task's permutation is
                              the union of two label sub-mappings, so
                              any two tasks share half of their digit ->
                              label assignments.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. SyntheticTasks — DataGenSine with pairwise-union correlated concept masks
# ---------------------------------------------------------------------------


class SyntheticTasks:
    """
    Synthetic correlated multi-task data generator.

    The data-generating model has a latent hidden space partitioned into
    ``n_components`` concept components, each occupying ``task_size``
    consecutive dimensions.  Each task is the union of a different
    pairwise combination of two components, so each component is shared
    by exactly two tasks.

    Forward pass per task t:
        z = enc(x) * m_t                    # (B, C*task_size)
        z = sin(f * z) * m_t
        y = classify(sum(z))                # clf: 2-way; reg: scalar

    Args:
        input_size:    Dimensionality of x.
        output_size:   Number of classes (clf) or scalar (reg).
        n_tasks:       Number of tasks.
        n_components:  Number of latent concept components.  Defaults to
                       ``n_tasks`` so the pairwise-union table fills.
        task_size:     Latent dims per component.
        frequency:     Sin nonlinearity frequency (higher = harder).
        classification:If True, binarise output to one-hot.
        seed:          RNG seed.
        device:        Torch device.
    """

    def __init__(
        self,
        input_size: int = 2,
        output_size: int = 2,
        n_tasks: int = 6,
        n_components: int | None = None,
        task_size: int = 4,
        frequency: float = 8.0,
        classification: bool = True,
        seed: int = 0,
        device: str = "cpu",
        test_size: int = 2048,
        orthogonal: bool = False,
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.n_tasks = n_tasks
        self.task_size = task_size
        self.frequency = frequency
        self.is_classification = classification
        self.device = device
        self.orthogonal = orthogonal

        rng = torch.Generator().manual_seed(seed)

        if orthogonal:
            # Disjoint task supports: each task gets its own pair of two
            # private components, so no two tasks share any component
            # and there is no latent structure to transfer.
            C = 2 * n_tasks if n_components is None else n_components
            if C < 2 * n_tasks:
                raise ValueError(
                    f"orthogonal needs n_components >= 2*n_tasks; got {C}"
                )
            self.n_components = C
            task_component_table = torch.zeros(n_tasks, C)
            comp_perm = torch.randperm(C, generator=rng).tolist()
            for ti in range(n_tasks):
                a, b = comp_perm[2 * ti], comp_perm[2 * ti + 1]
                task_component_table[ti, a] = 1.0
                task_component_table[ti, b] = 1.0
            self._concept_table = task_component_table.numpy()
        else:
            C = n_components if n_components is not None else n_tasks
            # We need at least n_tasks distinct pairs out of C(C,2).
            if C * (C - 1) // 2 < n_tasks:
                raise ValueError(
                    f"n_components={C} is too small to yield {n_tasks} unique pairs"
                )
            self.n_components = C

            pairs = list(combinations(range(C), 2))
            pair_idx = torch.randperm(len(pairs), generator=rng)[:n_tasks].tolist()
            task_component_table = torch.zeros(n_tasks, C)
            for ti, pi in enumerate(pair_idx):
                i, j = pairs[pi]
                task_component_table[ti, i] = 1.0
                task_component_table[ti, j] = 1.0
            self._concept_table = task_component_table.numpy()

        # Expand each component into `task_size` consecutive latent dims.
        self._task_masks = torch.kron(task_component_table, torch.ones(task_size))
        self.latent_size = C * task_size

        self.enc = nn.Linear(input_size, self.latent_size, bias=False)
        nn.init.kaiming_uniform_(self.enc.weight, a=math.sqrt(5), generator=rng)

        # Pre-generate test set on CPU with a separate seeded generator
        # for reproducibility, then move to device.
        test_rng = torch.Generator().manual_seed(seed + 9999)
        self._test = [
            self._generate(t, test_size, test_rng) for t in range(n_tasks)
        ]
        self.enc.to(device)
        self._task_masks = self._task_masks.to(device)
        self._test = [(x.to(device), y.to(device)) for x, y in self._test]

    def concept_table(self) -> np.ndarray:
        """(n_tasks, n_components) binary concept-usage table."""
        return self._concept_table

    def _generate(
        self, task_idx: int, n: int, rng: Optional[torch.Generator] = None
    ) -> tuple:
        with torch.no_grad():
            m = self._task_masks[task_idx]
            if rng is not None:
                x = torch.randn(n, self.input_size, generator=rng)
            else:
                x = torch.randn(n, self.input_size, device=m.device)
            z = self.enc(x) * m
            z = torch.sin(self.frequency * z) * m
            if self.is_classification:
                y_raw = (z.sum(-1) > 0).long()
                y = F.one_hot(y_raw, num_classes=self.output_size).float()
            else:
                y = z.sum(-1, keepdim=True) / max(1.0, m.sum().item() / 2.0)
        return x, y

    def sample_batch(
        self,
        task_idx: int,
        batch_size: int = 256,
        split: str = "train",
        device: str = "cpu",
    ) -> tuple:
        if split == "test":
            x, y = self._test[task_idx]
            idx = torch.randperm(len(x), device=x.device)[:batch_size]
            return x[idx].to(device), y[idx].to(device)
        x, y = self._generate(task_idx, batch_size)
        return x.to(device), y.to(device)

    def evaluate(
        self,
        model: nn.Module,
        mask: torch.Tensor,
        task_idx: int,
        split: str = "test",
        batch_size: int = 1024,
        device: str = "cuda",
    ) -> float:
        mask = mask.to(device)
        model.eval()
        with torch.no_grad():
            x, y = self._test[task_idx]
            x, y = x.to(device), y.to(device)
            y_pred = model(x, mask)
            if self.is_classification:
                acc = (y_pred.argmax(-1) == y.argmax(-1)).float().mean().item()
                return acc
            return F.mse_loss(y_pred.squeeze(), y.squeeze()).item()


# ---------------------------------------------------------------------------
# 2. MNISTSharedShuffle — structured digit->label permutations
# ---------------------------------------------------------------------------


class MNISTSharedShuffle:
    """
    MNIST with partially-shared label permutations (concept shift).

    Define ``n_components`` concept components, each a partial digit ->
    label sub-mapping over ``component_size`` distinct digits.  Each task
    is the union of a pair of components: the task-specific permutation
    is a fresh random base permutation overridden on the digits covered
    by its two components with those components' shared mappings.

    Two tasks that share a component therefore agree on the digit ->
    label mapping within that component, providing realistic structured
    concept-sharing for backward-transfer evaluation.

    The (n_tasks, n_components) concept_table is binary: 1 iff the task
    is one of the two tasks containing that component.
    """

    is_classification = True
    input_size = 784
    output_size = 10

    def __init__(
        self,
        n_tasks: int = 6,
        n_components: int = 4,
        component_size: int = 5,
        seed: int = 0,
        data_dir: str = "./data",
    ) -> None:
        self.n_tasks = n_tasks
        self.n_components = n_components
        self.component_size = component_size
        self.data_dir = data_dir

        rng = np.random.default_rng(seed)

        # Build C components; each maps a chosen subset of digits to a
        # chosen subset of labels.
        self._components: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(n_components):
            digs = rng.choice(10, size=component_size, replace=False)
            labs = rng.choice(10, size=component_size, replace=False)
            self._components.append((digs, labs))

        pairs = list(combinations(range(n_components), 2))
        if len(pairs) < n_tasks:
            raise ValueError(
                f"n_components={n_components} can only form {len(pairs)} unique "
                f"pairs, but n_tasks={n_tasks}"
            )
        pair_idx = rng.permutation(len(pairs))[:n_tasks]
        self._pairs: list[tuple[int, int]] = [pairs[i] for i in pair_idx]

        table = np.zeros((n_tasks, n_components), dtype=np.float32)
        for ti, (a, b) in enumerate(self._pairs):
            table[ti, a] = 1.0
            table[ti, b] = 1.0
        self._concept_table = table

        # Build per-task explicit permutations.
        self.permutations: list[np.ndarray] = []
        for (a, b) in self._pairs:
            perm = rng.permutation(10)
            for c_idx in (a, b):
                digs, labs = self._components[c_idx]
                for d, l in zip(digs, labs):
                    perm[d] = l
            self.permutations.append(perm)

        self._train_x: Optional[torch.Tensor] = None
        self._test_x: Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None
        self._test_y: Optional[torch.Tensor] = None
        self._loaded = False
        self._gpu_device: Optional[str] = None

    def concept_table(self) -> np.ndarray:
        return self._concept_table

    def _load(self) -> None:
        if self._loaded:
            return
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                transforms.Lambda(lambda x: x.view(-1)),
            ]
        )
        train_ds = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=transform
        )
        test_ds = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=transform
        )
        self._train_x = torch.stack([x for x, _ in train_ds])
        self._train_y = torch.tensor([y for _, y in train_ds])
        self._test_x = torch.stack([x for x, _ in test_ds])
        self._test_y = torch.tensor([y for _, y in test_ds])
        self._loaded = True

    def _ensure_gpu(self, device: str) -> None:
        if self._gpu_device == device:
            return
        if "cuda" in device:
            self._train_x = self._train_x.to(device)
            self._train_y = self._train_y.to(device)
            self._test_x = self._test_x.to(device)
            self._test_y = self._test_y.to(device)
            self._gpu_device = device

    def _apply_perm(self, labels: torch.Tensor, task_idx: int) -> torch.Tensor:
        perm = torch.tensor(
            self.permutations[task_idx], dtype=torch.long, device=labels.device
        )
        return perm[labels]

    def sample_batch(
        self,
        task_idx: int,
        batch_size: int = 256,
        split: str = "train",
        device: str = "cpu",
    ) -> tuple:
        self._load()
        self._ensure_gpu(device)
        x_all = self._train_x if split == "train" else self._test_x
        y_all = self._train_y if split == "train" else self._test_y
        idx = torch.randint(0, len(x_all), (batch_size,), device=x_all.device)
        x = x_all[idx].to(device)
        y_raw = y_all[idx]
        y_perm = self._apply_perm(y_raw, task_idx)
        y = F.one_hot(y_perm.to(device), num_classes=10).float()
        return x, y

    def evaluate(
        self,
        model: nn.Module,
        mask: torch.Tensor,
        task_idx: int,
        split: str = "test",
        batch_size: int = 1000,
        device: str = "cuda",
    ) -> float:
        self._load()
        self._ensure_gpu(device)
        x_all = self._test_x if split == "test" else self._train_x
        y_all = self._test_y if split == "test" else self._train_y
        total, correct = 0, 0
        mask = mask.to(device)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(x_all), batch_size):
                x = x_all[start : start + batch_size].to(device)
                y_raw = y_all[start : start + batch_size]
                y_perm = self._apply_perm(y_raw, task_idx).to(device)
                y_pred = model(x, mask).argmax(-1)
                correct += (y_pred == y_perm).sum().item()
                total += len(x)
        return correct / total


# ---------------------------------------------------------------------------
# 3. PermutedMNIST — domain-incremental: each task is a fixed pixel permutation
# ---------------------------------------------------------------------------


class _MNISTBase:
    """Shared MNIST loader and per-task transformation hook."""

    is_classification = True
    input_size = 784
    output_size = 10

    def __init__(self, n_tasks: int, seed: int, data_dir: str) -> None:
        self.n_tasks = n_tasks
        self.data_dir = data_dir
        self._rng = np.random.default_rng(seed)
        self._train_x: Optional[torch.Tensor] = None
        self._test_x: Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None
        self._test_y: Optional[torch.Tensor] = None
        self._loaded = False
        self._gpu_device: Optional[str] = None

    def _load(self) -> None:
        if self._loaded:
            return
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_ds = datasets.MNIST(
            self.data_dir, train=True, download=True, transform=transform
        )
        test_ds = datasets.MNIST(
            self.data_dir, train=False, download=True, transform=transform
        )
        self._train_x = torch.stack([x for x, _ in train_ds])  # (N,1,28,28)
        self._train_y = torch.tensor([y for _, y in train_ds])
        self._test_x = torch.stack([x for x, _ in test_ds])
        self._test_y = torch.tensor([y for _, y in test_ds])
        self._loaded = True

    def _ensure_gpu(self, device: str) -> None:
        if self._gpu_device == device:
            return
        if "cuda" in device:
            self._train_x = self._train_x.to(device)
            self._train_y = self._train_y.to(device)
            self._test_x = self._test_x.to(device)
            self._test_y = self._test_y.to(device)
            self._gpu_device = device

    def _apply_task(
        self, x: torch.Tensor, task_idx: int, device: str
    ) -> torch.Tensor:
        raise NotImplementedError

    def sample_batch(
        self,
        task_idx: int,
        batch_size: int = 256,
        split: str = "train",
        device: str = "cpu",
    ) -> tuple:
        self._load()
        self._ensure_gpu(device)
        x_all = self._train_x if split == "train" else self._test_x
        y_all = self._train_y if split == "train" else self._test_y
        idx = torch.randint(0, len(x_all), (batch_size,), device=x_all.device)
        x = self._apply_task(x_all[idx], task_idx, device)
        y = F.one_hot(y_all[idx].to(device), num_classes=10).float()
        return x, y

    def evaluate(
        self,
        model: nn.Module,
        mask: torch.Tensor,
        task_idx: int,
        split: str = "test",
        batch_size: int = 1000,
        device: str = "cuda",
    ) -> float:
        self._load()
        self._ensure_gpu(device)
        x_all = self._test_x if split == "test" else self._train_x
        y_all = self._test_y if split == "test" else self._train_y
        total, correct = 0, 0
        mask = mask.to(device)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(x_all), batch_size):
                x = self._apply_task(x_all[start : start + batch_size], task_idx, device)
                y = y_all[start : start + batch_size].to(device)
                y_pred = model(x, mask).argmax(-1)
                correct += (y_pred == y).sum().item()
                total += len(x)
        return correct / total

    def concept_table(self) -> np.ndarray:
        # Domain-incremental MNIST has no shared concept structure to recover.
        return np.eye(self.n_tasks, dtype=np.float32)


class PermutedMNIST(_MNISTBase):
    """
    Domain-incremental MNIST with a fixed pixel permutation per task.

    Task 0 is the identity; tasks 1..n_tasks-1 each apply a distinct
    random pixel permutation (drawn once, deterministic given seed).
    All tasks share the same digit -> label mapping.
    """

    def __init__(
        self,
        n_tasks: int = 20,
        seed: int = 0,
        data_dir: str = "./data",
    ) -> None:
        super().__init__(n_tasks=n_tasks, seed=seed, data_dir=data_dir)
        # Task 0 is identity; subsequent tasks each get a distinct random
        # permutation of the 784 pixels.
        perms = [np.arange(784)]
        for _ in range(n_tasks - 1):
            perms.append(self._rng.permutation(784))
        self._perms = [torch.tensor(p, dtype=torch.long) for p in perms]

    def _apply_task(
        self, x: torch.Tensor, task_idx: int, device: str
    ) -> torch.Tensor:
        # x: (B, 1, 28, 28) -> flatten -> permute -> return flat (B, 784)
        flat = x.reshape(x.shape[0], -1).to(device)
        perm = self._perms[task_idx].to(device)
        return flat[:, perm]


class RotatedMNIST(_MNISTBase):
    """
    Domain-incremental MNIST with a fixed rotation angle per task.

    Angles span 0 to ``max_angle`` degrees in ``n_tasks`` equal steps.
    Task 0 is identity (0 degrees); subsequent tasks apply progressively
    larger rotations. All tasks share the same digit -> label mapping.
    """

    def __init__(
        self,
        n_tasks: int = 20,
        max_angle: float = 180.0,
        seed: int = 0,
        data_dir: str = "./data",
    ) -> None:
        super().__init__(n_tasks=n_tasks, seed=seed, data_dir=data_dir)
        self.max_angle = max_angle
        self._angles = [
            i * max_angle / max(1, n_tasks - 1) for i in range(n_tasks)
        ]

    def _apply_task(
        self, x: torch.Tensor, task_idx: int, device: str
    ) -> torch.Tensor:
        # x: (B, 1, 28, 28) on whatever device -> rotate -> flatten to (B, 784)
        from torchvision.transforms.functional import rotate
        x = x.to(device)
        angle = self._angles[task_idx]
        if angle != 0.0:
            x = rotate(x, angle)
        return x.reshape(x.shape[0], -1)
