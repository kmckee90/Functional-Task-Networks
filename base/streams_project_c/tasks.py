"""
tasks.py — Task/dataset generators for project_c experiments.

Four task families:
  1. SyntheticTasks    — DataGenSine: synthetic multi-task classification/regression.
  2. MNISTShuffled     — MNIST with permuted class labels (concept shift).
  3. PermutedMNIST     — MNIST with permuted pixel positions (domain shift).
  4. ElectricityELEC2  — Real Australian electricity-market data (UCI / OpenML id 151)
                         split into contiguous time regimes (tabular concept drift).

All task classes expose a consistent interface:
    task.sample_batch(task_idx, batch_size, split='train') → (x, y)
    task.evaluate(model, mask, task_idx, split='test')     → scalar metric
    task.n_tasks        : int
    task.input_size     : int
    task.output_size    : int
    task.is_classification : bool
"""

import math
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. SyntheticTasks (DataGenSine-based)
# ---------------------------------------------------------------------------

class SyntheticTasks:
    """
    Synthetic tasks using sinusoidal projections with task-specific masks.

    Each task applies a shared encoder to the input, masks the result
    with a task-specific binary vector, passes through sin(f * z),
    and decodes to classification or regression output.

    This provides clean, controlled experiments with tunable inter-task
    correlation and task difficulty.

    Args:
        input_size:       Dimensionality of x.
        output_size:      Number of classes (classification) or output dims (regression).
        n_tasks:          Number of tasks.
        task_size:        Hidden dimensions per task.
        n_samples:        Samples per batch.
        frequency:        Sin nonlinearity frequency (higher = harder).
        classification:   If True, binarise output to one-hot.
        correlated_tasks: If True, tasks share some hidden dims.
        seed:             RNG seed.
    """

    is_classification = True  # overridden per instance

    def __init__(
        self,
        input_size: int = 4,
        output_size: int = 2,
        n_tasks: int = 5,
        task_size: int = 4,
        n_samples: int = 256,
        frequency: float = 8.0,
        classification: bool = True,
        correlated_tasks: bool = False,
        seed: int = 0,
        device: str = "cpu",
        test_size: int = 2048,
    ) -> None:
        self.input_size    = input_size
        self.output_size   = output_size
        self.n_tasks       = n_tasks
        self.n_samples     = n_samples
        self.frequency     = frequency
        self.is_classification = classification
        self.device        = device

        rng = torch.Generator().manual_seed(seed)

        # Hidden space
        hidden_size = task_size * n_tasks * 2  # disjoint blocks

        # Disjoint task masks (orthogonal concept shift)
        self._task_masks = torch.kron(
            torch.eye(n_tasks),
            torch.ones(task_size * 2)
        )  # (n_tasks, hidden_size)

        if correlated_tasks:
            # Tasks share some dimensions (pairwise unions)
            from itertools import combinations
            pairs = list(combinations(range(n_tasks), 2))
            m = torch.eye(n_tasks)
            m = m[torch.tensor([list(p) for p in pairs])].max(1).values
            self._task_masks = torch.kron(m, torch.ones(task_size))

        # Fixed encoder/decoder — init weights on CPU (rng is a CPU generator),
        # then move to device along with everything else.
        self.enc = nn.Linear(input_size, hidden_size, bias=False)
        self.dec = nn.Linear(hidden_size, output_size, bias=False)
        nn.init.kaiming_uniform_(self.enc.weight, a=math.sqrt(5), generator=rng)
        nn.init.kaiming_uniform_(self.dec.weight, a=math.sqrt(5), generator=rng)

        # Pre-generate fixed test sets while enc/dec are still on CPU so the
        # seeded rng produces identical results regardless of device.
        test_rng = torch.Generator().manual_seed(seed + 9999)
        self._test = [
            self._generate(t, test_size, test_rng) for t in range(n_tasks)
        ]

        # Move everything to the target device once.
        self.enc.to(device)
        self.dec.to(device)
        self._task_masks = self._task_masks.to(device)
        self._test = [(x.to(device), y.to(device)) for x, y in self._test]

    def _generate(
        self,
        task_idx: int,
        n: int,
        rng: Optional[torch.Generator] = None,
    ) -> tuple:
        with torch.no_grad():
            m = self._task_masks[task_idx]
            # rng is a CPU generator — used only for reproducible test-set
            # pre-generation in __init__ (before device transfer).
            # Training batches pass rng=None and generate directly on device.
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
                y = z.sum(-1, keepdim=True) / max(1, m.sum().item() / 2)
        return x, y

    def sample_batch(
        self,
        task_idx: int,
        batch_size: int = 256,
        split: str = "train",
        device: str = "cpu",
    ) -> tuple:
        if split == "test":
            x, y = self._test[task_idx]  # already on self.device
            idx = torch.randperm(len(x), device=x.device)[:batch_size]
            return x[idx].to(device), y[idx].to(device)
        # Training: generate directly on self.device — no CPU→GPU copy.
        x, y = self._generate(task_idx, batch_size)
        return x.to(device), y.to(device)  # no-op when device == self.device

    def evaluate(
        self,
        model: nn.Module,
        mask: torch.Tensor,
        task_idx: int,
        split: str = "test",
        batch_size: int = 512,
        device: str = "cuda",
    ) -> float:
        x, y = self._test[task_idx]
        x, y = x.to(device), y.to(device)
        mask = mask.to(device)
        model.eval()
        with torch.no_grad():
            y_pred = model(x, mask)
            if self.is_classification:
                acc = (y_pred.argmax(-1) == y.argmax(-1)).float().mean().item()
                return acc
            else:
                mse = F.mse_loss(y_pred.squeeze(), y.squeeze()).item()
                return mse
        model.train()


# ---------------------------------------------------------------------------
# 2. MNISTShuffled — concept shift via label permutations
# ---------------------------------------------------------------------------

class MNISTShuffled:
    """
    MNIST dataset with shuffled class-label assignments per task.

    Each task applies a different permutation π_t : {0..9} → {0..9} to
    the true digit labels before one-hot encoding.  The input images are
    IDENTICAL across tasks; only the output semantics differ.

    This is a pure concept-shift benchmark: the same visual input maps
    to a different correct answer depending on which task is active.
    A model without masking will catastrophically forget task t-1's
    label mapping when trained on task t.

    Args:
        n_tasks:   Number of different shufflings (default 5).
        seed:      RNG seed.
        data_dir:  Directory to cache MNIST data.
    """

    is_classification = True
    input_size  = 784
    output_size = 10

    def __init__(
        self,
        n_tasks: int = 5,
        seed: int = 0,
        data_dir: str = "./data",
    ) -> None:
        self.n_tasks  = n_tasks
        self.data_dir = data_dir

        # Build per-task label permutations
        rng = np.random.default_rng(seed)
        self.permutations = [np.arange(10)]  # task 0 = standard labels
        for _ in range(n_tasks - 1):
            perm = rng.permutation(10)
            self.permutations.append(perm)

        self._train_x: Optional[torch.Tensor] = None
        self._test_x:  Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None
        self._test_y:  Optional[torch.Tensor] = None
        self._loaded  = False

    def _load(self) -> None:
        if self._loaded:
            return
        from torchvision import datasets, transforms
        # Normalize BEFORE flattening — torchvision Normalize requires image tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),  # flatten 28×28 → 784
        ])
        train_ds = datasets.MNIST(self.data_dir, train=True,  download=True, transform=transform)
        test_ds  = datasets.MNIST(self.data_dir, train=False, download=True, transform=transform)

        self._train_x = torch.stack([x for x, _ in train_ds])
        self._train_y = torch.tensor([y for _, y in train_ds])
        self._test_x  = torch.stack([x for x, _ in test_ds])
        self._test_y  = torch.tensor([y for _, y in test_ds])
        self._loaded  = True
        self._gpu_device: Optional[str] = None  # will be set on first GPU request

    def _ensure_gpu(self, device: str) -> None:
        """Move MNIST tensors to GPU once for fast sampling."""
        if self._gpu_device == device:
            return
        if "cuda" in device:
            self._train_x = self._train_x.to(device)
            self._train_y = self._train_y.to(device)
            self._test_x  = self._test_x.to(device)
            self._test_y  = self._test_y.to(device)
            self._gpu_device = device

    def _apply_perm(self, labels: torch.Tensor, task_idx: int) -> torch.Tensor:
        """Apply task t's permutation to a tensor of digit labels."""
        perm = torch.tensor(self.permutations[task_idx], dtype=torch.long,
                            device=labels.device)
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
        x_all = self._test_x if split == "test" else self._train_x
        y_all = self._test_y if split == "test" else self._train_y

        total, correct = 0, 0
        mask = mask.to(device)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(x_all), batch_size):
                x = x_all[start:start+batch_size].to(device)
                y_raw = y_all[start:start+batch_size]
                y_perm = self._apply_perm(y_raw, task_idx).to(device)
                y_pred = model(x, mask).argmax(-1)
                correct += (y_pred == y_perm).sum().item()
                total   += len(x)
        model.train()
        return correct / total


# ---------------------------------------------------------------------------
# 3. PermutedMNIST — domain shift via pixel permutations
# ---------------------------------------------------------------------------

class PermutedMNIST:
    """
    MNIST with a different fixed pixel permutation applied per task.

    Unlike MNISTShuffled, the labels are unchanged — each task is
    still standard digit recognition but the pixels are scrambled
    differently.  This tests resilience to feature-distribution shift
    in a block-sequential CL protocol.

    Standard benchmark in the CL literature (Kirkpatrick et al. 2017,
    Zenke et al. 2017, etc.).

    Args:
        n_tasks:   Number of permutations (default 10).
        seed:      RNG seed.
        data_dir:  Directory to cache MNIST data.
    """

    is_classification = True
    input_size  = 784
    output_size = 10

    def __init__(
        self,
        n_tasks: int = 10,
        seed: int = 0,
        data_dir: str = "./data",
    ) -> None:
        self.n_tasks  = n_tasks
        self.data_dir = data_dir

        rng = np.random.default_rng(seed)
        # Task 0: identity permutation; tasks 1+ are random
        self.permutations = [np.arange(784)]
        for _ in range(n_tasks - 1):
            self.permutations.append(rng.permutation(784))

        self._train_x: Optional[torch.Tensor] = None
        self._test_x:  Optional[torch.Tensor] = None
        self._train_y: Optional[torch.Tensor] = None
        self._test_y:  Optional[torch.Tensor] = None
        self._loaded  = False

    def _load(self) -> None:
        if self._loaded:
            return
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Lambda(lambda x: x.view(-1)),
        ])
        train_ds = datasets.MNIST(self.data_dir, train=True,  download=True, transform=transform)
        test_ds  = datasets.MNIST(self.data_dir, train=False, download=True, transform=transform)

        self._train_x = torch.stack([x for x, _ in train_ds])
        self._train_y = torch.tensor([y for _, y in train_ds])
        self._test_x  = torch.stack([x for x, _ in test_ds])
        self._test_y  = torch.tensor([y for _, y in test_ds])
        self._loaded  = True
        self._gpu_device: Optional[str] = None

    def _ensure_gpu(self, device: str) -> None:
        if self._gpu_device == device:
            return
        if "cuda" in device:
            self._train_x = self._train_x.to(device)
            self._train_y = self._train_y.to(device)
            self._test_x  = self._test_x.to(device)
            self._test_y  = self._test_y.to(device)
            self._gpu_device = device

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
        x = x_all[idx]
        perm = torch.tensor(self.permutations[task_idx], dtype=torch.long,
                            device=x_all.device)
        x = x[:, perm]
        y = F.one_hot(y_all[idx], num_classes=10).float()
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

        perm = torch.tensor(self.permutations[task_idx], dtype=torch.long,
                            device=x_all.device)
        total, correct = 0, 0
        mask = mask.to(device)
        model.eval()
        with torch.no_grad():
            for start in range(0, len(x_all), batch_size):
                x = x_all[start:start+batch_size][:, perm]
                y = y_all[start:start+batch_size]
                y_pred = model(x, mask).argmax(-1)
                correct += (y_pred == y).sum().item()
                total   += len(x)
        model.train()
        return correct / total


# ---------------------------------------------------------------------------
# 4. ElectricityELEC2 — real tabular concept drift
# ---------------------------------------------------------------------------

class ElectricityELEC2:
    """
    Real Australian electricity-market dataset (ELEC2 / UCI Electricity,
    OpenML dataset id 151; Harries 1999).  Widely used as a canonical
    concept-drift benchmark (Gama et al. 2014; Bifet et al. MOA).

    Data:
        45 312 half-hourly records from NSW and VIC electricity markets
        between 07 May 1996 and 05 Dec 1998.
        Target: binary label (UP / DOWN) — whether the NSW price rose
        relative to a 24-hour moving average.

    Concept-drift task structure:
        The rows are chronologically ordered; we split them into
        `n_tasks` contiguous equal-sized chunks, each treated as one
        "regime" in the block-sequential CL protocol.  Distribution over
        features and over the feature → label mapping drifts across
        chunks because the underlying market dynamics (supply mix,
        demand patterns, price-setting rules) evolved during the
        observation window.

    Timestamp-leakage warning:
        Žliobaitė (2013) shows that the raw `date`, `day`, and `period`
        columns leak the time-of-day / day-of-week cycles that are
        correlated with the (time-defined) label.  A model can "solve"
        ELEC2 by simply memorising the hour of day.  To avoid this
        trivial shortcut we DROP those three temporal columns and keep
        only the market-price and market-demand features:

            nswprice   NSW electricity spot price
            nswdemand  NSW electricity demand
            vicprice   VIC electricity spot price
            vicdemand  VIC electricity demand
            transfer   scheduled NSW↔VIC transfer

        These 5 features match the previous (synthetic) input dimension,
        so the rest of the pipeline is unchanged.  All features are
        already normalised to [0, 1] by the OpenML distributor.

    Args:
        n_tasks:    Number of contiguous time-regime tasks (default 4).
        seed:       RNG seed (affects the 80/20 train/test split within
                    each chunk — the task ordering is deterministic).
        data_dir:   OpenML cache directory.
    """

    is_classification = True
    input_size   = 5
    output_size  = 2

    # Indices of the 5 non-temporal (leakage-free) features
    _FEATURE_COLS = ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer"]
    _DROPPED_COLS = ["date", "day", "period"]  # dropped per Žliobaitė 2013

    def __init__(
        self,
        n_tasks: int = 4,
        seed: int = 0,
        data_dir: str = "./data/openml",
    ) -> None:
        self.n_tasks = n_tasks
        self.data_dir = data_dir
        self.seed = seed
        self._gpu_device: Optional[str] = None
        self._train, self._test = self._load_and_split(n_tasks, seed, data_dir)

    def _load_and_split(self, n_tasks: int, seed: int, data_dir: str):
        from sklearn.datasets import fetch_openml

        os.makedirs(data_dir, exist_ok=True)
        ds = fetch_openml(
            name="electricity", version=1, as_frame=True,
            cache=True, data_home=data_dir,
        )
        df = ds.data
        x_all = df[self._FEATURE_COLS].to_numpy().astype(np.float32)
        y_all = (ds.target.to_numpy() == "UP").astype(np.int64)

        x_all_t = torch.from_numpy(x_all)
        y_all_t = torch.from_numpy(y_all)

        # Split into n_tasks contiguous chunks (concept-drift regimes)
        n = len(x_all_t)
        rng = torch.Generator().manual_seed(seed)

        train_tasks: list[tuple[torch.Tensor, torch.Tensor]] = []
        test_tasks:  list[tuple[torch.Tensor, torch.Tensor]] = []
        for t in range(n_tasks):
            start = t * n // n_tasks
            end   = (t + 1) * n // n_tasks
            xs = x_all_t[start:end]
            ys = y_all_t[start:end]

            # 80/20 train/test split within the chunk, shuffled with `seed`
            perm = torch.randperm(len(xs), generator=rng)
            split = int(0.8 * len(xs))
            tr_idx, te_idx = perm[:split], perm[split:]
            y_oh_tr = F.one_hot(ys[tr_idx], num_classes=2).float()
            y_oh_te = F.one_hot(ys[te_idx], num_classes=2).float()
            train_tasks.append((xs[tr_idx], y_oh_tr))
            test_tasks.append((xs[te_idx], y_oh_te))

        return train_tasks, test_tasks

    def _ensure_gpu(self, device: str) -> None:
        if self._gpu_device == device:
            return
        if "cuda" in device:
            self._train = [(x.to(device), y.to(device)) for x, y in self._train]
            self._test  = [(x.to(device), y.to(device)) for x, y in self._test]
            self._gpu_device = device

    def sample_batch(
        self,
        task_idx: int,
        batch_size: int = 256,
        split: str = "train",
        device: str = "cpu",
    ) -> tuple:
        self._ensure_gpu(device)
        x_all, y_all = (self._train if split == "train" else self._test)[task_idx]
        idx = torch.randint(0, len(x_all), (batch_size,), device=x_all.device)
        return x_all[idx].to(device), y_all[idx].to(device)

    def evaluate(
        self,
        model: nn.Module,
        mask: torch.Tensor,
        task_idx: int,
        split: str = "test",
        batch_size: int = 512,
        device: str = "cuda",
    ) -> float:
        self._ensure_gpu(device)
        x_all, y_all = (self._test if split == "test" else self._train)[task_idx]
        x_all = x_all.to(device)
        y_all = y_all.to(device)
        mask  = mask.to(device)

        total, correct = 0, 0
        model.eval()
        with torch.no_grad():
            for start in range(0, len(x_all), batch_size):
                xb = x_all[start:start+batch_size]
                yb = y_all[start:start+batch_size]
                pred = model(xb, mask).argmax(-1)
                correct += (pred == yb.argmax(-1)).sum().item()
                total   += len(xb)
        model.train()
        return correct / total


# Back-compat shim: some scripts still import EnergyRegime by name
EnergyRegime = ElectricityELEC2
