"""
tasks.py - Data generation for masked network experiments.

Contains task/data generators that produce (x, y) pairs for training.
Each task represents a different input-output mapping that the model
must learn to handle via mask reconfiguration.
"""

import math
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DataGen:
    """
    Generates classification tasks with controllable correlation structure.

    Each task uses a subset of hidden dimensions (defined by mask 'm') to
    transform inputs x into outputs y. Tasks can share dimensions (correlated)
    or use disjoint dimensions (orthogonal).

    Args:
        input_dim: Dimension of input features x
        output_dim: Dimension of output (number of classes)
        n_tasks: Number of distinct tasks
        task_size: Base size for task-specific hidden dimensions
        n_samples: Number of samples per batch
        data_temp: Temperature for softmax output generation
        correlated_tasks: If True, tasks share some hidden dimensions
        concept_shift: If True, different tasks use different hidden subsets
        domain_shift_factor: Scale factor for input distribution shift per task
        rng: Optional torch.Generator for local reproducibility. When provided,
             all random operations use this generator instead of the global RNG.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_tasks: int,
        task_size: int = 1,
        n_samples: int = 2048,
        data_temp: float = 1e-8,
        correlated_tasks: bool = True,
        concept_shift: bool = True,
        domain_shift_factor: float = 0.0,
        rng: torch.Generator = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_tasks = n_tasks
        self.n_samples = n_samples
        self.task_size = task_size
        self.data_temp = data_temp
        self.hidden_size = task_size * n_tasks
        self.rng = rng

        # Build task masks defining which hidden dims each task uses
        if concept_shift:
            if correlated_tasks:
                # Tasks share dimensions via pairwise combinations
                m = torch.eye(n_tasks)
                pairs = list(combinations(range(n_tasks), 2))
                m = m[pairs, :].max(1)[0]
                self.m = torch.kron(m, torch.ones(task_size))
                # Shuffle for variety
                self.m = self.m[torch.randperm(len(m), generator=rng)]
            else:
                # Tasks use disjoint dimensions
                self.hidden_size *= 2
                self.m = torch.kron(torch.eye(n_tasks), torch.ones(task_size * 2))
        else:
            # All tasks use same dimensions
            self.m = torch.ones(task_size * 2)
            self.hidden_size = len(self.m)

        # Linear projections: input -> hidden -> output
        self.enc = nn.Linear(self.input_dim, self.hidden_size, bias=False)
        self.dec = nn.Linear(self.hidden_size, self.output_dim, bias=False)

        # Reinitialize weights with local RNG if provided
        if rng is not None:
            nn.init.kaiming_uniform_(self.enc.weight, a=math.sqrt(5), generator=rng)
            nn.init.kaiming_uniform_(self.dec.weight, a=math.sqrt(5), generator=rng)

        # Domain shift per task dimension
        self.dshift = torch.arange(len(self.m)).float()
        self.dshift = (self.dshift - self.dshift.mean()) * domain_shift_factor

    def __call__(self, task_idx: int, seed: int = None, temp: float = None):
        """
        Generate data for a specific task.

        Args:
            task_idx: Index of the task to generate data for
            seed: Random seed for reproducibility (sets global RNG; prefer
                  passing rng to __init__ instead)
            is_test: If True, use test mask instead of task mask
            temp: Temperature for softmax (uses self.data_temp if None)

        Returns:
            x: Input tensor of shape (n_samples, input_dim)
            y: One-hot output tensor of shape (n_samples, output_dim)
        """
        if temp is None:
            temp = self.data_temp
        if seed is not None:
            if self.rng is not None:
                self.rng.manual_seed(seed)
            else:
                torch.manual_seed(seed)

        with torch.no_grad():
            # Generate random inputs with optional domain shift
            m = self.m[task_idx]
            # x = torch.rand((self.n_samples, self.input_dim), generator=self.rng) * 4 - 2
            x = torch.randn(self.n_samples, self.input_dim, generator=self.rng)
            x = x + self.dshift[task_idx]
            z = self.enc(x) * m

            # Simple decoder
            # y = (z.sum(-1) > 0).float().reshape(-1, 1)
            # y = torch.cat((y, 1 - y), -1)

            # Original decoder
            y = self.dec(z)
            probs = torch.softmax(y / temp, dim=-1)
            y = torch.multinomial(probs, 1, generator=self.rng).squeeze(-1)
            y = F.one_hot(y, num_classes=self.output_dim).float()
        return x, y


class DataGenSine:
    """
    Generates classification tasks with controllable correlation structure.

    Each task uses a subset of hidden dimensions (defined by mask 'm') to
    transform inputs x into outputs y. Tasks can share dimensions (correlated)
    or use disjoint dimensions (orthogonal).

    Args:
        input_dim: Dimension of input features x
        output_dim: Dimension of output (number of classes)
        n_tasks: Number of distinct tasks
        task_size: Base size for task-specific hidden dimensions
        n_samples: Number of samples per batch
        data_temp: Temperature for softmax output generation
        correlated_tasks: If True, tasks share some hidden dimensions
        concept_shift: If True, different tasks use different hidden subsets
        domain_shift_factor: Scale factor for input distribution shift per task
        rng: Optional torch.Generator for local reproducibility. When provided,
             all random operations use this generator instead of the global RNG.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_tasks: int,
        task_size: int = 1,
        n_samples: int = 2048,
        data_temp: float = 1e-8,
        frequency=8,
        classify: bool = False,
        correlated_tasks: bool = True,
        concept_shift: bool = True,
        domain_shift_factor: float = 0.0,
        rng: torch.Generator = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_tasks = n_tasks
        self.n_samples = n_samples
        self.task_size = task_size
        self.data_temp = data_temp
        self.hidden_size = task_size * n_tasks
        self.rng = rng
        self.frequency = frequency
        self.classify = classify

        # Build task masks defining which hidden dims each task uses
        if concept_shift:
            if correlated_tasks:
                # Tasks share dimensions via pairwise combinations
                m = torch.eye(n_tasks)
                pairs = list(combinations(range(n_tasks), 2))
                m = m[pairs, :].max(1)[0]
                self.m = torch.kron(m, torch.ones(task_size))
                # Shuffle for variety
                self.m = self.m[torch.randperm(len(m), generator=rng)]
            else:
                # Tasks use disjoint dimensions
                self.hidden_size *= 2
                self.m = torch.kron(torch.eye(n_tasks), torch.ones(task_size * 2))
        else:
            # All tasks use same dimensions
            self.m = torch.ones(task_size * 2)
            self.hidden_size = len(self.m)

        # Linear projections: input -> hidden -> output
        self.enc = nn.Linear(self.input_dim, self.hidden_size, bias=False)
        self.dec = nn.Linear(self.hidden_size, self.output_dim, bias=False)

        # Reinitialize weights with local RNG if provided
        if rng is not None:
            nn.init.kaiming_uniform_(self.enc.weight, a=math.sqrt(5), generator=rng)
            nn.init.kaiming_uniform_(self.dec.weight, a=math.sqrt(5), generator=rng)

        # Domain shift per task dimension
        self.dshift = torch.arange(len(self.m)).float()
        self.dshift = (self.dshift - self.dshift.mean()) * domain_shift_factor

    def __call__(self, task_idx: int, seed: int = None, temp: float = None):
        """
        Generate data for a specific task.

        Args:
            task_idx: Index of the task to generate data for
            seed: Random seed for reproducibility (sets global RNG; prefer
                  passing rng to __init__ instead)
            is_test: If True, use test mask instead of task mask
            temp: Temperature for softmax (uses self.data_temp if None)

        Returns:
            x: Input tensor of shape (n_samples, input_dim)
            y: One-hot output tensor of shape (n_samples, output_dim)
        """
        if temp is None:
            temp = self.data_temp
        if seed is not None:
            if self.rng is not None:
                self.rng.manual_seed(seed)
            else:
                torch.manual_seed(seed)

        with torch.no_grad():
            # Generate random inputs with optional domain shift
            m = self.m[task_idx]
            # x = torch.rand( #     (self.n_samples, self.input_dim), generator=self.rng # ) * torch.pi - (torch.pi / 2)
            x = torch.randn(self.n_samples, self.input_dim, generator=self.rng)

            x = x + self.dshift[task_idx]
            z = self.enc(x) * m
            z = torch.sin(self.frequency * z) * m

            # y = self.dec(z)
            if self.classify:
                # probs = torch.softmax(y / temp, dim=-1)
                # y = torch.multinomial(probs, 1, generator=self.rng).squeeze(-1)
                # y = F.one_hot(y, num_classes=self.output_dim).float()
                # Simple decoder
                y = (z.sum(-1) > 0).float().reshape(-1, 1)
                y = torch.cat((y, 1 - y), -1)
            else:
                y = z.sum(-1).unsqueeze(1) / self.task_size

        return x, y


class DataGenPiecewise(DataGenSine):
    """One latent-sinusoid function, split into N contiguous input sub-domains.

    Unlike DataGenSine (where each task_idx selects a different latent subset,
    i.e. a different function), here EVERY task_idx is the SAME function; the
    index instead selects a contiguous slab of the input domain along SPLIT_DIM.

    The function itself is the identical latent-sinusoid transform used by
    DataGenSine:  y = decode( sin(frequency * enc(x)) ), summed over
    n_components latent components. Only the input sampling differs: inputs are
    drawn UNIFORMLY to fill each sub-domain's slab of a bounded box, rather than
    from a centered Gaussian (which would leave the outer sub-domains modelling
    only sparse tails).
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        n_domains,
        n_components,
        frequency=8,
        classify=False,
        n_samples=256,
        domain_half=math.pi,
        split_dim=0,
        data_temp=1e-8,
        rng: torch.Generator = None,
    ):
        # NOTE: we deliberately do NOT call DataGenSine.__init__ -- its mask
        # construction assumes multiple *functions* (n_tasks) and breaks for a
        # single shared function. We set up the same enc/dec machinery directly.
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_tasks = n_domains
        self.n_samples = n_samples
        self.task_size = n_components  # used by the regression normalization
        self.data_temp = data_temp
        self.frequency = frequency
        self.classify = classify
        self.hidden_size = n_components
        self.rng = rng

        self.n_domains = n_domains
        self.domain_half = domain_half
        self.split_dim = split_dim
        # Edges of the N equal-width sub-domains along `split_dim`.
        self.edges = torch.linspace(-domain_half, domain_half, n_domains + 1)

        # One shared latent function: every component active for every domain.
        self.m = torch.ones(n_domains, n_components)

        # Latent-sinusoid projections (same construction as DataGenSine).
        self.enc = nn.Linear(input_dim, n_components, bias=False)
        self.dec = nn.Linear(n_components, output_dim, bias=False)
        if rng is not None:
            nn.init.kaiming_uniform_(self.enc.weight, a=math.sqrt(5), generator=rng)
            nn.init.kaiming_uniform_(self.dec.weight, a=math.sqrt(5), generator=rng)

        # No additive per-task domain shift: the domain split provides locality.
        self.dshift = torch.zeros(n_domains)

    def transform(self, x):
        """Apply the shared latent-sinusoid function to inputs x -> targets y.

        Identical math to DataGenSine.__call__, factored out so it can be reused
        for dense visualization over the domain.
        """
        m = self.m[0]  # shared function; all rows are identical
        z = self.enc(x) * m
        z = torch.sin(self.frequency * z) * m
        if self.classify:
            y = (z.sum(-1) > 0).float().reshape(-1, 1)
            y = torch.cat((y, 1 - y), -1)
        else:
            y = z.sum(-1).unsqueeze(1) / self.task_size
        return y

    def _sample_domain(self, task_idx):
        """Uniformly sample inputs that fill sub-domain `task_idx`'s slab.

        Every axis spans the bounded box [-domain_half, domain_half] uniformly;
        the split axis is further restricted to this sub-domain's contiguous
        interval so each sub-domain is uniformly (not Gaussian-) covered.
        """
        x = (
            torch.rand(self.n_samples, self.input_dim, generator=self.rng)
            * (2 * self.domain_half)
            - self.domain_half
        )
        lo = self.edges[task_idx]
        hi = self.edges[task_idx + 1]
        col = torch.rand(self.n_samples, generator=self.rng) * (hi - lo) + lo
        x[:, self.split_dim] = col
        return x

    def __call__(self, task_idx, seed=None, temp=None):
        if seed is not None:
            if self.rng is not None:
                self.rng.manual_seed(seed)
            else:
                torch.manual_seed(seed)
        with torch.no_grad():
            x = self._sample_domain(task_idx)
            y = self.transform(x)
        return x, y


def make_rng(seed):
    """Create a local torch.Generator on the current default device."""
    device = torch.empty(0).device
    return torch.Generator(device=device).manual_seed(seed)


if __name__ == "__main__":
    import config

    cfg = config.Default()

    # CLASSIFICATION
    dg = DataGenSine(
        cfg.X_DIM,
        output_dim=cfg.Y_DIM,
        n_tasks=cfg.N_TASK,
        task_size=cfg.TASK_SIZE,
        frequency=cfg.FREQUENCY,
        classify=cfg.CLASSIFICATION,
        n_samples=4096 * 4,
        correlated_tasks=cfg.CORRELATED_TASKS,
        rng=make_rng(cfg.SEED + 100),
    )

    import numpy as np
    from PIL import Image

    res = 64
    half_pi = math.pi / 2
    grids = []

    for task_idx in range(3):
        xy, val = dg(task_idx)
        grid = np.full((res, res), 128, dtype=np.uint8)

        ix = ((xy[:, 0] + half_pi) / math.pi * res).clamp(0, res - 1).long()
        iy = ((xy[:, 1] + half_pi) / math.pi * res).clamp(0, res - 1).long()

        if dg.classify:
            pixel = (val.argmax(dim=1).float() * 255).byte().cpu().numpy()
        else:
            v = val[:, 0].cpu()
            v = (v - v.min()) / (v.max() - v.min() + 1e-8)
            pixel = (v * 255).byte().numpy()
        grid[iy.cpu(), ix.cpu()] = pixel
        grids.append(grid)
        if task_idx < 2:
            grids.append(np.zeros((res, 2), dtype=np.uint8))

    combined = np.concatenate(grids, axis=1)
    img = Image.fromarray(combined, mode="L")
    img = img.resize((256 * 3, 256), Image.NEAREST)
    img.save("task_vis.png")
    print("Saved task_vis.png")
