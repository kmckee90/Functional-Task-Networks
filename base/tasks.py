"""
tasks.py - Data generation for FTN experiments (stand-alone copy).

DataGenSine produces (x, y) pairs per task. Each task uses a subset of hidden
dimensions of a random generator network; tasks can share dimensions
(correlated) or use disjoint ones.
"""

import math
from itertools import combinations

import torch
import torch.nn as nn


def make_rng(seed):
    """Create a local torch.Generator on the current default device."""
    device = torch.empty(0).device
    return torch.Generator(device=device).manual_seed(seed)


class DataGenSine:
    """
    Sine-nonlinearity task generator with controllable correlation structure.

    Each task uses a subset of hidden dimensions (defined by mask 'm') to
    transform inputs x into outputs y through y ~ sin(frequency * enc(x)).

    Args:
        input_dim: Dimension of input features x
        output_dim: Dimension of output (n classes for classification, else 1)
        n_tasks: Number of distinct tasks
        task_size: Base size for task-specific hidden dimensions
        n_samples: Number of samples per batch
        data_temp: Temperature for softmax output generation (classification)
        frequency: Frequency of the sine nonlinearity; higher = harder targets
        classify: If True, binary one-hot labels; else scalar regression target
        correlated_tasks: If True, tasks share some hidden dimensions
        concept_shift: If True, different tasks use different hidden subsets
        domain_shift_factor: Scale factor for input distribution shift per task
        domain_shift: If True, dimensional domain shift: each task's input_dim
            values occupy their own slice of a wider input vector (width
            (n_tasks-1)*stride + input_dim), offset by task_idx * stride,
            zeros elsewhere. The task function itself is unchanged.
        domain_shift_stride: Element-index shift between consecutive tasks'
            slices. 0 = concept shift (all tasks share the same elements);
            >= input_dim = fully disjoint (orthogonal) input supports.
        domain_dense_projection: If True, apply a fixed random orthogonal
            projection (same dimension) to the final input vector, making the
            sparse domain-shifted inputs dense while preserving their
            information (a rotation of the domain).
        domain_per_task_projection: If True, use n_tasks distinct projections
            (one per task); if False, one shared projection for all tasks.
        rng: Optional torch.Generator for local reproducibility
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
        domain_shift: bool = False,
        domain_shift_stride: int = 0,
        domain_dense_projection: bool = False,
        domain_per_task_projection: bool = False,
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

        # Dimensional domain shift: task t's input occupies elements
        # [t*stride, t*stride + input_dim) of a fixed-width vector sized so
        # every task fits. The model consumes total_input_dim, not input_dim.
        self.domain_shift = domain_shift
        self.domain_shift_stride = domain_shift_stride
        if domain_shift:
            self.total_input_dim = (n_tasks - 1) * domain_shift_stride + input_dim
        else:
            self.total_input_dim = input_dim

        # Fixed random orthogonal projection(s) applied to the final input:
        # makes sparse domain-shifted inputs dense without losing information
        # (a rotation of the domain). One per task, or one shared.
        self.domain_dense_projection = domain_dense_projection
        self.domain_per_task_projection = domain_per_task_projection
        if domain_dense_projection:
            n_proj = n_tasks if domain_per_task_projection else 1
            d = self.total_input_dim
            self.proj = torch.stack(
                [
                    torch.linalg.qr(torch.randn(d, d, generator=rng))[0]
                    for _ in range(n_proj)
                ]
            )

    def __call__(self, task_idx: int, seed: int = None, temp: float = None):
        """
        Generate one batch of data for a task.

        Returns:
            x: Input tensor (n_samples, input_dim)
            y: One-hot targets (n_samples, output_dim) if classify,
               else scalars (n_samples, 1)
        """
        if temp is None:
            temp = self.data_temp
        if seed is not None:
            if self.rng is not None:
                self.rng.manual_seed(seed)
            else:
                torch.manual_seed(seed)

        with torch.no_grad():
            m = self.m[task_idx]
            x = torch.randn(self.n_samples, self.input_dim, generator=self.rng)
            x = x + self.dshift[task_idx]
            z = self.enc(x) * m
            z = torch.sin(self.frequency * z) * m

            if self.classify:
                y = (z.sum(-1) > 0).float().reshape(-1, 1)
                y = torch.cat((y, 1 - y), -1)
            else:
                y = z.sum(-1).unsqueeze(1) / self.task_size

            # Embed the task-local input into its slice of the full vector.
            if self.domain_shift:
                off = task_idx * self.domain_shift_stride
                x_full = torch.zeros(self.n_samples, self.total_input_dim)
                x_full[:, off : off + self.input_dim] = x
                x = x_full

            # Rotate into a dense domain (shared or per-task projection).
            if self.domain_dense_projection:
                p = self.proj[task_idx if self.domain_per_task_projection else 0]
                x = x @ p.T

        return x, y


def save_task_vis(cfg, path="task_vis.png", n_show=3, res=64):
    """Render the first tasks' target functions to a PNG (experiment aid)."""
    import numpy as np
    from PIL import Image

    dg = DataGenSine(
        cfg.X_DIM,
        output_dim=cfg.Y_DIM,
        n_tasks=cfg.N_TASK,
        task_size=cfg.TASK_SIZE,
        frequency=cfg.FREQUENCY,
        classify=cfg.CLASSIFICATION,
        n_samples=4096 * 4,
        correlated_tasks=cfg.CORRELATED_TASKS,
        domain_shift=getattr(cfg, "DOMAIN_SHIFT", False),
        domain_shift_stride=getattr(cfg, "DOMAIN_SHIFT_STRIDE", 0),
        # DOMAIN_DENSE_PROJECTION intentionally omitted: the vis plots each
        # task's function over its own 2-D input slice, which a dense
        # rotation would scramble. The underlying task is identical.
        rng=make_rng(cfg.SEED + 100),
    )

    half_pi = math.pi / 2
    grids = []
    n_show = min(n_show, cfg.N_TASK)
    for task_idx in range(n_show):
        xy, val = dg(task_idx)
        grid = np.full((res, res), 128, dtype=np.uint8)

        # Plot over the task's own input slice (offset 0 without domain shift).
        off = task_idx * dg.domain_shift_stride if dg.domain_shift else 0
        ix = ((xy[:, off] + half_pi) / math.pi * res).clamp(0, res - 1).long()
        iy = ((xy[:, off + 1] + half_pi) / math.pi * res).clamp(0, res - 1).long()

        if dg.classify:
            pixel = (val.argmax(dim=1).float() * 255).byte().cpu().numpy()
        else:
            v = val[:, 0].cpu()
            v = (v - v.min()) / (v.max() - v.min() + 1e-8)
            pixel = (v * 255).byte().numpy()
        grid[iy.cpu(), ix.cpu()] = pixel
        grids.append(grid)
        if task_idx < n_show - 1:
            grids.append(np.zeros((res, 2), dtype=np.uint8))

    combined = np.concatenate(grids, axis=1)
    img = Image.fromarray(combined, mode="L")
    img = img.resize((256 * n_show, 256), Image.NEAREST)
    img.save(path)
    print(f"Saved {path}")


if __name__ == "__main__":
    import config

    save_task_vis(config.Default())
