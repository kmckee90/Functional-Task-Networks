"""
masks.py - Mask configuration for masked network experiments.

Contains mask configurers that take a current mask 'm' and reconfigure it
for a new task. Mask reconfiguration involves gradient-based optimization
followed by lateral dynamics and sparsification (k-winner-take-all).
"""

import torch
from reconfig_optimizer import make_reconfig_optimizer, validate_reconfig_optimizer
import torch.nn as nn
import torch.nn.functional as F


def kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    K-winner-take-all sparsification.

    Returns a binary mask with 1s at the top-k values and 0s elsewhere.

    Args:
        x: Input tensor of shape (n,)
        k: Number of winners to select

    Returns:
        Binary mask tensor of shape (n,)
    """
    idx = torch.topk(x, k).indices
    out = torch.zeros_like(x)
    out[idx] = 1
    return out


def normalize_mask(m: torch.Tensor) -> torch.Tensor:
    """
    Normalize mask to [0, 1] range.

    Args:
        m: Mask tensor

    Returns:
        Normalized mask tensor
    """
    m = m - m.min()
    m = m / m.max()
    return m


def mask_to_grid(m: torch.Tensor, block_dim: int) -> torch.Tensor:
    """
    Reshape a flat mask into 2D by arranging block_dim x block_dim blocks
    side by side. For a single block this is just reshape(block_dim, block_dim).
    For multiple blocks it produces (block_dim, n_blocks * block_dim).

    Args:
        m: Flat mask tensor of shape (n_blocks * block_dim * block_dim,)
        block_dim: Side length of each square block

    Returns:
        2D tensor of shape (block_dim, n_blocks * block_dim)
    """
    block_size = block_dim * block_dim
    blocks = m.split(block_size)
    return torch.cat([b.reshape(block_dim, block_dim) for b in blocks], dim=-1)


def mask_to_vec(grid: torch.Tensor, block_dim: int) -> torch.Tensor:
    """
    Inverse of mask_to_grid. Splits a 2D grid into block_dim x block_dim blocks
    and flattens each back into a vector.

    Args:
        grid: 2D tensor of shape (block_dim, n_blocks * block_dim)
        block_dim: Side length of each square block

    Returns:
        Flat tensor of shape (n_blocks * block_dim * block_dim,)
    """
    blocks = grid.split(block_dim, dim=-1)
    return torch.cat([b.flatten() for b in blocks])


class SmoothKWTA:
    """
    Mask configurer using lateral smoothing + k-winner-take-all sparsification.

    Gradient optimization followed by convolutional smoothing and kwta.
    Simple single-channel lateral dynamics with uniform averaging kernel.

    Args:
        hidden_size: Dimension of the mask
        square_dim: Side length of square grid
        **kwargs: Optional overrides for class-specific defaults
    """

    # Class-specific defaults
    DEFAULT_K_RATIO = 6
    DEFAULT_RECONFIG_STEPS = 1
    DEFAULT_RECONFIG_LR = 1.0
    DEFAULT_RECONFIG_SAMPLE_SIZE = 2048
    DEFAULT_ENTROPY_COEF = 0.0
    DEFAULT_KERNEL_SIZE = 17
    DEFAULT_LATERAL_STEPS = 2
    DEFAULT_LATERAL_NOISE_SD = 0.0
    DEFAULT_LATERAL_POLYAK = 0.0
    DEFAULT_TORUS = False
    DEFAULT_K = 128

    def __init__(
        self, hidden_size: int, square_dim: int, classification=False, **kwargs
    ):
        self.hidden_size = hidden_size
        self.square_dim = square_dim
        self.classification = classification

        # Use class defaults, allow override via kwargs
        self.k = kwargs.get("k", self.DEFAULT_K)
        self.reconfig_steps = kwargs.get("reconfig_steps", self.DEFAULT_RECONFIG_STEPS)
        self.reconfig_lr = kwargs.get("reconfig_lr", self.DEFAULT_RECONFIG_LR)
        self.reconfig_optimizer = validate_reconfig_optimizer(kwargs.get("reconfig_optimizer", "Adam"))
        self.reconfig_sample_size = kwargs.get(
            "reconfig_sample_size", self.DEFAULT_RECONFIG_SAMPLE_SIZE
        )
        self.entropy_coef = kwargs.get("entropy_coef", self.DEFAULT_ENTROPY_COEF)
        self.lateral_steps = kwargs.get("lateral_steps", self.DEFAULT_LATERAL_STEPS)
        self.lateral_noise_sd = kwargs.get(
            "lateral_noise_sd", self.DEFAULT_LATERAL_NOISE_SD
        )
        self.lateral_polyak = kwargs.get("lateral_polyak", self.DEFAULT_LATERAL_POLYAK)

        kernel_size = kwargs.get("kernel_size", self.DEFAULT_KERNEL_SIZE)
        torus = kwargs.get("torus", self.DEFAULT_TORUS)

        # Lateral smoothing kernel (single channel, uniform averaging)
        self.lateral = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
            padding_mode="circular" if torus else "zeros",
        ).requires_grad_(False)
        self.lateral.weight.data.fill_(1)  # / kernel_size**2)

    def step_lateral(self, m: torch.Tensor) -> torch.Tensor:
        """
        Apply lateral smoothing to mask.

        Splits the mask into square_dim x square_dim blocks and applies
        lateral dynamics to each block independently. For a single block
        this is equivalent to a plain reshape; for multiple blocks (e.g.
        after neurogenesis) each block preserves its own spatial topology.

        Args:
            m: Mask tensor of shape (hidden_size,)

        Returns:
            Smoothed mask tensor of shape (hidden_size,)
        """
        m = mask_to_grid(m, self.square_dim).unsqueeze(0).unsqueeze(0)
        m = self.lateral(m).squeeze(0).squeeze(0)
        if self.lateral_noise_sd > 0:
            m = m + torch.randn_like(m) * self.lateral_noise_sd
        return mask_to_vec(m, self.square_dim)

    def __call__(self, model: nn.Module, data: tuple, m: torch.Tensor) -> torch.Tensor:
        """
        Reconfigure mask for a new task.

        Optimizes mask via gradient descent on task loss, applies lateral
        dynamics for spatial coherence, and sparsifies with k-winner-take-all.

        Args:
            model: Neural network with forward(x, m) signature
            data: Tuple of (x, y) task data tensors
            m: Current mask tensor of shape (hidden_size,)

        Returns:
            Updated binary mask tensor of shape (hidden_size,)
        """
        x, y = data
        n = min(self.reconfig_sample_size, x.shape[0])

        # Make mask a learnable parameter
        # Can determine the prior based on the scale here.
        # m = nn.Parameter(1 * (m.data.clone() - 0.5))

        m = nn.Parameter(torch.randn_like(m) * 0.00001)
        # m = nn.Parameter(torch.ones_like(m) * -2)
        # m = nn.Parameter(torch.zeros_like(m))

        optimizer = make_reconfig_optimizer([m], lr=self.reconfig_lr, name=self.reconfig_optimizer)

        # Gradient-based mask optimization
        z = model.parallel(x[:n]).detach()
        lowest_loss = torch.inf
        best = None
        for i in range(self.reconfig_steps):
            # y_pred = model(x[:n], torch.sigmoid(10 * (m - 0.5)))
            sm = torch.sigmoid(m)
            # mask = sm
            # mask = torch.sigmoid(m / 1e-8).detach() + sm - sm.detach()
            tau = m.detach().std() * torch.exp(
                -torch.ones(1) * i / (self.reconfig_steps / 4)
            )
            noise = torch.randn_like(m) * tau
            mask = kwta(m + noise, k=self.k) + sm - sm.detach()

            # sm = m
            # mask = kwta(sm, self.k) + sm - sm.detach()

            y_pred = model.linear(z * mask)

            if self.classification:
                target = torch.where(y == 1)[1]
                loss = F.cross_entropy(input=y_pred[:n], target=target[:n])
            else:
                loss = F.mse_loss(y_pred.squeeze()[:n], y[:n].squeeze())

            if self.entropy_coef != 0:
                entropy = (
                    torch.distributions.Categorical(logits=y_pred).entropy().mean()
                )
                loss = loss + self.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss < lowest_loss:
                best = mask.clone().detach()
                lowest_loss = loss
            # if (i + 1) % 100 == 0:
            # print(f"Reconfig step {i + 1}: loss={loss.item():.4f}")

        # Apply lateral dynamics (spatial smoothing)
        if self.lateral_polyak < 1.0:
            for _ in range(self.lateral_steps):
                m_lateral = self.step_lateral(best)
                best = (
                    self.lateral_polyak * best + (1 - self.lateral_polyak) * m_lateral
                )
            best = kwta(best, k=self.k)

        # K-winner-take-all sparsification

        return best


class DualKernelMaskConfigurer:
    """
    Mask configurer using dual excitatory/inhibitory convolutional kernels.

    Uses a 2-channel representation [active, inactive] processed through:
    1. Excitatory lateral kernel (local averaging)
    2. Inhibitory kernel (surround suppression)
    3. Softmax normalization across channels

    This approach replaces k-winner-take-all with emergent sparsity from
    lateral inhibition dynamics, producing spatially coherent patterns.

    Args:
        hidden_size: Dimension of the mask (must be square_dim^2)
        square_dim: Side length of square grid
        **kwargs: Optional overrides for class-specific defaults
    """

    # Class-specific defaults
    DEFAULT_RECONFIG_STEPS = 1
    DEFAULT_RECONFIG_LR = 1.0
    DEFAULT_RECONFIG_SAMPLE_SIZE = 2048
    DEFAULT_ENTROPY_COEF = 0.0
    DEFAULT_LATERAL_STEPS = 5
    DEFAULT_LATERAL_NOISE_SD = 0.0
    DEFAULT_SOFTMAX_HARDNESS = 0.5
    DEFAULT_EXC_KERNEL_SIZE = 3
    DEFAULT_INH_KERNEL_SIZE = 9
    DEFAULT_INH_STR_ACTIVE = -0.15
    DEFAULT_INH_STR_INACTIVE = -0.05
    DEFAULT_TORUS = False

    def __init__(
        self, hidden_size: int, square_dim: int, classification=False, **kwargs
    ):
        self.hidden_size = hidden_size
        self.square_dim = square_dim
        self.classification = classification

        # Use class defaults, allow override via kwargs
        self.reconfig_steps = kwargs.get("reconfig_steps", self.DEFAULT_RECONFIG_STEPS)
        self.reconfig_lr = kwargs.get("reconfig_lr", self.DEFAULT_RECONFIG_LR)
        self.reconfig_optimizer = validate_reconfig_optimizer(kwargs.get("reconfig_optimizer", "Adam"))
        self.reconfig_sample_size = kwargs.get(
            "reconfig_sample_size", self.DEFAULT_RECONFIG_SAMPLE_SIZE
        )
        self.entropy_coef = kwargs.get("entropy_coef", self.DEFAULT_ENTROPY_COEF)
        self.lateral_steps = kwargs.get("lateral_steps", self.DEFAULT_LATERAL_STEPS)
        self.lateral_noise_sd = kwargs.get(
            "lateral_noise_sd", self.DEFAULT_LATERAL_NOISE_SD
        )
        self.softmax_hardness = kwargs.get(
            "softmax_hardness", self.DEFAULT_SOFTMAX_HARDNESS
        )
        self.torus = kwargs.get("torus", self.DEFAULT_TORUS)

        exc_kernel_size = kwargs.get("exc_kernel_size", self.DEFAULT_EXC_KERNEL_SIZE)
        inh_kernel_size = kwargs.get("inh_kernel_size", self.DEFAULT_INH_KERNEL_SIZE)
        inh_str_active = kwargs.get("inh_str_active", self.DEFAULT_INH_STR_ACTIVE)
        inh_str_inactive = kwargs.get("inh_str_inactive", self.DEFAULT_INH_STR_INACTIVE)

        # Excitatory lateral kernel (local averaging, 2 groups for on/off channels)
        self.lateral = nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=exc_kernel_size,
            stride=1,
            padding=exc_kernel_size // 2,
            groups=2,
            bias=False,
            padding_mode="circular" if self.torus else "zeros",
        ).requires_grad_(False)
        self.lateral.weight.data.fill_(1.0)

        # Inhibitory kernel (surround suppression)
        self.inh = nn.Conv2d(
            in_channels=2,
            out_channels=2,
            kernel_size=inh_kernel_size,
            stride=1,
            padding=inh_kernel_size // 2,
            groups=2,
            bias=False,
            padding_mode="circular",
        ).requires_grad_(False)
        # Different inhibition strengths for active vs inactive channels
        self.inh.weight.data[0].fill_(inh_str_active)
        self.inh.weight.data[1].fill_(inh_str_inactive)

    def step_lateral(self, m: torch.Tensor) -> torch.Tensor:
        """
        Apply dual-kernel lateral dynamics to mask.

        Splits into square_dim x square_dim blocks and applies dual-kernel
        dynamics to each independently. For each block, creates a 2-channel
        [active, inactive] representation, applies excitatory and inhibitory
        convolutions, and normalizes with softmax.

        Args:
            m: Mask tensor of shape (hidden_size,)

        Returns:
            Updated mask tensor of shape (hidden_size,)
        """
        grid = mask_to_grid(m, self.square_dim)
        b = torch.stack((grid, 1 - grid)).unsqueeze(0)  # (1, 2, H, W)
        b = self.lateral(b) + self.inh(b)
        if self.lateral_noise_sd > 0:
            b = b + torch.randn_like(b) * self.lateral_noise_sd
        b = torch.softmax(b / self.softmax_hardness, dim=1)
        return mask_to_vec(b.squeeze(0)[0], self.square_dim)

    def __call__(self, model: nn.Module, data: tuple, m: torch.Tensor) -> torch.Tensor:
        """
        Reconfigure mask for a new task.

        Optimizes mask via gradient descent, then applies dual-kernel
        lateral dynamics for emergent sparsity patterns.

        Args:
            model: Neural network with forward(x, m) signature
            data: Tuple of (x, y) task data tensors
            m: Current mask tensor of shape (hidden_size,)

        Returns:
            Updated mask tensor of shape (hidden_size,)
        """
        x, y = data
        n = min(self.reconfig_sample_size, x.shape[0])

        # Make mask a learnable parameter
        m = nn.Parameter(m.data.clone())
        optimizer = make_reconfig_optimizer([m], lr=self.reconfig_lr, name=self.reconfig_optimizer)

        # Gradient-based mask optimization
        for i in range(self.reconfig_steps):
            y_pred = model(x[:n], torch.sigmoid(m))

            if self.classification:
                target = torch.where(y == 1)[1]
                loss = F.cross_entropy(y_pred.squeeze()[:n], target[:n])
            else:
                loss = F.mse_loss(y_pred.squeeze()[:n], y[:n].squeeze())
            if self.entropy_coef != 0:
                entropy = (
                    torch.distributions.Categorical(logits=y_pred).entropy().mean()
                )
                loss = loss + self.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f"Reconfig step {i + 1}: loss={loss.item():.4f}")

        # Apply lateral dynamics (train) or kwta (test)
        for _ in range(self.lateral_steps):
            m = self.step_lateral(m)
        # else:
        #     # Fallback to kwta for test mode
        #     k = self.hidden_size // 6  # Approximate sparsity
        #     m = kwta(m.data, k=k)

        return m.detach()


# Mask subclasses


class SmoothKWTAFast(SmoothKWTA):
    DEFAULT_K_RATIO = 12
    DEFAULT_KERNEL_SIZE = 17
    DEFAULT_LATERAL_STEPS = 2
    DEFAULT_TORUS = False

    def __init__(self, hidden_size: int, square_dim: int, **kwargs):
        super().__init__(hidden_size, square_dim, **kwargs)


class SmoothKWTAFastSmall(SmoothKWTA):
    DEFAULT_K_RATIO = 12
    DEFAULT_KERNEL_SIZE = 3
    DEFAULT_LATERAL_STEPS = 2
    DEFAULT_TORUS = False

    def __init__(self, hidden_size: int, square_dim: int, **kwargs):
        super().__init__(hidden_size, square_dim, **kwargs)


class SmoothKWTASlow(SmoothKWTA):
    DEFAULT_K_RATIO = 6
    DEFAULT_KERNEL_SIZE = 3
    DEFAULT_LATERAL_STEPS = 15

    def __init__(self, hidden_size: int, square_dim: int, **kwargs):
        super().__init__(hidden_size, square_dim, **kwargs)


class PolyakKWTAFast(SmoothKWTA):
    DEFAULT_K_RATIO = 12
    DEFAULT_KERNEL_SIZE = 7
    DEFAULT_LATERAL_STEPS = 2
    DEFAULT_LATERAL_POLYAK = 0.95
    DEFAULT_TORUS = False

    def __init__(self, hidden_size: int, square_dim: int, **kwargs):
        super().__init__(hidden_size, square_dim, **kwargs)


class PolyakKWTASlow(SmoothKWTA):
    DEFAULT_K_RATIO = 6
    DEFAULT_KERNEL_SIZE = 7
    DEFAULT_LATERAL_STEPS = 1
    DEFAULT_LATERAL_POLYAK = 0.95
    DEFAULT_TORUS = False

    def __init__(self, hidden_size: int, square_dim: int, **kwargs):
        super().__init__(hidden_size, square_dim, **kwargs)


class KWTA(SmoothKWTA):
    DEFAULT_K_RATIO = 12
    DEFAULT_KERNEL_SIZE = 3
    DEFAULT_LATERAL_STEPS = 0

    def __init__(self, hidden_size: int, square_dim: int, **kwargs):
        super().__init__(hidden_size, square_dim, **kwargs)


class GridCell(DualKernelMaskConfigurer):
    """
    Dual-kernel configurer tuned for grid-cell-like patterns.

    Produces hexagonal/grid-like activation patterns through strong
    local inhibition with moderate kernel size.

    Preset: kernel=9, active=-0.15, inactive=-0.05
    """

    # Override defaults for grid-cell pattern
    DEFAULT_INH_KERNEL_SIZE = 9
    DEFAULT_INH_STR_ACTIVE = -0.15
    DEFAULT_INH_STR_INACTIVE = -0.05

    def __init__(self, hidden_size: int, square_dim: int, **kwargs):
        super().__init__(hidden_size, square_dim, **kwargs)


class MexicanHatCA(DualKernelMaskConfigurer):
    """
    Dual-kernel configurer tuned for mexican-hat cellular automata patterns.

    Produces blob-like task networks through weak but wide-range inhibition.
    Good for task-specific network separation.

    Preset: kernel=27, active=-0.01, inactive=-0.001
    """

    # Override defaults for mexican-hat pattern
    DEFAULT_INH_KERNEL_SIZE = 27
    DEFAULT_INH_STR_ACTIVE = -0.001
    DEFAULT_INH_STR_INACTIVE = -0.0001
    DEFAULT_TORUS = False

    def __init__(self, hidden_size: int, square_dim: int, **kwargs):
        super().__init__(hidden_size, square_dim, **kwargs)


class Disabled:
    """
    Null mask configurer for ablation studies.

    Returns a fixed mask with the first n units active,
    ignoring task data entirely.

    Args:
        hidden_size: Dimension of the mask
        square_dim: Side length of square grid (for interface compatibility)
        **kwargs: Optional overrides (active_ratio)
    """

    DEFAULT_ACTIVE_RATIO = 0.5
    DEFAULT_K = 8

    def __init__(self, hidden_size: int, square_dim: int, **kwargs):
        self.hidden_size = hidden_size
        self.square_dim = square_dim
        self.active_ratio = kwargs.get("active_ratio", self.DEFAULT_ACTIVE_RATIO)
        self.k = kwargs.get("k", self.DEFAULT_K)

    def __call__(self, model: nn.Module, data: tuple, m: torch.Tensor) -> torch.Tensor:
        """
        Return fixed mask ignoring task data.

        Args:
            model: Unused
            data: Unused
            m: Template for mask shape

        Returns:
            Fixed binary mask tensor
        """
        _ = model, data  # Unused
        m = torch.ones_like(m)
        # n_active = int(self.hidden_size * self.active_ratio)
        # n_active = self.k
        # m[:n_active] = 1
        return m


class Random:
    """
    Null mask configurer for ablation studies.

    Returns a fixed mask with the first n units active,
    ignoring task data entirely.

    Args:
        hidden_size: Dimension of the mask
        square_dim: Side length of square grid (for interface compatibility)
        **kwargs: Optional overrides (active_ratio)
    """

    DEFAULT_ACTIVE_RATIO = 0.5
    DEFAULT_K = 8

    def __init__(self, hidden_size: int, square_dim: int, **kwargs):
        self.hidden_size = hidden_size
        self.square_dim = square_dim
        self.active_ratio = kwargs.get("active_ratio", self.DEFAULT_ACTIVE_RATIO)
        self.k = kwargs.get("k", self.DEFAULT_K)

    def __call__(self, model: nn.Module, data: tuple, m: torch.Tensor) -> torch.Tensor:
        """
        Return fixed mask ignoring task data.

        Args:
            model: Unused
            data: Unused
            m: Template for mask shape

        Returns:
            Fixed binary mask tensor
        """
        _ = model, data  # Unused
        m = kwta(torch.rand_like(m), k=self.k)
        # n_active = int(self.hidden_size * self.active_ratio)
        # n_active = self.k
        # m[:n_active] = 1
        return m


class KWTA_Search:
    """
    Mask configurer using lateral smoothing + k-winner-take-all sparsification.

    Gradient optimization followed by convolutional smoothing and kwta.
    Simple single-channel lateral dynamics with uniform averaging kernel.

    Args:
        hidden_size: Dimension of the mask
        square_dim: Side length of square grid
        **kwargs: Optional overrides for class-specific defaults
    """

    # Class-specific defaults
    DEFAULT_K_RATIO = 6
    DEFAULT_RECONFIG_STEPS = 1
    DEFAULT_RECONFIG_LR = 1.0
    DEFAULT_RECONFIG_SAMPLE_SIZE = 2048
    DEFAULT_ENTROPY_COEF = 0.0
    DEFAULT_KERNEL_SIZE = 17
    DEFAULT_LATERAL_STEPS = 2
    DEFAULT_LATERAL_NOISE_SD = 0.0
    DEFAULT_LATERAL_POLYAK = 0.0
    DEFAULT_TORUS = False
    DEFAULT_K = 128

    def __init__(
        self, hidden_size: int, square_dim: int, classification=False, **kwargs
    ):
        self.hidden_size = hidden_size
        self.square_dim = square_dim
        self.classification = classification

        # Use class defaults, allow override via kwargs
        self.k = kwargs.get("k", self.DEFAULT_K)
        self.reconfig_steps = kwargs.get("reconfig_steps", self.DEFAULT_RECONFIG_STEPS)
        self.reconfig_lr = kwargs.get("reconfig_lr", self.DEFAULT_RECONFIG_LR)
        self.reconfig_optimizer = validate_reconfig_optimizer(kwargs.get("reconfig_optimizer", "Adam"))
        self.reconfig_sample_size = kwargs.get(
            "reconfig_sample_size", self.DEFAULT_RECONFIG_SAMPLE_SIZE
        )

    def __call__(self, model: nn.Module, data: tuple, m: torch.Tensor) -> torch.Tensor:
        """
        Reconfigure mask for a new task.

        Optimizes mask via gradient descent on task loss, applies lateral
        dynamics for spatial coherence, and sparsifies with k-winner-take-all.

        Args:
            model: Neural network with forward(x, m) signature
            data: Tuple of (x, y) task data tensors
            m: Current mask tensor of shape (hidden_size,)

        Returns:
            Updated binary mask tensor of shape (hidden_size,)
        """
        x, y = data
        n = min(self.reconfig_sample_size, x.shape[0])
        m = nn.Parameter(torch.zeros_like(m))
        optimizer = make_reconfig_optimizer([m], lr=self.reconfig_lr, name=self.reconfig_optimizer)

        # Gradient-based mask optimization
        lowest_loss = torch.inf
        best = None
        z = model.parallel(x[:n]).detach()
        for i in range(self.reconfig_steps):
            tau = m.std() * torch.exp(-torch.ones(1) * i / (self.reconfig_steps / 2))
            noise = torch.randn_like(m) * tau
            mask = kwta(m + noise, k=self.k) + m - m.detach()
            y_pred = model.linear(z * mask)
            if self.classification:
                target = torch.where(y == 1)[1]
                loss = F.cross_entropy(input=y_pred[:n], target=target[:n])
            else:
                loss = F.mse_loss(y_pred.squeeze()[:n], y[:n].squeeze())

            if loss.item() < lowest_loss:
                best = mask
                lowest_loss = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return best
