"""
models.py - Neural network models for masked network experiments.

Contains model architectures where forward passes accept an additional
mask argument 'm' that gates hidden activations. This enables dynamic
network reconfiguration for multi-task learning.
"""

import math

import torch
import torch.nn as nn


def reinit(module, scale=1.0):
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d, nn.Conv3d)):
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))  # PyTorch default-like
            m.weight.data.mul_(scale)
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound * scale, bound * scale)

    module.apply(_init)


def init_module(module, scale=1.0):
    for m in module.modules():
        if hasattr(m, "weight") and m.weight is not None:
            fan_in = m.weight.size(1) if m.weight.dim() >= 2 else m.weight.numel()
            std = scale / math.sqrt(fan_in)
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.zeros_(m.bias)
            # m.bias.data = std * m.bias / m.bias.std()


class MLP(nn.Module):
    """
    Multi-layer perceptron with mask-gated hidden activations.

    Each hidden layer's output is element-wise multiplied by mask 'm',
    enabling selective activation of hidden units per task.

    Args:
        input_size: Dimension of input features
        output_size: Dimension of output
        hidden_size: Dimension of hidden layers
        n_layers: Number of hidden layers
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.dec = nn.Linear

    def forward(self, x: torch.Tensor, m: torch.Tensor):
        return self.fc(x)


class MaskedMLP(nn.Module):
    """
    Multi-layer perceptron with mask-gated hidden activations.

    Each hidden layer's output is element-wise multiplied by mask 'm',
    enabling selective activation of hidden units per task.

    Args:
        input_size: Dimension of input features
        output_size: Dimension of output
        hidden_size: Dimension of hidden layers
        n_layers: Number of hidden layers
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        n_layers: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = kwargs["layers"]

        # Build hidden layers
        self.fc = nn.ModuleList(
            [
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(self.layers)
            ]
        )
        self.skip = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()
        self.register_buffer("lr", torch.ones(self.hidden_size))
        self.mask_pars = nn.Linear(8, hidden_size)

        # reinit(self.fc, scale=1.0)
        # reinit(self.out, scale=1.0)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mask gating.

        Args:
            x: Input tensor of shape (..., input_size)
            m: Mask tensor of shape (hidden_size,) applied to hidden activations

        Returns:
            Output tensor of shape (..., output_size)
        """
        # Linear skip path
        h_skip = self.skip(x) * m

        # Deep nonlinear path (residual from first layer)
        h0 = self.act(self.fc[0](x)) * m
        h = h0
        for fc in self.fc[1:]:
            h = self.act(fc(h)) * m + h0
        return self.out(h + h_skip)


class MaskedMLPSpatial(nn.Module):
    """
    Multi-layer perceptron with mask-gated hidden activations.

    Each hidden layer's output is element-wise multiplied by mask 'm',
    enabling selective activation of hidden units per task.

    Args:
        input_size: Dimension of input features
        output_size: Dimension of output
        hidden_size: Dimension of hidden layers
        n_layers: Number of hidden layers
    """

    DEFAULT_B = 0.0
    DEFAULT_SQUARE_DIM = 32
    DEFAULT_KERNEL_SIZE = 3
    DEFAULT_STEPS = 1
    DEFAULT_LAYERS = 3
    DEFAULT_DROPOUT = 0.2

    TORUS = False

    def __init__(self, input_size: int, output_size: int, hidden_size: int, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = kwargs.get("layers", self.DEFAULT_LAYERS)
        self.drop_p = kwargs.get("dropout", self.DEFAULT_DROPOUT)

        # Build hidden layers
        self.fc = nn.ModuleList(
            [
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(self.layers)
            ]
        )
        self.lin = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.drop_p)

        # The scale
        # reinit(self.lin, scale=0.1)
        # reinit(self.fc, scale=0.1)
        # reinit(self.out, scale=0.1)
        init_module(self.lin, scale=0.1)
        init_module(self.fc, scale=0.1)
        init_module(self.out, scale=0.1)

        # for fc in self.fc:
        #     fc.weight.data *= 1
        #     fc.bias.data *= 1

        # Lateral smoothing kernel (single channel, uniform averaging)
        self.square_dim = kwargs.get("sqsuare_dim", self.DEFAULT_SQUARE_DIM)
        kernel_size = kwargs.get("kernel_size", self.DEFAULT_KERNEL_SIZE)
        self.b = kwargs.get("b", self.DEFAULT_B)
        self.steps = kwargs.get("steps", self.DEFAULT_STEPS)

        torus = kwargs.get("torus", self.TORUS)
        self.lateral = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
            padding_mode="circular" if torus else "zeros",
        ).requires_grad_(False)
        self.lateral.weight.data.fill_(1 / kernel_size**2)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mask gating.

        Args:
            x: Input tensor of shape (..., input_size)
            m: Mask tensor of shape (hidden_size,) applied to hidden activations

        Returns:
            Output tensor of shape (..., output_size)
        """
        # Compute linear layer with lateral and masking:

        h_lin = self.lin(x)
        dm = torch.bernoulli(torch.ones_like(h_lin) * (1 - self.drop_p)) * m
        h_lin = self.step_lateral(h_lin * dm) * dm
        # Compute masked and lateral MLP
        h = x
        for fc in self.fc:
            h = fc(h)
            h = self.step_lateral(h * dm) * dm
            h = self.act(h) * dm
        # Decode combined result
        h_out = h + h_lin
        m2 = (h_out > 0).float()
        # h_out = h_out * m2
        y = self.out(h_out)
        return y

    def step_lateral(self, h):
        if self.steps == 0 or self.b == 1.0:
            return h
        hshape = h.shape
        h = h.reshape(-1, 1, self.square_dim, self.square_dim)
        for _ in range(self.steps):
            h = self.b * h + (1 - self.b) * self.lateral(h)

        return h.reshape(hshape)


class Spatial0(nn.Module):
    """
    Multi-layer perceptron with mask-gated hidden activations.

    Each hidden layer's output is element-wise multiplied by mask 'm',
    enabling selective activation of hidden units per task.

    Args:
        input_size: Dimension of input features
        output_size: Dimension of output
        hidden_size: Dimension of hidden layers
        n_layers: Number of hidden layers
    """

    DEFAULT_B = 0.0
    DEFAULT_SQUARE_DIM = 32
    DEFAULT_KERNEL_SIZE = 3
    DEFAULT_STEPS = 1
    DEFAULT_LAYERS = 3
    DEFAULT_DROPOUT = 0.2

    TORUS = False

    def __init__(self, input_size: int, output_size: int, hidden_size: int, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = kwargs.get("layers", self.DEFAULT_LAYERS)
        self.drop_p = kwargs.get("dropout", self.DEFAULT_DROPOUT)

        # Build hidden layers
        self.fc = nn.ModuleList(
            [
                nn.Linear(input_size if i == 0 else hidden_size, hidden_size)
                for i in range(self.layers)
            ]
        )
        self.lin = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.drop_p)

        # The scale
        # reinit(self.lin, scale=0.1)
        # reinit(self.fc, scale=0.1)
        # reinit(self.out, scale=0.1)
        init_module(self.lin, scale=0.1)
        init_module(self.fc, scale=0.1)
        init_module(self.out, scale=0.1)

        # for fc in self.fc:
        #     fc.weight.data *= 1
        #     fc.bias.data *= 1

        # Lateral smoothing kernel (single channel, uniform averaging)
        self.square_dim = kwargs.get("sqsuare_dim", self.DEFAULT_SQUARE_DIM)
        kernel_size = kwargs.get("kernel_size", self.DEFAULT_KERNEL_SIZE)
        self.b = kwargs.get("b", self.DEFAULT_B)
        self.steps = kwargs.get("steps", self.DEFAULT_STEPS)

        torus = kwargs.get("torus", self.TORUS)
        self.lateral = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
            padding_mode="circular" if torus else "zeros",
        ).requires_grad_(False)
        self.lateral.weight.data.fill_(1 / kernel_size**2)

        # self.register_buffer(
        #     "m", nn.Parameter(torch.randn(self.square_dim * self.square_dim))
        # )

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with mask gating.

        Args:
            x: Input tensor of shape (..., input_size)
            m: Mask tensor of shape (hidden_size,) applied to hidden activations

        Returns:
            Output tensor of shape (..., output_size)
        """
        # Compute linear layer with lateral and masking:

        h_lin = self.lin(x)
        dm = torch.bernoulli(torch.ones_like(h_lin) * (1 - self.drop_p)) * m
        h_lin = self.step_lateral(h_lin * dm) * dm
        h_lin = self.act(h_lin)
        # Compute masked and lateral MLP
        h = x
        for fc in self.fc:
            h = fc(h)
            h = self.step_lateral(h * dm) * dm
            h = self.act(h) * dm
        # Decode combined result
        h_out = h + h_lin

        # Mask and suppress?
        sm = torch.sigmoid(2 * h_out - 2)
        mask = torch.bernoulli(sm) + sm - sm.detach()
        h_out = h_out * mask
        # h_out[h_out < 0] = 0

        y = self.out(h_out)
        return y

    def step_lateral(self, h):
        if self.steps == 0 or self.b == 1.0:
            return h
        hshape = h.shape
        h = h.reshape(-1, 1, self.square_dim, self.square_dim)
        for _ in range(self.steps):
            h = self.b * h + (1 - self.b) * self.lateral(h)

        return h.reshape(hshape)


class ParallelMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, **kwargs):
        super().__init__()
        self.n_layers = kwargs["layers"]
        self.inner_size = kwargs["inner_size"]
        sizes = [input_size] + [self.inner_size] * self.n_layers + [output_size]
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(hidden_size, sizes[i], sizes[i + 1]))
                for i in range(len(sizes) - 1)
            ]
        )
        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.empty(hidden_size, sizes[i + 1]))
                for i in range(len(sizes) - 1)
            ]
        )
        self._init_weights(sizes)

    def _init_weights(self, sizes):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            fan_in = sizes[i]
            fan_out = sizes[i + 1]
            s = 6.0
            s_b = 0.1
            if i < self.n_layers:
                # Kaiming uniform for ReLU hidden layers
                bound = math.sqrt(s / fan_in)
            else:
                # Xavier uniform for the linear output layer
                bound = math.sqrt(s / (fan_in + fan_out))
            nn.init.uniform_(w, -bound, bound)
            # Bias: uniform in [-1/sqrt(fan_in), 1/sqrt(fan_in)] (PyTorch convention)
            nn.init.uniform_(b, -s_b / math.sqrt(fan_in), s_b / math.sqrt(fan_in))

    def forward(self, x):
        h = x.squeeze().unsqueeze(-2)  # (B, 1, D) — broadcasts over K
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.einsum("bki,kij->bkj", h, W) + b  # (B, K, out_i)
            if i < self.n_layers:  # relu on hidden layers only, not the output
                h = torch.relu(h)
        out = h.squeeze(-1)
        return out  # (B, K)


class NeuroModel(nn.Module):
    DEFAULT_B = 0.0
    DEFAULT_SQUARE_DIM = 32
    DEFAULT_KERNEL_SIZE = 3
    DEFAULT_STEPS = 1
    DEFAULT_DROPOUT = 0.2
    TORUS = True

    def __init__(self, input_size, output_size, hidden_size, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.parallel = ParallelMLP(input_size, 1, hidden_size, **kwargs)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.dropout = nn.Dropout(p=kwargs.get("dropout_p", self.DEFAULT_DROPOUT))

        self.square_dim = kwargs.get("sqsuare_dim", self.DEFAULT_SQUARE_DIM)
        kernel_size = kwargs.get("kernel_size", self.DEFAULT_KERNEL_SIZE)
        self.b = kwargs.get("b", self.DEFAULT_B)
        self.steps = kwargs.get("steps", self.DEFAULT_STEPS)

        torus = kwargs.get("torus", self.TORUS)
        self.lateral = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
            padding_mode="circular" if torus else "zeros",
        ).requires_grad_(False)
        self.lateral.weight.data.fill_(1 / kernel_size**2)

    def forward(self, x, m):
        z = self.parallel(x)  # (B, K)
        z = self.dropout(z)
        z = self.step_lateral(z)
        if hasattr(self, "linear"):
            return self.linear(z * m.reshape(1, -1))
        else:
            return (z * m.reshape(1, -1, 1)).mean(-2)

    def step_lateral(self, h):
        if self.steps == 0 or self.b == 1.0:
            return h
        hshape = h.shape
        h = h.reshape(-1, 1, self.square_dim, self.square_dim)
        for _ in range(self.steps):
            h = self.b * h + (1 - self.b) * self.lateral(h)

        return h.reshape(hshape)
