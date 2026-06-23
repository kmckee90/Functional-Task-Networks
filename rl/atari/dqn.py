"""
dqn.py - Masked DQN model for Atari continual learning.

CNN feature extractor (shared across all games) feeds into masked FC layers
that use the same spatial masking approach as MaskedMLPSpatial from the parent
project.  The mask gates hidden activations in the FC layers, enabling
task-specific subnetwork allocation per game.

The output head has max_actions outputs.  Per-game valid action masking is
applied externally (invalid actions get -inf before argmax).
"""

import math

import torch
import torch.nn as nn


def init_module(module, scale=1.0):
    for m in module.modules():
        if hasattr(m, "weight") and m.weight is not None:
            fan_in = m.weight.size(1) if m.weight.dim() >= 2 else m.weight.numel()
            std = scale / math.sqrt(fan_in)
            nn.init.normal_(m.weight, mean=0.0, std=std)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.zeros_(m.bias)


class MaskedDQN(nn.Module):
    """
    DQN with masked hidden FC layers for continual learning.

    Architecture:
        Conv feature extractor  (shared, no masking)
        → Flatten
        → FC hidden layers      (masked via cortical grid + KWTA)
        → Q-value output head   (shared, max_actions outputs)

    The mask 'm' is a binary vector of shape (hidden_size,) that gates
    activations in every hidden FC layer, identical to MaskedMLPSpatial.

    Args:
        max_actions: Maximum action space across all games
        hidden_size: Dimension of masked hidden layers (square_dim²)
        frame_stack: Number of stacked frames (input channels)
        img_size: Height/width of preprocessed frames (84)
        **kwargs: Lateral dynamics params (b, steps, kernel_size, layers, dropout)
    """

    DEFAULT_B = 0.0
    DEFAULT_SQUARE_DIM = 32
    DEFAULT_KERNEL_SIZE = 3
    DEFAULT_STEPS = 0
    DEFAULT_LAYERS = 2
    DEFAULT_DROPOUT = 0.0

    def __init__(
        self,
        max_actions: int,
        hidden_size: int,
        frame_stack: int = 4,
        img_size: int = 84,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_actions = max_actions
        n_fc_layers = kwargs.get("layers", self.DEFAULT_LAYERS)
        self.drop_p = kwargs.get("dropout", self.DEFAULT_DROPOUT)

        # ── CNN feature extractor (Nature DQN architecture) ────────────
        in_channels = frame_stack  # grayscale stacked frames
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            conv_out = self.conv(dummy)
            self.conv_flat_size = conv_out.view(1, -1).shape[1]

        # ── Masked FC layers ───────────────────────────────────────────
        self.fc = nn.ModuleList()
        for i in range(n_fc_layers):
            in_dim = self.conv_flat_size if i == 0 else hidden_size
            self.fc.append(nn.Linear(in_dim, hidden_size))

        # Linear skip connection (conv features → hidden, masked)
        self.lin = nn.Linear(self.conv_flat_size, hidden_size)

        # ── Output head ────────────────────────────────────────────────
        self.out = nn.Linear(hidden_size, max_actions)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.drop_p)

        # Init FC layers with small scale (matching parent project)
        # init_module(self.lin, scale=0.1)
        # init_module(self.fc, scale=0.1)
        # init_module(self.out, scale=0.1)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Observations, shape (batch, frame_stack, 84, 84), uint8 or float
            m: Mask, shape (hidden_size,)

        Returns:
            Q-values, shape (batch, max_actions)
        """
        # Normalize pixel values to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        # CNN feature extraction (unmasked)
        features = self.conv(x).view(x.size(0), -1)  # (batch, conv_flat_size)

        # Linear skip path (masked)
        dm = (
            torch.bernoulli(torch.ones_like(m) * (1 - self.drop_p)) * m
            if self.training
            else m
        )
        h_lin = self.lin(features)

        # Masked FC path
        h = features
        for fc in self.fc:
            h = fc(h) * dm
            h = self.act(h) * dm

        # Combine and produce Q-values
        q = self.out(h + h_lin)
        return q


class MaskedDQNMinAtar(nn.Module):
    """
    Masked DQN for MinAtar (10x10 observations with variable channels).

    Uses a lightweight CNN appropriate for MinAtar's small spatial resolution:
        Conv2d(in_channels, 16, 3, stride=1, padding=1) → ReLU   (preserves 10x10)
        → Flatten (16 * 10 * 10 = 1600)
        → Masked FC layers (same masking as MaskedDQN)
        → Q-value output head

    Args:
        max_actions: Maximum action space across all games
        hidden_size: Dimension of masked hidden layers (square_dim²)
        in_channels: Number of input channels (max across all MinAtar games)
        **kwargs: layers, dropout (same as MaskedDQN)
    """

    DEFAULT_LAYERS = 2
    DEFAULT_DROPOUT = 0.0

    def __init__(
        self,
        max_actions: int,
        hidden_size: int,
        in_channels: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_actions = max_actions
        n_fc_layers = kwargs.get("layers", self.DEFAULT_LAYERS)
        self.drop_p = kwargs.get("dropout", self.DEFAULT_DROPOUT)

        # ── CNN feature extractor (small, for 10x10 inputs) ──────────
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 10, 10)
            conv_out = self.conv(dummy)
            self.conv_flat_size = conv_out.view(1, -1).shape[1]  # 16*10*10 = 1600

        # ── Masked FC layers ─────────────────────────────────────────
        self.fc = nn.ModuleList()
        for i in range(n_fc_layers):
            in_dim = self.conv_flat_size if i == 0 else hidden_size
            self.fc.append(nn.Linear(in_dim, hidden_size))

        # Linear skip connection (conv features → hidden, masked)
        self.lin = nn.Linear(self.conv_flat_size, hidden_size)

        # ── Output head ──────────────────────────────────────────────
        self.out = nn.Linear(hidden_size, max_actions)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(self.drop_p)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Observations, shape (batch, in_channels, 10, 10), uint8 or float
            m: Mask, shape (hidden_size,)

        Returns:
            Q-values, shape (batch, max_actions)
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        features = self.conv(x).reshape(x.size(0), -1)

        dm = (
            torch.bernoulli(torch.ones_like(m) * (1 - self.drop_p)) * m
            if self.training
            else m
        )
        h_lin = self.lin(features)

        h = features
        for fc in self.fc:
            h = fc(h) * dm
            h = self.act(h) * dm

        q = self.out(h + h_lin)
        return q


class Conv2dDQNMinAtar(nn.Module):
    """
    Masked DQN for MinAtar (10x10 observations with variable channels).

    Uses a lightweight CNN appropriate for MinAtar's small spatial resolution:
        Conv2d(in_channels, 16, 3, stride=1, padding=1) → ReLU   (preserves 10x10)
        → Flatten (16 * 10 * 10 = 1600)
        → Masked FC layers (same masking as MaskedDQN)
        → Q-value output head

    Args:
        max_actions: Maximum action space across all games
        hidden_size: Dimension of masked hidden layers (square_dim²)
        in_channels: Number of input channels (max across all MinAtar games)
        **kwargs: layers, dropout (same as MaskedDQN)
    """

    DEFAULT_LAYERS = 2
    DEFAULT_DROPOUT = 0.0

    def __init__(
        self,
        max_actions: int,
        hidden_size: int,
        in_channels: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_actions = max_actions
        self.drop_p = kwargs.get("dropout", self.DEFAULT_DROPOUT)
        n_fc_layers = kwargs.get("layers", self.DEFAULT_LAYERS)

        # ── CNN feature extractor (small, for 10x10 inputs) ──────────
        hs = self.hidden_size
        self.act = nn.ReLU()
        self.fc = nn.ModuleList()
        for i in range(n_fc_layers):
            in_dim = in_channels if i == 0 else hs
            self.fc.append(nn.Conv2d(in_dim, hs, 3, 1, 1))

        # Compute conv output size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 10, 10)
            conv_out = self.fc[0](dummy)
            self.conv_flat_size = conv_out.view(1, -1).shape[1]  # 16*10*10 = 1600

        # ── Output head ──────────────────────────────────────────────
        self.out = nn.Linear(self.conv_flat_size, max_actions)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Observations, shape (batch, in_channels, 10, 10), uint8 or float
            m: Mask, shape (hidden_size,)

        Returns:
            Q-values, shape (batch, max_actions)
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        dm = (
            torch.bernoulli(torch.ones_like(m) * (1 - self.drop_p)) * m
            if self.training
            else m
        ).reshape(1, -1, 1, 1)
        h = self.fc[0](x) * dm
        # h = self.act(h0.clone())
        # h0 = h0.reshape(x.size(0), -1)
        for fc in self.fc[1:]:
            h = fc(h) * dm
            h = self.act(h) * dm

        q = self.out(h.reshape(x.size(0), -1))
        return q
