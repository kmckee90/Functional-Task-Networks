"""
ftn.py - Self-contained Functional Task Network (FTN).

An FTN is a masked network that owns its task mask and the loop that
reconfigures it. Everything mask-related lives on this class:

  - backbone: parallel per-unit MLPs (ParallelMLP) + linear readout,
  - mask state: `self.m`, a persistent (hidden_size,) binary buffer,
  - reconfiguration: annealed straight-through-estimator k-winner-take-all
    (the ftn/masks.KWTA algorithm), with optional lateral smoothing of the
    mask (the ftn/masks.SmoothKWTA behavior). mask_lateral_steps=0 recovers
    plain KWTA.

Usage:
    pred = ftn(x, y)          # reconfigure mask on this batch, then predict
    pred = ftn(x)             # predict with the current mask (holdout eval)
    pred = ftn(x, m=m_stored) # predict with an explicit mask (e.g. replay)
    ftn.reconfigure(x, y)     # reconfigure only; result lands in ftn.m

The optimizer for the weights stays outside the class. reconfigure() detaches
all weights, so it never leaves gradients on FTN parameters and never needs a
zero_grad() from the caller.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Mask utilities (module-level so visualization code can reuse them)
# =============================================================================


def kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    """K-winner-take-all: binary mask with 1s at the top-k values."""
    idx = torch.topk(x, k).indices
    out = torch.zeros_like(x)
    out[idx] = 1
    return out


def normalize_mask(m: torch.Tensor) -> torch.Tensor:
    """Normalize mask to [0, 1] range."""
    m = m - m.min()
    m = m / m.max()
    return m


def mask_to_grid(m: torch.Tensor, block_dim: int) -> torch.Tensor:
    """
    Reshape a flat mask into 2D by arranging block_dim x block_dim blocks
    side by side. For a single block this is just reshape(block_dim, block_dim).
    For multiple blocks (e.g. after neurogenesis) it produces
    (block_dim, n_blocks * block_dim).
    """
    block_size = block_dim * block_dim
    blocks = m.split(block_size)
    return torch.cat([b.reshape(block_dim, block_dim) for b in blocks], dim=-1)


def mask_to_vec(grid: torch.Tensor, block_dim: int) -> torch.Tensor:
    """Inverse of mask_to_grid."""
    blocks = grid.split(block_dim, dim=-1)
    return torch.cat([b.flatten() for b in blocks])


def _uniform_conv(kernel_size: int, fill: float, torus: bool) -> nn.Conv2d:
    """Single-channel, frozen, uniform 2D convolution used for lateral dynamics."""
    conv = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        bias=False,
        padding_mode="circular" if torus else "zeros",
    ).requires_grad_(False)
    conv.weight.data.fill_(fill)
    return conv


# =============================================================================
# Backbone
# =============================================================================


class ParallelMLP(nn.Module):
    """hidden_size independent small MLPs, each mapping input -> a scalar unit."""

    def __init__(self, input_size, output_size, hidden_size, layers, inner_size):
        super().__init__()
        self.n_layers = layers
        sizes = [input_size] + [inner_size] * layers + [output_size]
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
            nn.init.uniform_(b, -s_b / math.sqrt(fan_in), s_b / math.sqrt(fan_in))

    def forward(self, x):
        h = x.unsqueeze(-2)  # (B, 1, D) — broadcasts over K
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.einsum("bki,kij->bkj", h, W) + b  # (B, K, out_i)
            if i < self.n_layers:  # relu on hidden layers only, not the output
                h = torch.relu(h)
        return h.squeeze(-1)  # (B, K)


# =============================================================================
# FTN
# =============================================================================


class FTN(nn.Module):
    """
    Functional Task Network: masked backbone + task mask + reconfiguration.

    forward(x) predicts with the current mask; forward(x, y) first
    reconfigures the mask on that batch (per-batch reconfiguration),
    forward(x, m=...) uses an explicit mask.

    Args:
        input_size:          Input feature dimension.
        output_size:         Output dimension (n classes or 1).
        hidden_size:         Number of maskable units (K of ParallelMLP).
        square_dim:          Side length of the square grid each mask block is
                             reshaped into for 2D lateral dynamics. hidden_size
                             must be a multiple of square_dim**2.
        classification:      Cross-entropy (one-hot targets) vs MSE, used for
                             both reconfiguration and self.loss().
        layers, inner_size:  ParallelMLP depth / inner width.
        dropout_p:           Dropout on backbone features (forward only).
        lateral_b, lateral_steps, lateral_kernel_size:
                             Optional lateral smoothing of backbone features;
                             lateral_steps=0 disables (default).
        k:                   kWTA budget - number of active mask units.
        reconfig_steps, reconfig_lr, reconfig_sample_size:
                             Mask-optimization budget per reconfiguration.
        mask_kernel_size, mask_lateral_steps, mask_lateral_polyak,
        mask_lateral_noise_sd:
                             SmoothKWTA lateral dynamics on the mask after the
                             gradient loop; mask_lateral_steps=0 = plain KWTA.
        torus:               Circular (wrap-around) padding for both lateral
                             convolutions.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        square_dim: int,
        classification: bool = False,
        # Backbone
        layers: int = 3,
        inner_size: int = 8,
        dropout_p: float = 0.2,
        lateral_b: float = 0.0,
        lateral_steps: int = 0,
        lateral_kernel_size: int = 3,
        # Mask / reconfiguration
        k: int = 128,
        reconfig_steps: int = 30,
        reconfig_lr: float = 0.3,
        reconfig_sample_size: int = 2048,
        mask_kernel_size: int = 3,
        mask_lateral_steps: int = 0,
        mask_lateral_polyak: float = 0.0,
        mask_lateral_noise_sd: float = 0.0,
        torus: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.square_dim = square_dim
        self.classification = classification

        self.k = k
        self.reconfig_steps = reconfig_steps
        self.reconfig_lr = reconfig_lr
        self.reconfig_sample_size = reconfig_sample_size
        self.mask_lateral_steps = mask_lateral_steps
        self.mask_lateral_polyak = mask_lateral_polyak
        self.mask_lateral_noise_sd = mask_lateral_noise_sd

        self.lateral_b = lateral_b
        self.lateral_steps = lateral_steps

        # Backbone
        self.parallel = ParallelMLP(input_size, 1, hidden_size, layers, inner_size)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)
        # Feature smoothing (averaging kernel); no-op when lateral_steps == 0
        self.lateral = _uniform_conv(
            lateral_kernel_size, 1 / lateral_kernel_size**2, torus
        )
        # Mask smoothing (summing kernel) for SmoothKWTA lateral dynamics
        self.mask_lateral = _uniform_conv(mask_kernel_size, 1.0, torus)

        # Current task mask
        self.register_buffer("m", torch.zeros(hidden_size))

        # Kept for expand(): everything needed to rebuild this FTN
        self._config = dict(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            square_dim=square_dim,
            classification=classification,
            layers=layers,
            inner_size=inner_size,
            dropout_p=dropout_p,
            lateral_b=lateral_b,
            lateral_steps=lateral_steps,
            lateral_kernel_size=lateral_kernel_size,
            k=k,
            reconfig_steps=reconfig_steps,
            reconfig_lr=reconfig_lr,
            reconfig_sample_size=reconfig_sample_size,
            mask_kernel_size=mask_kernel_size,
            mask_lateral_steps=mask_lateral_steps,
            mask_lateral_polyak=mask_lateral_polyak,
            mask_lateral_noise_sd=mask_lateral_noise_sd,
            torus=torus,
        )

    # ---------------------------------------------------------------- forward

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone features before masking: (B, hidden_size)."""
        z = self.parallel(x)
        z = self.dropout(z)
        return self._step_feature_lateral(z)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = None, m: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Predict. If y is given (and no explicit mask), the mask is first
        reconfigured on this batch - per-batch reconfiguration lives here.

        Args:
            x: Inputs (B, input_size)
            y: Optional targets used only to reconfigure the mask
            m: Optional explicit mask overriding the internal one (no reconfig)

        Returns:
            Predictions (B, output_size)
        """
        if m is None:
            if y is not None:
                self.reconfigure(x, y)
            m = self.m
        z = self.features(x)
        return self.linear(z * m.reshape(1, -1))

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Task loss: cross-entropy on one-hot targets, or MSE."""
        if self.classification:
            target = torch.where(y == 1)[1]
            return F.cross_entropy(pred, target)
        return F.mse_loss(pred.squeeze(), y.squeeze())

    # ---------------------------------------------------------- reconfiguration

    @torch.enable_grad()
    def reconfigure(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Reconfigure the mask for the task represented by (x, y).

        Annealed straight-through KWTA: optimize a real-valued mask logit with
        Adam, where the forward mask is kwta(logits + noise) with an annealed
        noise temperature and a sigmoid STE for gradients. Keeps the best mask
        by loss, then (optionally) applies lateral smoothing + re-sparsifies.

        All weights are detached: gradients flow only into the mask logits.
        Updates self.m in place and returns it.
        """
        n = min(self.reconfig_sample_size, x.shape[0])
        x, y = x[:n], y[:n]

        m = nn.Parameter(torch.randn_like(self.m) * 0.00001)
        optimizer = torch.optim.Adam([m], lr=self.reconfig_lr)

        z = self.parallel(x).detach()
        w = self.linear.weight.detach()

        lowest_loss = torch.inf
        best = None
        for i in range(self.reconfig_steps):
            sm = torch.sigmoid(m)
            tau = m.detach().std() * math.exp(-i / (self.reconfig_steps / 4))
            noise = torch.randn_like(m) * tau
            mask = kwta(m + noise, k=self.k) + sm - sm.detach()

            y_pred = F.linear(z * mask, w)
            loss = self.loss(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < lowest_loss:
                best = mask.clone().detach()
                lowest_loss = loss.item()

        # Lateral dynamics (spatial smoothing) + re-sparsification.
        # With mask_lateral_steps == 0 this leaves the KWTA mask unchanged.
        if self.mask_lateral_polyak < 1.0:
            for _ in range(self.mask_lateral_steps):
                m_lat = self._step_mask_lateral(best)
                best = (
                    self.mask_lateral_polyak * best
                    + (1 - self.mask_lateral_polyak) * m_lat
                )
            best = kwta(best, k=self.k)

        self.m.copy_(best)
        return self.m

    # ------------------------------------------------------------- neurogenesis

    def expand(self, new_hidden_size: int) -> "FTN":
        """
        Neurogenesis: return a larger FTN with this one's weights copied into
        the corresponding sub-tensors. New units are freshly initialized.
        """
        cfg = dict(self._config, hidden_size=new_hidden_size)
        new = FTN(**cfg)
        with torch.no_grad():
            for w_old, w_new in zip(self.parallel.weights, new.parallel.weights):
                w_new[: w_old.shape[0]] = w_old
            for b_old, b_new in zip(self.parallel.biases, new.parallel.biases):
                b_new[: b_old.shape[0]] = b_old
            new.linear.weight[:, : self.hidden_size] = self.linear.weight
            new.m[: self.hidden_size] = self.m
        return new

    # ----------------------------------------------------------------- lateral

    def _step_feature_lateral(self, h: torch.Tensor) -> torch.Tensor:
        """Optional lateral smoothing of backbone features (no-op by default)."""
        if self.lateral_steps == 0 or self.lateral_b == 1.0:
            return h
        hshape = h.shape
        h = h.reshape(-1, 1, self.square_dim, self.square_dim)
        for _ in range(self.lateral_steps):
            h = self.lateral_b * h + (1 - self.lateral_b) * self.lateral(h)
        return h.reshape(hshape)

    def _step_mask_lateral(self, m: torch.Tensor) -> torch.Tensor:
        """
        One lateral-smoothing step on the mask. Blocks of square_dim**2 units
        are smoothed independently, so each block (e.g. pre/post neurogenesis
        populations) keeps its own spatial topology.
        """
        g = mask_to_grid(m, self.square_dim).unsqueeze(0).unsqueeze(0)
        g = self.mask_lateral(g).squeeze(0).squeeze(0)
        if self.mask_lateral_noise_sd > 0:
            g = g + torch.randn_like(g) * self.mask_lateral_noise_sd
        return mask_to_vec(g, self.square_dim)
