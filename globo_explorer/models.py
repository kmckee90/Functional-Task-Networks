"""Vision models for the globo explorer.

PyTorch port of the conv2d reservoir + deep conv2d decoder design from
Fluid_1S (obelisk: research/modeling/agents/fluid_1S/fluid_1.py +
research/modeling/reservoirs/conv2d_esn.py).

The reservoir is a *fixed* (untrained) 2D echo-state network operating on the
first-person RGB image. Only the deep conv2d decoder ("readout") is trained.
A feedback vector (previous action / reward / binned novelty) is projected by a
*fixed* random projection into an extra input channel, exactly as Fluid_1S folds
feedback into a 2D channel before the reservoir.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _freeze(module: nn.Module) -> nn.Module:
    for p in module.parameters():
        p.requires_grad = False
    return module


class Conv2dReservoir(nn.Module):
    """Fixed 2D echo-state network reservoir for image input.

    Mirrors ``EchoStateNetwork2D_Deep``:
      h_{t+1} = (1-tc)*h_t + tc*tanh(cnn1(input_scale*x) + cnn2(recur_scale*h_t))
    with ``cnn1`` a strided downsampling conv (kernel=stride, sparsified) and
    ``cnn2`` a circular-padded recurrent conv. All weights are frozen.
    """

    def __init__(
        self,
        in_channels: int,
        num_channels: int = 64,
        kernel_size: int = 7,
        stride: int = 7,
        input_scale: float = 2.375 * 10,
        recur_scale: float = 1.49,
        tc: float = 1.0,
        in_sparsity: float = 0.5,
        seed: int = 0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.input_scale = input_scale
        self.recur_scale = recur_scale
        self.tc = tc

        gen = torch.Generator().manual_seed(seed)

        self.cnn1 = nn.Conv2d(in_channels, num_channels, kernel_size=kernel_size,
                              stride=stride, padding=0, bias=False)
        self.cnn2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1,
                              padding=1, padding_mode="circular", bias=False)

        with torch.no_grad():
            w1 = torch.empty_like(self.cnn1.weight).normal_(generator=gen)
            mask = (torch.rand(self.cnn1.weight.shape, generator=gen) < in_sparsity).float()
            self.cnn1.weight.copy_(w1 * mask)
            self.cnn2.weight.normal_(generator=gen)

        _freeze(self.cnn1)
        _freeze(self.cnn2)

    def hidden_shape(self, in_h: int, in_w: int) -> tuple[int, int, int]:
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1
        return (self.num_channels, out_h, out_w)

    def init_hidden(self, batch: int, in_h: int, in_w: int, device=None) -> torch.Tensor:
        c, h, w = self.hidden_shape(in_h, in_w)
        return torch.zeros(batch, c, h, w, device=device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) image (+feedback) channels in [0,1]; hidden: (B, num_channels, h, w)."""
        pre = self.cnn1(self.input_scale * x) + self.cnn2(self.recur_scale * hidden)
        return (1 - self.tc) * hidden + self.tc * torch.tanh(pre)


class FeedbackToChannel(nn.Module):
    """Fixed random projection of the feedback vector into a single 2D channel.

    Fixed (not trained) so that stored reservoir features stay valid across
    training (the reservoir + its input encoding are a stationary function).
    """

    def __init__(self, feedback_size: int, in_h: int, in_w: int, seed: int = 1):
        super().__init__()
        self.in_h, self.in_w = in_h, in_w
        self.proj = nn.Linear(feedback_size, in_h * in_w)
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.proj.weight.normal_(generator=gen)
            self.proj.bias.zero_()
        _freeze(self.proj)

    @torch.no_grad()
    def forward(self, feedback: torch.Tensor) -> torch.Tensor:
        b = feedback.shape[0]
        return torch.tanh(self.proj(feedback)).reshape(b, 1, self.in_h, self.in_w)


class DeepConvDecoder(nn.Module):
    """Trained deep conv2d readout on the reservoir hidden state.

    Mirrors the Fluid_1S "classifier": 3x (Conv2d 3x3 s1 p1 + ReLU) -> flatten
    -> Linear to ``out_size`` (number of discrete actions for DQN).
    """

    def __init__(self, in_channels: int, hidden_h: int, hidden_w: int,
                 out_size: int, layer_size: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, layer_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(layer_size, layer_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(layer_size, layer_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.head = nn.Linear(layer_size * hidden_h * hidden_w, out_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(hidden))


def downscale_embedding(img_chw: torch.Tensor, out_h: int, out_w: int,
                        grayscale: bool = True) -> torch.Tensor:
    """Downscale an image (or batch) to a flat embedding for RECODE.

    img_chw: (C, H, W) or (B, C, H, W) float in [0,1].
    Returns (D,) or (B, D) where D = out_h*out_w*(1 or C).
    """
    single = img_chw.dim() == 3
    if single:
        img_chw = img_chw.unsqueeze(0)
    x = img_chw
    if grayscale and x.shape[1] == 3:
        # luminance
        w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        x = (x * w).sum(dim=1, keepdim=True)
    x = F.adaptive_avg_pool2d(x, (out_h, out_w))
    emb = x.flatten(1)
    return emb.squeeze(0) if single else emb
