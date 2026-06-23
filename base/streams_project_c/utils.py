"""
utils.py - Utility functions for masked network experiments.
"""

import numpy as np
import torch
from PIL import Image


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a PyTorch module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def create_frame(m: torch.Tensor, dim: int) -> Image.Image:
    """Create a grayscale PIL Image frame from a mask tensor."""
    frame_data = m.reshape(dim, dim) * 255
    frame = Image.fromarray(frame_data.detach().cpu().numpy().astype(np.uint8))
    return frame


def create_rgb_frame(m: torch.Tensor) -> Image.Image:
    """Create an RGB PIL Image frame from a tensor."""
    frame = Image.fromarray(m.detach().cpu().numpy().astype(np.uint8))
    return frame
