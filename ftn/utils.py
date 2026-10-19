"""
utils.py - Utility functions for FTN experiments.
"""

import torch


def count_parameters(module: torch.nn.Module, trainable_only: bool = True) -> int:
    """Count the number of parameters in a PyTorch module."""
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())
