"""Exact-k masks and shared task loss."""
import torch
import torch.nn.functional as F

def kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    """K-winner-take-all: binary mask with 1s at the top-k values of x."""
    idx = torch.topk(x, k).indices
    out = torch.zeros_like(x)
    out[idx] = 1
    return out

def task_loss(pred, y, classification: bool=True):
    """Single-label softmax cross-entropy on one-hot targets (classification),
    else MSE. Shared by training and the mask search."""
    if classification:
        target = torch.where(y == 1)[1]
        return F.cross_entropy(pred, target)
    return F.mse_loss(pred.squeeze(), y.squeeze())

def mask_to_grid(m, block_dim):
    """View one flat atom mask as its square display grid."""
    if m.numel() != block_dim ** 2:
        raise ValueError("An atom mask must contain square_dim**2 entries")
    return m.reshape(block_dim, block_dim)
