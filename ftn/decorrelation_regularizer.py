"""Neuron-output decorrelation, explicitly scoped to weight-training forwards."""

import math

import torch
import torch.nn.functional as F

from ftn import ParallelMLP, ParallelQuadMLP


def neuron_decorrelation(outputs):
    """Mean squared Pearson r over the strict lower triangle of (B, N) outputs.

    Constant neurons have zero correlation. A batch with fewer than two rows,
    or a population with fewer than two neurons, contributes zero.
    """
    if outputs.ndim != 2:
        raise ValueError("neuron outputs must have shape (batch, neurons)")
    z = outputs.float() if outputs.dtype in (torch.float16, torch.bfloat16) else outputs
    batch, neurons = z.shape
    if batch < 2 or neurons < 2:
        return z.sum() * 0
    centered = z - z.mean(dim=0, keepdim=True)
    normalized = F.normalize(centered, p=2, dim=0, eps=1e-8)
    correlation = normalized.T @ normalized
    return correlation.tril(diagonal=-1).square().sum() / (neurons * (neurons - 1) / 2)


def weight_training_forward(model, x, y=None, m=None, coefficient=0.0):
    """Return (predictions, weighted penalty) for one weight-training batch.

    Search finishes before hooks are installed. Capture scalar backbone outputs
    before dropout/masking, so every neuron participates. For stacked/unrolled
    models concatenate the populations across layers/steps on the neuron axis.
    Hooks and captured tensors never persist on the model. LoRA/HML backbones
    are unsupported because they do not expose ParallelMLP scalar populations.
    """
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("DECORRELATION_COEF must be finite and nonnegative")
    if coefficient == 0:
        pred = model(x, y, m=m)
        return pred, pred.new_zeros(())
    populations = [module for module in model.modules()
                   if isinstance(module, (ParallelMLP, ParallelQuadMLP))]
    if not populations:
        raise ValueError("Neuron decorrelation requires a ParallelMLP or ParallelQuadMLP backbone")
    if m is None and y is not None:
        model._reconfig(x, y)
    outputs = []

    def capture(module, args, output):
        outputs.append(output)

    hooks = [module.register_forward_hook(capture) for module in populations]
    try:
        pred = model(x, m=m)
    finally:
        for hook in hooks:
            hook.remove()
    if not outputs:
        raise RuntimeError("Weight-training forward produced no neuron outputs")
    penalty = neuron_decorrelation(torch.cat(outputs, dim=1))
    return pred, coefficient * penalty
