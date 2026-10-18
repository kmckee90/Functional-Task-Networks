"""Optimizer selection shared by parent-base mask reconfiguration paths."""
import torch


def validate_reconfig_optimizer(name):
    if name not in ("Adam", "SGD", "RMSprop"):
        raise ValueError("reconfig_optimizer must be Adam, SGD, or RMSprop")
    return name


def make_reconfig_optimizer(params, *, lr, name="Adam"):
    validate_reconfig_optimizer(name)
    return getattr(torch.optim, name)(params, lr=lr)
