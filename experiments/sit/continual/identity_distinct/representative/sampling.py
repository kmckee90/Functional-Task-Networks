"""Uniform admission across known task horizons; no model RNG consumption."""
import numpy as np


def storage_steps(seed, block, steps=1500, capacity=384, tasks=3, entry_size=4):
    if capacity % (tasks * entry_size):
        raise ValueError('Capacity must split into equal whole entries per task')
    quota = capacity // (tasks * entry_size)
    if not 0 < quota <= steps:
        raise ValueError('Invalid task quota')
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7319, block]))
    # Equivalent to sequential admission with probability remaining_quota /
    # remaining_steps. Every step has the same marginal inclusion probability.
    return sorted(rng.choice(steps, size=quota, replace=False).tolist())
