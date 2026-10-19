"""Torch-free random-walk collection worker for the parallel trainers.

Kept in its own module (numpy + the env only, NO torch import) so that
multiprocessing 'spawn' workers start fast and never touch CUDA. Each worker
builds its own env (whichever ``env_name`` the trainer selected) from the SAME
``world_kwargs`` (so the floor patchwork is identical to the main process) and
runs a random walk, returning the collected transitions as numpy arrays.
"""

from __future__ import annotations

import numpy as np

from envs import make_env


def collect_worker(args):
    """args = (env_name, world_kwargs, n_steps, seed).
    Returns (obs, act, next_obs, pos, next_pos, boundary) as float32 arrays."""
    env_name, world_kwargs, n_steps, seed = args
    rng = np.random.default_rng(seed)
    env = make_env(env_name, **world_kwargs)
    obs, info = env.reset(seed=seed)

    O = np.empty((n_steps, env.obs_dim), np.float32)
    A = np.empty((n_steps, 2), np.float32)
    NO = np.empty((n_steps, env.obs_dim), np.float32)
    P = np.empty((n_steps, 2), np.float32)
    NP = np.empty((n_steps, 2), np.float32)
    B = np.empty((n_steps,), np.float32)

    for t in range(n_steps):
        a = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)  # uniform random action
        nobs, _, _, trunc, ninfo = env.step(a)
        O[t] = obs; A[t] = a; NO[t] = nobs
        P[t] = info["pos"]; NP[t] = ninfo["pos"]; B[t] = float(trunc)
        if trunc:
            obs, info = env.reset()
        else:
            obs, info = nobs, ninfo
    # Mark the chunk's final transition as an episode boundary. Workers' chunks are
    # concatenated downstream, and each is a SEPARATE trajectory; without this the
    # graph builder would add a spurious edge (and exploration.png a jump line)
    # from this worker's last membership to the next worker's first.
    if n_steps > 0:
        B[-1] = 1.0
    return O, A, NO, P, NP, B
