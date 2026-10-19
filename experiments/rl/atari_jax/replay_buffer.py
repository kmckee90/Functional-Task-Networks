"""
replay_buffer.py - JIT-compatible circular replay buffers using fixed-size JAX arrays.

Two buffer types:
  1. ReplayBufferState: standard DQN replay for the current game.
  2. MaskReplayState: cross-game replay tagged with per-game masks.
     Uses Gumbel-max trick for JIT-compatible masked sampling.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ═══════════════════════════════════════════════════════════════════════
# Standard DQN Replay Buffer
# ═══════════════════════════════════════════════════════════════════════


class ReplayBufferState(NamedTuple):
    obs: jnp.ndarray        # (capacity, C, H, W)
    actions: jnp.ndarray    # (capacity,) int32
    rewards: jnp.ndarray    # (capacity,) float32
    next_obs: jnp.ndarray   # (capacity, C, H, W)
    dones: jnp.ndarray      # (capacity,) float32
    write_pos: jnp.ndarray  # scalar int32
    size: jnp.ndarray       # scalar int32


def init_buffer(capacity, obs_shape):
    """Allocate a zeroed replay buffer of fixed capacity."""
    return ReplayBufferState(
        obs=jnp.zeros((capacity, *obs_shape), dtype=jnp.float32),
        actions=jnp.zeros(capacity, dtype=jnp.int32),
        rewards=jnp.zeros(capacity, dtype=jnp.float32),
        next_obs=jnp.zeros((capacity, *obs_shape), dtype=jnp.float32),
        dones=jnp.zeros(capacity, dtype=jnp.float32),
        write_pos=jnp.int32(0),
        size=jnp.int32(0),
    )


def add(buf, obs, action, reward, next_obs, done, capacity):
    """Write one transition at write_pos % capacity, return updated state."""
    idx = buf.write_pos % capacity
    return ReplayBufferState(
        obs=buf.obs.at[idx].set(obs),
        actions=buf.actions.at[idx].set(action),
        rewards=buf.rewards.at[idx].set(reward),
        next_obs=buf.next_obs.at[idx].set(next_obs),
        dones=buf.dones.at[idx].set(done),
        write_pos=buf.write_pos + 1,
        size=jnp.minimum(buf.size + 1, capacity),
    )


def sample(buf, key, batch_size):
    """Sample a random batch from valid entries."""
    indices = jax.random.randint(key, (batch_size,), 0, buf.size)
    return (
        buf.obs[indices],
        buf.actions[indices],
        buf.rewards[indices],
        buf.next_obs[indices],
        buf.dones[indices],
    )


# ═══════════════════════════════════════════════════════════════════════
# Mask-Correlated Cross-Game Replay Buffer
# ═══════════════════════════════════════════════════════════════════════


class MaskReplayState(NamedTuple):
    obs: jnp.ndarray        # (capacity, C, H, W)
    actions: jnp.ndarray    # (capacity,) int32
    rewards: jnp.ndarray    # (capacity,) float32
    next_obs: jnp.ndarray   # (capacity, C, H, W)
    dones: jnp.ndarray      # (capacity,) float32
    masks: jnp.ndarray      # (capacity, hidden_size)
    write_pos: jnp.ndarray  # scalar int32
    size: jnp.ndarray       # scalar int32
    game_count: jnp.ndarray # per-game write counter, scalar int32


def init_mask_buffer(capacity, obs_shape, hidden_size):
    """Allocate a zeroed mask replay buffer."""
    return MaskReplayState(
        obs=jnp.zeros((capacity, *obs_shape), dtype=jnp.float32),
        actions=jnp.zeros(capacity, dtype=jnp.int32),
        rewards=jnp.zeros(capacity, dtype=jnp.float32),
        next_obs=jnp.zeros((capacity, *obs_shape), dtype=jnp.float32),
        dones=jnp.zeros(capacity, dtype=jnp.float32),
        masks=jnp.zeros((capacity, hidden_size), dtype=jnp.float32),
        write_pos=jnp.int32(0),
        size=jnp.int32(0),
        game_count=jnp.int32(0),
    )


def mask_add(buf, obs, action, reward, next_obs, done, mask, capacity):
    """Write one transition with mask tag. Returns updated state."""
    idx = buf.write_pos % capacity
    return MaskReplayState(
        obs=buf.obs.at[idx].set(obs),
        actions=buf.actions.at[idx].set(action),
        rewards=buf.rewards.at[idx].set(reward),
        next_obs=buf.next_obs.at[idx].set(next_obs),
        dones=buf.dones.at[idx].set(done),
        masks=buf.masks.at[idx].set(mask),
        write_pos=buf.write_pos + 1,
        size=jnp.minimum(buf.size + 1, capacity),
        game_count=buf.game_count + 1,
    )


def sample_correlated(buf, key, current_mask, n_samples, corr_low, corr_high, capacity):
    """
    Sample transitions whose stored masks partially overlap with current_mask.

    Uses Gumbel-max trick for JIT-compatible categorical sampling from the
    valid set (entries with correlation in [corr_low, corr_high]).

    Returns:
        (obs, actions, rewards, next_obs, dones, stored_mask, has_valid)
    """
    mask_sum = jnp.maximum(current_mask.sum(), 1.0)
    corrs = (buf.masks * current_mask[None, :]).sum(-1) / mask_sum

    valid_idx = jnp.arange(capacity) < buf.size
    valid_corr = (corrs > corr_low) & (corrs <= corr_high)
    valid = valid_idx & valid_corr

    # Gumbel-max trick: sample from valid set without replacement
    gumbel = jax.random.gumbel(key, shape=(capacity,))
    scores = jnp.where(valid, gumbel, -1e30)
    chosen = jnp.argsort(-scores)[:n_samples]

    has_valid = valid.any()

    return (
        buf.obs[chosen],
        buf.actions[chosen],
        buf.rewards[chosen],
        buf.next_obs[chosen],
        buf.dones[chosen],
        buf.masks[chosen[0]],
        has_valid,
    )
