"""RECODE density estimator — JAX port of ../globo_explorer/recode/recode.py.

RECODE (arXiv 2305.01521) keeps a fixed-size set of atoms (centroids) with
per-atom visitation counts and reports novelty = count-weighted kNN distance.
This is the default ``knn_reward`` path only (the alternative kernels in the
torch original are unused by globo_explorer's config).

Faithful to the original, with two deliberate, GPU-friendly changes for the
single-world lifetime:
  * State is an immutable pytree (`RecodeState`); `update` is a jitted pure
    function using `lax.cond` for vacant-insert vs merge/evict.
  * The eviction target `d2_nearest` is recomputed fresh from a pairwise matmul
    each eviction, instead of the torch version's lazily-maintained (and stale)
    incremental array. For 1 world this is a single M×M matmul — negligible on
    GPU — and is arguably more correct.
"""
from __future__ import annotations

from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

_INF = jnp.float32(jnp.inf)


class RecodeState(NamedTuple):
    memory: jnp.ndarray     # (M, D) f32 — atoms
    counts: jnp.ndarray     # (M,)  f32 — visitation mass (0 == vacant slot)
    d2_nearest: jnp.ndarray  # (M,) f32 — each atom's sq-dist to its nearest atom
    d2_ema: jnp.ndarray     # scalar f32 — EMA of mean sq-dist to k nearest
    size: jnp.ndarray       # scalar int32 — occupied atoms


def _sqdist_to(vec: jnp.ndarray, memory: jnp.ndarray) -> jnp.ndarray:
    """Squared L2 distance from a single (D,) vector to each row of (M,D)."""
    diff = memory - vec[None, :]
    return jnp.sum(diff * diff, axis=1)


def _pairwise_nearest_sq(memory: jnp.ndarray) -> jnp.ndarray:
    """For each atom, squared distance to its nearest *other* atom. (M,) via matmul."""
    sq = jnp.sum(memory * memory, axis=1)               # (M,)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (memory @ memory.T)
    d2 = d2 + jnp.eye(memory.shape[0], dtype=memory.dtype) * _INF  # exclude self
    return jnp.min(jnp.maximum(d2, 0.0), axis=1)


def _update_d2_ema(d2_ema, d2_masked, ema_k: int, decay_rate: float):
    """EMA of the mean of the k smallest (masked) squared distances."""
    vals = -jax.lax.top_k(-d2_masked, ema_k)[0]          # k smallest
    mean = jnp.mean(vals)
    finite = jnp.isfinite(mean)
    blended = jnp.where(d2_ema == 0.0, mean, decay_rate * d2_ema + (1.0 - decay_rate) * mean)
    return jnp.where(finite, blended, d2_ema)


def knn_reward(embeddings: jnp.ndarray, memory: jnp.ndarray, counts: jnp.ndarray,
               reward_k: int) -> jnp.ndarray:
    """Count-weighted kNN novelty for each (B,D) embedding. Vacant slots (count==0)
    are masked to +inf distance so they never enter the neighborhood. Returns (B,)
    Euclidean distance; 0 when there is insufficient occupied memory."""
    valid = counts > 0.0                                  # (M,)
    sq = (jnp.sum(embeddings * embeddings, axis=1)[:, None]
          + jnp.sum(memory * memory, axis=1)[None, :]
          - 2.0 * (embeddings @ memory.T))                # (B,M) squared
    d = jnp.sqrt(jnp.maximum(sq, 0.0))
    d = jnp.where(valid[None, :], d, _INF)
    min_d, idxs = jax.lax.top_k(-d, reward_k)             # k smallest
    min_d = -min_d                                        # (B,k)
    cnt = counts[idxs]                                    # (B,k)
    cum = jnp.cumsum(cnt, axis=1)                         # (B,k)
    # searchsorted(cum, k, 'left'): first position with cum >= k == count of cum < k
    kth = jnp.sum(cum < float(reward_k), axis=1)
    kth = jnp.clip(kth, 0, reward_k - 1)
    reward = jnp.take_along_axis(min_d, kth[:, None], axis=1)[:, 0]
    return jnp.nan_to_num(reward, posinf=0.0, neginf=0.0)  # too-few atoms → 0 novelty


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def recode_update(state: RecodeState, emb: jnp.ndarray, key,
                  memory_size: int, ema_k: int, relative_tolerance: float,
                  decay_rate: float, gamma: float, insertion_probability):
    """Insert one embedding (D,). Vacant slots fill first; once full, merge into the
    nearest atom or (with prob p, if far enough) evict the most-redundant atom."""
    M = memory_size

    def vacant(st: RecodeState) -> RecodeState:
        i = st.size
        mem = st.memory.at[i].set(emb)
        cnts = st.counts.at[i].set(1.0)
        new_size = i + 1
        d2 = _sqdist_to(emb, mem)
        occupied = jnp.arange(M) < new_size
        d2 = jnp.where(occupied, d2, _INF)
        d2 = jnp.where(d2 == 0.0, _INF, d2)
        d2ema = jax.lax.cond(new_size > ema_k,
                             lambda: _update_d2_ema(st.d2_ema, d2, ema_k, decay_rate),
                             lambda: st.d2_ema)
        return st._replace(memory=mem, counts=cnts, size=new_size, d2_ema=d2ema)

    def full(st: RecodeState) -> RecodeState:
        d2 = _sqdist_to(emb, st.memory)
        d2ema = _update_d2_ema(st.d2_ema, jnp.where(d2 == 0.0, _INF, d2), ema_k, decay_rate)
        counts = st.counts * gamma
        nearest = jnp.argmin(d2)
        do_evict = (d2[nearest] > relative_tolerance * d2ema) & \
                   (jax.random.uniform(key) < insertion_probability)

        def evict(_):
            d2n = _pairwise_nearest_sq(st.memory)
            remove_idx = jnp.argmin(d2n)
            d2_rm = _sqdist_to(st.memory[remove_idx], st.memory).at[remove_idx].set(_INF)
            redist = jnp.argmin(d2_rm)
            cr, rm = counts[redist], counts[remove_idx]
            new_red = (cr * st.memory[redist] + rm * st.memory[remove_idx]) / (cr + rm)
            mem = st.memory.at[redist].set(new_red).at[remove_idx].set(emb)
            cnts = counts.at[redist].add(rm).at[remove_idx].set(1.0)
            d2n = d2n.at[remove_idx].set(d2[nearest])
            return st._replace(memory=mem, counts=cnts, d2_nearest=d2n, d2_ema=d2ema)

        def merge(_):
            cn = counts[nearest]
            new_atom = (cn * st.memory[nearest] + emb) / (cn + 1.0)
            mem = st.memory.at[nearest].set(new_atom)
            cnts = counts.at[nearest].add(1.0)
            return st._replace(memory=mem, counts=cnts, d2_ema=d2ema)

        return jax.lax.cond(do_evict, evict, merge, operand=None)

    return jax.lax.cond(state.size < M, vacant, full, state)


class RECODE:
    """Stateful wrapper mirroring the torch RECODE surface (get_reward/update/state)."""

    def __init__(self, observation_size: int, memory_size: int, ema_k: int = 10,
                 reward_k: int = 10, relative_tolerance: float = 0.2,
                 decay_rate: float = 0.9995, gamma: float = 0.9995,
                 insertion_probability: float = 0.5):
        self.obs_size = observation_size
        self.memory_size = memory_size
        self.ema_k = ema_k
        self.reward_k = reward_k
        self.relative_tolerance = relative_tolerance
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.insertion_probability = insertion_probability
        self.state = RecodeState(
            memory=jnp.zeros((memory_size, observation_size), jnp.float32),
            counts=jnp.zeros((memory_size,), jnp.float32),
            d2_nearest=jnp.full((memory_size,), _INF, jnp.float32),
            d2_ema=jnp.float32(0.0),
            size=jnp.int32(0),
        )

    def get_reward(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        if embeddings.ndim == 1:
            embeddings = embeddings[None]
        return knn_reward(embeddings, self.state.memory, self.state.counts, self.reward_k)

    def update(self, emb: jnp.ndarray, key) -> None:
        if emb.ndim == 2:  # single-lifetime: one embedding per step
            emb = emb[0]
        self.state = recode_update(
            self.state, emb, key, self.memory_size, self.ema_k,
            self.relative_tolerance, self.decay_rate, self.gamma,
            jnp.float32(self.insertion_probability))

    @property
    def size(self) -> int:
        return int(self.state.size)

    def get_state(self) -> dict[str, Any]:
        return {k: getattr(self.state, k) for k in self.state._fields}

    def set_state(self, d: dict[str, Any]) -> None:
        self.state = RecodeState(**{k: d[k] for k in self.state._fields})
