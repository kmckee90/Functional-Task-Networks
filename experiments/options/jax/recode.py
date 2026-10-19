"""JAX RECODE: online k-NN clustering of observations into a fixed-size atom
memory, as a single jitted lax.scan over the embedding batch.

Faithful port of m2e/recode/recode.py's clustering path (the only part
train_options_2 uses -- the novelty reward is unused under random_explore). Per
embedding, in order:

  * size < memory_size -> VACANT insert (counts=1); update d2_ema once size>k.
  * size == memory_size -> _merge_update: either evict-and-redistribute the
    weakest atom (when the point is far from all atoms AND a Bernoulli(p) fires)
    or merge the point into its nearest atom (running mean, counts++).

State (memory, counts, d2_nearest, d2_ema, size) is carried through the scan;
data-dependent branches use lax.cond so it all stays inside one XLA program.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


def init_state(memory_size, obs_dim):
    return {
        "memory": jnp.zeros((memory_size, obs_dim), jnp.float32),
        "counts": jnp.zeros((memory_size,), jnp.float32),
        "d2_nearest": jnp.full((memory_size,), jnp.inf, jnp.float32),
        "d2_ema": jnp.float32(0.0),
        "size": jnp.int32(0),
    }


def _topk_mean(d2, k):
    """Mean of the k smallest entries (matches torch d2.topk(k, largest=False).mean)."""
    return -jax.lax.top_k(-d2, k)[0].mean()


def _ema_update(d2_ema, mean, decay):
    """torch _update_d2_ema: skip if mean is inf; seed if ema==0 else decay-mix."""
    valid = jnp.isfinite(mean)
    seeded = jnp.where(d2_ema == 0, mean, decay * d2_ema + (1.0 - decay) * mean)
    return jnp.where(valid, seeded, d2_ema)


@functools.partial(jax.jit, static_argnums=(2, 3, 4, 5, 6, 7))
def update(state, embeddings, memory_size, k, kappa, decay, gamma, insertion_prob, key):
    """Fold `embeddings` (N, obs_dim) into the atom memory. Returns new state."""
    M = memory_size

    def vacant(st, e, kk):
        i = st["size"]
        memory = st["memory"].at[i].set(e)
        counts = st["counts"].at[i].set(1.0)
        size = i + 1
        idx = jnp.arange(M)
        d2 = jnp.sum((e - memory) ** 2, axis=1)
        d2 = jnp.where((idx < size) & (idx != i), d2, jnp.inf)  # valid, non-self
        mean = _topk_mean(d2, k)
        d2_ema = jnp.where(size > k, _ema_update(st["d2_ema"], mean, decay), st["d2_ema"])
        return {**st, "memory": memory, "counts": counts, "size": size, "d2_ema": d2_ema}

    def merge(st, e, kk):
        memory, counts = st["memory"], st["counts"]
        d2 = jnp.sum((e - memory) ** 2, axis=1)
        d2_ema = _ema_update(st["d2_ema"], _topk_mean(d2, k), decay)
        counts = counts * gamma
        nearest = jnp.argmin(d2)
        kk, bk = jax.random.split(kk)
        far = d2[nearest] > kappa * d2_ema
        fire = jax.random.uniform(bk) < insertion_prob
        evict = far & fire

        def do_merge(_):
            cn = counts[nearest]
            m = memory.at[nearest].set((cn * memory[nearest] + e) / (cn + 1.0))
            c = counts.at[nearest].add(1.0)
            return m, c, st["d2_nearest"]

        def do_evict(_):
            # Lazily (re)compute d2_nearest the first time it is all-inf.
            def recompute(_):
                D = jnp.sum((memory[:, None, :] - memory[None, :, :]) ** 2, axis=-1)
                D = jnp.where(jnp.eye(M, dtype=bool), jnp.inf, D)  # self -> inf
                return jnp.min(D, axis=1)
            d2n = jax.lax.cond(jnp.all(jnp.isinf(st["d2_nearest"])),
                               recompute, lambda _: st["d2_nearest"], None)
            remove = jnp.argmin(d2n)
            d2_rem = jnp.sum((memory[remove] - memory) ** 2, axis=1).at[remove].set(jnp.inf)
            redist = jnp.argmin(d2_rem)
            cr, cm = counts[redist], counts[remove]
            m = memory.at[redist].set((cr * memory[redist] + cm * memory[remove]) / (cr + cm))
            c = counts.at[redist].set(cr + cm)
            m = m.at[remove].set(e)
            c = c.at[remove].set(1.0)
            d2n = d2n.at[remove].set(d2[nearest])
            return m, c, d2n

        memory, counts, d2_nearest = jax.lax.cond(evict, do_evict, do_merge, None)
        return {**st, "memory": memory, "counts": counts, "d2_nearest": d2_nearest, "d2_ema": d2_ema}

    def step(carry, e):
        st, key = carry
        key, sk = jax.random.split(key)
        st = jax.lax.cond(st["size"] < M,
                          lambda s: vacant(s, e, sk),
                          lambda s: merge(s, e, sk), st)
        return (st, key), None

    (state, _), _ = jax.lax.scan(step, (state, key), embeddings)
    return state


def all_pos(state):
    """Atoms currently in memory (memory[:size]) -- host numpy for FPS."""
    n = int(state["size"])
    return jax.device_get(state["memory"][:n])
