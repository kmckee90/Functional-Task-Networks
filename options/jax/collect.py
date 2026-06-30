"""Parallel random-walk collection as ONE jitted vmap+scan.

The torch trainer (train_options_2.py / _par_collect.py) spawns W OS processes,
each running an independent allocentric random walk over the SAME world for a
slice of the step budget, then concatenates their transitions. Here the *exact*
same semantics are a single device program:

  * ``rollout`` is a ``lax.scan`` over one worker's chunk (pure env dynamics, no
    Python per-step overhead).
  * ``vmap(rollout)`` runs all W worker chunks in parallel on-device.
  * each worker's final transition is flagged as an episode boundary (matching
    _par_collect's ``B[-1] = 1.0``), so the concatenated stream has no spurious
    cross-worker graph edges / trajectory jumps.

There is no torch buffer copy: the concatenated arrays ARE the round's batch.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from color_world import block_move, color_at, coord_at, sample_free
from ftn import actor_forward


def _aug(o, oh, ah):
    """Flatten current obs + history window into one augmented vector / row.
    o:(...,raw) oh:(...,n,raw) ah:(...,n,act) -> (...,raw*(n+1)+act*n)."""
    lead = o.shape[:-1]
    return jnp.concatenate([o, oh.reshape(*lead, -1), ah.reshape(*lead, -1)], axis=-1)


def _env_obs(pos, sites, colors, size, coord_obs):
    """Per-step observation: continuous (x,y) in [-1,1]^2 when coord_obs, else the
    nearest-Voronoi-site color vector. coord_obs is a static Python flag."""
    return coord_at(pos, size) if coord_obs else color_at(pos, sites, colors)


def _rollout(key, sites, colors, n_steps, max_speed, size, max_steps, torus, n_hist, polys,
             coord_obs):
    """One worker: `n_steps` of allocentric uniform random walk via lax.scan.

    Returns (obs, act, next_obs, pos, next_pos, boundary, aug_obs, aug_next), each
    leading dim n_steps. obs/next_obs are the RAW color (used by RECODE/graph);
    aug_obs/aug_next are the history-augmented vectors [o, n prev obs, n prev acts]
    fed to the actor/critic. n_hist=0 -> aug == raw. Episode truncates at max_steps
    (reset to a fresh pose, history cleared); the chunk's last transition is forced
    to a boundary. torus=True wraps the box edges instead of clipping. polys (P,V,2)
    are convex obstacles the walk cannot enter (empty P -> the exact original walk)."""
    raw, ad = (2 if coord_obs else colors.shape[-1]), 2
    has_obs = polys.shape[0] > 0
    key, rk = jax.random.split(key)
    pos0 = (sample_free(rk, (2,), size, polys) if has_obs
            else jax.random.uniform(rk, (2,), minval=0.02 * size, maxval=0.98 * size))
    oh0, ah0 = jnp.zeros((n_hist, raw)), jnp.zeros((n_hist, ad))

    def step(carry, _):
        pos, steps, oh, ah, key = carry
        key, ak, rk = jax.random.split(key, 3)
        a = jax.random.uniform(ak, (2,), minval=-1.0, maxval=1.0)
        disp = a * max_speed
        npos = jnp.mod(pos + disp, size) if torus else jnp.clip(pos + disp, 0.0, size)
        if has_obs:
            npos = block_move(pos, npos, polys)        # forbid passing into obstacles
        steps1 = steps + 1
        trunc = steps1 >= max_steps
        o = _env_obs(pos, sites, colors, size, coord_obs)
        no = _env_obs(npos, sites, colors, size, coord_obs)
        aug_obs = _aug(o, oh, ah)
        noh = jnp.concatenate([o[None], oh], axis=0)[:n_hist]   # roll in current
        nah = jnp.concatenate([a[None], ah], axis=0)[:n_hist]
        aug_next = _aug(no, noh, nah)
        out = (o, a, no, pos, npos, trunc.astype(jnp.float32), aug_obs, aug_next)
        rpos = (sample_free(rk, (2,), size, polys) if has_obs
                else jax.random.uniform(rk, (2,), minval=0.02 * size, maxval=0.98 * size))
        pos_next = jnp.where(trunc, rpos, npos)
        steps_next = jnp.where(trunc, 0, steps1)
        oh_next = jnp.where(trunc, jnp.zeros_like(noh), noh)    # clear history on reset
        ah_next = jnp.where(trunc, jnp.zeros_like(nah), nah)
        return (pos_next, steps_next, oh_next, ah_next, key), out

    _, outs = jax.lax.scan(step, (pos0, jnp.int32(0), oh0, ah0, key), None, length=n_steps)
    O, A, NO, P, NP, B, AO, AN = outs
    B = B.at[-1].set(1.0)
    return O, A, NO, P, NP, B, AO, AN


# Jit the raw worker (for the remainder chunk) and a vmapped fleet (W workers).
# n_steps (3), torus (7), n_hist (8), coord_obs (10) are static; only the PRNG key
# is vmapped (sites/colors/polys are shared device arrays).
rollout = jax.jit(_rollout, static_argnums=(3, 7, 8, 10))
_rollout_vmapped = jax.jit(
    jax.vmap(_rollout, in_axes=(0, None, None, None, None, None, None, None, None, None, None), out_axes=0),
    static_argnums=(3, 7, 8, 10),
)


def collect_parallel(key, world, total_steps, workers, history_len=0):
    """Run `workers` parallel worker chunks covering `total_steps` env steps.

    Returns (obs, act, next_obs, pos, next_pos, boundary, aug_obs, aug_next) as
    concatenated jnp arrays in worker order. aug_* carry the history window for
    training; the raw obs/next_obs slots are unchanged so RECODE/graph still use
    them directly."""
    per = total_steps // workers
    rem = total_steps - per * workers
    sites, colors, polys = world.sites_j, world.colors_j, world.polys_j
    ms, sz, mx, tor = world.max_speed, world.size, world.max_steps, world.torus
    co = bool(world.coord_obs)
    n = int(history_len)

    keys = jax.random.split(key, workers)
    outs = _rollout_vmapped(keys, sites, colors, per, ms, sz, mx, tor, n, polys, co)
    # (W, per, ...) -> (W*per, ...), preserving worker order.
    flat = [x.reshape((-1,) + x.shape[2:]) for x in outs]

    if rem > 0:  # remainder chunk (torch gives the remainder to the last worker)
        rk = jax.random.fold_in(key, workers)
        ro = rollout(rk, sites, colors, per + rem, ms, sz, mx, tor, n, polys, co)
        flat = [jnp.concatenate([f, r], axis=0) for f, r in zip(flat, ro)]
    return tuple(flat)


# ---------------------------------------------------------------------------
# On-policy collection: roll out one option's actor mask, epsilon-greedy.
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(8, 9, 12, 14, 16))
def _policy_scan(key, actor, m_a, sites, colors, centroids, cid,
                 epsilon, n_steps, workers, max_speed, size, torus, max_steps, n_hist, polys,
                 coord_obs):
    """`n_steps` of the option-`cid` policy over `workers` parallel episodes.

    Action = epsilon-greedy over the FTN actor under mask m_a, applied to the
    history-augmented obs (current color + n previous obs/actions). Continuous-
    control: episodes reset ONLY on the per-episode horizon (max_steps), which
    clears the history. Returns the same 8-tuple as the random collector
    (obs, act, next_obs, pos, next_pos, boundary, aug_obs, aug_next). polys (P,V,2)
    are convex obstacles the policy cannot enter (empty P -> the original walk)."""
    raw, ad = (2 if coord_obs else colors.shape[-1]), 2
    has_obs = polys.shape[0] > 0
    key, rk = jax.random.split(key)
    pos0 = (sample_free(rk, (workers, 2), size, polys) if has_obs
            else jax.random.uniform(rk, (workers, 2), minval=0.02 * size, maxval=0.98 * size))
    oh0, ah0 = jnp.zeros((workers, n_hist, raw)), jnp.zeros((workers, n_hist, ad))

    def step(carry, _):
        pos, steps, oh, ah, key = carry
        key, ek, gk, rk = jax.random.split(key, 4)
        o = _env_obs(pos, sites, colors, size, coord_obs)  # (W, raw)
        aug_obs = _aug(o, oh, ah)                          # (W, aug)
        greedy = actor_forward(actor, aug_obs, m_a)        # (W, 2), in [-1, 1]
        rand_a = jax.random.uniform(gk, (workers, 2), minval=-1.0, maxval=1.0)
        explore = (jax.random.uniform(ek, (workers,)) < epsilon)[:, None]
        a = jnp.where(explore, rand_a, greedy)
        disp = a * max_speed
        npos = jnp.mod(pos + disp, size) if torus else jnp.clip(pos + disp, 0.0, size)
        if has_obs:
            npos = block_move(pos, npos, polys)            # forbid passing into obstacles
        no = _env_obs(npos, sites, colors, size, coord_obs)
        steps1 = steps + 1
        ended = steps1 >= max_steps
        noh = jnp.concatenate([o[:, None], oh], axis=1)[:, :n_hist]
        nah = jnp.concatenate([a[:, None], ah], axis=1)[:, :n_hist]
        aug_next = _aug(no, noh, nah)
        out = (o, a, no, pos, npos, ended.astype(jnp.float32), aug_obs, aug_next)
        rpos = (sample_free(rk, (workers, 2), size, polys) if has_obs
                else jax.random.uniform(rk, (workers, 2), minval=0.02 * size, maxval=0.98 * size))
        pos_next = jnp.where(ended[:, None], rpos, npos)
        steps_next = jnp.where(ended, 0, steps1)
        m = ended[:, None, None]
        oh_next = jnp.where(m, jnp.zeros_like(noh), noh)
        ah_next = jnp.where(m, jnp.zeros_like(nah), nah)
        return (pos_next, steps_next, oh_next, ah_next, key), out

    _, outs = jax.lax.scan(step, (pos0, jnp.zeros(workers, jnp.int32), oh0, ah0, key), None, length=n_steps)
    # (n_steps, W, ...) -> (n_steps*W, ...). Order is irrelevant: train_option
    # samples random minibatches and recomputes reward/done per transition.
    return tuple(x.reshape((-1,) + x.shape[2:]) for x in outs)


def collect_on_policy(key, agent, cid, world, total_steps, workers, epsilon):
    """Collect ~`total_steps` transitions under option `cid`'s epsilon-greedy policy.

    Returns (obs, act, next_obs, pos, next_pos, boundary, aug_obs, aug_next) jnp
    arrays, matching collect_parallel so the on-policy trainer feeds train_option
    the same way. The actor is rolled out on the history-augmented obs."""
    per = max(1, total_steps // workers)
    cents = jnp.asarray(agent.registry.centroids)
    return _policy_scan(key, agent.actor, agent.options[cid]["m_a"], world.sites_j,
                        world.colors_j, cents, int(cid), float(epsilon), per, int(workers),
                        world.max_speed, world.size, bool(world.torus), world.max_steps,
                        int(agent.n_hist), world.polys_j, bool(world.coord_obs))
