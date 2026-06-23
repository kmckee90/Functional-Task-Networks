"""FTN-DDPG debug harness — fully in JAX on a fast 2-D point-mass control env.

A from-scratch JAX re-implementation of the FTN-DDPG continual-learning pipeline
in `train.py` (MT10 cheat-init), but with MuJoCo/MT10 replaced by a minimal,
GPU-resident continuous-control environment so the whole thing runs in seconds
and FTN+DDPG can be tuned before going back to `train.py`. NOTHING here touches
`train.py`; this is a standalone sandbox.

Environment (the continuous-control analog of train_grid_dqn's grid-world):
    A 2-D double-integrator point mass in the box [-1, 1]^2. State = (pos, vel),
    obs = concat(pos, vel) (4-D), action = 2-D force in [-1, 1]^2. Each "task" is
    a different GOAL location (mutually exclusive: a policy that drives to goal i
    is wrong for goal j). Crucially the goal is NOT in the observation — exactly
    like the grid-world reward-location tasks, the task identity must be carried
    by the FTN mask, not the obs. So the backbone has to host several distinct,
    simple continuous-control policies selected by mask reconfiguration.

    Reward is dense (−distance−to−goal) plus a terminal success bonus, mirroring
    MetaWorld's dense reward (the dense signal is what the r_returns mask search
    correlates Q against). "Cheat init" = spawn the mass within `spawn_radius` of
    the goal so random actions hit success at a useful rate (the analog of
    train.py's `_cheat_reset`); tune it with `--calibrate`.

Continual-learning protocol (identical in spirit to train.py, per task):
    1. Exploration: random-action rollouts on the cheat-initialized env until
       `n_successful_explore` rewarded episodes (or `explore_episodes` total).
    2. Boundary mask reconfig on the rewarded support — critic mask via
       `reconfig_objective` (default −Pearson(Q(s,a), G_t)), then actor mask via
       −Q under the new critic mask. AnnealedSTE.
    3. Exploitation: `n_iters_per_task` iterations of a DECOUPLED collect/train
       loop. Each iteration (a) collects `collect_steps` env steps on
       `n_parallel_train` continuing (auto-reset) envs in ONE jitted scan and
       pushes them to a device replay ring, then (b) runs `train_steps` joint
       DDPG updates in ONE jitted scan. The joint loss is the current-data loss
       (current masks) summed with the loss over up to `replay_max_chunks`
       mask-overlapping archived chunks, each forwarded under its own (m_a, m_c).
       Every `reconfig_every_iters` iterations a mid-phase reconfig closes off the
       current chunk (tagged with the outgoing masks) before committing new ones.
    4. Archive the final chunk, then eval every task: gather a rewarded support by
       random rollouts, then one cold-start AnnealedSTE reconfig per eval episode
       and a greedy rollout.

Speed: the env step, rollouts (lax.scan over time, vmap over parallel envs), the
joint DDPG update (jitted lax.scan over `train_steps`), and the AnnealedSTE mask
search (jitted lax.scan) all run on the GPU in float32. Crucially, exploitation
DECOUPLES inference from training — collect N steps across many parallel envs in
one big scan, then run M gradient updates in one big scan — so the GPU stays
saturated instead of bouncing between a single inference step and a single train
step (which is what made the naive interleaving as slow as MT10). Tune N/M and the
collection parallelism via collect_steps / train_steps / n_parallel_train.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from functools import partial

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# JAX defaults to the GPU (CudaDevice) for array placement and jit; everything
# below is float32 on device. No per-array device juggling needed.


# ---------------------------------------------------------------------------
# Tasks: per-task (hidden) goal locations in the [-1, 1]^2 box. Continuous analog
# of train_grid_dqn's DEFAULT_REWARDS — corners + edge midpoints, all distinct
# directions from the spawn region, so each needs a different control policy.
# ---------------------------------------------------------------------------
DEFAULT_GOALS = (
    (-0.7, -0.7),
    (0.7, -0.7),
    (-0.7, 0.7),
    (0.7, 0.7),
    (-0.7, 0.0),
    (0.0, -0.7),
    (0.0, 0.7),
    (0.7, 0.0),
)


# ---------------------------------------------------------------------------
# DDPGConfig glossary (parameter reference; fields listed bare below). Mirrors
# train.py's DDPGConfig; extra fields are the point-mass env knobs + JAX batching.
# ---------------------------------------------------------------------------
# goals                   : per-task hidden goal xy; one phase each, learned in order.
#
# -- point-mass env (the continuous-control "task") --
# dt                      : integration timestep.
# accel_scale             : force→acceleration gain (action ∈ [-1,1]^2 → accel).
# vel_damp                : per-step velocity damping (<1 bleeds momentum).
# max_speed               : velocity-magnitude clip.
# bound                   : half-extent of the square arena; pos clipped to [-b,b].
# success_radius          : distance-to-goal under which the episode succeeds.
# spawn_radius            : [difficulty] cheat-init disk radius around the goal —
#                           smaller = easier (random actions find reward sooner).
# dist_coef               : weight on the dense −distance reward term.
# step_penalty            : per-step constant penalty (encourages short paths).
# terminal_reward         : bonus added on the success step (episode terminates).
#
# -- experiment --
# seed                    : RNG seed (JAX PRNGKey + numpy).
# explore_episodes        : cap on exploration episodes per task / per eval cell.
# n_successful_explore    : stop exploration after this many rewarded episodes.
# eval_episodes           : greedy rollouts per (phase, task) eval cell.
# max_episode_steps       : truncation horizon.
# n_parallel_explore      : parallel envs per exploration/eval-support scan.
# output_path             : .npz/.png results path (suffixed per baseline + seed).
# log_every               : print a training summary every this many iterations.
#
# -- exploitation: decoupled collect / train (actor-learner loop) --
# n_iters_per_task        : collect→train iterations per task (after boundary reconfig).
# n_parallel_train        : parallel (auto-reset) envs during exploitation collection.
# collect_steps           : N — env steps collected per iteration, per parallel env,
#                           in one jitted scan, before any training.
# train_steps             : M — gradient updates run per iteration, in one jitted scan,
#                           after collection. Decoupling N from M is what keeps the GPU
#                           saturated (vs the per-step inference↔train bouncing).
#
# -- DDPG --
# gamma                   : discount for TD targets and MC returns.
# tau                     : Polyak target-network averaging rate.
# actor_lr / critic_lr    : Adam learning rates.
# batch_size              : transitions per gradient step (current data).
# noise_std               : Gaussian action noise added in exploitation collection
#                           (and prev-mask exploration), pre-clip to [-1,1].
# noise_eval_explore      : Gaussian action noise for eval-time support gathering.
#
# -- FTN backbone (ParallelMLP bank + KWTA-masked readout) --
# ftn_hidden_size         : H — number of parallel sub-networks per network.
# ftn_k                   : active sub-networks selected by the KWTA mask.
# ftn_inner_size          : width inside each parallel MLP.
# ftn_layers              : hidden layers inside each parallel MLP.
# ftn_dropout_p           : dropout on the per-neuron vector (train-time forwards).
#
# -- AnnealedSTE mask reconfiguration --
# reconfig_steps          : inner optimization steps per mask search.
# reconfig_lr             : SGD learning rate for the soft-mask logits.
# reconfig_n              : support batch size (rewarded transitions, resampled to
#                           exactly this many rows so the search shape is fixed).
# mask_init_noise_scale   : cold-start logit init = noise_scale * randn(H).
# mask_warm_start         : warm-start logits from the current mask (eval cold-starts
#                           regardless).
# reconfig_objective      : critic-mask objective; one of:
#     "td"          — MSE(Q(s,a)[m], r + γ·(1-done)·Q_target(s', μ(s'))[m])
#     "mse_returns" — MSE(Q(s,a)[m], G_t)
#     "cos_returns" — −cos(Q(s,a)[m], G_t)
#     "r_returns"   — −Pearson(Q(s,a)[m], G_t)   (default)
#   The actor mask is always selected second, via −Q under the new critic mask.
# reconfig_gamma          : discount for MC returns G_t; None → fall back to gamma.
# reconfig_every_iters    : mid-phase reconfig cadence (iterations); 0 disables. The
#                           support is the most recent collect burst (MC returns over
#                           its auto-reset episodes); the current chunk is archived
#                           under the outgoing masks before the new ones commit.
#
# -- mask-overlap-filtered cross-task replay (summed-loss, per-chunk) --
# replay_max_chunks       : max archived chunks summed into one joint step.
# replay_max_tuples       : transitions sampled per recalled chunk.
# rho_low / rho_high      : recall a chunk if actor OR critic mask overlap
#                           ρ ∈ (rho_low, rho_high].
#
# -- exploration / eval-time mask recovery --
# warmup_policy           : explore policy after task 0 — "random" or "prev_mask".
#
# -- baselines (mutually exclusive) --
# fixed_mask  : disjoint oracle allocation (task i owns [i*k,(i+1)*k); needs N*k<=H).
# no_mask     : all-ones mask (k=H), frozen — dense DDPG over the FTN backbone.
# stored_mask : full FTN training, eval recalls each task's last training mask.
# ---------------------------------------------------------------------------
@dataclass
class DDPGConfig:
    # --- tasks ---
    goals: tuple = DEFAULT_GOALS

    # --- point-mass env ---
    dt: float = 0.05
    accel_scale: float = 10.0
    vel_damp: float = 0.9
    max_speed: float = 2.0
    bound: float = 1.0
    success_radius: float = 0.15
    spawn_radius: float = 0.35
    dist_coef: float = 1.0
    step_penalty: float = 0.0
    terminal_reward: float = 1.0

    # --- experiment ---
    seed: int = 0
    explore_episodes: int = 2000
    n_successful_explore: int = 100
    eval_episodes: int = 8
    max_episode_steps: int = 150
    n_parallel_explore: int = 256
    output_path: str = "results_jax.npz"
    log_every: int = 5

    # --- exploitation: decoupled collect / train (the actor-learner loop) ---
    # Each iteration = collect `collect_steps` env steps on `n_parallel_train`
    # continuing (auto-reset) envs in ONE jitted scan, push them to the replay
    # ring, then run `train_steps` gradient updates in ONE jitted scan. This
    # keeps the GPU busy in big batched chunks instead of bouncing between
    # inference and training every step.
    n_iters_per_task: int = 150  # collect→train iterations per task
    n_parallel_train: int = 128  # parallel envs during exploitation collection
    collect_steps: int = 128  # N — env steps collected per iteration (per env)
    train_steps: int = 128  # M — gradient updates per iteration

    # --- DDPG ---
    gamma: float = 0.95
    tau: float = 0.01
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    batch_size: int = 64
    noise_std: float = (
        0.1  # Gaussian action noise (exploitation collect + prev-mask explore)
    )
    noise_eval_explore: float = (
        0.1  # Gaussian action noise for eval-time support gathering
    )

    # --- FTN backbone ---
    ftn_hidden_size: int = 2048
    ftn_k: int = 512
    ftn_inner_size: int = 8
    ftn_layers: int = 8
    ftn_dropout_p: float = 0.05

    # --- AnnealedSTE ---
    reconfig_steps: int = 120
    reconfig_lr: float = 0.003
    reconfig_n: int = 4096
    mask_init_noise_scale: float = 0.00001
    mask_warm_start: bool = True
    reconfig_objective: str = "r_returns"
    reconfig_gamma: float | None = None
    reconfig_every_iters: int = (
        10  # mid-phase reconfig cadence (iterations); 0 disables
    )

    # --- mask-overlap-filtered cross-task replay ---
    replay_max_chunks: int = 4
    replay_max_tuples: int = 64
    rho_low: float = 0.2
    rho_high: float = 0.8

    # --- exploration / eval-time mask recovery ---
    warmup_policy: str = "random"

    # --- baselines ---
    fixed_mask: bool = False
    no_mask: bool = False
    stored_mask: bool = False


def smoke_ddpg_config() -> DDPGConfig:
    return DDPGConfig(
        goals=((-0.7, -0.7), (0.7, 0.7), (0.7, -0.7)),
        explore_episodes=200,
        n_successful_explore=20,
        eval_episodes=4,
        max_episode_steps=80,
        n_parallel_explore=128,
        n_iters_per_task=40,
        n_parallel_train=32,
        collect_steps=32,
        train_steps=32,
        reconfig_every_iters=10,
        batch_size=32,
        ftn_hidden_size=64,
        ftn_inner_size=4,
        ftn_layers=2,
        ftn_k=16,
        reconfig_steps=30,
        reconfig_n=512,
        log_every=10,
        output_path="results_jax_smoke.npz",
    )


OBS_DIM = 4  # (pos_x, pos_y, vel_x, vel_y)
ACT_DIM = 2


# ---------------------------------------------------------------------------
# Point-mass environment (pure JAX, vectorized over a leading batch dim B)
# ---------------------------------------------------------------------------
#
# EnvParams is a flat pytree of floats passed as a (dynamic) argument so the
# jitted rollouts don't retrace when an unrelated config field changes.


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "dt",
        "accel_scale",
        "vel_damp",
        "max_speed",
        "bound",
        "success_radius",
        "spawn_radius",
        "dist_coef",
        "step_penalty",
        "terminal_reward",
        "gamma",
    ],
    meta_fields=[],
)
@dataclass
class EnvParams:
    dt: jnp.ndarray
    accel_scale: jnp.ndarray
    vel_damp: jnp.ndarray
    max_speed: jnp.ndarray
    bound: jnp.ndarray
    success_radius: jnp.ndarray
    spawn_radius: jnp.ndarray
    dist_coef: jnp.ndarray
    step_penalty: jnp.ndarray
    terminal_reward: jnp.ndarray
    gamma: jnp.ndarray


def env_params_from_cfg(cfg: DDPGConfig) -> EnvParams:
    f = jnp.float32
    return EnvParams(
        dt=f(cfg.dt),
        accel_scale=f(cfg.accel_scale),
        vel_damp=f(cfg.vel_damp),
        max_speed=f(cfg.max_speed),
        bound=f(cfg.bound),
        success_radius=f(cfg.success_radius),
        spawn_radius=f(cfg.spawn_radius),
        dist_coef=f(cfg.dist_coef),
        step_penalty=f(cfg.step_penalty),
        terminal_reward=f(cfg.terminal_reward),
        gamma=f(cfg.gamma),
    )


def _cheat_reset(key, goal, B, ep: EnvParams):
    """Spawn B point masses within `spawn_radius` of the goal (cheat init), vel=0.
    Returns (pos (B,2), vel (B,2))."""
    kang, krad = jax.random.split(key)
    ang = jax.random.uniform(kang, (B,), minval=0.0, maxval=2.0 * jnp.pi)
    r = ep.spawn_radius * jnp.sqrt(jax.random.uniform(krad, (B,)))
    off = jnp.stack([r * jnp.cos(ang), r * jnp.sin(ang)], axis=-1)
    pos = jnp.clip(goal[None, :] + off, -ep.bound, ep.bound)
    vel = jnp.zeros((B, 2), dtype=jnp.float32)
    return pos, vel


def _env_step(pos, vel, action, goal, ep: EnvParams):
    """Double-integrator step, vectorized over the batch. Returns
    (pos', vel', reward (B,), success (B,) bool, done (B,) bool)."""
    a = jnp.clip(action, -1.0, 1.0)
    vel = (vel + ep.dt * ep.accel_scale * a) * ep.vel_damp
    speed = jnp.linalg.norm(vel, axis=-1, keepdims=True)
    vel = vel * jnp.minimum(1.0, ep.max_speed / jnp.maximum(speed, 1e-8))
    new_pos = pos + ep.dt * vel
    clipped = jnp.clip(new_pos, -ep.bound, ep.bound)
    hit = clipped != new_pos
    vel = jnp.where(hit, 0.0, vel)  # zero the velocity component into a wall
    dist = jnp.linalg.norm(clipped - goal[None, :], axis=-1)
    success = dist < ep.success_radius
    reward = -ep.dist_coef * dist - ep.step_penalty + success * ep.terminal_reward
    done = success
    return clipped, vel, reward.astype(jnp.float32), success, done


# ---------------------------------------------------------------------------
# FTN backbone (ParallelMLP bank + KWTA-masked readout) — pure-functional JAX.
# Param pytrees mirror ftn.py's ParallelMLP / FTNActor / FTNCritic init.
# ---------------------------------------------------------------------------


def _init_parallel(key, in_size, H, layers, inner_size):
    sizes = [in_size] + [inner_size] * layers + [1]
    Ws, bs = [], []
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        bound = (
            np.sqrt(6.0 / fan_in) if i < layers else np.sqrt(6.0 / (fan_in + fan_out))
        )
        bbound = 0.1 / np.sqrt(fan_in)
        key, kw, kb = jax.random.split(key, 3)
        Ws.append(
            jax.random.uniform(
                kw, (H, sizes[i], sizes[i + 1]), minval=-bound, maxval=bound
            ).astype(jnp.float32)
        )
        bs.append(
            jax.random.uniform(
                kb, (H, sizes[i + 1]), minval=-bbound, maxval=bbound
            ).astype(jnp.float32)
        )
    return {"W": Ws, "b": bs}, key


def _init_net(key, in_size, out_size, H, layers, inner_size, drop_p):
    pp, key = _init_parallel(key, in_size, H, layers, inner_size)
    key, klin = jax.random.split(key)
    lin_bound = 1.0 / np.sqrt(H)
    lin = jax.random.uniform(
        klin, (out_size, H), minval=-lin_bound, maxval=lin_bound
    ).astype(jnp.float32)
    return {"parallel": pp, "lin": lin, "drop_p": jnp.float32(drop_p)}, key


def _parallel_forward(pp, x):
    """x (B, D) -> (B, H). Each of the H sub-MLPs applies its own weights to x."""
    Ws, bs = pp["W"], pp["b"]
    H = Ws[0].shape[0]
    h = jnp.broadcast_to(x[:, None, :], (x.shape[0], H, x.shape[1]))
    n = len(Ws)
    for i in range(n):
        h = jnp.einsum("bhi,hij->bhj", h, Ws[i]) + bs[i]
        if i < n - 1:
            h = jax.nn.relu(h)
    return h[..., 0]  # (B, H), output_size==1 squeezed


def _maybe_dropout(z, key, p):
    if key is None:  # eval / rollout forward: dropout off
        return z
    keep = 1.0 - p
    mask = jax.random.bernoulli(key, keep, z.shape).astype(z.dtype)
    return z * mask / keep


def actor_forward(params, obs, mask, dkey=None):
    z = _parallel_forward(params["parallel"], obs)
    z = _maybe_dropout(z, dkey, params["drop_p"])
    return jnp.tanh((z * mask[None, :]) @ params["lin"].T)  # (B, act_dim)


def critic_forward(params, obs, act, mask, dkey=None):
    x = jnp.concatenate([obs, act], axis=-1)
    z = _parallel_forward(params["parallel"], x)
    z = _maybe_dropout(z, dkey, params["drop_p"])
    return ((z * mask[None, :]) @ params["lin"].T)[:, 0]  # (B,)


# ---------------------------------------------------------------------------
# KWTA + AnnealedSTE mask search (JAX)
# ---------------------------------------------------------------------------


def kwta(x, k):
    """Binary top-k mask over a 1-D tensor (exactly k ones)."""
    idx = jax.lax.top_k(x, k)[1]
    return jnp.zeros_like(x).at[idx].set(1.0)


def _ste_search(loss_fn, m_param0, H, k, n_steps, lr, key):
    """AnnealedSTE soft-mask search (port of ftn.anneal_ste_mask).

    `loss_fn(soft_mask, dropout_key) -> scalar`. `m_param0` is the initial logit
    vector (already noise-scaled). Returns (best_hard_mask, final_logits,
    first_loss, lowest_loss)."""
    denom = max(1.0, n_steps / 4.0)

    def body(carry, step):
        m_param, best, lowest, first, key = carry
        key, nk, dk = jax.random.split(key, 3)
        std = jnp.std(m_param)
        scale = jnp.where(std > 1e-6, std, 0.1)
        tau = scale * jnp.exp(-step / denom)
        noise = jax.random.normal(nk, (H,)) * tau

        def g(mp):
            hard = kwta(jax.lax.stop_gradient(mp + noise), k)
            mask_ste = hard + mp - jax.lax.stop_gradient(mp)  # straight-through
            return loss_fn(mask_ste, dk), hard

        (loss, hard), grad = jax.value_and_grad(g, has_aux=True)(m_param)
        m_param = m_param - lr * grad
        first = jnp.where(step == 0, loss, first)
        improved = loss < lowest
        best = jnp.where(improved, hard, best)
        lowest = jnp.where(improved, loss, lowest)
        return (m_param, best, lowest, first, key), None

    carry0 = (m_param0, jnp.zeros(H), jnp.float32(jnp.inf), jnp.float32(0.0), key)
    (m_param, best, lowest, first, _), _ = jax.lax.scan(
        body, carry0, jnp.arange(n_steps)
    )
    return best, m_param, first, lowest


@partial(
    jax.jit,
    static_argnames=("objective", "H", "k", "n_steps", "warm"),
)
def run_reconfig(
    critic_params,
    actor_params,
    actor_t,
    critic_t,
    support,  # (obs, act, rew, next_obs, done, returns)
    m_a_cur,
    init_a,
    init_c,
    key,
    *,
    objective,
    H,
    k,
    n_steps,
    lr,
    noise_scale,
    gamma,
    warm,
):
    """Critic mask reconfig via `objective`, then actor mask via −Q under the new
    critic mask (AnnealedSTE). Cold-starts from noise unless `warm` (then from the
    passed logits). Returns (m_a, m_c, m_a_logits, m_c_logits, stats)."""
    obs, act, rew, next_obs, done, returns = support
    key, kcold_c, kcold_a, ks_c, ks_a = jax.random.split(key, 5)

    def critic_loss(soft_m, dk):
        q = critic_forward(critic_params, obs, act, soft_m, dk)
        if objective == "td":
            a_next = actor_forward(actor_t, next_obs, m_a_cur, dk)
            q_next = critic_forward(critic_t, next_obs, a_next, soft_m, dk)
            target = jax.lax.stop_gradient(rew + gamma * (1.0 - done) * q_next)
            return jnp.mean((q - target) ** 2)
        elif objective == "mse_returns":
            return jnp.mean((q - returns) ** 2)
        elif objective == "cos_returns":
            return -_cosine(q, returns)
        elif objective == "r_returns":
            return -_cosine(q - q.mean(), returns - returns.mean())
        else:
            raise ValueError(f"unknown reconfig_objective={objective!r}")

    base_c = init_c if warm else jax.random.normal(kcold_c, (H,))
    m_c, lc, ci, cf = _ste_search(
        critic_loss, noise_scale * base_c, H, k, n_steps, lr, ks_c
    )

    def actor_loss(soft_m, dk):
        a = actor_forward(actor_params, obs, soft_m, dk)
        return -jnp.mean(critic_forward(critic_params, obs, a, m_c, dk))

    base_a = init_a if warm else jax.random.normal(kcold_a, (H,))
    m_a, la, ai, af = _ste_search(
        actor_loss, noise_scale * base_a, H, k, n_steps, lr, ks_a
    )

    stats = {"critic_init": ci, "critic_final": cf, "actor_init": ai, "actor_final": af}
    return m_a, m_c, la, lc, stats


def _cosine(a, b):
    return jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-8)


# ---------------------------------------------------------------------------
# Vectorized rollouts (lax.scan over time, leading batch dim = parallel episodes)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnames=("B", "max_steps", "use_actor"))
def rollout(
    key, goal, actor_params, mask_a, noise_std, ep: EnvParams, B, max_steps, use_actor
):
    """Run B parallel episodes in lockstep. If `use_actor`, act with the (greedy +
    Gaussian-noise) actor under `mask_a`; else uniform-random actions. Returns a
    dict of arrays: obs/act/next_obs/rew/done/valid shaped (max_steps, B, ...),
    plus per-episode success (B,) and length (B,). `valid` marks steps before the
    episode terminated (post-termination steps are excluded downstream)."""
    kreset, kscan = jax.random.split(key)
    pos, vel = _cheat_reset(kreset, goal, B, ep)

    def step(carry, key_t):
        pos, vel, done_prev = carry
        obs = jnp.concatenate([pos, vel], axis=-1)
        ka, kn = jax.random.split(key_t)
        if use_actor:
            a = actor_forward(actor_params, obs, mask_a)  # dropout off
            a = a + noise_std * jax.random.normal(kn, a.shape)
            a = jnp.clip(a, -1.0, 1.0)
        else:
            a = jax.random.uniform(ka, (B, ACT_DIM), minval=-1.0, maxval=1.0)
        npos, nvel, rew, succ, done_step = _env_step(pos, vel, a, goal, ep)
        next_obs = jnp.concatenate([npos, nvel], axis=-1)
        valid = (~done_prev).astype(jnp.float32)
        done_now = done_prev | done_step
        out = {
            "obs": obs,
            "act": a,
            "rew": rew,
            "next_obs": next_obs,
            "done": done_step.astype(jnp.float32),
            "valid": valid,
            "succ": (succ & ~done_prev),
        }
        return (npos, nvel, done_now), out

    keys = jax.random.split(kscan, max_steps)
    (_, _, _), outs = jax.lax.scan(step, (pos, vel, jnp.zeros(B, bool)), keys)
    success = (outs["succ"].sum(axis=0) > 0).astype(jnp.int32)  # (B,)
    length = outs["valid"].sum(axis=0).astype(jnp.int32)  # (B,)
    return {**outs, "success": success, "length": length}


@partial(jax.jit, static_argnames=("B", "n_steps", "max_steps"))
def collect_rollout(
    key,
    goal,
    actor_params,
    mask_a,
    noise_std,
    ep: EnvParams,
    B,
    n_steps,
    max_steps,
    pos0,
    vel0,
    len0,
):
    """Step-based collection for the exploitation loop: run B continuing envs for
    `n_steps` under the noisy actor, in ONE jitted lax.scan. Envs auto-reset on
    success OR after `max_steps` (timeout) — the timeout matters: without it an
    env that drives into a wall and stalls would never reset and stop producing
    rewarded data. `pos0/vel0/len0` carry the env state + per-env episode length
    across iterations so episodes continue between bursts. Returns (transitions
    dict shaped (n_steps, B, ...), final pos, final vel, final ep-len). `done`
    (= success OR timeout, mirroring train.py's `term or trunc`) cuts the TD
    bootstrap; `succ` marks true goal reaches."""

    def step(carry, kt):
        pos, vel, ep_len = carry
        obs = jnp.concatenate([pos, vel], axis=-1)
        kn, kr = jax.random.split(kt)
        a = actor_forward(actor_params, obs, mask_a)  # dropout off
        a = jnp.clip(a + noise_std * jax.random.normal(kn, a.shape), -1.0, 1.0)
        npos, nvel, rew, succ, term = _env_step(pos, vel, a, goal, ep)
        next_obs = jnp.concatenate([npos, nvel], axis=-1)
        ep_len = ep_len + 1
        reset = term | (ep_len >= max_steps)
        rpos, rvel = _cheat_reset(kr, goal, B, ep)  # auto-reset spawn (cheat init)
        pos2 = jnp.where(reset[:, None], rpos, npos)
        vel2 = jnp.where(reset[:, None], rvel, nvel)
        ep_len2 = jnp.where(reset, 0, ep_len)
        out = {
            "obs": obs,
            "act": a,
            "rew": rew,
            "next_obs": next_obs,
            "done": reset.astype(jnp.float32),
            "succ": succ,
        }
        return (pos2, vel2, ep_len2), out

    keys = jax.random.split(key, n_steps)
    (pos, vel, ep_len), outs = jax.lax.scan(step, (pos0, vel0, len0), keys)
    return outs, pos, vel, ep_len


def _support_from_burst(burst_np: dict, gamma: float, n: int):
    """Mid-phase reconfig support from the most recent collect burst (shaped
    (T, B, ...)). MC returns are computed per env-column (so the auto-reset
    episode boundaries inside the burst are respected via `done`), then the
    transitions are flattened and resampled to exactly `n` rows."""
    rew = burst_np["rew"]  # (T, B)
    done = burst_np["done"]  # (T, B)
    T, B = rew.shape
    ret = np.zeros((T, B), dtype=np.float32)
    g = np.zeros(B, dtype=np.float32)
    for t in reversed(range(T)):
        g = rew[t] + gamma * (1.0 - done[t]) * g
        ret[t] = g

    def flat(a):
        return a.reshape(-1, a.shape[-1]) if a.ndim == 3 else a.reshape(-1)

    obs, act = flat(burst_np["obs"]), flat(burst_np["act"])
    rewf, nextf, donef, retf = (
        flat(rew),
        flat(burst_np["next_obs"]),
        flat(done),
        ret.reshape(-1),
    )
    N = obs.shape[0]
    idx = np.random.randint(0, N, size=n) if N > 0 else np.zeros(n, dtype=int)
    return (
        jnp.asarray(obs[idx]),
        jnp.asarray(act[idx]),
        jnp.asarray(rewf[idx]),
        jnp.asarray(nextf[idx]),
        jnp.asarray(donef[idx]),
        jnp.asarray(retf[idx]),
    )


# ---------------------------------------------------------------------------
# Host-side helpers: MC returns, support assembly, replay
# ---------------------------------------------------------------------------


def compute_mc_returns(
    rewards: np.ndarray, dones: np.ndarray, gamma: float
) -> np.ndarray:
    T = len(rewards)
    ret = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            g = 0.0
        g = float(rewards[t]) + gamma * g
        ret[t] = g
    return ret


def _episode_transitions(roll_np: dict, b: int) -> dict:
    """Pull episode b's valid transitions (the prefix before termination)."""
    L = int(roll_np["length"][b])
    L = max(L, 1)
    sl = slice(0, L)
    return {
        "obs": roll_np["obs"][sl, b],
        "act": roll_np["act"][sl, b],
        "rew": roll_np["rew"][sl, b],
        "next_obs": roll_np["next_obs"][sl, b],
        "done": roll_np["done"][sl, b],
    }


def _support_from_episodes(episodes: list[dict], gamma: float, n: int):
    """Concatenate transitions, MC returns on the concat (respects boundaries via
    `done`), then resample to exactly `n` rows (with replacement) so the reconfig
    search has a fixed shape. Returns a tuple of jnp arrays
    (obs, act, rew, next_obs, done, returns)."""
    obs = np.concatenate([e["obs"] for e in episodes], axis=0)
    act = np.concatenate([e["act"] for e in episodes], axis=0)
    rew = np.concatenate([e["rew"] for e in episodes], axis=0)
    next_obs = np.concatenate([e["next_obs"] for e in episodes], axis=0)
    done = np.concatenate([e["done"] for e in episodes], axis=0)
    returns = compute_mc_returns(rew, done, gamma)
    N = obs.shape[0]
    idx = np.random.randint(0, N, size=n) if N > 0 else np.zeros(n, dtype=int)
    return (
        jnp.asarray(obs[idx]),
        jnp.asarray(act[idx]),
        jnp.asarray(rew[idx]),
        jnp.asarray(next_obs[idx]),
        jnp.asarray(done[idx]),
        jnp.asarray(returns[idx]),
    )


class ReplayDDPG:
    """Device-resident current-segment ring buffer + per-chunk archive. The ring
    holds the transitions collected since the last reconfig (cleared each commit);
    the jitted train burst samples from it directly on the GPU (no host copies).
    Each archived chunk is tagged with the (m_a, m_c) its transitions were
    collected under and stored padded to `cap` rows so the jitted update sees a
    fixed shape. A chunk is recalled if its actor OR critic mask overlaps the
    current masks in (rho_low, rho_high]. Mirrors MaskedReplayBufferDDPG."""

    def __init__(self, cap: int):
        self.cap = cap
        self.obs = jnp.zeros((cap, OBS_DIM))
        self.act = jnp.zeros((cap, ACT_DIM))
        self.rew = jnp.zeros((cap,))
        self.next_obs = jnp.zeros((cap, OBS_DIM))
        self.done = jnp.zeros((cap,))
        self.ptr = 0
        self.size = 0
        self.past: list[dict] = []

    def add(self, obs, act, rew, next_obs, done):
        """Bulk-add a flat batch of transitions (device arrays, shape (L, ...))."""
        L = obs.shape[0]
        idx = jnp.asarray((self.ptr + np.arange(L)) % self.cap)
        self.obs = self.obs.at[idx].set(obs)
        self.act = self.act.at[idx].set(act)
        self.rew = self.rew.at[idx].set(rew)
        self.next_obs = self.next_obs.at[idx].set(next_obs)
        self.done = self.done.at[idx].set(done)
        self.ptr = int((self.ptr + L) % self.cap)
        self.size = int(min(self.size + L, self.cap))

    def clear_current(self):
        self.ptr = 0
        self.size = 0

    def current_arrays(self):
        """The full ring arrays (sampling clips indices to `size`)."""
        return {
            "obs": self.obs,
            "act": self.act,
            "rew": self.rew,
            "next_obs": self.next_obs,
            "done": self.done,
        }, self.size

    def archive_chunk(self, m_a, m_c):
        if self.size == 0:
            return
        n = self.size  # ring is cleared each reconfig interval, so [:size] is the chunk
        self.past.append(
            {
                "obs": self.obs,
                "act": self.act,
                "rew": self.rew,
                "next_obs": self.next_obs,
                "done": self.done,
                "len": float(n),
                "m_a": jnp.asarray(m_a),
                "m_c": jnp.asarray(m_c),
            }
        )

    def select_chunks(self, m_a_cur, m_c_cur, max_chunks, rho_low, rho_high):
        """Host-side overlap filter → up to max_chunks qualifying chunk indices."""
        if not self.past:
            return []
        ma = np.asarray(m_a_cur)
        mc = np.asarray(m_c_cur)
        denom_a = max(ma.sum(), 1.0)
        denom_c = max(mc.sum(), 1.0)
        cand = []
        for i, ph in enumerate(self.past):
            rho_a = float((np.asarray(ph["m_a"]) * ma).sum() / denom_a)
            rho_c = float((np.asarray(ph["m_c"]) * mc).sum() / denom_c)
            if (rho_low < rho_a <= rho_high) or (rho_low < rho_c <= rho_high):
                cand.append(i)
        if len(cand) > max_chunks:
            cand = list(np.random.permutation(cand)[:max_chunks])
        return cand

    def stacked_chunks(self, indices, C, H):
        """Stack C chunks (padded to replay_max_chunks with zero-weight dummies)
        into batched device arrays for the jitted update."""
        pad = self.cap
        keys = ("obs", "act", "rew", "next_obs", "done")
        dims = {
            "obs": OBS_DIM,
            "act": ACT_DIM,
            "rew": None,
            "next_obs": OBS_DIM,
            "done": None,
        }
        stacks = {k: [] for k in keys}
        lens, mas, mcs, valid = [], [], [], []
        for c in range(C):
            if c < len(indices):
                ph = self.past[indices[c]]
                for k in keys:
                    stacks[k].append(ph[k])
                lens.append(ph["len"])
                mas.append(ph["m_a"])
                mcs.append(ph["m_c"])
                valid.append(1.0)
            else:  # dummy padding chunk
                for k in keys:
                    shape = (pad,) if dims[k] is None else (pad, dims[k])
                    stacks[k].append(jnp.zeros(shape, jnp.float32))
                lens.append(1.0)
                mas.append(jnp.zeros(H, jnp.float32))
                mcs.append(jnp.zeros(H, jnp.float32))
                valid.append(0.0)
        return {
            **{k: jnp.stack(stacks[k]) for k in keys},
            "len": jnp.asarray(lens, jnp.float32),
            "m_a": jnp.stack(mas),
            "m_c": jnp.stack(mcs),
            "valid": jnp.asarray(valid, jnp.float32),
        }


def disjoint_task_mask(ti: int, H: int, k: int):
    m = np.zeros(H, dtype=np.float32)
    m[ti * k : (ti + 1) * k] = 1.0
    return jnp.asarray(m)


# ---------------------------------------------------------------------------
# Joint DDPG loss terms (pure functions; chunk terms vmapped) + update step
# ---------------------------------------------------------------------------


def _td_term(
    critic_params,
    actor_t,
    critic_t,
    obs,
    act,
    rew,
    next_obs,
    done,
    m_a,
    m_c,
    gamma,
    key,
):
    k1, k2, k3 = jax.random.split(key, 3)
    a_next = actor_forward(actor_t, next_obs, m_a, k1)
    q_next = critic_forward(critic_t, next_obs, a_next, m_c, k2)
    target = jax.lax.stop_gradient(rew + gamma * (1.0 - done) * q_next)
    q = critic_forward(critic_params, obs, act, m_c, k3)
    return jnp.mean((q - target) ** 2)


def _actor_term(actor_params, critic_params, obs, m_a, m_c, key):
    k1, k2 = jax.random.split(key)
    a = actor_forward(actor_params, obs, m_a, k1)
    return -jnp.mean(critic_forward(critic_params, obs, a, m_c, k2))


def _critic_total(critic_params, actor_t, critic_t, cur, m_a, m_c, ch, gamma, key):
    k0, kc = jax.random.split(key)
    cur_loss = _td_term(critic_params, actor_t, critic_t, *cur, m_a, m_c, gamma, k0)
    C = ch["valid"].shape[0]
    keys = jax.random.split(kc, C)
    ch_losses = jax.vmap(
        _td_term, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, 0, None, 0)
    )(
        critic_params,
        actor_t,
        critic_t,
        ch["obs"],
        ch["act"],
        ch["rew"],
        ch["next_obs"],
        ch["done"],
        ch["m_a"],
        ch["m_c"],
        gamma,
        keys,
    )
    return cur_loss + jnp.sum(ch["valid"] * ch_losses)


def _actor_total(actor_params, critic_params, cur_obs, m_a, m_c, ch, key):
    k0, kc = jax.random.split(key)
    cur_loss = _actor_term(actor_params, critic_params, cur_obs, m_a, m_c, k0)
    C = ch["valid"].shape[0]
    keys = jax.random.split(kc, C)
    ch_losses = jax.vmap(_actor_term, in_axes=(None, None, 0, 0, 0, 0))(
        actor_params, critic_params, ch["obs"], ch["m_a"], ch["m_c"], keys
    )
    return cur_loss + jnp.sum(ch["valid"] * ch_losses)


# ---------------------------------------------------------------------------
# Exploration + unified eval (host orchestration over jitted rollouts)
# ---------------------------------------------------------------------------


def _gather_rewarded(key, goal, agent, cfg, env_p, use_actor, mask_a, noise):
    """Random (or prev-mask actor) rollouts until n_successful_explore rewarded
    episodes or explore_episodes total. Returns (rewarded_eps, n_explore_eps,
    n_steps)."""
    B = cfg.n_parallel_explore
    rewarded, total_eps, total_steps = [], 0, 0
    while total_eps < cfg.explore_episodes and len(rewarded) < cfg.n_successful_explore:
        key, k = jax.random.split(key)
        roll = rollout(
            k,
            goal,
            agent["actor"],
            mask_a,
            jnp.float32(noise),
            env_p,
            B,
            cfg.max_episode_steps,
            use_actor,
        )
        roll_np = jax.tree_util.tree_map(np.asarray, roll)
        total_eps += B
        total_steps += int(roll_np["length"].sum())
        for b in np.nonzero(roll_np["success"])[0]:
            rewarded.append(_episode_transitions(roll_np, int(b)))
            if len(rewarded) >= cfg.n_successful_explore:
                break
    return rewarded, total_eps, total_steps, key


def _eval_task(key, goal, agent, cfg, env_p, gamma_rc, recall_m_a):
    """One eval cell's success rate. Recall baselines (recall_m_a given or no_mask)
    skip explore+reconfig and just run greedy rollouts under the recall mask;
    otherwise gather a rewarded support, then per eval episode cold-start reconfig
    + greedy rollout."""
    if recall_m_a is not None or cfg.no_mask:
        m = recall_m_a if recall_m_a is not None else agent["m_a"]
        key, k = jax.random.split(key)
        roll = rollout(
            k,
            goal,
            agent["actor"],
            m,
            jnp.float32(0.0),
            env_p,
            cfg.eval_episodes,
            cfg.max_episode_steps,
            True,
        )
        return float(np.asarray(roll["success"]).mean()), key

    rewarded, _, _, key = _gather_rewarded(
        key,
        goal,
        agent,
        cfg,
        env_p,
        use_actor=(cfg.warmup_policy != "random"),
        mask_a=agent["m_a"],
        noise=cfg.noise_eval_explore,
    )
    if not rewarded:
        return 0.0, key
    support = _support_from_episodes(rewarded, gamma_rc, cfg.reconfig_n)

    H, k_ = cfg.ftn_hidden_size, cfg.ftn_k
    n_succ = 0
    for _ in range(cfg.eval_episodes):
        key, krc, kro = jax.random.split(key, 3)
        m_a_e, _, _, _, _ = run_reconfig(
            agent["critic"],
            agent["actor"],
            agent["actor_t"],
            agent["critic_t"],
            support,
            agent["m_a"],
            jnp.zeros(H),
            jnp.zeros(H),
            krc,
            objective=cfg.reconfig_objective,
            H=H,
            k=k_,
            n_steps=cfg.reconfig_steps,
            lr=cfg.reconfig_lr,
            noise_scale=cfg.mask_init_noise_scale,
            gamma=gamma_rc,
            warm=False,  # eval always cold-starts
        )
        roll = rollout(
            kro,
            goal,
            agent["actor"],
            m_a_e,
            jnp.float32(0.0),
            env_p,
            1,
            cfg.max_episode_steps,
            True,
        )
        n_succ += int(np.asarray(roll["success"])[0])
    return n_succ / cfg.eval_episodes, key


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(cfg: DDPGConfig):
    np.random.seed(cfg.seed)
    key = jax.random.PRNGKey(cfg.seed)

    env_p = env_params_from_cfg(cfg)
    goals = [jnp.asarray(g, jnp.float32) for g in cfg.goals]
    N = len(goals)
    H, k = cfg.ftn_hidden_size, cfg.ftn_k
    gamma_rc = cfg.reconfig_gamma if cfg.reconfig_gamma is not None else cfg.gamma

    if cfg.fixed_mask and N * k > H:
        raise ValueError(f"fixed_mask needs N*k<=H, got {N}*{k}>{H}")

    no_reconfig = cfg.fixed_mask or cfg.no_mask
    baseline_label = (
        "no-mask" if cfg.no_mask else ("fixed-mask" if cfg.fixed_mask else "")
    )
    recall_eval = cfg.fixed_mask or cfg.stored_mask

    # --- networks ---
    key, ka, kc = jax.random.split(key, 3)
    actor, _ = _init_net(
        ka, OBS_DIM, ACT_DIM, H, cfg.ftn_layers, cfg.ftn_inner_size, cfg.ftn_dropout_p
    )
    critic, _ = _init_net(
        kc,
        OBS_DIM + ACT_DIM,
        1,
        H,
        cfg.ftn_layers,
        cfg.ftn_inner_size,
        cfg.ftn_dropout_p,
    )
    actor_t = jax.tree_util.tree_map(jnp.array, actor)
    critic_t = jax.tree_util.tree_map(jnp.array, critic)

    actor_opt = optax.adam(cfg.actor_lr)
    critic_opt = optax.adam(cfg.critic_lr)

    # --- initial masks ---
    key, kma, kmc = jax.random.split(key, 3)
    if cfg.no_mask:
        m_a = jnp.ones(H)
        m_c = jnp.ones(H)
    elif cfg.fixed_mask:
        m_a = disjoint_task_mask(0, H, k)
        m_c = disjoint_task_mask(0, H, k)
    else:
        m_a = kwta(jax.random.normal(kma, (H,)), k)
        m_c = kwta(jax.random.normal(kmc, (H,)), k)

    agent = {
        "actor": actor,
        "critic": critic,
        "actor_t": actor_t,
        "critic_t": critic_t,
        "aopt": actor_opt.init(actor),
        "copt": critic_opt.init(critic),
        "m_a": m_a,
        "m_c": m_c,
        "m_a_logits": None,
        "m_c_logits": None,
    }
    stored_m_a: dict[int, jnp.ndarray] = {}
    stored_m_c: dict[int, jnp.ndarray] = {}

    # Ring capacity holds (more than) one reconfig interval of transitions, so
    # the current segment never wraps before it is archived + cleared.
    interval_iters = cfg.reconfig_every_iters or cfg.n_iters_per_task
    cap = (interval_iters + 1) * cfg.collect_steps * cfg.n_parallel_train
    replay = ReplayDDPG(cap)
    C = cfg.replay_max_chunks

    def _gather(arr, ix):
        return jax.vmap(lambda a, i: a[i])(arr, ix)

    # --- jitted train burst: M=train_steps joint DDPG updates in one scan,
    # sampling the current ring + recalled chunks on device each step. ---
    @jax.jit
    def train_burst(state, m_a, m_c, cur, cur_size, ch, key):
        def step(carry, kt):
            st = carry
            kcur, kch, kup = jax.random.split(kt, 3)
            # current batch (sample with replacement from the filled prefix)
            idx = (jax.random.uniform(kcur, (cfg.batch_size,)) * cur_size).astype(
                jnp.int32
            )
            idx = jnp.clip(idx, 0, cur["obs"].shape[0] - 1)
            cur_b = (
                cur["obs"][idx],
                cur["act"][idx],
                cur["rew"][idx],
                cur["next_obs"][idx],
                cur["done"][idx],
            )
            # chunk tuples (per-chunk sampling within each chunk's true length)
            cidx = (
                jax.random.uniform(kch, (C, cfg.replay_max_tuples)) * ch["len"][:, None]
            ).astype(jnp.int32)
            cidx = jnp.clip(cidx, 0, ch["obs"].shape[1] - 1)
            gather = _gather
            ch_s = {
                "obs": gather(ch["obs"], cidx),
                "act": gather(ch["act"], cidx),
                "rew": gather(ch["rew"], cidx),
                "next_obs": gather(ch["next_obs"], cidx),
                "done": gather(ch["done"], cidx),
                "m_a": ch["m_a"],
                "m_c": ch["m_c"],
                "valid": ch["valid"],
            }
            kcl, kal = jax.random.split(kup)
            gC = jax.grad(_critic_total)(
                st["critic"],
                st["actor_t"],
                st["critic_t"],
                cur_b,
                m_a,
                m_c,
                ch_s,
                cfg.gamma,
                kcl,
            )
            cu, copt = critic_opt.update(gC, st["copt"], st["critic"])
            critic2 = optax.apply_updates(st["critic"], cu)
            gA = jax.grad(_actor_total)(
                st["actor"], critic2, cur_b[0], m_a, m_c, ch_s, kal
            )
            au, aopt = actor_opt.update(gA, st["aopt"], st["actor"])
            actor2 = optax.apply_updates(st["actor"], au)
            tau = cfg.tau
            actor_t2 = jax.tree_util.tree_map(
                lambda t, p: (1 - tau) * t + tau * p, st["actor_t"], actor2
            )
            critic_t2 = jax.tree_util.tree_map(
                lambda t, p: (1 - tau) * t + tau * p, st["critic_t"], critic2
            )
            return {
                **st,
                "actor": actor2,
                "critic": critic2,
                "actor_t": actor_t2,
                "critic_t": critic_t2,
                "aopt": aopt,
                "copt": copt,
            }, None

        keys = jax.random.split(key, cfg.train_steps)
        state, _ = jax.lax.scan(step, state, keys)
        return state

    def reconfig(support, warm, key):
        """Run a reconfig and return (m_a, m_c, logits_a, logits_c, stats)."""
        init_a = (
            agent["m_a_logits"]
            if (warm and agent["m_a_logits"] is not None)
            else jnp.zeros(H)
        )
        init_c = (
            agent["m_c_logits"]
            if (warm and agent["m_c_logits"] is not None)
            else jnp.zeros(H)
        )
        do_warm = warm and agent["m_a_logits"] is not None
        return run_reconfig(
            agent["critic"],
            agent["actor"],
            agent["actor_t"],
            agent["critic_t"],
            support,
            agent["m_a"],
            init_a,
            init_c,
            key,
            objective=cfg.reconfig_objective,
            H=H,
            k=k,
            n_steps=cfg.reconfig_steps,
            lr=cfg.reconfig_lr,
            noise_scale=cfg.mask_init_noise_scale,
            gamma=gamma_rc,
            warm=do_warm,
        )

    matrix = np.full((N, N), np.nan, dtype=np.float32)
    t_start = time.time()

    for ti, goal in enumerate(goals):
        print(f"\n=== Phase {ti}/{N - 1}: goal {tuple(np.asarray(goal).tolist())} ===")
        replay.clear_current()
        # Per-task optimizer-state reset (paper protocol): stale Adam momentum
        # must not keep moving the previous task's now-masked neurons.
        agent["aopt"] = actor_opt.init(agent["actor"])
        agent["copt"] = critic_opt.init(agent["critic"])

        if cfg.fixed_mask:
            agent["m_a"] = disjoint_task_mask(ti, H, k)
            agent["m_c"] = disjoint_task_mask(ti, H, k)
            print(f"  fixed-mask: neurons [{ti * k}, {(ti + 1) * k})")

        # 1) Exploration -----------------------------------------------------
        use_actor = ti > 0 and cfg.warmup_policy != "random"
        explore_eps, n_eps, n_steps_exp, key = _gather_rewarded(
            key,
            goal,
            agent,
            cfg,
            env_p,
            use_actor,
            mask_a=agent["m_a"],
            noise=cfg.noise_std,
        )
        print(
            f"  exploration ({'prev_mask' if use_actor else 'random'}): "
            f"{n_eps} eps, {n_steps_exp} steps, {len(explore_eps)} successful"
        )

        # 2) Boundary mask reconfig -----------------------------------------
        if not no_reconfig:
            if explore_eps:
                support = _support_from_episodes(explore_eps, gamma_rc, cfg.reconfig_n)
                key, krc = jax.random.split(key)
                m_a, m_c, la, lc, stats = reconfig(support, cfg.mask_warm_start, krc)
                agent.update(m_a=m_a, m_c=m_c, m_a_logits=la, m_c_logits=lc)
                print(
                    f"  reconfig: critic {float(stats['critic_init']):.3f}->"
                    f"{float(stats['critic_final']):.3f}  actor -Q "
                    f"{float(stats['actor_init']):.3f}->{float(stats['actor_final']):.3f}"
                )
            else:
                print("  WARN: no reward found during exploration; keeping prior masks")
        else:
            print(f"  {baseline_label} baseline: skipping reconfig")
        stored_m_a[ti] = agent["m_a"]
        stored_m_c[ti] = agent["m_c"]

        # 3) Exploitation: decoupled collect (N steps) → train (M steps) loop -
        B = cfg.n_parallel_train
        key, kre = jax.random.split(key)
        pos, vel = _cheat_reset(kre, goal, B, env_p)  # continuing env state
        ep_len = jnp.zeros(B)
        last_burst = None
        for it in range(cfg.n_iters_per_task):
            # --- COLLECT: N env steps across B continuing envs, one jitted scan.
            key, kc = jax.random.split(key)
            burst, pos, vel, ep_len = collect_rollout(
                kc,
                goal,
                agent["actor"],
                agent["m_a"],
                jnp.float32(cfg.noise_std),
                env_p,
                B,
                cfg.collect_steps,
                cfg.max_episode_steps,
                pos,
                vel,
                ep_len,
            )
            burst_np = jax.tree_util.tree_map(np.asarray, burst)
            last_burst = burst_np
            replay.add(
                burst["obs"].reshape(-1, OBS_DIM),
                burst["act"].reshape(-1, ACT_DIM),
                burst["rew"].reshape(-1),
                burst["next_obs"].reshape(-1, OBS_DIM),
                burst["done"].reshape(-1),
            )

            # --- TRAIN: M gradient updates, one jitted scan over the ring + chunks.
            cur, cur_size = replay.current_arrays()
            cand = (
                []
                if no_reconfig
                else replay.select_chunks(
                    agent["m_a"], agent["m_c"], C, cfg.rho_low, cfg.rho_high
                )
            )
            ch = replay.stacked_chunks(cand, C, H)
            state = {
                kk: agent[kk]
                for kk in ("actor", "critic", "actor_t", "critic_t", "aopt", "copt")
            }
            key, ku = jax.random.split(key)
            state = train_burst(
                state, agent["m_a"], agent["m_c"], cur, jnp.float32(cur_size), ch, ku
            )
            for kk in state:
                agent[kk] = state[kk]

            # --- Mid-phase reconfig (skipped under baselines / when disabled).
            if (
                not no_reconfig
                and cfg.reconfig_every_iters > 0
                and (it + 1) % cfg.reconfig_every_iters == 0
            ):
                support = _support_from_burst(last_burst, gamma_rc, cfg.reconfig_n)
                replay.archive_chunk(
                    agent["m_a"], agent["m_c"]
                )  # tag with outgoing masks
                replay.clear_current()
                key, krc = jax.random.split(key)
                m_a, m_c, la, lc, _ = reconfig(support, cfg.mask_warm_start, krc)
                agent.update(m_a=m_a, m_c=m_c, m_a_logits=la, m_c_logits=lc)

            if (it + 1) % cfg.log_every == 0 or it == cfg.n_iters_per_task - 1:
                nd = max(int(burst_np["done"].sum()), 1)
                ns = int(burst_np["succ"].sum())
                mr = float(burst_np["rew"].mean())
                print(
                    f"  iter {it + 1:>4d}  step_r={mr:6.3f}  "
                    f"succ/done={ns}/{nd}={ns / nd:.2f}  ({time.time() - t_start:.0f}s)"
                )

        # 4) Archive the final chunk ----------------------------------------
        stored_m_a[ti] = agent["m_a"]
        stored_m_c[ti] = agent["m_c"]
        if not no_reconfig:
            replay.archive_chunk(agent["m_a"], agent["m_c"])

        # 5) Eval all tasks --------------------------------------------------
        print(f"  --- Eval after phase {ti} ---")
        for tj in range(N):
            tag = "<-" if tj == ti else "  "
            if recall_eval and tj > ti:
                matrix[ti, tj] = np.nan
                print(
                    f"    [{tag}] goal {tuple(np.asarray(goals[tj]).tolist())} sr=  — (skipped)"
                )
                continue
            recall = stored_m_a[tj] if recall_eval else None
            sr, key = _eval_task(key, goals[tj], agent, cfg, env_p, gamma_rc, recall)
            matrix[ti, tj] = sr
            print(
                f"    [{tag}] goal {tuple(np.asarray(goals[tj]).tolist())} sr={sr:.2f}"
            )

        np.savez(
            cfg.output_path,
            matrix=matrix,
            goals=np.array(cfg.goals),
            phases_done=ti + 1,
        )

    print("\n=== Eval matrix (rows: training phase, cols: eval task) ===")
    for ti in range(N):
        row = " ".join(
            "  — " if np.isnan(matrix[ti, tj]) else f"{matrix[ti, tj]:.2f}"
            for tj in range(N)
        )
        print(f"  phase {ti}: {row}")

    plot_matrix(
        matrix,
        [str(tuple(g)) for g in cfg.goals],
        cfg.output_path.replace(".npz", ".png"),
    )
    with open(cfg.output_path.replace(".npz", ".config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    return matrix


def plot_matrix(matrix: np.ndarray, names, out_path: str):
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n + 2), max(5, 0.6 * n + 1.5)))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel("Eval task (goal)")
    ax.set_ylabel("Trained through task")
    ax.set_title("Sequential point-mass FTN-DDPG: success-rate matrix")
    for i in range(n):
        for j in range(n):
            if np.isnan(matrix[i, j]):
                continue
            color = "white" if matrix[i, j] < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="success rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


# ---------------------------------------------------------------------------
# Cheat-init calibration: random-action rollouts only, no learning. Tune
# spawn_radius / dynamics so random actions hit success at ~30-70% within
# max_episode_steps (the analog of train.py's calibrate_cheats).
# ---------------------------------------------------------------------------


def calibrate(cfg: DDPGConfig):
    key = jax.random.PRNGKey(cfg.seed)
    env_p = env_params_from_cfg(cfg)
    B = cfg.n_parallel_explore
    print(
        f"\nCalibration: {B} random-action eps per goal, max_steps={cfg.max_episode_steps}"
    )
    print(f"{'goal':16s} {'sr':>5s}  {'mean_len':>8s}")
    for g in cfg.goals:
        key, k = jax.random.split(key)
        roll = rollout(
            k,
            jnp.asarray(g, jnp.float32),
            None,
            jnp.zeros(cfg.ftn_hidden_size),
            jnp.float32(0.0),
            env_p,
            B,
            cfg.max_episode_steps,
            False,
        )
        sr = float(np.asarray(roll["success"]).mean())
        ml = float(np.asarray(roll["length"]).mean())
        print(f"{str(g):16s} {sr:5.2f}  {ml:8.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke-test config")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Random-action rollouts to tune the cheat init; no learning.",
    )
    parser.add_argument(
        "--fixed-mask",
        action="store_true",
        help="Disjoint-subset baseline (needs N*k<=H). Output suffixed _fixed.",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="All-ones-mask baseline (k=H), frozen. Output suffixed _nomask.",
    )
    parser.add_argument(
        "--stored-mask",
        action="store_true",
        help="FTN training; eval recalls each task's last mask. Suffixed _stored.",
    )
    parser.add_argument("--name", default="")
    parser.add_argument("--seed", default="0")
    args = parser.parse_args()

    if args.fixed_mask + args.no_mask + args.stored_mask > 1:
        parser.error(
            "--fixed-mask, --no-mask, and --stored-mask are mutually exclusive."
        )

    cfg = smoke_ddpg_config() if args.smoke else DDPGConfig()
    cfg.seed = int(args.seed)

    if args.calibrate:
        calibrate(cfg)
    else:
        cfg.fixed_mask = args.fixed_mask
        cfg.no_mask = args.no_mask
        cfg.stored_mask = args.stored_mask
        suffix = (
            "_nomask"
            if cfg.no_mask
            else "_fixed"
            if cfg.fixed_mask
            else "_stored"
            if cfg.stored_mask
            else ""
        )
        cfg.output_path = cfg.output_path.replace(
            ".", f"_{args.name + suffix}_{args.seed}."
        )
        print("Config:", asdict(cfg))
        train(cfg)
