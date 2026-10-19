"""
run.py - Main entry point for JAX-based masked DQN continual learning on MinAtar.

Core optimizations:
  - Single jax.lax.scan for collect + train (no Python overhead per step)
  - jax.lax.cond for conditional training (both branches traced, one executed)
  - jax.vmap for batched model inference and evaluation episodes
  - Target network sync without mutation via jax.tree.map + jnp.where

Entry points:
  main(cfg)           -- continual learning across a game sequence
  single_game(name)   -- train vanilla DQN on one game (no masking)
"""

import os
import time
import warnings
from typing import NamedTuple

# gymnax MinAtar sets int/float into boolean state arrays internally
warnings.filterwarnings(
    "ignore",
    message="scatter inputs have incompatible types",
    category=FutureWarning,
    module=r"jax\._src\.ops\.scatter",
)

import equinox as eqx
import gymnax
import jax
import jax.numpy as jnp
import optax
from config import GAME_SPECS, Config
from masks import reconfigure_mask
from model import Conv2dDQNMinAtar
from replay_buffer import (
    MaskReplayState,
    ReplayBufferState,
    add,
    init_buffer,
    init_mask_buffer,
    mask_add,
    sample,
    sample_correlated,
)

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def preprocess_obs(obs, n_channels, max_channels):
    """Transpose HWC -> CHW and pad channels to max_channels."""
    obs_chw = jnp.transpose(obs, (2, 0, 1)).astype(jnp.float32)
    padded = jnp.zeros((max_channels, 10, 10), dtype=jnp.float32)
    padded = padded.at[:n_channels].set(obs_chw)
    return padded


def compute_epsilon(step, warmup_steps, eps_start, eps_end, eps_decay_steps):
    """Linear epsilon decay, forced 1.0 during warmup."""
    eps = jnp.maximum(
        eps_end,
        eps_start - (eps_start - eps_end) * step / eps_decay_steps,
    )
    return jnp.where(step < warmup_steps, 1.0, eps)


# ═══════════════════════════════════════════════════════════════════════
# DQN Loss
# ═══════════════════════════════════════════════════════════════════════


def dqn_loss(
    model, target_model, batch, mask, gamma, n_actions, max_actions, *, drop_key=None
):
    """
    DQN Bellman error with Huber loss, vmapped over batch.

    Args:
        model: online Conv2dDQNMinAtar
        target_model: target Conv2dDQNMinAtar
        batch: (states, actions, rewards, next_states, dones)
        mask: binary mask (hidden_size,)
        gamma: discount factor
        n_actions: valid actions for this game
        max_actions: max actions (model output dim)
        drop_key: PRNG key for mask dropout (None = no dropout)
    """
    states, actions, rewards, next_states, dones = batch

    # Apply mask dropout once, shared across the batch (matches PyTorch)
    train_mask = model.drop_mask(mask, drop_key)

    # Vmapped model application
    q_all = jax.vmap(lambda o: model(o, train_mask))(states)
    q_current = q_all[jnp.arange(states.shape[0]), actions]

    # Target model: no dropout
    q_next = jax.vmap(lambda o: target_model(o, mask))(next_states)
    valid = jnp.arange(max_actions) < n_actions
    q_next = jnp.where(valid[None, :], q_next, -jnp.inf)
    q_target = rewards + gamma * (1 - dones) * jnp.max(q_next, axis=-1)

    return jnp.mean(optax.huber_loss(q_current, q_target))


# ═══════════════════════════════════════════════════════════════════════
# Scan-Based Training Loop
# ═══════════════════════════════════════════════════════════════════════


class StepMetrics(NamedTuple):
    step: jnp.ndarray  # int32
    grad_steps: jnp.ndarray  # int32
    episode_reward: jnp.ndarray  # float32
    episode_count: jnp.ndarray  # int32
    total_reward: jnp.ndarray  # float32
    loss_ema: jnp.ndarray  # float32
    mask_replay_writes: jnp.ndarray  # int32 (per-game counter)


class StepLog(NamedTuple):
    """Per-step logged data (output axis of scan)."""

    episode_done: jnp.ndarray  # bool
    episode_reward: jnp.ndarray  # float32 (valid only when episode_done)
    loss: jnp.ndarray  # float32
    epsilon: jnp.ndarray  # float32


def make_train_step(cfg, env, env_params, mask, n_actions, n_channels, optimizer):
    """
    Build the JIT-compatible scan step function for one game.

    Captures cfg, env, env_params, mask, n_actions, n_channels, optimizer
    as compile-time constants.

    Returns a function: (carry, _) -> (carry, StepLog)
    where carry = (model, target, opt_state, replay_buf, mask_replay, env_state, obs, key, metrics)
    """
    max_actions = cfg.max_actions
    max_channels = cfg.max_channels
    gamma = cfg.gamma
    batch_size = cfg.batch_size
    replay_capacity = cfg.replay_capacity
    mask_replay_capacity = cfg.mask_replay_capacity
    min_replay_size = cfg.min_replay_size
    target_update_freq = cfg.target_update_freq
    train_freq = cfg.train_freq
    gradient_steps = cfg.gradient_steps
    warmup_steps = cfg.warmup_steps
    eps_start = cfg.eps_start
    eps_end = cfg.eps_end
    eps_decay_steps = cfg.eps_decay_steps
    grad_clip = cfg.grad_clip
    loss_ema_alpha = cfg.loss_ema_smoothing
    mask_replay_enabled = cfg.mask_replay_enabled
    n_replay_samples = cfg.n_replay_samples
    corr_low = cfg.mask_replay_corr_low
    corr_high = cfg.mask_replay_corr_high
    n_sample_per_game = cfg.n_sample_per_game

    def step_fn(carry, _):
        (
            model,
            target_model,
            opt_state,
            replay_buf,
            mask_replay,
            env_state,
            obs,
            key,
            metrics,
        ) = carry

        step = metrics.step
        key, k_act, k_env, k_train, k_drop = jax.random.split(key, 5)

        # ── Epsilon-greedy action ────────────────────────────────────
        eps = compute_epsilon(step, warmup_steps, eps_start, eps_end, eps_decay_steps)

        q_values = model(obs, model.drop_mask(mask, k_drop))
        valid_mask = jnp.arange(max_actions) < n_actions
        q_values = jnp.where(valid_mask, q_values, -jnp.inf)
        greedy_action = jnp.argmax(q_values)

        random_action = jax.random.randint(k_act, (), 0, n_actions)
        explore = jax.random.uniform(k_act) < eps
        action = jnp.where(explore, random_action, greedy_action).astype(jnp.int32)

        # ── Environment step ─────────────────────────────────────────
        raw_obs, new_env_state, reward, done, _ = env.step(
            k_env, env_state, action, env_params
        )
        new_obs = preprocess_obs(raw_obs, n_channels, max_channels)

        # ── Write to replay buffer ───────────────────────────────────
        replay_buf = add(
            replay_buf,
            obs,
            action,
            reward,
            new_obs,
            done.astype(jnp.float32),
            replay_capacity,
        )

        # ── Write to mask replay (trace-time decision) ───────────────
        if mask_replay_enabled:
            should_write_mr = metrics.mask_replay_writes < n_sample_per_game
            mask_replay = jax.lax.cond(
                should_write_mr,
                lambda mr: mask_add(
                    mr,
                    obs,
                    action,
                    reward,
                    new_obs,
                    done.astype(jnp.float32),
                    mask,
                    mask_replay_capacity,
                ),
                lambda mr: mr,
                mask_replay,
            )
            new_mr_writes = metrics.mask_replay_writes + should_write_mr.astype(
                jnp.int32
            )
        else:
            new_mr_writes = metrics.mask_replay_writes

        # ── Episode tracking ─────────────────────────────────────────
        ep_reward = metrics.episode_reward + reward
        new_episode_count = metrics.episode_count + done.astype(jnp.int32)
        new_total_reward = metrics.total_reward + jnp.where(done, ep_reward, 0.0)
        logged_ep_reward = jnp.where(done, ep_reward, 0.0)
        new_ep_reward = jnp.where(done, 0.0, ep_reward)

        # ── Conditional training ─────────────────────────────────────
        can_train = (
            (step >= min_replay_size)
            & (replay_buf.size >= batch_size)
            & (step % train_freq == 0)
        )

        def _single_grad_step(carry, _):
            """One gradient update: sample batch, compute loss+grads, update."""
            model_, target_, opt_state_, grad_steps_, k_ = carry
            k_, k_sample, k_drop = jax.random.split(k_, 3)

            batch = sample(replay_buf, k_sample, batch_size)

            loss_val, grads = eqx.filter_value_and_grad(
                lambda m: dqn_loss(
                    m,
                    target_,
                    batch,
                    mask,
                    gamma,
                    n_actions,
                    max_actions,
                    drop_key=k_drop,
                )
            )(model_)

            # Mask-correlation replay loss (trace-time decision)
            if mask_replay_enabled:
                k_, k_mr_sample = jax.random.split(k_)

                def add_replay_loss(lr_carry):
                    loss_v, grads_v, k_r = lr_carry
                    k_r, k_r_drop = jax.random.split(k_r)
                    r_obs, r_act, r_rew, r_nobs, r_done, r_mask, has_valid = (
                        sample_correlated(
                            mask_replay,
                            k_r,
                            mask,
                            n_replay_samples,
                            corr_low,
                            corr_high,
                            mask_replay_capacity,
                        )
                    )
                    r_batch = (r_obs, r_act, r_rew, r_nobs, r_done)
                    r_loss, r_grads = eqx.filter_value_and_grad(
                        lambda m: dqn_loss(
                            m,
                            target_,
                            r_batch,
                            r_mask,
                            gamma,
                            n_actions,
                            max_actions,
                            drop_key=k_r_drop,
                        )
                    )(model_)
                    new_loss = loss_v + jnp.where(has_valid, r_loss, 0.0)
                    new_grads = jax.tree.map(
                        lambda g, rg: g + jnp.where(has_valid, rg, 0.0),
                        grads_v,
                        r_grads,
                    )
                    return (new_loss, new_grads, k_r)

                def skip_replay_loss(lr_carry):
                    return lr_carry

                use_mr = mask_replay.size > 0
                loss_val, grads, _ = jax.lax.cond(
                    use_mr,
                    add_replay_loss,
                    skip_replay_loss,
                    (loss_val, grads, k_mr_sample),
                )

            grads = jax.tree.map(lambda g: jnp.clip(g, -grad_clip, grad_clip), grads)
            updates, opt_state_new = optimizer.update(grads, opt_state_, model_)
            model_new = eqx.apply_updates(model_, updates)

            grad_steps_new = grad_steps_ + 1
            should_sync = (grad_steps_new % target_update_freq) == 0
            target_new = jax.tree.map(
                lambda t, m: jnp.where(should_sync, m, t),
                target_,
                model_new,
            )
            return (model_new, target_new, opt_state_new, grad_steps_new, k_), loss_val

        def do_train(args):
            carry_init = args
            carry_final, losses = jax.lax.scan(
                _single_grad_step,
                carry_init,
                None,
                length=gradient_steps,
            )
            model_, target_, opt_state_, grad_steps_, _ = carry_final
            return (model_, target_, opt_state_, grad_steps_, losses[-1])

        def skip_train(args):
            model_, target_, opt_state_, grad_steps_, _ = args
            return (model_, target_, opt_state_, grad_steps_, jnp.float32(0.0))

        train_args = (model, target_model, opt_state, metrics.grad_steps, k_train)
        model, target_model, opt_state, grad_steps, loss_val = jax.lax.cond(
            can_train,
            do_train,
            skip_train,
            train_args,
        )

        # ── Loss EMA ────────────────────────────────────────────────
        new_loss_ema = jnp.where(
            can_train,
            loss_ema_alpha * metrics.loss_ema + (1 - loss_ema_alpha) * loss_val,
            metrics.loss_ema,
        )

        new_metrics = StepMetrics(
            step=step + 1,
            grad_steps=grad_steps,
            episode_reward=new_ep_reward,
            episode_count=new_episode_count,
            total_reward=new_total_reward,
            loss_ema=new_loss_ema,
            mask_replay_writes=new_mr_writes,
        )

        new_carry = (
            model,
            target_model,
            opt_state,
            replay_buf,
            mask_replay,
            new_env_state,
            new_obs,
            key,
            new_metrics,
        )

        log = StepLog(
            episode_done=done,
            episode_reward=logged_ep_reward,
            loss=loss_val,
            epsilon=eps,
        )
        return new_carry, log

    return step_fn


def train_game(
    model, target_model, opt_state, mask, game_name, game_idx, cfg, mask_replay, key
):
    """
    Train DQN on a single game using chunked scans for logging.

    Returns: (model, target_model, opt_state, mask_replay, key, all_episode_rewards)
    """
    n_channels = cfg.game_channels(game_name)
    n_actions = cfg.game_actions(game_name)
    max_channels = cfg.max_channels
    optimizer = optax.adam(cfg.lr)

    # Re-init optimizer state for this game
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Init env
    env, env_params = gymnax.make(game_name)
    key, env_key = jax.random.split(key)
    raw_obs, env_state = env.reset(env_key, env_params)
    obs = preprocess_obs(raw_obs, n_channels, max_channels)

    # Init replay buffer
    obs_shape = (max_channels, 10, 10)
    replay_buf = init_buffer(cfg.replay_capacity, obs_shape)

    # When mask replay is disabled, use a capacity-1 placeholder so the
    # carry pytree stays tiny (~10 KB instead of ~315 MB).
    if not cfg.mask_replay_enabled:
        mask_replay = init_mask_buffer(1, obs_shape, cfg.hidden_size)

    # Build step function
    step_fn = make_train_step(
        cfg, env, env_params, mask, n_actions, n_channels, optimizer
    )

    # JIT compile the chunked scan
    @jax.jit
    def run_chunk(carry):
        return jax.lax.scan(step_fn, carry, None, length=cfg.scan_chunk_size)

    metrics = StepMetrics(
        step=jnp.int32(0),
        grad_steps=jnp.int32(0),
        episode_reward=jnp.float32(0.0),
        episode_count=jnp.int32(0),
        total_reward=jnp.float32(0.0),
        loss_ema=jnp.float32(0.0),
        mask_replay_writes=jnp.int32(0),
    )

    carry = (
        model,
        target_model,
        opt_state,
        replay_buf,
        mask_replay,
        env_state,
        obs,
        key,
        metrics,
    )

    n_chunks = cfg.max_steps_per_game // cfg.scan_chunk_size
    all_episode_rewards = []
    chunk_times = []

    for chunk_idx in range(n_chunks):
        t_chunk = time.time()
        carry, logs = run_chunk(carry)
        # Block to get accurate per-chunk timing
        jax.tree.map(
            lambda x: x.block_until_ready(), eqx.filter(carry[0], eqx.is_array)
        )
        chunk_times.append(time.time() - t_chunk)

        # Extract episode rewards from this chunk
        done_mask = logs.episode_done
        ep_rewards = logs.episode_reward[done_mask]
        all_episode_rewards.extend(ep_rewards.tolist())

        metrics = carry[8]
        step = int(metrics.step)
        ep_count = int(metrics.episode_count)
        loss_ema = float(metrics.loss_ema)
        eps = float(logs.epsilon[-1])

        if (chunk_idx + 1) % max(1, cfg.log_every_steps // cfg.scan_chunk_size) == 0:
            recent = ep_rewards.tolist()
            avg_r = sum(recent) / max(len(recent), 1)
            tag = " (incl. JIT)" if chunk_idx == 0 else ""
            print(
                f"    [{game_name}] step {step}/{cfg.max_steps_per_game}, "
                f"eps={eps:.3f}, loss={loss_ema:.4f}, "
                f"episodes={ep_count}, avg_reward={avg_r:.1f}, "
                f"chunk={chunk_times[-1]:.1f}s{tag}"
            )

    if chunk_times:
        jit_time = chunk_times[0]
        exec_times = chunk_times[1:]
        exec_total = sum(exec_times)
        print(
            f"    Timing: JIT+first chunk={jit_time:.1f}s, "
            f"remaining {len(exec_times)} chunks={exec_total:.1f}s "
            f"({exec_total / max(len(exec_times), 1):.2f}s/chunk)"
        )

    # Extract final state
    (
        model,
        target_model,
        opt_state,
        replay_buf,
        mask_replay,
        env_state,
        obs,
        key,
        metrics,
    ) = carry

    return model, target_model, opt_state, mask_replay, key, all_episode_rewards


# ═══════════════════════════════════════════════════════════════════════
# Mask Discovery
# ═══════════════════════════════════════════════════════════════════════


def discover_mask(model, game_name, cfg, key):
    """
    Iterative mask discovery: alternate between collecting on-policy
    observations and gradient-based mask reconfiguration.

    1. Initialize mask logits at zeros
    2. Collect observations under epsilon-greedy (eps=1.0 on first iter)
    3. Reconfigure mask logits on collected data
    4. Repeat 2-3 with the reconfigured mask guiding the policy
    """
    n_channels = cfg.game_channels(game_name)
    n_actions = cfg.game_actions(game_name)
    max_channels = cfg.max_channels
    max_actions = cfg.max_actions
    hidden_size = cfg.hidden_size
    n_collect = cfg.reconfig_sample_size

    env, env_params = gymnax.make(game_name)

    key, reset_key = jax.random.split(key)
    raw_obs, env_state = env.reset(reset_key, env_params)

    # ── JIT'd epsilon-greedy collection ──────────────────────────────
    @jax.jit
    def run_collect(env_state, raw_obs, mask, epsilon, collect_key):
        def collect_step(carry, _):
            env_st, obs, rng = carry
            rng, k_act, k_step, k_eps = jax.random.split(rng, 4)
            processed = preprocess_obs(obs, n_channels, max_channels)

            q = model(processed, mask)
            valid = jnp.arange(max_actions) < n_actions
            q = jnp.where(valid, q, -jnp.inf)
            greedy = jnp.argmax(q)
            random_a = jax.random.randint(k_act, (), 0, n_actions)
            action = jnp.where(
                jax.random.uniform(k_eps) < epsilon, random_a, greedy
            ).astype(jnp.int32)

            new_raw, new_env_st, _, _, _ = env.step(k_step, env_st, action, env_params)
            return (new_env_st, new_raw, rng), processed

        (final_env, final_obs, _), obs_batch = jax.lax.scan(
            collect_step, (env_state, raw_obs, collect_key), None, length=n_collect
        )
        return final_env, final_obs, obs_batch

    reconfig_jit = jax.jit(reconfigure_mask, static_argnums=(2,))

    # ── Iterative discover loop ──────────────────────────────────────
    m_logits = jnp.zeros(hidden_size)
    mask = jnp.zeros(hidden_size)  # placeholder for first iter (eps=1.0)

    for i in range(cfg.discover_iterations):
        # First iteration: pure random; subsequent: epsilon-greedy
        eps = jnp.float32(1.0 if i == 0 else cfg.discover_epsilon)

        key, collect_key = jax.random.split(key)
        env_state, raw_obs, obs_batch = run_collect(
            env_state, raw_obs, mask, eps, collect_key
        )

        key, reconfig_key = jax.random.split(key)
        mask, m_logits = reconfig_jit(model, obs_batch, cfg, reconfig_key, m_logits)

    return mask, key


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════


def evaluate_game(model, game_name, mask, cfg, key, n_episodes=None):
    """
    Evaluate agent on a game using vmapped episodes with greedy policy.

    Returns: mean episode reward (float)
    """
    if n_episodes is None:
        n_episodes = cfg.eval_episodes
    n_channels = cfg.game_channels(game_name)
    n_actions = cfg.game_actions(game_name)
    max_channels = cfg.max_channels
    max_actions = cfg.max_actions

    env, env_params = gymnax.make(game_name)
    max_ep_steps = 1000

    def run_episode(ep_key):
        reset_key, step_key = jax.random.split(ep_key)
        raw_obs, env_state = env.reset(reset_key, env_params)
        obs = preprocess_obs(raw_obs, n_channels, max_channels)

        def eval_step(carry, _):
            env_state, obs, reward_sum, done, rng = carry
            rng, k_step = jax.random.split(rng)

            q = model(obs, mask)
            valid = jnp.arange(max_actions) < n_actions
            q = jnp.where(valid, q, -jnp.inf)
            action = jnp.argmax(q).astype(jnp.int32)

            new_raw, new_env_state, reward, step_done, _ = env.step(
                k_step, env_state, action, env_params
            )
            new_obs = preprocess_obs(new_raw, n_channels, max_channels)

            reward_sum = reward_sum + jnp.where(done, 0.0, reward)
            done = done | step_done

            return (new_env_state, new_obs, reward_sum, done, rng), None

        init_carry = (env_state, obs, jnp.float32(0.0), jnp.bool_(False), step_key)
        (_, _, total_reward, _, _), _ = jax.lax.scan(
            eval_step, init_carry, None, length=max_ep_steps
        )
        return total_reward

    eval_keys = jax.random.split(key, n_episodes)
    rewards = jax.jit(jax.vmap(run_episode))(eval_keys)
    return float(jnp.mean(rewards))


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════


def save_reward_plot(episode_rewards, game_name, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 4))
    plt.plot(episode_rewards, linewidth=0.6, alpha=0.4, label="episode")
    if len(episode_rewards) >= 10:
        window = min(50, len(episode_rewards) // 2)
        avg = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, window - 1 + len(avg)),
            avg,
            linewidth=1.5,
            label=f"avg({window})",
        )
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(f"{game_name} (JAX)")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"reward_{game_name}.png"), dpi=120)
    plt.close()


def save_eval_plot(eval_history, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if not eval_history:
        return

    all_games = set()
    for _, results in eval_history:
        all_games.update(results.keys())

    for game in sorted(all_games):
        xs, medians, q25s, q75s = [], [], [], []
        for game_idx, results in eval_history:
            if game in results:
                scores = results[game]
                xs.append(game_idx)
                medians.append(np.median(scores))
                q25s.append(np.percentile(scores, 25))
                q75s.append(np.percentile(scores, 75))
        label = game.replace("-MinAtar", "")
        plt.plot(xs, medians, marker="o", markersize=3, linewidth=1, label=label)
        plt.fill_between(xs, q25s, q75s, alpha=0.2)

    plt.xlabel("Games completed")
    plt.ylabel("Mean episode reward")
    plt.title("Eval retention (median + IQR over mask recovery attempts)")
    plt.legend(fontsize=6, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "eval_retention.png"), dpi=150)
    plt.close()


def save_mask_visualization(game_masks, square_dim, outdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    n_games = len(game_masks)
    if n_games == 0:
        return

    cmap = plt.colormaps.get_cmap("tab10").resampled(max(n_games, 10))

    # ── RGB overlap image ────────────────────────────────────────────
    rgb = np.zeros((square_dim, square_dim, 3), dtype=np.float32)
    for i, (game, mask) in enumerate(game_masks.items()):
        color = np.array(cmap(i % 10)[:3], dtype=np.float32)
        m2d = np.array(mask).reshape(square_dim, square_dim)
        for c in range(3):
            rgb[:, :, c] = np.maximum(rgb[:, :, c], m2d * color[c])

    rgb_uint8 = (rgb * 255).astype(np.uint8)
    img = Image.fromarray(rgb_uint8, mode="RGB")
    img = img.resize((square_dim * 4, square_dim * 4), Image.NEAREST)
    img.save(os.path.join(outdir, "mask_allocation.png"))

    # ── Individual mask heatmaps ─────────────────────────────────────
    fig, axes = plt.subplots(1, n_games, figsize=(3 * n_games, 3))
    if n_games == 1:
        axes = [axes]
    for i, (game, mask) in enumerate(game_masks.items()):
        m2d = np.array(mask).reshape(square_dim, square_dim)
        axes[i].imshow(m2d, cmap="hot", vmin=0, vmax=1)
        axes[i].set_title(game.replace("-MinAtar", ""), fontsize=8)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "individual_masks.png"), dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main: Continual Learning
# ═══════════════════════════════════════════════════════════════════════


def main(cfg=None):
    if cfg is None:
        cfg = Config()

    key = jax.random.PRNGKey(cfg.seed)

    outdir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(outdir, exist_ok=True)

    print("=" * 60)
    print("MinAtar Continual Learning with Masked DQN (JAX)")
    print("=" * 60)
    print(f"Games: {list(cfg.game_sequence)}")
    print(f"Steps per game: {cfg.max_steps_per_game:,}")
    print(f"Hidden size: {cfg.hidden_size} ({cfg.square_dim}x{cfg.square_dim})")
    print(f"K (KWTA winners): {cfg.k}")
    print(f"Mask reconfig: {cfg.reconfig_steps} steps @ lr={cfg.reconfig_lr}")
    print(f"Max channels: {cfg.max_channels}, Max actions: {cfg.max_actions}")
    print()

    # Print game specs
    print("Game specs:")
    for game in cfg.game_sequence:
        ch, na = GAME_SPECS[game]
        print(f"  {game}: {ch} channels, {na} actions")
    print()

    # Create model
    key, model_key = jax.random.split(key)
    model = Conv2dDQNMinAtar(
        max_actions=cfg.max_actions,
        hidden_size=cfg.hidden_size,
        in_channels=cfg.max_channels,
        n_layers=cfg.fc_layers,
        key=model_key,
        drop_p=cfg.dropout,
    )
    target_model = model  # initial copy (immutable in JAX)

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model: {n_params:,} parameters")

    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Init mask replay
    obs_shape = (cfg.max_channels, 10, 10)
    mask_replay = init_mask_buffer(cfg.mask_replay_capacity, obs_shape, cfg.hidden_size)

    # Training loop over games
    game_masks = {}
    eval_history = []

    for game_idx, game_name in enumerate(cfg.game_sequence):
        print(f"\n{'=' * 60}")
        print(f"GAME {game_idx + 1}/{cfg.n_games}: {game_name}")
        print(f"{'=' * 60}")

        # Discover mask
        print(f"  Discovering mask for {game_name}...")
        key, mask_key = jax.random.split(key)
        game_mask, key = discover_mask(model, game_name, cfg, mask_key)
        game_masks[game_name] = game_mask
        active = int(game_mask.sum())
        print(f"  Mask: {active}/{cfg.hidden_size} neurons active")

        # Train
        t0 = time.time()
        model, target_model, opt_state, mask_replay, key, episode_rewards = train_game(
            model,
            target_model,
            opt_state,
            game_mask,
            game_name,
            game_idx,
            cfg,
            mask_replay,
            key,
        )
        dt = time.time() - t0
        print(f"  {game_name} done in {dt:.1f}s, {len(episode_rewards)} episodes")

        # Re-discover mask after training
        if cfg.reconfig_steps > 0:
            key, mask_key2 = jax.random.split(key)
            game_mask, key = discover_mask(model, game_name, cfg, mask_key2)
            game_masks[game_name] = game_mask
            print(f"  Post-training mask: {int(game_mask.sum())} active neurons")

        # Save reward plot
        save_reward_plot(episode_rewards, game_name, outdir)

        # Evaluation: multiple mask recovery attempts to capture variability
        if (game_idx + 1) % cfg.eval_every_n_games == 0 or game_idx == cfg.n_games - 1:
            n_att = cfg.eval_mask_attempts
            print(
                f"\n  -- Evaluating retention ({len(game_masks)} games, "
                f"{n_att} mask recovery attempts) --"
            )
            import numpy as _np

            eval_results = {}
            for g_name in game_masks:
                scores = []
                for _ in range(n_att):
                    key, mask_key, eval_key = jax.random.split(key, 3)
                    attempt_mask, _ = discover_mask(model, g_name, cfg, mask_key)
                    mean_r = evaluate_game(model, g_name, attempt_mask, cfg, eval_key)
                    scores.append(mean_r)
                eval_results[g_name] = scores
                med = _np.median(scores)
                lo, hi = _np.percentile(scores, [25, 75])
                print(f"    {g_name:25s}: median={med:.1f}  IQR=[{lo:.1f}, {hi:.1f}]")
            eval_history.append((game_idx + 1, eval_results))

        # Save plots
        save_eval_plot(eval_history, outdir)
        save_mask_visualization(game_masks, cfg.square_dim, outdir)

    # Final summary
    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")

    if eval_history:
        import numpy as _np

        print("\nFinal evaluation:")
        _, final_results = eval_history[-1]
        for game, scores in final_results.items():
            med = _np.median(scores)
            lo, hi = _np.percentile(scores, [25, 75])
            print(f"  {game:25s}: median={med:.1f}  IQR=[{lo:.1f}, {hi:.1f}]")

    print("\nMask allocation:")
    for game, mask in game_masks.items():
        print(f"  {game:25s}: {int(mask.sum())}/{cfg.hidden_size} active")

    # Mask overlaps
    names = list(game_masks.keys())
    if len(names) > 1:
        print("\nMask overlaps:")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                mi, mj = game_masks[names[i]], game_masks[names[j]]
                overlap = float((mi * mj).sum())
                union = float(((mi + mj) > 0).sum())
                iou = overlap / max(union, 1)
                print(
                    f"  {names[i]:20s} & {names[j]:20s}: {overlap:.0f} (IoU={iou:.2f})"
                )

    print(f"\nOutputs saved to {outdir}/")


# ═══════════════════════════════════════════════════════════════════════
# Single Game Mode
# ═══════════════════════════════════════════════════════════════════════


def single_game(game_name, cfg=None):
    """Train vanilla DQN on a single game with all-ones mask (no masking)."""
    if cfg is None:
        cfg = Config(game_sequence=(game_name,), mask_replay_enabled=False)
    key = jax.random.PRNGKey(cfg.seed)

    outdir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(outdir, exist_ok=True)

    n_actions = cfg.game_actions(game_name)
    n_channels = cfg.game_channels(game_name)
    print(f"Single-game baseline (JAX MinAtar): {game_name}")
    print(f"  channels={n_channels}, actions={n_actions}")
    print(f"  Steps: {cfg.max_steps_per_game:,}")

    key, model_key = jax.random.split(key)
    model = Conv2dDQNMinAtar(
        max_actions=cfg.max_actions,
        hidden_size=cfg.hidden_size,
        in_channels=cfg.max_channels,
        n_layers=cfg.fc_layers,
        key=model_key,
        drop_p=cfg.dropout,
    )
    target_model = model

    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # All-ones mask = no masking
    mask = jnp.ones(cfg.hidden_size)

    obs_shape = (cfg.max_channels, 10, 10)
    # Disabled mask replay: capacity-1 placeholder
    mask_replay = init_mask_buffer(1, obs_shape, cfg.hidden_size)

    n_params = sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    print(f"  Model: {n_params:,} parameters")

    t0 = time.time()
    model, target_model, opt_state, mask_replay, key, episode_rewards = train_game(
        model,
        target_model,
        opt_state,
        mask,
        game_name,
        0,
        cfg,
        mask_replay,
        key,
    )
    dt = time.time() - t0
    print(f"  Training done in {dt:.1f}s, {len(episode_rewards)} episodes")

    # Evaluate
    key, eval_key = jax.random.split(key)
    mean_r = evaluate_game(model, game_name, mask, cfg, eval_key, n_episodes=10)
    print(f"  Eval (10 episodes): mean reward = {mean_r:.1f}")

    save_reward_plot(episode_rewards, game_name, outdir)
    print(f"  Plot saved to {outdir}/")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="JAX Masked DQN on MinAtar")
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Single game mode (e.g. --single Breakout-MinAtar)",
    )
    parser.add_argument(
        "--steps", type=int, default=None, help="Override max_steps_per_game"
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=None,
        help="Env steps between training bursts (default: 1)",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=None,
        help="Gradient updates per training burst (default: 1)",
    )
    args = parser.parse_args()

    # Build config overrides from CLI args
    cfg_overrides = {}
    if args.steps:
        cfg_overrides["max_steps_per_game"] = args.steps
    if args.train_freq:
        cfg_overrides["train_freq"] = args.train_freq
    if args.gradient_steps:
        cfg_overrides["gradient_steps"] = args.gradient_steps

    if args.single:
        # Convenience: allow short names
        name = args.single
        if not name.endswith("-MinAtar"):
            for full_name in GAME_SPECS:
                if full_name.lower().startswith(name.lower()):
                    name = full_name
                    break
        cfg = Config(game_sequence=(name,), mask_replay_enabled=False, **cfg_overrides)
        single_game(name, cfg)
    else:
        cfg = Config(**cfg_overrides)
        main(cfg)
