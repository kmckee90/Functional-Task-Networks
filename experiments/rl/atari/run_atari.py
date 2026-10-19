"""
run_atari.py - Continual learning on Atari with masked DQN.

One agent learns a sequence of Atari games.  For each game:
  1. Discover a mask via gradient-based optimization (SmoothKWTA from parent).
  2. Train DQN with that mask for up to MAX_STEPS_PER_GAME env steps.
  3. Store mask + small replay sample for cross-game replay.

Every EVAL_EVERY_N_GAMES games, evaluate on ALL previously learned games
(using their stored masks) to demonstrate retention.

The masking system (SmoothKWTA, KWTA, etc.) is imported directly from the
parent project's masks.py.
"""

import copy
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Import local modules first (before modifying sys.path)
from config import AtariConfig
from dqn import Conv2dDQNMinAtar, MaskedDQN, MaskedDQNMinAtar
from env_utils import (
    get_action_space_size,
    get_max_action_space,
    get_minatar_max_channels,
    make_atari_env,
    make_env,
)
from replay_buffer import MaskReplayBuffer, ReplayBuffer

# Add parent dir to path so we can import masks.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import masks as parent_masks


# ═══════════════════════════════════════════════════════════════════════
# Mask reconfiguration for RL
# ═══════════════════════════════════════════════════════════════════════
def reconfigure_mask_rl_iterate(
    model, target_model, mask_configurer, env, cfg, current_mask
):
    def gather(m, eps):
        obs_list = []
        with torch.no_grad():
            obs, _ = env.reset()
            for _ in range(cfg.RECONFIG_SAMPLE_SIZE):
                obs_list.append(obs)
                if torch.rand(1) < eps:
                    action = env.action_space.sample()
                else:
                    q = model(torch.tensor(obs).unsqueeze(0), m)
                    action = q.argmax()
                obs, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    obs, _ = env.reset()

            states = torch.from_numpy(np.stack(obs_list)).to(torch.empty(0).device)
            # next_states = states[1:]
            # states = states[:-1]
            # Use the mask configurer's gradient-based optimization.
            # We create a pseudo-data tuple (x, y) where:
            #   x = stacked observations
            #   y = current Q-value predictions (self-supervised target)
            with torch.no_grad():
                # Use a uniform mask to get initial Q estimates as targets
                q_targets = model(states, m_init)

            # Target Q-values (max over valid actions only)
            # with torch.no_grad():
            #     q_next = target_model(next_states, mask)
            #     # Mask invalid actions
            # q_next[:, n_actions:] = -float("inf")
            #     q_target = rewards + gamma * (1 - dones) * q_next.max(1)[0]

        return states, q_targets

    # Steps:
    # Init any mask (unusused)
    # Collect initial eps=1.0 data
    def reconfig_q(data, m_init_logits, steps=None):
        if steps is None:
            steps = cfg.RECONFIG_STEPS

        x, y = data
        n = min(cfg.RECONFIG_SAMPLE_SIZE, x.shape[0])
        # Initialize mask parameter
        m_logit = torch.nn.Parameter(m_init_logits)
        optimizer = torch.optim.RMSprop([m_logit], lr=cfg.RECONFIG_LR)

        for _ in range(steps):
            sm = torch.sigmoid(m_logit)
            mask = torch.bernoulli(sm) + sm - sm.detach()  # straight-through
            q_pred = model(x[:n], mask)
            # loss = F.mse_loss(q_pred[:n], y[:n])
            loss = F.smooth_l1_loss(q_pred[:n], y[:n].detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Create a temporary SmoothKWTA-like configurer for lateral + KWTA
        # We reuse the parent's lateral step and kwta functions
        hidden_size = model.hidden_size
        square_dim = cfg.SQUARE_DIM

        # Apply lateral smoothing
        configurer = parent_masks.SmoothKWTAFast(
            hidden_size=hidden_size,
            square_dim=square_dim,
            **cfg.MASK_PARS,
        )
        # Run lateral steps on the optimized continuous mask
        m_smooth = m_logit.data.clone()
        if configurer.lateral_polyak < 1.0:
            for _ in range(configurer.lateral_steps):
                m_lateral = configurer.step_lateral(m_smooth)
                m_smooth = (
                    configurer.lateral_polyak * m_smooth
                    + (1 - configurer.lateral_polyak) * m_lateral
                )

        # KWTA sparsification
        m_final = parent_masks.kwta(m_smooth, k=cfg.K)

        return m_logit, m_final

    m_init = torch.ones_like(current_mask)
    mask = m_init
    mask_logit = m_init * 0

    # breakpoint()
    # cfg.RECONFIG_LR = 0.1
    # cfg.RECONFIG_SAMPLE_SIZE = 2048 * 4

    # mask = torch.ones_like(mask)
    # mask_logit = mask * 0
    # dat = gather(mask, 1)

    # dat = gather(mask, 0.1)
    # mask = torch.ones_like(mask)
    # mask_logit = mask * 0

    # mask_logit, mask = reconfig_q(dat, mask_logit, steps=100)
    # (current_mask != mask).sum()
    if cfg.DISABLE_MASK:
        mask = torch.zeros_like(mask)
        mask[: cfg.MASK_PARS["k"]] = 1.0
        return mask
    else:
        for i in range(1):
            dat = gather(mask, 1.0 if i == 0 else cfg.EPS_END)
            mask_logit, mask = reconfig_q(dat, mask_logit, steps=None)
    return mask


# ═══════════════════════════════════════════════════════════════════════
# DQN training step
# ═══════════════════════════════════════════════════════════════════════


def dqn_loss(model, target_model, batch, mask, gamma, n_actions):
    """
    Compute DQN loss (standard Bellman error).

    Args:
        model: Online DQN network
        target_model: Target DQN network
        batch: (states, actions, rewards, next_states, dones) tensors
        mask: Current game's mask
        gamma: Discount factor
        n_actions: Number of valid actions for this game

    Returns:
        Scalar loss tensor
    """
    states, actions, rewards, next_states, dones = batch

    # Current Q-values for chosen actions
    q_values = model(states, mask)
    q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q-values (max over valid actions only)
    with torch.no_grad():
        q_next = target_model(next_states, mask)
        # Mask invalid actions
        q_next[:, n_actions:] = -float("inf")
        q_target = rewards + gamma * (1 - dones) * q_next.max(1)[0]

    return F.smooth_l1_loss(q_current, q_target)


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════


def evaluate_game(model, game_name, mask, cfg, n_episodes=None):
    """
    Evaluate the agent on a single game using its stored mask.

    Returns mean episode reward over n_episodes.
    """
    if n_episodes is None:
        n_episodes = cfg.EVAL_EPISODES

    env = make_env(
        game_name,
        cfg,
        seed=cfg.SEED + 9999,
        pad_channels=getattr(cfg, "_pad_channels", None),
    )
    n_actions = env.action_space.n
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            state = (
                torch.from_numpy(np.array(obs)).unsqueeze(0).to(torch.empty(0).device)
            )
            with torch.no_grad():
                q = model(state, mask)
                q[:, n_actions:] = -float("inf")
                action = q.argmax(1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)

    env.close()
    return np.mean(rewards)


def evaluate_all_games(model, game_masks, cfg):
    """
    Evaluate on all learned games so far.

    Args:
        model: Trained DQN
        game_masks: dict {game_name: mask_tensor}
        cfg: AtariConfig

    Returns:
        dict {game_name: mean_reward}
    """
    results = {}
    model.eval()
    for game_name, mask in game_masks.items():
        mean_reward = evaluate_game(model, game_name, mask, cfg)
        results[game_name] = mean_reward
        print(f"    {game_name:20s}: {mean_reward:.1f}")
    model.train()
    return results


# ═══════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════


def save_reward_plot(episode_rewards, game_name, outdir):
    """Save per-episode reward curve for a single game (overwritten in-place)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(episode_rewards, linewidth=0.6, alpha=0.4, label="episode")
    # Running average
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
    plt.title(game_name)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"reward_{game_name}.png"), dpi=120)
    plt.close()


def save_training_plot(all_losses, outdir):
    """Save per-game training loss curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for game, losses in all_losses.items():
        if losses:
            plt.plot(losses, linewidth=0.8, label=game)
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "training_losses.png"), dpi=150)
    plt.close()


def save_eval_plot(eval_history, outdir, train_history=None, null_baselines=None):
    """Save evaluation reward scatter + mean line over cycles.

    eval_history: list of (game_idx, {game_name: reward}, cycle)
    train_history: {game_name: {cycle: mean_train_reward}} or None
    null_baselines: {game_name: early_reward_from_first_cycle} or None
    X-axis is cycle, individual eval scores are scattered, mean per cycle on top.
    Training scores shown as dashed lines, null baselines as dotted lines.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not eval_history:
        return

    # Collect all game names
    all_games = set()
    for results, _ in eval_history:
        all_games.update(results.keys())

    # X range for null baseline lines
    all_cycles = sorted({c for _, c in eval_history})

    cmap = plt.cm.get_cmap("tab10", max(len(all_games), 10))

    for i, game in enumerate(sorted(all_games)):
        color = cmap(i)
        # Gather (cycle, score) pairs for this game
        cycle_scores = {}  # {cycle: [scores]}
        for results, cycle in eval_history:
            if game in results:
                cycle_scores.setdefault(cycle, []).append(results[game])

        # Scatter all individual points (jittered slightly for visibility)
        all_xs, all_ys = [], []
        for cycle, scores in sorted(cycle_scores.items()):
            for s in scores:
                all_xs.append(cycle)
                all_ys.append(s)
        jitter = (np.random.default_rng(i).random(len(all_xs)) - 0.5) * 0.15
        plt.scatter(
            np.array(all_xs) + jitter,
            all_ys,
            color=color,
            alpha=0.35,
            s=27,
            zorder=2,
        )

        # Mean eval line on top
        mean_xs = sorted(cycle_scores.keys())
        mean_ys = [np.mean(cycle_scores[c]) for c in mean_xs]
        plt.plot(
            mean_xs,
            mean_ys,
            marker="o",
            markersize=6,
            linewidth=1.5,
            color=color,
            label=f"{game} (eval)",
            zorder=3,
        )

        # Training score line (dashed, same color)
        if train_history and game in train_history:
            train_xs = sorted(train_history[game].keys())
            train_ys = [train_history[game][c] for c in train_xs]
            plt.plot(
                train_xs,
                train_ys,
                marker="s",
                markersize=4.5,
                linewidth=1.2,
                color=color,
                linestyle="--",
                alpha=0.7,
                label=f"{game} (train)",
                zorder=2,
            )

        # Null baseline (horizontal dotted line from first-cycle early episodes)
        if null_baselines and game in null_baselines:
            plt.axhline(
                y=null_baselines[game],
                linewidth=1,
                color=color,
                linestyle=":",
                alpha=0.5,
                label=f"{game} (null)",
                zorder=1,
            )

    plt.xlabel("Cycle")
    plt.ylabel("Reward")
    plt.legend(fontsize=6, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "eval_retention.png"), dpi=150)
    plt.close()


def save_mask_visualization(game_masks, square_dim, outdir):
    """Save RGB mask visualization like the parent project."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    n_games = len(game_masks)
    if n_games == 0:
        return

    # Assign colors to games (cycle through a colormap)
    cmap = plt.cm.get_cmap("tab10", max(n_games, 10))

    grid_h = square_dim
    hidden = square_dim**2
    n_blocks = 1  # assuming single block for now
    grid_w = n_blocks * square_dim

    rgb = torch.zeros(grid_h, grid_w, 3, device="cpu")
    for i, (game, mask) in enumerate(game_masks.items()):
        color = cmap(i % 10)[:3]
        m2d = parent_masks.mask_to_grid(mask.detach().cpu(), square_dim)
        for c in range(3):
            rgb[:, :, c] = torch.max(rgb[:, :, c], m2d * color[c])

    rgb = (rgb.numpy() * 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((grid_w * 4, grid_h * 4), Image.NEAREST)
    img.save(os.path.join(outdir, "mask_allocation.png"))

    # Also save individual masks
    fig, axes = plt.subplots(1, n_games, figsize=(3 * n_games, 3))
    if n_games == 1:
        axes = [axes]
    for i, (game, mask) in enumerate(game_masks.items()):
        m2d = parent_masks.mask_to_grid(mask.detach().cpu(), square_dim)
        axes[i].imshow(m2d.numpy(), cmap="hot", vmin=0, vmax=1)
        axes[i].set_title(game, fontsize=8)
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "individual_masks.png"), dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════


def train_game(
    model,
    target_model,
    optimizer,
    game_name,
    game_idx,
    mask,
    cfg,
    replay_buf,
    mask_replay,
    outdir,
):
    """
    Train DQN on a single Atari game for up to MAX_STEPS_PER_GAME steps.

    Args:
        model: Online MaskedDQN
        target_model: Target MaskedDQN
        optimizer: AdamW optimizer
        game_name: e.g. "Pong"
        game_idx: Index in the curriculum
        mask: Binary mask for this game
        cfg: AtariConfig
        replay_buf: Standard DQN replay buffer (reset per game)
        mask_replay: Cross-game mask replay buffer (persistent)
        outdir: Output directory

    Returns:
        list of loss values
    """
    env = make_env(
        game_name,
        cfg,
        seed=cfg.SEED + game_idx,
        pad_channels=getattr(cfg, "_pad_channels", None),
    )
    n_actions = env.action_space.n

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_count = 0
    episode_rewards = []
    losses = []
    loss_smooth = 0.0
    grad_steps_total = 0

    mask_replay.new_game()

    for step in range(1, cfg.MAX_STEPS_PER_GAME + 1):
        # ── Epsilon-greedy action selection ─────────────────────────
        eps = max(
            cfg.EPS_END,
            cfg.EPS_START - (cfg.EPS_START - cfg.EPS_END) * step / cfg.EPS_DECAY_STEPS,
        )

        if step < cfg.WARMUP_STEPS or np.random.random() < eps:
            action = env.action_space.sample()
        else:
            state_t = (
                torch.from_numpy(np.array(obs)).unsqueeze(0).to(torch.empty(0).device)
            )
            with torch.no_grad():
                q = model(state_t, mask)
                q[:, n_actions:] = -float("inf")
                action = q.argmax(1).item()

        # ── Step environment ────────────────────────────────────────
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        clipped_reward = np.clip(reward, -1.0, 1.0) if cfg.REWARD_CLIP else reward

        # Store in standard replay buffer
        replay_buf.push(
            np.array(obs, dtype=np.uint8),
            action,
            clipped_reward,
            np.array(next_obs, dtype=np.uint8),
            float(done),
        )

        # Store in mask replay buffer (limited samples per game)
        if (mask_replay.count < cfg.N_SAMPLE_PER_GAME) and (
            torch.rand(1) < (cfg.N_SAMPLE_PER_GAME / cfg.MAX_STEPS_PER_GAME)
        ):
            mask_replay.add(
                np.array(obs, dtype=np.uint8),
                action,
                clipped_reward,
                np.array(next_obs, dtype=np.uint8),
                float(done),
                mask,
            )

        episode_reward += reward
        obs = next_obs

        if done:
            obs, _ = env.reset()
            episode_count += 1
            episode_rewards.append(episode_reward)
            if episode_count % 10 == 0:
                print(
                    f"    [{game_name}] ep {episode_count}, "
                    f"step {step}/{cfg.MAX_STEPS_PER_GAME}, "
                    f"reward={episode_reward:.0f}, eps={eps:.3f}"
                )
            if episode_count % cfg.PLOT_EVERY == 0:
                save_reward_plot(episode_rewards, game_name, outdir)
            episode_reward = 0.0

        # ── Training burst ──────────────────────────────────────────
        if step >= cfg.MIN_REPLAY_SIZE and step % cfg.TRAIN_FREQ == 0:
            for _grad_step in range(cfg.GRADIENT_STEPS):
                batch = replay_buf.sample(cfg.BATCH_SIZE)
                loss = dqn_loss(model, target_model, batch, mask, cfg.GAMMA, n_actions)

                # Mask-correlation replay (cross-game)
                if cfg.MASK_REPLAY_ENABLED and len(mask_replay) > 0:
                    replay_sample = mask_replay.sample_correlated(
                        mask,
                        cfg.N_REPLAY_SAMPLES,
                        cfg.MASK_REPLAY_CORR_LOW,
                        cfg.MASK_REPLAY_CORR_HIGH,
                    )
                    if replay_sample is not None:
                        (
                            r_states,
                            r_actions,
                            r_rewards,
                            r_next_states,
                            r_dones,
                            r_mask,
                        ) = replay_sample
                        r_batch = (
                            r_states,
                            r_actions,
                            r_rewards,
                            r_next_states,
                            r_dones,
                        )
                        r_loss = dqn_loss(
                            model, target_model, r_batch, r_mask, cfg.GAMMA, n_actions
                        )
                        loss = loss + r_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
                optimizer.step()

                # Target network update (counted by gradient steps)
                grad_steps_total += 1
                if grad_steps_total % cfg.TARGET_UPDATE_FREQ == 0:
                    target_model.load_state_dict(model.state_dict())

            li = loss.item()
            loss_smooth = (
                cfg.LOSS_EMA_SMOOTHING * loss_smooth + (1 - cfg.LOSS_EMA_SMOOTHING) * li
            )
            losses.append(loss_smooth)

        # ── Logging ─────────────────────────────────────────────────
        if step % cfg.LOG_EVERY_STEPS == 0 and losses:
            print(
                f"    [{game_name}] step {step}/{cfg.MAX_STEPS_PER_GAME}, "
                f"loss={losses[-1]:.4f}, eps={eps:.3f}, "
                f"replay={len(replay_buf)}, episodes={episode_count}"
            )

    env.close()
    mean_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0.0
    early_reward = np.mean(episode_rewards[:10]) if episode_rewards else 0.0
    return losses, mean_reward, early_reward


def main(cfg=None):
    if cfg is None:
        cfg = AtariConfig()

    torch.set_default_device("cuda")
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)

    outdir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(outdir, exist_ok=True)

    mode_label = "MinAtar" if cfg.USE_MINATAR else "Atari"
    print("=" * 60)
    print(f"{mode_label} Continual Learning with Masked DQN")
    print("=" * 60)
    print(f"Games: {cfg.GAME_SEQUENCE}")
    print(f"Steps per game: {cfg.MAX_STEPS_PER_GAME:,}")
    print(f"Hidden size: {cfg.HIDDEN_SIZE} ({cfg.SQUARE_DIM}x{cfg.SQUARE_DIM})")
    print(f"K (KWTA winners): {cfg.K}")
    print(f"Mask reconfig: {cfg.RECONFIG_STEPS} steps @ lr={cfg.RECONFIG_LR}")
    print()

    # ── Determine max action space ──────────────────────────────────
    print("Scanning action spaces...")
    action_sizes = {}
    for game in cfg.GAME_SEQUENCE:
        n = get_action_space_size(game, cfg)
        action_sizes[game] = n
        print(f"  {game}: {n} actions")
    max_actions = max(action_sizes.values())
    print(f"Max actions: {max_actions}\n")

    # ── Create model ────────────────────────────────────────────────
    if cfg.USE_MINATAR:
        pad_channels = get_minatar_max_channels(cfg.GAME_SEQUENCE)
        cfg._pad_channels = pad_channels
        print(f"MinAtar max channels: {pad_channels}")
        model = Conv2dDQNMinAtar(
            max_actions=max_actions,
            hidden_size=cfg.HIDDEN_SIZE,
            in_channels=pad_channels,
            square_dim=cfg.SQUARE_DIM,
            **cfg.MODEL_PARS,
        )
    else:
        model = Conv2dDQNMinAtar(
            max_actions=max_actions,
            hidden_size=cfg.HIDDEN_SIZE,
            frame_stack=cfg.FRAME_STACK,
            img_size=cfg.IMG_SIZE,
            square_dim=cfg.SQUARE_DIM,
            **cfg.MODEL_PARS,
        )
    target_model = copy.deepcopy(model)
    target_model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters")

    # ── Buffers ─────────────────────────────────────────────────────
    mask_replay = MaskReplayBuffer(max_per_game=cfg.N_SAMPLE_PER_GAME)

    # ── Training loop over games ────────────────────────────────────
    game_masks = {}  # {game_name: mask}
    all_losses = {}  # {game_name: [loss_values]}
    eval_history = []  # [(game_idx, {game: reward}, cycle)]
    train_history = {}  # {game_name: {cycle: mean_reward}}
    null_baselines = {}  # {game_name: early_reward} from first cycle

    for cycle in range(cfg.CYCLES):
        for game_idx, game_name in enumerate(cfg.GAME_SEQUENCE):
            print(f"\n{'=' * 60}")
            print(f"GAME {game_idx + 1}/{cfg.N_GAMES}: {game_name}")
            print(f"{'=' * 60}")

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
            )

            # ── Discover mask for this game ─────────────────────────────
            print(f"  Discovering mask for {game_name}...")
            env_tmp = make_env(
                game_name,
                cfg,
                seed=None,  # cfg.SEED + game_idx,
                pad_channels=getattr(cfg, "_pad_channels", None),
            )
            current_mask = torch.zeros(cfg.HIDDEN_SIZE)
            mask = reconfigure_mask_rl_iterate(
                model, target_model, None, env_tmp, cfg, current_mask
            )
            env_tmp.close()
            game_masks[game_name] = mask
            active = mask.sum().item()
            print(f"  Mask: {active:.0f}/{cfg.HIDDEN_SIZE} neurons active")

            # ── Train on this game ──────────────────────────────────────
            replay_buf = ReplayBuffer(capacity=cfg.REPLAY_CAPACITY)
            model.train()

            t0 = time.time()
            losses, mean_train_reward, early_reward = train_game(
                model,
                target_model,
                optimizer,
                game_name,
                game_idx,
                mask,
                cfg,
                replay_buf,
                mask_replay,
                outdir,
            )
            dt = time.time() - t0
            all_losses[game_name] = losses
            train_history.setdefault(game_name, {})[cycle] = mean_train_reward
            if cycle == 0:
                null_baselines[game_name] = early_reward
            print(
                f"  {game_name} done in {dt:.0f}s, {len(losses)} training steps, mean reward={mean_train_reward:.1f}"
            )

        # ── Re-discover mask after training (may have shifted) ──────
        for eval_cycle in range(cfg.EVAL_CYCLES):
            for game_idx_re, game_name_re in enumerate(cfg.GAME_SEQUENCE):
                if cfg.RECONFIG_STEPS > 0:
                    env_tmp_re = make_env(
                        game_name_re,
                        cfg,
                        seed=None,  # cfg.SEED + game_idx + 1000,
                        pad_channels=getattr(cfg, "_pad_channels", None),
                    )
                    mask_re = reconfigure_mask_rl_iterate(
                        model, target_model, None, env_tmp_re, cfg, mask
                    )
                    env_tmp_re.close()
                    game_masks[game_name_re] = mask_re
                    # print(f"  Post-training mask: {mask_re.sum().item():.0f} active neurons")

                # ── Periodic evaluation on all learned games ────────────────
            print(f"\n  ── Evaluating retention ({len(game_masks)} games) ──")
            eval_results = evaluate_all_games(model, game_masks, cfg)
            eval_history.append((eval_results, cycle))

        # ── Save plots ──────────────────────────────────────────────
        save_training_plot(all_losses, outdir)
        save_eval_plot(
            eval_history,
            outdir,
            train_history=train_history,
            null_baselines=null_baselines,
        )
        save_mask_visualization(game_masks, cfg.SQUARE_DIM, outdir)

        # ── Checkpoint ──────────────────────────────────────────────
        if (game_idx + 1) % cfg.CHECKPOINT_EVERY_GAMES == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "game_masks": game_masks,
                "game_idx": game_idx,
                "eval_history": eval_history,
                "config": {
                    k: v
                    for k, v in vars(cfg.__class__).items()
                    if not k.startswith("_") and not callable(v)
                },
            }
            torch.save(ckpt, os.path.join(outdir, f"checkpoint_game{game_idx + 1}.pt"))
            print("  Checkpoint saved")

    # ── Final summary ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 60}")

    if eval_history:
        print("\nFinal evaluation:")
        _, final_results, _ = eval_history[-1]
        for game, reward in final_results.items():
            print(f"  {game:20s}: {reward:.1f}")

    print("\nMask allocation:")
    for game, mask in game_masks.items():
        print(f"  {game:20s}: {mask.sum().item():.0f}/{cfg.HIDDEN_SIZE} active")

    # Compute mask overlaps
    names = list(game_masks.keys())
    if len(names) > 1:
        print("\nMask overlaps (fraction of shared active neurons):")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                mi, mj = game_masks[names[i]], game_masks[names[j]]
                overlap = (mi * mj).sum().item()
                union = ((mi + mj) > 0).sum().item()
                iou = overlap / max(union, 1)
                print(
                    f"  {names[i]:15s} ∩ {names[j]:15s}: {overlap:.0f} neurons (IoU={iou:.2f})"
                )

    print(f"\nOutputs saved to {outdir}/")


def single_game(game_name, use_minatar=False):
    """
    Train vanilla DQN on a single game with all-ones mask (no masking).
    Quick sanity check that the base DQN learns.
    """
    cfg = AtariConfig()
    if use_minatar:
        cfg.enable_minatar()

    torch.set_default_device("cuda")
    if cfg.SEED is not None:
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)

    outdir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(outdir, exist_ok=True)

    n_actions = get_action_space_size(game_name, cfg)
    mode_label = "MinAtar" if cfg.USE_MINATAR else "Atari"
    print(f"Single-game baseline ({mode_label}): {game_name} ({n_actions} actions)")
    print(f"Steps: {cfg.MAX_STEPS_PER_GAME:,}")

    if cfg.USE_MINATAR:
        from env_utils import get_minatar_max_channels

        pad_channels = get_minatar_max_channels([game_name])
        cfg._pad_channels = pad_channels
        model = Conv2dDQNMinAtar(
            max_actions=n_actions,
            hidden_size=cfg.HIDDEN_SIZE,
            in_channels=pad_channels,
            square_dim=cfg.SQUARE_DIM,
            **cfg.MODEL_PARS,
        )
    else:
        model = Conv2dDQNMinAtar(
            max_actions=n_actions,
            hidden_size=cfg.HIDDEN_SIZE,
            frame_stack=cfg.FRAME_STACK,
            img_size=cfg.IMG_SIZE,
            square_dim=cfg.SQUARE_DIM,
            **cfg.MODEL_PARS,
        )
    target_model = copy.deepcopy(model)
    target_model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
    )

    # All-ones mask = every neuron active, no masking
    mask = torch.ones(cfg.HIDDEN_SIZE)
    replay_buf = ReplayBuffer(capacity=cfg.REPLAY_CAPACITY)
    mask_replay = MaskReplayBuffer(max_per_game=0)  # disabled

    model.train()
    t0 = time.time()
    losses, mean_train_reward, _early_reward = train_game(
        model,
        target_model,
        optimizer,
        game_name,
        0,
        mask,
        cfg,
        replay_buf,
        mask_replay,
        outdir,
    )
    dt = time.time() - t0
    print(
        f"Training done in {dt:.0f}s, {len(losses)} gradient steps, mean reward={mean_train_reward:.1f}"
    )

    # Evaluate
    model.eval()
    mean_r = evaluate_game(model, game_name, mask, cfg, n_episodes=10)
    print(f"Eval (10 episodes): mean reward = {mean_r:.1f}")

    save_training_plot({game_name: losses}, outdir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="Train vanilla DQN on a single game (e.g. --single Pong)",
    )
    parser.add_argument(
        "--minatar",
        action="store_true",
        help="Use MinAtar environments instead of full Atari",
    )
    args = parser.parse_args()

    if args.single:
        single_game(args.single, use_minatar=args.minatar)
    else:
        if args.minatar:
            cfg = AtariConfig()
            cfg.enable_minatar()
            main(cfg)
        else:
            main()
