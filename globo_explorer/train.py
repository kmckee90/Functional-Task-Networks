"""Train the conv2d-reservoir RECODE explorer on globo, saving GIF artifacts.

RECODE provides the intrinsic (exploration) reward computed over the downscaled
images themselves; globo's in-game reward is optional (env_reward_coef, default 0).

GIF peeks are taken *within* an episode: every --render-interval collected steps,
the next --render-window steps are recorded into one GIF. Each frame is annotated
with the global step and the cumulative in-game time (seconds), and the GIF plays
back at real game-time speed (from the engine's per-step dtime), so you can read
off the steps<->seconds relationship directly.

Training cadence is decoupled from data collection: every --train-interval (J)
collected steps, the agent does --train-iters (K) gradient updates (cf. the
explorer's per-episode update loop).

All run parameters come from a YAML config (see default.yaml); the only CLI
argument is --config.

Examples:
    python train.py                         # uses default.yaml
    python train.py --config smoke.yaml     # tiny end-to-end smoke run
    python train.py --config my_run.yaml    # overlays default.yaml with your keys
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from agent import GloboExplorerDQN
from env import make_globo_env, obs_to_chw

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")
CENTROIDS = os.path.join(ARTIFACTS, "centroids")


def _annotate(rgb_big, text):
    """Draw a caption strip with `text` under an (upscaled) RGB frame."""
    from PIL import Image, ImageDraw
    h, w, _ = rgb_big.shape
    bar = 16
    canvas = np.zeros((h + bar, w, 3), dtype=np.uint8)
    canvas[:h] = rgb_big
    img = Image.fromarray(canvas)
    ImageDraw.Draw(img).text((2, h + 2), text, fill=(255, 255, 0))
    return np.asarray(img)


def save_gif(frames, captions, durations, path, upscale=6):
    """Save an annotated GIF. `durations` is per-frame seconds (real game-time)."""
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = []
    for f, cap in zip(frames, captions):
        big = np.kron(f[:, :, :3], np.ones((upscale, upscale, 1), dtype=np.uint8))
        out.append(_annotate(big, cap))
    imageio.mimsave(path, out, duration=durations, loop=0)


class PeekRecorder:
    """Records `window` consecutive frames every `interval` collected steps.

    Frames are annotated with global step + cumulative game-time, and the GIF
    plays back at real game-time speed (per-frame dtime, clamped for visibility)
    unless a fixed `gif_seconds` is given.
    """

    def __init__(self, window, interval, gif_seconds, max_frames=200, min_ms=30, max_ms=250):
        self.window = window                    # number of collected steps a GIF spans
        self.interval = interval
        self.gif_seconds = gif_seconds          # >0: fixed playback; else real-time
        self.stride = max(1, round(window / max(1, max_frames)))  # subsample to cap frames
        self.min_s, self.max_s = min_ms / 1000, max_ms / 1000
        self.active = False
        self._reset_buffers()

    def _reset_buffers(self):
        self.frames, self.caps, self.durs = [], [], []
        self.start_step = None
        self.start_gt = 0.0
        self.last_gt = 0.0
        self.n_steps = 0       # steps elapsed in this window
        self._dt_accum = 0.0   # game-time since last kept frame

    def maybe_start(self, collect_step):
        if (not self.active) and self.interval > 0 and collect_step % self.interval == 0:
            self.active = True
            self._reset_buffers()

    def add(self, obs, ep, gstep, game_time, dtime):
        if not self.active:
            return None
        if self.start_step is None:
            self.start_step, self.start_gt = gstep, game_time
        self.last_gt = game_time
        self.n_steps += 1
        self._dt_accum += dtime
        if (self.n_steps - 1) % self.stride == 0:   # keep every stride-th frame
            self.frames.append(np.asarray(obs))
            self.caps.append(f"ep{ep} s{gstep} t={game_time:6.1f}s")
            self.durs.append(float(np.clip(self._dt_accum, self.min_s, self.max_s)))
            self._dt_accum = 0.0
        if self.n_steps >= self.window:
            return self._flush(ep)
        return None

    def _flush(self, ep):
        n = len(self.frames)
        gt_span = max(0.0, self.last_gt - self.start_gt)        # real in-game seconds covered
        durs = [self.gif_seconds / n] * n if self.gif_seconds > 0 else self.durs
        path = os.path.join(ARTIFACTS, f"globo_ep{ep:03d}_s{self.start_step:07d}.gif")
        save_gif(self.frames, self.caps, durs, path)
        info = dict(path=path, n=n, start=self.start_step, gt_span=gt_span,
                    mean_ms=1000 * gt_span / max(1, n))
        self.active = False
        self._reset_buffers()
        return info


def _load_config(path):
    """Load a YAML/JSON config; returns {dest: value}. Keys may use dashes or
    underscores and must match CLI option names (e.g. max-steps / max_steps)."""
    import yaml  # available via craftium; safe_load also parses JSON
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config {path} must be a mapping of option:value")
    return {str(k).replace("-", "_"): v for k, v in cfg.items()}


def load_config():
    """Single CLI argument: the YAML config to run. Every run parameter lives in
    the YAML — there are no other CLI flags. `default.yaml` defines the complete
    parameter set; any other `--config` is overlaid on top of it, so a custom file
    only needs the keys it changes."""
    default_cfg = os.path.join(HERE, "default.yaml")
    p = argparse.ArgumentParser(description="Train the globo explorer. All parameters come from the YAML config.")
    p.add_argument("--config", type=str, default=default_cfg,
                   help="YAML/JSON config with run parameters (see default.yaml); defaults to default.yaml")
    cli = p.parse_args()

    base = _load_config(default_cfg)            # the full parameter set + default values
    cfg = dict(base)
    if os.path.abspath(cli.config) != os.path.abspath(default_cfg):
        over = _load_config(cli.config)
        unknown = set(over) - set(base)
        if unknown:
            raise SystemExit(f"Unknown config keys: {sorted(unknown)}. Known: {sorted(base)}")
        cfg.update(over)
    cfg["config"] = cli.config
    return SimpleNamespace(**cfg)


def main():
    args = load_config()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(ARTIFACTS, exist_ok=True)

    print(f"[setup] device={args.device} obs={args.obs_width}x{args.obs_height} "
          f"recode_down={args.recode_down_h}x{args.recode_down_w} grayscale={not args.recode_rgb} "
          f"invincible={args.invincible}")
    print(f"[setup] action_mode={args.action_mode}; train: {args.train_iters} updates every "
          f"{args.train_interval} steps; peek: {args.render_window} steps every {args.render_interval} steps")

    max_timesteps = None if args.max_steps <= 0 else args.max_steps
    n_episodes = None if args.episodes <= 0 else args.episodes
    print(f"[setup] seed={args.seed} spawn={args.spawn} weather_seed={args.weather_seed} "
          f"freeze_daylight={args.freeze_daylight} invincible={args.invincible} "
          f"episodes={'inf' if n_episodes is None else n_episodes} "
          f"max_steps={'none' if max_timesteps is None else max_timesteps}")

    def build_env():
        return make_globo_env(
            obs_width=args.obs_width, obs_height=args.obs_height,
            max_timesteps=max_timesteps, init_frames=args.init_frames, seed=args.seed,
            action_mode=args.action_mode,
            invincible=args.invincible, freeze_daylight=args.freeze_daylight, spawn=args.spawn,
            weather_seed=args.weather_seed,
        )

    env = build_env()
    factored = args.action_mode == "factored"
    if factored:
        action_groups = [int(n) for n in env.action_space.nvec]
        print(f"[env] factored action_space={env.action_space} groups={env.group_names} "
              f"sizes={action_groups}")
    else:
        action_groups = None
        print(f"[env] action_space={env.action_space} (DQN over {env.action_space.n} actions)")

    # Graceful shutdown: close the env (kills the luanti subprocess) on SIGTERM/SIGINT.
    def _shutdown(signum, _frame):
        print(f"\n[signal {signum}] shutting down: closing env...", flush=True)
        try:
            env.close()
        except Exception:
            pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    agent = GloboExplorerDQN(
        num_actions=(None if factored else env.action_space.n),
        action_groups=action_groups, in_h=args.obs_height, in_w=args.obs_width,
        reservoir_channels=args.reservoir_channels, decoder_layer_size=args.decoder_layer_size,
        num_novelty_bins=args.num_novelty_bins,
        recode_down_h=args.recode_down_h, recode_down_w=args.recode_down_w,
        recode_grayscale=not args.recode_rgb, recode_memory_size=args.recode_memory,
        recode_k=args.recode_k, recode_knn_k=args.recode_knn_k,
        recode_kappa=args.recode_kappa, recode_decay=args.recode_decay,
        recode_density_gamma=args.recode_gamma,
        recode_insertion_probability=args.recode_insertion_probability,
        gamma=args.gamma, lr=args.lr, grad_clip=args.grad_clip,
        minibatch_size=args.batch, memory_length=args.replay_capacity,
        target_update_tau=args.target_update_tau,
        epsilon=args.epsilon_start, epsilon_end=args.epsilon_end, epsilon_decay_steps=args.epsilon_decay,
        env_reward_coef=args.env_reward_coef,
        online_coef=args.online_coef, offline_coef=args.offline_coef,
        reservoir_kernel=args.reservoir_kernel, reservoir_stride=args.reservoir_stride,
        reservoir_recur_kernel=args.reservoir_recur_kernel,
        reservoir_input_scale=args.reservoir_input_scale, reservoir_recur_scale=args.reservoir_recur_scale,
        reservoir_tc=args.reservoir_tc, reservoir_sparsity=args.reservoir_sparsity,
        use_online_novelty=args.use_online_novelty, use_offline_novelty=args.use_offline_novelty,
        main_sparse_reward_scale=args.main_sparse_reward_scale,
        goal_sparse_reward_scale=args.goal_sparse_reward_scale,
        goal_memory_length=args.goal_memory_length, prioritized_replay=args.prioritized_replay,
        offline_scan_max=args.offline_scan_max,
        device=args.device, seed=args.seed,
    )
    print(f"[agent] reservoir hidden {agent.hidden_shape}; feedback_size={agent.feedback_size}")

    rec = PeekRecorder(args.render_window, args.render_interval, args.gif_seconds,
                       max_frames=args.render_max_frames)

    global_step = 0
    game_time = 0.0           # cumulative in-game seconds
    steps_since_train = 0
    recent_loss = []
    t_start = time.time()

    ep = 0
    while n_episodes is None or ep < n_episodes:
        # Rebuild a pristine env each restart so terrain (seed), spawn location, and
        # player state don't accumulate across deaths => identical world every time.
        if ep > 0:
            env.close()
            env = build_env()
        obs, info = env.reset(seed=args.seed)
        agent.reset_hidden()
        img = obs_to_chw(obs, device=args.device)
        novelty = agent.intrinsic_reward(img)
        z = agent.encode(img, novelty)
        agent.update_density(img)

        ep_intr = 0.0
        t = 0
        while True:
            rec.maybe_start(global_step)

            action = agent.act(z)   # list of per-head indices (len 1 if discrete)
            env_action = action if factored else int(action[0])
            next_obs, env_r, term, trunc, info = env.step(env_action)
            dtime = float(info.get("mt_dtime", 0.0) or 0.0)
            game_time += dtime

            next_img = obs_to_chw(next_obs, device=args.device)
            intr = agent.intrinsic_reward(next_img)          # online novelty
            emb_next = agent.embed(next_img)                 # stored for offline novelty
            reward = args.online_coef * float(intr) + args.env_reward_coef * float(env_r)
            # Record the action/reward that produced next_img BEFORE encoding it, so the
            # reservoir's feedback channel carries a_t/r_t with the state they led to.
            agent.note_action(action, reward)
            z_next = agent.encode(next_img, intr)
            agent.update_density(next_img)
            # boundary marks any episode reset (death or truncation) for trajectory
            # segmentation; terminal (death only) controls TD bootstrapping.
            agent.remember(z, action, float(env_r), float(intr), emb_next, z_next,
                           terminal=bool(term), boundary=bool(term or trunc))

            saved = rec.add(next_obs, ep, global_step, game_time, dtime)
            if saved:
                print(f"        [gif] {os.path.relpath(saved['path'], HERE)} "
                      f"{saved['n']} frames = {saved['gt_span']:.1f}s game-time "
                      f"(~{saved['mean_ms']:.0f} ms/step)", flush=True)
                n = agent.dump_centroids(CENTROIDS)
                print(f"        [centroids] wrote {n} -> {os.path.relpath(CENTROIDS, HERE)}/", flush=True)

            ep_intr += float(intr)

            # offline reward + goal/frontier prioritized replay: K updates every J steps
            steps_since_train += 1
            if steps_since_train >= args.train_interval and agent.memory.size >= max(args.warmup, args.batch):
                steps_since_train = 0
                loss = agent.train_round(args.train_iters)
                if loss is not None:
                    recent_loss.append(loss)

            agent.step_schedule()
            z = z_next
            global_step += 1
            t += 1

            if global_step % 1000 == 0:
                lo = np.mean(recent_loss[-200:]) if recent_loss else float("nan")
                print(f"[step {global_step:7d}] ep{ep} t_game={game_time:7.1f}s eps={agent.epsilon:.3f} "
                      f"recode_atoms={int(getattr(agent.density,'size',0))} goal_buf={int(agent.goal.size)} "
                      f"loss={lo:.4f} ({global_step/max(1e-9,time.time()-t_start):.1f} steps/s)", flush=True)

            if (args.total_steps and global_step >= args.total_steps) or term or trunc:
                break

        ended = "DIED" if term else ("total_steps" if (args.total_steps and global_step >= args.total_steps) else "trunc")
        print(f"[ep {ep:3d}] steps={t} intr_sum={ep_intr:.3f} t_game={game_time:.1f}s "
              f"eps={agent.epsilon:.3f} {ended}", flush=True)
        ep += 1
        if args.total_steps and global_step >= args.total_steps:
            break

    agent.dump_centroids(CENTROIDS)
    env.close()
    sps = global_step / max(1e-9, time.time() - t_start)
    print(f"[done] {global_step} steps over {game_time:.1f}s game-time in "
          f"{time.time()-t_start:.1f}s wall ({sps:.1f} steps/s); artifacts in {ARTIFACTS}")


if __name__ == "__main__":
    main()
