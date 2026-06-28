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

All run parameters come from a YAML config (see default.yaml); the CLI flags are
--config (the YAML to run), --name (a prefix for all output filenames), and --resume
(a checkpoint .pt to continue training from).

Examples:
    python train.py                         # uses default.yaml
    python train.py --config smoke.yaml     # tiny end-to-end smoke run
    python train.py --config my_run.yaml    # overlays default.yaml with your keys
    python train.py --name run1             # outputs prefixed: run1_globo_*.gif, run1_*.png
    python train.py --resume artifacts/checkpoints/run1_ckpt_s0050000.pt --name run1
                                            # continue run1 from its checkpoint (its config is
                                            # the base; add --config to overlay tweaks like total_steps)
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
from env import downsample_obs, make_globo_env, obs_to_chw, resolve_render_size

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")
CENTROIDS = os.path.join(ARTIFACTS, "centroids")
CHECKPOINTS = os.path.join(ARTIFACTS, "checkpoints")


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

    def __init__(self, window, interval, gif_seconds, max_frames=200, min_ms=30, max_ms=250, prefix=""):
        self.window = window                    # number of collected steps a GIF spans
        self.interval = interval
        self.gif_seconds = gif_seconds          # >0: fixed playback; else real-time
        self.prefix = prefix                    # filename prefix for the saved GIFs
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
        path = os.path.join(ARTIFACTS, f"{self.prefix}globo_ep{ep:03d}_s{self.start_step:07d}.gif")
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
    """CLI args: --config (the YAML config to run), --name (an optional output-filename
    prefix), and --resume (a checkpoint .pt to continue training from). Every run parameter
    lives in the YAML. `default.yaml` defines the complete parameter set; any other `--config`
    is overlaid on top of it, so a custom file only needs the keys it changes.

    With --resume, the CHECKPOINT's own config is used as the base (so the architecture matches
    and the weights load), and --config (if given) is overlaid on top for tweaks (e.g. a larger
    total_steps). Returns (args, resume_ckpt) where resume_ckpt is the loaded checkpoint or None."""
    default_cfg = os.path.join(HERE, "default.yaml")
    p = argparse.ArgumentParser(description="Train the globo explorer. All parameters come from the YAML config.")
    p.add_argument("--config", type=str, default=default_cfg,
                   help="YAML/JSON config with run parameters (see default.yaml); defaults to default.yaml")
    p.add_argument("--name", type=str, default="",
                   help="prefix for all output filenames (gif, centroids, masses), e.g. --name run1")
    p.add_argument("--resume", type=str, default="",
                   help="path to a checkpoint .pt to continue training from (restores weights, "
                        "optimizer, RECODE density, epsilon schedule and step count; the replay "
                        "buffer is re-warmed). The checkpoint's config is the base; --config overlays it.")
    cli = p.parse_args()

    resume_ckpt = None
    if cli.resume:
        resume_ckpt = torch.load(cli.resume, map_location="cpu", weights_only=False)
        base = dict(resume_ckpt["config"])      # continue with the architecture/params it trained with
    else:
        base = _load_config(default_cfg)        # the full parameter set + default values
    cfg = dict(base)
    if os.path.abspath(cli.config) != os.path.abspath(default_cfg):
        over = _load_config(cli.config)
        unknown = set(over) - set(base)
        if unknown:
            raise SystemExit(f"Unknown config keys: {sorted(unknown)}. Known: {sorted(base)}")
        cfg.update(over)
    cfg["config"] = cli.config
    cfg["resume"] = cli.resume
    # filename prefix for all outputs (empty => unchanged); "run1" => "run1_…"
    cfg["name"] = f"{cli.name}_" if cli.name else ""
    return SimpleNamespace(**cfg), resume_ckpt


def build_agent(args, env):
    """Construct the agent from a config namespace + a built env. Shared by training
    (train.py) and the demo (play.py --agent) so both build the identical model."""
    factored = args.action_mode == "factored"
    action_groups = [int(n) for n in env.action_space.nvec] if factored else None
    return GloboExplorerDQN(
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
        gamma=args.gamma, gamma_env=args.gamma_env, death_penalty=args.death_penalty,
        lr=args.lr, grad_clip=args.grad_clip,
        minibatch_size=args.batch, memory_length=args.replay_capacity,
        target_update_tau=args.target_update_tau, target_update_tau_env=args.target_update_tau_env,
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


def save_checkpoint(agent, args, global_step, prefix=""):
    """Save the agent's model + a copy of the config it trained with. Writes a
    self-contained <prefix>ckpt_s<step>.pt (model + embedded config) plus a
    human-readable <prefix>ckpt_s<step>.yaml sidecar. The .pt is all the demo needs."""
    import yaml
    os.makedirs(CHECKPOINTS, exist_ok=True)
    # config as a plain dict of just the run parameters (drop the CLI-only keys),
    # which is exactly the default.yaml key set -> reusable as a --config overlay.
    cfg = {k: v for k, v in vars(args).items() if k not in ("config", "name", "resume")}
    path = os.path.join(CHECKPOINTS, f"{prefix}ckpt.pt")
    torch.save({"agent": agent.state_dict(), "config": cfg, "global_step": global_step}, path)
    with open(path.replace(".pt", ".yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def main():
    args, resume_ckpt = load_config()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(ARTIFACTS, exist_ok=True)

    render_w, render_h = resolve_render_size(args.obs_width, args.obs_height,
                                             args.render_width, args.render_height)
    downscaling = (render_w, render_h) != (args.obs_width, args.obs_height)
    print(f"[setup] device={args.device} obs={args.obs_width}x{args.obs_height} "
          f"render={render_w}x{render_h}{' (downsampled->obs)' if downscaling else ' (==obs)'} "
          f"recode_down={args.recode_down_h}x{args.recode_down_w} grayscale={not args.recode_rgb} "
          f"invincible={args.invincible}")
    print(f"[setup] action_mode={args.action_mode}; train: {args.train_iters} updates every "
          f"{args.train_interval} steps; peek: {args.render_window} steps every {args.render_interval} steps")
    print(f"[setup] reward streams: intrinsic gamma={args.gamma} tau={args.target_update_tau}; "
          f"environment gamma_env={args.gamma_env} tau_env={args.target_update_tau_env} "
          f"env_reward_coef={args.env_reward_coef} death_penalty={args.death_penalty}")

    max_timesteps = None if args.max_steps <= 0 else args.max_steps
    n_episodes = None if args.episodes <= 0 else args.episodes
    print(f"[setup] seed={args.seed} spawn={args.spawn} weather_seed={args.weather_seed} "
          f"freeze_daylight={args.freeze_daylight} invincible={args.invincible} "
          f"episodes={'inf' if n_episodes is None else n_episodes} "
          f"max_steps={'none' if max_timesteps is None else max_timesteps}")

    def build_env():
        return make_globo_env(
            obs_width=render_w, obs_height=render_h,   # engine renders at the (possibly higher) render size
            max_timesteps=max_timesteps, init_frames=args.init_frames, seed=args.seed,
            action_mode=args.action_mode,
            invincible=args.invincible, freeze_daylight=args.freeze_daylight,
            clear_sky=getattr(args, "clear_sky", False), spawn=args.spawn,
            weather_seed=args.weather_seed,
            frameskip=args.frameskip,
            move_speed=args.move_speed, move_accel=args.move_accel, climb_speed=args.climb_speed,
            jump_speed=args.jump_speed, gravity=args.gravity, mouse_mov=args.mouse_mov,
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

    agent = build_agent(args, env)
    print(f"[agent] reservoir hidden {agent.hidden_shape}; feedback_size={agent.feedback_size}")

    resume_step = 0
    if resume_ckpt is not None:
        agent.load_state_dict(resume_ckpt["agent"])
        resume_step = int(resume_ckpt.get("global_step", 0))
        print(f"[resume] {args.resume} -> step={resume_step} eps={agent.epsilon:.3f} "
              f"cur_step={agent.cur_step} atoms={agent.density.size} "
              f"(weights/optimizer/density restored; replay buffer re-warms from empty)", flush=True)

    rec = PeekRecorder(args.render_window, args.render_interval, args.gif_seconds,
                       max_frames=args.render_max_frames, prefix=args.name)

    global_step = resume_step
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
        img = obs_to_chw(downsample_obs(obs, args.obs_width, args.obs_height), device=args.device)
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

            next_small = downsample_obs(next_obs, args.obs_width, args.obs_height)  # render-size -> agent input
            next_img = obs_to_chw(next_small, device=args.device)
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

            saved = rec.add(next_small, ep, global_step, game_time, dtime)   # GIF peeks = the agent's view
            if saved:
                print(f"        [gif] {os.path.relpath(saved['path'], HERE)} "
                      f"{saved['n']} frames = {saved['gt_span']:.1f}s game-time "
                      f"(~{saved['mean_ms']:.0f} ms/step)", flush=True)
                n = agent.dump_centroids(CENTROIDS, prefix=args.name)
                print(f"        [centroids] wrote {n} -> {os.path.relpath(CENTROIDS, HERE)}/", flush=True)
                ckpt = save_checkpoint(agent, args, global_step, prefix=args.name)
                print(f"        [ckpt] {os.path.relpath(ckpt, HERE)}", flush=True)

            # optional fixed-interval checkpointing, independent of GIF peeks
            if args.checkpoint_interval and global_step % args.checkpoint_interval == 0:
                ckpt = save_checkpoint(agent, args, global_step, prefix=args.name)
                print(f"        [ckpt] {os.path.relpath(ckpt, HERE)}", flush=True)

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

    agent.dump_centroids(CENTROIDS, prefix=args.name)
    ckpt = save_checkpoint(agent, args, global_step, prefix=args.name)
    print(f"[ckpt] final -> {os.path.relpath(ckpt, HERE)}", flush=True)
    env.close()
    sps = global_step / max(1e-9, time.time() - t_start)
    print(f"[done] {global_step} steps over {game_time:.1f}s game-time in "
          f"{time.time()-t_start:.1f}s wall ({sps:.1f} steps/s); artifacts in {ARTIFACTS}")


if __name__ == "__main__":
    main()
