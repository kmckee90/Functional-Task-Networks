"""Train the conv2d-reservoir RECODE explorer on globo, saving GIF artifacts.

RECODE provides the intrinsic (exploration) reward computed over the downscaled
images themselves; globo's in-game reward is optional (env_reward_coef, default 0).

Run a small smoke test:
    python train.py --smoke

Typical small run:
    python train.py --episodes 20 --max-steps 300 --obs-width 64 --obs-height 32
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch

from agent import GloboExplorerDQN
from env import make_globo_env, obs_to_chw

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.join(HERE, "artifacts")


def save_gif(frames, path, upscale=6, duration=0.08):
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    big = [np.kron(f[:, :, :3], np.ones((upscale, upscale, 1), dtype=np.uint8)) for f in frames]
    imageio.mimsave(path, big, duration=duration, loop=0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--obs-width", type=int, default=64)
    p.add_argument("--obs-height", type=int, default=32)
    p.add_argument("--init-frames", type=int, default=200)
    # RECODE / downscaling (lower these if OOM)
    p.add_argument("--recode-down-h", type=int, default=16)
    p.add_argument("--recode-down-w", type=int, default=32)
    p.add_argument("--recode-rgb", action="store_true", help="cluster over RGB (default grayscale)")
    p.add_argument("--recode-memory", type=int, default=2000)
    # RL
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--updates-per-step", type=int, default=1)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--env-reward-coef", type=float, default=0.0)
    p.add_argument("--exploration-coef", type=float, default=1.0)
    p.add_argument("--epsilon-decay", type=int, default=20000)
    p.add_argument("--gif-every", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run (2 eps, 40 steps)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke:
        args.episodes, args.max_steps, args.init_frames = 2, 40, 100
        args.warmup, args.gif_every, args.epsilon_decay = 20, 1, 400

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(ARTIFACTS, exist_ok=True)

    print(f"[setup] device={args.device} obs={args.obs_width}x{args.obs_height} "
          f"recode_down={args.recode_down_h}x{args.recode_down_w} "
          f"grayscale={not args.recode_rgb}")

    env = make_globo_env(
        obs_width=args.obs_width, obs_height=args.obs_height,
        max_timesteps=args.max_steps, init_frames=args.init_frames, seed=args.seed,
    )
    num_actions = env.action_space.n
    print(f"[env] action_space={env.action_space} (DQN over {num_actions} actions)")

    agent = GloboExplorerDQN(
        num_actions=num_actions, in_h=args.obs_height, in_w=args.obs_width,
        recode_down_h=args.recode_down_h, recode_down_w=args.recode_down_w,
        recode_grayscale=not args.recode_rgb, recode_memory_size=args.recode_memory,
        gamma=args.gamma, lr=args.lr, minibatch_size=args.batch,
        env_reward_coef=args.env_reward_coef, exploration_coef=args.exploration_coef,
        epsilon_decay_steps=args.epsilon_decay, device=args.device, seed=args.seed,
    )
    print(f"[agent] reservoir hidden {agent.hidden_shape}; feedback_size={agent.feedback_size}")

    global_step = 0
    t_start = time.time()
    for ep in range(args.episodes):
        obs, info = env.reset()
        agent.reset_hidden()
        img = obs_to_chw(obs, device=args.device)
        novelty = agent.intrinsic_reward(img)
        z = agent.encode(img, novelty)
        agent.update_density(img)

        frames = [np.asarray(obs)]
        ep_intr = 0.0
        ep_env = 0.0
        ep_loss = []
        record = (ep % args.gif_every == 0) or (ep == args.episodes - 1)

        for t in range(args.max_steps):
            action = agent.act(z)
            next_obs, env_r, term, trunc, info = env.step(action)
            next_img = obs_to_chw(next_obs, device=args.device)

            intr = agent.intrinsic_reward(next_img)
            reward = args.exploration_coef * float(intr) + args.env_reward_coef * float(env_r)
            novelty_next = intr
            z_next = agent.encode(next_img, novelty_next)
            agent.update_density(next_img)

            done = bool(term)
            agent.remember(z, action, reward, z_next, done)

            if record:
                frames.append(np.asarray(next_obs))
            ep_intr += float(intr)
            ep_env += float(env_r)

            if agent.memory.size >= max(args.warmup, args.batch):
                for _ in range(args.updates_per_step):
                    loss = agent.replay()
                    if loss is not None:
                        ep_loss.append(loss)

            agent.step_schedule()
            z = z_next
            global_step += 1
            if term or trunc:
                break

        mem = agent.metrics
        msg = (f"[ep {ep:3d}] steps={t+1} intr_sum={ep_intr:.3f} env_sum={ep_env:.3f} "
               f"eps={agent.epsilon:.3f} recode_atoms={int(getattr(agent.density,'size',0))} "
               f"loss={np.mean(ep_loss):.4f}" if ep_loss else
               f"[ep {ep:3d}] steps={t+1} intr_sum={ep_intr:.3f} eps={agent.epsilon:.3f} "
               f"recode_atoms={int(getattr(agent.density,'size',0))} (no updates yet)")
        print(msg, flush=True)

        if record and len(frames) > 1:
            gif = os.path.join(ARTIFACTS, f"globo_explorer_ep{ep:03d}.gif")
            save_gif(frames, gif)
            print(f"        saved {os.path.relpath(gif, HERE)} ({len(frames)} frames)", flush=True)

    env.close()
    print(f"[done] {args.episodes} episodes in {time.time()-t_start:.1f}s; artifacts in {ARTIFACTS}")


if __name__ == "__main__":
    main()
