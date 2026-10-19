"""Single-lifetime training loop for the GPU globo explorer (JAX + warpcraft).

Mirrors ../globo_explorer/train.py: one continuous lifetime, decoupled
collection and learning ("K updates every J steps"), GIF peeks, checkpoints.
Everything on-device; the env is warpcraft globo at num_worlds=1.

  uv run python train.py --config default.yaml
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config(path: str) -> dict:
    with open(os.path.join(HERE, "default.yaml")) as f:
        cfg = yaml.safe_load(f)
    if path and os.path.abspath(path) != os.path.join(HERE, "default.yaml"):
        with open(path) as f:
            overlay = yaml.safe_load(f) or {}
        unknown = set(overlay) - set(cfg)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg.update(overlay)
    return cfg


def build(cfg, key):
    from env import GloboGPUEnv
    from agent import GloboExplorerBranchingDQN
    env = GloboGPUEnv(
        seed=cfg["seed"], obs_width=cfg["obs_width"], obs_height=cfg["obs_height"],
        render_width=cfg["render_width"], render_height=cfg["render_height"],
        recode_down_h=cfg["recode_down_h"], recode_down_w=cfg["recode_down_w"],
        recode_grayscale=not cfg["recode_rgb"], active_channels=tuple(cfg["active_channels"]),
        max_episode_steps=cfg["max_episode_steps"], advance_weather=cfg["advance_weather"])
    agent = GloboExplorerBranchingDQN(cfg, env.head_sizes, key)
    return env, agent


def save_checkpoint(agent, cfg, gstep, path):
    blob = {
        "params": jax.tree_util.tree_map(np.asarray, agent.params),
        "target": jax.tree_util.tree_map(np.asarray, agent.target),
        "density": {k: np.asarray(v) for k, v in agent.density.get_state().items()},
        "epsilon": agent.epsilon, "cur_step": agent.cur_step,
        "config": cfg, "global_step": gstep,
    }
    with open(path, "wb") as f:
        pickle.dump(blob, f)


def write_gif(frames, path, fps=20):
    import imageio
    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "default.yaml"))
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    name = args.name or cfg["name"]
    art = os.path.join(HERE, "artifacts")
    os.makedirs(art, exist_ok=True)

    key = jax.random.PRNGKey(cfg["seed"])
    key, bk = jax.random.split(key)
    env, agent = build(cfg, bk)
    print(f"[setup] heads={env.head_sizes}{env.head_names} obs={cfg['obs_width']}x{cfg['obs_height']} "
          f"render={cfg['render_width']}x{cfg['render_height']} emb={agent.emb_dim} "
          f"hidden={agent.hidden_shape}", flush=True)
    print(f"[setup] intrinsic gamma={cfg['gamma']} survival gamma_env={cfg['gamma_env']} "
          f"env_coef={cfg['env_reward_coef']} death_penalty={cfg['death_penalty']}; "
          f"train {cfg['train_iters']} updates every {cfg['train_interval']} steps", flush=True)

    online_coef, env_coef = jnp.float32(cfg["online_coef"]), jnp.float32(cfg["env_reward_coef"])
    zero_act = jnp.zeros((agent.n_heads,), jnp.int32)
    zero = jnp.float32(0.0)

    hidden = agent.init_hidden()
    obs = env.reset()
    key, pk = jax.random.split(key)
    # First perception (no previous action/reward/novelty).
    z, action, emb, nov = agent.perceive(hidden, obs["image"], zero_act, zero, zero, pk)

    total = cfg["total_steps"] or (1 << 62)
    gstep, t0 = 0, time.time()
    peek_frames, peeking = [], False
    while gstep < total:
        # --- foreign GPU env step (only non-JAX call in the hot loop) ---
        obs2, env_r, terminal, timeout = env.step(action)
        boundary = jnp.maximum(terminal, timeout)                       # device
        r_fb = online_coef * nov + env_coef * env_r                     # device (no host sync)
        # functional recurrence reset on episode boundary (no host branch)
        hidden_carry = jnp.where(boundary > 0.5, jnp.zeros_like(z), z)
        key, pk, dk = jax.random.split(key, 3)
        z2, action2, emb2, nov2 = agent.perceive(hidden_carry, obs2["image"], action, r_fb, nov, pk)

        agent.remember(z[0], z2[0], emb, action, env_r, nov, terminal, boundary)
        agent.update_density(emb, dk)
        agent.step_schedule()

        # GIF peek window (only host copies during an active window)
        if (not peeking) and cfg["render_interval"] and gstep % cfg["render_interval"] == 0 and gstep > 0:
            peeking, peek_frames = True, []
        if peeking:
            peek_frames.append((np.asarray(obs2["render_rgb"][0]) * 255).astype(np.uint8))
            if len(peek_frames) >= cfg["render_window"]:
                stride = max(1, len(peek_frames) // cfg["render_max_frames"])
                write_gif(peek_frames[::stride], os.path.join(art, f"{name}_s{gstep:08d}.gif"))
                save_checkpoint(agent, cfg, gstep, os.path.join(art, f"{name}_ckpt.pkl"))
                peeking, peek_frames = False, []

        obs, z, action, emb, nov = obs2, z2, action2, emb2, nov2
        gstep += 1
        if gstep % cfg["train_interval"] == 0:
            key, tk = jax.random.split(key)
            loss = agent.train_round(cfg["train_iters"], tk)   # host sync only here (logging)
            rate = gstep / (time.time() - t0)
            print(f"[step {gstep:8d}] eps={agent.epsilon:.3f} recode={agent.density.size} "
                  f"nov={float(nov):.4f} loss={loss if loss is None else round(loss, 4)} "
                  f"({rate:.1f} steps/s)", flush=True)


if __name__ == "__main__":
    main()
