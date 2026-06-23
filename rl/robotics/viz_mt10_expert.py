"""Visualize a trained MT10 expert-BC FTN model: arm rollout + live FTN mask.

Loads a checkpoint saved by train_mt10_expert.py (`--name ...` -> `*.pt`), then
for each task in the training order rolls out one or more greedy eval episodes
and renders a side-by-side GIF:

    +-----------------------------+-------------------------------+
    |   Meta-World render          |   FTN mask state (EMA)        |
    |   (arm executing the policy) |   blue/purple decay trail     |
    +-----------------------------+-------------------------------+

The mask panel is exactly the visualization from
func_net_spatial/run_experiments.py: the active k-hot mask is laid onto the
square grid (mask_to_grid) and accumulated into a per-channel max-decay EMA so
the currently-active units are white, recently-active units fade through
purple/magenta, and older ones linger in blue before decaying out. The EMA is
continuous across the whole task sequence, so each task switch shows the
previous task's mask fading while the new task's mask lights up.

By default the per-task mask is the one stored at training time
(`stored_m_a[ti]`); `--mask-source reconfig` instead recovers each task's mask
at eval from its expert support (full-FTN protocol), applying the same lateral
spatial smoothing the trainer used.

Example:
    uv run python viz_mt10_expert.py --ckpt results_expert_myrun_ftn_0.pt \
        --episodes 1 --out rollout.gif
"""

from __future__ import annotations

import argparse
import dataclasses
import os

# MuJoCo offscreen rendering: must be set before mujoco / metaworld import.
os.environ.setdefault("MUJOCO_GL", "egl")

import metaworld
import numpy as np
import torch
from ftn import mask_to_grid
from metaworld.policies import ENV_POLICY_MAP  # noqa: F401  (kept for parity)
from PIL import Image, ImageDraw
from train_mt10_expert import (
    BCAgent,
    ExpertBCConfig,
    generate_or_load_demos,
    make_support,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(DEVICE)


# ---------------------------------------------------------------------------
# Render-capable env (StandardTaskEnv + an rgb_array camera)
# ---------------------------------------------------------------------------


class RenderTaskEnv:
    """One MT10 env class with offscreen rendering. Standard reset (random goal
    variant), same obs/act handling as train_mt10_expert.StandardTaskEnv."""

    def __init__(self, name, env_cls, tasks, seed, camera, height, width):
        self.name = name
        self.env = env_cls(
            render_mode="rgb_array", camera_name=camera, height=height, width=width
        )
        self.tasks = tasks
        self.rng = np.random.default_rng(seed)
        self.env.set_task(self.tasks[0])
        obs, _ = self.env.reset()
        self.obs_dim = int(obs.shape[0])
        self.act_dim = int(self.env.action_space.shape[0])

    def reset(self):
        task = self.tasks[self.rng.integers(len(self.tasks))]
        self.env.set_task(task)
        obs, info = self.env.reset()
        return obs.astype(np.float32), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action.astype(np.float32))
        return obs.astype(np.float32), float(r), bool(term), bool(trunc), info

    def render(self) -> np.ndarray:
        # EGL offscreen framebuffers come out vertically flipped; un-flip them.
        return np.flipud(self.env.render()).copy()


def build_render_envs(task_names, seed, camera, height, width):
    mt10 = metaworld.MT10(seed=seed)
    out = []
    for i, name in enumerate(task_names):
        tasks = [t for t in mt10.train_tasks if t.env_name == name]
        out.append(
            RenderTaskEnv(
                name,
                mt10.train_classes[name],
                tasks,
                seed + i + 1,
                camera,
                height,
                width,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Checkpoint -> agent
# ---------------------------------------------------------------------------


def load_agent(ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    valid = {f.name for f in dataclasses.fields(ExpertBCConfig)}
    cfg_kwargs = {k: v for k, v in ck["cfg"].items() if k in valid}
    if "tasks" in cfg_kwargs and isinstance(cfg_kwargs["tasks"], list):
        cfg_kwargs["tasks"] = tuple(cfg_kwargs["tasks"])
    cfg = ExpertBCConfig(**cfg_kwargs)

    agent = BCAgent(ck["obs_dim"], ck["act_dim"], cfg)
    agent.actor.load_state_dict(ck["actor"])
    agent.actor.eval()
    agent.set_normalizer(
        ck["obs_mean"].to(DEVICE, torch.float32),
        ck["obs_std"].to(DEVICE, torch.float32),
    )
    agent.stored_m_a = {
        int(i): m.to(DEVICE, torch.float32) for i, m in ck["stored_m_a"].items()
    }
    return agent, cfg, ck


# ---------------------------------------------------------------------------
# Mask EMA panel (exact recipe from func_net_spatial/run_experiments.py)
# ---------------------------------------------------------------------------


class MaskEMA:
    """Per-channel max-decay EMA of the active mask on the square grid.
    R decays at 0.95, G at 0.30, B at 0.975 -> active=white, recent=purple,
    old=blue (matches run_experiments.py)."""

    def __init__(self, hidden_size: int, square_dim: int):
        self.square_dim = square_dim
        n_blocks = hidden_size // (square_dim * square_dim)
        self.gh, self.gw = square_dim, n_blocks * square_dim
        self.ema = torch.zeros(self.gh, self.gw, 3)

    def update(self, mask: torch.Tensor):
        m2d = mask_to_grid(mask.detach().float(), self.square_dim)
        self.ema[:, :, 0] = torch.max(0.95 * self.ema[:, :, 0], m2d)
        self.ema[:, :, 1] = torch.max(0.30 * self.ema[:, :, 1], m2d)
        self.ema[:, :, 2] = torch.max(0.975 * self.ema[:, :, 2], m2d)

    def image(self) -> np.ndarray:
        return (self.ema.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Frame composition
# ---------------------------------------------------------------------------


def _upscale_nn(img: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(img, np.ones((scale, scale, 1), dtype=np.uint8))


def compose_frame(
    render_rgb: np.ndarray,
    mask_rgb: np.ndarray,
    title: str,
    panel: int,
    bar_h: int = 30,
) -> Image.Image:
    """Left = arm render (panel x panel), right = mask EMA centered in a
    (panel x panel) tile, with a title bar across the top."""
    # Left panel: resize render to panel x panel.
    left = Image.fromarray(render_rgb).resize((panel, panel), Image.BILINEAR)

    # Right panel: scale the mask grid up to fill the panel width, center it.
    gh, gw = mask_rgb.shape[:2]
    scale = max(1, panel // gw)
    m_up = _upscale_nn(mask_rgb, scale)
    m_img = Image.fromarray(m_up)
    right = Image.new("RGB", (panel, panel), (12, 12, 18))
    ox = (panel - m_img.width) // 2
    oy = (panel - m_img.height) // 2
    right.paste(m_img, (max(0, ox), max(0, oy)))

    canvas = Image.new("RGB", (panel * 2, panel + bar_h), (0, 0, 0))
    canvas.paste(left, (0, bar_h))
    canvas.paste(right, (panel, bar_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title, fill=(255, 255, 255))
    draw.text(
        (panel + 6, 8),
        "FTN mask (active=white -> purple -> blue)",
        fill=(180, 180, 255),
    )
    return canvas


# ---------------------------------------------------------------------------
# Main rollout
# ---------------------------------------------------------------------------


def reconfig_mask(agent, demo, cfg):
    """Cold-start recover an actor mask from a task's expert support
    (full-FTN protocol; commit=False so the agent's running state is untouched)."""
    s_obs, s_act = make_support(demo, agent, cfg.reconfig_n)
    agent.m_a_logits = None  # cold start
    mask, _ = agent.reconfig_actor_mask(s_obs, s_act, commit=False)
    return mask


def rollout_episode(env, agent, mask, ema, cfg, args, ti, ep, attempt):
    """Roll out one greedy episode, appending composed frames to a local buffer.

    The mask EMA is mutated as frames are composed; the caller is responsible
    for snapshotting/restoring it so a discarded (failed) attempt leaves no
    trail. Returns (buffered_frames, success)."""
    obs, _ = env.reset()
    success = False
    tail = 0
    buf: list[Image.Image] = []
    for t in range(args.max_steps):
        # Periodic in-rollout mask reconfig from expert support
        # (0 = keep the per-task mask fixed for the whole rollout).
        if args.reconfig_every > 0 and t % args.reconfig_every == 0:
            mask = reconfig_mask(agent, demos[env.name], cfg)
        if t % args.frame_stride == 0 or success:
            ema.update(mask)
            att = f" try {attempt}" if attempt else ""
            title = f"[{ti}] {env.name}   ep {ep}{att}   step {t}" + (
                "   SUCCESS" if success else ""
            )
            buf.append(
                compose_frame(env.render(), ema.image(), title, args.render_size)
            )
        a = agent.act(obs, mask_a=mask)
        obs, _, term, trunc, info = env.step(a)
        if float(info.get("success", 0.0)) > 0.5:
            success = True
        if success:
            tail += 1
            if tail >= args.success_tail:
                break
        if term or trunc:
            break
    return buf, success


# `demos` is module-level so rollout_episode can reach it for in-rollout reconfig.
demos = None


def run(args):
    global demos
    agent, cfg, ck = load_agent(args.ckpt)
    tasks = ck["tasks"]
    phases_done = ck.get("phases_done", len(tasks))
    tasks = list(tasks[:phases_done])  # only visualize tasks the model trained on
    print(
        f"Loaded {args.ckpt}: {len(tasks)} task(s) trained, "
        f"square_dim={cfg.ftn_square_dim}, k={cfg.ftn_k}, H={cfg.ftn_hidden_size}"
    )

    seed = cfg.seed if args.seed is None else args.seed
    envs = build_render_envs(
        tasks, seed, args.camera, args.render_size, args.render_size
    )

    # Expert demos are needed to recover masks at eval — either for the
    # per-task mask (--mask-source reconfig) or for periodic in-rollout
    # reconfig (--reconfig-every > 0).
    if args.mask_source == "reconfig" or args.reconfig_every > 0:
        demos = generate_or_load_demos(envs, cfg)

    ema = MaskEMA(cfg.ftn_hidden_size, cfg.ftn_square_dim)
    frames: list[Image.Image] = []

    for ti, env in enumerate(envs):
        for ep in range(args.episodes):
            # Determine the active mask. For reconfig, recover a fresh mask
            # per episode from the expert support (an independent mask search,
            # so it varies episode-to-episode); the stored mask is constant.
            if args.mask_source == "reconfig":
                mask = reconfig_mask(agent, demos[env.name], cfg)
            else:
                mask = agent.stored_m_a[ti]

            # With --success-only, retry the episode until it succeeds (or we
            # run out of attempts), committing only the successful attempt's
            # frames. The EMA is snapshotted before each attempt and rolled
            # back on failure so discarded episodes leave no decay trail.
            attempts = args.max_attempts if args.success_only else 1
            for attempt in range(attempts):
                ema_snapshot = ema.ema.clone()
                buf, success = rollout_episode(
                    env, agent, mask, ema, cfg, args, ti, ep, attempt
                )
                last_attempt = attempt == attempts - 1
                if success or not args.success_only or last_attempt:
                    # Keep on success, when retries are disabled, or as a
                    # fallback on the final attempt so the GIF isn't empty.
                    frames.extend(buf)
                    break
                # Failed attempt: discard its frames and undo its EMA trail.
                ema.ema = ema_snapshot
                print(
                    f"  [{ti}] {env.name:25s} ep {ep} try {attempt}: "
                    f"fail    (retrying)"
                )
            print(
                f"  [{ti}] {env.name:25s} ep {ep}: "
                f"{'success' if success else 'fail   '} ({len(frames)} frames so far)"
            )

    if not frames:
        raise RuntimeError("No frames rendered.")
    frames[0].save(
        args.out,
        save_all=True,
        append_images=frames[1:],
        duration=args.duration,
        loop=0,
        optimize=True,
    )
    print(f"Saved {len(frames)} frames -> {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt", required=True, help="Checkpoint .pt from train_mt10_expert."
    )
    p.add_argument("--out", default="rollout.gif", help="Output GIF path.")
    p.add_argument("--episodes", type=int, default=1, help="Eval episodes per task.")
    p.add_argument(
        "--max-steps", type=int, default=200, help="Max env steps per episode."
    )
    p.add_argument(
        "--success-tail",
        type=int,
        default=8,
        help="Extra steps rendered after first success before cutting.",
    )
    p.add_argument(
        "--success-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry each episode until it succeeds and render only the "
        "successful episode's frames (failed attempts are discarded, EMA "
        "rolled back). Use --no-success-only to render the first attempt as-is.",
    )
    p.add_argument(
        "--max-attempts",
        type=int,
        default=20,
        help="Max retries per episode when --success-only is set; after this "
        "the last (failed) attempt's frames are kept so the GIF isn't empty.",
    )
    p.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Render/append every Nth env step (smaller = smoother, bigger GIF).",
    )
    p.add_argument(
        "--render-size", type=int, default=320, help="Side length of each panel."
    )
    p.add_argument("--duration", type=int, default=60, help="GIF frame duration (ms).")
    p.add_argument(
        "--camera",
        default="corner3",
        help="Meta-World camera (corner, corner2, corner3, topview, behindGripper, gripperPOV).",
    )
    p.add_argument(
        "--mask-source",
        choices=["stored", "reconfig"],
        default="reconfig",
        help="stored: per-task mask saved at training time. "
        "reconfig: recover each task's mask from its expert support at eval.",
    )
    p.add_argument(
        "--reconfig-every",
        type=int,
        default=0,
        help="Re-recover the FTN mask from expert support every N env "
        "steps during the rollout (0 = once per task; the mask stays "
        "fixed for the whole rollout). N>0 forces loading expert demos "
        "and shows the mask adapting mid-episode.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override env seed (default: ckpt cfg seed).",
    )
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
