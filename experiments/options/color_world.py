"""Simplified continuous top-down world: allocentric control + pure color obs.

An alternative to ``continuous_world.ContinuousWorld`` for proving the options
pipeline in an easy setting (no aliasing, no egocentric view):

  * Floor = a Voronoi patchwork of ``n_cells`` cells, each with a UNIQUE random
    "hyper-color" vector of length ``color_dim`` (>= 3). The first 3 components are
    the RGB used to render the cell; the extra components just make every cell's
    vector distinct so clustering is trivially unambiguous.
  * **Allocentric control**: action ``[dx, dy] in [-1, 1]^2`` moves the agent in
    absolute x/y (left/right/up/down). There is NO heading/rotation.
  * **Observation = the color vector of the cell the agent is currently on**
    (``color_dim`` floats). Nothing else -- no rays, no pose. Two different cells
    never share an observation, so a centroid == a cell, exactly.
  * **No obstacles** -- an open room.

No rewards, no termination (optional step truncation). Interface matches the bits
of ContinuousWorld that the trainer/demo use: ``size``, ``obstacles`` (empty),
``obs_dim``, ``action_space``, ``reset``/``step`` (info has pos/heading/speed),
``_background`` and ``render_world`` for the top-down PNGs/GIF.
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ColorWorld(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        size: float = 16.0,
        n_cells: int = 10,
        color_dim: int = 8,        # >= 3; first 3 dims are the visualized RGB
        world_seed: int = 0,       # fixes the patchwork (independent of reset seed)
        max_speed: float = 0.6,    # max |displacement| per step (per axis at |a|=1)
        max_steps: int = 400,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        assert color_dim >= 3, "color_dim must be >= 3 (first 3 dims are RGB)"
        self.size = float(size)
        self.n_cells = int(n_cells)
        self.color_dim = int(color_dim)
        self.world_seed = int(world_seed)
        self.max_speed = max_speed
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.obstacles: list = []  # open room (kept for render/save compatibility)

        wrng = np.random.default_rng(world_seed)
        self._sites = wrng.uniform(0.0, self.size, size=(self.n_cells, 2))
        # Unique random color vector per cell; saturated so RGB renders vividly.
        self._cell_colors = wrng.uniform(0.12, 0.95, size=(self.n_cells, self.color_dim)).astype(np.float32)
        self._bg_image: np.ndarray | None = None

        self.obs_dim = self.color_dim
        self.observation_space = spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

        self.pos = np.zeros(2)
        self.heading = 0.0
        self.speed = 0.0
        self.steps = 0
        if seed is not None:
            self.reset(seed=seed)

    def config(self) -> dict:
        """All constructor params needed to rebuild an identical world. Saved at
        training time so the demo reproduces the EXACT dynamics (esp. max_speed,
        which scales the policy's per-step displacement)."""
        return dict(size=self.size, n_cells=self.n_cells, color_dim=self.color_dim,
                    world_seed=self.world_seed, max_speed=self.max_speed, max_steps=self.max_steps)

    # ---- geometry / color ---------------------------------------------

    def _cell_color_at(self, points: np.ndarray) -> np.ndarray:
        """Nearest-site color vector for each point. points: (M, 2) -> (M, color_dim)."""
        d2 = np.sum((points[:, None, :] - self._sites[None, :, :]) ** 2, axis=2)
        return self._cell_colors[np.argmin(d2, axis=1)]

    def _in_obstacle(self, p: np.ndarray) -> bool:
        return False  # open room

    # ---- gymnasium API -------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.pos = self.np_random.uniform(0.02 * self.size, 0.98 * self.size, size=2)
        self.heading = 0.0
        self.speed = 0.0
        self.steps = 0
        return self._obs(), self._info()

    def step(self, action: np.ndarray):
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        disp = a * self.max_speed
        self.pos = np.clip(self.pos + disp, 0.0, self.size)
        self.speed = float(np.hypot(disp[0], disp[1]))
        if self.speed > 1e-6:
            self.heading = float(np.arctan2(disp[1], disp[0]))
        self.steps += 1
        return self._obs(), 0.0, False, self.steps >= self.max_steps, self._info()

    def _obs(self) -> np.ndarray:
        return self._cell_color_at(self.pos[None, :])[0].astype(np.float32)

    def _info(self) -> dict:
        return {"pos": self.pos.copy(), "heading": self.heading,
                "speed": self.speed, "steps": self.steps}

    # ---- rendering -----------------------------------------------------

    def _background(self, res: int = 320) -> np.ndarray:
        if self._bg_image is not None and self._bg_image.shape[0] == res:
            return self._bg_image
        xs = (np.arange(res) + 0.5) / res * self.size
        gx, gy = np.meshgrid(xs, xs)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        cols = self._cell_color_at(pts)[:, :3].reshape(res, res, 3)
        self._bg_image = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
        return self._bg_image

    def render_world(
        self,
        centroids: list | None = None,
        centroid_ids: list | None = None,
        edges: list | None = None,
        path: list | None = None,
        trail: list | None = None,
        current_id=None,
        goal_id=None,
        agent_pos: np.ndarray | None = None,
        agent_heading: float | None = None,
        caption: str | None = None,
        dpi: int = 100,
        figsize: tuple = (5, 5),
    ) -> np.ndarray:
        """Top-down RGB frame with optional graph/path/trail/membership overlays."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(self._background(), extent=[0, self.size, 0, self.size],
                  origin="lower", interpolation="nearest", zorder=0)

        if edges:
            for (p, q) in edges:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=(0.95, 0.95, 0.95), lw=0.7, alpha=0.35, zorder=2)
        if trail and len(trail) >= 2:
            tx = [t[0] for t in trail]; ty = [t[1] for t in trail]
            ax.plot(tx, ty, color=(0.05, 0.05, 0.05), lw=1.2, alpha=0.55, zorder=2)
        if path and len(path) >= 2:
            px = [p[0] for p in path]; py = [p[1] for p in path]
            ax.plot(px, py, color=(1.0, 1.0, 1.0), lw=2.6, alpha=0.95, zorder=3)
        if centroids:
            for k, (cx, cy) in enumerate(centroids):
                cid = centroid_ids[k] if centroid_ids is not None else k
                is_cur = current_id is not None and cid == current_id
                is_goal = goal_id is not None and cid == goal_id
                color, size_ = (1.0, 1.0, 1.0), 22
                if is_goal:
                    color, size_ = (0.1, 1.0, 1.0), 150
                elif is_cur:
                    color, size_ = (1.0, 0.15, 0.15), 120
                ax.scatter([cx], [cy], s=size_, c=[color], edgecolors="black",
                           linewidths=0.7, zorder=5)

        ap = self.pos if agent_pos is None else agent_pos
        ah = self.heading if agent_heading is None else agent_heading
        tri = np.array([[0.6, 0], [-0.35, 0.3], [-0.35, -0.3]]) * 0.95
        rot = np.array([[np.cos(ah), -np.sin(ah)], [np.sin(ah), np.cos(ah)]])
        tri = tri @ rot.T + ap
        ax.fill(tri[:, 0], tri[:, 1], facecolor=(1, 1, 1), edgecolor=(0, 0, 0), zorder=7)

        if caption:
            ax.text(0.02 * self.size, 0.98 * self.size, caption, color="white",
                    fontsize=8, va="top", zorder=8,
                    bbox=dict(facecolor=(0, 0, 0, 0.65), edgecolor="none", pad=2))

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return buf


try:
    gym.register(id="ColorWorld-v0", entry_point=__name__ + ":ColorWorld")
except Exception:
    pass


def _smoke(steps: int, seed: int) -> None:
    env = ColorWorld(seed=seed)
    obs, info = env.reset(seed=seed)
    print(f"obs_dim={env.obs_dim} (color_dim)  n_cells={env.n_cells}  action_space={env.action_space}")
    rng = np.random.default_rng(seed)
    seen = set()
    for _ in range(steps):
        obs, _, _, trunc, info = env.step(rng.uniform(-1, 1, size=2))
        seen.add(tuple(np.round(obs, 4)))
        if trunc:
            obs, info = env.reset()
    print(f"ran {steps} steps; distinct color-obs seen={len(seen)} (<= n_cells={env.n_cells})")
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image
    Image.fromarray(env.render_world(caption=f"ColorWorld ({env.n_cells} cells, color_dim={env.color_dim})")).save(
        os.path.join(out_dir, "color_world.png"))
    print(f"wrote {os.path.join(out_dir, 'color_world.png')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--random", action="store_true")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    _smoke(args.steps if args.random else 300, args.seed)
