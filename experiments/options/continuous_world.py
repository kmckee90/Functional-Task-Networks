"""Continuous top-down 2D world with an egocentric point of view (gymnasium).

A bounded square arena whose floor is a **random patchwork of colored polygons**
(a Voronoi tessellation of >=30 cells, each a random color) covering the whole
space, plus a few static rectangular **obstacles** that block movement and the
agent's view. The dense color field makes the egocentric observation vary
locally everywhere, so clustering tiles the space finely enough for local-step
planning (the earlier version had only a handful of sparse colored circles).

The agent is a point with a heading. Its action space is continuous:
``[turn, accel]`` in ``[-1, 1]^2`` -- a bounded turning rotation and a bounded
forward acceleration ("turn left / turn right / go forward").

There are **no rewards and no episode termination** (only an optional step-count
truncation). This is an exploration substrate: an external agent supplies its own
(intrinsic) objective.

Observation (``float32``, all in [0, 1], flattened) is **purely egocentric** --
NO absolute pose, mirroring ``options/grid_world.py``:
  * ``obstacle_dist``  : one normalized distance per ray to the nearest wall /
    obstacle / border, for ``n_rays`` rays fanned over 360 degrees *relative to
    the agent's heading* (ray 0 points straight ahead).
  * ``cell_rgb``       : the floor color sampled a short distance ahead along each
    ray (a "color compass" of the surrounding cells), per ray.
  * ``floor_rgb``      : the color of the cell directly under the agent.

The true ``(x, y, heading)`` is tracked internally and exposed only via ``info``
and ``render`` -- the agent never observes it. The patchwork is fixed by
``world_seed`` (independent of the per-episode reset seed) so the world is
identical across training and demo runs.
"""

from __future__ import annotations

import argparse
import os

import gymnasium as gym
import numpy as np
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Geometry helper (vectorized ray/segment casting)
# ---------------------------------------------------------------------------


def _ray_segments_dist(
    origin: np.ndarray,
    direction: np.ndarray,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    max_range: float,
) -> float:
    """Nearest forward hit distance of a ray against a batch of segments.

    Ray  : ``origin + t * direction`` (``direction`` unit, ``t >= 0``).
    Segs : ``seg_a + s * (seg_b - seg_a)`` (``s in [0, 1]``), shapes ``(M, 2)``.
    Returns the smallest valid ``t`` (clamped to ``max_range`` if no hit)."""
    v1 = origin[None, :] - seg_a          # (M, 2)
    v2 = seg_b - seg_a                    # (M, 2)
    v3 = np.array([-direction[1], direction[0]])  # perp of ray dir
    denom = v2 @ v3                       # (M,)
    nonzero = np.abs(denom) > 1e-12
    cross = v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = cross / denom                # distance along the ray
        t2 = (v1 @ v3) / denom            # position along the segment
    valid = nonzero & (t1 >= 0) & (t2 >= 0) & (t2 <= 1)
    if not np.any(valid):
        return max_range
    return float(min(np.min(t1[valid]), max_range))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class ContinuousWorld(gym.Env):
    """Continuous, egocentric, reward-free top-down navigation substrate."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        size: float = 16.0,
        n_rays: int = 16,
        n_cells: int = 36,            # Voronoi cells in the floor patchwork (>= 30)
        world_seed: int = 0,          # fixes the patchwork (independent of reset seed)
        ray_sense_range: float | None = None,  # how far ahead each ray samples color
        max_turn: float = 0.9,       # rad / step at |turn| = 1
        max_accel: float = 0.18,      # speed gained / step at accel = 1
        friction: float = 0.06,       # speed bled off per step
        max_speed: float = 0.7,
        max_steps: int = 400,
        obstacles: list[tuple[float, float, float, float]] | None = None,
        seed: int | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.size = float(size)
        self.n_rays = int(n_rays)
        self.n_cells = int(n_cells)
        self.world_seed = int(world_seed)
        self.max_turn = max_turn
        self.max_accel = max_accel
        self.friction = friction
        self.max_speed = max_speed
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.max_range = float(np.hypot(size, size))
        self.ray_sense_range = float(ray_sense_range) if ray_sense_range is not None else 0.18 * size

        s = self.size
        if obstacles is None:
            obstacles = [
                (0.30 * s, 0.30 * s, 0.45 * s, 0.70 * s),   # left bar
                (0.55 * s, 0.55 * s, 0.72 * s, 0.68 * s),   # mid block
                (0.55 * s, 0.20 * s, 0.68 * s, 0.40 * s),   # lower-right block
            ]
        self.obstacles = obstacles

        # --- random colored-polygon patchwork (Voronoi by nearest site) ----
        wrng = np.random.default_rng(world_seed)
        self._sites = wrng.uniform(0.0, s, size=(self.n_cells, 2))
        # Saturated, near-unique random colors so adjacent cells are
        # distinguishable and each cell gives a distinct sensory signature.
        self._cell_colors = wrng.uniform(0.12, 0.95, size=(self.n_cells, 3))
        self._bg_image: np.ndarray | None = None  # lazily rasterized for render

        # Wall segments: obstacle edges + the four borders.
        self._seg_a, self._seg_b = self._build_walls()

        # Observation: per-ray obstacle distance + per-ray cell RGB + floor RGB.
        self.obs_dim = self.n_rays + self.n_rays * 3 + 3
        self.observation_space = spaces.Box(0.0, 1.0, (self.obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

        self.pos = np.zeros(2)
        self.heading = 0.0
        self.speed = 0.0
        self.steps = 0
        if seed is not None:
            self.reset(seed=seed)

    def config(self) -> dict:
        """All constructor params needed to rebuild an identical world (mirrors
        ColorWorld.config()). Saved at training time so the parallel collection
        workers and the demo reproduce the EXACT dynamics and floor patchwork."""
        return dict(size=self.size, n_rays=self.n_rays, n_cells=self.n_cells,
                    world_seed=self.world_seed, ray_sense_range=self.ray_sense_range,
                    max_turn=self.max_turn, max_accel=self.max_accel,
                    friction=self.friction, max_speed=self.max_speed,
                    max_steps=self.max_steps, obstacles=self.obstacles)

    # ---- geometry ------------------------------------------------------

    def _build_walls(self) -> tuple[np.ndarray, np.ndarray]:
        a, b = [], []

        def rect(x0, y0, x1, y1):
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            for i in range(4):
                a.append(corners[i])
                b.append(corners[(i + 1) % 4])

        s = self.size
        rect(0, 0, s, s)  # border
        for (x0, y0, x1, y1) in self.obstacles:
            rect(x0, y0, x1, y1)
        return np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)

    def _blocked(self, p0: np.ndarray, p1: np.ndarray) -> bool:
        d = p1 - p0
        dist = float(np.hypot(d[0], d[1]))
        if dist < 1e-9:
            return False
        direction = d / dist
        t = _ray_segments_dist(p0, direction, self._seg_a, self._seg_b, self.max_range)
        return t <= dist

    def _cell_color_at(self, points: np.ndarray) -> np.ndarray:
        """Nearest-site (Voronoi cell) color for each point. points: (M, 2) ->
        (M, 3)."""
        d2 = np.sum((points[:, None, :] - self._sites[None, :, :]) ** 2, axis=2)  # (M, K)
        return self._cell_colors[np.argmin(d2, axis=1)]

    # ---- gymnasium API -------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        for _ in range(1000):
            p = self.np_random.uniform(0.08 * self.size, 0.92 * self.size, size=2)
            if not self._in_obstacle(p):
                break
        self.pos = p
        self.heading = float(self.np_random.uniform(0, 2 * np.pi))
        self.speed = 0.0
        self.steps = 0
        return self._obs(), self._info()

    def _in_obstacle(self, p: np.ndarray) -> bool:
        for (x0, y0, x1, y1) in self.obstacles:
            if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
                return True
        return False

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        turn = float(np.clip(action[0], -1, 1))
        accel = float(np.clip(action[1], -1, 1))

        self.heading = (self.heading + turn * self.max_turn) % (2 * np.pi)
        self.speed = float(np.clip(self.speed + accel * self.max_accel - self.friction, 0.0, self.max_speed))

        d = self.speed * np.array([np.cos(self.heading), np.sin(self.heading)])
        new_pos = self.pos + d
        if self._blocked(self.pos, new_pos):
            self.speed = 0.0
        else:
            self.pos = new_pos

        self.steps += 1
        truncated = self.steps >= self.max_steps
        return self._obs(), 0.0, False, truncated, self._info()

    # ---- observation ---------------------------------------------------

    def _obs(self) -> np.ndarray:
        angles = self.heading + np.linspace(0, 2 * np.pi, self.n_rays, endpoint=False)
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (R, 2)
        obst = np.empty(self.n_rays, dtype=np.float32)
        sample_pts = np.empty((self.n_rays, 2), dtype=np.float64)
        for i in range(self.n_rays):
            wall_d = _ray_segments_dist(self.pos, dirs[i], self._seg_a, self._seg_b, self.max_range)
            obst[i] = wall_d / self.max_range
            # Sample the floor color a short distance ahead (capped just short of
            # the wall) -> an egocentric "color compass" of the nearby cells.
            sd = min(self.ray_sense_range, max(0.0, wall_d - 0.05))
            sample_pts[i] = self.pos + sd * dirs[i]
        cell_rgb = self._cell_color_at(sample_pts).astype(np.float32)      # (R, 3)
        floor = self._cell_color_at(self.pos[None, :])[0].astype(np.float32)
        return np.concatenate([obst, cell_rgb.ravel(), floor]).astype(np.float32)

    def _info(self) -> dict:
        return {"pos": self.pos.copy(), "heading": self.heading,
                "speed": self.speed, "steps": self.steps}

    # ---- rendering -----------------------------------------------------

    def _background(self, res: int = 320) -> np.ndarray:
        """Rasterize the Voronoi patchwork to an (res, res, 3) uint8 image once."""
        if self._bg_image is not None and self._bg_image.shape[0] == res:
            return self._bg_image
        xs = (np.arange(res) + 0.5) / res * self.size
        gx, gy = np.meshgrid(xs, xs)                         # (res, res), origin lower
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)     # (res*res, 2)
        cols = self._cell_color_at(pts).reshape(res, res, 3)
        self._bg_image = (cols * 255).astype(np.uint8)
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
        """Top-down RGB frame (uint8 H x W x 3) of the whole world with optional
        graph/path/trail/membership overlays. matplotlib Agg backend."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.set_xticks([]); ax.set_yticks([])

        # Colored-polygon patchwork as the floor.
        ax.imshow(self._background(), extent=[0, self.size, 0, self.size],
                  origin="lower", interpolation="nearest", zorder=0)

        for (x0, y0, x1, y1) in self.obstacles:
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor=(0.12, 0.12, 0.13), edgecolor="black", lw=1, zorder=1))

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
        ax.fill(tri[:, 0], tri[:, 1], color=(0, 0, 0), zorder=7)
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
    gym.register(id="ContinuousWorld-v0", entry_point=__name__ + ":ContinuousWorld")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Smoke test: random self-drive, dump one frame.
# ---------------------------------------------------------------------------


def _smoke(steps: int, seed: int) -> None:
    env = ContinuousWorld(seed=seed)
    obs, info = env.reset(seed=seed)
    print(f"obs_dim={env.obs_dim}  obs.shape={obs.shape}  action_space={env.action_space}")
    print(f"  n_rays={env.n_rays}  n_cells={env.n_cells}  {len(env.obstacles)} obstacles")
    rng = np.random.default_rng(seed)
    moved = 0
    for t in range(steps):
        obs, r, term, trunc, info = env.step(rng.uniform(-1, 1, size=2))
        if info["speed"] > 0:
            moved += 1
        if trunc:
            obs, info = env.reset()
    print(f"ran {steps} random steps  (moving steps: {moved})")
    print(f"  final obs range [{obs.min():.3f}, {obs.max():.3f}]")
    out_dir = os.path.join(os.path.dirname(__file__), "artifacts")
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image
    frame = env.render_world(caption=f"ContinuousWorld ({env.n_cells}-cell patchwork)")
    Image.fromarray(frame).save(os.path.join(out_dir, "world.png"))
    print(f"wrote {os.path.join(out_dir, 'world.png')}  frame.shape={frame.shape}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--random", action="store_true", help="random self-drive smoke test")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    _smoke(args.steps if args.random else 200, args.seed)
