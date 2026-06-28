"""JAX ColorWorld: a parameter-identical port of options/color_world.py.

The world geometry (Voronoi sites + per-cell hyper-colors) is generated on the
HOST with numpy.default_rng(world_seed) using the EXACT same calls as the torch
env, so a given world_seed produces a byte-identical patchwork -- the JAX and
torch experiments explore the same map and their PNGs line up.

Dynamics are pure functions of (pos, action) so they fuse into the collection
scan. ``obs = color vector of the nearest Voronoi site`` (allocentric control,
no heading/aliasing). The matplotlib ``render_world`` / ``_background`` helpers
are kept on the numpy side, identical to the torch env, for the visualizations.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp


class ColorWorld:
    """Host-side world: holds the numpy geometry + JAX device copies, exposes the
    same render surface as options/color_world.ColorWorld."""

    def __init__(self, size: float = 16.0, n_cells: int = 10, color_dim: int = 8,
                 world_seed: int = 0, max_speed: float = 0.6, max_steps: int = 400,
                 torus: bool = False):
        assert color_dim >= 3, "color_dim must be >= 3 (first 3 dims are RGB)"
        self.size = float(size)
        self.n_cells = int(n_cells)
        self.color_dim = int(color_dim)
        self.world_seed = int(world_seed)
        self.max_speed = float(max_speed)
        self.max_steps = int(max_steps)
        self.torus = bool(torus)  # wrap the box edges (no boundaries) vs clip at walls
        self.obstacles: list = []  # open room (render compatibility)

        # IDENTICAL generation order to the torch env.
        wrng = np.random.default_rng(world_seed)
        self._sites = wrng.uniform(0.0, self.size, size=(self.n_cells, 2))
        self._cell_colors = wrng.uniform(0.12, 0.95, size=(self.n_cells, self.color_dim)).astype(np.float32)
        self._bg_image: np.ndarray | None = None

        self.obs_dim = self.color_dim
        # Device copies for the JIT loops.
        self.sites_j = jnp.asarray(self._sites, dtype=jnp.float32)
        self.colors_j = jnp.asarray(self._cell_colors, dtype=jnp.float32)
        # Pose used only by render_world (set by the demo/driver if desired).
        self.pos = np.zeros(2)
        self.heading = 0.0

    def config(self) -> dict:
        return dict(size=self.size, n_cells=self.n_cells, color_dim=self.color_dim,
                    world_seed=self.world_seed, max_speed=self.max_speed, max_steps=self.max_steps,
                    torus=self.torus)

    # ---- numpy geometry/color (render only) ----------------------------
    def _cell_color_at(self, points: np.ndarray) -> np.ndarray:
        d2 = np.sum((points[:, None, :] - self._sites[None, :, :]) ** 2, axis=2)
        return self._cell_colors[np.argmin(d2, axis=1)]

    def _background(self, res: int = 320) -> np.ndarray:
        if self._bg_image is not None and self._bg_image.shape[0] == res:
            return self._bg_image
        xs = (np.arange(res) + 0.5) / res * self.size
        gx, gy = np.meshgrid(xs, xs)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        cols = self._cell_color_at(pts)[:, :3].reshape(res, res, 3)
        self._bg_image = (np.clip(cols, 0, 1) * 255).astype(np.uint8)
        return self._bg_image

    def render_world(self, centroids=None, centroid_ids=None, edges=None, path=None,
                     trail=None, current_id=None, goal_id=None, agent_pos=None,
                     agent_heading=None, caption=None, dpi: int = 100, figsize=(5, 5)) -> np.ndarray:
        """Top-down RGB frame -- identical to options/color_world.render_world."""
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
            ax.plot([t[0] for t in trail], [t[1] for t in trail], color=(0.05, 0.05, 0.05), lw=1.2, alpha=0.55, zorder=2)
        if path and len(path) >= 2:
            ax.plot([p[0] for p in path], [p[1] for p in path], color=(1.0, 1.0, 1.0), lw=2.6, alpha=0.95, zorder=3)
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
                ax.scatter([cx], [cy], s=size_, c=[color], edgecolors="black", linewidths=0.7, zorder=5)
        ap = self.pos if agent_pos is None else agent_pos
        ah = self.heading if agent_heading is None else agent_heading
        tri = np.array([[0.6, 0], [-0.35, 0.3], [-0.35, -0.3]]) * 0.95
        rot = np.array([[np.cos(ah), -np.sin(ah)], [np.sin(ah), np.cos(ah)]])
        tri = tri @ rot.T + ap
        ax.fill(tri[:, 0], tri[:, 1], facecolor=(1, 1, 1), edgecolor=(0, 0, 0), zorder=7)
        if caption:
            ax.text(0.02 * self.size, 0.98 * self.size, caption, color="white", fontsize=8, va="top",
                    zorder=8, bbox=dict(facecolor=(0, 0, 0, 0.65), edgecolor="none", pad=2))
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return buf


# ---------------------------------------------------------------------------
# JAX-pure dynamics (used inside the collection scan)
# ---------------------------------------------------------------------------


@jax.jit
def color_at(pos, sites, colors):
    """Nearest-site color vector. pos (...,2) -> (...,color_dim)."""
    d2 = jnp.sum((pos[..., None, :] - sites) ** 2, axis=-1)  # (..., n_cells)
    idx = jnp.argmin(d2, axis=-1)
    return colors[idx]
