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
                 torus: bool = False, n_obstacles: int = 0,
                 obstacle_min_r: float = 1.0, obstacle_max_r: float = 2.5,
                 obstacle_max_verts: int = 6, coord_obs: bool = False):
        assert color_dim >= 3, "color_dim must be >= 3 (first 3 dims are RGB)"
        # coord_obs=True -> obs is the continuous (x, y) position in [-1,1]^2 instead
        # of the nearest-site color vector; the Voronoi geometry is still generated
        # (RNG stream unchanged) but unused for obs, and the demo renders white space.
        self.coord_obs = bool(coord_obs)
        self.size = float(size)
        self.n_cells = int(n_cells)
        self.color_dim = int(color_dim)
        self.world_seed = int(world_seed)
        self.max_speed = float(max_speed)
        self.max_steps = int(max_steps)
        self.torus = bool(torus)  # wrap the box edges (no boundaries) vs clip at walls
        self.n_obstacles = int(n_obstacles)
        self.obstacle_min_r = float(obstacle_min_r)
        self.obstacle_max_r = float(obstacle_max_r)
        self.obstacle_max_verts = int(obstacle_max_verts)
        self.obstacles: list = []  # rect (x0,y0,x1,y1) list -- kept empty for viz compat

        # IDENTICAL generation order to the torch env: sites then cell_colors are
        # drawn FIRST from wrng, so any world_seed yields the same patchwork whether
        # or not obstacles are enabled. Obstacles consume the stream only afterward.
        wrng = np.random.default_rng(world_seed)
        self._sites = wrng.uniform(0.0, self.size, size=(self.n_cells, 2))
        self._cell_colors = wrng.uniform(0.12, 0.95, size=(self.n_cells, self.color_dim)).astype(np.float32)
        # Random convex polygon obstacles (CCW). _polys: per-obstacle (Vi,2) for
        # rendering; _polys_padded: (P, Vmax, 2) padded by repeating the last vertex
        # for the vectorized convex point-in-polygon test (host + device).
        self._polys: list[np.ndarray] = _gen_obstacles(
            wrng, self.size, self.n_obstacles, self.obstacle_min_r,
            self.obstacle_max_r, self.obstacle_max_verts)
        self._polys_padded = _pad_polys(self._polys)
        self._bg_image: np.ndarray | None = None

        self.obs_dim = 2 if self.coord_obs else self.color_dim
        # Device copies for the JIT loops.
        self.sites_j = jnp.asarray(self._sites, dtype=jnp.float32)
        self.colors_j = jnp.asarray(self._cell_colors, dtype=jnp.float32)
        self.polys_j = jnp.asarray(self._polys_padded, dtype=jnp.float32)  # (P, Vmax, 2)
        # Pose used only by render_world (set by the demo/driver if desired).
        self.pos = np.zeros(2)
        self.heading = 0.0

    def config(self) -> dict:
        return dict(size=self.size, n_cells=self.n_cells, color_dim=self.color_dim,
                    world_seed=self.world_seed, max_speed=self.max_speed, max_steps=self.max_steps,
                    torus=self.torus, n_obstacles=self.n_obstacles,
                    obstacle_min_r=self.obstacle_min_r, obstacle_max_r=self.obstacle_max_r,
                    obstacle_max_verts=self.obstacle_max_verts, coord_obs=self.coord_obs)

    def in_obstacle(self, points: np.ndarray) -> np.ndarray:
        """points (...,2) -> (...,) bool: inside ANY obstacle polygon (host/numpy)."""
        return _inside_convex_np(np.asarray(points, np.float32), self._polys_padded)

    # ---- numpy geometry/color (render only) ----------------------------
    def _cell_color_at(self, points: np.ndarray) -> np.ndarray:
        d2 = np.sum((points[:, None, :] - self._sites[None, :, :]) ** 2, axis=2)
        return self._cell_colors[np.argmin(d2, axis=1)]

    def _background(self, res: int = 320) -> np.ndarray:
        if self._bg_image is not None and self._bg_image.shape[0] == res:
            return self._bg_image
        if self.coord_obs:  # no patchwork in coordinate-obs mode: plain gray space
            # mid-gray so the white plan path / centroids stay visible against it
            self._bg_image = np.full((res, res, 3), 128, dtype=np.uint8)
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
        for poly in self._polys:
            ax.fill(poly[:, 0], poly[:, 1], facecolor=(0.12, 0.12, 0.13),
                    edgecolor="black", lw=1.0, zorder=1)
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
# Obstacle geometry (host generation + vectorized convex point-in-polygon)
# ---------------------------------------------------------------------------


def _convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain. pts (k,2) -> CCW hull vertices (h,2), h>=3."""
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(points):
        chain = []
        for p in points:
            while len(chain) >= 2 and np.cross(chain[-1] - chain[-2], p - chain[-2]) <= 0:
                chain.pop()
            chain.append(p)
        return chain[:-1]

    lower = half(pts)
    upper = half(pts[::-1])
    return np.asarray(lower + upper, dtype=np.float32)  # CCW


def _gen_obstacles(rng, size, n, min_r, max_r, max_verts, margin=0.5):
    """`n` random convex polygons (CCW) inside [margin, size-margin]^2. Each is the
    convex hull of points sampled in a disk, so it is guaranteed convex (the
    half-plane inside-test below requires convexity)."""
    polys = []
    for _ in range(int(n)):
        c = rng.uniform(margin, size - margin, size=2)
        r = float(rng.uniform(min_r, max_r))
        k = int(rng.integers(3, max(3, max_verts) + 1))
        ang = rng.uniform(0.0, 2.0 * np.pi, size=k)
        rad = r * rng.uniform(0.55, 1.0, size=k)
        pts = c + np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=1)
        hull = _convex_hull(pts.astype(np.float32))
        if len(hull) >= 3:
            polys.append(hull)
    return polys


def _pad_polys(polys: list[np.ndarray]) -> np.ndarray:
    """Pack a list of (Vi,2) CCW polygons into (P, Vmax, 2), padding each by
    repeating its last vertex (degenerate edges contribute 0 to the inside test).
    Empty list -> (0, 1, 2) so the device test trivially returns all-False."""
    if not polys:
        return np.zeros((0, 1, 2), dtype=np.float32)
    vmax = max(len(p) for p in polys)
    out = np.zeros((len(polys), vmax, 2), dtype=np.float32)
    for i, p in enumerate(polys):
        out[i, :len(p)] = p
        out[i, len(p):] = p[-1]
    return out


def _inside_convex(pos, polys, eps: float = 1e-6):
    """JAX: pos (...,2), polys (P,Vmax,2) CCW convex padded -> (...,) bool, inside
    ANY polygon. A point is inside a CCW convex polygon iff it is left of (or on)
    every edge; padded degenerate edges give cross==0 and pass."""
    a = polys                            # (P, V, 2)
    b = jnp.roll(polys, -1, axis=1)      # (P, V, 2) -- edge a->b, closing edge wraps
    e = b - a                            # (P, V, 2)
    rp = pos[..., None, None, :] - a     # (..., P, V, 2)
    cross = e[..., 0] * rp[..., 1] - e[..., 1] * rp[..., 0]   # (..., P, V)
    inside_poly = jnp.all(cross >= -eps, axis=-1)             # (..., P)
    return jnp.any(inside_poly, axis=-1)                      # (...,)


def _inside_convex_np(pos, polys, eps: float = 1e-6):
    """Numpy mirror of _inside_convex (host step + reset rejection)."""
    if polys.shape[0] == 0:
        return np.zeros(pos.shape[:-1], dtype=bool)
    a = polys
    b = np.roll(polys, -1, axis=1)
    e = b - a
    rp = pos[..., None, None, :] - a
    cross = e[..., 0] * rp[..., 1] - e[..., 1] * rp[..., 0]
    return np.any(np.all(cross >= -eps, axis=-1), axis=-1)


def sample_free(key, shape2, size, polys, tries: int = 12):
    """JAX: sample positions uniform in [0.02,0.98]*size avoiding obstacles. shape2
    is (2,) for one walker or (W,2) for a fleet. Draws `tries` candidates and keeps
    the first one outside every obstacle (falls back to the first if all are stuck).
    Caller should branch on polys.shape[0]==0 to keep the no-obstacle RNG stream."""
    cand = jax.random.uniform(key, (tries,) + tuple(shape2),
                              minval=0.02 * size, maxval=0.98 * size)
    free = ~_inside_convex(cand, polys)          # (tries,) or (tries, W)
    idx = jnp.argmax(free, axis=0)               # () or (W,)
    return jnp.take_along_axis(cand, idx[None, ..., None], axis=0)[0]


def block_move(pos, npos, polys):
    """JAX: forbid entering an obstacle -- if npos is inside one, stay at pos.
    pos/npos (...,2). No-op when polys is empty. Permits leaving (pos inside)."""
    blocked = _inside_convex(npos, polys) & (~_inside_convex(pos, polys))  # (...,)
    return jnp.where(blocked[..., None], pos, npos)


# ---------------------------------------------------------------------------
# JAX-pure dynamics (used inside the collection scan)
# ---------------------------------------------------------------------------


@jax.jit
def color_at(pos, sites, colors):
    """Nearest-site color vector. pos (...,2) -> (...,color_dim)."""
    d2 = jnp.sum((pos[..., None, :] - sites) ** 2, axis=-1)  # (..., n_cells)
    idx = jnp.argmin(d2, axis=-1)
    return colors[idx]


@jax.jit
def coord_at(pos, size):
    """Continuous coordinate obs: position in [0,size] normalized to [-1,1]^2.
    pos (...,2) -> (...,2)."""
    return pos / size * 2.0 - 1.0
