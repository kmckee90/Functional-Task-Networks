"""Visualizations -- pixel-for-pixel the same plots as the torch trainer.

Ports save_centroids_png / save_graph_png / save_exploration_png (train_options.py)
and save_masks_png (train_options_2.py), reading from the JAX agent's numpy
registry + masks. Matplotlib styling, zorders, captions and colors are unchanged.
"""

from __future__ import annotations

import os

import numpy as np


def _new_topdown(env):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure(figsize=(5, 5), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(env._background(), extent=[0, env.size, 0, env.size],
              origin="lower", interpolation="nearest", zorder=0)
    for (x0, y0, x1, y1) in env.obstacles:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor=(0.12, 0.12, 0.13), edgecolor="black", lw=1, zorder=1))
    return fig, ax


def _caption(ax, env, text):
    ax.text(0.02 * env.size, 0.98 * env.size, text, color="white", fontsize=7.5, va="top",
            zorder=9, bbox=dict(facecolor=(0, 0, 0, 0.6), edgecolor="none", pad=2))


def _centroid_xy(agent):
    pos = agent.registry.positions_np()
    cnt = agent.registry.pos_count
    ids = [c for c in range(agent.registry.n) if cnt[c] > 0]
    return ids, pos


def save_centroids_png(env, agent, path):
    import matplotlib.pyplot as plt
    ids, pos = _centroid_xy(agent)
    fig, ax = _new_topdown(env)
    for cid in ids:
        x, y = pos[cid]
        ax.scatter([x], [y], s=42, c=[(1, 1, 1)], edgecolors="black", linewidths=0.7, zorder=5)
        ax.text(x + 0.12, y + 0.12, str(cid), color="white", fontsize=6, zorder=6,
                bbox=dict(facecolor=(0, 0, 0, 0.45), edgecolor="none", pad=0.4))
    _caption(ax, env, f"RECODE centroids: {len(ids)} shown / {agent.registry.n} total "
                      f"(at mean footprint)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_graph_png(env, agent, graph, path):
    import matplotlib.pyplot as plt
    ids, pos = _centroid_xy(agent)
    idset = set(ids)
    fig, ax = _new_topdown(env)
    el = graph.undirected_edge_list()
    wmax = max((w for _, _, w in el), default=1.0)
    for a, b, w in el:
        if a in idset and b in idset:
            pa, pb = pos[a], pos[b]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=(1, 1, 1),
                    lw=0.4 + 1.8 * (w / wmax), alpha=0.5, zorder=2)
    for cid in ids:
        x, y = pos[cid]
        ax.scatter([x], [y], s=30, c=[(1, 0.55, 0.0)], edgecolors="black", linewidths=0.6, zorder=5)
        ax.text(x + 0.12, y + 0.12, str(cid), color="white", fontsize=6, zorder=6,
                bbox=dict(facecolor=(0, 0, 0, 0.45), edgecolor="none", pad=0.4))
    _caption(ax, env, f"behavioral graph: {len(graph.nodes)} nodes, {len(el)} undirected edges "
                      f"(width ~ #transitions)")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_exploration_png(env, agent, path):
    import matplotlib.pyplot as plt
    pos = agent.buffer_next_pos  # numpy (N, 2)
    bnd = agent.buffer_boundary  # numpy (N,)
    if len(pos) < 2:
        return
    fig, ax = _new_topdown(env)
    ends = list(np.where(bnd > 0.5)[0])
    if not ends or ends[-1] != len(pos) - 1:
        ends.append(len(pos) - 1)
    cmap = plt.get_cmap("turbo")
    n_eps = len(ends)
    start = 0
    for e, end in enumerate(ends):
        seg = pos[start:end + 1]
        if len(seg) >= 2:
            ax.plot(seg[:, 0], seg[:, 1], lw=0.5, alpha=0.6,
                    color=cmap(e / max(1, n_eps - 1)), zorder=2)
        start = end + 1
    ax.text(0.02 * env.size, 0.98 * env.size,
            f"exploration trajectory: {len(pos)} steps, {n_eps} episodes (color = time)",
            color="white", fontsize=7.5, va="top", zorder=5,
            bbox=dict(facecolor=(0, 0, 0, 0.6), edgecolor="none", pad=2))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---- mask overlap diagnostic (train_options_2.save_masks_png) -------------


def _mask_grid(m_np, sq, H):
    if H % (sq * sq) == 0:
        bs = sq * sq
        n_blocks = H // bs
        return np.concatenate([m_np[i * bs:(i + 1) * bs].reshape(sq, sq)
                               for i in range(n_blocks)], axis=-1)
    c = int(np.ceil(np.sqrt(H)))
    r = int(np.ceil(H / c))
    pad = np.zeros(r * c, np.float32)
    pad[:H] = m_np
    return pad.reshape(r, c)


def save_masks_png(agent, path, which="m_a", alpha=0.3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cids = sorted(agent.options.keys())
    if not cids:
        return
    H, sq = agent.cfg.ftn_hidden_size, agent.cfg.lateral_square_dim
    masks_np = {c: np.asarray(agent.options[c][which]) for c in cids}
    g0 = _mask_grid(masks_np[cids[0]], sq, H)
    gh, gw = g0.shape
    rng = np.random.default_rng(0)
    canvas = np.full((gh, gw, 3), 0.08, dtype=np.float32)
    usage = np.zeros((gh, gw), dtype=np.float32)
    for cid in cids:
        g = _mask_grid(masks_np[cid], sq, H)
        color = rng.uniform(0.25, 1.0, size=3).astype(np.float32)
        a = alpha * (g > 0.5)[..., None]
        canvas = canvas * (1 - a) + color[None, None, :] * a
        usage += (g > 0.5)

    M = np.stack([masks_np[c] for c in cids]).astype(np.float32)
    kk = float(M[0].sum())
    inter = (M @ M.T) / max(kk, 1.0)
    n = len(cids)
    off = inter[~np.eye(n, dtype=bool)]
    mean_ov = float(off.mean()) if n > 1 else 0.0

    fig = plt.figure(figsize=(max(5, gw / gh * 5), 5), dpi=130)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(canvas, origin="lower", interpolation="nearest")
    ax.text(0.01, 0.99, f"{n} option {which} masks on {gh}x{gw} grid  (k={int(kk)}/{H})\n"
            f"mean pairwise overlap={mean_ov:.2f}  max cell usage={int(usage.max())}",
            transform=ax.transAxes, color="white", fontsize=8, va="top",
            bbox=dict(facecolor=(0, 0, 0, 0.6), edgecolor="none", pad=2))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
