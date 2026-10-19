"""Parallel-collection variant of train_options.py with SEQUENTIAL mask training.

This is train_options_par.py but with the option-training phase reverted to the
serial scheme of train_options.py. Everything else (RECODE clustering, centroid
registry, behavioral graph, checkpoints, the top-down PNGs) reuses train_options.

1. DATA COLLECTION (multiprocessing). With a forward-biased RANDOM walk the env
   step is pure-CPU ray casting and embarrassingly parallel. `--workers W` spawns
   W processes (torch-free, see `_par_collect.py`), each running its own copy of
   the SAME world (fixed `world_seed`) for a slice of the step budget. Their
   transitions are concatenated and bulk-inserted into the agent's buffer; RECODE
   is then fed the batch on the (GPU) main process. This variant is random-walk
   only -- the neural explorer is inherently sequential, so it is not ported here.

2. OPTION TRAINING (sequential, multi-epoch). Unlike train_options_par.py, which
   reconfigures ALL masks then DDPG-trains them simultaneously, this trains the
   options ONE AT A TIME: reconfig mask 0 -> train option 0, reconfig mask 1 ->
   train option 1, ... Because all options share ONE FTN backbone, each
   subsequent option's mask search and DDPG updates see the backbone already
   updated by the earlier options, so it can capitalize on their solutions. The
   `--option-epochs` loop then loops back to option 0 so early options get
   retrained against the later, co-adapted backbone.

    uv run --project .. python train_options_2.py --workers 16 --device cuda
    uv run --project .. python train_options_2.py --smoke --workers 4 --device cpu
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

torch.set_num_threads(4)

from _par_collect import collect_worker
from envs import make_env
from options_agent import OptionsAgent
# All tunable parameters live in options_config.py (single source of truth).
from options_config import AgentConfig, ParConfig, TrainConfig, smoke_config
# Reuse all shared machinery from the serial trainer (graph build, PNG dumps, etc).
from train_options import (
    _fmt_dt,
    build_graph,
    save_centroids_png,
    save_exploration_png,
    save_graph_png,
)

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


# ---------------------------------------------------------------------------
# 1. Parallel random-walk collection
# ---------------------------------------------------------------------------


def collect_parallel(total_steps, env_name, world_kwargs, workers, base_seed):
    """Run `workers` random-walk processes covering `total_steps` env steps total.
    Returns concatenated (obs, act, next_obs, pos, next_pos, boundary) arrays."""
    per = total_steps // workers
    chunks = [per] * workers
    chunks[-1] += total_steps - per * workers  # remainder to the last worker
    args = [(env_name, world_kwargs, chunks[i], base_seed * 100003 + i) for i in range(workers)]
    if workers == 1:
        results = [collect_worker(args[0])]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers) as pool:
            results = pool.map(collect_worker, args)
    return tuple(np.concatenate([r[f] for r in results], axis=0) for f in range(6))


def bulk_add(buffer, arrays, device):
    """Slice-assign a batch of transitions into the preallocated EpisodeBuffer."""
    O, A, NO, P, NP, B = arrays
    n = min(len(O), buffer.capacity - buffer.size)
    if n <= 0:
        print("[buffer] capacity reached; dropping collected transitions")
        return 0
    sl = slice(buffer.size, buffer.size + n)
    buffer.obs[sl] = torch.as_tensor(O[:n], device=device)
    buffer.act[sl] = torch.as_tensor(A[:n], device=device)
    buffer.next_obs[sl] = torch.as_tensor(NO[:n], device=device)
    buffer.pos[sl] = torch.as_tensor(P[:n], device=device)
    buffer.next_pos[sl] = torch.as_tensor(NP[:n], device=device)
    buffer.boundary[sl] = torch.as_tensor(B[:n], device=device)
    buffer.size += n
    return n


# ---------------------------------------------------------------------------
# 2. Sequential (one-option-at-a-time) DDPG training, multi-epoch
# ---------------------------------------------------------------------------


def _mean_overlap(agent, n_opt, which="m_a"):
    """Mean off-diagonal pairwise mask overlap (fraction of k shared)."""
    if n_opt < 2:
        return 0.0
    M = torch.stack([agent.options[c][which] for c in range(n_opt)]).float()
    inter = (M @ M.t()) / M[0].sum().clamp(min=1)
    off = inter[~torch.eye(n_opt, dtype=torch.bool, device=inter.device)]
    return float(off.mean().item())


def train_options_sequential(agent, tc, rnd, epochs):
    """Run `epochs` passes of sequential per-option (reconfig -> DDPG-train).

    Each epoch walks the options in order and trains them ONE AT A TIME:
    reconfig mask cid -> DDPG-train option cid -> next cid. Because all options
    share ONE FTN backbone, each subsequent option sees the backbone already
    updated by the earlier options this epoch, so it can capitalize on their
    solutions (rather than every mask being reconfigured against the same frozen
    backbone and then trained simultaneously, as in train_options_par.py). The
    `epochs` loop then loops back to option 0 so early options get retrained
    against the later, co-adapted backbone. (Needs option_reconfig=True for the
    re-search; with fixed random masks the epochs just keep training the same
    subspaces.) Mean mask overlap is logged per epoch so the consolidation/
    segregation trend is visible."""
    n_opt = agent.registry.n
    if n_opt == 0 or agent.buffer.size < agent.cfg.batch_size:
        return
    tag = "reconfig" if tc.option_reconfig else "fixed-mask"
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        recon = []
        for i, cid in enumerate(range(n_opt), 1):
            if tc.option_reconfig:
                st = agent.reconfig_option(cid)
                recon.append(st["critic"][1])
            else:
                agent.ensure_option(cid)
            agent.train_option(cid, tc.option_train_steps)
            if i % tc.log_every_options == 0 or i == n_opt:
                dt = time.time() - t_ep
                rate = i / max(dt, 1e-6)
                last = f"  last reconfig_r={-recon[-1]:.2f}" if recon else ""
                print(f"    [r{rnd + 1} ep{epoch} options] {i}/{n_opt}  {rate:.2f} opt/s  "
                      f"elapsed {_fmt_dt(dt)}  eta {_fmt_dt((n_opt - i) / max(rate, 1e-6))}{last}",
                      flush=True)
        ov = _mean_overlap(agent, n_opt)
        print(f"  [r{rnd + 1} epoch {epoch}/{epochs} {tag}] trained {n_opt} options "
              f"sequentially in {_fmt_dt(time.time() - t_ep)}  mean_overlap={ov:.2f}", flush=True)


# ---------------------------------------------------------------------------
# Mask-overlap diagnostic image
# ---------------------------------------------------------------------------


def _mask_grid(m_np, sq, H):
    """Lay a flat mask on the SmoothKWTA square grid (sq x n_blocks*sq) when H is
    a multiple of sq**2, else a near-square padded reshape."""
    if H % (sq * sq) == 0:
        from ftn import mask_to_grid
        return mask_to_grid(torch.as_tensor(m_np), sq).cpu().numpy()
    c = int(np.ceil(np.sqrt(H)))
    r = int(np.ceil(H / c))
    pad = np.zeros(r * c, np.float32)
    pad[:H] = m_np
    return pad.reshape(r, c)


def save_masks_png(agent, path, which="m_a", alpha=0.3):
    """Overlay every trained option mask on the square grid, each a different
    random color at low alpha, so overlap is visible (cells used by many options
    blend toward a muddy mix; isolated cells keep a clean hue)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cids = sorted(agent.options.keys())
    if not cids:
        return
    H, sq = agent.cfg.ftn_hidden_size, agent.cfg.lateral_square_dim
    g0 = _mask_grid(agent.options[cids[0]][which].detach().cpu().numpy(), sq, H)
    gh, gw = g0.shape
    rng = np.random.default_rng(0)
    canvas = np.full((gh, gw, 3), 0.08, dtype=np.float32)  # dark background
    usage = np.zeros((gh, gw), dtype=np.float32)           # #options per cell
    for cid in cids:
        g = _mask_grid(agent.options[cid][which].detach().cpu().numpy(), sq, H)
        color = rng.uniform(0.25, 1.0, size=3).astype(np.float32)
        a = alpha * (g > 0.5)[..., None]
        canvas = canvas * (1 - a) + color[None, None, :] * a
        usage += (g > 0.5)

    # Overlap stats.
    masks = torch.stack([agent.options[c][which] for c in cids]).float()
    k = float(masks[0].sum().item())
    inter = (masks @ masks.t()) / max(k, 1.0)
    n = len(cids)
    off = inter[~torch.eye(n, dtype=torch.bool, device=inter.device)]
    mean_ov = float(off.mean().item()) if n > 1 else 0.0

    fig = plt.figure(figsize=(max(5, gw / gh * 5), 5), dpi=130)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(canvas, origin="lower", interpolation="nearest")
    ax.text(0.01, 0.99, f"{n} option {which} masks on {gh}x{gw} grid  (k={int(k)}/{H})\n"
            f"mean pairwise overlap={mean_ov:.2f}  max cell usage={int(usage.max())}",
            transform=ax.transAxes, color="white", fontsize=8, va="top",
            bbox=dict(facecolor=(0, 0, 0, 0.6), edgecolor="none", pad=2))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # CLI flag defaults come from ParConfig (the single source of truth); flags
    # override individual fields, then we rebuild `pc` from the parsed values.
    d = ParConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=d.workers,
                   help="parallel random-walk collection processes")
    p.add_argument("--par-batch", type=int, default=d.par_batch,
                   help="per-option batch size in the vectorized option step")
    p.add_argument("--option-epochs", type=int, default=d.option_epochs,
                   help="passes over the options; each pass trains them sequentially (reconfig->train per option) so they co-adapt")
    p.add_argument("--env", default=d.env, choices=["color", "continuous"],
                   help="which world to train in (color=ColorWorld, continuous=ContinuousWorld)")
    p.add_argument("--n-cells", type=int, default=d.n_cells, help="ColorWorld Voronoi cells (color env only)")
    p.add_argument("--color-dim", type=int, default=d.color_dim, help="ColorWorld obs/color vector length (>=3; color env only)")
    p.add_argument("--max-speed", type=float, default=d.max_speed,
                   help="per-step displacement scale (default = env default); recorded for the demo")
    args = p.parse_args()

    pc = ParConfig(workers=args.workers, par_batch=args.par_batch,
                   option_epochs=args.option_epochs, env=args.env, n_cells=args.n_cells,
                   color_dim=args.color_dim, max_speed=args.max_speed)

    if args.smoke:
        tc, ac = smoke_config()
    else:
        tc, ac = TrainConfig(), AgentConfig()
    tc.seed = args.seed
    tc.random_explore = True  # this variant is random-walk collection only
    ac.device = args.device
    if pc.env == "color":
        # ColorWorld colors are random per-cell, so observation-space distances are
        # NOT spatially meaningful and distance shaping would mislead -- use pure
        # membership reward. ContinuousWorld keeps the AgentConfig default shaping.
        ac.option_shaping = 0.0

    np.random.seed(tc.seed)
    torch.manual_seed(tc.seed)

    # Build the selected world. Shared kwargs apply to both; color-only shape
    # params (n_cells/color_dim) are passed for ColorWorld, while ContinuousWorld
    # uses its own (denser) defaults plus any max_speed override.
    common = dict(world_seed=pc.world_seed, max_steps=tc.max_steps, seed=tc.seed)
    if pc.max_speed is not None:
        common["max_speed"] = pc.max_speed
    if pc.env == "color":
        env = make_env("color", n_cells=pc.n_cells, color_dim=pc.color_dim, **common)
    else:
        env = make_env("continuous", **common)
    # FULL env config (incl. max_speed/size) -> workers + demo rebuild identically.
    env_kwargs = env.config()
    ac.obs_dim = env.obs_dim
    ac.act_dim = env.action_space.shape[0]
    agent = OptionsAgent(ac)

    os.makedirs(ART, exist_ok=True)
    print(f"config[par]: env={pc.env} obs_dim={ac.obs_dim} act_dim={ac.act_dim} "
          f"device={ac.device} workers={pc.workers} par_batch={pc.par_batch} "
          f"option_epochs={pc.option_epochs} rounds={tc.rounds} explore_steps={tc.explore_steps} "
          f"H={ac.ftn_hidden_size} k={ac.ftn_k} target_centroids={ac.target_centroids} "
          f"option_reconfig={tc.option_reconfig} option_train_steps={tc.option_train_steps} "
          f"option_shaping={ac.option_shaping}", flush=True)

    t0 = time.time()
    for rnd in range(tc.rounds):
        print(f"\n=== round {rnd + 1}/{tc.rounds} ===", flush=True)

        # 1) Parallel random-walk collection ------------------------------
        t_c = time.time()
        arrays = collect_parallel(tc.explore_steps, pc.env, env_kwargs, pc.workers, tc.seed + rnd)
        n_added = bulk_add(agent.buffer, arrays, agent.device)
        # Feed RECODE the collected next_obs in one batch (clustering source).
        agent.recode.update(torch.as_tensor(arrays[2], device=agent.device))
        agent.recode.step()
        dt = time.time() - t_c
        print(f"  [r{rnd + 1} collect] {n_added} steps via {pc.workers} workers in {_fmt_dt(dt)} "
              f"({n_added / max(dt, 1e-6):.0f} st/s)  buffer={agent.buffer.size} recode={agent.recode.size}",
              flush=True)

        # 2) Centroids + graph + PNGs (before the long option phase) ------
        added = agent.registry.sync(agent.recode)
        arr = agent.buffer.arrays()
        agent.registry.update_footprints(arr["next_obs"], arr["next_pos"])
        graph = build_graph(agent)
        save_centroids_png(env, agent, os.path.join(ART, "centroids.png"))
        save_graph_png(env, agent, graph, os.path.join(ART, "graph.png"))
        save_exploration_png(env, agent, os.path.join(ART, "exploration.png"))
        print(f"  [r{rnd + 1} graph] +{added} centroids -> {agent.registry.n}, "
              f"{len(graph.edges)} edges  (wrote centroids.png, graph.png, exploration.png)", flush=True)

        # 3) Sequential option training (multi-epoch co-adaptation) -------
        train_options_sequential(agent, tc, rnd, pc.option_epochs)
        save_masks_png(agent, os.path.join(ART, "masks.png"))
        print(f"  [r{rnd + 1} masks] wrote masks.png ({agent.registry.n} option masks overlaid)", flush=True)

        elapsed = time.time() - t0
        print(f"[round {rnd + 1}/{tc.rounds}] centroids={agent.registry.n} edges={len(graph.edges)} "
              f"buffer={agent.buffer.size}  ({_fmt_dt(elapsed)})", flush=True)

        agent.save(os.path.join(ART, "checkpoint.pt"))
        torch.save({"graph": graph.to_dict(),
                    "centroid_pos": agent.registry.positions_np(),
                    "n_centroids": agent.registry.n,
                    "env": pc.env, "env_kwargs": env_kwargs},
                   os.path.join(ART, "graph.pt"))

    print(f"done. wrote {ART}/checkpoint.pt, graph.pt, centroids.png, graph.png, exploration.png, masks.png", flush=True)


if __name__ == "__main__":
    main()
