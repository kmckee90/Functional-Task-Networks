"""Train the explorer + discover options over RECODE centroids.

Pipeline (one round, repeated R times -- honors "explores ... creates new
clusters ... continually trains offline to achieve old clusters too"):

  1. EXPLORE: run the env with the forward-biased novelty/DDPG explorer for
     `explore_steps` env steps (episodes truncate at `max_steps`, the agent's
     buffer is continuous with boundary flags). RECODE clusters online; the
     explorer is DDPG-updated on RECODE novelty reward periodically.
  2. SYNC: fold RECODE atoms into the stable `CentroidRegistry` (FPS to target).
  3. GRAPH: stream membership over the whole buffer -> rebuild the behavioral
     adjacency graph (episode boundaries break the chain).
  4. OPTIONS: for every centroid, reconfigure its FTN mask pair on the
     membership-relabeled data, then DDPG-train that masked subspace (with
     mask-overlap consolidation across sibling options). Old options retrained
     each round.

Outputs (to options/artifacts/): `checkpoint.pt` (agent: nets, masks, centroids,
footprints, RECODE state) and `graph.pt` (behavioral graph + centroid positions),
consumed by demo.py.

    uv run --project .. python train_options.py            # prototype
    uv run --project .. python train_options.py --smoke    # tiny, fast
"""

from __future__ import annotations

import argparse
import os
import time

# The FTN backbone runs many tiny tensor ops on CPU; without a thread cap torch
# oversubscribes all cores and thrashes (huge %CPU, ~no progress). Cap before
# torch initializes its thread pools.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

torch.set_num_threads(4)

from centroid_graph import BehavioralGraph
from continuous_world import ContinuousWorld
from options_agent import OptionsAgent
# All tunable parameters live in options_config.py (single source of truth).
from options_config import AgentConfig, TrainConfig, smoke_config

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def _fmt_dt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:d}m{s:02d}s" if m else f"{s:d}s"


def _random_action(agent) -> np.ndarray:
    """Forward-biased random action -- bypasses the net entirely (fast path)."""
    turn = np.random.uniform(-1.0, 1.0)
    accel = np.random.uniform(agent.cfg.forward_bias - 1.0, 1.0)
    return np.array([turn, accel], dtype=np.float32)


def explore(env, agent, n_steps, tc, rnd=0):
    """Run exploration for n_steps, logging progress every log_every_steps.

    With ``tc.random_explore`` the explorer neural net is bypassed completely:
    actions are a forward-biased random walk and NO explorer DDPG updates run --
    a fast way to collect coverage/clustering data. RECODE clustering and the
    buffer are still fed either way. Otherwise the FTN novelty/DDPG explorer runs."""
    mode = "random" if tc.random_explore else "policy"
    obs, info = env.reset()
    t_phase = time.time()
    for step in range(1, n_steps + 1):
        a = _random_action(agent) if tc.random_explore else agent.act_explore(obs)
        nobs, _, _, trunc, ninfo = env.step(a)
        agent.observe(obs, a, nobs, info["pos"], ninfo["pos"], float(trunc))
        if not tc.random_explore:
            agent.decay_epsilon()
            if len(agent.buffer) % tc.explorer_update_every == 0:
                agent.train_explorer(tc.explorer_updates)
        if trunc:
            obs, info = env.reset()
        else:
            obs, info = nobs, ninfo

        if step % tc.log_every_steps == 0 or step == n_steps:
            dt = time.time() - t_phase
            rate = step / max(dt, 1e-6)
            with torch.no_grad():
                nov = float(agent.recode.get_reward(
                    torch.as_tensor(np.asarray(nobs, np.float32), device=agent.device)[None, :]).mean())
            print(f"  [r{rnd + 1} explore:{mode}] {step}/{n_steps} steps  eps={agent.epsilon:.3f}  "
                  f"buffer={len(agent.buffer)}  recode={agent.recode.size}  novelty~{nov:.3f}  "
                  f"{rate:.0f} st/s  eta {_fmt_dt((n_steps - step) / max(rate, 1e-6))}", flush=True)


def build_graph(agent) -> BehavioralGraph:
    arr = agent.buffer.arrays()
    ids = agent.registry.membership(arr["next_obs"]).cpu().numpy()
    bnd = arr["boundary"].cpu().numpy()
    stream = [(-1 if bnd[i] else int(ids[i])) for i in range(len(ids))]
    return BehavioralGraph().build_from_memberships(stream)


def _new_topdown(env):
    """Open a (fig, ax) with the patchwork floor + obstacles drawn (zorder 0-1)."""
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
    """(ids, positions) for centroids that have a real footprint (skip never-visited)."""
    pos = agent.registry.positions_np()
    cnt = agent.registry.pos_count.detach().cpu().numpy()
    ids = [c for c in range(agent.registry.n) if cnt[c] > 0]
    return ids, pos


def save_centroids_png(env, agent, path: str) -> None:
    """Top-down view of the RECODE centroids at their mean physical footprints."""
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


def save_graph_png(env, agent, graph, path: str) -> None:
    """Top-down view of the behavioral adjacency graph over the centroids
    (edge width scales with transition count)."""
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


def save_exploration_png(env, agent, path: str) -> None:
    """Plot the agent's full exploration trajectory over the patchwork floor.

    Uses the true (x, y) poses stored in the buffer (NOT available to the agent).
    The path is split at episode boundaries (so resets don't draw teleport lines)
    and each episode is colored by its time order (early -> late) to show how
    coverage builds up. Only producible during training (the buffer is in-memory,
    not saved to the checkpoint)."""
    import matplotlib.pyplot as plt

    arr = agent.buffer.arrays()
    pos = arr["next_pos"].cpu().numpy()
    bnd = arr["boundary"].cpu().numpy()
    if len(pos) < 2:
        return

    fig, ax = _new_topdown(env)
    # Split into episodes at boundaries; color each by time order.
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


@torch.no_grad()
def eval_option(env, agent, cid, episodes, horizon) -> float:
    """Fraction of probe rollouts (from random starts) in which firing option
    `cid` lands the agent in centroid `cid` within `horizon` steps."""
    hits = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        for _ in range(horizon):
            if agent.membership_np(obs) == cid:
                hits += 1
                break
            obs, _, _, trunc, _ = env.step(agent.act_option(obs, cid))
            if trunc:
                obs, _ = env.reset()
    return hits / max(1, episodes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--random-explore", action="store_true",
                   help="bypass the explorer net: forward-biased random walk, no explorer DDPG (fast)")
    args = p.parse_args()

    if args.smoke:
        tc, ac = smoke_config()
    else:
        tc, ac = TrainConfig(), AgentConfig()
    tc.seed = args.seed
    if args.random_explore:
        tc.random_explore = True
    ac.device = args.device

    np.random.seed(tc.seed)
    torch.manual_seed(tc.seed)

    env = ContinuousWorld(max_steps=tc.max_steps, seed=tc.seed)
    ac.obs_dim = env.obs_dim
    ac.act_dim = env.action_space.shape[0]
    agent = OptionsAgent(ac)

    os.makedirs(ART, exist_ok=True)
    print(f"config: device={ac.device} rounds={tc.rounds} explore_steps={tc.explore_steps} "
          f"explore={'random' if tc.random_explore else 'policy'} "
          f"H={ac.ftn_hidden_size} k={ac.ftn_k} target_centroids={ac.target_centroids} "
          f"option_reconfig={tc.option_reconfig} option_train_steps={tc.option_train_steps}", flush=True)
    t0 = time.time()
    for rnd in range(tc.rounds):
        print(f"\n=== round {rnd + 1}/{tc.rounds} ===", flush=True)
        explore(env, agent, tc.explore_steps, tc, rnd)

        added = agent.registry.sync(agent.recode)
        print(f"  [r{rnd + 1} sync] +{added} centroids -> {agent.registry.n} total "
              f"(recode atoms={agent.recode.size})", flush=True)
        # Refresh footprints over the whole buffer.
        arr = agent.buffer.arrays()
        agent.registry.update_footprints(arr["next_obs"], arr["next_pos"])
        graph = build_graph(agent)
        # Emit the top-down centroid + graph + trajectory images NOW, before the
        # long option-training phase, so they can be inspected immediately.
        save_centroids_png(env, agent, os.path.join(ART, "centroids.png"))
        save_graph_png(env, agent, graph, os.path.join(ART, "graph.png"))
        save_exploration_png(env, agent, os.path.join(ART, "exploration.png"))
        print(f"  [r{rnd + 1} graph] {len(graph.nodes)} nodes, {len(graph.edges)} edges  "
              f"(wrote centroids.png, graph.png, exploration.png)", flush=True)

        # Train one option per centroid (the long phase) with progress + ETA.
        n_opt = agent.registry.n
        t_opt = time.time()
        recon = []
        print(f"  [r{rnd + 1} options] training {n_opt} options "
              f"({'reconfig+' if tc.option_reconfig else 'fixed-mask, '}{tc.option_train_steps} DDPG steps each)...",
              flush=True)
        for i, cid in enumerate(range(n_opt), 1):
            if tc.option_reconfig:
                st = agent.reconfig_option(cid)
                recon.append(st["critic"][1])
            agent.train_option(cid, tc.option_train_steps)
            if i % tc.log_every_options == 0 or i == n_opt:
                dt = time.time() - t_opt
                rate = i / max(dt, 1e-6)
                tail = f"  last reconfig_r={-recon[-1]:.2f}" if recon else ""
                print(f"    [r{rnd + 1} options] {i}/{n_opt}  {rate:.2f} opt/s  "
                      f"elapsed {_fmt_dt(dt)}  eta {_fmt_dt((n_opt - i) / max(rate, 1e-6))}{tail}",
                      flush=True)

        # Optional reach-rate probes on a few options.
        reach = []
        if tc.eval_option_episodes > 0 and agent.registry.n > 0:
            probe_ids = list(range(min(agent.registry.n, 5)))
            for cid in probe_ids:
                reach.append(eval_option(env, agent, cid, tc.eval_option_episodes, tc.eval_option_horizon))

        elapsed = time.time() - t0
        msg = (f"[round {rnd + 1}/{tc.rounds}] eps={agent.epsilon:.2f} "
               f"buffer={len(agent.buffer)} recode={agent.recode.size} "
               f"centroids={agent.registry.n}(+{added}) "
               f"edges={len(graph.edges)}")
        if recon:
            msg += f" reconfig_r={np.mean(recon):.2f}"
        if reach:
            msg += f" reach@probe={np.mean(reach):.2f}"
        msg += f"  ({elapsed:.0f}s)"
        print(msg, flush=True)

        # Checkpoint after every round.
        agent.save(os.path.join(ART, "checkpoint.pt"))
        torch.save({"graph": graph.to_dict(),
                    "centroid_pos": agent.registry.positions_np(),
                    "n_centroids": agent.registry.n},
                   os.path.join(ART, "graph.pt"))

    print(f"done. wrote {ART}/checkpoint.pt, graph.pt, centroids.png, graph.png, exploration.png", flush=True)


if __name__ == "__main__":
    main()
