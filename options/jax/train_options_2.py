"""JAX port of options/train_options_2.py: parallel collection + SEQUENTIAL,
multi-epoch option training, end to end on-device.

Per round:
  1. COLLECT  -- one jitted vmap+scan random walk (collect.collect_parallel).
  2. RECODE   -- one jitted scan folds next_obs into the atom memory.
  3. CENTROIDS+GRAPH -- host FPS registry sync, footprints, behavioral graph, PNGs.
  4. OPTIONS  -- `option_epochs` passes; each pass trains options ONE AT A TIME
                 (reconfig mask cid -> DDPG-train cid) so each option capitalizes
                 on the shared backbone the earlier options just moved. Every
                 reconfig + train phase is a single fused jit call.

The two heavy phases (collection, option training) are each a JIT loop -- the
Python level only dispatches a handful of compiled programs per round.

    uv run --project ../.. python train_options_2.py            # full
    uv run --project ../.. python train_options_2.py --smoke    # tiny, fast
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

# Reuse the framework-neutral behavioral graph + Dijkstra from the parent package.
# APPEND (not insert) so local jax/ modules (color_world, recode, ...) win over the
# parent options/ modules of the same name.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from centroid_graph import BehavioralGraph

import collect as collect_mod
import recode as recode_mod
import viz
from agent import OptionsAgent
from color_world import ColorWorld
from config import AgentConfig, ParConfig, TrainConfig, smoke_config

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def _fmt_dt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:d}m{s:02d}s" if m else f"{s:d}s"


def build_graph(agent, next_obs_np, boundary_np, pos_np=None, next_pos_np=None,
                world_size=None) -> BehavioralGraph:
    """Behavioral graph over the centroid memberships. The plan graph is kept in
    the SQUARE even when the env is a torus: a transition that wraps a seam (real
    per-step displacement is << size, so any per-axis jump > size/2 is a wrap) is
    flagged as a boundary so it forms NO edge -- otherwise wrap edges would short-
    circuit the map and make plans trivial. Toroidal dynamics/training are
    unaffected; only graph edges are suppressed."""
    ids = agent.registry.membership_np(next_obs_np)
    break_at = np.asarray(boundary_np).astype(bool)
    if pos_np is not None and next_pos_np is not None and world_size is not None:
        wrap = (np.abs(np.asarray(next_pos_np) - np.asarray(pos_np)) > 0.5 * world_size).any(axis=1)
        break_at = break_at | wrap
    stream = [(-1 if break_at[i] else int(ids[i])) for i in range(len(ids))]
    return BehavioralGraph().build_from_memberships(stream)


def train_options_sequential(agent, tc, buffer, rnd, epochs):
    """`epochs` passes of sequential per-option (reconfig -> DDPG-train).

    One option at a time so each sees the backbone updated by the earlier ones;
    the epoch loop revisits option 0 against the later, co-adapted backbone."""
    n_opt = agent.registry.n
    if n_opt == 0:
        return
    tag = "reconfig" if tc.option_reconfig else "fixed-mask"
    for epoch in range(1, epochs + 1):
        t_ep = time.time()
        recon = []
        for i, cid in enumerate(range(n_opt), 1):
            if tc.option_reconfig:
                c0, c1 = agent.reconfig_option(cid, buffer)
                recon.append(c1)
            else:
                agent.ensure_option(cid)
            agent.train_option(cid, buffer)
            if i % tc.log_every_options == 0 or i == n_opt:
                dt = time.time() - t_ep
                rate = i / max(dt, 1e-6)
                last = f"  last reconfig_r={-recon[-1]:.2f}" if recon else ""
                print(f"    [r{rnd + 1} ep{epoch} options] {i}/{n_opt}  {rate:.2f} opt/s  "
                      f"elapsed {_fmt_dt(dt)}  eta {_fmt_dt((n_opt - i) / max(rate, 1e-6))}{last}",
                      flush=True)
        ov = agent.mean_overlap()
        print(f"  [r{rnd + 1} epoch {epoch}/{epochs} {tag}] trained {n_opt} options "
              f"sequentially in {_fmt_dt(time.time() - t_ep)}  mean_overlap={ov:.2f}", flush=True)


def main():
    d = ParConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=d.workers,
                   help="parallel random-walk collection worker chunks (vmapped)")
    p.add_argument("--option-epochs", type=int, default=d.option_epochs,
                   help="passes over the options; each pass trains them sequentially (reconfig->train per option)")
    p.add_argument("--n-cells", type=int, default=d.n_cells)
    p.add_argument("--color-dim", type=int, default=d.color_dim)
    p.add_argument("--max-speed", type=float, default=d.max_speed)
    p.add_argument("--torus", action=argparse.BooleanOptionalAction, default=d.torus,
                   help="wrap the world edges (no boundaries) during collection (--no-torus to clip at walls)")
    args = p.parse_args()

    pc = ParConfig(workers=args.workers, option_epochs=args.option_epochs,
                   n_cells=args.n_cells, color_dim=args.color_dim, max_speed=args.max_speed,
                   torus=args.torus)

    if args.smoke:
        tc, ac = smoke_config()
    else:
        tc, ac = TrainConfig(), AgentConfig()
    tc.seed = args.seed
    tc.random_explore = True
    # Color-space distances aren't spatially meaningful -> membership reward only.
    ac.option_shaping = 0.0

    # World (identical patchwork to the torch env for the same world_seed).
    ms = 0.6 if pc.max_speed is None else pc.max_speed
    env = ColorWorld(world_seed=pc.world_seed, n_cells=pc.n_cells, color_dim=pc.color_dim,
                     max_steps=tc.max_steps, max_speed=ms, torus=pc.torus)
    env_kwargs = env.config()
    ac.obs_dim = env.obs_dim
    ac.act_dim = 2
    agent = OptionsAgent(ac, seed=tc.seed, option_train_steps=tc.option_train_steps)

    os.makedirs(ART, exist_ok=True)
    print(f"config[jax]: backend={jax.default_backend()} workers={pc.workers} torus={pc.torus} "
          f"option_epochs={pc.option_epochs} rounds={tc.rounds} explore_steps={tc.explore_steps} "
          f"H={ac.ftn_hidden_size} k={ac.ftn_k} target_centroids={ac.target_centroids} "
          f"option_reconfig={tc.option_reconfig} option_train_steps={tc.option_train_steps}", flush=True)

    key = jax.random.PRNGKey(tc.seed)
    buf = None  # (obs, act, next_obs, pos, next_pos, boundary) jnp arrays
    t0 = time.time()
    for rnd in range(tc.rounds):
        print(f"\n=== round {rnd + 1}/{tc.rounds} ===", flush=True)

        # 1) Parallel random-walk collection (JIT vmap+scan) --------------
        t_c = time.time()
        key, ck = jax.random.split(key)
        arrays = collect_mod.collect_parallel(ck, env, tc.explore_steps, pc.workers, ac.history_len)
        jax.block_until_ready(arrays)
        if buf is None:
            buf = arrays
        else:
            cap = ac.buffer_capacity
            buf = tuple(jnp.concatenate([b, a], axis=0)[:cap] for b, a in zip(buf, arrays))
        n_added = int(arrays[0].shape[0])
        # Feed RECODE the collected next_obs in one jitted scan.
        agent.recode_update(arrays[2])
        rsize = int(agent.recode["size"])
        dt = time.time() - t_c
        print(f"  [r{rnd + 1} collect] {n_added} steps via {pc.workers} workers in {_fmt_dt(dt)} "
              f"({n_added / max(dt, 1e-6):.0f} st/s)  buffer={buf[0].shape[0]} recode={rsize}", flush=True)

        # 2) Centroids + graph + PNGs ------------------------------------
        atoms = recode_mod.all_pos(agent.recode)
        added = agent.registry.sync(atoms)
        next_obs_np = np.asarray(buf[2])
        pos_np = np.asarray(buf[3])
        next_pos_np = np.asarray(buf[4])
        boundary_np = np.asarray(buf[5])
        agent.registry.update_footprints(next_obs_np, next_pos_np)
        # On a torus, suppress seam-wrapping transitions so the PLAN graph stays in
        # the square (toroidal dynamics/training are unaffected).
        graph = build_graph(agent, next_obs_np, boundary_np,
                            pos_np=pos_np if env.torus else None,
                            next_pos_np=next_pos_np if env.torus else None,
                            world_size=env.size if env.torus else None)
        agent.buffer_next_pos = next_pos_np
        agent.buffer_boundary = boundary_np
        viz.save_centroids_png(env, agent, os.path.join(ART, "centroids.png"))
        viz.save_graph_png(env, agent, graph, os.path.join(ART, "graph.png"))
        viz.save_exploration_png(env, agent, os.path.join(ART, "exploration.png"))
        print(f"  [r{rnd + 1} graph] +{added} centroids -> {agent.registry.n}, "
              f"{len(graph.edges)} edges  (wrote centroids.png, graph.png, exploration.png)", flush=True)

        # 3) Sequential option training (multi-epoch co-adaptation) ------
        # Train on the history-AUGMENTED obs (slots 6/7); raw obs/next_obs (0/2)
        # still feed RECODE/graph above. With history_len=0 aug == raw.
        train_buf = (buf[6], buf[1], buf[7], buf[5])  # aug_obs, act, aug_next, boundary
        train_options_sequential(agent, tc, train_buf, rnd, pc.option_epochs)
        viz.save_masks_png(agent, os.path.join(ART, "masks.png"))
        print(f"  [r{rnd + 1} masks] wrote masks.png ({agent.registry.n} option masks overlaid)", flush=True)

        elapsed = time.time() - t0
        print(f"[round {rnd + 1}/{tc.rounds}] centroids={agent.registry.n} edges={len(graph.edges)} "
              f"buffer={buf[0].shape[0]}  ({_fmt_dt(elapsed)})", flush=True)

        # Checkpoints (pickled host arrays; .pkl to not clash with the torch .pt).
        ckpt = {
            "actor": jax.device_get(agent.actor), "critic": jax.device_get(agent.critic),
            "actor_t": jax.device_get(agent.actor_t), "critic_t": jax.device_get(agent.critic_t),
            "options": {c: {k: np.asarray(v) for k, v in o.items()} for c, o in agent.options.items()},
            "centroids": agent.registry.centroids, "centroid_pos": agent.registry.positions_np(),
            "cfg": ac,
        }
        with open(os.path.join(ART, "checkpoint.pkl"), "wb") as f:
            pickle.dump(ckpt, f)
        with open(os.path.join(ART, "graph.pkl"), "wb") as f:
            pickle.dump({"graph": graph.to_dict(), "centroid_pos": agent.registry.positions_np(),
                         "n_centroids": agent.registry.n, "env": "ColorWorld", "env_kwargs": env_kwargs}, f)

    print(f"done. wrote {ART}/checkpoint.pkl, graph.pkl, centroids.png, graph.png, exploration.png, masks.png",
          flush=True)


if __name__ == "__main__":
    main()
