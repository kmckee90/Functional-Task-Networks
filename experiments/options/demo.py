"""Plan over the centroid graph and execute the plan by firing FTN options.

Loads a trained checkpoint (agent + behavioral graph), then:
  1. resets the world and reads the agent's CURRENT centroid membership;
  2. picks a RANDOM goal centroid (reachable in the graph);
  3. plans a path with Dijkstra over the behavioral adjacency graph;
  4. executes CLOSED-LOOP: every `replan_every` steps it re-reads the current
     centroid membership, re-runs Dijkstra to the goal, and fires the option
     (actor mask) of the NEXT centroid on the fresh path -- so the active mask is
     re-determined as the agent moves, until it reaches the goal (or times out).

Renders the whole top-down world each step with the centroid graph + edges, the
planned path highlighted, the live current-membership marker, and the goal
marker, and writes an animated GIF to options/artifacts/demo.gif.

    uv run --project .. python demo.py
    uv run --project .. python demo.py --seed 3 --goal 7
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from PIL import Image

from centroid_graph import BehavioralGraph
from envs import env_key, make_env
from options_agent import OptionsAgent
# All tunable parameters live in options_config.py (single source of truth);
# DemoConfig documents each flag below and supplies its default.
from options_config import DemoConfig

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def load(device="cpu"):
    agent = OptionsAgent.load(os.path.join(ART, "checkpoint.pt"), map_location=device)
    gd = torch.load(os.path.join(ART, "graph.pt"), weights_only=False)
    graph = BehavioralGraph.from_dict(gd["graph"])
    env_name = env_key(gd.get("env"))      # which world was trained ("color"/"continuous")
    env_kwargs = gd.get("env_kwargs", {})  # constructor params used at training time
    return agent, graph, env_name, env_kwargs


# Each --flag below is documented in options_config.DemoConfig (the parameter
# glossary) and takes its default from that dataclass; flags override per-field.
def main():
    d = DemoConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--tours", type=int, default=d.tours, help="number of random plan->execute cycles to chain")
    p.add_argument("--goal", type=int, default=d.goal, help="fix the FIRST tour's goal (-1 = random)")
    p.add_argument("--replan-every", type=int, default=d.replan_every,
                   help="re-plan (membership->Dijkstra->next-centroid mask) every N steps")
    p.add_argument("--tour-timeout", type=int, default=d.tour_timeout, help="max steps per tour to reach the goal")
    p.add_argument("--exec-noise", type=float, default=d.exec_noise, help="action noise during option execution (helps unstick)")
    p.add_argument("--max-steps", type=int, default=d.max_steps)
    p.add_argument("--render-every", type=int, default=d.render_every)
    p.add_argument("--continuous", action="store_true", default=d.continuous,
                   help="legs continue from the previous pose instead of restarting at a random non-obstacle location")
    p.add_argument("--max-speed", type=float, default=d.max_speed,
                   help="override env max_speed (use to match training for checkpoints saved before it was recorded)")
    p.add_argument("--out", default=d.out or os.path.join(ART, "demo.gif"))
    args = p.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent, graph, env_name, env_kwargs = load()
    # A checkpoint trained before options had masks (e.g. option_reconfig=False
    # without the lazy ensure_option fix) has no options -> planning would KeyError
    # mid-run. Fail early with an actionable message instead.
    if not agent.options:
        raise SystemExit(
            "checkpoint has no trained options (agent.options is empty). Re-run "
            "train_options.py with the current code so every centroid gets an "
            "option mask (reconfig on, or lazy random masks when reconfig is off)."
        )
    missing = [c for c in sorted(graph.nodes) if c not in agent.options]
    if missing:
        print(f"WARNING: {len(missing)} graph centroids have no option {missing[:10]}"
              f"{'...' if len(missing) > 10 else ''}; planning will avoid them.")
    pos = agent.registry.positions_np()                 # (N, 2) footprints
    centroid_ids = list(range(agent.registry.n))
    centroids_xy = [tuple(pos[i]) for i in centroid_ids]
    edges_xy = [(tuple(pos[a]), tuple(pos[b])) for a, b, _ in graph.undirected_edge_list()]

    # Plan only over centroids that actually have an option (render still shows
    # the full graph above). Drops missing-option nodes from the planning graph.
    if missing:
        mset = set(missing)
        pg = BehavioralGraph()
        pg.nodes = set(graph.nodes) - mset
        pg.edges = {(a, b): w for (a, b), w in graph.edges.items()
                    if a not in mset and b not in mset}
        graph = pg

    # Rebuild the SAME world used in training (env_name + env_kwargs from graph.pt),
    # but with a long horizon so legs aren't truncated. max_speed MUST match
    # training or the policy's per-step displacement will be mis-scaled.
    ekw = {**env_kwargs, "max_steps": 10_000}
    if args.max_speed is not None:
        ekw["max_speed"] = args.max_speed
    elif "max_speed" not in env_kwargs:
        print(f"WARNING: this checkpoint's graph.pt did not record max_speed; using the "
              f"{env_name} env default. If training used a different value the demo will be "
              f"mis-scaled -- pass --max-speed <training value> (or retrain to record it).")
    env = make_env(env_name, **ekw, seed=args.seed)
    print(f"demo env: {env_name} max_speed={env.max_speed} n_cells={env.n_cells} "
          f"world_seed={env.world_seed}", flush=True)
    obs, info = env.reset(seed=args.seed)

    # Settle to a defined membership.
    cur = agent.membership_np(obs)
    for _ in range(20):
        if cur >= 0:
            break
        obs, _, _, _, info = env.step(np.array([0.3, 0.5], np.float32))
        cur = agent.membership_np(obs)

    # Mutable state shared with snap() (caption/path/goal change per tour).
    frames: list[np.ndarray] = []
    trail: list = [tuple(info["pos"])]
    state = {"plan": [cur], "goal": cur, "caption": ""}

    def snap():
        path_xy = [tuple(pos[c]) for c in state["plan"]]
        frames.append(env.render_world(
            centroids=centroids_xy, centroid_ids=centroid_ids, edges=edges_xy,
            path=path_xy, trail=trail[-400:], current_id=agent.membership_np(obs),
            goal_id=state["goal"], agent_pos=info["pos"], agent_heading=info["heading"],
            caption=state["caption"]))

    def pick_goal(src):
        """Random reachable goal, preferring a non-trivial (multi-leg) one."""
        reach = sorted(graph.reachable_from(src))
        if not reach:
            return None
        for _ in range(8):
            g = int(np.random.choice(reach))
            pl = graph.dijkstra(src, g)
            if pl and len(pl) >= 2:
                return g
        return int(np.random.choice(reach))

    total = 0
    n_success = 0
    for tour in range(1, args.tours + 1):
        # Optionally restart the tour from a fresh random non-obstacle pose.
        if not args.continuous:
            obs, info = env.reset()
            trail = [tuple(info["pos"])]
        cur = agent.membership_np(obs)
        goal = args.goal if (tour == 1 and args.goal >= 0) else pick_goal(cur)
        if goal is None:
            break
        state["goal"] = goal  # render the CURRENT goal marker (was stale before)
        print(f"tour {tour}/{args.tours}: start={cur} goal={goal}", flush=True)

        # CLOSED-LOOP execution: every `replan_every` steps, re-read the current
        # membership, re-run Dijkstra to the goal, and fire the option of the NEXT
        # centroid on the fresh path (re-determining the active mask each step).
        target = cur
        reached = False
        for t in range(args.tour_timeout):
            if t % args.replan_every == 0:
                cur = agent.membership_np(obs)
                plan = graph.dijkstra(cur, goal) or [cur]
                target = plan[1] if len(plan) >= 2 else goal
                state["plan"] = plan
            obs, _, _, _, info = env.step(agent.act_option(obs, target, noise=args.exec_noise))
            total += 1
            trail.append(tuple(info["pos"]))
            if t % args.render_every == 0:
                state["caption"] = (f"tour {tour}/{args.tours}  now=C{cur} ->C{target}  "
                                    f"goal=C{goal}  plan={plan}")
                snap()
            if agent.membership_np(obs) == goal:
                reached = True
                break
            if total >= args.max_steps:
                break
        final = agent.membership_np(obs)
        n_success += int(reached)
        state["caption"] = f"tour {tour}/{args.tours}: {'REACHED' if reached else 'timeout at'} C{final}  (goal C{goal})"
        for _ in range(8):  # hold at tour end
            snap()
        print(f"  -> final={final} goal={goal} {'SUCCESS' if reached else 'miss'} ({total} steps total)", flush=True)
        if total >= args.max_steps:
            break

    print(f"done: {n_success}/{args.tours} tours reached goal, {total} steps, {len(frames)} frames", flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(args.out, save_all=True, append_images=imgs[1:], duration=60, loop=0)
    print(f"wrote {args.out}  ({len(imgs)} frames)")


if __name__ == "__main__":
    main()
