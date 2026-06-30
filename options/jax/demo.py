"""Plan over the centroid graph and execute the plan by firing FTN options (JAX).

JAX port of options/demo.py. Loads a train_options_2.py checkpoint (the pickled
`checkpoint.pkl` + `graph.pkl`, NOT the torch `.pt`), then:
  1. resets the world and reads the agent's CURRENT centroid membership;
  2. picks a RANDOM goal centroid (reachable in the graph);
  3. plans a path with Dijkstra over the behavioral adjacency graph;
  4. executes CLOSED-LOOP: every `replan_every` steps it re-reads the current
     centroid membership, re-runs Dijkstra to the goal, and fires the option
     (actor mask) of the NEXT centroid on the fresh path -- so the active mask is
     re-determined as the agent moves, until it reaches the goal (or times out).

Renders the whole top-down world each step with the centroid graph + edges, the
planned path highlighted, the live current-membership marker, and the goal
marker, and writes an animated GIF to options/jax/artifacts/demo.gif.

    uv run --project ../.. python demo.py
    uv run --project ../.. python demo.py --seed 3 --goal 7
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import jax
import jax.numpy as jnp
from PIL import Image

# Reuse the framework-neutral behavioral graph + Dijkstra (see train_options_2.py).
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from centroid_graph import BehavioralGraph

from agent import CentroidRegistry
from color_world import ColorWorld
from config import DemoConfig
from ftn import actor_forward
from viz import _mask_grid

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def _render_mask_pane(mask_np, H, sq, cid, size_px=500, dpi=100):
    """Square heatmap of the currently-active actor mask (the H units reshaped via
    the FTN lateral square layout). Returns an (size_px, size_px, 3) uint8 frame to
    sit beside the world view. Near-binary when settled; shows the blend mid-morph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = _mask_grid(np.asarray(mask_np, np.float32), sq, H)
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(g, cmap="magma", vmin=0.0, vmax=1.0, interpolation="nearest",
              aspect="auto", zorder=0)
    on = int((g > 0.5).sum())
    ax.text(0.02, 0.98, f"active mask  C{cid}\n{on}/{H} units on", color="white",
            fontsize=8, va="top", ha="left", transform=ax.transAxes, zorder=2,
            bbox=dict(facecolor=(0, 0, 0, 0.65), edgecolor="none", pad=2))
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def _break_wraps(trail, size):
    """Insert (nan, nan) breaks where the trail jumps across a torus seam, so the
    rendered polyline doesn't draw a long streak across the frame on wrap."""
    thresh = 0.5 * size
    out: list = []
    for i, p in enumerate(trail):
        if i and (abs(p[0] - trail[i - 1][0]) > thresh or abs(p[1] - trail[i - 1][1]) > thresh):
            out.append((np.nan, np.nan))
        out.append(p)
    return out


class DemoAgent:
    """Minimal executor reconstructed from a pickled checkpoint: the shared actor
    backbone + per-option actor masks + the centroid registry for membership."""

    def __init__(self, ckpt):
        cfg = ckpt["cfg"]
        self.cfg = cfg
        # device_get left the pytree as numpy; lift back to jnp for the forward.
        self.actor = jax.tree_util.tree_map(jnp.asarray, ckpt["actor"])
        self.options = {int(c): jnp.asarray(o["m_a"]) for c, o in ckpt["options"].items()}
        self.registry = CentroidRegistry(cfg.obs_dim, cfg)
        self.registry.centroids = np.asarray(ckpt["centroids"], np.float32)
        self.registry.pos = np.asarray(ckpt["centroid_pos"], np.float32)
        # History-augmented obs: keep a rolling window of the last n (obs, action).
        self.raw_obs_dim = cfg.obs_dim
        self.act_dim = cfg.act_dim
        self.n_hist = int(getattr(cfg, "history_len", 0))
        self.reset_history()

    def reset_history(self):
        self._oh = np.zeros((self.n_hist, self.raw_obs_dim), np.float32)
        self._ah = np.zeros((self.n_hist, self.act_dim), np.float32)

    def _aug(self, obs_np: np.ndarray) -> np.ndarray:
        return np.concatenate([obs_np.astype(np.float32), self._oh.reshape(-1), self._ah.reshape(-1)])

    def membership_np(self, obs_np: np.ndarray) -> int:
        return int(self.registry.membership_np(obs_np[None].astype(np.float32))[0])

    def act_with_mask(self, obs_np: np.ndarray, mask, noise: float = 0.0) -> np.ndarray:
        """Greedy action under an explicit actor mask (may be a blend of options),
        applied to the history-augmented obs. Updates the rolling history with the
        executed (obs, action) so the next step sees this transition."""
        aug = self._aug(obs_np)
        a = np.asarray(actor_forward(self.actor, jnp.asarray(aug[None], jnp.float32),
                                     jnp.asarray(mask))[0])
        if noise > 0:
            a = a + np.random.randn(self.cfg.act_dim).astype(np.float32) * noise
        a = np.clip(a, -1.0, 1.0).astype(np.float32)
        if self.n_hist > 0:  # roll in current (obs, action)
            self._oh = np.concatenate([obs_np.astype(np.float32)[None], self._oh], 0)[:self.n_hist]
            self._ah = np.concatenate([a[None], self._ah], 0)[:self.n_hist]
        return a

    def act_option(self, obs_np: np.ndarray, cid: int, noise: float = 0.0) -> np.ndarray:
        """Greedy action of option `cid` (its actor mask)."""
        return self.act_with_mask(obs_np, self.options[cid], noise)


class DemoEnv:
    """Host-side reset/step over ColorWorld's dynamics (the JAX env only exposes
    the fused collection scan, so the demo's per-step loop runs them on the host).
    Mirrors collect._rollout: disp = action * max_speed. With torus=True the box
    edges wrap (no boundaries); otherwise the position is clipped as in training.
    Either way the position stays in [0, size) -- the domain the policy saw."""

    def __init__(self, world: ColorWorld, torus: bool = False):
        self.w = world
        self.size = world.size
        self.torus = torus
        self.max_speed = world.max_speed
        self.n_cells = world.n_cells
        self.world_seed = world.world_seed
        self.pos = np.zeros(2, np.float32)
        self.heading = 0.0

    def _obs(self, pos: np.ndarray) -> np.ndarray:
        if self.w.coord_obs:  # continuous (x,y) in [-1,1]^2 (matches color_world.coord_at)
            return (pos.astype(np.float32) / self.w.size * 2.0 - 1.0)
        return self.w._cell_color_at(pos[None].astype(np.float32))[0]

    def reset(self, seed=None):
        s = self.w.size
        for _ in range(64):  # rejection-sample a spawn outside every obstacle
            p = np.random.uniform(0.02 * s, 0.98 * s, size=2).astype(np.float32)
            if not bool(self.w.in_obstacle(p)):
                break
        self.pos = p
        self.heading = 0.0
        return self._obs(self.pos), {"pos": self.pos.copy(), "heading": self.heading}

    def step(self, action):
        disp = np.asarray(action, np.float32) * self.max_speed
        if self.torus:
            npos = np.mod(self.pos + disp, self.w.size).astype(np.float32)
        else:
            npos = np.clip(self.pos + disp, 0.0, self.w.size).astype(np.float32)
        # Can't pass through obstacles: reject a move that would enter one.
        if bool(self.w.in_obstacle(npos)) and not bool(self.w.in_obstacle(self.pos)):
            npos = self.pos
        if float(np.linalg.norm(disp)) > 1e-6:
            self.heading = float(np.arctan2(disp[1], disp[0]))
        self.pos = npos
        return self._obs(npos), 0.0, False, False, {"pos": npos.copy(), "heading": self.heading}

    def render_world(self, **kw):
        return self.w.render_world(**kw)


def load(checkpoint="checkpoint.pkl", graph="graph.pkl"):
    with open(os.path.join(ART, checkpoint), "rb") as f:
        ckpt = pickle.load(f)
    with open(os.path.join(ART, graph), "rb") as f:
        gd = pickle.load(f)
    agent = DemoAgent(ckpt)
    bgraph = BehavioralGraph.from_dict(gd["graph"])
    env_kwargs = gd.get("env_kwargs", {})  # ColorWorld.config() captured at training
    return agent, bgraph, env_kwargs


def main():
    # Defaults live in config.DemoConfig (single source of truth); flags override.
    d = DemoConfig()
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--tours", type=int, default=d.tours, help="number of random plan->execute cycles to chain")
    p.add_argument("--goal", type=int, default=d.goal, help="fix the FIRST tour's goal (-1 = random)")
    p.add_argument("--replan-every", type=int, default=d.replan_every,
                   help="re-plan (membership->Dijkstra->next-centroid mask) every N steps")
    p.add_argument("--tour-timeout", type=int, default=d.tour_timeout, help="max steps per tour to reach the goal")
    p.add_argument("--exec-noise", type=float, default=d.exec_noise, help="action noise during option execution (helps unstick)")
    p.add_argument("--mask-polyak", type=float, default=d.mask_polyak,
                   help="inertia when switching options: the active actor mask is blended toward the target "
                        "option's mask each step as m = (1-rate)*m + rate*m_target, rate = 1 - mask_polyak. "
                        "0 = switch instantly (default); ->1 = morph slowly over many steps")
    p.add_argument("--mask-delay", type=int, default=d.mask_delay,
                   help="hold the active mask for N steps after the planner picks a NEW target option "
                        "before it starts changing toward it (latency on the switch; 0 = no delay). "
                        "Composes with --mask-polyak: delay, THEN morph")
    p.add_argument("--max-steps", type=int, default=d.max_steps)
    p.add_argument("--render-every", type=int, default=d.render_every)
    p.add_argument("--continuous", action="store_true", default=d.continuous,
                   help="legs continue from the previous pose instead of restarting at a random location")
    p.add_argument("--max-speed", type=float, default=d.max_speed,
                   help="override env max_speed (default: the value recorded in graph.pkl)")
    p.add_argument("--torus", action=argparse.BooleanOptionalAction, default=d.torus,
                   help="wrap the world edges (no boundaries) instead of clipping at the walls (--no-torus)")
    p.add_argument("--checkpoint", default=d.checkpoint,
                   help="checkpoint file in artifacts/ (e.g. checkpoint_on_policy.pkl)")
    p.add_argument("--graph", default=d.graph, help="graph file in artifacts/")
    p.add_argument("--out", default=os.path.join(ART, d.out))
    args = p.parse_args()

    np.random.seed(args.seed)

    agent, graph, env_kwargs = load(args.checkpoint, args.graph)
    if not agent.options:
        raise SystemExit(
            "checkpoint has no trained options (options is empty). Re-run "
            "train_options_2.py so every centroid gets an option mask."
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

    # Rebuild the SAME world used in training (env_kwargs from graph.pkl), but with
    # a long horizon so legs aren't truncated. max_speed MUST match training or the
    # policy's per-step displacement will be mis-scaled.
    ekw = {**env_kwargs, "max_steps": 10_000}
    if args.max_speed is not None:
        ekw["max_speed"] = args.max_speed
    env = DemoEnv(ColorWorld(**ekw), torus=args.torus)
    print(f"demo env: ColorWorld max_speed={env.max_speed} n_cells={env.n_cells} "
          f"world_seed={env.world_seed} torus={env.torus}", flush=True)
    obs, info = env.reset(seed=args.seed)
    agent.reset_history()
    cur = agent.membership_np(obs)

    H, sq = agent.cfg.ftn_hidden_size, agent.cfg.lateral_square_dim

    # Mutable state shared with snap() (caption/path/goal/active-mask change per tour).
    frames: list[np.ndarray] = []
    trail: list = [tuple(info["pos"])]
    state = {"plan": [cur], "goal": cur, "caption": "",
             "mask": np.asarray(agent.options[cur]) if cur in agent.options
                     else np.zeros(H, np.float32),
             "target": cur}

    def snap():
        path_xy = [tuple(pos[c]) for c in state["plan"]]
        tr = trail[-400:]
        if env.torus:
            tr = _break_wraps(tr, env.size)
        world = env.render_world(
            centroids=centroids_xy, centroid_ids=centroid_ids, edges=edges_xy,
            path=path_xy, trail=tr, current_id=agent.membership_np(obs),
            goal_id=state["goal"], agent_pos=info["pos"], agent_heading=info["heading"],
            caption=state["caption"])
        pane = _render_mask_pane(state["mask"], H, sq, state["target"], size_px=world.shape[0])
        frames.append(np.concatenate([world, pane], axis=1))

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
        # Optionally restart the tour from a fresh random pose.
        if not args.continuous:
            obs, info = env.reset()
            agent.reset_history()
            trail = [tuple(info["pos"])]
        cur = agent.membership_np(obs)
        goal = args.goal if (tour == 1 and args.goal >= 0) else pick_goal(cur)
        if goal is None:
            break
        state["goal"] = goal
        print(f"tour {tour}/{args.tours}: start={cur} goal={goal}", flush=True)

        # CLOSED-LOOP execution: every `replan_every` steps, re-read the current
        # membership, re-run Dijkstra to the goal, and fire the option of the NEXT
        # centroid on the fresh path (re-determining the active mask each step).
        # With --mask-polyak > 0 the applied mask trails the target's mask, morphing
        # toward it over several steps instead of switching instantly. With
        # --mask-delay N the mask additionally HOLDS for N steps after the planner
        # picks a new target before it starts changing toward it (delay, then morph).
        rate = float(np.clip(1.0 - args.mask_polyak, 0.0, 1.0))
        target = cur          # what the planner wants right now
        eff_target = cur      # what the mask is actually morphing toward
        pending = cur         # last requested target (to detect a fresh switch)
        delay_left = 0        # steps still to wait before committing a switch
        plan = [cur]
        m_active = None  # running actor mask; lazily seeded with the first target's mask
        reached = False
        for t in range(args.tour_timeout):
            if t % args.replan_every == 0:
                cur = agent.membership_np(obs)
                plan = graph.dijkstra(cur, goal) or [cur]
                target = plan[1] if len(plan) >= 2 else goal
                state["plan"] = plan
            # A newly requested target arms the delay; the mask keeps tracking the
            # old eff_target until the countdown elapses, then commits the switch.
            if target != pending:
                pending = target
                delay_left = args.mask_delay
            if delay_left > 0:
                delay_left -= 1
            else:
                eff_target = pending
            m_target = agent.options[eff_target]
            m_active = m_target if m_active is None else (1.0 - rate) * m_active + rate * m_target
            state["mask"], state["target"] = np.asarray(m_active), eff_target
            obs, _, _, _, info = env.step(agent.act_with_mask(obs, m_active, noise=args.exec_noise))
            total += 1
            trail.append(tuple(info["pos"]))
            if t % args.render_every == 0:
                held = "" if eff_target == target else f" (hold {delay_left})"
                state["caption"] = (f"tour {tour}/{args.tours}  now=C{cur} ->C{target}{held}  "
                                    f"mask=C{eff_target}  goal=C{goal}  plan={plan}")
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
