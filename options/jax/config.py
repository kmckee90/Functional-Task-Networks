"""Config for the JAX port of train_options_2.py.

Mirrors options/options_config.py 1:1 (same field names + defaults) so the JAX
experiment is parameter-identical to the torch one. AgentConfig, TrainConfig and
ParConfig drive the sequential-mask trainer; OnPolicyConfig drives the on-policy
bootstrap (train_on_policy.py) and DemoConfig the planner demo (demo.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    obs_dim: int = 51
    act_dim: int = 2

    # FTN backbone
    ftn_hidden_size: int = 32*32      # H parallel sub-networks
    ftn_k: int = 32                 # active neurons per (option) mask
    ftn_inner_size: int = 8
    ftn_layers: int = 8
    ftn_dropout_p: float = 0.1       # (no-op in the JAX port: dropout disabled here)

    # Observation history: the policy/critic see the current obs plus the last
    # `history_len` (obs, action) pairs -> network input = obs*(n+1) + act*n.
    # n=0 = current obs only (unchanged behavior). Helps disambiguate the hidden
    # within-cell position from the piecewise-constant color obs. Networks must be
    # retrained when this changes (it resizes the actor/critic input).
    history_len: int = 3

    # SmoothKWTA
    lateral_steps: int = 2
    lateral_polyak: float = 0.0
    lateral_square_dim: int = 32
    lateral_kernel_size: int = 13
    lateral_noise_sd: float = 0.0
    lateral_torus: bool = True

    # DDPG
    gamma: float = 0.80
    tau: float = 1.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    noise_std: float = 0.0

    # Conservative Q-Learning (offline overestimation control; Kumar et al. 2020).
    # cql_alpha=0 -> pure DDPG (unchanged). When >0, the critic loss gains
    #   alpha * ( logsumexp_a Q(s,a) - Q(s, a_data) )
    # pushing Q DOWN on OOD/random actions and UP on the dataset action, yielding a
    # conservative (lower-bound) critic so the actor can't chase hallucinated value
    # on actions the random-walk buffer never took. The continuous logsumexp is the
    # CQL(H) approximation: cql_n_actions uniform action samples over the tanh box
    # [-1,1]^act_dim, plus the current policy action. Penalty is applied per-option
    # on each option's own critic mask. Typical alpha ~0.1-10; start ~1.0 and sweep.
    cql_alpha: float = 1.0
    cql_n_actions: int = 10

    # AnnealedSTE mask reconfig
    reconfig_steps: int = 1
    reconfig_lr: float = 1
    reconfig_n: int = 2048
    mask_init_noise_scale: float = 1e-5

    # Exploration
    epsilon: float = 1.0
    epsilon_decay: float = 0.9995
    epsilon_min: float = 0.1
    forward_bias: float = 0.6

    # Option (artificial membership) reward
    option_goal_reward: float = 1.0
    option_step_cost: float = 0.0
    option_shaping: float = 0.0
    # When True the per-step option reward is the NEGATIVE Euclidean distance from
    # the (next) obs to the option's target centroid -- a dense distance-to-goal
    # reward -- instead of the sparse membership reward (goal*member - shaping*dist
    # - step_cost). train_options_2.py auto-enables this in coordinate-obs mode
    # (ParConfig.coord_obs), where the obs IS the position so distance is meaningful.
    dist_reward: bool = False

    # RECODE clustering / novelty
    recode_memory_size: int = 1000
    recode_k: int = 10
    recode_kappa: float = 0.2
    recode_gamma: float = 1.0
    recode_insertion_probability: float = 0.2
    recode_decay: float = 0.9999

    # Stable centroid registry (FPS over RECODE atoms)
    target_centroids: int = 40
    max_centroids: int = 45
    centroid_min_sep: float = 1e-3
    centroid_anchor_ema: float = 0.05

    buffer_capacity: int = 200_000
    device: str = "cuda"  # advisory only; JAX picks the default backend


@dataclass
class TrainConfig:
    rounds: int = 1
    explore_steps: int = 64000
    explorer_update_every: int = 4
    explorer_updates: int = 4
    option_reconfig: bool = True
    option_train_steps: int = 400
    eval_option_episodes: int = 0
    eval_option_horizon: int = 120
    max_steps: int = 64000
    seed: int = 0
    random_explore: bool = True
    log_every_steps: int = 1000
    log_every_options: int = 1


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 4) - 2)


@dataclass
class ParConfig:
    workers: int = field(default_factory=_default_workers)
    par_batch: int = 64
    option_epochs: int = 1
    n_cells: int = 30
    color_dim: int = 8
    # When True the observation is simply the agent's continuous (x, y) position
    # normalized to [-1, 1]^2 (obs_dim=2) -- the Voronoi patchwork + color obs are
    # bypassed entirely. The world geometry is still generated (unchanged RNG
    # stream) but unused for obs; the demo renders gray space. Networks must be
    # retrained when this changes (it resizes obs_dim).
    coord_obs: bool = True
    max_speed: float | None = None
    world_seed: int = 2
    torus: bool = False  # wrap the world edges (no boundaries) for collection + dynamics
    n_obstacles: int = 14          # random convex polygon obstacles the agent can't enter (0 = open room)
    obstacle_min_r: float = 1.0   # obstacle size range (world units)
    obstacle_max_r: float = 4.0
    obstacle_max_verts: int = 9   # max vertices per random convex polygon


@dataclass
class OnPolicyConfig:
    """On-policy bootstrap (train_on_policy.py): after train_options_2.py, for each
    option in turn -- activate its mask, collect fresh data under that mask's
    epsilon-greedy policy (reward = centroid membership), then DDPG-train the mask
    on that on-policy data. One pass over all options = one epoch."""
    epochs: int = 50                  # passes over all options
    steps_per_option: int = 24000     # on-policy transitions collected per option per epoch
    workers: int = 8                # parallel rollout workers (batched through the actor)
    epsilon: float = 0.4             # epsilon-greedy exploration during collection
    episode_max_steps: int = 20     # per-option-episode truncation horizon (resets on goal/timeout)
    option_train_steps: int = 1   # DDPG steps per option per epoch on the on-policy buffer
    consolidate: bool = True         # sum mask-overlapping sibling terms in each DDPG step
    seed: int = 0
    # in_checkpoint: str = "checkpoint.pkl"        # train_options_2.py output to bootstrap from
    in_checkpoint: str = "checkpoint_on_policy.pkl"        # train_options_2.py output to bootstrap from
    graph: str = "graph.pkl"                     # graph/world to reuse (env_kwargs, centroids)
    out_checkpoint: str = "checkpoint_on_policy.pkl"


@dataclass
class DemoConfig:
    """Planner demo (demo.py): plan over the centroid graph and execute options."""
    seed: int = 0
    tours: int = 10                   # number of random plan->execute cycles to chain
    goal: int = -1                   # fix the FIRST tour's goal (-1 = random)
    replan_every: int = 1            # re-plan (membership->Dijkstra->next-centroid mask) every N steps
    tour_timeout: int = 150          # max steps per tour to reach the goal
    exec_noise: float = 0.3          # action noise during option execution (helps unstick)
    mask_polyak: float = 0.25         # switch inertia: blend active mask toward target (0 = instant)
    mask_delay: int = 3              # hold the active mask N steps after a new target before morphing
    max_steps: int = 5000            # global step cap across all tours
    render_every: int = 1
    continuous: bool = False         # legs continue from the previous pose vs restart at a random location
    max_speed: float | None = None   # override env max_speed (default: value recorded in the graph)
    torus: bool = True               # wrap the world edges (no boundaries) vs clip at walls
    checkpoint: str = "checkpoint.pkl"  # which checkpoint to load (e.g. checkpoint_on_policy.pkl)
    # checkpoint: str = "checkpoint_on_policy.pkl"  # which checkpoint to load (e.g. checkpoint_on_policy.pkl)
    graph: str = "graph.pkl"
    out: str = "demo.gif"


def smoke_config() -> tuple[TrainConfig, AgentConfig]:
    tc = TrainConfig(rounds=2, explore_steps=1500, option_train_steps=40,
                     eval_option_episodes=2, eval_option_horizon=60, max_steps=200)
    # lateral_square_dim=2 so H=32 is a multiple of square_dim**2=4 (the torch
    # smoke leaves it at 32 and so crashes LateralSmoother; here we keep smoothing
    # ON but valid). Full AgentConfig (H=1024, sq=32) is unaffected.
    ac = AgentConfig(ftn_hidden_size=32, ftn_k=8, reconfig_steps=20, reconfig_n=512,
                     target_centroids=8, epsilon_decay=0.998, lateral_square_dim=2)
    return tc, ac
