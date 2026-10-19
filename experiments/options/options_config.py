"""Single source of truth for every tunable parameter across the options/ prototype.

Four dataclasses, one per concern, plus the smoke override:

* ``AgentConfig`` -- agent / RECODE / FTN / DDPG knobs. Consumed by
  ``options_agent.OptionsAgent`` (and its ``CentroidRegistry``).
* ``TrainConfig`` -- the explore -> sync -> graph -> train-options loop schedule.
  Shared by ``train_options.py`` and ``train_options_par.py``.
* ``ParConfig``   -- extras specific to the parallel trainer
  (``train_options_par.py``): worker count, vectorized per-option batch, option
  co-adaptation epochs, ColorWorld shape, and the world seed.
* ``DemoConfig``  -- plan-and-execute rollout knobs for ``demo.py``.

``smoke_config()`` returns the tiny ``(TrainConfig, AgentConfig)`` pair selected by
the ``--smoke`` flag.

Every script reads its DEFAULTS from here (CLI flag defaults reference these
dataclasses), so changing a value in one place changes it everywhere; individual
CLI flags still override single fields at launch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# AgentConfig -- agent / RECODE / FTN / DDPG knobs (options_agent.py)
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    # NOT a tunable: these are dimensions of the ACTIVE environment, filled in at
    # launch from ``env.obs_dim`` / ``env.action_space.shape[0]`` before the agent
    # is built. Left 0 here so a forgotten assignment fails loudly (FTN with
    # in_dim=0) rather than silently training on a stale hardcoded size.
    obs_dim: int = 0
    act_dim: int = 0

    # FTN backbone
    ftn_hidden_size: int = 1024      # H parallel sub-networks
    ftn_k: int = 128                # active neurons per (option) mask
    ftn_inner_size: int = 8
    ftn_layers: int = 7
    ftn_dropout_p: float = 0.1

    # SmoothKWTA
    lateral_steps: int = 1
    lateral_polyak: float = 0.5      # mix coeff per pass; must be < 1.0 to enable
    lateral_square_dim: int = 32     # block side; H must be a multiple of its square
    lateral_kernel_size: int = 5     # smoothing kernel (odd)
    lateral_noise_sd: float = 0.0
    lateral_torus: bool = True      # wrap the grid (circular conv) if True

    # DDPG
    gamma: float = 0.90
    tau: float = 0.01
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    batch_size: int = 128
    noise_std: float = 0.1

    # AnnealedSTE mask reconfig
    reconfig_steps: int = 60
    reconfig_lr: float = 0.3
    reconfig_n: int = 2048
    mask_init_noise_scale: float = 1e-5

    # Exploration
    epsilon: float = 1.0
    epsilon_decay: float = 0.9995
    epsilon_min: float = 0.1
    forward_bias: float = 0.6      # random-explore accel sampled from [bias-1, 1]

    # Option (artificial membership) reward: a sparse bonus for being IN the
    # target centroid plus a dense shaping term that rewards getting closer to it
    # in observation space (gives the reactive policy a gradient to descend).
    option_goal_reward: float = 1.0
    option_step_cost: float = 0.0
    option_shaping: float = 0.4

    # RECODE clustering / novelty
    recode_memory_size: int = 4000
    recode_k: int = 10
    recode_kappa: float = 0.2
    recode_gamma: float = 1.0
    recode_insertion_probability: float = 0.2
    recode_decay: float = 0.9999

    # Stable centroid registry: farthest-point-sampled from RECODE atoms up to
    # `target_centroids` (existing IDs are preserved and re-anchored each sync).
    target_centroids: int = 20
    max_centroids: int = 25
    centroid_min_sep: float = 1e-3   # stop adding once atoms are this close to a centroid
    centroid_anchor_ema: float = 0.05

    buffer_capacity: int = 200_000
    device: str = "cuda"


# ---------------------------------------------------------------------------
# TrainConfig -- explore -> sync -> graph -> train-options loop schedule
# ---------------------------------------------------------------------------
# rounds                : number of explore -> sync -> graph -> train-options
#                         cycles. Centroids/options accumulate across rounds, so
#                         later rounds retrain old options on more data.
# explore_steps         : env steps of (forward-biased) novelty exploration per
#                         round. The agent's buffer is continuous across rounds.
# explorer_update_every : run the explorer DDPG update every N env steps.
# explorer_updates      : explorer DDPG gradient steps taken at each such update.
# option_reconfig       : if True, re-search each option's FTN mask pair (AnnealedSTE)
#                         every round before training it; if False, keep prior masks.
# option_train_steps    : DDPG gradient steps spent training EACH option's masked
#                         subspace per round (membership-relabeled reward).
# eval_option_episodes  : reach-rate probe rollouts per option after training
#                         (0 = skip). Reported as reach@probe (noisy; few rollouts).
# eval_option_horizon   : max env steps per reach-rate probe rollout.
# max_steps             : env truncation horizon (one "episode"); marks buffer
#                         boundaries that break behavioral-graph edge chains.
# seed                  : RNG seed for python/numpy/torch and agent spawns.
# random_explore        : bypass the explorer net entirely -- pure forward-biased
#                         random walk, no explorer DDPG (fast coverage collection).
# log_every_steps       : explore-phase progress cadence (env steps).
# log_every_options     : option-training progress cadence (options).
@dataclass
class TrainConfig:
    rounds: int = 1
    explore_steps: int = 24000          # env steps per round
    explorer_update_every: int = 4     # DDPG updates batched every N env steps
    explorer_updates: int = 4
    option_reconfig: bool = True
    option_train_steps: int = 5000      # DDPG steps per option per round
    eval_option_episodes: int = 0      # reach-rate probes per option (0 = skip)
    eval_option_horizon: int = 120
    max_steps: int = 24000         # env truncation horizon
    seed: int = 0
    random_explore: bool = True       # bypass the explorer net: pure random walk
    log_every_steps: int = 1000       # explore-phase progress cadence (env steps)
    log_every_options: int = 5         # option-training progress cadence (options)


def _default_workers() -> int:
    """Leave two cores free for the GPU main process / RECODE."""
    return max(1, (os.cpu_count() or 4) - 2)


# ---------------------------------------------------------------------------
# ParConfig -- extras for the parallel trainer (train_options_par.py)
# ---------------------------------------------------------------------------
# workers       : parallel random-walk collection processes.
# par_batch     : per-option batch size in the vectorized (all-options-at-once)
#                 DDPG step.
# option_epochs : passes of (reconfig-all-masks -> train-all) per round so masks
#                 co-adapt (consolidate/segregate) against the shared backbone.
# env           : which world to train in -- "color" (ColorWorld: allocentric
#                 [dx,dy] control, obs = current cell's unique color vector) or
#                 "continuous" (ContinuousWorld: egocentric [turn,accel] control,
#                 obs = per-ray wall distances + color compass + floor color, with
#                 obstacles). Selects the env class via envs.make_env; the chosen
#                 name is recorded into graph.pt so workers + demo rebuild it.
# n_cells       : ColorWorld Voronoi cell count (ignored for "continuous", which
#                 uses its own denser default patchwork).
# color_dim     : ColorWorld obs/color vector length (>= 3); ColorWorld only.
# max_speed     : per-step displacement scale (None = env default; ColorWorld 0.6,
#                 ContinuousWorld 0.7); recorded into graph.pt so the demo rebuilds
#                 the env identically.
# world_seed    : fixed map seed; MUST match the main env + every collection
#                 worker so they all explore the same world.
@dataclass
class ParConfig:
    workers: int = field(default_factory=_default_workers)
    par_batch: int = 64
    option_epochs: int = 1
    env: str = "color"          # "color" (ColorWorld) | "continuous" (ContinuousWorld)
    n_cells: int = 10
    color_dim: int = 8
    max_speed: float | None = None
    world_seed: int = 0


# ---------------------------------------------------------------------------
# DemoConfig -- plan-and-execute rollout knobs (demo.py)
# ---------------------------------------------------------------------------
# seed         : RNG seed for python/numpy/torch -> agent spawn + random goals.
# tours        : number of goal-reaching episodes chained into one GIF.
# goal         : fix the FIRST tour's goal centroid id (-1 = random). Later tours
#                are always random.
# replan_every : closed-loop control -- every N steps, re-read membership, re-run
#                Dijkstra to the goal, and switch to the next centroid's option
#                mask. 1 = re-plan every step.
# tour_timeout : max env steps per tour to reach the goal before giving up.
# exec_noise   : Gaussian action noise added during option execution (a little
#                noise helps the greedy option escape stuck states).
# max_steps    : hard cap on total executed env steps across all tours.
# render_every : capture a GIF frame every N executed env steps.
# continuous   : if True, legs continue from the previous pose; if False (default)
#                each leg restarts from a fresh random non-obstacle pose.
# max_speed    : override ColorWorld max_speed (match training for checkpoints
#                saved before max_speed was recorded into graph.pt).
# out          : output GIF path ("" -> artifacts/demo.gif, resolved by demo.py).
@dataclass
class DemoConfig:
    seed: int = 6
    tours: int = 5
    goal: int = -1
    replan_every: int = 1
    tour_timeout: int = 100
    exec_noise: float = 0.0
    max_steps: int = 5000
    render_every: int = 1
    continuous: bool = False
    max_speed: float = 0.6
    out: str = ""  # "" -> os.path.join(ART, "demo.gif"); resolved in demo.py


def smoke_config() -> tuple[TrainConfig, AgentConfig]:
    """Tiny, fast config selected by ``--smoke`` (both trainers)."""
    tc = TrainConfig(rounds=2, explore_steps=1500, option_train_steps=40,
                     eval_option_episodes=2, eval_option_horizon=60, max_steps=200)
    ac = AgentConfig(ftn_hidden_size=32, ftn_k=8, reconfig_steps=20, reconfig_n=512,
                     target_centroids=8, epsilon_decay=0.998)
    return tc, ac
