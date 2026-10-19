"""Sequential DDPG on Meta-World MT10 (cheat-init) — FTN backbone + masked replay.

Direct port of train_grid_dqn.py (which works on grid-world) to DDPG / continuous
control on the MT10 suite (cheat-init; run any subset via DEFAULT_TASKS). Same
continual-learning protocol:

Per task:
    1. Exploration: random-action rollouts on a cheat-initialized env (arm/world
       pre-placed near the reward state). Stop after `n_successful_explore`
       rewarded episodes or `explore_episodes` total.
    2. Mask reconfig on the rewarded support — critic mask via `reconfig_objective`
       (default: −Pearson(Q(s,a), G_t)), then actor mask via −Q under the new
       critic mask. AnnealedSTE.
    3. Exploitation: K episodes of cheat-init rollouts with Gaussian-noise
       actor; per-env-step joint DDPG step = current-data loss (current masks)
       summed with the loss over up to `replay_max_chunks` mask-overlapping
       archived chunks, each forwarded under its own (m_a, m_c). Data is chunked
       per reconfig interval so each chunk's transitions match exactly one mask.
    4. Periodic mid-phase reconfig every `reconfig_every_trials` trials using
       recent episodes; each reconfig closes off the current chunk (tagged with
       the outgoing masks) before committing the new ones.
    5. Archive the final chunk, then eval all tasks. Eval mirrors exploration: random
       rollouts under cheat init until `n_successful_explore` rewarded → one
       cold-started AnnealedSTE reconfig per eval episode, greedy rollout.

This is the robotic-arm analog of the grid-world demo. The "cheat" hides the
exploration problem so we can focus on whether the FTN mask-search recovers a
working policy from a handful of rewarded trajectories.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import metaworld
import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ftn import (
    FTNActor,
    FTNCritic,
    anneal_ste_mask,
    kwta,
)

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Full MT10 suite — every name has a cheat handler in `_cheat_reset`. Comment
# out any line to skip that task; the rest of the pipeline keys off len(tasks),
# so any subset works. Listed in canonical MT10 order.
# NOTE: pick-place and peg-insert-side need a *sustained grasp* (the gripper
# must hold the object aloft / through insertion), which random actions rarely
# achieve even cheat-initialized — they are the hard ones to calibrate.
DEFAULT_TASKS = (
    # "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "window-open-v3",
    "window-close-v3",
    # "reach-v3",
    # "push-v3",
    # "peg-insert-side-v3",  # hard: needs a sustained grasp + insertion
    # "pick-place-v3",  # hard: needs a sustained grasp
)


# ---------------------------------------------------------------------------
# DDPGConfig glossary (parameter reference; fields are listed bare below)
# ---------------------------------------------------------------------------
# tasks                   : MT10 task names, learned sequentially (one phase each).
#
# -- experiment --
# seed                    : RNG seed for python/numpy/torch and env goal sampling.
# explore_episodes        : cap on exploration episodes per task / per eval cell.
# n_successful_explore    : stop exploration after this many rewarded episodes.
# n_trials_per_task       : K — exploitation episodes run after the boundary reconfig.
# eval_episodes           : greedy rollouts per (phase, task) eval cell.
# max_episode_steps       : truncated horizon; matches what cheat init allows.
# output_path             : .npz/.png results path (suffixed per baseline + seed).
# log_every               : print a training summary every this many trials.
#
# -- DDPG --
# gamma                   : discount factor for TD targets and MC returns.
# tau                     : Polyak target-network averaging rate.
# actor_lr / critic_lr    : Adam learning rates for actor and critic.
# batch_size              : transitions per gradient step.
# replay_capacity         : ring-buffer capacity (current-task transitions).
# updates_per_step        : DDPG gradient steps per env step.
# noise_std               : Gaussian exploration noise added to actions in rollouts.
#
# -- FTN backbone (ParallelMLP bank + KWTA-masked readout) --
# ftn_hidden_size         : H — number of parallel sub-networks per network.
# ftn_k                   : active sub-networks selected by the KWTA mask.
# ftn_inner_size          : width inside each parallel MLP.
# ftn_layers              : hidden layers inside each parallel MLP.
# ftn_dropout_p           : dropout on the per-neuron output vector (train-time only).
#
# -- AnnealedSTE mask reconfiguration --
# reconfig_steps          : inner optimization steps per mask search.
# reconfig_lr             : learning rate for the soft-mask logits.
# reconfig_n              : support-batch cap (subsampled rewarded transitions).
# mask_init_noise_scale   : cold-start logit init = noise_scale * randn(H).
# mask_warm_start         : warm-start logits from the current mask (eval cold-starts
#                           regardless).
# reconfig_objective      : critic-mask objective; one of:
#     "td"          — MSE(Q(s,a)[m], r + γ·(1-done)·Q_target(s', μ(s'))[m])
#     "mse_returns" — MSE(Q(s,a)[m], G_t)
#     "cos_returns" — −cos(Q(s,a)[m], G_t)
#     "r_returns"   — −Pearson(Q(s,a)[m], G_t)   (default)
#   The actor mask is always selected second, via −Q under the new critic mask.
# reconfig_gamma          : discount for MC returns G_t; None → fall back to gamma.
# reconfig_every_trials   : mid-phase reconfig cadence (trials); 0 disables.
# recent_eps_for_reconfig : # recent episodes used as mid-phase reconfig support.
#
# -- mask-overlap-filtered cross-task replay (summed-loss, per-chunk) --
# replay_max_chunks       : max archived chunks summed into one joint step.
# replay_max_tuples       : transitions sampled per recalled chunk (per-chunk
#                           batch size; each chunk is forwarded under its own
#                           (m_a, m_c), so its data-to-mask match stays exact).
# rho_low / rho_high      : recall a chunk if either actor OR critic mask overlap
#                           ρ ∈ (rho_low, rho_high].
#
# -- exploration / eval-time mask recovery --
# warmup_policy           : explore policy after task 0 — "random" or "prev_mask".
#
# -- bootstrapped exploration (eval-time support collection) --
# explore_method          : how the eval-time rewarded support is gathered
#                           (`_eval_with_reconfig`); eval then runs a fresh
#                           cold-start reconfig per eval episode either way.
#                           CLI override: --bootstrap selects "bootstrap".
#     "random"    — collect random / warmup_policy rollouts until
#                   n_successful_explore rewarded episodes. Off-policy: returns
#                   come from meandering random rollouts, so the Q-vs-returns
#                   correlation target is noisy.
#     "bootstrap" — on-policy collection. Random actions until the first reward;
#                   thereafter EVERY exploration episode is preceded by a fresh
#                   COLD-START reconfig on the *cumulative* rewarded support
#                   (never committed), so each episode draws a fresh plausible
#                   mask and a bad early mask can't trap exploration. Rewarded
#                   episodes accumulate. Stops at n_successful_explore rewarded
#                   episodes or explore_episodes total.
# noise_std_explore       : Gaussian exploration noise for the "bootstrap"
#                           collection rollouts (DDPG analog of an exploration ε,
#                           separate from noise_std).
#
# -- baselines: catastrophic-forgetting references (mutually exclusive) --
# fixed_mask    : oracle disjoint allocation — task i owns neurons [i*k, (i+1)*k)
#                 (requires N*k <= H), recalled exactly at eval (oracle task
#                 identity). Disables all reconfig and cross-task replay (the
#                 disjoint masks never overlap); forward-transfer cells are NaN.
# no_mask       : all-ones mask (k = ftn_hidden_size), frozen for the run — plain
#                 dense DDPG over the FTN backbone. Disables reconfig + replay.
# stored_mask   : full FTN training, but eval recalls the last mask used in each
#                 task's training phase instead of explore + cold-start reconfig;
#                 forward-transfer cells NaN. Isolates mask recovery from storage.
# Everything else (buffer, DDPG updates, exploration) is identical across
# conditions, so the result matrices are directly cell-comparable.
# ---------------------------------------------------------------------------
@dataclass
class DDPGConfig:
    # --- tasks ---
    tasks: tuple = DEFAULT_TASKS

    # --- experiment ---
    seed: int = 0
    explore_episodes: int = 2000
    n_successful_explore: int = 100
    n_trials_per_task: int = 100
    eval_episodes: int = 8
    max_episode_steps: int = 200
    output_path: str = "results.npz"
    log_every: int = 10

    # --- DDPG ---
    gamma: float = 0.95
    tau: float = 0.01
    actor_lr: float = 1e-4
    critic_lr: float = 3e-4
    batch_size: int = 64
    replay_capacity: int = 200_000
    updates_per_step: int = 1
    noise_std: float = 1

    # --- FTN backbone ---
    ftn_hidden_size: int = 64
    ftn_k: int = 64
    ftn_inner_size: int = 8
    ftn_layers: int = 8
    ftn_dropout_p: float = 0.00

    # --- AnnealedSTE ---
    reconfig_steps: int = 120
    reconfig_lr: float = 0.003
    reconfig_n: int = 4096
    mask_init_noise_scale: float = 0.00001
    mask_warm_start: bool = True
    reconfig_objective: str = "r_returns"
    reconfig_gamma: float | None = None
    reconfig_every_trials: int = 10
    recent_eps_for_reconfig: int = 10

    # --- mask-overlap-filtered cross-task replay (summed-loss, per-chunk) ---
    replay_max_chunks: int = 4
    replay_max_tuples: int = 64
    rho_low: float = 0.2
    rho_high: float = 0.8

    # --- exploration / eval-time mask recovery ---
    warmup_policy: str = "random"
    # Push exploration trajectories into the replay ring (in addition to seeding
    # recent_eps for reconfig). Off by default: under FTN the phase is archived
    # under its final committed mask, and pre-mask exploration tuples have no
    # valid mask correspondence for cross-task recall. But they are valid
    # off-policy (s,a,r,s') data for the *current* task's DDPG updates — enable
    # this (e.g. for --no-mask, where the mask is a constant all-ones) so that
    # successful random rollouts actually train the critic/actor instead of
    # being discarded.
    replay_explore: bool = True

    # --- bootstrapped exploration (explore_method="bootstrap") ---
    explore_method: str = "random"
    noise_std_explore: float = 0.1

    # --- baselines: catastrophic-forgetting references ---
    fixed_mask: bool = False
    no_mask: bool = False
    stored_mask: bool = False


def smoke_ddpg_config() -> DDPGConfig:
    return DDPGConfig(
        tasks=("reach-v3", "drawer-close-v3"),
        explore_episodes=20,
        n_successful_explore=5,
        n_trials_per_task=20,
        eval_episodes=2,
        max_episode_steps=50,
        batch_size=32,
        replay_capacity=10_000,
        ftn_hidden_size=32,
        ftn_inner_size=4,
        ftn_layers=2,
        ftn_k=8,
        reconfig_steps=10,
        reconfig_n=128,
        recent_eps_for_reconfig=4,
        log_every=5,
        output_path="results_smoke.npz",
    )


# ---------------------------------------------------------------------------
# Cheat-init wrappers for the MT10 suite
# ---------------------------------------------------------------------------
#
# CALIBRATION CONVENTION: every task branch below opens with its difficulty
# knob(s) as named local variables — spawn radii, z-distances, pre-open amounts,
# remaining-travel gaps, etc. Tune ONLY those first line(s) per task to make
# random actions hit success at a useful rate (~30-70% within max_episode_steps);
# use `--calibrate` to measure. Larger radii / bigger remaining gaps = harder.


def _disk_offset(rng: np.random.Generator, radius: float) -> np.ndarray:
    """Uniformly-sampled xy offset within a disk of the given radius (meters)."""
    ang = float(rng.uniform(0.0, 2.0 * np.pi))
    r = radius * float(np.sqrt(rng.uniform(0.0, 1.0)))
    return np.array([r * np.cos(ang), r * np.sin(ang)], dtype=np.float32)


def _cheat_reset(env, task_name: str, rng: np.random.Generator) -> np.ndarray:
    """Reset env, then place arm/world close to the reward state so random
    actions can hit success quickly. Returns the post-cheat observation.

    For free-body objects (push / pick-place / peg) we call `env._set_obj_xyz`
    (the env's own helper — sets the object free-joint and zeros its velocity)
    AFTER `_reset_hand`, so the freshly-placed object isn't disturbed by the
    hand-settle steps. For articulated tasks (door / drawer / window / button)
    we set the relevant joint, then re-assert it after `_reset_hand` (which
    steps the sim) so the pre-set pose survives."""
    env.reset()

    if task_name == "reach-v3":
        xy_radius = 0.1  # [difficulty] hand spawn radius around target in xy (m)
        z_offset = 0.075  # [difficulty] hand height above the target (m)
        target = env._target_pos.copy()
        xy = _disk_offset(rng, xy_radius)
        env.hand_init_pos = np.array(
            [target[0] + xy[0], target[1] + xy[1], target[2] + z_offset],
            dtype=np.float32,
        )
        env._reset_hand(steps=100)

    elif task_name == "push-v3":
        obj_radius = 0.1  # [difficulty] puck spawn radius around the goal (m); keep > 0.05 success radius or it pre-succeeds
        hand_x = 0.05  # [difficulty] hand x-offset from the puck (m); keep > ~0.02 — MetaWorld's caging reward errors if the hand starts ~on the object in x-z (margin < 0, xz_thresh=0.005)
        hand_z = 0.03  # [difficulty] hand height above the puck (m)
        # Success: puck within 0.05 of _target_pos. Spawn it just shy of the
        # goal so a little random nudging finishes the push.
        target = env._target_pos.copy()
        xy = _disk_offset(rng, obj_radius)
        obj = np.array(
            [target[0] + xy[0], target[1] + xy[1], target[2]], dtype=np.float32
        )
        # Keep obj_init_pos consistent with the placed puck — it feeds the dense
        # reward (and thus the MC returns the mask reconfig fits).
        env.obj_init_pos = obj.copy()
        env.hand_init_pos = np.array(
            [obj[0] + hand_x, obj[1], obj[2] + hand_z], dtype=np.float32
        )
        env._reset_hand()
        env._set_obj_xyz(obj)

    elif task_name == "pick-place-v3":
        # HARD: success (obj within 0.07 of an AIR target) needs a sustained
        # grasp; an ungrasped object just falls. Best effort: spawn the object
        # right at the air-goal with the hand on it. Calibrate expecting low SR.
        obj_radius = 0.02  # [difficulty] object spawn radius around the air-goal (m)
        hand_x = 0.05  # [difficulty] hand x-offset from the object (m); keep > ~0.02 (caging-margin safety)
        hand_z = 0.0  # [difficulty] hand height above the object (m)
        target = env._target_pos.copy()
        xy = _disk_offset(rng, obj_radius)
        obj = np.array(
            [target[0] + xy[0], target[1] + xy[1], target[2]], dtype=np.float32
        )
        env.obj_init_pos = obj.copy()  # consistent bookkeeping (feeds the dense reward)
        env.hand_init_pos = np.array(
            [obj[0] + hand_x, obj[1], obj[2] + hand_z], dtype=np.float32
        )
        env._reset_hand()
        env._set_obj_xyz(obj)

    elif task_name == "door-open-v3":
        door_angle = -1.2  # [difficulty] pre-opened hinge angle (rad); more negative = more open = easier (full open ≈ -1.57)
        hand_off = np.array(
            [0.0, 0.0, 0.0], dtype=np.float32
        )  # [difficulty] hand offset from the handle (m)
        # Door starts closed (angle 0); opening drives the hinge negative.
        env.data.joint("doorjoint").qpos[:] = door_angle
        mujoco.mj_forward(env.model, env.data)
        handle = env._get_pos_objects().copy()
        env.hand_init_pos = handle.astype(np.float32) + hand_off
        env._reset_hand()
        env.data.joint("doorjoint").qpos[:] = door_angle
        mujoco.mj_forward(env.model, env.data)

    elif task_name == "drawer-open-v3":
        drawer_pull = 0.14  # [difficulty] pre-pulled-open amount (m); bigger = easier (full open ~0.16, success within 0.03)
        hand_z = 0.04  # [difficulty] hand height above the handle (m)
        # Drawer starts closed; pulling open drives qpos[9] negative.
        qpos = env.data.qpos.flat.copy()
        qvel = env.data.qvel.flat.copy()
        qpos[9] = -drawer_pull
        qvel[9] = 0.0
        env.set_state(qpos, qvel)
        mujoco.mj_forward(env.model, env.data)
        handle = env._get_pos_objects().copy()
        env.hand_init_pos = handle.astype(np.float32) + np.array(
            [0.0, 0.0, hand_z], dtype=np.float32
        )
        env._reset_hand()
        # _reset_hand stepped the sim; reassert the joint pos so the drawer is
        # still pre-opened.
        qpos = env.data.qpos.flat.copy()
        qvel = env.data.qvel.flat.copy()
        qpos[9] = -drawer_pull
        qvel[9] = 0.0
        env.set_state(qpos, qvel)
        mujoco.mj_forward(env.model, env.data)

    elif task_name == "drawer-close-v3":
        drawer_open_remaining = 0.04  # [difficulty] how far the drawer starts pulled out (m); bigger = harder (full open ~0.16)
        hand_z = 0.01  # [difficulty] hand height above the handle (m)
        # Drawer starts pulled out (qpos[9] negative); closing drives it to 0.
        handle = env._get_pos_objects().copy()
        env.hand_init_pos = handle.astype(np.float32) + np.array(
            [0.0, 0.0, hand_z], dtype=np.float32
        )
        qpos = env.data.qpos.flat.copy()
        qvel = env.data.qvel.flat.copy()
        qpos[9] = -drawer_open_remaining
        qvel[9] = 0.0
        env.set_state(qpos, qvel)
        env._reset_hand()

    elif task_name == "button-press-topdown-v3":
        press_remaining = 0.01  # [difficulty] button travel left to the target (m); bigger = harder (full travel 0.06)
        hand_z = 0.05  # [difficulty] hand height above the button (m)
        # Slider qpos[9] ∈ [-0.06, 0]; -0.06 = fully pressed. Pre-press to leave
        # `press_remaining`, then place the hand right above so random downward
        # motion finishes it. `_set_obj_xyz` sets the slider (qpos[9]) for this env.
        button_pre = -(0.06 - press_remaining)
        env._set_obj_xyz(button_pre)
        button = env._get_pos_objects().copy()
        env.hand_init_pos = np.array(
            [button[0], button[1], button[2] + hand_z], dtype=np.float32
        )
        env._reset_hand()
        env._set_obj_xyz(button_pre)  # reassert after _reset_hand stepped the sim

    elif task_name == "peg-insert-side-v3":
        # HARD: needs a sustained grasp + aligned insertion. Best effort: spawn
        # the peg just outside the hole along the insertion (+x) axis with the
        # hand on it. Calibrate expecting low SR.
        insert_gap = 0.05  # [difficulty] peg distance from the hole along +x (m); bigger = harder
        hand_x = 0.05  # [difficulty] hand x-offset from the peg (m); keep > ~0.02 (caging-margin safety, xz_thresh=0.005)
        # Success: peg tip within 0.07 of _target_pos (the hole); insertion is
        # toward -x, so spawn the peg at the hole + a small +x gap.
        target = env._target_pos.copy()
        peg = np.array([target[0] + insert_gap, target[1], target[2]], dtype=np.float32)
        env.obj_init_pos = peg.copy()  # consistent bookkeeping (feeds the dense reward)
        env.hand_init_pos = np.array(
            [peg[0] + hand_x, peg[1], peg[2]], dtype=np.float32
        )
        env._reset_hand()
        env._set_obj_xyz(peg)

    elif task_name == "window-open-v3":
        slide_pos = 0.10  # [difficulty] window slide position at reset (toward open); success within 0.05 of target
        hand_off = np.array(
            [0.0, -0.03, 0.0], dtype=np.float32
        )  # [difficulty] hand offset from the handle (m)
        env.data.joint("window_slide").qpos[:] = slide_pos
        mujoco.mj_forward(env.model, env.data)
        handle = env._get_pos_objects().copy()
        env.hand_init_pos = handle.astype(np.float32) + hand_off
        env._reset_hand()
        env.data.joint("window_slide").qpos[:] = slide_pos
        mujoco.mj_forward(env.model, env.data)

    elif task_name == "window-close-v3":
        slide_remaining = 0.15  # [difficulty] slide distance left to closed (m); window_slide 0.2=open, 0.0=closed, success within 0.05
        hand_off = np.array(
            [0.03, -0.03, -0.08], dtype=np.float32
        )  # [difficulty] hand offset from the handle (m), mirrors expert approach
        # Window starts open (slide 0.2); closing drives the slide to 0.
        env.data.joint("window_slide").qpos[:] = slide_remaining
        mujoco.mj_forward(env.model, env.data)
        handle = env._get_pos_objects().copy()
        env.hand_init_pos = handle.astype(np.float32) + hand_off
        env._reset_hand()
        env.data.joint("window_slide").qpos[:] = slide_remaining
        mujoco.mj_forward(env.model, env.data)

    else:
        raise ValueError(f"no cheat handler for {task_name!r}")

    return env._get_obs().astype(np.float32)


class CheatTaskEnv:
    """Wraps one MT10 env class together with its 50 goal-variant tasks. Every
    reset places the arm/world close to reward so random exploration finds it."""

    def __init__(self, name: str, env_cls, tasks, seed: int):
        self.name = name
        self.env = env_cls()
        self.tasks = tasks
        self.rng = np.random.default_rng(seed)
        # Probe obs/action dims.
        self.env.set_task(self.tasks[0])
        obs = _cheat_reset(self.env, self.name, self.rng)
        self.obs_dim = int(obs.shape[0])
        self.act_dim = int(self.env.action_space.shape[0])

    def reset(self):
        task = self.tasks[self.rng.integers(len(self.tasks))]
        self.env.set_task(task)
        obs = _cheat_reset(self.env, self.name, self.rng)
        return obs, {}

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action.astype(np.float32))
        return obs.astype(np.float32), float(r), bool(term), bool(trunc), info


def build_cheat_envs(task_names, seed: int) -> list[CheatTaskEnv]:
    mt10 = metaworld.MT10(seed=seed)
    out = []
    for i, name in enumerate(task_names):
        tasks = [t for t in mt10.train_tasks if t.env_name == name]
        out.append(CheatTaskEnv(name, mt10.train_classes[name], tasks, seed + i + 1))
    return out


# ---------------------------------------------------------------------------
# MC returns (port of compute_mc_returns from train_grid.py)
# ---------------------------------------------------------------------------


def compute_mc_returns(
    rewards: np.ndarray, dones: np.ndarray, gamma: float
) -> np.ndarray:
    T = len(rewards)
    ret = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            g = 0.0
        g = float(rewards[t]) + gamma * g
        ret[t] = g
    return ret


# ---------------------------------------------------------------------------
# Mask-overlap-filtered DDPG replay (GPU-resident; bulk-add per episode)
# ---------------------------------------------------------------------------


class MaskedReplayBufferDDPG:
    """Continuous-action analog of MaskedReplayBufferDQN from train_grid_dqn.py.

    Per-chunk archive: each chunk is tagged with the (m_a, m_c) its transitions
    were collected under. Chunks are sub-phase — the caller archives (and clears)
    the current segment at every reconfig commit, so each chunk's data matches
    exactly one mask pair. A chunk is recalled if either its actor OR critic mask
    overlaps the current mask in (rho_low, rho_high]. Storage lives on the default
    device; one H2D copy per episode field."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = torch.zeros(capacity, obs_dim)
        self.act = torch.zeros(capacity, act_dim)
        self.rew = torch.zeros(capacity)
        self.next_obs = torch.zeros(capacity, obs_dim)
        self.done = torch.zeros(capacity)
        self.ptr = 0
        self.size = 0
        self.past: list[dict] = []
        self._past_masks_a: torch.Tensor | None = None
        self._past_masks_c: torch.Tensor | None = None

    def add_episode(self, ep: dict):
        L = ep["length"]
        if L == 0:
            return
        obs_t = torch.from_numpy(ep["obs"])
        act_t = torch.from_numpy(ep["act"])
        rew_t = torch.from_numpy(ep["reward"])
        next_t = torch.from_numpy(ep["next_obs"])
        done_t = torch.from_numpy(ep["done"])
        end = self.ptr + L
        if end <= self.capacity:
            self.obs[self.ptr : end].copy_(obs_t)
            self.act[self.ptr : end].copy_(act_t)
            self.rew[self.ptr : end].copy_(rew_t)
            self.next_obs[self.ptr : end].copy_(next_t)
            self.done[self.ptr : end].copy_(done_t)
        else:
            first = self.capacity - self.ptr
            tail = L - first
            self.obs[self.ptr :].copy_(obs_t[:first])
            self.obs[:tail].copy_(obs_t[first:])
            self.act[self.ptr :].copy_(act_t[:first])
            self.act[:tail].copy_(act_t[first:])
            self.rew[self.ptr :].copy_(rew_t[:first])
            self.rew[:tail].copy_(rew_t[first:])
            self.next_obs[self.ptr :].copy_(next_t[:first])
            self.next_obs[:tail].copy_(next_t[first:])
            self.done[self.ptr :].copy_(done_t[:first])
            self.done[:tail].copy_(done_t[first:])
        self.ptr = end % self.capacity
        self.size = min(self.size + L, self.capacity)

    def sample(self, batch):
        idx = torch.randint(0, self.size, (batch,))
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.done[idx],
        )

    def clear_current(self):
        self.ptr = 0
        self.size = 0

    def archive_chunk(self, m_a: torch.Tensor, m_c: torch.Tensor):
        """Archive the current segment as a chunk tagged with the (m_a, m_c) it
        was collected under. Per-reconfig-interval chunking: callers invoke this
        right before committing new masks (and at phase end), then
        clear_current(), so every chunk's transitions match exactly one mask
        pair."""
        if self.size == 0:
            return
        self.past.append(
            {
                "obs": self.obs[: self.size].clone(),
                "act": self.act[: self.size].clone(),
                "rew": self.rew[: self.size].clone(),
                "next_obs": self.next_obs[: self.size].clone(),
                "done": self.done[: self.size].clone(),
                "m_a": m_a.detach().clone(),
                "m_c": m_c.detach().clone(),
            }
        )
        self._past_masks_a = torch.stack([ph["m_a"] for ph in self.past])
        self._past_masks_c = torch.stack([ph["m_c"] for ph in self.past])

    def sample_filtered_multi(
        self,
        m_a_cur: torch.Tensor,
        m_c_cur: torch.Tensor,
        max_chunks: int,
        max_tuples: int,
        rho_low: float,
        rho_high: float,
    ):
        """Return up to `max_chunks` (batch, m_a, m_c) triples — one per archived
        chunk whose actor OR critic mask overlap with the current masks lies in
        (rho_low, rho_high]. Each batch holds `max_tuples` transitions sampled
        (with replacement) from that chunk, to be forwarded under that chunk's
        own masks. Returns [] if none qualify. Chunks are addressed purely by
        mask overlap — no task identity. If more than max_chunks qualify, a
        uniform random subset is taken (a per-chunk weighting would instead bias
        this selection / the loss)."""
        if not self.past or self._past_masks_a is None:
            return []
        # With fixed-k masks |m_cur| == |m_chunk| == k, so each ratio is symmetric.
        denom_a = m_a_cur.sum().clamp(min=1)
        denom_c = m_c_cur.sum().clamp(min=1)
        rho_a = (self._past_masks_a * m_a_cur).sum(dim=-1) / denom_a
        rho_c = (self._past_masks_c * m_c_cur).sum(dim=-1) / denom_c
        ok = ((rho_a > rho_low) & (rho_a <= rho_high)) | (
            (rho_c > rho_low) & (rho_c <= rho_high)
        )
        candidates = ok.nonzero(as_tuple=True)[0]
        n_cand = int(candidates.numel())
        if n_cand == 0:
            return []
        if n_cand > max_chunks:
            candidates = candidates[torch.randperm(n_cand)[:max_chunks]]
        out = []
        for ci in candidates.tolist():
            ph = self.past[ci]
            N = ph["obs"].shape[0]
            idx = torch.randint(0, N, (max_tuples,))
            batch = (
                ph["obs"][idx],
                ph["act"][idx],
                ph["rew"][idx],
                ph["next_obs"][idx],
                ph["done"][idx],
            )
            out.append((batch, ph["m_a"], ph["m_c"]))
        return out


def disjoint_task_mask(ti: int, H: int, k: int) -> torch.Tensor:
    """k-hot mask for the `fixed_mask` baseline: task ti owns neurons
    [ti*k, (ti+1)*k). Caller must ensure (ti+1)*k <= H."""
    m = torch.zeros(H)
    m[ti * k : (ti + 1) * k] = 1.0
    return m


# ---------------------------------------------------------------------------
# FTN DDPG agent (mirrors FTNDQN in train_grid_dqn.py; two masks instead of one)
# ---------------------------------------------------------------------------


class FTNDDPG:
    def __init__(self, obs_dim: int, act_dim: int, cfg: DDPGConfig):
        self.cfg = cfg
        self.act_dim = act_dim
        H, k = cfg.ftn_hidden_size, cfg.ftn_k
        L, IS, D = cfg.ftn_layers, cfg.ftn_inner_size, cfg.ftn_dropout_p

        self.actor = FTNActor(obs_dim, act_dim, H, L, IS, D)
        self.critic = FTNCritic(obs_dim, act_dim, H, L, IS, D)
        self.actor_t = FTNActor(obs_dim, act_dim, H, L, IS, D)
        self.critic_t = FTNCritic(obs_dim, act_dim, H, L, IS, D)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        for p in self.actor_t.parameters():
            p.requires_grad_(False)
        for p in self.critic_t.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        if cfg.no_mask:
            # All-ones mask: linear readout sees every parallel-MLP neuron. No
            # sparsity at all. Used as a baseline alongside `fixed_mask`.
            self.m_a = torch.ones(H)
            self.m_c = torch.ones(H)
        elif cfg.fixed_mask:
            # Disjoint-subset baseline: the train loop reassigns task ti's
            # index mask at each phase start; start on task 0's subset.
            self.m_a = disjoint_task_mask(0, H, k)
            self.m_c = disjoint_task_mask(0, H, k)
        else:
            self.m_a = kwta(torch.randn(H), k=k)
            self.m_c = kwta(torch.randn(H), k=k)
        self.m_a_logits: torch.Tensor | None = None
        self.m_c_logits: torch.Tensor | None = None
        self.stored_m_a: dict[int, torch.Tensor] = {}
        self.stored_m_c: dict[int, torch.Tensor] = {}

    def reset_optimizer_state(self):
        """Per-task Adam state reset (func_net_kt paper protocol). Without it,
        momentum from the previous phase keeps stepping that task's now-masked
        neurons for ~tens of updates after the boundary even though their
        gradients are exactly zero."""
        self.actor_opt.state = defaultdict(dict)
        self.critic_opt.state = defaultdict(dict)

    @torch.no_grad()
    def act(
        self,
        obs_np: np.ndarray,
        mask_a: torch.Tensor | None = None,
        noise: float = 0.0,
    ) -> np.ndarray:
        m = self.m_a if mask_a is None else mask_a
        obs = torch.tensor(obs_np).unsqueeze(0)
        # Deterministic action selection: dropout stays a train-time
        # regularizer (gradient updates), not rollout/eval policy noise.
        was_training = self.actor.training
        self.actor.eval()
        a = self.actor(obs, m).squeeze(0).cpu().numpy()
        if was_training:
            self.actor.train()
        if noise > 0.0:
            a = a + np.random.randn(self.act_dim).astype(np.float32) * noise
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def reconfig_masks(self, support, commit: bool = True):
        """Critic mask reconfig via `cfg.reconfig_objective`, then actor mask
        reconfig via −Q under the freshly chosen critic mask.

        support = (obs, act, rew, next_obs, done, returns)."""
        obs, act, rew, next_obs, done, returns = support
        cfg = self.cfg
        H, k = cfg.ftn_hidden_size, cfg.ftn_k
        warm = cfg.mask_warm_start
        init_a = self.m_a_logits if warm else None
        init_c = self.m_c_logits if warm else None
        objective = cfg.reconfig_objective

        # --- critic mask ---
        def critic_loss(soft_m):
            q_sa = self.critic(obs, act, soft_m)
            if objective == "td":
                with torch.no_grad():
                    a_next = self.actor_t(next_obs, self.m_a)
                    q_next = self.critic_t(next_obs, a_next, soft_m)
                    target = rew + cfg.gamma * (1.0 - done) * q_next
                return F.mse_loss(q_sa, target)
            elif objective == "mse_returns":
                return F.mse_loss(q_sa, returns)
            elif objective == "cos_returns":
                cos = F.cosine_similarity(
                    q_sa.unsqueeze(0), returns.unsqueeze(0), dim=1
                ).squeeze(0)
                return -cos
            elif objective == "r_returns":
                q_c = q_sa - q_sa.mean()
                r_c = returns - returns.mean()
                r = F.cosine_similarity(
                    q_c.unsqueeze(0), r_c.unsqueeze(0), dim=1
                ).squeeze(0)
                return -r
            else:
                raise ValueError(f"unknown reconfig_objective={objective!r}")

        m_c_new, m_c_logits_new, c_init, c_final = anneal_ste_mask(
            H,
            k,
            critic_loss,
            cfg.reconfig_steps,
            cfg.reconfig_lr,
            init_logits=init_c,
            noise_scale=cfg.mask_init_noise_scale,
        )

        # --- actor mask: maximize Q under the new critic mask ---
        m_c_frozen = m_c_new

        def actor_loss(soft_m):
            a_pi = self.actor(obs, soft_m)
            return -self.critic(obs, a_pi, m_c_frozen).mean()

        m_a_new, m_a_logits_new, a_init, a_final = anneal_ste_mask(
            H,
            k,
            actor_loss,
            cfg.reconfig_steps,
            cfg.reconfig_lr,
            init_logits=init_a,
            noise_scale=cfg.mask_init_noise_scale,
        )

        stats = {
            "critic_init": c_init,
            "critic_final": c_final,
            "actor_init": a_init,
            "actor_final": a_final,
        }
        if commit:
            self.m_a = m_a_new
            self.m_c = m_c_new
            self.m_a_logits = m_a_logits_new
            self.m_c_logits = m_c_logits_new
        return m_a_new, m_c_new, stats

    def update_joint(self, batch, recalled):
        """One DDPG gradient step on the *summed* loss across the current batch
        (under the current masks) and each recalled chunk (under the masks it was
        collected under). The critic step sums the per-term TD losses; the actor
        step sums the per-term −Q losses. Every term gets its own forward pass,
        so the mask-to-data match is exact; gradients accumulate on neurons
        shared across the masks (the cross-task consolidation mechanism).

        `recalled` is a list of (batch, m_a, m_c) triples (possibly empty). One
        critic step, one actor step, and one target soft-update per call."""
        gamma, tau = self.cfg.gamma, self.cfg.tau
        terms = [(batch, self.m_a, self.m_c)]
        terms.extend(recalled)

        # --- critic: summed TD loss across all terms ---
        critic_total = None
        for b, m_a, m_c in terms:
            obs, act, rew, next_obs, done = b
            with torch.no_grad():
                a_next = self.actor_t(next_obs, m_a)
                q_next = self.critic_t(next_obs, a_next, m_c)
                target = rew + gamma * (1.0 - done) * q_next
            critic_loss = F.mse_loss(self.critic(obs, act, m_c), target)
            # --- per-chunk weighting hook (critic) -------------------------
            # Unweighted sum for now. To weight (by overlap ρ, 1/len(terms), or
            # per-neuron usage), scale critic_loss here before accumulating.
            critic_total = (
                critic_loss if critic_total is None else critic_total + critic_loss
            )
        self.critic_opt.zero_grad()
        critic_total.backward()
        self.critic_opt.step()

        # --- actor: summed −Q loss across all terms ---
        actor_total = None
        for b, m_a, m_c in terms:
            obs = b[0]
            actor_loss = -self.critic(obs, self.actor(obs, m_a), m_c).mean()
            # --- per-chunk weighting hook (actor) — see critic note above ---
            actor_total = (
                actor_loss if actor_total is None else actor_total + actor_loss
            )
        self.actor_opt.zero_grad()
        actor_total.backward()
        self.actor_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.mul_(1 - tau).add_(p, alpha=tau)
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.mul_(1 - tau).add_(p, alpha=tau)


# ---------------------------------------------------------------------------
# Episode collection (DDPG variant — continuous actions, needs next_obs)
# ---------------------------------------------------------------------------


def _collect_episode(env: CheatTaskEnv, policy_fn, max_steps: int) -> dict:
    obs_buf, act_buf, r_buf, next_buf, d_buf = [], [], [], [], []
    obs, _ = env.reset()
    success = 0
    for _ in range(max_steps):
        a = policy_fn(obs)
        ns, r, term, trunc, info = env.step(a)
        obs_buf.append(obs)
        act_buf.append(a)
        r_buf.append(r)
        next_buf.append(ns)
        d_buf.append(float(term or trunc))
        obs = ns
        if float(info.get("success", 0.0)) > 0.5:
            success = 1
        if term or trunc:
            break
    return {
        "obs": np.stack(obs_buf).astype(np.float32),
        "act": np.stack(act_buf).astype(np.float32),
        "next_obs": np.stack(next_buf).astype(np.float32),
        "reward": np.asarray(r_buf, dtype=np.float32),
        "done": np.asarray(d_buf, dtype=np.float32),
        "success": success,
        "length": len(obs_buf),
    }


def _support_from_episodes(
    episodes: list[dict],
    gamma: float,
    n_max: int,
) -> tuple[torch.Tensor, ...]:
    """Concatenate transitions, MC returns on the full concat (respects episode
    boundaries via `done`), then subsample. Returns
    (obs, act, rew, next_obs, done, returns)."""
    obs = np.concatenate([e["obs"] for e in episodes], axis=0)
    act = np.concatenate([e["act"] for e in episodes], axis=0)
    rew = np.concatenate([e["reward"] for e in episodes], axis=0)
    next_obs = np.concatenate([e["next_obs"] for e in episodes], axis=0)
    done = np.concatenate([e["done"] for e in episodes], axis=0)
    returns = compute_mc_returns(rew, done, gamma)
    N = obs.shape[0]
    if N > n_max:
        idx = np.random.choice(N, size=n_max, replace=False)
    else:
        idx = np.arange(N)
    return (
        torch.tensor(obs[idx]),
        torch.tensor(act[idx]),
        torch.tensor(rew[idx]),
        torch.tensor(next_obs[idx]),
        torch.tensor(done[idx]),
        torch.tensor(returns[idx]),
    )


# ---------------------------------------------------------------------------
# Exploration + unified eval
# ---------------------------------------------------------------------------


def _explore_for_reward(
    env: CheatTaskEnv,
    agent: FTNDDPG,
    max_episodes: int,
    max_steps: int,
    policy: str,
    prev_m_a: torch.Tensor | None = None,
    n_successful: int = 1,
) -> tuple[list[dict], int]:
    episodes = []
    n_success = 0
    act_dim = env.act_dim
    for _ in range(max_episodes):
        if policy == "random":

            def policy_fn(_obs):
                return np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:  # "prev_mask"

            def policy_fn(obs):
                return agent.act(obs, mask_a=prev_m_a, noise=agent.cfg.noise_std)

        ep = _collect_episode(env, policy_fn, max_steps)
        episodes.append(ep)
        if ep["success"]:
            n_success += 1
            if n_success >= n_successful:
                break
    return episodes, n_success


def _bootstrap_collect_rewarded(
    env: CheatTaskEnv,
    agent: FTNDDPG,
    cfg: DDPGConfig,
    gamma_rc: float,
) -> list[dict]:
    """Collect rewarded episodes for eval-time recovery (`explore_method='bootstrap'`).

    Until the first reward is found there is no support, so exploration uses
    random actions. Once any rewarded episode exists, EVERY subsequent
    exploration episode is preceded by a fresh COLD-START mask reconfiguration
    on the full cumulative rewarded support (commit=False — no single mask is
    committed). With low support the recovered masks are unreliable, so we
    deliberately avoid committing to any one of them: each episode re-derives
    the policy from scratch, drawing a fresh mask (cold-start init noise) from
    the same support, so a bad early mask can't trap exploration when it is
    still failing to find reward. Rewarded episodes accumulate into the support.
    Stops at cfg.n_successful_explore rewarded episodes or cfg.explore_episodes
    total episodes. The caller clears/restores the warm-start logits so every
    reconfig here cold-starts. Returns the rewarded episodes."""
    rewarded: list[dict] = []
    act_dim = env.act_dim
    for _ in range(cfg.explore_episodes):
        if len(rewarded) >= cfg.n_successful_explore:
            break
        if not rewarded:
            # No support yet: uniform random actions.
            def policy_fn(_obs):
                return np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            # Cold-start reconfig on the cumulative support before every episode.
            support = _support_from_episodes(rewarded, gamma_rc, cfg.reconfig_n)
            m_a_cur = agent.reconfig_masks(support, commit=False)[0]

            def policy_fn(obs, _m=m_a_cur):
                return agent.act(obs, mask_a=_m, noise=cfg.noise_std_explore)

        ep = _collect_episode(env, policy_fn, cfg.max_episode_steps)
        if ep["success"]:
            rewarded.append(ep)
    return rewarded


def _eval_with_reconfig(
    env: CheatTaskEnv,
    agent: FTNDDPG,
    cfg: DDPGConfig,
    recall_m_a: torch.Tensor | None = None,
) -> float:
    """Mirror of _eval_with_reconfig in train_grid_dqn.py. Gather a rewarded
    support set on the cheat-init env, then for each of cfg.eval_episodes trials
    run a fresh cold-started AnnealedSTE reconfig on that support and one greedy
    rollout. Support collection depends on `cfg.explore_method`:
      "random"    — random / warmup_policy rollouts until cfg.n_successful_explore
                    rewarded episodes.
      "bootstrap" — on-policy: each exploration episode after the first reward
                    acts under a fresh cold-start reconfig on the cumulative
                    rewarded support (`_bootstrap_collect_rewarded`).

    Recall baselines: when `recall_m_a` is given (fixed-mask / stored-mask —
    oracle task identity) or `cfg.no_mask` is set (all-ones mask), skip
    exploration + reconfig and just run cfg.eval_episodes greedy rollouts
    under that mask."""
    if recall_m_a is not None or cfg.no_mask:
        n_eval_succ = 0
        for _ in range(cfg.eval_episodes):

            def policy_fn(obs):
                return agent.act(obs, mask_a=recall_m_a, noise=0.0)

            ep = _collect_episode(env, policy_fn, cfg.max_episode_steps)
            if ep["success"]:
                n_eval_succ += 1
        return n_eval_succ / cfg.eval_episodes

    gamma_rc = cfg.reconfig_gamma if cfg.reconfig_gamma is not None else cfg.gamma

    # Clear warm-start logits for the whole eval so every reconfig below (the
    # bootstrap exploration and the per-eval-episode reconfigs alike) cold-starts
    # and is not contaminated by whichever task was trained last; restore at the
    # end. No mask is ever committed during eval (commit=False), so agent.m_a/m_c
    # need no save/restore.
    saved_la, saved_lc = agent.m_a_logits, agent.m_c_logits
    agent.m_a_logits = None
    agent.m_c_logits = None

    # Gather the rewarded support set.
    if cfg.explore_method == "bootstrap":
        rewarded = _bootstrap_collect_rewarded(env, agent, cfg, gamma_rc)
    else:
        eps, _ = _explore_for_reward(
            env,
            agent,
            cfg.explore_episodes,
            cfg.max_episode_steps,
            cfg.warmup_policy,
            prev_m_a=agent.m_a.clone(),
            n_successful=cfg.n_successful_explore,
        )
        rewarded = [e for e in eps if e["success"]]

    if not rewarded:
        agent.m_a_logits, agent.m_c_logits = saved_la, saved_lc
        return 0.0

    support = _support_from_episodes(rewarded, gamma_rc, cfg.reconfig_n)

    # For each eval episode, run a fresh cold-started reconfig on the rewarded
    # support and one greedy rollout (no commitment to a single mask).
    n_eval_succ = 0
    for _ in range(cfg.eval_episodes):
        agent.m_a_logits = None
        agent.m_c_logits = None
        m_a_e = agent.reconfig_masks(support, commit=False)[0]

        def policy_fn(obs, _m_a=m_a_e):
            return agent.act(obs, mask_a=_m_a, noise=0.0)

        ep = _collect_episode(env, policy_fn, cfg.max_episode_steps)
        if ep["success"]:
            n_eval_succ += 1

    agent.m_a_logits, agent.m_c_logits = saved_la, saved_lc
    return n_eval_succ / cfg.eval_episodes


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train(cfg: DDPGConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    envs = build_cheat_envs(cfg.tasks, cfg.seed)
    obs_dim = envs[0].obs_dim
    act_dim = envs[0].act_dim

    agent = FTNDDPG(obs_dim, act_dim, cfg)
    replay = MaskedReplayBufferDDPG(cfg.replay_capacity, obs_dim, act_dim)

    N = len(envs)
    matrix = np.zeros((N, N), dtype=np.float32)
    gamma_rc = cfg.reconfig_gamma if cfg.reconfig_gamma is not None else cfg.gamma

    if cfg.fixed_mask and N * cfg.ftn_k > cfg.ftn_hidden_size:
        raise ValueError(
            f"fixed_mask needs N*k <= H for disjoint subsets, got "
            f"{N}*{cfg.ftn_k} > {cfg.ftn_hidden_size}"
        )

    # fixed-mask / no-mask never run reconfig or cross-task replay; the
    # stored-mask baseline trains exactly like FTN (it only differs at eval).
    no_reconfig = cfg.fixed_mask or cfg.no_mask
    baseline_label = (
        "no-mask" if cfg.no_mask else ("fixed-mask" if cfg.fixed_mask else "")
    )
    # fixed-mask and stored-mask recall a per-task mask at eval; there is no
    # mask-selection process for unseen tasks, so skip forward-transfer cells.
    recall_eval = cfg.fixed_mask or cfg.stored_mask

    t_start = time.time()
    for ti, env in enumerate(envs):
        print(f"\n=== Phase {ti}/{N - 1}: {env.name} ===")
        replay.clear_current()
        recent_eps: deque = deque(maxlen=cfg.recent_eps_for_reconfig)
        # Optimizer-state reset at task boundaries (paper protocol, applies to
        # every condition): stale Adam momentum must not keep moving the
        # previous task's neurons once they fall outside the active mask.
        agent.reset_optimizer_state()

        # Disjoint-subset baseline: assign this task's index mask for the
        # whole phase (recalled exactly at eval).
        if cfg.fixed_mask:
            agent.m_a = disjoint_task_mask(ti, cfg.ftn_hidden_size, cfg.ftn_k)
            agent.m_c = disjoint_task_mask(ti, cfg.ftn_hidden_size, cfg.ftn_k)
            print(f"  fixed-mask: neurons [{ti * cfg.ftn_k}, {(ti + 1) * cfg.ftn_k})")

        # 1) Exploration ---------------------------------------------------
        prev_m_a = agent.m_a.clone()
        policy = "random" if ti == 0 else cfg.warmup_policy
        explore_eps, n_succ = _explore_for_reward(
            env,
            agent,
            cfg.explore_episodes,
            cfg.max_episode_steps,
            policy,
            prev_m_a=prev_m_a,
            n_successful=cfg.n_successful_explore,
        )
        n_steps_exp = sum(e["length"] for e in explore_eps)
        print(
            f"  exploration ({policy}): {len(explore_eps)} eps, "
            f"{n_steps_exp} steps, {n_succ} successful"
        )

        # 2) Mask reconfig (skipped under fixed-mask / no-mask baselines) -
        if not no_reconfig:
            if n_succ > 0:
                rewarded = [e for e in explore_eps if e["success"]]
                support = _support_from_episodes(rewarded, gamma_rc, cfg.reconfig_n)
                _, _, stats = agent.reconfig_masks(support, commit=True)
                print(
                    f"  reconfig: critic {stats['critic_init']:.3f}->"
                    f"{stats['critic_final']:.3f}  actor -Q {stats['actor_init']:.3f}->"
                    f"{stats['actor_final']:.3f}"
                )
            else:
                print("  WARN: no reward found during exploration; keeping prior masks")
        else:
            print(f"  {baseline_label} baseline: skipping reconfig")
        agent.stored_m_a[ti] = agent.m_a.detach().clone()
        agent.stored_m_c[ti] = agent.m_c.detach().clone()

        # Exploration trajectories were collected under the prior/random policy
        # *before* this phase's mask was committed. By default they are NOT
        # pushed to the replay ringbuffer — the phase is archived under its final
        # mask, and those pre-mask tuples have no valid mask correspondence. They
        # still seed recent_eps as support for the mid-phase mask reconfig. With
        # cfg.replay_explore they are also added to the ring as off-policy data
        # for the current task's DDPG updates (valid (s,a,r,s'); useful for
        # --no-mask, where the mask is constant, so successful random rollouts
        # are not discarded).
        for ep in explore_eps:
            recent_eps.append(ep)
            if cfg.replay_explore:
                replay.add_episode(ep)

        # 3) Exploitation: K episodes -------------------------------------
        train_returns, train_lengths, train_succs = [], [], []
        for trial in range(cfg.n_trials_per_task):

            def policy_fn(obs):
                return agent.act(obs, noise=cfg.noise_std)

            ep = _collect_episode(env, policy_fn, cfg.max_episode_steps)
            replay.add_episode(ep)
            recent_eps.append(ep)
            train_returns.append(float(ep["reward"].sum()))
            train_lengths.append(ep["length"])
            train_succs.append(ep["success"])

            # Gradient updates: one joint DDPG step per env step. Each step sums
            # the loss on the current segment (current masks) with the loss on up
            # to replay_max_chunks mask-overlapping archived chunks, each
            # forwarded under its own (m_a, m_c). The current segment is reset at
            # each reconfig (per-chunk exactness), so it can be small; sampling is
            # with replacement, so `size > 0` suffices. Under fixed-mask / no-mask
            # cross-task replay is a no-op (masks are disjoint resp. identical, so
            # the overlap filter never matches) — pass no recalled chunks to keep
            # those baselines pure per-mask DDPG.
            if replay.size > 0:
                n_updates = ep["length"] * cfg.updates_per_step
                for _ in range(n_updates):
                    recalled = (
                        []
                        if no_reconfig
                        else replay.sample_filtered_multi(
                            agent.m_a,
                            agent.m_c,
                            cfg.replay_max_chunks,
                            cfg.replay_max_tuples,
                            cfg.rho_low,
                            cfg.rho_high,
                        )
                    )
                    agent.update_joint(replay.sample(cfg.batch_size), recalled)

            # Periodic mid-phase reconfig using recent episodes (skipped under
            # fixed-mask / no-mask baselines).
            if (
                not no_reconfig
                and cfg.reconfig_every_trials > 0
                and (trial + 1) % cfg.reconfig_every_trials == 0
            ):
                support_eps = [e for e in recent_eps if e["success"]] or list(
                    recent_eps
                )
                if support_eps:
                    support = _support_from_episodes(
                        support_eps, gamma_rc, cfg.reconfig_n
                    )
                    # Close off the chunk collected under the outgoing masks
                    # before committing the new ones, so each archived chunk's
                    # transitions match exactly the masks they were collected
                    # under (per-reconfig-interval chunking).
                    replay.archive_chunk(agent.m_a, agent.m_c)
                    replay.clear_current()
                    agent.reconfig_masks(support, commit=True)

            if (trial + 1) % cfg.log_every == 0:
                k = min(cfg.log_every, len(train_returns))
                mean_r = float(np.mean(train_returns[-k:]))
                mean_sr = float(np.mean(train_succs[-k:]))
                mean_l = float(np.mean(train_lengths[-k:]))
                elapsed = time.time() - t_start
                print(
                    f"  trial {trial + 1:>4d}  ep_ret={mean_r:7.2f}  "
                    f"sr={mean_sr:.2f}  len={mean_l:5.1f}  ({elapsed:.0f}s)"
                )

        # 4) Archive the final chunk (data since the last reconfig) under the
        # phase's last masks (skipped under fixed-mask / no-mask). stored_m_a/c[ti]
        # now holds the *last* mask used in this task's training phase — the
        # stored-mask baseline recalls it at eval.
        agent.stored_m_a[ti] = agent.m_a.detach().clone()
        agent.stored_m_c[ti] = agent.m_c.detach().clone()
        if not no_reconfig:
            replay.archive_chunk(agent.m_a, agent.m_c)

        # 5) Eval all tasks (unified procedure) ---------------------------
        print(f"  --- Eval after phase {ti} ---")
        for tj in range(N):
            tag = "<-" if tj == ti else "  "
            if recall_eval:
                if tj > ti:
                    # No mask-selection process for unseen tasks: skip
                    # forward-transfer cells.
                    matrix[ti, tj] = np.nan
                    print(f"    [{tag}] {envs[tj].name:25s} sr=  — (skipped)")
                    continue
                sr = _eval_with_reconfig(
                    envs[tj], agent, cfg, recall_m_a=agent.stored_m_a[tj]
                )
            else:
                sr = _eval_with_reconfig(envs[tj], agent, cfg)
            matrix[ti, tj] = sr
            print(f"    [{tag}] {envs[tj].name:25s} sr={sr:.2f}")

        np.savez(
            _routed(cfg.output_path, "results", ".npz"),
            matrix=matrix,
            tasks=np.array(cfg.tasks),
            phases_done=ti + 1,
        )

    print("\n=== Eval matrix (rows: training phase, cols: eval task) ===")
    for ti in range(N):
        row = " ".join(
            "  — " if np.isnan(matrix[ti, tj]) else f"{matrix[ti, tj]:.2f}"
            for tj in range(N)
        )
        print(f"  phase {ti} ({envs[ti].name}): {row}")

    # np.savez(
    #     cfg.output_path,
    #     matrix=matrix,
    #     tasks=np.array(cfg.tasks),
    #     phases_done=N,
    # )
    plot_matrix(
        matrix,
        [e.name for e in envs],
        _routed(cfg.output_path, "images", ".png"),
    )
    config_path = _routed(cfg.output_path, "configs", ".config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    print(f"Saved config to {config_path}")
    return matrix


def _routed(output_path: str, kind: str, ext: str) -> str:
    """Route an artifact into the organized output/<kind>/ folder (gitignored).

    `output_path` is the configured results path (e.g. "results_tag_seed.npz").
    Outputs are grouped by kind: results/ (.npz), images/ (.png), configs/.
    """
    base = os.path.dirname(output_path) or "output"
    stem = os.path.splitext(os.path.basename(output_path))[0]
    out_dir = os.path.join(base, kind)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, stem + ext)


def plot_matrix(matrix: np.ndarray, names, out_path: str):
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n + 2), max(5, 0.6 * n + 1.5)))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel("Eval task")
    ax.set_ylabel("Trained through task")
    ax.set_title("Sequential MT10 cheat-init DDPG: success rate matrix")
    for i in range(n):
        for j in range(n):
            if np.isnan(matrix[i, j]):
                continue  # skipped forward-transfer cell
            color = "white" if matrix[i, j] < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="success rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


# ---------------------------------------------------------------------------
# Cheat-init calibration: random-action rollouts only, no learning.
# Use this to iterate on `_cheat_reset` per task — aim for a target band of
# success rates (e.g. 30–70%) and a non-trivial mean steps-to-first-success.
# ---------------------------------------------------------------------------


def calibrate_cheats(
    task_names: tuple[str, ...] = DEFAULT_TASKS,
    n_episodes: int = 30,
    max_steps: int = 150,
    seed: int = 0,
):
    envs = build_cheat_envs(task_names, seed)
    print(
        f"\nCalibration: {n_episodes} random-action eps per task, max_steps={max_steps}"
    )
    print(f"{'task':28s} {'sr':>5s}  {'mean_len':>8s}  {'mean_steps_to_succ':>18s}")
    for env in envs:
        n_succ = 0
        lens: list[int] = []
        first_succ_steps: list[int] = []
        for _ in range(n_episodes):
            env.reset()
            ep_succ = 0
            first = -1
            for t in range(max_steps):
                a = np.random.uniform(-1.0, 1.0, size=env.act_dim).astype(np.float32)
                _, _, term, trunc, info = env.step(a)
                if float(info.get("success", 0.0)) > 0.5 and first < 0:
                    first = t + 1
                if float(info.get("success", 0.0)) > 0.5:
                    ep_succ = 1
                if term or trunc:
                    break
            n_succ += ep_succ
            lens.append(t + 1)
            if first >= 0:
                first_succ_steps.append(first)
        sr = n_succ / n_episodes
        mean_len = float(np.mean(lens)) if lens else float("nan")
        mean_first = (
            float(np.mean(first_succ_steps)) if first_succ_steps else float("nan")
        )
        print(f"{env.name:28s} {sr:5.2f}  {mean_len:8.1f}  {mean_first:18.1f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke-test config")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run random-action rollouts under cheat init to tune _cheat_reset; no learning.",
    )
    parser.add_argument(
        "--calibrate-eps",
        type=int,
        default=30,
        help="Episodes per task in --calibrate.",
    )
    parser.add_argument(
        "--calibrate-steps",
        type=int,
        default=150,
        help="Max steps per ep in --calibrate.",
    )
    parser.add_argument(
        "--fixed-mask",
        action="store_true",
        help="Disjoint-subset baseline: task i uses neurons [i*k, (i+1)*k), "
        "recalled exactly at eval (oracle task identity); no reconfig or "
        "cross-task replay, forward-transfer evals skipped. Output suffixed "
        "with _fixed.",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="No-sparsity baseline: all-ones mask (k=H), frozen. Skips all "
        "reconfig + cross-task replay. Output suffixed with _nomask.",
    )
    parser.add_argument(
        "--stored-mask",
        action="store_true",
        help="FTN training, but eval recalls the last mask used in each "
        "task's training phase instead of explore + cold-start reconfig; "
        "forward-transfer evals skipped. Output suffixed with _stored.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Use bootstrapped eval-time exploration (explore_method='bootstrap'): "
        "after the first reward, every exploration episode cold-start-reconfigs "
        "on the cumulative rewarded support. Default is random exploration.",
    )
    parser.add_argument(
        "--replay-explore",
        action="store_true",
        help="Push exploration trajectories into the replay ring as off-policy "
        "data for the current task's DDPG updates (otherwise they only seed the "
        "reconfig support). Recommended with --no-mask so successful random "
        "rollouts actually train the agent.",
    )
    parser.add_argument("--name", default="")
    parser.add_argument("--seed", default="0")
    args = parser.parse_args()

    if args.fixed_mask + args.no_mask + args.stored_mask > 1:
        parser.error(
            "--fixed-mask, --no-mask, and --stored-mask are mutually exclusive."
        )

    if args.calibrate:
        calibrate_cheats(
            DEFAULT_TASKS,
            n_episodes=args.calibrate_eps,
            max_steps=args.calibrate_steps,
            seed=int(args.seed),
        )
    else:
        cfg = smoke_ddpg_config() if args.smoke else DDPGConfig()
        cfg.fixed_mask = args.fixed_mask
        cfg.no_mask = args.no_mask
        cfg.stored_mask = args.stored_mask
        cfg.replay_explore = args.replay_explore
        if args.bootstrap:
            cfg.explore_method = "bootstrap"
        if cfg.no_mask:
            suffix = "_nomask"
        elif cfg.fixed_mask:
            suffix = "_fixed"
        elif cfg.stored_mask:
            suffix = "_stored"
        else:
            suffix = ""
        tag = args.name + suffix
        cfg.output_path = cfg.output_path.replace(".", f"_{tag}_{args.seed}.")
        cfg.seed = int(args.seed)
        print("Config:", asdict(cfg))
        train(cfg)
