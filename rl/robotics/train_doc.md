# Documentation: robotics/train.py — Sequential FTN-DDPG continual learning on Meta-World MT10 (cheat-init)

## Overview

`train.py` is the robotics / reinforcement-learning instantiation of Functional
Task Networks (FTN). It trains a single agent **sequentially** on four robotic
manipulation tasks from Meta-World's MT10 benchmark, using DDPG (Deep
Deterministic Policy Gradient) for continuous control on top of an FTN backbone:
a wide bank of parallel sub-networks gated by a sparse top-k mask. The central
question is the same as the FTN paper's: after learning a sequence of tasks, can a
support-set-driven *mask reconfiguration* recover a working policy for any task
(retention and transfer) without being told the task identity? Here the
"support set" is a handful of *rewarded trajectories* the agent gathers by
exploration, and the mask is selected to make the critic's value estimates line
up with the observed returns.

It is a direct port of an earlier grid-world DQN version (`train_grid_dqn.py`) to
continuous-control DDPG, so it carries two masks (one for the actor, one for the
critic) instead of the single Q-network mask of the discrete version.

## The "cheat init" and the four tasks

Meta-World exploration is hard: random actions almost never complete a task, so
there would be no reward signal to reconfigure a mask from. To sidestep the pure
exploration problem and focus on the FTN mechanism, the environments are
**cheat-initialized**: on every reset the arm and the world are placed close to
the reward/goal state, so random actions succeed within ~150 steps a meaningful
fraction of the time. The four tasks (with hand-written cheat handlers in
`_cheat_reset`) are: `reach-v3`, `drawer-open-v3`, `window-open-v3`, and
`drawer-close-v3`. For each, `_cheat_reset` resets the env, then nudges the hand
initial position near the object and/or pre-sets the relevant joint (e.g. the
drawer slide `qpos[9]` to nearly-closed/open, or the window slide joint near the
target) and re-runs the MuJoCo forward kinematics so random rollouts can hit
success quickly.

`CheatTaskEnv` wraps one MT10 environment class together with its 50 goal-variant
tasks; each `reset()` samples a goal variant and applies the cheat. `step()`
forwards the action and returns `(obs, reward, terminated, truncated, info)`,
where `info["success"]` flags task completion. `build_cheat_envs` constructs one
`CheatTaskEnv` per task name from `metaworld.MT10`. `calibrate_cheats` is a tool
that runs random-action rollouts only (no learning) to tune the cheat handlers,
targeting a useful band of success rates (e.g. 30–70%).

## The FTN-DDPG agent

`FTNDDPG` holds four networks: an actor and critic, plus Polyak-averaged target
copies (`actor_t`, `critic_t`). The actor (`FTNActor`) maps observations to a
continuous action; the critic (`FTNCritic`) maps (observation, action) to a scalar
Q-value. Both are built on the FTN backbone (`ftn_hidden_size = H` parallel
sub-networks, each a small deep MLP of `ftn_layers` layers and `ftn_inner_size`
width, with dropout `ftn_dropout_p`), and both take a binary mask argument that
selects `ftn_k` active sub-networks. There are **two separate masks**: `m_a` for
the actor and `m_c` for the critic.

`act(obs, mask_a, noise)` selects a deterministic action under the actor mask
(dropout is disabled for action selection so it stays a train-time regularizer,
not rollout noise), optionally adding Gaussian exploration noise and clipping to
[-1, 1]. `reset_optimizer_state()` clears the Adam state at every task boundary —
the FTN paper protocol — because stale momentum would otherwise keep updating the
previous task's now-masked-out neurons even though their gradients are zero.

### Mask reconfiguration (`reconfig_masks`)

This is the FTN core. Given a support set of rewarded transitions
`(obs, act, rew, next_obs, done, returns)`, it selects new masks by optimizing a
soft mask with an **annealed straight-through estimator** (`anneal_ste_mask`,
imported from `ftn`) over `reconfig_steps` inner steps at learning rate
`reconfig_lr`, optionally warm-started from the current logits
(`mask_warm_start`).

- **Critic mask first**, by minimizing `cfg.reconfig_objective`. The default is
  `"r_returns"`: maximize the Pearson correlation between the critic's Q-values
  `Q(s,a)` and the Monte-Carlo returns `G_t` (implemented as negative
  centered-cosine, i.e. `-Pearson`). Other objectives are `"td"` (TD error MSE
  against the bootstrapped target), `"mse_returns"` (MSE to returns), and
  `"cos_returns"` (negative cosine to returns).
- **Actor mask second**, by maximizing the critic's value under the freshly
  chosen (frozen) critic mask: the actor loss is `-Q(s, μ(s))` averaged.

The masks are committed (and their logits cached for warm starting) when
`commit=True`. So the agent infers *which sub-networks make the value function
explain the observed returns*, then *which sub-networks make the policy maximize
that value* — all from a small batch of rewarded experience, never from an
explicit task label.

### DDPG update (`update`)

A standard DDPG step under the current (or supplied) masks: form the bootstrapped
target `r + γ(1-done)·Q_target(s', μ_target(s'))[m_c]`, take an MSE critic step,
then a deterministic-policy-gradient actor step `-Q(s, μ(s))`, and finally Polyak
(soft) update of both target networks with rate `tau`.

## Replay buffer with mask-overlap filtering

`MaskedReplayBufferDDPG` is a GPU-resident ring buffer of transitions for the
*current* task, plus an archive of *past* phases. When a task phase finishes,
`archive_phase(m_a, m_c)` snapshots that phase's transitions together with the
actor and critic masks in use. During later tasks, `sample_filtered` draws a
replay batch only from past phases whose stored masks **overlap the current masks
moderately** — fractional overlap `rho` in `(rho_low, rho_high]` (default
(0.2, 0.8]) for either the actor or critic mask. This is the FTN retention
mechanism: replay supervision is spent on exactly the sub-networks shared between
the current task and a prior task (so interference is corrected there), while
disjoint sub-networks are protected by the mask gating and never replayed. Past
phases that share too little (rho ≤ 0.2) or almost everything (rho > 0.8) are
skipped. `add_episode` bulk-copies an episode into the ring buffer; `sample`
draws uniformly from the current task's data; `clear_current` resets the ring at
each task boundary (the data is preserved in the archive if it was archived).

`compute_mc_returns` computes discounted Monte-Carlo returns over a concatenation
of episodes, respecting episode boundaries via the `done` flags;
`_support_from_episodes` concatenates rewarded episodes, computes returns, and
subsamples up to `reconfig_n` transitions to form the reconfiguration support set.

## The training protocol (per task)

`train` loops over the four tasks in order. For each task phase `ti`:

1. **Exploration** — `_explore_for_reward` runs cheat-init rollouts (random
   actions on task 0, or `warmup_policy` thereafter) until `n_successful_explore`
   rewarded episodes are collected or `explore_episodes` is hit. The rewarded
   episodes are the support set.
2. **Boundary mask reconfig** — if any reward was found, `reconfig_masks` selects
   new actor/critic masks from that support (critic via −Pearson(Q, returns),
   actor via −Q). If no reward was found, the prior masks are kept.
3. **Exploitation** — `n_trials_per_task` (K) episodes with a Gaussian-noise
   actor. After each episode, the buffer gets one DDPG update per env step from
   the current ring buffer, **plus** `mask_replay_n_per_step` mask-overlap-filtered
   updates drawn from archived past-task transitions (the retention channel).
4. **Periodic mid-phase reconfig** — every `reconfig_every_trials` trials, the
   masks are re-selected using the most recent (preferably rewarded) episodes as
   support, so the masks track the policy as it improves. Per-batch / frequent
   reconfiguration is what builds a broad basis of partially trained sub-networks,
   mirroring the FTN paper's finding.
5. **Archive + eval** — the phase's data and final masks are archived for future
   cross-task replay, then **all** tasks are evaluated.

### Unified evaluation (`_eval_with_reconfig`)

Evaluation mirrors training and never uses task identity: random-action explore
under cheat init until `n_successful_explore` rewarded episodes, then for each of
`eval_episodes` trials a **fresh cold-started** AnnealedSTE reconfiguration on the
rewarded support (warm-start logits cleared so each eval reconfig starts from
scratch), followed by one greedy (noise-free) rollout. The success rate fills the
`matrix[ti, tj]` entry (row = training phase, column = eval task). The
lower-triangular entries measure retention of earlier tasks; the diagonal is the
just-trained task; the upper triangle is forward transfer to not-yet-trained
tasks.

## Baselines (catastrophic-forgetting references)

Three flags switch the FTN agent into matched-architecture controls so the result
matrices are directly cell-comparable:

- **`fixed_mask`** — oracle parameter isolation: task `i` is permanently assigned
  the disjoint neuron block `[i*k, (i+1)*k)` (`disjoint_task_mask`, requires
  `N*k ≤ H`), recalled exactly at eval using oracle task identity. No reconfig,
  no cross-task replay (the disjoint masks never overlap). Forward-transfer cells
  are skipped (NaN), since there is no mask-selection process for unseen tasks.
- **`no_mask`** — no sparsity at all: an all-ones mask (`k = H`), frozen for the
  whole run. Plain dense DDPG over the FTN backbone. Isolates the value of *any*
  sparsity. No reconfig or cross-task replay.
- **`stored_mask`** — full FTN training (boundary + mid-phase reconfig,
  cross-task replay), but at eval it recalls the last mask used during each task's
  training phase instead of running explore + cold-start reconfig. Isolates eval-
  time mask *recovery* from mask *storage*.

Everything else (buffer, DDPG updates, exploration) is identical across
conditions, so the matrices compare cleanly. The script names outputs with
suffixes `_nomask`, `_fixed`, `_stored` accordingly.

## Key configuration (`DDPGConfig`)

- Tasks: `reach-v3`, `drawer-open-v3`, `window-open-v3`, `drawer-close-v3`.
- Exploration: `n_successful_explore = 1000` rewarded eps target,
  `explore_episodes = 10000` cap, `max_episode_steps = 150`.
- Exploitation: `n_trials_per_task = 150` (K), `eval_episodes = 8`.
- DDPG: `gamma = 0.95`, `tau = 0.01`, `actor_lr = 3e-4`, `critic_lr = 5e-4`,
  `batch_size = 64`, `replay_capacity = 200_000`, `noise_std = 0.1`.
- FTN backbone: `ftn_hidden_size = 2048` (H), `ftn_k = 512` active,
  `ftn_inner_size = 8`, `ftn_layers = 8`, `ftn_dropout_p = 0.1`.
- AnnealedSTE: `reconfig_steps = 60`, `reconfig_lr = 0.3`, `reconfig_n = 4096`
  support cap, `mask_init_noise_scale = 1e-4`, `mask_warm_start = True`.
- Critic objective `reconfig_objective = "r_returns"` (negative Pearson of Q vs
  returns); mid-phase reconfig every `reconfig_every_trials = 10` trials using
  `recent_eps_for_reconfig = 10` recent episodes.
- Cross-task replay: `mask_replay_n_per_step = 1`, overlap window
  `rho_low = 0.2`, `rho_high = 0.8`.

`smoke_ddpg_config` is a tiny version (2 tasks, small bank, few episodes) for fast
testing. The CLI exposes `--smoke`, `--calibrate` (tune cheats, no learning),
`--fixed-mask`, `--no-mask`, `--stored-mask`, `--name`, and `--seed`; results
(the success-rate matrix and a heatmap PNG via `plot_matrix`) are saved to an
`.npz` named per the baseline and seed.

## Summary

In one sentence: `train.py` trains an FTN-DDPG agent sequentially over four
cheat-initialized Meta-World tasks, where a support set of self-collected rewarded
trajectories drives an AnnealedSTE reconfiguration of separate actor and critic
top-k masks (critic mask chosen to maximize Pearson correlation between Q-values
and Monte-Carlo returns, actor mask chosen to maximize Q), retention is protected
by mask-overlap-filtered cross-task replay, and evaluation recovers each task's
policy from a fresh support set without any task identity — with fixed-mask,
no-mask, and stored-mask baselines for comparison.
