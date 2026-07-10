# FTN Robotics — continual RL with Functional Task Networks

Reinforcement-learning / robotic-control instantiation of **Functional Task
Networks (FTN)**. One agent learns a *sequence* of tasks on a single shared
backbone: a wide bank of `H` parallel small MLPs gated by a sparse top-`k`
binary mask. Task identity lives in the **mask**, never in the observation.
The central question: after sequential training, can a support-set-driven
*mask reconfiguration* (AnnealedSTE search) recover a working policy for any
task — including retention of old tasks and **transfer to tasks never
trained on** — without being told the task identity?

Core mechanisms (all in `ftn.py`):

- **FTN backbone** — `ParallelMLP` bank + top-k mask; `FTNActor` (tanh policy)
  and `FTNCritic` (Q-value) wrappers.
- **AnnealedSTE mask reconfig** — gradient search over mask logits against a
  task-specific objective (TD error, −Q, BC loss, ...), annealed
  straight-through top-k. Optional lateral spatial smoothing
  (`LateralSmoother`) clusters each task's mask into a contiguous blob on a
  square grid.
- **Mask-overlap-filtered replay (consolidation)** — archived past-task data
  is replayed *under its own mask* whenever the current mask overlaps it in
  `(rho_low, rho_high]`, so shared neurons get gradients from both tasks.

## Running

Everything runs with `uv` from this directory:

```bash
uv run python <script>.py [flags]        # every trainer has --smoke for a fast sanity run
```

## Tasks / environments

| Suite | Tasks | Used by |
|---|---|---|
| Grid-world | Reward hidden at a different cell per task; obs = agent position only | `train_grid_ppo.py`, `train_grid_dqn.py`, `train_grid_dqn_async.py` |
| JAX point-mass | 2-D double integrator, one goal location per task (goal not in obs) | `train_jax_debug.py` |
| Meta-World MT10 (cheat-init) | `reach`, `drawer-open`, `drawer-close`, `window-open` (v3) — envs reset near the goal so random exploration finds reward | `train.py`, `train_mt10_ddpg.py` |
| Continual World CW10/CW20 | Wołczyk et al. protocol reproduced on MetaWorld 3.x, standard resets | `train_cw.py` |
| Meta-World MT10 (expert BC) | `door-open`, `drawer-open`, `drawer-close`, `button-press-topdown`, `window-open`, `window-close`, `peg-insert-side`, `pick-place` (v3), standard resets | `train_mt10_bc.py` |

## Training pipelines

All trainers share the same continual protocol — per task: collect a support
set → reconfigure the mask on it → train under the committed mask (+
consolidation replay) → archive → eval *every* task → success-rate matrix
(`results_*.npz` / `.png`). They differ in algorithm and how the support set
is obtained:

- **`train_grid_ppo.py` / `train_grid_dqn.py`** — grid-world proof of concept
  (PPO / DQN). Support = the first rewarded exploration trajectories; DQN
  variant has the full mask-overlap replay. `train_grid_dqn_async.py`
  vectorizes eval. `run_many.sh <name>` launches 8 seeds of the DQN version.
- **`train_jax_debug.py`** — standalone all-JAX re-implementation on the
  point-mass env; runs in seconds, used to tune FTN+DDPG hyperparameters.
- **`train.py`** — FTN-DDPG on cheat-init MT10, per-env-step interleaved
  updates. Critic mask found by maximizing Pearson(Q, returns) on rewarded
  rollouts, actor mask by −Q. See `train_doc.md` for a full walkthrough.
- **`train_mt10_ddpg.py`** — same, but a decoupled collect/train loop with
  parallel envs (faster; mirrors the JAX harness structure).
- **`train_cw.py`** — the same FTN-DDPG on the Continual World sequence
  (standard resets, no cheat).
- **`train_mt10_bc.py`** — **behavior cloning** from Meta-World's scripted
  expert policies (no critic, no exploration). The expert `(obs, action)`
  data *is* the support set: mask reconfig minimizes BC loss on it. Demos are
  cached in `expert_data/` and shared across conditions. This is the trainer
  the visualizer below consumes.

Every trainer supports the same four conditions:

| Flag | Condition |
|---|---|
| *(default)* | Full FTN: masks recovered from support data at eval |
| `--stored-mask` | FTN training, but eval recalls each task's stored training mask |
| `--fixed-mask` | Oracle: task *i* owns disjoint neurons `[i*k, (i+1)*k)` |
| `--no-mask` | All-ones mask (shared net) — catastrophic-forgetting reference |

### `train_mt10_bc.py` outputs

```bash
uv run python train_mt10_bc.py --name myrun --seed 0
```

- `results_expert_myrun_ftn_0.npz` / `.png` — the eval matrix (rows = trained
  through task *i*, cols = eval task *j*). **Off-diagonal cells with j > i are
  forward transfer**: tasks the model has *not* trained on yet, solved purely
  by mask reconfig. Use this matrix to pick knowledge-transfer demos.
- `results_expert_myrun_ftn_0.pt` — rolling checkpoint, overwritten after each
  task (disable with `--no-ckpt`).
- `knowledge_transfer/results_expert_myrun_ftn_0_ntasksK.pt` — a snapshot
  saved **before** training on each task: `ntasks0` is the untrained net,
  `ntasks3` has trained on exactly the first 3 tasks, ..., plus a final
  `ntasksN` after the last phase. These are the partially-trained models for
  transfer demos.

## Visualizing with `viz_mt10_expert.py`

Renders greedy rollouts of a `train_mt10_bc.py` checkpoint as a side-by-side
GIF: MuJoCo arm render on the left, live FTN mask activity on the right
(active units white, fading purple → blue as the mask moves between tasks).
By default it retries each episode until success (`--no-success-only` to
disable) and, with `--mask-source reconfig` (the default), recovers each
task's mask at eval time from that task's expert support — model weights are
never updated.

### 1. Visualize learned solutions

Point it at a checkpoint; with no `--tasks` it rolls out every task the model
has trained on, in training order:

```bash
uv run python viz_mt10_expert.py --ckpt results_expert_myrun_ftn_0.pt \
    --episodes 1 --out rollout.gif
```

`--mask-source stored` uses each task's mask saved at training time instead
of re-searching it. Other useful knobs: `--camera`, `--frame-stride`,
`--reconfig-every N` (re-search the mask mid-rollout to show it adapting).

### 2. Visualize *unlearned* solutions (knowledge transfer)

Load a partially-trained `ntasksK` snapshot and select hold-out tasks with
`--tasks`. The mask is recovered from the hold-out task's expert
demonstrations by cold-start AnnealedSTE reconfig — the FTN
knowledge-transfer mechanism — while the weights remain those of a model that
never trained on that task:

```bash
# Model that has trained on the first 3 tasks, demoed on two hold-outs:
uv run python viz_mt10_expert.py \
    --ckpt knowledge_transfer/results_expert_myrun_ftn_0_ntasks3.pt \
    --tasks window-close-v3 drawer-close-v3 --out transfer.gif
```

Workflow: read row `K` of the training-run eval matrix — hold-out columns
(`j >= K`) with high success rates are the tasks worth demoing from the
`ntasksK` checkpoint. `--tasks` accepts any MT10 task names in any order;
hold-outs require `--mask-source reconfig` (default), since no stored mask
exists for them. Missing expert demos for a hold-out are generated on the fly
and cached in `expert_data/`.
