# options/

Toward learning Sutton-style **options** with FTN (`/src/`, `/base/`) + DQN
(see `rl/robotics/train_dqn*`).

## Step 1 — survival grid world (done)

A minimal 2D top-down grid world where the agent must simply **survive**.

- `grid_world.py` — `SurvivalGridEnv`, a gymnasium env (also registered as
  `SurvivalGrid-v0`).
- `play.py` — turn-based human play + ASCII diagnostics (full world + the
  agent's egocentric obs), WASD controls.

### Mechanics

- Bounded grid with static random **obstacles** (`#`), randomly-wandering
  **NPCs** (`N`), and occasionally-respawning **nutrients** (`*`).
- Agent (`@`) has `health` in (0, 1] and `hunger` in [0, 1]. Hunger rises
  linearly; at 1 it bleeds health. NPC contact deals direct health damage.
  Eating a nutrient lowers hunger.
- **Reward:** `0` per surviving step, `-1` on death (terminated). Episodes also
  truncate at `max_steps`.

### Observation (`Box([0,1], shape=(149,))` with defaults)

Flattened `7×7×3` egocentric one-hot kernel (channels = obstacle|wall, npc,
nutrient; border reads as obstacle; the centre = agent, unencoded), then
`health` and `hunger`. Actions: `Discrete(4)` = up/down/left/right.

### Run

```bash
uv run --project .. python play.py            # human play
uv run --project .. python play.py --size 24 --npcs 6 --nutrients 8
```

```python
from grid_world import SurvivalGridEnv
env = SurvivalGridEnv()           # tune size, n_npcs, n_nutrients, *_steps, npc_damage, ...
obs, info = env.reset(seed=0)
```

## Step 2 — options over RECODE centroids with FTN-DDPG

A reward-free **option-discovery** demo: an explorer agent wanders a continuous
2D world, RECODE online-clusters its egocentric observations into **centroids**,
an **FTN masked subspace** is trained into a sub-policy ("option") for *reaching
membership in each centroid*, a **behavioral adjacency graph** is built over the
centroids, and the agent then **plans (Dijkstra) and executes** a path by firing
options in sequence. Final artifact: an annotated top-down GIF.

### Pieces

- `continuous_world.py` — `ContinuousWorld`, a continuous top-down gym env.
  Action `Box(-1,1,(2,))` = `[turn, accel]` (bounded turning rotation + forward
  acceleration). A few rectangular **obstacles** (block movement/view) and several
  distinctly-colored circular **landmark** patches on the floor. **No reward, no
  termination** (optional step truncation). Observation is **purely egocentric**
  (like `grid_world`): per-ray obstacle distances + per-ray landmark RGB + floor
  color, rays fanned 360° relative to heading. True pose is tracked internally
  only for rendering.
- `options_agent.py` — `OptionsAgent`. Reuses **RECODE** (`explorer/recode`) for
  online clustering + intrinsic novelty reward, and **FTN-DDPG** (`rl/robotics/
  ftn.py`: `FTNActor`/`FTNCritic` + KWTA masks) where **each option is a mask pair
  `(m_a,m_c)`**. A `CentroidRegistry` gives RECODE's churning atoms stable IDs
  (farthest-point sampled to a target count). Options are trained by relabeling
  exploration data with an artificial reward `R·[member(next_obs)==C] − cost` and
  running `anneal_ste_mask` (mask search) + DDPG on the masked subspace.
- `centroid_graph.py` — `BehavioralGraph` (edge `i→j` when membership transitions
  directly `i→j`) + `dijkstra`.
- `train_options.py` — explore → sync centroids → rebuild graph → train every
  option, repeated for several rounds; checkpoints to `artifacts/`.
- `demo.py` — load checkpoint, read current membership, pick a random goal,
  Dijkstra-plan, fire each leg's option until membership flips, render the GIF.

> **Note on pure-egocentric obs:** centroids are *sensory situations*, so the
> graph is a sensorimotor topological map (a centroid can alias across locations).
> Each option's reward is a function of the observation the policy sees, so a
> reactive FTN policy can still learn it. Centroids are drawn at their mean
> physical footprint, so any aliasing is visible rather than hidden.

### Run

```bash
uv run --project .. python continuous_world.py --random   # env smoke -> artifacts/world.png
uv run --project .. python train_options.py --smoke       # tiny end-to-end run
uv run --project .. python train_options.py               # prototype run (~4 rounds)
uv run --project .. python demo.py                         # -> artifacts/demo.gif (5 random plan->execute tours)
uv run --project .. python demo.py --tours 1 --goal 6      # single plan to a fixed goal
uv run --project .. python demo.py --tours 8 --seed 4      # longer journey
```

`demo.py` chains `--tours` cycles: from the agent's current centroid it picks a
random reachable goal, Dijkstra-plans, fires each leg's option until membership
flips (or a per-leg timeout), then picks the next goal — one continuous journey.
The GIF overlays the centroid graph, the current plan (orange), the agent's
actual track (dim cyan), and live current/goal markers.

Notes:
- Run training on GPU (`--device cuda`) — the FTN backbone is many tiny CPU ops and
  oversubscribes cores otherwise; `train_options.py` caps CPU threads as a guard.
- A full prototype run is ~4 rounds × ~100 s on one GPU. Options reach their target
  centroid roughly a third to a half of the time from a cold start (a typical
  6-tour journey reaches ~3 goals), so multi-leg plans partially succeed; `demo.py`
  is timeout-tolerant (advances each leg on a per-leg timeout) so the GIF always
  renders full planned attempts whether or not each goal is reached.

