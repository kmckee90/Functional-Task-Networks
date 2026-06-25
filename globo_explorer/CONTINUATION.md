# globo_explorer — continuation notes

Handoff context for resuming this project. Last updated 2026-06-24.

## What this is
A vision RL "explorer" trained on the **globo** Minetest game through a first-person RGB
gym env. It ports the explorer agent (`/explorer/agents/agent_recode.py`, an RDDPG_Dual
DDPG + RECODE intrinsic reward) to:
- a **conv2d echo-state reservoir + deep conv2d decoder** (Fluid_1S design, ported to
  PyTorch in `models.py`); reservoir + feedback projection are **frozen**, only the
  decoder (Q-heads) trains;
- **DQN** (discrete) or **branching DQN** (factored) instead of DDPG;
- **RECODE intrinsic reward over the downscaled image itself** (not latents), with an
  **offline** (recomputed each train round) and **online** (collection-time) novelty term,
  a **goal buffer** + **prioritized trajectory replay** toward the exploration frontier.

Status: end-to-end working; a 50k-step test passed earlier. Currently in a control/physics
+ hyperparameter tuning phase (user editing `default.yaml`).

## Files
- `env.py` — builds the craftium env-dir for globo; `make_globo_env(action_mode=discrete|factored|raw, ...)`; control mod (HUD off, midday, terminate-on-death, invincibility, weather reseed, fly/fast revoke); physics via minetest_conf.
- `models.py` — `Conv2dReservoir` (frozen ESN), `DeepConvDecoder` (one Q-head per action group), `FeedbackToChannel`, `downscale_embedding`.
- `agent.py` — `GloboExplorerDQN`: reservoir front-end + conv-decoder Q-net + RECODE (online+offline) + goal buffer + prioritized trajectory replay + centroid dump.
- `train.py` — single CLI arg `--config` (default `default.yaml`); loads config, runs train loop, saves GIF peeks + centroids.
- `play.py` — human play in browser over SSH (MJPEG + key input); `--parity` = agent-faithful one-step-per-action.
- `default.yaml` — THE config; glossary + RECODE tuning guide + params. Single source of truth.
- `smoke.yaml`, `factored.yaml`, `test_50k.yaml` — overlay configs.
- `globo_game/` — vendored globo game (committed); env built from here, so `/globo` can be deleted.
- `recode/` — RECODE density estimator (copied from `/explorer`).
- `artifacts/` — GIF peeks + `centroids/<i>.png` + `!masses.csv`; `cleanup.sh` clears them.

## Config system
- **Only one CLI arg**: `--config` (defaults to `default.yaml`). Every parameter lives in the YAML.
- A custom `--config` is **overlaid on top of `default.yaml`** (only specify keys you change).
- `default.yaml` mirrors the code defaults and documents every param (glossary) + a RECODE tuning guide.

## Critical env / craftium gotchas (hard-won — see also project memory)
- globo has **no Python gym**; we run it inside **craftium** (Luanti 5.12 fork) via its
  prebuilt **cp311 manylinux wheel** (GitHub release `mikelma/craftium` v0.0.1), installed in
  an **isolated `.venv`** (craftium hard-pins `gymnasium==0.29.1`, conflicting with the repo root).
- **`torch` is pinned to `2.6.0`** in this project's `pyproject.toml`. A `uv sync` once pulled
  `torch 2.12.1` with a broken cu13 build (missing `libcudnn.so.9`) → torch wouldn't import.
  If it breaks again: `uv pip install 'torch==2.6.0'`. Box: 8× H100, driver 575 / CUDA 12.9.
- **`sync_mode=False` is mandatory** (True crashes with `[SyncClient] Failed to open semaphore A`).
- Env is built fresh each restart from `globo_game/`; `craftium_globo_env/` and `craftium_play_env/` are gitignored/regenerable.
- Reproducible world = `seed` (terrain) + `spawn` "x,y,z" (`static_spawnpoint`; engine spawn search varies otherwise) + control mod fixes initial facing + `freeze_daylight` + `weather_seed`
  (globo picks weather via `math.random`, engine seeds Lua RNG from wall-clock; our control mod
  reseeds before `climate` loads — `climate` is made `optional_depends = craftium_env`). Mobs still vary.
- **Seed 7** = a dry, above-ground spawn at `8,8,115` (most seeds put y=8 underground or in water; seed 7's spawn is in a climbable bush though). `weather_seed`: 0→overcast_light_snow, 2→superstorm, 68→clear.
- craftium obs are `(H, W, 3)`; turn magnitude = `int(mouse × obs_width//2)` so it scales with
  resolution — `make_globo_env` normalizes `mouse_mov` to a 64px reference so yaw/step is fixed.

## Physics (recent, important)
The "floaty / low-gravity" feel was diagnosed: gravity is actually ~2× the setting (strong).
The real cause is **slow-motion** — at `frameskip=1` each env step is only ~5 ms of game-time, so
a 0.6 s fall = ~120 steps; acceleration phases (falls/jumps) look floaty, constant-velocity
walking/cows don't. **Fix = `frameskip`** (exposed; default 4; higher = more game-time/step,
≈real-time at ~10). Also replaced craftium's opaque `pmul` with explicit physics:
`move_speed`, `move_accel` (high → instant start/stop, no momentum), `climb_speed` (small →
climbable canes/plants don't launch you), `jump_speed`, `gravity`, plus revoking `fly/fast/noclip`.

## Reward / goal system (matches the stated intent, not agent_recode's literal dead-code)
- Novelty = RECODE kNN distance **divided by sqrt(embedding dim)** (RMS per-dim → scale stable across `recode_down`).
- Separate **`online_coef`** and **`offline_coef`** (replaced the single `exploration_coef`).
- Each train round: find the lowest-density (highest-novelty) **frontier**, give it the sparse bonus,
  move the cohesive trajectory leading to it into the **goal buffer**, oversample goal/last/random
  contiguous trajectories so Bellman backups propagate the bonus (long-range plan to frontier).
- NOTE: agent_recode.replay overwrote the offline reward with the stored reward (`rewards_o = reward`) —
  user said that was a test artifact; here the offline reward DOES enter the loss when enabled.
- `grad_clip` added (default 10).

## Open issues / watch
- **Loss divergence** with aggressive settings (`recode_knn_k:1`, sparse scales 5, high update ratio,
  `target_update_tau:1.0` = hard target sync). The dimensionality normalization + online/offline coefs
  + grad_clip were added to help. If it still diverges: lower `target_update_tau` (~0.005–0.01),
  raise `recode_knn_k`, lower update ratio, or normalize/clip reward further.
- **`default.yaml` currently has `move_speed: 200`, `move_accel: 5000`** (user was compensating for the
  slow-mo). With `frameskip` now handling time-scale, `move_speed` ~8–15 is saner (200 m/s × 20 ms/step
  ≈ 4 m/step is very fast). Suggest retune.
- **Climbable plants** (canes + one plant node, `walkable=false, climbable=true`) still cap vertical to
  `climb_speed` when overlapped ("jump up a plant = climb/hang"). Could make non-climbable via mod override if wanted.
- **globo HUD** (hunger/thirst/energy/temp) is drawn into the obs (its own hud elements, not removed by
  `hud_set_flags`). Tiny at 64×32, prominent at higher res. Could call `globo.deallocate_hud_for_player` to remove.
- RECODE granularity for 3D frames: see the tuning guide in `default.yaml` (raise `recode_kappa`,
  coarsen `recode_down`, raise `recode_insertion_probability` to let atoms spread off the initial sky).

## Run / kill / play
```sh
cd globo_explorer
uv run python train.py                      # uses default.yaml (background: nohup ... > logs/train.log & ; echo $! > logs/run.pid)
kill $(cat logs/run.pid)                     # graceful stop (traps SIGTERM, closes env)
pkill -9 -f '[l]uanti/bin/luanti'            # backstop for stray engine procs
uv run python train.py --config smoke.yaml   # tiny smoke
uv run python train.py --config test_50k.yaml
uv run python play.py [--parity]             # human play; ssh -L 8000:localhost:8000 ; http://localhost:8000
```
A running train job uses old in-memory code — relaunch to pick up edits. play.py uses no GPU/torch.

## Quick verify pattern (used throughout)
Build a raw env with a temp `env_dir` under `/tmp`, step it, read `info["player_vel"]/["player_pos"]/["mt_dtime"]`,
or load config via `train.load_config()` and construct `GloboExplorerDQN` to check shapes — all on CPU,
isolated from any running job. Render frames to PNG and Read them to eyeball.
