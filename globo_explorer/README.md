# globo_explorer

Trains the **explorer RECODE agent** (from `/explorer/agents/agent_recode.py`) on the
**globo** Minetest environment, using a first-person **vision** observation and the
**conv2d reservoir + deep conv2d decoder** design from Fluid_1S
(`obelisk: research/modeling/agents/fluid_1S`).

* **Reservoir computing**: a *fixed* 2D echo-state network (`Conv2dReservoir`, a port of
  Fluid_1S `EchoStateNetwork2D_Deep`) encodes the RGB image; only the **deep conv2d
  decoder** readout is trained. A feedback vector (prev action / reward / binned novelty)
  is folded into an extra input channel via a fixed random projection (Fluid_1S style).
* **RECODE over the images themselves**: clusters/densities are computed over the
  *downscaled pixels* (not any model latent), giving the intrinsic exploration reward and
  keeping memory small. Downscaling is configurable (`--recode-down-h/-w`, `--recode-rgb`).
* **DQN** (not DDPG): globo via craftium exposes discrete key/mouse actions, so the agent
  is the explorer logic adapted to a discrete action space.
* **Artifacts**: per-episode **GIF renders** of the explorer's first-person view in
  `artifacts/`.

## The environment

globo ships only as a Minetest *game* (Lua mods). There is no Python gym interface in
the repo, and the obelisk `minetest` conda build is private/conda-only. We therefore run
globo inside **[craftium](https://github.com/mikelma/craftium)** — a Luanti/Minetest fork
with a gymnasium API — entirely under **uv** (no pixi/conda/micromamba).

The game is **vendored** at `globo_game/` (a committed copy), so this project is
self-contained and does not depend on the repo's top-level `/globo`. `env.py` builds a
craftium `env_dir` (`craftium_globo_env/`, regenerable/gitignored) containing `games/globo`
(copied from `globo_game/`), a dependency-free `craftium_env` control mod (HUD off, midday,
terminate on death), and a `world.mt` selecting `gameid = globo`. Runs headless
(`offscreen_sdl`, no X server needed) at a small **64×32** first-person view.

> **Note:** `sync_mode` must be `False` — `sync_mode=True` crashes craftium with
> `[SyncClient] Failed to open semaphore A`.

## Setup (uv)

This is a standalone uv project (isolated venv) because craftium pins `gymnasium==0.29.1`,
which conflicts with the repo root's `gymnasium>=1.0.0`.

```sh
cd globo_explorer
uv sync            # installs craftium (prebuilt cp311 wheel, ~328MB), torch, numpy, imageio
```

If `uv sync` cannot fetch the wheel, download it once and install directly:

```sh
gh release download v0.0.1 -R mikelma/craftium \
    -p 'craftium-0.0.1-cp311-cp311-manylinux_2_28_x86_64.whl' -D /tmp/wheels
uv pip install /tmp/wheels/craftium-0.0.1-cp311-cp311-manylinux_2_28_x86_64.whl torch numpy imageio
```

## Run

Everything is driven by a YAML config — the **only** CLI argument is `--config` (defaults to
`default.yaml`). Edit `default.yaml` to change a run, or point `--config` at another file.

```sh
# uses default.yaml (indefinite die-and-restart, daytime, fixed world, weather_seed 0)
uv run python train.py

# tiny end-to-end smoke run (overlays default.yaml with small values, on CPU)
uv run python train.py --config smoke.yaml

# your own run: a small file that overrides only the keys you change
uv run python train.py --config my_run.yaml
```

A `--config` file is **overlaid on top of `default.yaml`**, so a custom file only needs the keys it
changes; everything else falls back to `default.yaml`. Unknown keys are rejected with a list of
known ones. `default.yaml` lists and documents every parameter and mirrors the code defaults.

### Parameters (all set in the YAML)
- **Control scheme** — `action_mode: discrete` = one action per step, a single DQN over `Discrete(10)`; `action_mode: factored` = a **branching/factored DQN**: 5 independent Q-heads over the mutually-exclusive groups `look_h[left/noop/right]`, `look_v[up/noop/down]`, `jump[noop/jump]`, `move[fwd/noop/back]`, `strafe[left/noop/right]` (`MultiDiscrete([3,3,2,3,3])`). The engine natively accepts these concurrently — move + strafe + jump + look in one step. Each head trains by its own TD target (shared reward); the conv-decoder trunk is shared.
- **View / RECODE** — `obs_width/obs_height` (64×32); `recode_down_h/_w`, `recode_rgb` (embedding resolution/color); `recode_memory`, `recode_k`, `recode_knn_k`, `recode_kappa`, `recode_gamma`, `recode_insertion_probability`, `recode_decay`; `exploration_coef` / `env_reward_coef` (mix RECODE vs in-game reward).
- **DQN** — `gamma`, `lr`, `batch`, `replay_capacity`, `target_update_tau` (polyak), `epsilon_start`/`epsilon_end`/`epsilon_decay`, `reservoir_channels`, `decoder_layer_size`, `num_novelty_bins`.
- **Training cadence** — `train_iters` (K) updates every `train_interval` (J) collected steps (cf. the explorer's per-episode update loop); `warmup`.
- **In-episode GIF peeks** — `render_window` (steps each GIF spans; ~10s ≈ 2700–3700 steps at ~3–7 ms/step), `render_interval` (steps between peeks), `render_max_frames` (frame cap; window stride-subsampled), `gif_seconds`. Frames are captioned `ep s{step} t={game_seconds}s` and play back at real game-time.
- **Episode control** — `episodes: 0` runs indefinitely; `max_steps: 0` removes the per-episode cap (episode ends only on death). `invincible: true` pins HP/stats so the player can't die (one endless episode); `false` lets it die and restart.
- **Reproducible world** — `seed` fixes terrain; `spawn: "x,y,z"` fixes the spawn location (`static_spawnpoint`) and the control mod fixes the initial facing; `freeze_daylight: true` stops the day/night cycle (stays midday) so observations don't go dark; `weather_seed` makes weather & clouds reproducible/selectable (globo picks weather via `math.random`, which the engine otherwise seeds from wall-clock time). The control mod reseeds Lua's RNG before globo's `climate` mod loads (climate is made to `optional_depends` on it), so a given `weather_seed` always yields the same weather, e.g. `0→overcast_light_snow`, `2→superstorm`, `68→clear`.

> **Determinism note:** seed + spawn + facing + daylight + weather_seed make the *terrain, start location, viewpoint, lighting, and weather/clouds* identical on every restart, and the env is rebuilt fresh each restart so player state doesn't accumulate. The only remaining variation is globo's wandering animals/mobs (spawned via runtime RNG), so observations are very close but not pixel-identical — by design of the living world.

## Files

| File | Purpose |
|---|---|
| `env.py` | Builds the craftium globo `env_dir`; headless discrete-action gym env + obs preprocessing. |
| `models.py` | `Conv2dReservoir` (fixed ESN), `DeepConvDecoder` (trained readout), feedback→channel projection, RECODE downscaling. |
| `agent.py` | `GloboExplorerDQN`: reservoir front-end + conv-decoder Q-net + RECODE-over-images intrinsic reward + replay. |
| `train.py` | Training loop; saves GIF artifacts. |
| `recode/` | RECODE density estimator (from `/explorer`). |
| `globo_game/` | Vendored copy of the globo Minetest game (committed; makes this project self-contained). |
| `craftium_globo_env/` | Generated craftium env dir (built from `globo_game/`; gitignored/regenerable). |
