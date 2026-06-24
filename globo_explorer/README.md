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

globo ships only as a Minetest *game* (Lua mods in `/globo`). There is no Python gym
interface in the repo, and the obelisk `minetest` conda build is private/conda-only. We
therefore run globo inside **[craftium](https://github.com/mikelma/craftium)** — a
Luanti/Minetest fork with a gymnasium API — entirely under **uv** (no pixi/conda/micromamba).

`env.py` builds a craftium `env_dir` (`craftium_globo_env/`) containing `games/globo`
(copied from `/globo`), a dependency-free `craftium_env` control mod (HUD off, midday,
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

```sh
# tiny end-to-end smoke test (2 episodes, 40 steps) -> artifacts/globo_explorer_ep*.gif
uv run python train.py --smoke --device cpu

# small real run
uv run python train.py --episodes 50 --max-steps 500

# reduce RECODE memory footprint if OOM (smaller downscale / grayscale / fewer atoms)
uv run python train.py --recode-down-h 8 --recode-down-w 16 --recode-memory 1000
```

Key flags: `--obs-width/--obs-height` (view size, default 64×32), `--recode-down-h/-w`
and `--recode-rgb` (RECODE embedding resolution / color), `--recode-memory` (cluster
count), `--env-reward-coef` / `--exploration-coef` (mix in-game reward vs RECODE),
`--gif-every`, `--device`.

## Files

| File | Purpose |
|---|---|
| `env.py` | Builds the craftium globo `env_dir`; headless discrete-action gym env + obs preprocessing. |
| `models.py` | `Conv2dReservoir` (fixed ESN), `DeepConvDecoder` (trained readout), feedback→channel projection, RECODE downscaling. |
| `agent.py` | `GloboExplorerDQN`: reservoir front-end + conv-decoder Q-net + RECODE-over-images intrinsic reward + replay. |
| `train.py` | Training loop; saves GIF artifacts. |
| `recode/` | RECODE density estimator (from `/explorer`). |
| `craftium_globo_env/` | Generated craftium env dir (globo game + control mod + world). |
