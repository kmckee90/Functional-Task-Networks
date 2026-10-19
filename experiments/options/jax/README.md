# JAX port of `train_options_2.py`

A from-scratch JAX reimplementation of the **sequential-mask** options experiment
(`options/train_options_2.py`): environment, model, RECODE clustering, DDPG, mask
search and all four visualizations. It exists alongside the torch version (which is
untouched) — same experiment, same plots, two backends.

```bash
cd options/jax
uv run --project ../.. python train_options_2.py            # full
uv run --project ../.. python train_options_2.py --smoke    # tiny, fast sanity run
uv run --project ../.. python train_options_2.py --workers 16
```

Outputs go to `options/jax/artifacts/`: `centroids.png`, `graph.png`,
`exploration.png`, `masks.png` (identical styling to the torch trainer) plus
`checkpoint.pkl` / `graph.pkl` (pickled host arrays; `.pkl` so they don't clash
with the torch `.pt`).

## What's where

| file | role |
|------|------|
| `config.py` | dataclasses, mirror of `options_config.py` (same defaults) |
| `color_world.py` | ColorWorld: JAX-pure dynamics (`color_at`) + identical numpy/matplotlib render. World geometry generated with `numpy.default_rng(world_seed)` exactly as the torch env, so the same `world_seed` gives the same patchwork |
| `collect.py` | **JIT loop #1** — `vmap(scan)` parallel random-walk collection |
| `recode.py` | RECODE online clustering as one `lax.scan` (vacant-insert → merge / evict-redistribute) |
| `ftn.py` | ParallelMLP / actor / critic forwards, KWTA, SmoothKWTA, AnnealedSTE mask search |
| `agent.py` | params, manual Adam, host-side FPS `CentroidRegistry`, **JIT loop #2** — per-option DDPG `scan` with sibling consolidation, and the jitted reconfig |
| `train_options_2.py` | the driver (mirrors the torch one) |
| `viz.py` | the four PNG renderers, ported verbatim in styling |
| *(reused)* `../centroid_graph.py` | framework-neutral behavioral graph + Dijkstra |

## Design / performance

The two heavy phases are each a single fused XLA program — the Python level only
dispatches a handful of compiled calls per round:

- **Collection** is `vmap` over W worker chunks × `lax.scan` over steps. No
  multiprocessing, no torch buffer copy — the concatenated arrays *are* the batch.
- **Option training** keeps the sequential structure (reconfig→train, one option at
  a time, so each capitalizes on the shared backbone the earlier ones moved). Each
  option's reconfig (`reconfig_steps` AnnealedSTE iters) and DDPG (`option_train_steps`
  iters, with up to 3 mask-overlapping sibling terms) are fused `scan`s.

**Backend.** GPU is enabled — `jax.default_backend()` is `gpu` and the code is
device-agnostic. Full-size `H=1024` reconfig + 1000 DDPG steps is ~2.6 s/option on
an H100, so a full run is well under a minute (vs ~tens of seconds/option on CPU).

GPU was turned on *surgically* so torch's CUDA setup is untouched — only the two
plugin packages, no-deps, reusing torch's existing nvidia cu12 wheels:

```bash
uv pip install --no-deps "jax-cuda12-plugin==0.4.36" "jax-cuda12-pjrt==0.4.36"
```

The plugin/pjrt version must match `jaxlib` exactly (0.4.36 here). Installing the
`jax[cuda12]` extra instead would churn nvidia wheels in the shared uv cache and can
poison torch's cuDNN — see the `shared-uv-cache-cudnn-poison` note; the no-deps
plugin install avoids that. A benign `ptxas ... miscompile` advisory prints at
startup. To force CPU: `JAX_PLATFORMS=cpu`.

## Faithfulness notes

- RNG and float math differ from torch, so plots are not bit-identical — but the
  pipeline, config values and plot styling are the same. In ColorWorld there are
  only `n_cells` distinct observations, so FPS reliably recovers one centroid per
  cell regardless of backend.
- `smoke_config` sets `lateral_square_dim=2` (valid for `H=32`). The torch smoke
  leaves it at 32 and so actually crashes `LateralSmoother` — there was no working
  reference to match, so the JAX smoke is made runnable. The full config (`H=1024`,
  `square_dim=32`) is unchanged.
</content>
