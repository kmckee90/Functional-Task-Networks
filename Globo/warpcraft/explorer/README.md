# globo_explorer_gpu

A GPU port of [`../globo_explorer`](../globo_explorer): the same **conv2d-reservoir
RECODE explorer**, re-implemented in **JAX/Flax** and driving the **warpcraft**
GPU globo engine instead of craftium/Luanti. Everything — env, model inference,
RECODE, and the DQN training loop — runs on the GPU.

## Design

This simulates a single **real-world robot-env interaction lifetime**, so the env
runs at `num_worlds=1` (one continuous life, auto-reset on death/timeout) — *not*
thousands of parallel worlds. The GPU win comes from keeping the whole
encode→act→learn path on-device and fused, not from batching lives.

- **Frozen conv2d echo-state reservoir** encodes the first-person RGB obs (+ a
  feedback channel of prev action/reward/binned-novelty). Only the decoder trains.
- **Branching dual-stream DQN** — an intrinsic stream (RECODE novelty, `gamma`)
  and a survival stream (env reward − death penalty, `gamma_env`), acting on the
  combined greedy action, with independent Polyak target rates.
- **RECODE** intrinsic novelty (count-weighted kNN) over the downscaled grayscale
  image, with online + offline terms, a goal buffer, and prioritized trajectory
  replay pushing a sparse frontier bonus backward.
- **Action** uses warpcraft's native `[1,4]` channels; the branching heads default
  to LOCOMOTION(10) + LOOK(5) (concurrent move+look — the closest analog of the
  craftium `factored` control).

`models.py` / `recode.py` / `agent.py` are faithful JAX ports of the originals;
`env.py` is the warpcraft adapter (zero-copy torch↔JAX via Warp dlpack).

## Speed

The per-step hot loop is one jitted `perceive` (encode+act+embed+novelty) with
**no host syncs**, then the one foreign GPU call (`warpcraft` step). ~29 steps/s
on an H100 — faster than the craftium version despite the full survival sim.
(The env step can't go inside `jax.jit` — it's opaque torch/Warp — so it's the
per-step floor.)

## Setup (on an actl GPU pod)

```sh
bash setup_pod.sh          # git-safe-dir fix + uv sync (warpcraft@git, jax[cuda12], flax, optax) + Warp prewarm
```
Requires GitHub auth on the pod (warpcraft is private) and a CUDA GPU
(`exec_mode=1`). Warp JIT-compiles kernels on first run (slow once, then cached).

## Run

```sh
uv run python train.py --config default.yaml     # single-lifetime training
uv run python train.py --config smoke.yaml        # tiny end-to-end smoke
uv run python train.py --config bench.yaml        # steady-state speed benchmark

# Human/agent viewer (forward the port first, e.g. `actl pod exec --port 8000`):
uv run python play.py                              # free-roam: WASD move, arrows look, space jump
uv run python play.py --agent artifacts/run_ckpt.pkl   # watch a trained checkpoint
```

`default.yaml` is the single source of truth; a `--config` overlay only needs the
keys it changes. Artifacts (GIF peeks + checkpoints) land in `artifacts/`.

## Status / notes

- End-to-end verified on an H100 (`env`, full train loop, `play` internals).
- Loss stability follows the same deadly-triad caveats as the craftium version
  (see `../globo_explorer/CONTINUATION.md`): high `gamma_env` + hard intrinsic
  target `tau=1.0` can diverge with aggressive settings; tune `target_update_tau`,
  `recode_knn_k`, or the update ratio if so.
- Differences from the craftium original are noted inline: reservoir weights use
  JAX RNG (same architecture/distribution, not identical values); RECODE's
  eviction `d2_nearest` is recomputed fresh (cleaner than the original's stale
  incremental array); obs downscale uses bilinear.
