# CLRS Generalist Replication — Continuation Notes

*Written 2026-07-16 by Claude Code after repo verification + timing probes on the actl pod. Read this before resuming.*

## Goal

Replicate "A Generalist Neural Algorithmic Learner" (Ibarz et al. 2022, arXiv:2209.11142) using this repo
(google-deepmind/clrs @ 9f77b53), then likely port the experiment to PyTorch.

## Verified findings about the repo

- `clrs/examples/run.py` IS the paper's training script. Pass all 30 algorithms via `--algorithms` to get the
  generalist: one shared processor (`triplet_gmpnn` default = paper's Triplet-GMPNN), per-algorithm
  encoders/decoders selected by `algorithm_index` (`clrs/_src/nets.py:71`). All paper improvements are default
  flags (Sinkhorn permutation decoding, pred_h-as-input, randomized pos, xavier-on-scalars, grad clip 1.0,
  train sizes 4–16, OOD test at 64).
- Model stack is **JAX/Haiku/Optax, not TensorFlow**. TF/tfds is only the test-split data loader
  (`dataset.py`, `probing.py`). Samplers/algorithms/eval are pure NumPy — reusable as-is in a PyTorch port.
- PyTorch port = ~3,000 lines (processors.py, nets.py, encoders/decoders/losses.py, slim baselines.py).
  `hk.scan` over hints → plain loop. Reference (deviating) port: SALSA-CLRS (github.com/jkminder/SALSA-CLRS).
  JAX buys 2–5x over eager PyTorch on this launch-bound workload via XLA fusion, but pays a ~50 min compile
  wall + recompile spikes that eager PyTorch doesn't. torch.compile should land near parity.

## Pod state (actl pod `clrs`, namespace `obelisk`, H100 80GB)

Everything below is already in place on the pod (survives container restarts; home = PVC):

- **Env**: `~/venv` (PVC) created from `/app/.venv/bin/python3`, with
  `~/venv/lib/python3.12/site-packages/app_venv.pth` → `/app/.venv/lib/python3.12/site-packages` to inherit the
  image's jax 0.10.1 + CUDA plugin. Container-local `/app/.venv` additions are WIPED on restart — never install
  there. Do not let pip put jax>0.10.1 in the venv (dm-haiku 0.0.16 breaks on jax 0.11: `jax.core.DropVar`).
- **Pins**: `protobuf~=6.31.1` (TF 2.21 needs >=6.31; tfds-nightly breaks on 7.x).
- **Dataset**: `$HOME/CLRS30` (86 MB tarball, already downloaded). Checkpoints (`best.pkl`) same path.
- **Compile cache**: `JAX_COMPILATION_CACHE_DIR=$HOME/jax_cache` (~85 MB, warm for: train shapes at length 4,
  val at 16, and ALL 30 algorithms' size-64 test shapes — including the OOM-causing ones).
- **Memory**: pod resized to **240Gi** (in-place; `actl pod resize clrs --memory NGi` works without restart).
- Timing logs on pod: `$HOME/timing_run.log` (52-step run), `$HOME/test_sweep.log` (+`test_sweep2.log`).
  Local analyzer: scratchpad `analyze_timing.py` (session-specific; rewrite from git if needed).

## Known bugs / workarounds (would also bite a PyTorch port — samplers are shared)

1. **Upstream bug**: default `--length_needle=-8` crashes val-sampler creation for string matchers
   (`samplers.py:675` `rng.choice(0)`: needle=8 can fill the size-16 val haystack; run.py 5/4-scales train
   lengths but `main()` pins val to 16). Workaround: `--length_needle=-7`.
2. **OOM history**: container OOM-killed at 64Gi (step 52, kmp_matcher train-shape compile) and at 160Gi
   (test sweep, lcs_length size-64 compile). Kill mechanism = XLA *transient* compile memory on top of
   accumulated executables. 240Gi + warm cache should hold, but see watchdog below. Check kills with:
   `kubectl -n obelisk get pod actl-ws-kmckee90-clrs-devspace-* -o jsonpath='{.status.containerStatuses[0].lastState}'`
   (exec sessions dying with 137 usually = container restarted underneath them).

## Measured timings (H100, all 30 algos, paper defaults)

- Startup (samplers+dataset+init): ~4 min
- First-time compile wall (30 algos × 5 train lengths): ~50 min (≈0 on reruns via cache)
- Steady-state 30-algo training step: **median 2.4 s** (~40–80 ms/algo; GPU ~95% idle);
  sporadic 20–60 s recompile spikes, decaying over time (steps 5–52 mean 5.5 s)
- Val sweep (30 algos): ~4 s cached (170 s first time) → 200 sweeps ≈ 15 min total
- Test sweep (30 algos @ size 64): ~6.5 min measured
- **Full 10,000-step estimate: ≈ 8–12 h. One H100 is enough; multi-GPU only buys parallel seeds.**
- Note: `run.py` has NO mid-training resume — a crash loses training progress (compile cache survives).

## To launch the real run

```bash
actl pod exec clrs -- bash -lc 'cd ~/workspace && \
  export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=1 \
         JAX_COMPILATION_CACHE_DIR=$HOME/jax_cache && \
  nohup $HOME/venv/bin/python -m clrs.examples.run \
    --algorithms=articulation_points,activity_selector,bellman_ford,bfs,binary_search,bridges,bubble_sort,dag_shortest_paths,dfs,dijkstra,find_maximum_subarray_kadane,floyd_warshall,graham_scan,heapsort,insertion_sort,jarvis_march,kmp_matcher,lcs_length,matrix_chain_order,minimum,mst_kruskal,mst_prim,naive_string_matcher,optimal_bst,quickselect,quicksort,segments_intersect,strongly_connected_components,task_scheduling,topological_sort \
    --length_needle=-7 --train_steps=10000 \
    --dataset_path=$HOME/CLRS30 --checkpoint_path=$HOME/CLRS30 \
    > $HOME/full_run.log 2>&1 & echo $!'
```

Arm a memory watchdog alongside (poll `/sys/fs/cgroup/memory.current` in the pod every ~60 s; if it exceeds
~80% of `memory.max`, `actl pod resize clrs --memory <bigger>` in place — node has ~1 TB). Grep progress with
`grep "step N " full_run.log`; per-algo test results print at the end. Results are single-seed: expect the
paper's qualitative picture, with per-algorithm variance on hard OOD tasks.

## Decisions still open

- Fidelity choices for full replication: paper's multi-task learner used chunked training
  (`--chunked_training`, untested here) — decide whether to match it or run the default non-chunked setup.
- PyTorch port: green light was "feasible"; not started.
