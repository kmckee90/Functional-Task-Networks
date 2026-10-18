# Phase 2: fresh exploration comparison

All runs start from randomly initialized models and empty replay/density/frontier buffers. No Phase 1 checkpoint is an initializer. Runs have no step limit; the supervisor saves checkpoints every 600 seconds, resumes interrupted runs, and detects stale progress. The eight runs use seeds 19 and 101 and their existing matched spawn locations.

| Model | Pod | GPUs (seed 19, seed 101) |
|---|---|---|
| Dreamer | globo-dreamer | 1, 4 |
| Plan2Explore | globo-dreamer | 5, 6 |
| M2E | globo1 | 0, 1 |
| M2E+convFTN | globo2 | 0, 1 |

The MLP-FTN pair is deferred at the user's request because globo3 could not be scheduled. Phase 1 retains flat seed 19 (GPU 0), conv seeds 19/101 (GPUs 2/3), and waypoint seed 101 (GPU 7). Other Phase 1 workers are paused with checkpoints and results retained.

## Implementations

These retain the existing method configurations, not a new parameter-matched architecture experiment. Dreamer uses the scaled RSSM (hidden 1024, stochastic 128×32, encoder channels 32, decoder channels 64). Plan2Explore uses size12m, decoder channels 16, five disagreement predictors and reward scale 10000. Both use training ratio 256 and 100k Dreamer replay. Their action objectives are intrinsic, without survival rewards.

M2E uses config 24, tuned frozen 128-channel reservoir (20% recurrent connections, radius .99), float16 replay of 2M transitions, 500k frontier transitions and 100k RECODE centroids. ConvFTN replaces only the trainable decoder, retaining the matched active parameter budget and one sheet of independent eight-layer, width-eight neurons; K=296 of 2368, Bayesian replay and two reconfiguration steps per batch. No Dreamer component is used in these two methods. Existing frontier rotation (top five distinct alternatives, increasing switch hazard and 1000-step maximum when alternatives exist) is retained.

Every run uses the same 256×256 game rendering mode and downsamples for its agent. All have achievement tracking from the beginning. A narrow animal-attack nil-height fallback prevents a known game crash; it changes neither learning rewards nor agent inputs.

## Storage and isolation

Code is edited locally and synced by actl. Runs/checkpoints/replay stay outside the synced workspace under `/home/dev/globo_dreamer_state/runs/phase2_*` on their own pods. Each pod owns its own supervisor, cache, environment directories, run directories, and publisher. New pods also need system packages `libx11-6 libxext6 libegl1 libgles2 libgl1-mesa-dri libgl1 libopengl0` for headless software rendering. Shared dependencies are read-only by convention; Python bytecode writes are disabled.

`/mnt/obelisk-shared/kmckee90/globo_phase2` contains only published monitor data and terrain cache, not shared agent replay/checkpoints. Each publisher writes disjoint globally named run directories and its own pod manifest. JSON and videos are atomically replaced. Clips overwrite rather than accumulating. Compact progress/event histories remain available for long-term comparison.

## Monitor and terrain

Local monitor: http://127.0.0.1:8765/phase2/
Phase 1: http://127.0.0.1:8765/continuous/

The pod-side aggregator runs independently of the laptop. The local HTTP bridge reads its output; sleeping the laptop interrupts viewing but not training, publication, terrain mapping, or monitor generation. Rehost using `presentation/site/rehost_local.sh`.

Per-seed top charts show spatial discovery and cumulative checked achievement boxes. Repeating an achievement never raises that curve. Columns list gained categories and individual discoveries. Per-run video, status, stagnation history, all detailed metrics, and FTN mask pop-outs are retained. Clip length is shared across Phase 2, separately from Phase 1.

Visitation uses the tracker's 4×4 metre horizontal cells, counted once per environment observation. All runs on a seed share a bounding rectangle from the union of visited positions plus margin, rounded to 64m tiles, expanding without shrinking. Same terrain, axes, pixel scale and log visitation scale are used for every model on that seed. Recent paths break at life boundaries. White marks current position/heading; blue marks initial spawn.

Independent CPU-only headless game servers generate the original seeded terrain. They have no player and are never connected to a learner. Tiles sample the top visible solid/liquid/canopy surface every 2m. The background is a reference surface, not a live representation of agent modifications or underground paths. Unknown terrain is visibly unfilled until generated. Cached tiles persist as the view expands. Neither terrain nor positional instrumentation enters observations, replay, policy decisions, or rewards.

## Operations (on the relevant pod)

Use Python `/home/dev/globo_dreamer_state/venv/bin/python` with `PYTHONPATH=/home/dev/globo_dreamer_state/packages:/home/dev/globo_dreamer_state/vendor/dreamerv3` and `PYTHONDONTWRITEBYTECODE=1`.

- `-m globo_dreamer.phase2 install`: attach the shared dependency runtime on a new pod.
- `-m globo_dreamer.phase2 launch --pod globo1` (or globo2/globo-dreamer): idempotently register intended runs and start the pod supervisor/publisher.
- `-m globo_dreamer.phase2_monitor`: aggregate monitor on globo-dreamer (singleton lock).
- `-m globo_dreamer.terrain_worker`: terrain servers on globo-dreamer.
- `-m globo_dreamer.test_phase2`: CPU bookkeeping checks.

Do not terminate the continuous supervisor with SIGTERM when retaining its workers: its shutdown handler stops them. Deployment utility handles safe supervisor replacement explicitly.
