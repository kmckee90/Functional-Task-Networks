# Phase 5: twelve Plan2Explore replicates

October 16 allocation update: runs 7–12 are gracefully paused with final
checkpoints retained. Runs 1–6 continue unchanged on globo-dreamer GPUs 0–5.
The freed six GPUs now host Phase 4 M2E; globo3/globo4 host Phase 4 FTN-M2E.
Paused Phase 5 runs remain visible as saved results on the Phase 5 monitor.

All twelve runs resumed October 16, 2026 from their existing checkpoints
(approximately 59k–103k observations), after gracefully stopping Phase 4.
They use the same twelve GPUs across globo-dreamer, globo1 and globo2, with
no step limit. Replay and learned state are retained; each resumed worker
starts a new game instance. The live monitor is /phase5/.

World seed 19, spawn `12.0,39.1,120.0`; independent learner seeds 1901–1912.
Twelve fresh, unlimited runs, one per GPU: runs 1–8 on globo-dreamer,
9–10 on globo1, 11–12 on globo2. No imported checkpoints or replay.

Uses the existing Plan2Explore baseline: Dreamer size12m, decoder channels 16,
five embedding-disagreement predictors, intrinsic reward scale 10000,
train ratio 256, Dreamer replay 100k. All actions from Plan2Explore, no M2E,
FTN, or survival reward. Same 256px render/downsample mode and achievement
instrumentation as the preceding experiments.

Monitor `/phase5/`: selectable run curves and recomputed mean/range, spatial
progress, unique achievements, twelve-column achievement table, per-run videos,
shared terrain heatmaps, survival, stagnation logs, and world-model video.
The shared-step summary lists twelve runs rather than six cross-method pairs.
FTN mask panes and FTN-only event captures do not apply to Plan2Explore.

Phase 3 and Phase 4 original checkpoints/replay remain in each pod's persistent
`/home/dev/globo_dreamer_state/runs/phase{3,4}_*` directories. Separate snapshots
of results, checkpoints, event clips and source-at-archive are stored under
`/mnt/obelisk-shared/kmckee90/globo_archives/phase{3,4}/POD`, with SHA-256 manifests.
Replay is retained in the original run directory, not duplicated in snapshots.
Frozen monitors remain at `/phase3/` and `/phase4/`.

Phase 5 runtime data: `/home/dev/globo_dreamer_state/runs/phase5_*` per pod.
Published data: `/mnt/obelisk-shared/kmckee90/globo_phase5`.
