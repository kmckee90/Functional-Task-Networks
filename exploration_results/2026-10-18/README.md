# Exploration monitor results — October 18, 2026

Downloaded from the shared results storage through the globo1 pod. Approximately 62 MB on disk. No checkpoints, replay buffers, training datasets, worlds, terrain data, images, or videos are included.

- `runs.csv`: compact index of the 44 published Phase 2–5 runs, their last recorded steps, spatial coverage, and result directories.
- `phase2/` through `phase5/`: published run configurations, status, achievement records, progress histories, survival summaries, events, and monitor HTML snapshots.
- `historical/`: retained Phase 3 and Phase 4 archive snapshots. These are historical records, not additional independent trials.
- `globo1/`: local supervisor state and any available continuous-monitor HTML.
- `manifest.json`: original source paths, file sizes, checksums, exclusions, and unavailable archives.
- `SHA256SUMS`: checksums for every downloaded result file, verified after extraction.

Each run's `live/history.jsonl` supplies the chart histories; `live/events.jsonl` records events; `environment/achievements.json` and `status.json` retain achievement and final-state information. Some runs also include compact visitation summaries. These are numerical result records, not terrain or training data.

Monitor HTML is retained as a snapshot. Embedded charts and tables remain available, but media and server-dependent controls are intentionally unavailable. Raw result files are the durable source for replotting. A recorded status is a snapshot, not proof that a worker is currently running.

## Coverage limitation

The full original Phase 0 and Phase 1 archives were not found on globo1 or its shared results storage. The original globo-dreamer pod could not execute commands at export time. This archive therefore does not claim to contain those full histories. Existing proposal figures elsewhere in the repository remain untouched.

The remote files were not deleted or modified. The temporary local transfer archive was removed after extraction to avoid duplicate storage.
