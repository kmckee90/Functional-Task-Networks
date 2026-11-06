# Benchmark storage

The three workspaces `bm1`, `bm2`, and `bm3` sync only this directory to
`/home/dev/workspace`. All have the shared PVC mounted at
`/mnt/obelisk-shared`.

Before running any download, installation, or experiment on a pod:

```bash
source /home/dev/workspace/remote_env.sh
```

All bulk paths are outside the sync tree:

| Content | Remote path |
| --- | --- |
| Root | `/mnt/obelisk-shared/benchmarking/kmckee90/neurips_ftn_2026` |
| Datasets | `$BM_ROOT/data` |
| Checkpoints | `$BM_ROOT/checkpoints` |
| Raw results, logs, run manifests | `$BM_ROOT/runs` |
| Model, package, framework caches | `$BM_ROOT/cache` |
| Socket and temporary scratch | `/tmp/bm` on the remote pod (outside sync) |

Use separate method/benchmark/seed directories and claim jobs atomically to
avoid concurrent writers. Environments belong under `$BM_ROOT/envs`, outside
sync. Keep full datasets and checkpoints remote. Copy back only small summary
tables, manifests, plots, and the final report, deliberately.

`.actlignore` also excludes common dataset/checkpoint/cache/output names and
large binary extensions as a second layer of protection. Do not create symlinks
to remote bulk directories inside the sync tree.

Each workspace also has its own 500 GiB home PVC; these are not the shared
benchmark data store and do not sync except for `/home/dev/workspace`.

## Local result snapshot

The 2026-10-31 recovery from `bm3` is in the Git-ignored directory
`artifacts/neurips_ftn_2026/`. It preserves the remote layout for `runs/`,
`manifests/`, `reevaluations/`, `reports/`, and `validation/`: raw results,
progress, failure records, native logs, configurations, split indices, timing,
and reports. Smoke tests and incomplete runs are included; presence in this
snapshot does not imply a completed experiment.

The snapshot contains 274 files (95,485,599 bytes), with remote SHA-256 hashes
in `artifacts/neurips_ftn_2026/SHA256SUMS`. Verify from that directory with
`sha256sum -c SHA256SUMS`. Datasets, checkpoints, caches, environments,
serialized model masks (`block*_mask.pt`), and transient lock/PID files were
not downloaded. Remote originals remain intact.

Experiment source is visible to Git; result snapshots and generated reports
remain ignored. `artifacts/` is also excluded from actl sync.
