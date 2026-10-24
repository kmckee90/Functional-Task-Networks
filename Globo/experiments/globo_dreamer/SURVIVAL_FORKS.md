# Survival forks — 2026-10-23

## Expanded reward sweep

The six remaining explorer runs were saved and stopped, then forked into new
`survival_sweep_<source-name>` directories. Their original checkpoint directories
remain intact. The two first survival runs continue unchanged. All eight use
the survival reward. This is a deliberately scattered joint-parameter pilot,
not an isolated-factor experiment.

| Source / agent seed | Gamma (both streams) | Updates | LR multiplier | Ordinary novelty |
|---|---:|---:|---:|---:|
| γ=.90 explorer r1 / 1901 | .999 | 200 | .25 | 1/8 |
| γ=.90 explorer r2 / 1902 | .999 | 200 | .25 | 1/16 |
| γ=.90 explorer r3 / 1903 | .995 | 200 | 1 | 1/4 |
| γ=.90 explorer r4 / 1904 | .999 | 50 | .25 | 1/8 |
| γ=.99 explorer r1 / 1901 | .995 | 50 | .25 | 1/16 |
| γ=.99 explorer r2 / 1902 | .999 | 200 | 1 | 1/4 |
| First survival r1 / 1901, unchanged | .99 | 50 | 1 | 1/4 |
| First survival r2 / 1902, unchanged | .99 | 50 | 1 | 1/4 |

World seed remains 19. Base LR is 0.0001899492997635358; quarter LR is
0.00004748732494088395. Explicit LR is applied after loading the optimizer,
retaining moments while overriding the checkpoint's LR. Runtime settings and
buffer sizes are saved in each new run's `survival_settings.json`.
Goal reward stays unscaled, as in the first survival runs. The death penalty
matches that unscaled goal reward. Only ordinary novelty changes to 1/8 or 1/16.

`expand_survival.py pause --pod <pod>` saves/stops only explorer sources and
retains existing survival workers. `fork` refuses live sources or existing
destination directories, copies checkpoints/mutable state, links immutable
replay chunks, and launches separate workers with separate future checkpoints.

## Original two-run pilot

On globo1, stop and retain the two gamma .99 / 200-update runs. GPUs 2 and 3
instead run `lifetime2_survival_r1_s19` and `lifetime2_survival_r2_s19`, forked
from the matching gamma .99 / 50-update runs. World seed 19; agent seeds 1901
and 1902. Originals continue on GPUs 0 and 1. All retain whole-life recording.

Each fork retains weights, optimizer, FTN mask archive, RECODE density, replay,
frontier buffer, schedules, training step and accumulated discoveries. Its
`fork.json` records the source checkpoint and step. Replay chunks are immutable
hard links with independent manifests and garbage collection; checkpoints and
mutable environment state are copied. Source and fork save independently.

Only the survival forks enable `GLOBO_SURVIVAL=1`:

- Compute the existing combined online-density/offline-density reward at replay
  time. Keep the sparse goal/frontier target unchanged; multiply other rewards
  in that batch by .25.
- Give the death transition a negative environment-stream target equal to the
  magnitude of that batch's original boosted goal reward. This is a dynamic,
  batch-relative scale, not the frontier multiplier used as an absolute cost.
- Retain gamma .99 and 50 offline updates per 50 interactions. Acting maximizes
  the sum of the novelty and environment Q streams, as before.
- Mark death separately using replay `env_r=-1` on the transition from the last
  executed action to the respawn observation. Keep terminal/boundary false,
  retaining cross-life bootstrapping, reservoir and feedback semantics.
- Prior continuing-life replay did not retain death labels. Its exploration
  data remains usable, but death penalties start with newly observed deaths.

The monitor labels the new condition **Survival**. Its curves start at the fork
step; inherited achievements and coverage are not represented as fresh
discoveries. Completed-life statistics start with the fork's new lives.
Stopped 200-update runs remain stored but leave the active comparison.

Validation: `python -m globo_dreamer.test_survival_rewards` on the pod checks
reward ratios, goal preservation, equal-scale death penalty, bootstrapping,
and exactly one replay death label across death and respawn.
