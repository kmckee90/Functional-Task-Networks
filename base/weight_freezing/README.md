# weight_freezing — FTN with usage-density weight freezing instead of replay

Fork of `base_rf/` in which replay is removed entirely and replaced by a
single rule: **only train the least-globally-used neurons in the current
mask** (`freezing.py`). No stored data, no stored masks, no per-task records,
no task ids, no boundary knowledge — the only state is one `(hidden_size,)`
vector of counts.

The core model and training loop are byte-equivalent to `base_rf/` with
replay disabled (`ftn.py`, `tasks.py`, `viz.py`, `utils.py` are identical
copies; `run_experiments.py` is `base_rf`'s with the replay buffer deleted
and the freezer inserted at the same point in the step).

## Mechanism

- **Density**: `counts[j] += m[j]` after every training batch — *batch
  involvement*, counted whether or not the unit's weights were frozen that
  step. That choice is load-bearing: a reused (frozen) senior unit keeps pace
  with the batch counter, so later-recruited units can never overtake it —
  the seniority ordering that encodes "consolidated memory" is permanent.
  No decay/EMA: decay would assume a task-recurrence timescale, which is
  a-priori task knowledge.
- **Freeze rule** (per step, within the current mask only): sort the active
  units' counts; find the *lowest adjacent gap* larger than `FREEZE_MARGIN`
  (raw batches); freeze every active unit above it — they are the discretely
  senior cohort that carries another solution's training. Gradient slices of
  frozen units (per-unit MLP weights/biases + readout column) are zeroed
  before the optimizer step.
- **Emergent replay-gate behavior**: a uniformly fresh mask (no overlap) has
  no significant gap → everything trains (old gate's `corr ≤ 0.2`: nothing to
  protect). A uniformly senior mask (pure revisit) has no gap either →
  everything trains, i.e. rehearsal/refinement (`corr > 0.8`). Only mixed
  masks freeze their seniors. Both edges of the hand-tuned replay band fall
  out of the relative comparison.
- **Guards**: `FREEZE_MIN_PLASTIC` — if the junior cohort is a sliver of the
  mask (e.g. one freshly swapped-in unit next to a settled population,
  mid-task churn), the gap is noise, not a task boundary; nothing freezes.
  Doubles as the depletion guard: at least this fraction of the working mask
  always stays trainable.

Why unit-level hard freezing is enough: the EWC study archived in
`archive_ewc/` found the xfer matrix insensitive to the penalty strength λ
across 2.5 orders of magnitude — the soft, Fisher-weighted anchor was
behaviorally equivalent to freezing whole units, because within one per-unit
MLP all gradients share the unit-level scale factor. Freezing also composes
safely with FTN's reconfiguration: if a frozen unit fits a new task poorly,
kWTA routes around it instead of paying a permanent tax.

## Hyperparameters

| name | default | meaning |
|---|---|---|
| `FREEZE_MARGIN` | 250 | seniority gap (batches of involvement) that counts as a discrete split; must exceed within-task mask-churn spread and sit below a full training block |
| `FREEZE_MIN_PLASTIC` | 0.1 | min fraction of the mask that must remain trainable |

`FREEZE_MARGIN` is the one real hyperparameter — a timescale ("how much more
veteran than the newest recruits before you're read-only"), not a task label.

## Run

```bash
python run_experiments.py                       # Default config (freezing on)
python run_experiments.py --config Smoke        # tiny sanity check
python run_experiments.py --margin 0 --name off # baseline (= base_rf, no replay)
python run_experiments.py --margin 500 --name m500
```

Outputs land in `images/`; the headline evaluation is
`images/xfer_p1_mtl*.png` (rows = after training task, cols = evaluated task;
lower triangle = retention, upper = forward transfer).

`archive_ewc/` preserves the preceding mask-gated-EWC iteration (code,
README, run logs, result images) for reference.

## Results (Default config, seed 123, 4 tasks)

| setup | final-row mean acc | task-0 retention | notes |
|---|---|---|---|
| replay (base_rf) | 0.858 | 0.89 → 0.82 → 0.83 → 0.83 | stores data + masks |
| baseline (margin off) | 0.833 | 0.91 → 0.80 → **0.72** → 0.82 | dips at mask overlap |
| EWC λ=3 (archive_ewc) | 0.881 | flat | O(tasks × params) state, task boundaries |
| **freezing (margin 250, lag 1000)** | **0.872** | 0.91 → 0.90 → 0.91 → 0.89 | **two H-vectors, no task knowledge** |
| freezing margin 150 | 0.871 | flat, task 0 self-freezes a bit early | |
| freezing margin 400 | 0.871 | mild dips (0.83) — protection arrives late | |

Forward transfer (upper triangle) 0.75–0.83 — the best of all variants
(replay 0.68–0.81, EWC 0.75–0.83). Unlike EWC, late tasks keep learning
(task 3's training loss descends instead of flatlining): hard-freezing the
senior cohort leaves fresh units fully plastic instead of dragging the whole
mask with a soft penalty.

Freeze dynamics from the logs: task 0 trains essentially unfrozen
(mean_frozen 9.6/step, mild end-of-block self-consolidation), later tasks
freeze their senior cohorts (~100–200 of the 256 mask units).

Empirical notes:
- **Batch involvement is a continuum, not bimodal.** Per-batch
  reconfiguration churns masks while a task settles (count quantiles
  0/5/66/996 after one task, coverage 0.72), so the original gap-detection
  rule never fired; a plain within-mask seniority comparison on
  *instantaneous* counts cannot distinguish the current task's core from a
  past task's core. The lagged counts are what make the rule work.
- **Margin/lag interplay**: seniors arrive at the next task with a lagged
  lead of ≈ 0.37·lag; the margin must sit below that so protection is
  immediate at transitions (margin 400 > 368 → late onset → dips), and above
  the within-task churn spread (~100). 250 with lag 1000 is comfortably in
  the window.
