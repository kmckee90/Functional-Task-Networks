# ewc — FTN with mask-gated EWC instead of replay

Fork of `base_rf/` in which **replay is removed entirely** and replaced by
**Elastic Weight Consolidation gated by mask overlap** (`ewc.py`). The goal:
handle mask overlap between tasks — optimal task recall *and* forward
transfer in the xfer matrix — without rehearsing a single stored sample.

## Why EWC fits FTN structurally

The backbone is a `ParallelMLP`: every hidden unit owns its own parameter
slice (per-layer weights are `(hidden, in, out)`, biases `(hidden, out)`, and
each readout column belongs to one unit). Under a binary mask, task gradients
reach **only units with m = 1**. So the *only* interference channel between
tasks is the set of units their masks share — exactly the population the old
replay rule targeted ("replay only memories whose masks overlap the current
one"). EWC here is the direct translation of that rule from data-space into
weight-space:

| Replay (base_rf) | Mask-gated EWC (here) |
|---|---|
| Store first batches + masks per task | Store Fisher + anchor θ\* + usage mask per task, at end of task |
| Each step: find stored batches with `0.2 < corr(m_stored, m) ≤ 0.8` | Each step: find stored tasks with the same overlap gate |
| Accumulate task-loss gradients on those batches under their stored masks | Accumulate quadratic anchor gradients `λ·F·(θ−θ*)`, **gated per unit to the mask intersection** |

The two gates that keep the population from being depleted:

1. **Overlap gate (task level)** — same as replay. Stored tasks with
   correlation ≤ `EWC_OVERLAP_MIN` face no interference (their units aren't
   being trained) and are skipped; correlation > `EWC_OVERLAP_MAX` means the
   current solution basically *is* that task, so training it is rehearsal —
   also skipped, so a task never fights its own refinement.
2. **Unit gate (per-unit)** — the penalty is multiplied by
   `usage_t · m` (stored soft usage mask × current mask), so only units in
   the *intersection* carry any constraint. Units of an old task the current
   task doesn't touch need no protection; units of the current task no old
   task used stay fully plastic. Constraint mass exists only where
   interference can actually occur, and Fisher weighting keeps even shared
   units soft along directions that didn't matter to the old task.

Consolidation (`MaskedEWC.consolidate`) runs at the end of each task:
`EWC_FISHER_BATCHES` fresh batches, each **reconfigured on first** (mirroring
how the task will be retrieved later), empirical Fisher diagonal of the task
loss with dropout off, usage mask = mean of the reconfigured masks. The
Fisher is normalized to mean 1 over the used units' entries so `EWC_LAMBDA`
means the same thing for every task. Re-consolidating a task replaces its
record. The runner logs `population coverage` — the fraction of units under
any consolidation pressure — as the depletion monitor.

## Layout (vs base_rf)

| File | Change |
|---|---|
| `ewc.py` | **new** — `MaskedEWC`: consolidate / penalty / coverage |
| `run_experiments.py` | `ReplayBuffer` + `replay_step` removed; `ewc_bank.penalty_backward(ftn)` sits where `replay_step` sat; end-of-task consolidation; `--lam` CLI override |
| `config.py` | replay params removed; `EWC_LAMBDA`, `EWC_FISHER_BATCHES`, `EWC_OVERLAP_MIN/MAX` added |
| `ftn.py`, `tasks.py`, `viz.py`, `utils.py` | unchanged |

## Run

```bash
python run_experiments.py                          # Default config (EWC on)
python run_experiments.py --config Smoke           # tiny sanity check
python run_experiments.py --lam 0 --name noewc     # pure-masking baseline
python run_experiments.py --lam 300 --name lam300  # strength sweep
```

Outputs land in `images/`; the headline evaluation is
`images/xfer_p1_mtl*.png` (rows = after training task, cols = evaluated task;
lower triangle = retention, upper = forward transfer).

## Results (Default config, seed 123, 4 tasks)

Final-row mean accuracy across the λ sweep, vs the two references:

| setup | final mean acc | task-0 retention trajectory | notes |
|---|---|---|---|
| replay (base_rf) | 0.858 | 0.89 → 0.82 → 0.83 → 0.83 | reference |
| λ=0 (masking only) | 0.838 | 0.91 → 0.80 → **0.68** → 0.85 | dips at mask overlap |
| λ=0.3 | 0.871 | 0.91 → 0.87 → 0.86 → 0.89 | transition |
| **λ=3 (default)** | **0.881** | **0.91 → 0.90 → 0.90 → 0.91** | flat retention |
| λ=10 … 1000, mean-reduce variants | 0.876–0.880 | flat | plateau |

The λ=3 xfer matrix dominates replay in retention (all lower-triangle cells
0.86–0.91) and forward transfer (upper triangle 0.75–0.83 vs replay's
0.68–0.81); its one concession is late-task acquisition (task-3 diagonal
0.86 vs replay's 0.90), because high-Fisher shared entries genuinely encode
the old tasks' functions and only data (replay's joint gradient) could move
them cooperatively.

Two empirical notes for tuning:
- Retention saturates for λ ≥ 3 and the matrix is *insensitive to λ over
  2.5 orders of magnitude*: through Adam, whichever entries have
  λ·F·Δθ above the task-gradient scale are pinned regardless of the exact λ,
  so λ selects a pinning threshold along the (heavy-tailed) Fisher spectrum
  rather than a smooth strength. λ=3 sits at the least-constraining end of
  the plateau — flat retention with minimal population depletion (the
  `population coverage` log line tracks how much of the population carries
  any consolidation pressure; it stays ≈ the union of task masks, ~0.4).
- `EWC_REDUCE="mean"` (averaging over gated records instead of summing)
  behaves like a ~n×-smaller λ; on the plateau it makes no difference.
