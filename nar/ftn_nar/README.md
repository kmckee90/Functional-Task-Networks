# FTN Neural Algorithmic Reasoning (layered-DAG dynamic programs)

Implementation of `ftn_neural_algorithmic_reasoning_experiment_design.txt`:
support-conditioned sparse-operator (FTN) continual learning and held-out
compositional induction on the 2x4 factorial family of graph dynamic programs
(MIN/MAX reducer x ADD/SUB/MAX_EDGE/MIN_EDGE extender).

## Layout

- `src/ftn_nar/` — flat package (matches sibling experiment style):
  `algorithms.py`/`reference_b.py` (two independent exact references),
  `generator.py` (layered DAG distributions), `batch.py`, `support.py`
  (identifiable support selection), `model.py` (atom banks + recurrent
  executor + oracle/dense variants), `masks.py`, `configurer.py` (annealed
  STE top-k inner loop), `losses.py`, `replay.py` (overlap-filtered
  reservoir), `trainer.py` (block-sequential continual training, all
  methods), `evaluator.py`, `evalrun.py`, `exhaustive.py` (tiny-bank
  enumeration), `aggregate.py`, `cli.py`.
- `tests/` — required tests (reference cross-check on 10k graphs, STE/topk,
  freeze, resume, overfit, identifiability, replay filter...).
- `scripts/launch_matrix.py` — fold x seed x method scheduler for the 4xH100 pod.

## Running (on the actl pod `ftn-nar`)

```bash
uv sync
uv run pytest -x -q                       # full test suite
uv run python -m ftn_nar.cli selftest     # quick startup self-test
# single run:
uv run python -m ftn_nar.cli train --method FTN_FULL --held-out MIN_ADD --seed 0 \
    --out /home/dev/runs
# matrix:
uv run python scripts/launch_matrix.py --methods FTN_FULL ORACLE_COMPOSITION \
    --seeds 0 1 2 --folds all --out /home/dev/runs
uv run python -m ftn_nar.cli aggregate --runs-dir /home/dev/runs --out outputs/reports
```

Methods: FTN_FULL, ORACLE_COMPOSITION, FTN_RETRIEVAL (derived from FTN_FULL's
checkpoint at eval time), FTN_RANDOM_MASK_TRAIN, FTN_NO_REPLAY,
DENSE_ACTIVE_MATCHED, SOFT_MIXTURE, DISJOINT_TASK_MODULES, JOINT_MULTITASK_FTN,
SINGLE_TASK (stage-B specialist), FTN_PER_TASK_MASK (ablation 12.6).

## Deviations from the design doc (documented intentionally)

- **fp32 everywhere** instead of bf16 autocast: the model is ~3M params; exact
  rounded-integer readout at OOD depth leaves no room for bf16's 2^-9 relative
  precision. Registers/masks/losses are fp32 as specified.
- Flat package layout instead of the deep `src/*` tree (matches the sibling
  `abelian`/`temporal` experiments in this repo).
- **Value readout is the scalar register `r_v`**, not a `1024*tanh(MLP(h))` head.
  With the tanh-from-latent head the optimizer routes value information through
  the message/GRU path and leaves the register unaligned (single-batch overfit
  plateaus at MAE~1-2, register saturates), and a tanh head trained on ID values
  (|v|<=48) cannot extrapolate to OOD-depth values (|v|~256). Reading out the
  register directly supervises the extender scalar path and the hard reduction —
  the alignment the design's registers/CLIP/trace machinery is built around.
  With this change single-batch overfit reaches 100% node accuracy, loss ~1e-6.
- The source node's value output is architecturally pinned to 0 (register clamp),
  consistent with excluding the source from value losses.
- Degree features use log1p(deg)/log1p(6) normalization (bounded ID, mildly >1
  OOD).
