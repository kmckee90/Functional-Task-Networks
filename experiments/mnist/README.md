# Manuscript MNIST experiments

Restored from `base/` at commit `3884dea0ed9c969aa4dea6cd5bad51bc44379d57`
(the parent of `e4b6543`, which replaced the old base implementation).
The 32 restored files are unchanged. `RESTORATION.json` records their original
paths and SHA-256 hashes; `README.original.md` preserves the historical README.
This restoration includes the synthetic task controls because the same runners
and table generators use them. It excludes the separate `streams_project_c`
prototype and LoRA planning notes.

## Main entry points

| File | Purpose |
| --- | --- |
| `experiments/run_revision.py` | Five-seed suite: shared-shuffle, Permuted, Rotated MNIST, and synthetic tasks |
| `experiments/run_kt_exp_mnist.py` | Original eight-seed shared-shuffle experiment (exp6) |
| `baselines.py` | Manuscript's parallel-MLP SupSup and Progressive adaptations |
| `tasks.py` | Original task generation, transformations, sampling, and evaluation |
| `trainer.py`, `masks.py`, `models.py` | Original FTN training, mask algorithms, and architecture |
| `experiments/generate_revision_tables.py` | Full revision results tables |
| `experiments/generate_figures_mnist.py` | Original exp6 plots and table |

The revision runner includes `FTN`, `FTN+Router`, `FTN_noreplay`, `FixedMask`,
`FixedMask+Router`, `NoMask`, `NoMask+ER`, `FTN_RandMask`, `SupSup`, and
`Progressive`. Defaults retain H=8192, K=512, 1,000 updates per task, batch 256,
and learning rate 3e-4. Shared-shuffle has six tasks; Permuted and Rotated each
have ten. These are the old manuscript settings, not the newer benchmark FTN
settings.

## Running on a compute pod

Use the repository's actl workflow for runtime work. From the remote source
checkout, with the dependencies in `requirements.txt` available:

```bash
# Inspect options without downloading data or starting training.
bash experiments/mnist/run.sh experiments/run_revision.py --help

# All original comparators on the three MNIST families, five seeds.
bash experiments/mnist/run.sh experiments/run_revision.py \
  --tasks mnist_shared permuted rotated --seeds 0 1 2 3 4

# A separate output file for the old learned-mask and column baselines.
bash experiments/mnist/run.sh experiments/run_revision.py \
  --tasks mnist_shared permuted rotated --methods SupSup Progressive \
  --seeds 0 1 2 3 4 --results-suffix _legacy_baselines
```

Set `PYTHON=/path/to/environment/bin/python` to choose an existing environment.
`run.sh` sets the working directory to this experiment. Original scripts keep
their historical relative paths: `data/`, `results/`, `figures/`, and
`paper/{neurips,arxiv}/tables/` are **inside this directory**. They do not update
`manuscripts/arxiv/v2` automatically. Runtime data/results should remain on the
pod and outside source sync; these paths are covered by `.actlignore` (with the
MNIST data exclusion added for this restoration).

For seed-sharded runs, give each process a distinct `--results-suffix`, such as
`_s0`, then use `experiments/run_revision_merge.py`. The historical shell launchers
are preserved for reference: some assume eight GPUs and list only a subset of
methods. Prefer explicit revision-runner commands appropriate to the actual pod.
Do not have multiple processes write the same result JSON.

The restoration does not include historical raw results, checkpoints, or a
locked original environment. It is source recovery, not a claim that the paper's
numbers have been reproduced. Current manuscript tables remain in
`manuscripts/arxiv/v2/tables/`.

## Baseline expansion agreed for the next comparison

| Task suite | Existing comparisons to retain | Additions |
| --- | --- | --- |
| Shared-shuffle / Permuted / Rotated MNIST | Original FTN and controls, SupSup, Progressive | DER++, L2P, HAT |
| CORe50 NC / TinyImageNet | New benchmark FTN variants, DER++, L2P | SupSup, Progressive, HAT |

This is the intended comparison matrix, **not a claim that these new adapters
or runs are already implemented**. The existing native-method integration is
in `benchmarking/run_mammoth.py`; frozen visual features and train/support
splits are in `benchmarking/vision_data.py`.

Before combining results, preserve these distinctions:

- The restored SupSup freezes a random parallel-MLP bank and learns per-task
  masks and readouts. Progressive uses disjoint trainable columns and cumulative
  feature access through per-task readouts. They are architecture adaptations,
  not unchanged upstream implementations. Progressive's active capacity grows
  with the number of columns.
- Both already implement support-loss retrieval without an evaluation task ID,
  as well as stored-mask/readout evaluation with oracle task identity. Carry
  both protocols to the new visual benchmarks and keep all output classes
  available when reporting the support-retrieval result.
- Historical MNIST support is drawn from the training pool. New visual
  benchmarks reserve a separate 10% support holdout. Preserve the historical
  protocol for reproduction; treat any MNIST support-holdout experiment as a
  new protocol and rerun all affected methods.
- Shared-shuffle reproduces the original component-overwrite label mappings.
  They need not be bijective digit permutations. Do not replace this task
  generator with a standard label-permutation dataset when adding baselines.
- Use the exact old pixel permutations, rotations, normalization, task order,
  and labels for the new MNIST methods. L2P additionally needs its native ViT
  image preprocessing/backbone; report that architectural difference. Do not
  silently substitute Mammoth's default Permuted/Rotated MNIST generators.
- HAT's native task-conditioned evaluation and any support-based head/mask
  retrieval adaptation must be reported separately. Native task information
  is not interchangeable with inferred context.
- Match or explicitly report updates, support examples, memory in examples
  and bytes, stored parameters, and active capacity. Do not pool seed-0 visual
  results with five-seed MNIST summaries as though their uncertainty matched.

## Validation

`check_restoration.py` checks the historical file hashes, imports the restored
modules, and exercises all ten training/evaluation paths on a tiny synthetic
fixture. It also checks the three MNIST generators with an in-memory image
fixture, without downloading the dataset. This is a restoration smoke check,
not an accuracy experiment:

```bash
bash experiments/mnist/run.sh check_restoration.py
```
