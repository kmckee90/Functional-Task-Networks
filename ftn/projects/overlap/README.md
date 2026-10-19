# Current-input overlap preservation experiment

See [RESULTS.md](RESULTS.md) for the completed comparison and transfer images.

`run.py` compares basic FTN training under three settings. `replay` is the
existing `ftn/run_experiments.py` replay rule: retain the first 100 labeled
batches and masks per task, select four stored masks per update from the
overlap band `(0.2, 0.8]` with replacement, and backpropagate their labeled
losses alongside the current batch loss. `none` uses one current-batch update
without replay. `overlap` retains only masks; on the *current* batch it takes
predictions under four selected stored masks before changing the weights, then
uses those detached predictions as MSE targets during repeated optimizer steps
on the same current inputs. The current labels supervise the primary mask in
every inner step. Both the primary mask and the selected old masks stay fixed
within the inner loop. The first old-mask MSE is zero with default dropout 0;
later inner steps can oppose the change caused by the primary update.

`none_multi` is a matched optimizer-step control: it takes the same number of
current-batch steps as `overlap` without old-mask losses. Thus any improvement
over `none_multi` is evidence for preservation beyond extra optimization.
The new method never stores earlier inputs, labels, or output targets. Targets
are recomputed at the beginning of each batch from the current model, so they
constrain immediate functional drift rather than recover an older function
once it has already been overwritten.

All arms use `config.Default` (8 tasks, 500 batches/task, 256 samples/batch,
4096 mask units, K=256, 30 training reconfiguration steps, Adam 3e-4) and
the root runner's holdout backwards/prospective evaluation (10 batches/task,
60 reconfiguration steps). The task generator uses seed `seed+100`. Evaluation
scores are reconfigured task accuracy; `xfer_*.png` shows accuracy after each
training block (rows) on each task (columns). New runs also save the final
mask from each task block and test it later without reconfiguration on an
independent holdout stream with the same task functions. `xfer_fixed_*.png`
shows this stricter fixed-solution accuracy. Runs emit per-block and summary
metrics in JSON under `results/`.

Example on the project's `actl` pod:

```bash
cd /home/dev/workspace
CUDA_VISIBLE_DEVICES=0 .venv/bin/python projects/overlap/run.py --mode overlap --seed 123
.venv/bin/python projects/overlap/render.py projects/overlap/results/overlap_s123_n2_lr0.0003.json
```

Use `--steps`, `--weight`, and `--lr` to test more inner steps, stronger
preservation, or lower learning rates. `render.py` accepts any number of result
JSON files and writes a corresponding `xfer_*.png` beside each one.
