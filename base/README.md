# func_net_kt

Knowledge-transfer-via-mask-optimisation experiments. Same parallel-neuron
backbone as `func_net_spatial/project_c_4`, but the mask configurer is
KWTA-only (no lateral spatial smoothing), the training protocol enables
mask-filtered replay, and the tasks are constructed with explicit
pairwise-shared concept components so backward transfer is meaningful.

## Project layout

```
func_net_kt/
  masks.py             # KWTA_Sigmoid, KWTA_AnnealedSTE, KWTA_Gumbel + baselines
  models.py            # NeuroModel = parallel-MLP bank + mask-gated readout
  tasks.py             # SyntheticTasks (correlated sinusoids), MNISTSharedShuffle
  trainer.py           # block-sequential trainer + mask-filtered replay
                       #   tracks per-task head-start loss after mask opt
  metrics.py           # ACC, FM, BWT
  plotting.py          # neuron/concept sharing matrices, comparison plots,
                       # everything else inherited from project_c_4
  experiments/
    run_kt.py          # main sweep: 3 methods × 4 budgets × N seeds × 2 tasks
    generate_figures.py# build figures and tables from results/*.json
  paper/icml/draft_kt.tex   # standalone ICML 2026 draft
  results/             # results/<task>.json (auto-merged across runs)
  figures/             # figures/<task>/*.pdf
  logs/                # sweep stdout
```

## Three mask optimisers

All three return a hard top-k binary mask. They differ in how the binary
constraint is handled inside the inner optimisation:

| Method      | Soft mask sample                                      | Init                | Optimiser |
|-------------|--------------------------------------------------------|---------------------|-----------|
| `Sigmoid`   | `sigmoid(m)`                                          | logits = 0          | RMSprop   |
| `AnnealSTE` | `KWTA(m + N(0,τ²)) + m - sg(m)` with τ → 0            | logits = 0          | SGD       |
| `Gumbel`    | `sigmoid((m + Logistic) / τ)` with τ → 0; budget pen. | logits = σ⁻¹(k/H)   | Adam      |

## Running

```bash
# Full sweep on synthetic tasks, three seeds:
uv run python experiments/run_kt.py --tasks synthetic --methods Sigmoid AnnealSTE Gumbel \
    --budgets 50 500 5000 50000 --seeds 0 1 2 --include-baselines

# Same on MNIST shared-shuffle:
uv run python experiments/run_kt.py --tasks mnist_shared --methods Sigmoid AnnealSTE Gumbel \
    --budgets 50 500 5000 50000 --seeds 0 1 2 --include-baselines

# Build figures + tables from existing results:
uv run python experiments/generate_figures.py --tasks synthetic mnist_shared

# Compile paper:
cd paper/icml && pdflatex draft_kt && bibtex draft_kt && pdflatex draft_kt && pdflatex draft_kt
```

## Tracked metrics

For each (method, budget, seed) run we store:

| Metric                   | Where                                            |
|--------------------------|--------------------------------------------------|
| ACC, FM, BWT             | `summary_stored` / `summary_reconfig`            |
| Per-task perf matrix     | `perf_matrix_stored` / `perf_matrix_reconfig`    |
| Head-start loss          | `head_start[t].task_loss_with_mask` per task t   |
| Reconfig init/final loss | `head_start[t].reconfig_initial_loss/final_loss` |
| First-epoch mean loss    | `first_epoch_loss[t]`                            |
| Neuron sharing matrix    | `model_sharing` (built from the per-task masks)  |
| Concept sharing matrix   | `gt_sharing` (built from `task_set.concept_table()`) |
| Sharing fidelity         | `sharing_corr` (Pearson r over off-diagonals)    |
