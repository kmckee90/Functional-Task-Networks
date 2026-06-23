# project_c_4 — Session Notes (paper draft_v4)

Session date: 2026-04-22
Purpose: record of what was done and key decisions while producing
`paper/draft_v4.tex`, so a future session can pick up without re-reading
the full codebase.

## What exists in project_c_4

- Parallel-neuron backbone (`models.py::NeuroModel`): H=1024 independent
  per-neuron MLPs (layers=8, inner_size=8, dropout_p=0.2), mask-gated
  scalar outputs, single linear readout. No weight sharing between slots
  except the readout columns (one column per slot).
- Three-stage mask configurer (`masks.py::SmoothKWTA`): sigmoid-relaxed
  gradient optimisation via Adam (cold-start from zeros each call)
  → lateral torus convolution → KWTA (k=128).
  - `SmoothKWTAFast` — kernel 17, 2 lateral steps (FTN-Fast)
  - `SmoothKWTASlow` — kernel 3, 15 lateral steps (FTN-Slow)
  - `KWTAOnly` — no lateral smoothing (ablation)
  - `FixedMask`, `NoMask` — baselines
- Trainer (`trainer.py`): block-sequential; runs BOTH `stored` and
  `reconfig` (mask-recovery) evaluation protocols after each task.
  Headline numbers use reconfig.
- Four experiments (`experiments/multi_seed_experiments.py`),
  8 seeds each, results in `results/exp{1,2,3,4}*.json`.

## Headline results (reconfig protocol, mean±std, 8 seeds)

- **Exp 1 Synthetic clf** (3 tasks): FTN-Fast ACC=0.891 FM=0.066;
  FTN-Slow 0.833/0.080; KWTA 0.561/0.201; FixedMask 0.949/0.061;
  NoMask/EWC ~0.67/0.47.
- **Exp 1 Synthetic reg** (MSE, lower=better): FTN-Fast 0.249;
  FTN-Slow 0.335; KWTA diverges (2.6); FixedMask 0.035; NoMask 0.314.
  Note the Exp-1 table caption says "$-$MSE" but the numbers are MSE.
- **Exp 2 MNIST Shuffled** (5 tasks, pure concept shift):
  FTN-Slow 0.976/0.004 (best); KWTA 0.951/0.025; FTN-Fast 0.938/0.052;
  NoMask/EWC ≈0.255/0.904 (chance).
- **Exp 3 Permuted MNIST** (10 tasks, domain shift):
  FTN-Slow 0.959/0.019; FTN-Fast 0.956/0.023; KWTA 0.926/0.042;
  EWC 0.941/0.037; NoMask 0.829/0.164.
- **Exp 4 Energy Demand** (4 tasks, tabular concept shift):
  FTN-Slow 0.891/0.020 (best adaptive); FTN-Fast 0.855/0.066;
  KWTA 0.786/0.159; FixedMask 0.903/0.004 (upper bound);
  NoMask/EWC ~0.625/0.375.

Consistent ranking across MNIST-style + Energy benchmarks:
FTN-Slow > FTN-Fast > KWTA > EWC > NoMask.

## Paper decisions (draft_v4.tex)

- **Standalone** research paper — does NOT reference project_c_3 or
  compare to earlier versions. draft_v3 (main.tex) was a changelog-style
  comparison to project_c_3; draft_v4 replaces that framing.
- **Title**: "Cortex-Inspired Continual Learning via Unsupervised Task
  Network Instantiation and Recovery" (simplified from earlier long title
  at user's request).
- **Story**: spatial masking enables fast gradient-based recovery of
  prior-task subnetworks (mask-recovery protocol). The
  `instantiate = recover` unification is the key hook: the same procedure
  installs and retrieves subnetworks.
- **Complexity claim**: spatial organisation reduces mask search from
  combinatorial ($\binom{N}{k}$) to ~$\mathcal{O}(N)$ compact
  neighbourhoods on the grid.
- **Neocortical motifs** explicitly listed with algorithmic role:
  dendritic trees (Poirazi/Papoutsi, Beniaguev), scalar spike bottleneck
  → MoE-like diversification, dropout, KWTA (Maass; Douglas & Martin),
  lateral excitation for spatial search reduction, gradient-as-dopamine
  gating cortical attractors (Mink, Schultz, Rolls).
- **Related work**: EWC, SI, PackNet, HAT, SupSup, Progressive, DEN;
  MoE routing (Shazeer, Switch); task-free CL (Aljundi); concept drift
  (Gama); Parisi review. All in refs.bib.
- **Discussion points requested by user**:
  - KWTA alone has harder search without spatial dynamics (done).
  - KWTA's combinatorial space is useful for OOD/knowledge transfer but
    NP-hard, explicitly framed as future work (done).
  - Mask-overlap as replay query mentioned as natural consequence
    (trainer has a replay buffer, but main experiments run without replay).
- **Structural forgetting theorem** included for the parallel-neuron
  backbone (disjoint masks → disjoint gradients).

## Files modified/created

- `paper/draft_v4.tex` — new standalone paper (12 pages compiled).
- `paper/refs.bib` — appended Poirazi/Papoutsi, Beniaguev, Maass,
  Douglas & Martin, Hubel & Wiesel, Mink, Schultz, Rolls, Srivastava,
  Shazeer, Fedus (Switch), Aljundi, Gama, Parisi.
- `paper/draft_v4.pdf` — compiled output.
- Did NOT modify: existing `main.tex`, `draft_v3.txt` (empty), results
  JSON, code.

## Known caveats

- Exp-1 regression table column is labelled "$-$MSE" but the values are
  actually MSE (lower = better). The prose in draft_v4 handles this by
  calling the values MSE directly in the discussion sentences.
- Exp-3 figures dir is missing `mask_grid_*.pdf` (only mask_tiling).
  The paper does not cite missing figures.
- Replay buffer is implemented in trainer but `use_replay=False` in all
  headline experiments. draft_v4 discusses it as a natural consequence
  of mask overlap, not as an evaluated component.
