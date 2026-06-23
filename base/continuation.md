---
name: NeurIPS revision (reviewer-driven) — func_net_kt
description: In-flight revision of func_net_kt (FTN/DNA) NeurIPS 2026 submission addressing reviewer feedback. Main sweep complete; support-size sweep partial; paper updated with new comparators and limitations.
type: project
originSessionId: 8dccde3c-2f7f-4a90-934a-694c653d9dfa
---
# Continuation: NeurIPS revision for func_net_kt (paused 2026-05-04)

Working dir: `/data/users/kevin/misc_experiments/experiments/func_net_kt`
Paused at 2026-05-04 15:38 PDT.
NeurIPS 2026 deadlines: abstract May 4 AOE, full paper May 6 AOE.

## What triggered this revision
A reviewer rejected the current submission as too narrow and prescribed a revision plan. Their reduced acceptance bar (per user 2026-05-04 prompt):

> Min acceptable package: Split CIFAR-100 + (RotatedMNIST or PermutedMNIST), with strong baselines and support-size / no-replay ablations. Strong package adds Split CUB-200.
>
> Required baselines: ER (matched memory), parameter-isolation/masking (SupSup/HAT/PackNet), support-set-conditioned router (learned, or argmin-support-loss retrieval over stored masks).

We scoped down further (architecture lift for CIFAR-100/CUB takes weeks; deadline is 2 days):
- Run Permuted/RotatedMNIST in addition to existing synthetic + shared-shuffle MNIST
- Add ER, FixedMask + Router (SupSup-like), FTN + Router, FTN-noreplay
- Acknowledge CIFAR-100 / TinyImageNet / CUB-200 as a deferred limitation in the paper

## What is DONE

### Code (all merged into the repo, not committed yet)
- `tasks.py`: added `PermutedMNIST` (10 tasks, fixed pixel permutations) and `RotatedMNIST` (10 tasks, 0–180° rotations), plus a shared `_MNISTBase` class.
- `trainer.py`: added `replay_mode` to `TrainConfig` (`"mask_overlap"` default vs `"random"` for ER baseline). Added `eval_mask_cfg` parameter to `block_sequential_train` so a different configurer can be used at evaluation. Plumbs `mask_store` to the eval configurer.
- `masks.py`: added `RetrievalRouter` — eval-time configurer that picks from `mask_store` by argmin support-set loss. Registered in `CONFIGURERS["Router"]`.
- `models.py`: **bugfix** — `ParallelMLP.forward` previously called `x.squeeze().unsqueeze(-2)` which collapses the batch dim when batch size = 1; replaced with a guarded `if x.dim() == 3 and x.shape[1] == 1: x = x.squeeze(1)` then `x.unsqueeze(-2)`. Required for support_size n=1.
- `experiments/run_revision.py`: unified runner for the 7 method specs (FTN, FTN+Router, FTN_noreplay, FixedMask, FixedMask+Router, NoMask, NoMask+ER). Has `--ablation main|support_size` flag.
- `experiments/launch_revision_parallel.sh`: 8-GPU launcher that shards by (task_family, seed) into per-shard JSONs.
- `experiments/launch_supportsize_parallel.sh`: similar launcher for the support-size sweep (uses GPUs 4–7).
- `experiments/run_revision_merge.py`: merges per-seed shards into `results/revision_<task>.json`.
- `experiments/generate_revision_tables.py`: emits `paper/neurips/tables/revision_main.tex` and `revision_per_benchmark.tex` (mirrors to `paper/arxiv/tables/`); has support-size table emitter that reads `revision_<task>_support_size.json`.
- `experiments/generate_revision_figs.py`: bar charts for ACC / FM / HS per method per benchmark, plus support-size sweep curves; outputs to `figures/revision/`.
- `experiments/update_paper_with_revision.sh`: end-to-end script: merge shards → regenerate tables → regenerate figures → rebuild PDF.

### Main sweep results: ALL 140 RUNS COMPLETE
- 4 task families × 5 seeds × 7 methods = 140 runs, saved in `results/revision_<task>_s<seed>.json`.
- Merged JSONs at `results/revision_<task>.json` (also includes per-(method, support_size=256) aggregation).
- Paper tables and figures regenerated from this data.

### Key results (5 seeds, reconfig protocol unless noted)

| Task | NoMask | NoMask+ER | FixedMask | FM+Router | **FTN** | FTN+Router | FTN-noreplay |
|---|---|---|---|---|---|---|---|
| Synthetic ACC | 0.61 | 0.65 | 0.95 | 0.95 | **0.92** | 0.90 | 0.78 |
| Synthetic FM | 0.40 | 0.17 | 0.00 | 0.00 | **0.03** | 0.05 | 0.20 |
| Synthetic HS | 2.87 | 1.07 | 0.70 | 0.70 | **0.40** | 0.39 | 0.52 |
| MNIST-shared ACC | 0.44 | 0.44 | 0.98 | 0.98 | **0.96** | 0.96 | 0.88 |
| MNIST-shared FM | 0.64 | 0.63 | 0.00 | 0.00 | **0.02** | 0.02 | 0.12 |
| MNIST-shared HS | 10.40 | 8.52 | 2.31 | 2.31 | **0.56** | 0.56 | 0.76 |
| Permuted ACC | 0.93 | 0.96 | 0.98 | 0.98 | **0.95** | 0.95 | 0.94 |
| Permuted FM | 0.05 | 0.02 | 0.00 | 0.00 | **0.02** | 0.03 | 0.03 |
| Permuted HS | 5.21 | 5.27 | 2.32 | 2.32 | **1.62** | 1.59 | 1.58 |
| Rotated ACC | 0.52 | 0.91 | 0.98 | 0.98 | **0.94** | 0.94 | 0.77 |
| Rotated FM | 0.51 | 0.08 | 0.00 | 0.00 | **0.04** | 0.05 | 0.24 |
| Rotated HS | 0.36 | 0.67 | 2.32 | 2.32 | **0.32** | 0.31 | 0.28 |

**Headline**: FTN dominates head-start vs every parameter-isolation baseline on every benchmark (4.1× vs FixedMask+Router on shared-shuffle, 7.2× on rotated). Retention is slightly worse than FixedMask oracle but materially better than NoMask. FTN+Router ≈ FTN, isolating the contribution to *training-time* per-batch reconfiguration. No-replay ablation: removing replay raises FM materially everywhere except permuted (where ER alone also recovers well — random permutations have weak interference).

### Paper (`paper/neurips/main.tex`) updates
- Abstract reframed as "controlled study"; explicitly notes Split CIFAR-100 / TinyImageNet / CUB-200 not yet evaluated.
- New §4.8 "Domain-incremental benchmarks: PermutedMNIST and RotatedMNIST" with comparator-suite description, no-replay paragraph, support-set-size paragraph.
- Discussion: "Mask overlap is not by itself a sharing claim" paragraph (addresses reviewer's interpretability concern about Appendix D and Figure 2 diagonal); "Limitations and scope" with future-work roadmap (HAT, PackNet, SupSup-proper, DER++, GEM/A-GEM, learned-router MoE on backbone benchmarks); "Outlook" rewritten around LoRA-bank scaling.
- Figure 2 caption explicitly explains diagonal entries are "unique to task i" (not self-overlap), and that $r_{\mathrm{share}}$ excludes the diagonal.
- Related work: added "Support-set-conditioned task inference" paragraph (cites SupSup, L2P, DualPrompt; declares the retrieval router as the central comparator class).
- §4.8 narrative now uses concrete numbers (head-start gaps per benchmark, no-replay FM deltas, NoMask+ER limits).
- Body fits 9 pages exactly (Outlook ends p9, appendix at p12, total 21 pages with all appendices/figures).

## What is NOT DONE

### 1. Support-size sweep on Permuted/RotatedMNIST (highest priority)

Reviewer asked for support sizes $\{1, 5, 10, 50, 256\}$ on the new benchmarks. Synthetic was already covered by the existing exp5 ($n \in \{4, 16, 64, 256\}$, 8 seeds), so the new sweep is permuted + rotated only.

State at pause: 6 partial shards each have 1 of 5 support sizes done.

Remaining: 6 shards × ~4 sizes = 24 runs. Each run is ~5–6 min on idle GPUs.

To resume:
```bash
cd /data/users/kevin/misc_experiments/experiments/func_net_kt

# Re-run the permuted/rotated portion only (skips already-completed sizes per shard).
for task in permuted rotated; do
  for seed in 0 1 2; do
    gpu=$(( (seed + (task == "rotated" ? 3 : 0)) % 8 + 4 ))   # GPUs 4-7
    log="logs/supportsize_${task}_s${seed}.log"
    CUDA_VISIBLE_DEVICES="${gpu}" \
      python3 experiments/run_revision.py \
        --tasks "${task}" --seeds "${seed}" \
        --methods FTN --ablation support_size \
        --support-sizes 1 5 10 50 256 \
        --results-suffix "_support_size_s${seed}" \
        > "${log}" 2>&1 &
  done
done; wait
```

Then `bash experiments/update_paper_with_revision.sh` regenerates the support-size table (`paper/neurips/tables/revision_supportsize.tex`) and rebuilds the PDF.

### 2. Optional: 5 seeds on permuted/rotated support-size

Currently scoped to 3 seeds for the support-size sweep. If we want to match the 5-seed convention for all numbers, bump `SEEDS=(0 1 2 3 4)` in the launcher. Adds another 4 shards × 5 sizes = 20 runs.

### 3. Mask-overlap statistics for the new benchmarks (low priority)

`experiments/analyze_mask_overlap.py` has been run only on the original synthetic + shared-shuffle data. Could be extended to permuted/rotated to populate the appendix mask-overlap table (`paper/neurips/tables/mask_overlap.tex`). Probably not necessary for the May 6 deadline.

### 4. Random-support-label ablation (deferred)

Reviewer also asked for a "random support-set labels or mismatched support set" ablation. Not yet implemented. Would test whether the configurer's mask is doing real label-conditional routing or just exploiting the input distribution. Implementation: draw the support set from task t but feed labels from task t' ≠ t. Simple ~30 LOC change to the `RetrievalRouter` and FTN configurer call sites.

### 5. CIFAR-100 / TinyImageNet / CUB-200 (deferred to next venue)

Requires a backbone-with-adapter-masking architecture (ResNet/WRN with masked LoRA or expert modules). Multi-day implementation + multi-day training time. The paper currently acknowledges this as the principal limitation (`§Discussion: "Limitations and scope of the empirical case."`) and proposes the LoRA-bank instantiation as the natural follow-up.

### 6. Stronger CL baselines (HAT, PackNet, EWC, SI, LwF, GEM/A-GEM, DER++, learned-router MoE) (deferred)

Same dependency on the backbone variant. Acknowledged in the same Limitations paragraph.

## Key files / artifacts

- `paper/neurips/main.tex` — current paper (9-page body, fits NeurIPS limit).
- `paper/neurips/tables/revision_main.tex` — 28-row main comparator table (4 tasks × 7 methods).
- `paper/neurips/tables/revision_per_benchmark.tex` — appendix copy.
- `paper/neurips/tables/revision_supportsize.tex` — placeholder; will be populated when support-size sweep finishes.
- `figures/revision/methods_acc.pdf`, `methods_fm.pdf`, `methods_hs.pdf` — bar charts per benchmark per method (not yet referenced in paper text; could add if useful).
- `results/revision_<task>_s<seed>.json` — per-shard saves from the main sweep (4 tasks × 5 seeds = 20 shards).
- `results/revision_<task>.json` — merged aggregates.
- `results/revision_<task>_support_size_s<seed>.json` — support-size shards (6 partial files at pause).
- `experiments/update_paper_with_revision.sh` — one-shot pipeline: merge → tables → figures → rebuild.

## Things NOT to redo / pitfalls

- Do NOT delete `results/revision_*_s[0-9].json` shards — they contain the 140 main-sweep runs and would take ~1.5h to recompute.
- The launcher uses `python3` directly (not `uv run`), but `uv run` from `/data/users/kevin/misc_experiments/` also works.
- `models.py::ParallelMLP.forward` had a batch-size-1 bug — fix is in place. If you see einsum dim mismatch errors, that's the symptom; check the squeeze logic.
- The synthetic support-size sweep is REDUNDANT with exp5 (already in `results/exp5_synthetic.json`, 8 seeds). Only run support-size on permuted + rotated.

## Quick resume script

```bash
cd /data/users/kevin/misc_experiments/experiments/func_net_kt

# 1. Finish support-size sweep (24 remaining runs across 6 shards):
for task in permuted rotated; do
  for seed in 0 1 2; do
    log="logs/supportsize_${task}_s${seed}.log"
    CUDA_VISIBLE_DEVICES="$((4 + ((seed + (task == "rotated" ? 3 : 0)) % 4)))" \
      python3 experiments/run_revision.py \
        --tasks "${task}" --seeds "${seed}" \
        --methods FTN --ablation support_size \
        --support-sizes 1 5 10 50 256 \
        --results-suffix "_support_size_s${seed}" \
        > "${log}" 2>&1 &
  done
done; wait

# 2. Regenerate everything and rebuild paper:
bash experiments/update_paper_with_revision.sh

# 3. Verify pagination (body should be 9 pages, Outlook ends p9):
grep -E "Outlook|appendix.A" paper/neurips/main.aux | head
```
