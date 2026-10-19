# LoRA-Mux: a continual-learning router for LoRA on a real LLM

This experiment is the empirical companion to Section 4 ("FTN as a
Continual-Learning Router for LoRA") of the project's ICML draft.  It
**does not modify any code in `project_c_4/`** outside this folder; it
implements LoRA-Mux from scratch on top of HuggingFace `transformers`.

## What it tests

A frozen pretrained causal LM (default: `distilgpt2`) is continually
fine-tuned on **N mutually-exclusive style-transformation tasks**.  Each
task takes the same source corpus and applies a different deterministic
text→text transformation (uppercase, lowercase, reverse-words, double-words,
starred-vowels, pirate, …).  The input distribution is shared across tasks
and only the target text differs, so this is **strict concept shift** —
the regime where regularization-based CL collapses (cf. MNIST Shuffled in
the main paper).

Three methods are compared, all with the same backbone and the same total
LoRA budget (a pool of `H` rank-`r` components per attention QKV
projection):

| Method            | Routing                                      | Per-task capacity | Expected behaviour |
|-------------------|----------------------------------------------|-------------------|--------------------|
| `LoRA-Naive`      | All-ones mask (single shared LoRA)           | `H · r`           | Catastrophic forgetting on prior styles |
| `LoRA-FixedMask`  | Pre-determined disjoint blocks of size `k`   | `k · r`           | Structural oracle — exactly zero forgetting |
| `LoRA-Mux` (ours) | Pool routed by SmoothKWTA (FTN configurer)   | `k · r`           | Near-oracle retention; mask recovered at inference time without task labels |

`LoRA-Mux` is evaluated under two protocols, exactly as in the parent
FTN paper:

* **stored** — use the mask saved when the task was trained (oracle).
* **reconfig** — cold-start the SmoothKWTA configurer on a fresh support
  batch from the task at evaluation time, with no task identity given
  (the realistic setting).

Reported metrics are per-token cross-entropy on a held-out test split:

* `avg_loss` — average final-task CE across all tasks (lower is better).
* `FM` — average rise in CE on prior tasks above the lowest level they
  reached during training (positive ⇒ forgetting).
* `BWT` — average final-CE − first-trained-CE on prior tasks (positive ⇒
  forgetting in the LM-loss convention).

## File layout

```
experiments/lora/
├── README.md          this file
├── lora_mux.py        LoRAMuxConv1D wrapper, MaskState, configurers
├── tasks.py           style transforms + embedded source corpus + dataloader
├── trainer.py         block-sequential training loop, eval protocols, metrics
├── run.py             CLI driver: loads HF model, sweeps methods, saves JSON
├── results/           output JSONs (created at first run)
└── figures/           output plots (created by plot.py)
```

Nothing here imports from `project_c_4/{models,masks,trainer,tasks}.py`.

## Quick start

Default is the headline 6-task setup on `distilgpt2`:

```bash
cd project_c_4
uv run python experiments/lora/run.py
```

Faster smoke test (3 tasks, 30 steps each, ~10s on H100):

```bash
uv run python experiments/lora/run.py --n-tasks 3 --steps 30
```

A larger run (8 tasks on `gpt2`, longer training):

```bash
uv run python experiments/lora/run.py --model gpt2 --n-tasks 8 --steps 400
```

## Key flags

* `--model`           HF model id (default: `distilgpt2`).
* `--n-tasks`         number of style regimes (1..8 with the default corpus).
* `--n-components`    pool size `H` per LoRA-Mux module (default: 64; also `--grid-dim 8`).
* `--rank`            per-component LoRA rank `r` (default: 4).
* `--k`               KWTA winners per task (default: 8 — i.e. `k/H = 12.5%` capacity per task).
* `--target-names`    which Conv1D modules to wrap in each transformer block.  Default: `c_attn` only.  Add `c_proj`, `c_fc`, `c_proj` of MLP, etc., to broaden.
* `--steps`           training steps per task.
* `--reconfig-steps`, `--reconfig-lr` — Stage-1 inner loop of the SmoothKWTA configurer.
* `--methods`         subset of `{LoRA-Naive, LoRA-FixedMask, LoRA-Mux}`.

## Why these tasks are gradient-discriminable

The mask configurer relies on Stage 1: take a small support batch from the
target task and run a few Adam steps on the relaxed mask logits to minimise
the per-token CE.  For two distinct style regimes, the gradient w.r.t.
mask logits points toward different LoRA components because the loss
surface for `(x, y_uppercase)` is qualitatively different from
`(x, y_reverse_words)`: each transformation requires a different token-
level pattern (case change, vowel removal, word-order flip, fixed-prefix
wrapping, etc.).  The lateral smoothing then projects the noisy proposal
onto compact contiguous blobs on the 2-D component grid, and KWTA
binarises.

## Headline result

Single seed, 6 styles (`upper`, `lower`, `reverse_words`, `double_words`,
`starred_vowels`, `pirate`), 200 train steps per task, `H=1024` (a $32\times 32$ component grid), `r=4`, `k=128`,
`distilgpt2`:

```
Method              Protocol      avg_loss          FM         BWT
LoRA-Naive          stored          8.3127     +7.5493     +7.5493
LoRA-Naive          reconfig        8.3127     +7.5493     +7.5493
LoRA-FixedMask      stored          1.3899     +0.0000     +0.0000
LoRA-FixedMask      reconfig        1.3899     +0.0000     +0.0000
LoRA-Mux            stored          1.7550     +0.3786     +0.3786
LoRA-Mux            reconfig        1.5349     +0.1273     +0.0133
```

**Reading**: `LoRA-Naive` forgets catastrophically (avg CE rises by ~7.5
nats on prior tasks).  `LoRA-FixedMask` retains exactly — `FM = 0.000`
across the board, an empirical confirmation of Theorem 1 in the ICML draft.
`LoRA-Mux` is **near-oracle** under the realistic mask-recovery protocol
(`avg_loss = 1.53` vs oracle `1.39`, `FM = 0.13` vs oracle `0.00`); the
$24\times 24$ component grid gives the configurer enough room to find
nearly disjoint masks across all six task pairs.

### Mask-overlap diagnostic (LoRA-Mux stored masks)

Pairwise `|m_i ∩ m_j| / k` from the headline run, `k = 72`:

```
           upper  lower  rev_w  dbl_w  starv  pirat
upper       1.00   0.36   0.00   0.00   0.00   0.00
lower       0.36   1.00   0.00   0.00   0.00   0.06
rev_w       0.00   0.00   1.00   0.24   0.00   0.00
dbl_w       0.00   0.00   0.24   1.00   0.00   0.00
starv       0.00   0.00   0.00   0.00   1.00   0.18
pirat       0.00   0.06   0.00   0.00   0.18   1.00
```

The largest pair-overlap is `0.36` (`upper`/`lower`, both case-change
tasks); the word-level `reverse_words`/`double_words` pair sits at `0.24`
(down from `0.88` on the smaller `8x8` grid).  All other pairs are exactly
disjoint, and the residual `FM = 0.13` is concentrated on the overlapping
pairs — Theorem 1 honored on the disjoint pairs, mask overlap converted
near-linearly to forgetting on the overlapping ones.

This is the same pattern reported in the parent FTN experiments, just on
a real LLM with LoRA components in place of single neurons.

### Mask layouts on the latent grid

To visualise how the SmoothKWTA configurer organizes the LoRA components,
re-run the experiment over multiple seeds and render the masks as RGB
overlays on the $32\times 32$ component grid (one colour per task):

```bash
for s in 0 1 2 3 4 5 6 7; do
  uv run python experiments/lora/run.py --seed $s \
    --methods LoRA-Mux LoRA-FixedMask \
    --out experiments/lora/results/multiseed/seed_$s.json
done
uv run python experiments/lora/plot_masks.py
```

This produces `figures/mask_tiling_LoRA-Mux.pdf` (and the disjoint-block
oracle for contrast).  Tasks with co-located colours are sharing
components; tasks with disjoint colours are exactly non-interfering by
Theorem 1.

## Tuning notes

* Increasing `--reconfig-steps` (default 10) tightens the mask-recovery
  optimization and improves the `reconfig` protocol, at a small extra
  inference cost.
* Larger lateral kernel / more lateral steps push masks further apart
  (less overlap, lower FM); the defaults (`3×3` kernel, `8` steps) are
  the FTN-Slow-style setting from the main paper.
* Wrapping more modules (`--target-names c_attn c_proj c_fc c_proj`
  picks both attention and MLP projections) raises capacity per task at
  the cost of more LoRA parameters.
