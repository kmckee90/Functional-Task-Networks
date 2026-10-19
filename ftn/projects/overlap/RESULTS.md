# Overlap preservation experiment

## Protocol

The model and task stream use `config.Default`: 8 sine-classification tasks,
500 batches/task, 256 samples/batch, 4,096 mask units with K=256, mask search
every batch, and Adam at 3e-4. Replay uses the unchanged root implementation:
100 stored batches/task, four selected stored masks per update, and overlap
fraction in `(0.2, 0.8]`. The proposed method retains the same number of masks
but no earlier data. It samples four eligible masks with replacement, freezes
their logits on the current inputs before updating, and takes two optimizer
steps on the same current batch. Each step uses current labels under the
primary mask plus the sum of four old-mask logit MSE losses (weight 1). The
first MSE is zero because dropout is zero; the second can correct drift caused
by the first primary update. `none_multi` takes two primary-only steps to
control for extra optimization. All arms share seeds, data order, architecture,
and evaluation schedule.

The standard accuracy evaluation reconfigures on each task and scores a fresh
holdout batch (10 batches/task, 60 reconfiguration steps). Mean recall pools
accuracy on previously trained tasks after each later task block. Final mean
is accuracy across all eight tasks after task 7. Each run has a transfer
matrix in `results/xfer_*.png`.

## Reconfigured task accuracy

Values below average seeds 123 and 124. These are exploratory runs, not
confidence intervals.

| Condition | Mean prior-task recall | Final mean accuracy | Mean current-task accuracy |
|---|---:|---:|---:|
| Replay | 0.931 | 0.930 | 0.938 |
| Current-input overlap, two steps | 0.878 | 0.880 | 0.952 |
| No replay, one step | 0.866 | 0.868 | 0.942 |
| No replay, two steps | 0.850 | 0.850 | 0.949 |

The mask-only method improves mean recall by 1.2 percentage points over the
one-step no-replay setting and by 2.8 points over the matched two-step control.
Replay remains 5.3 points higher. The results are consistent in direction
across both seeds. In seed 123, eligible masks were found on 99.9% of batches
(286.5 eligible masks/batch on average), so lack of overlap selection does not
explain the gap.

Two exploratory settings on seed 123 increased the old-mask MSE weight from 1
to 100. Two steps at 3e-4 gave prior-task recall 0.884, final mean 0.878, and
current-task accuracy 0.923. Four steps at 1e-4 gave 0.883, 0.883, and 0.918,
respectively. Neither closed the replay gap; both reduced current-task
accuracy relative to the weight-1 mask-only setting (0.955 on this seed).

## Frozen-mask accuracy

The standard evaluation can find a new mask and thereby hide damage to a
stored solution. We repeated seed 123 while saving the mask at the end of each
task block. After every later block, we evaluated each saved mask directly on
ten fresh holdout batches from its original task, without reconfiguration. An
independent copy of the task generator supplies this probe so training data
and mask search remain identical to the first seed-123 runs. The corresponding
matrices are `results/xfer_fixed_*.png`.

| Condition | Mean prior-task accuracy under frozen masks | Final mean under frozen masks |
|---|---:|---:|
| Replay | 0.909 | 0.915 |
| Current-input overlap, two steps | 0.716 | 0.753 |
| No replay, one step | 0.709 | 0.721 |
| No replay, two steps | 0.635 | 0.618 |

The mask-only method preserves a little more of the fixed solutions than no
replay, especially relative to the matched two-step control. Replay preserves
far more. The gap is much larger under fixed masks than under reconfigured
evaluation, so a high reconfigured score should not be read as preservation
of the original mask's function.

The likely limitation is intrinsic to this particular target choice: each
batch's teacher outputs come from the *current* weights and are discarded
after that batch. The method limits immediate changes where current inputs
probe old masks, but its target follows any cumulative drift. Current-task
inputs may also leave parts of an old task's input distribution untested.
Replay's old labels remain anchored to the original task. These are
interpretations of the observed pattern, not separate causal tests.

Run details and all block-level scores are in the corresponding JSON files in
`results/`. `render.py` regenerates the transfer PNGs from those files.

Key images: [replay transfer](results/xfer_replay_s123.png),
[mask-only transfer](results/xfer_overlap_s123.png),
[no-replay transfer](results/xfer_none_s123.png),
[replay fixed masks](results/xfer_fixed_replay_fixed_s123.png),
[mask-only fixed masks](results/xfer_fixed_overlap_fixed_s123.png), and
[no-replay fixed masks](results/xfer_fixed_none_fixed_s123.png).
