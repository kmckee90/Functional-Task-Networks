# FTN–M2E policy decoder experiment

Both models retain the fixed tuned 128-channel reservoir, RECODE, online
density feedback, offline re-evaluation of density, frontier buffer, and config
24 exploration hyperparameters. Render 256×256; area resize to 64×64 and then
32×64 RGB, with the existing fourth feedback channel. No Dreamer, no survival
reward, no warm-started weights. Seeds 19 and 101 use the existing spawn sites.

One cortical sheet / one Bayesian mask. Each independent neuron has **eight
hidden ReLU layers of width eight**, followed by one scalar tanh output, matching
the `layers=8` convention in base's ParallelMLP. The only downstream trainable
operation is the linear action-value decoder. There are no trainable shared
upstream features or inter-neuron connections.

| Decoder | Neuron input | K | Bank (8K) | Active parameters |
|---|---|---:|---:|---:|
| Flat | flattened 128×4×9 reservoir state | 91 | 728 | 3,406,585 |
| Convolutional | each padded 3×3×128 reservoir patch | 296 | 2,368 | 3,414,952 |

The baseline budget is 3,411,506: both existing 25-action value streams and their
linear heads, excluding frozen targets/reservoir. Both hybrids retain these two
streams but share the sheet and mask; environment reward remains zero. Each
convolutional neuron is the same private deep MLP at every spatial position;
the decoder sees its flattened 4×9 map. No new cross-neuron mixing occurs
between its eight layers. Display grids are 26×28 and 37×64, respectively;
rectangles preserve exactly 8K atoms rather than silently adding square padding.

## Bayesian inference and DQN

`legacy/bayes_reconfig.py` and `overlap_regularizer.py` are copied from
ftn/projects/bayesian_adv; only the sibling import is made package-relative.
Two posterior steps per training batch and every eight environment actions
after the initial 500-transition collection period. Support-loss change points
use three EMA standard deviations after 32 searches and a two-step fresh-prior
refit. Ordinary updates use persistent logits, Gumbel exact-K sampling, sigmoid
STE, Bernoulli KL weight 1, mask learning rate .3 and temperature 1.

Mask search uses the `td` alternative from rl/robotics/train_grid_dqn.py:
Q(s, recorded action) versus density/frontier reward plus masked target-network
bootstrap. Targets are stop-gradient and evaluated under the candidate mask.
The reference's single-search branch currently uses max Q instead of recorded
action (its batched branch gathers the action); this implementation deliberately
uses the recorded action consistently. No artificial episode returns are needed.

For this first comparison, keep M2E's gamma≈.9037, RMSprop lr≈.000190,
target tau=.01, batch≤64, gradient clipping 1, epsilon 1→.01537 over 10k, and
50 offline updates every 50 observations. The grid experiment instead uses
Adam .0005, tau .1, fixed epsilon .1, dropout .1, 50-step AnnealedSTE search,
and defaults to correlation with episodic returns. Those changes are not all
bundled into this experiment. An isolated RMSprop implementation updates only
active atoms and their state; inactive neurons cannot move through momentum.

Each main update can add one stored-mask replay loss using the grid experiment's
.2–.8 overlap filter. A bounded uniform lifetime reservoir of 256 support batches
(up to 16,384 transitions, collected every eight updates) retains their learned
masks and transition data. Replay rewards are recomputed against current density;
the stored policy mask is used for both prediction and bootstrap. Main replay
remains 2M, frontier 500k, RECODE 100k. All state is checkpointed outside sync.

## Monitor and runtime

Four unlimited runs on GPUs 0–3: m2eftn_flat_s19, m2eftn_flat_s101,
m2eftn_conv_s19, m2eftn_conv_s101. Previous run metadata/status snapshots are
in `$GLOBO_STATE/archives/pre_m2e_ftn`; original run directories are retained.
The supervisor continues updating the existing monitor and its comparisons.

Each overwrite-only video pairs the original 256×256 image with the concurrent
action mask. White is active; the old bw-style EMA color trails use .95/.30/.975
decays. The existing SpatialPermuteRandomCenter algorithm (random blob init,
radius 7, center .25, eight sweeps) changes display positions only. One shared
permutation is applied to the entire clip so its mask motion is interpretable.

Validation: `python -m globo_dreamer.verify_m2e_ftn` on the pod tests parameter
counts, actual eight-layer structure, full-bank/selected equivalence, inference
and optimizer isolation, two-step mask search, overlap replay, and checkpoint
roundtrip. Use `start_m2e_ftn` only once with the old supervisor/workers stopped;
thereafter resume `continuous_runs`, never recreate or overwrite the run set.
