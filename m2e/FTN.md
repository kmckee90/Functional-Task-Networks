# Bayesian FTN variant of RECODE M2E

Run on the m2e pod, from its workspace:

```bash
uv run python experiment_ftn.py --task Fixed --device cuda --seed 1901
```

`experiment_ftn.py` preserves the `experiment_recode.py` experiment and exposes
FTN controls in its `Hyperparameters` dataclass. `agent_ftn.RDDPG_Dual` inherits
the original agent. Its replay method keeps the original frontier selection,
goal-buffer movement, prioritized contiguous sampling and reservoir encoding;
only the decoder optimization block changes. The original files are untouched.

Unchanged: observation and feedback reservoirs, RECODE density and feedback,
epsilon behavior, action normalization, actor/dual-critic DDPG objectives,
discounts, OCP/FCP ablations, optional empirical goal-return blending (`a`),
replay cadence, and target Polyak update rate.

## FTN decoders

Each of the observation critic, feedback critic and actor has an independent
bank of deep neurons and a Bayesian exact-k mask. Separate masks follow
`experiments/rl/robotics/train_cw.py` and `train_mt10_ddpg.py`; a single DQN
mask cannot directly describe these three different DDPG functions.

Critics select masks by squared TD error against detached bootstrap labels.
Those labels are held fixed for each mask search, then recomputed for training.
The actor searches after the critics, maximizing the original OCP/FCP-weighted
target-critic value through its actions. This uses the original M2E target
critics rather than changing the actor's learning objective. Mask inference
does not train network weights. No task labels or external reward are added.

Globo's Bayesian posterior implementation is vendored under
`agents/ftn_support/`. Default searches take two steps, with persistent logits,
a KL prior, and change-point refitting when the combined critic loss exceeds
its EMA by three standard deviations after 32 searches. Action-time searches
use the latest replay support every eight interactions, once a minibatch is
available. They preserve all four reservoir hidden states; the normal forward
path still follows the original agent's epsilon and reservoir behavior.

All model-side FTN controls are exposed in `experiment_ftn.Hyperparameters`.
Ordinary posterior updates use `ftn_temperature`, `ftn_reconfig_take_best`,
`ftn_sample_type` (`gumbel` or `annealste`) and `ftn_reconfig_noise_scale`
(annealste only), alongside their existing step, learning-rate, support, KL,
prior-decay and stochastic-mask settings. Initial mask sampling has its own
`ftn_initial_temperature` and `ftn_initial_stochastic_mask`.

Changepoint refits use `ftn_changepoint_steps`, `ftn_changepoint_lr`,
`ftn_changepoint_support_size`, `ftn_changepoint_temperature`,
`ftn_changepoint_prior_decay`, `ftn_changepoint_kl_weight` and
`ftn_changepoint_stochastic_mask`. A learning rate of `None` shares
`ftn_reconfig_lr`; a support size of `None` uses all available support rows.
Both ordinary and changepoint temperatures still default to 1.0. Mask-search
overlap regularization uses `ftn_overlap_mode`, `ftn_overlap_lambda`,
`ftn_overlap_threshold` and `ftn_overlap_temperature`; its default weight is
zero. These controls are separate from archived-support overlap eligibility.
Hyperparameters are included in the experiment's MLflow parameter logging.

Defaults use width 8, depth 8 and eight times as many bank neurons as selected
neurons. If `ftn_k` is None, each decoder's active count is calculated to roughly
match its original decoder's trainable parameter count; integer rounding means
it is not an exact match. `ftn_hidden_size` overrides the total bank size.
`agent.ftn_spec` reports the actual counts for all three decoders.

Every eighth update archives a support and its three masks in a bounded,
uniform-lifetime reservoir (256 entries). One archived support with 0.2–0.8
overlap in at least one mask contributes an additional loss under its saved
masks. Supports retain their sampled reward values; bootstrap targets are
recomputed. They store frozen reservoir features, so replay does not overwrite
the live recurrent context. RMSprop updates only the union of active atoms,
including their optimizer state. Set `ftn_replay_capacity=0` to disable this
auxiliary replay without disabling M2E's original replay/frontier buffers.
Overlap eligibility is computed in one CPU batch per bank, with three current
mask transfers total. Float64 comparisons preserve the original threshold
boundaries, archive ordering, sampled support and random-number consumption.

Masks and posterior logits are registered buffers. Agent state dictionaries
also retain the FTN support archive and change-point statistics; whole-agent
serialization retains optimizers as in the original experiment. Resetting
memories also clears the FTN support archive.

## Experiment conditions

`--ftn_setting Current` preserves the existing Bayesian FTN schedule and the
256-entry support archive. `--ftn_setting Reactive_Low` adapts the reference
`ftn/config.py` loss-triggered schedule: one initial mask search, followed by
searches only when the combined enabled-critic TD loss exceeds its EMA mean
by five standard deviations. EMA retention is 0.9; triggers are suppressed
for `ceil(1 / (1 - EMA))` checks after a search. Statistics persist across
episodes and use post-search loss rather than recording the pre-search spike.
Triggered searches use the existing Bayesian changepoint refit. Stable checks
leave masks and logits unchanged. Action-time checks still occur every eight
interactions; they do not cause unconditional reconfiguration in this setting.

`--no_ftn_replay` disables the FTN archived-support replay (`ftn_replay_capacity=0`).
It retains the main DDPG experience/goal buffers and their ordinary training.
Combine it with `--ftn_setting Reactive_Low` for the Reactive_Low NoReplay group.
Both groups retain the current network sizes, objectives, optimizations, and
asynchronous red 32-rollout diagnostics. Settings are logged as MLflow params.

## Retrospective mask routes

Every 20 completed training episodes, `experiment_ftn.py` evaluates the most-used
actor mask from each of those episodes for 32 complete diagnostic episodes.
Mask counts are recorded after action-time reconfiguration and before stepping
the maze. Critic-only changes do not split an actor mask's usage count; ties
choose the first mask. The associated most-used critic masks are also shown.

Each artifact is a 4-by-5 grid of episodes: red visitation dots on the left, the
actor/OCP/FCP masks on the right. Diagnostic rollouts use block-end weights,
zero initial reservoir/feedback state, epsilon zero, fixed masks and no mask
search or learning. They use the original episode's maze geometry and start
position (including continual mazes). Each rollout receives an independent copy
of the block-end density estimator, updated locally to provide feedback. Common
noise seeds across masks make route differences easier to compare.

All distinct masks encountered during action selection contribute to one
cumulative display permutation per neuron bank. The vendored spatial algorithm
uses the same SpatialPermuteRandomCenter options as Globo: random blobs,
center 0.25, roundness 2, radius 7, temperature 0.05, seed 0, eight sweeps. Later
blocks refine the previous permutation; every cell in a block uses the same
layout. Neuron indices and the actual masks are never permuted in the model.
Mean neuron co-activation is precomputed once per block, so each proposed swap
uses neighboring pairwise affinities rather than the full growing mask history.
The objective and all cumulative masks are retained; floating-point differences
can change which equivalent display permutation is selected.

PNG grids and NPZ/JSON data are written under
`outputs/ftn_diagnostics/<run>/`. With `--log`, only the PNG grids are uploaded
to the run's `ftn_diagnostics/` artifact directory; NPZ/JSON stay local to the pod.
Data includes every within-episode
mask change, all cumulative masks, display permutations, maze walls, rollout
visitations, and evaluation settings. No external logging is needed for local
artifacts. Use `--ftn_diagnostic_dir` to choose a location or
`--no_ftn_diagnostics` to disable the extra evaluation work.

By default a spawned worker process evaluates diagnostics asynchronously on the
same CUDA device as training. Training briefly copies the actor, two reservoirs
and density estimator into a frozen CPU snapshot, then resumes. The worker has
independent RNG streams, hidden states, maze instances and CUDA storage. Snapshots
retain their exact block-end weights even if later training changes the model.
One worker processes blocks in order and retains the previous spatial layouts.

At most two blocks are outstanding (one executing, one queued). Training waits
if that queue fills; no completed block is dropped. Errors propagate on polling
or final drain. The worker uploads artifacts directly to the explicit originating
MLflow run ID, and all outstanding work finishes before the run closes. GPU
compute and memory are shared, so the speedup depends on contention. Set
`--ftn_diagnostic_sync` to compare against synchronous evaluation. An incomplete
final block is not evaluated.

Launch four full runs on the m2e 4x pod, one per GPU, with seeds 1901–1904:

```bash
uv run python launch_ftn_routes.py
```

The launcher records run IDs, source hashes, PIDs, seeds and log paths in
`outputs/ftn_route_runs/<UTC timestamp>/manifest.json`. Each detached process
runs the full experiment with Databricks logging and diagnostics enabled.

For two Current runs (GPUs 0–1) versus two Reactive_Low NoReplay runs (GPUs 2–3),
using seeds 1901 and 1902 in both conditions:

```bash
uv run python launch_ftn_routes.py --compare_reactive_low_no_replay
```

Tests on the pod:

```bash
OMP_NUM_THREADS=2 uv run python -m unittest test_ftn_reactive test_ftn_optimizations test_ftn_async test_ftn_diagnostics test_agent_ftn test_agent_recode -v
```

Bounded performance comparisons on the same pod:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 uv run python profile_ftn.py --k 256
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 uv run python benchmark_ftn_async.py
```

`benchmark_ftn_async.py --history <diagnostic.npz>` includes a real cumulative
mask history in the synchronous/asynchronous comparison. It measures 200
training updates plus 64 complete diagnostic rollouts, including worker startup
and final draining. `--validate-databricks` instead runs a bounded 20-episode
experiment and verifies the worker's three artifacts in Databricks.
