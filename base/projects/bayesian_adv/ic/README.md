# Input-conditioned reconfiguration experiment

## Gradient replay versus mask repair

`REPLAY_REPAIR_SWEEP.md` compares direct retrieval and retrospective repair
while holding the stored replay buffer at 100 batches per task and 16 rows per
batch. `ICGradReplay0`, `ICGradReplay4`, and `ICGradReplay8` vary the number of
replay-gradient batches per training step; `ICGradReplay8Normalized` keeps
eight sampled batches but divides each replay loss by eight. Compose one of
these with `ICEvalOnly ICNoRepairDirect` or `ICEvalOnly ICReplayRepair` after
`ICExperiment ICFull`. Add `ICRNNFull` for BayesLoRARNN. `ICRepairHeavy`
increases the offline search to five 200-step restarts. The predictive
model's training mask search keeps full support, and evaluation routes from
four inputs.

## Replay-retrieved mask logits

`ICReplayLogits` extends the existing replay tuples with a snapshot of the
predictor's mask logits `L`. The gradient replay rule is unchanged. At
evaluation, four support inputs select the replay batch whose saved inputs
have the smallest mean nearest-example distance. Its `L` supplies the mask
prior; by default the model directly uses its top-k mask. The support labels
do not influence retrieval. `ICReplayLogitsAverage` instead averages the `L`
vectors from eight nearest replay batches. `ICReplayLogitsRefine` optionally refines the
retrieved prior using those four labels, and `ICReplayLogitsZero` is a paired
four-label search control. `ICReplayLogitsSubspace` matches each replay batch
by rank-two reconstruction error instead of individual distances. No old
mask is repaired in these conditions.

For example, on the pod:

```sh
cd /home/dev/workspace/projects/bayesian_adv/ic
CUDA_VISIBLE_DEVICES=0 uv run --project /home/dev/workspace python run_experiments.py \
  --config BayesLoRA DomainShift ICExperiment ICFull ICReplayLogits \
  --name rp_lora_full_nearest --out-root artifacts --metrics-only
```

Add `ICRNNFull` for BayesLoRARNN, or omit `--metrics-only` for the usual plots.
The parent-scale predictive model still has full training change-point
support; only evaluation uses four support rows.

## Shared autoencoder mask → learned logit prior

`ICAutoencoder` is a second, separate experiment in this folder. A small Base
FTN learns `x -> x` using a Reactive_Low-style loss change detector and no
replay. Its mask `a` gates a separate Base FTN trained as `x, a -> L`, where
`L` is a vector of logits for the Bayesian predictor's mask. This logit FTN
never reconfigures its own mask. At evaluation, four input rows reconfigure the
autoencoder on `x -> x`; `x, a -> L` then predicts a prior for `x, m -> y`.
The predictive model's training change-point search keeps its full support.
The auxiliary networks receive no old-domain replay or retrospective updates.

`ICAutoencoder` defaults to applying the predicted logits' hard top-k mask
directly. `ICAutoencoderRefine` instead uses them as a Bayesian prior and
refines it with four labeled examples; this performed worse in the first
trials. `ICAutoencoderDirect` is an explicit alias for the default behavior.
`ICAutoencoderZero` trains both auxiliary networks but ignores them at eval for
a paired four-label search control. `ICAutoencoderColdTeacher` distills a
full-support current-block mask instead of the current posterior logits.
`ICAutoencoderFreeze` prevents its old active neurons from receiving later
gradients. All auxiliary sizes, loss search steps,
fit budget, and prior strengths are in `ic/config.py`.

Example on the pod, preserving the parent-scale predictor and four-row eval:

```sh
cd /home/dev/workspace/projects/bayesian_adv/ic
CUDA_VISIBLE_DEVICES=0 uv run --project /home/dev/workspace python run_experiments.py \
  --config BayesLoRA DomainShift ICExperiment ICFull ICAutoencoder ICAutoencoderDirect ICAutoencoderColdTeacher \
  --name ae_lora_full_coldteacher --out-root artifacts --metrics-only
```

Use `BayesLoRARNN` and add `ICRNNFull` for the recurrent model. `RESULTS.md`
records the experimental findings and the major limitation: the predictor's
old masks can become stale as its shared weights change, even if `x -> L`
remembers them exactly.
Omit `--metrics-only` to generate the pipeline's usual graphics.

## PCA retrieval experiment

This folder copies the parent `run_experiments.py` and inherits its models,
task generator, replay buffer, and metrics. All modifications are local to
`ic/`. Run it from this folder with the root project's `uv` environment.

The experiment uses `BayesLoRA` and `BayesLoRARNN` with `DomainShift`, while
setting `CONCEPT_SHIFT=False`. Each domain is a two-dimensional input subspace
rotated into a common input vector. The initial reduced experiment uses four
labeled support rows for mask search. The parent-scale follow-up restores
full-batch training change-point fitting, while retrospective evaluation
routes from four input rows. Training retains the original replay pipeline.

`ICNoPrior` and `ICPrior` use the same input reconstruction novelty trigger.
At a detected change, the old segment's posterior logits are archived with a
rank-two linear autoencoder learned from its unlabeled inputs. The old entries
are immutable. `ICNoPrior` follows the existing zero-logit change-point search.
`ICPrior` selects an archive entry by reconstruction error and starts the same
number of search steps from its logits, with a small KL penalty. Retrospective
evaluation does the corresponding input-conditioned versus zero-logit search.
When no entry matches, both conditions use the original cold search.
`ICTrainOnly` and `ICEvalOnly` isolate the two places where the prior is used.

For the parent-scale comparison, `ICFull` restores the parent's LoRA capacity,
six-task stream, and 1500 training batches per task while retaining four-row
retrospective support. `ICRNNFull` uses the parent's 500-batch recurrent
training duration. `ICReplayRepair` keeps the input router and its existing
entries frozen, but periodically re-searches archived prediction masks using
the labeled examples already in the training replay buffer. Replay batches are
assigned to domains by input reconstruction error, with no task ID in the
router. A held-out replay split accepts only masks that improve validation
loss. Evaluation then uses the retrieved hard mask directly, so its four input
rows supply routing evidence and no evaluation labels are needed. This avoids
short, unstable refinement from saturated mask logits.

The reconstruction model is the closed-form optimum of a linear autoencoder
(PCA), which is particularly suitable for this generator. This tests whether
an easier `x -> x` problem can select a prior for `x -> y`, while avoiding
forgetting in the routing function. It does **not** test whether the mask of a
second, independently trained FTN autoencoder transfers directly to the
predictor's atom bank; those mask coordinates have no intrinsic alignment.

Example, from `/home/dev/workspace/projects/bayesian_adv/ic` on the pod:

```sh
CUDA_VISIBLE_DEVICES=0 uv run --project /home/dev/workspace python run_experiments.py \
  --config BayesLoRA DomainShift ICExperiment ICNoPrior \
  --name lora_base_123 --seed 123 --out-root artifacts --metrics-only

CUDA_VISIBLE_DEVICES=1 uv run --project /home/dev/workspace python run_experiments.py \
  --config BayesLoRA DomainShift ICExperiment ICPrior \
  --name lora_prior_123 --seed 123 --out-root artifacts --metrics-only
```

Replace `BayesLoRA` with `BayesLoRARNN` for the recurrent model. The effective
config and scalar metrics are written under `ic/artifacts/<name>/`.

Parent-scale replay-repair run, launched from the same pod directory:

```sh
CUDA_VISIBLE_DEVICES=0 uv run --project /home/dev/workspace python run_experiments.py \
  --config BayesLoRA DomainShift ICExperiment ICFull ICEvalOnly ICReplayRepair \
  --name lora_full_repair_123 --seed 123 --out-root artifacts --metrics-only
```

For the fixed-I/O recurrent model, add `ICRNNFull` after `ICFull` and use
`BayesLoRARNN`. See `RESULTS.md` for both the initial small-model experiment
and the parent-scale follow-up.
