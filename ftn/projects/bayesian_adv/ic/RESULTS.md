# Input-conditioned routing on DomainShift (September 2026)

## Retrieving logits from the existing replay buffer

This simpler experiment stores the predictor's current mask logits `L` beside
each ordinary replay `(x, y, m)` batch. It leaves replay gradient sampling and
full-support training reconfiguration unchanged. At evaluation, four input
rows score each retained replay batch by their mean distance to its nearest
saved input; the best batch supplies `L`. The `average 8` variant averages
logits from the eight closest batches. Both use the logits' hard top-k mask
directly, without evaluation labels. The control performs the existing
four-label zero-prior mask search. All runs use six DomainShift tasks,
`CONCEPT_SHIFT=False`, one pass, and seed 123.

| Model | Current-block training accuracy | Four-label control recall | Nearest one `L` recall | Average eight `L` recall | Refine one `L` with four labels | Correct-domain retrieval |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BayesLoRA | 0.9133 | 0.5601 | 0.7624 | **0.7796** | 0.7080 | 1.000 |
| BayesLoRARNN | 0.8171 | 0.5122 | 0.7043 | **0.7063** | 0.6028 | 1.000 |

Recall is the retrospective held-out accuracy averaged after successive
training blocks. Training accuracy averages each block while current; the
columns therefore do not score identical examples. The predictor training
trajectories were paired within each model. The replay buffer retained 100
sampled batches per task and 16 input/label rows per batch. No retrospective
mask refresh or repair was used. Correct-domain retrieval is diagnostic only;
the matcher never sees task IDs or evaluation labels.

This confirms that a replay buffer can provide useful input-conditioned
routing without a learned router. Domain matching is already perfect on this
simple generator, so the remaining gap to training performance is not caused
by picking another domain. Averaging modestly helps LoRA and barely changes
the recurrent result. These are train-time posterior snapshots, and shared
predictor weights keep changing; the replay-repair experiment below explicitly
addresses that source of staleness. The fixed-size replay buffer would also
evict old domains in a longer task stream.
Starting a four-label Bayesian mask update from the retrieved `L` reduced
recall in both models, so direct use of the retrieved mask was the better
evaluation strategy here.

Reproduce with `ICReplayLogits`, optionally followed by
`ICReplayLogitsAverage`, or use `ICReplayLogitsZero` for the control. The
matching metric can be changed to rank-two replay-batch reconstruction with
`ICReplayLogitsSubspace`; its smoke run passed, but nearest-example matching
already identified every measured domain in the full runs. Scalar files and
logs are under `artifacts/rp_*/`.

## Shared FTN autoencoder mask and learned logit prior

This experiment implements the proposed three-network path. A small Base FTN
autoencodes inputs with a Reactive_Low-style detector and no replay. Its hard
mask `a` also gates an independent Base FTN that learns `x, a -> L`, where `L`
is the predictive model's mask-logit vector. The BayesLoRA or BayesLoRARNN
predictor learns `x, m -> y` through its existing training procedure. The
auxiliary FTNs only learn from the current task block. At retrospective
evaluation, four inputs reconfigure `x -> x`, and the resulting `x, a -> L`
output selects `m`. The learned prior is applied only at evaluation. Predictor
change-point fitting during training retains full support. `CONCEPT_SHIFT` is
false, and the domain generator is otherwise the existing DomainShift setting.

The best variant here trains `x -> L` at each block end from 256 labeled rows
of that *current training block*: it searches a teacher mask with 100 steps
and three restarts, then distills its positive and negative mask logits. No
old-domain rows, old targets, or retrospective repairs enter either auxiliary
network. The four-row limit applies to evaluation. The reported `direct`
strategy uses the predicted logits' top-k mask without evaluating the four
labels; the `zero` control instead searches a mask from those four labels.

| Model, seed | Training accuracy while current | Four-row recall, zero control | Four-row recall, learned direct mask | Stored teacher mask on retrospective queries |
| --- | ---: | ---: | ---: | ---: |
| BayesLoRA, reduced scale, 123 | 0.6779 | 0.5221 | **0.6565** | 0.6663 |
| BayesLoRA, parent scale, 123 | 0.9032 | 0.5576 | **0.5984** | 0.7312 |
| BayesLoRARNN, parent scale, 123 | 0.8117 | 0.5107 | **0.6102** | 0.6868 |
| BayesLoRARNN, parent scale, 124 | 0.8767 | 0.5530 | **0.6728** | 0.7623 |

Training accuracy is averaged across each task's training block when it was
current. Recall is retrospective held-out accuracy averaged after successive
blocks, so the columns do not score identical examples. The reduced LoRA run
has four tasks, two passes, and 120 batches per task; the parent-scale runs
have six tasks, one pass, and 1500 LoRA or 500 RNN batches per task. All
conditions in each row share a seed and predictor training stream.

The recurrent two-seed mean is 0.6415 with the learned mask versus 0.5319
with four-label search. The reduced experiment nearly reaches its training accuracy. Parent-scale
retention is much weaker. Even the stored teacher masks, supplied directly to
the *later* shared predictor as a diagnostic, score only 0.7312 for LoRA and
0.6868 for the RNN. This bounds what an exact reconstruction of those fixed
targets can achieve after predictor weights drift. The four-input
autoencoder search also recovers only 0.5722 of the training mask for full
LoRA and 0.6785 for full RNN; corresponding predicted-mask top-k recalls are
0.7904 and 0.7518. Thus both `a` instability and stale predictive targets
contribute. These probes use task IDs only for measurement, never for routing.
Training autoencoder masks also overlap across domains (0.317 of active
neurons on average for full LoRA), so later `x -> L` gradient steps can change
weights used by older routes. The scheme therefore does not guarantee that
the logit network cannot forget, even though it never trains on old data.

With a posterior-logit teacher on full LoRA, direct routing scored 0.5826;
four-label Bayesian refinement of that predicted prior scored 0.5582. The
cold-mask teacher raised direct routing to 0.5984. Freezing previously active
`x -> L` neurons made no meaningful difference in the reduced test (0.6139
versus 0.6126 with the posterior teacher). These results support useful
evaluation-time input conditioning, but do not support the stronger claim that
this no-maintenance construction recovers parent-scale training performance.
Unlike the replay-repair experiment below, it never updates old mask targets
using labeled replay.

Run configs, scalar metrics, and logs are under `artifacts/ae_*/`. Example
commands are in `README.md`.

## Parent-scale follow-up: four input rows at evaluation

This follow-up uses six tasks, one pass, `CONCEPT_SHIFT=False`, the parent's
LoRA dimensions (64 input dimensions, 1024 atoms, width 256, eight layers),
and 1500 training batches per task. The fixed-I/O recurrent model uses 500
batches per task. Full-batch change-point fitting, the original loss-based
change detector, and the original replay buffer remain active. Retrospective
evaluation provides four rows to the input router. `ICReplayRepair` routes
directly to a stored hard mask, so it uses **no evaluation labels or mask
optimization** in this condition.

Replay repair addresses the main defect of the first prior: archived masks
become stale as shared predictor weights change. After each task block, it
groups the existing replay examples by frozen rank-two input reconstruction
keys, without task IDs. For each old domain with at least 1600 replay rows, it
uses 800 labeled training rows for a fresh 100-step, three-restart mask search
and the other 800 to accept the candidate only if validation loss improves.
Only a new mask value is appended. The input projectors and predictor weights
are unchanged by repair. Thus routing cannot forget through gradient updates,
while archived masks can track predictor drift. This adds a substantial
training-memory and computation requirement; it is not four-shot learning of
the predictor mask.

| Model and seed | Training accuracy, current block | Recall, four input rows | Earlier small-support method |
| --- | ---: | ---: | ---: |
| BayesLoRA 123 | 0.9133 | 0.8948 | 0.7054 (30-step evaluation refinement) |
| BayesLoRA 124 | 0.9243 | 0.9150 | — |
| BayesLoRA 125 | 0.9272 | 0.9195 | — |
| **BayesLoRA mean** | **0.9216** | **0.9098** | — |
| BayesLoRARNN fixed I/O 123 | 0.8171 | 0.8214 | 0.7558 (256-row cold refresh) |
| BayesLoRARNN fixed I/O 124 | 0.8677 | 0.8883 | 0.7939 (256-row cold refresh) |
| BayesLoRARNN fixed I/O 125 | 0.8726 | 0.8992 | 0.8434 (256-row cold refresh) |
| **BayesLoRARNN mean** | **0.8525** | **0.8696** | **0.7977** |

The training column is the mean accuracy on each task's training block when
it was current. Recall is the mean retrospective held-out accuracy over all
tasks seen after each block. They are related but not identical populations;
recall can exceed that training average. The final routing probes matched the
correct input domain on all four trials per task for all six runs; their
reconstruction errors were about `1e-13`. In the LoRA seed-123 control, the
same training run with evaluation-only prior refinement and no repair scored
0.7054; zero-logit four-row search scored 0.5601. Across the three recurrent
seeds, replay repair improved on the 256-row cached refresh in every pair.
The full-scale maskable-I/O recurrent variant was not rerun.

The result supports the input-conditioned prior idea for this deliberately
simple domain geometry. It does not establish that four examples can find a
new deep mask: the archive is first repaired with extensive labeled *training*
replay. It also does not test semantic or concept shifts, noisy domains, or a
literal separately trained FTN autoencoder mask. The input reconstruction
router is a nonparametric linear autoencoder rather than a gradient-trained
neural router.

Metrics and logs: `artifacts/lora_full_repair_{123,124,125}/` and
`artifacts/rnn_full_repair_{123,124,125}/` on the pod. The results below are
the earlier reduced-capacity exploratory runs, so their low training scores
should not be read as a ceiling for the parent-scale models.

## Earlier reduced-capacity experiment

### Protocol

- Seeds 123–125, four tasks, two passes through the task stream.
- `CONCEPT_SHIFT=False`; each input domain is a rank-two subspace rotated into
  the common five-dimensional input space.
- BayesLoRA, BayesLoRARNN with fixed I/O, and BayesLoRARNN with maskable I/O.
- Four examples for change-point and retrospective mask search; 64 examples
  per training batch; 120 batches per task block. Replay and the parent
  Bayesian routine-update path remain enabled.
- Zero-logit and reconstruction-prior conditions both use 12 mask-search
  steps. Both use the same input reconstruction novelty trigger. Evaluation RNG
  is restored after every retrospective check to keep training streams paired.
- Reconstruction memory stores the optimal rank-two linear autoencoder of
  each observed segment, paired with its posterior mask logits. Entries are
  append-only; routing uses only the input support rows. A failed match falls
  back to the original zero-logit search.

### Retrospective recall accuracy

Mean `avg_recall_eval_acc` across the three seeds:

| Architecture | Zero-logit | Prior at change point and evaluation | Difference |
| --- | ---: | ---: | ---: |
| BayesLoRA | 0.5693 | 0.6824 | +0.1131 |
| BayesLoRARNN, fixed I/O | 0.5483 | 0.6472 | +0.0989 |
| BayesLoRARNN, maskable I/O | 0.5428 | 0.6254 | +0.0826 |

The prior won all nine paired seed/architecture comparisons. See
`artifacts/<name>/metrics_<name>.txt` for each run and `summarize.py` for the
aggregation.

### Where the gain occurs

| Architecture | Zero-logit | Prior only at training change points | Prior only at evaluation | Prior at both |
| --- | ---: | ---: | ---: | ---: |
| BayesLoRA | 0.5693 | 0.5648 | 0.6667 | 0.6824 |
| BayesLoRARNN, fixed I/O | 0.5483 | 0.5452 | 0.6257 | 0.6472 |

Most of the retrospective recall gain comes from using the retrieved prior
when inferring a mask at evaluation. A change-point prior alone has little
effect on later recovery by the *separate* zero-logit evaluation search.

The change-point prior does improve the immediate prediction after a known
domain returns. For the first change point of each returning block, the first
four rows selected/refined the mask and the remaining 60 rows were scored
before any weight update:

| Architecture | Zero-logit immediate loss | Prior immediate loss | Zero-logit immediate accuracy | Prior immediate accuracy |
| --- | ---: | ---: | ---: | ---: |
| BayesLoRA | 0.7977 | 0.6761 | 0.6333 | 0.6361 |
| BayesLoRARNN, fixed I/O | 0.8752 | 0.6696 | 0.6056 | 0.6458 |

Each number averages four returning-block measurements per seed over three
seeds. The parent loss-based change detector observes the whole batch before
reconfiguration; the 60 scored rows are unused by mask fitting, though their
labels can influence whether that detector fires. The same detector is used
in both conditions.

### Interpretation and limits

This supports the idea that an easy input reconstruction problem can retrieve
a prior for hard mask inference, including the deeper recurrent model. The
input router cannot forget old domains through weight updates because its old
reconstruction entries are frozen. The paired mask logits can still become
stale if the shared predictor's atom weights change; replay remains active.

The setup is deliberately favorable to reconstruction: noiseless rank-two
domain shifts are exactly modeled by rank-two PCA. These results do not show
that a literal FTN autoencoder mask transfers to an independently trained
predictor, nor that reconstruction would route semantic or concept shifts.
Only three seeds were run, so the table is an experiment result rather than a
precise estimate of general performance.
