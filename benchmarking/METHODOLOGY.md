## Scope and interpretation

This experiment suite evaluates the six requested FTN configurations on Split
CIFAR-10, Split CIFAR-100, Tiny-ImageNet, and the official CORe50 new-classes
stream. Numerical baselines are DER++, MNTDP-D, SPG, L2P, and DualPrompt. These
are the methods reviewers explicitly highlighted for broader empirical
comparison or as missing comparisons. PathNet and AIM are discussed below as
related routing mechanisms; numerical runs of those methods are not included
in this declared matrix.

There are 300 production runs: 15 method/backbone settings × 4 datasets × 5
seeds. A cell is final only after all five complete runs exist. Unit checks,
smoke runs, partial training, and failed runs never enter comparison tables.
An empty cell is missing evidence, not a measured zero. No rankings or claims
of superiority should be inferred from an incomplete report.

The suite addresses benchmark coverage, comparisons using native method
implementations, optimizer isolation, seed reliability, and compute reporting.
It does not establish that hard top-k masks or particular overlap thresholds
are theoretically optimal, and it is not a replacement for all routing and
replay ablations requested in the reviews.

## FTN implementations and defaults

Source snapshots under `sources/ftn/` and `sources/bayesian_adv/` are copied
from this repository, including uncommitted source changes present when the
suite was prepared. `sources/manifest.json` records their original paths and
SHA-256 digests. Production uses their model classes, loss, mask inference,
reactive detectors, replay admission, overlap filtering, and training loops.
Only benchmark input/output dimensions and the number of stream tasks replace
the synthetic data-generator dimensions. Seeds are 0–4. Graphics generation is
disabled; original training budgets and other requested configuration values
are retained. Every run writes its resolved configuration to remote storage.

| Configuration | Source selection | Bank H / active k | Training reconfiguration | Replay batches/update | LoRA rank / alpha | Updates/task |
|---|---|---|---|---:|---|---:|
| Base | `ftn/config.py: Base` | 4096 / 256 | 30 search steps per batch | 2, overlap filtered | — | 500 |
| Reactive_Low_NoReplay | `Reactive_Low NoReplay` | 4096 / 256 | Loss-triggered; 5 SD, EMA 0.9 | 0 | — | 500 |
| LoRA | `LoRA` | 4096 / 256 | 30 search steps per batch | 2, overlap filtered | 1 / 256 | 500 |
| LoRA_Reactive_Low_NoReplay | `LoRA Reactive_Low NoReplay` | 4096 / 256 | Loss-triggered; 5 SD, EMA 0.9 | 0 | 1 / 256 | 500 |
| Bayes | `bayesian_adv/config.py: Bayes` | 4096 / 256 | Persistent 2-step posterior updates; 30-step change-point refit | 1, overlap filtered | — | 500 |
| BayesLoRA | `bayesian_adv/config.py: BayesLoRA` | 4096 / 256 | Persistent 2-step posterior updates; 30-step change-point refit | 1, overlap filtered | 4 / 64 | 1500 |

The native base bank has independent eight-layer, width-eight scalar-output
MLPs. Its readout has separate columns per neuron and no shared output bias.
The native LoRA configuration has an eight-layer frozen random MLP of width
256, with independently parameterized atom banks. The four base-family
configurations use their native 60-step evaluation search; the Bayesian family
uses its native 30-step stateless evaluation search. Default batch and maximum
support size are 256. Native LoRA and BayesLoRA differ in rank, scaling, replay,
and update budget as well as inference; their comparison therefore evaluates
whole configurations and cannot isolate the causal effect of Bayesian routing.

The new Bayesian method is an approximate, persistent logit-based inference
procedure with a KL term and a hard top-k mask, not a claim of exact posterior
inference over all assemblies.

## Parameter isolation

The user explicitly required zero weight decay, Adam resets between mask
reconfigurations, and no shared trainable upstream computation. The original
training loops reset weight-optimizer state at task boundaries. The benchmark
optimizer adds the requested reset at every training reconfiguration, including
stable Bayesian posterior updates, while preserving the model's persistent
Bayesian mask state. This optimizer policy is an intentional, documented
addition to the native training loop, not an unreported change of configuration.

Every trainable tensor must have a verified atom/neuron axis. Base-bank
weights and biases have their own neuron axis, and readout columns belong to
individual neurons. LoRA A, B, and bias banks have an explicit atom axis. The
optimizer rejects any unrecognized shared trainable parameter. Dormant lateral
modules constructed by the root FTN are frozen when their configured iteration
counts are zero; they do not participate in the requested defaults, so freezing
them changes no forward computation or native weight update.

For every update, the allowed parameter set is the union of hard masks used by
the actual current and replay forwards. Weight decay is exactly zero. Adam
moments outside that union are zeroed before the step, preventing previously
replayed atoms from drifting when they are inactive on a subsequent update.
Mask search optimizes logits with bank weights detached; evaluations preserve
training optimizer state, inference state, and training RNG state.

`check_isolation.py` asserts zero inactive gradients, bitwise unchanged
inactive weights after warming Adam moments and varying replay masks, unchanged
predictions of a disjoint assembly, state reset after a real native mask search,
no weight gradients produced by that search, and no optimizer mutation during
evaluation. The same checks cover the transformer adapters. These checks are
required before FTN jobs are released. Raw assertion results remain in the
remote `validation/isolation/` directory.

## Transformer adaptation

For L2P/DualPrompt comparisons, the LoRA, LoRA Reactive_Low NoReplay, and
BayesLoRA variants are also evaluated on ViT-B/16. They keep their native atom
count, active budget, rank, initialization, scaling, reconfiguration, replay,
and training settings. Only the frozen backbone is replaced. Banks are added
to all 12 blocks' attention qkv and output projections, both MLP projections,
and the classifier: 49 insertion points sharing one routing mask.

The backbone, normalization, patch embedding, position embeddings, and
unbanked parameters are frozen. The bank is the only trainable computation.
Classifier parameters are banked too, avoiding a shared trainable head. Both
FTN and the native prompt methods use the released Google checkpoint
[ViT-B_16.npz](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz).
The checkpoint digest is recorded remotely. Pretraining is an additional
resource, so transformer results have their own table rather than being ranked
as resource-matched scratch results.

The transformer linear adaptation contracts the native equation
`sum_h m[h] * x @ A[h] @ B[h]` into an equivalent weight update. Every atom
remains differentiable to its mask logit during inference, including inactive
atoms through the native straight-through estimator. A float64 comparison
checks outputs, input gradients, atom gradients, and every mask gradient against
the original dense equation. Training/inference checks exercise all insertion
points and verify that frozen weights have no gradients.

## Data and inference protocol

Split CIFAR-10 uses five tasks of two classes. Split CIFAR-100 uses ten tasks
of ten classes; Tiny-ImageNet uses ten tasks of twenty classes. The same seeded
class order is used across methods. CORe50 uses official NC run metadata, with
ten classes in the first batch and five in each of the remaining eight batches,
and its official held-out sessions. The corresponding run index is the seed.

CIFAR uses torchvision's official archives and their reference checksums.
Tiny-ImageNet uses the distribution maintained by the Mammoth authors when
Stanford's host is unavailable, with its Hugging Face revision recorded and
100,000 training / 10,000 validation images and 200 labels checked. CORe50's
RGB archive is checked against its official MD5 before extraction; its original
paths, labels, and LUP metadata define the stream.

Ten percent of each training class is reserved for calibration/validation,
using identical indices across methods. Those samples are excluded from
weight training and replay. Official test images supply query metrics. No
test-query labels are passed to mask inference. FTN's primary class-incremental
evaluation uses one class-balanced support set pooled over seen classes,
independent of the query's task. Predictions range over all seen classes.

Separate task-incremental diagnostics provide the known class group and a
task-matched support set. A further FTN diagnostic uses pooled support with the
known class group, allowing the effect of task-matched support to be inspected
without silently comparing it with a task-agnostic baseline. The different
inference information is essential to interpreting these results.

NoReplay means no replay gradients during training; FTN still requires labeled
support at evaluation. It is not a claim of zero inference-time example memory.
The full calibration pool, support size, and native replay capacity must be
considered when comparing storage assumptions. Native replay uses the first
100 admitted batches per task and mask-overlap filtering. The harness stores
sample indices rather than duplicate transformed image tensors; deterministic
transforms reconstruct exactly the same stored inputs. Native subsampling and
replay sampling behavior are retained.

FTN MLPs consume flattened images at native dataset resolution (32, 64, or
128 pixels). Transformer evaluation uses the native prompt checkpoint's
224-pixel resize/crop and input convention. Native baselines retain their
method-specific augmentation. The FTN data adapter uses deterministic image
transforms, allowing exact indexed replay. This suite compares native recipes,
not matched augmentation, architecture, parameter count, or sample exposure.

## Baseline implementations

[DER++](https://github.com/aimagelab/mammoth), L2P, and DualPrompt use
Mammoth's implementations and complete training pipeline, including
`MammothDatasetWrapper`, `store_masked_loaders`, and `meta_*` hooks. A registered
external dataset adapter supplies the shared splits. It also converts the
upstream parser's custom-order list to the NumPy mapping expected by its
splitter. The underlying method code is unchanged.

DER++ is evaluated with its standard 500-example buffer and with a buffer
capacity matched to the full-replay FTN configurations, `100 × 256 × tasks`.
It uses a scratch ResNet-18, 50 epochs/task, batch size 32, learning rate 0.03,
alpha 0.3, and beta 0.5. L2P and DualPrompt use their released ViT-B/16 prompt
architectures, batch size 128, five epochs/task, and learning-rate input 0.03
before native batch-size scaling. Batchwise test prompting is disabled to
avoid using task-homogeneous query batches as a routing hint. These are fixed
recipes, not a claim of exhaustive per-benchmark hyperparameter tuning.

[SPG](https://github.com/UIC-Liu-Lab/SPG) uses the authors' AlexNet, training,
importance computation, and gradient modification. Its published task-
incremental protocol supplies an oracle task head. The native 32-pixel image
size, width 2048, SGD learning rate 0.25, validation scheduling, and early
stopping up to 500 epochs are preserved. Published CIFAR-100 and Tiny-ImageNet
dropout settings are reused; CIFAR-10 and CORe50 use the corresponding fixed
recipe rather than new tuning. A removed PyTorch scheduler verbosity argument
is handled in the external compatibility adapter.

[MNTDP-D](https://github.com/TomVeniat/MNTDP) uses the authors' residual blocks,
task-driven nearest-neighbor prior, candidate graph construction, deterministic
path search, and freezing/pruning. Each native candidate is trained using the
original Ignite trainer, with independent initialization copies and validation
accuracy selecting the path. Candidate execution is serial on a worker instead
of using the legacy Ray launcher. The native 32-pixel backbone, one prior source
model, Adam learning rate 0.001, maximum 10,000 candidate updates, and patience
100 are used. Oracle task identity selects the task's final path. API-name,
legacy ragged-array, and empty-array truth-value compatibility is provided in
`mntdp_compat.py`; no reduced substitute model is used.

## Metrics, statistics, and transfer analysis

For an accuracy matrix R whose row i is evaluation after task i, final ACC is
the mean of the last row across tasks. BWT is the mean over previous tasks of
`R[last,j] - R[j,j]`. Forgetting is the mean over previous tasks of the largest
post-acquisition accuracy minus final accuracy. Positive BWT means later
training improved earlier-task accuracy; lower forgetting is better. Average
incremental accuracy averages seen-task accuracy after each task block.

Five paired seeds share class orders and data partitions. Tables report mean
and sample standard deviation. Per-seed matrices and resolved recipes are
retained, so aggregate improvements can be checked against individual runs.
No significance claim is made from a one-seed or incomplete comparison.

CIFAR-100 additionally supplies its coarse-label hierarchy as a semantic
relatedness diagnostic. After training, task-matched support masks produce a
pairwise Jaccard matrix, compared descriptively with the Jaccard overlap of
tasks' coarse-category sets. This is a natural-data semantic analysis, not
ground-truth latent function sharing or a causal proof. Pairwise task entries
are dependent, so their nominal correlation p-values are not reported.

## Positioning and limits of the routing claim

PathNet searches and preserves modular paths. AIM learns input-conditioned
attention over independent mechanisms. SPG attenuates parameter gradients using
importance; MNTDP selects reusable modules under a task-driven prior. L2P and
DualPrompt select trainable prompts around a pretrained backbone. These are
substantial conceptual precedents for modularity, routing, and parameter reuse.
FTN's specific mechanism is labeled-support-conditioned selection of
overlapping atom/neuron assemblies, combined with reconfiguration and
overlap-filtered replay. Parameter masking or routing alone is not claimed as
novel.

Differences between pooled-support CIL and task-matched inference must remain
visible in the conclusions. Strong task-matched retention would not by itself
establish strong task-agnostic class-incremental recognition. Similarly,
configuration differences in rank, training duration, replay, capacity, and
pretraining prevent attributing an overall advantage to only one component.

## Reproducibility and storage

`PROVENANCE.json` pins third-party source revisions and licenses remain with
their repositories. Resolved FTN configs, baseline commands, split indices,
per-block progress, final matrices, assertions, failure logs, and final inference
checkpoints are under the shared remote benchmark root documented in
`STORAGE.md`. `manifests/environment.txt` records the runtime packages.

All three pods use the same shared data store. One locked worker per GPU claims
jobs without duplicate execution. A failed job is recorded for inspection and
is not silently rerun with a smaller batch, shorter stream, or simpler model.
The report is regenerated from complete production result files. Only this
small report and source sync locally; dataset archives, model files, package
caches, and raw logs stay outside the sync directory.
