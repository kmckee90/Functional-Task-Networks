"""
config.py - Experiment configurations for FTN experiments.

Each class is a self-contained experiment config consumed by
run_experiments.py. Attributes are read as `Config.<NAME>`; the `@property`
fields at the bottom derive values from them. FTN_PARS collects everything
the FTN class needs (backbone + mask/reconfiguration hyperparameters).

Compared to base/config.py this drops:
  - MASK_CONFIGURER      (FTN's annealed-STE KWTA is the only algorithm;
                          MASK_LATERAL_STEPS=0 gives plain KWTA, >0 gives
                          SmoothKWTA behavior, and MASK_LATERAL_POLYAK /
                          MASK_LATERAL_NOISE_SD cover the Polyak/noisy
                          variants — all inside the FTN class)
  - CONFIGURE_PER_BATCH  (replaced by RECONFIG_EVERY: per-batch, every-N, or
                          once per task block; RECONFIG_REACTIVE replaces the
                          schedule with loss-triggered reconfiguration)
  - SHUFFLE_ON_REPLAY    (replay always subsamples REPLAY_BATCH_SIZE rows once
                          at store time)
  - PHASES               (the experiment is exactly the old phase 1:
                          multi-task learning over the task stream)

v2 = base_rf + domain-shift options, reconfiguration scheduling (periodic /
per-block / reactive), and the SmoothKWTA mask parameters. Replay stays
available (N_REPLAY_SAMPLES=0 disables it). No weight freezing.

Parameter glossary
==================

Architecture
------------
ARCHITECTURE         Backbone family:
                     "base" (default) - the ParallelMLP FTN in ftn.py (compute
                       all z, mask selects a subspace).
                     "lora" - FTN-LoRA in lora_ftn.py (one frozen random MLP
                       backbone, mask selects k of H trainable LoRA atoms).
                     "stack" / "stack_residual" - the stacked-mask FTN in
                       stack_ftn.py (chain N ParallelMLP blocks, one mask per
                       layer: x -> z0*m0 -> ... -> readout). "stack_residual"
                       adds a linear skip from every masked layer to the output.
                     "rnn" / "rnn_residual" - the same N blocks as ONE
                       weight-tied ParallelMLP cell unrolled over N recurrent
                       steps (a linear input encoder + a shared readout), so it
                       is one FTN activating a different mask each step.
                     "hml" - the higher-order multilinear FTN in hml_ftn.py: a
                       distinct input population tanh(Wx) seeds the recurrence,
                       and a weight-tied recurrent population unrolled HML_DEPTH
                       steps replaces feed-forward depth with per-neuron low-rank
                       (HML_RANK) multilinear interactions up to HML_MAX_ORDER.
                       A linear decoder reads every masked level (depth+1 masks)
                       and their outputs are summed.
TASK                 Which data-generation task the experiment runs:
                     "sine"    - tasks.DataGenSine (the original sine task).
                     "compose" - task_comp.DataGenComposition (function
                       composition on binary matrices; each task is a distinct
                       fixed operator sequence). Both follow the same
                       continual-learning convention: N_TASK tasks, T samples
                       per batch, data_gen(task_idx) -> (x, y).

Experiment
----------
SEED                 Base RNG seed; local torch.Generators are seeded off it.

The loss mode is per-task, not a shared knob: the compose task is always
per-cell binary cross-entropy (the BINARY property returns TASK=="compose");
the sine task uses CLASSIFICATION (softmax cross-entropy) or MSE.

Data generator: shared (both TASK values)
-----------------------------------------
N_TASK               Number of distinct tasks in the task stream.
T                    Samples per generated batch.

Data generator: sine (TASK == "sine", tasks.DataGenSine)
--------------------------------------------------------
CLASSIFICATION       Single-label softmax cross-entropy on one-hot targets vs
                     scalar/vector regression (MSE) when False. Sine-only.
X_DIM, Y_DIM         Input / output dimensions.
TASK_SIZE            Hidden dims allocated per task (controls task overlap).
FREQUENCY            Sine frequency; higher = harder targets.
DATA_TEMP            Label-sampling softmax temperature (classification).
CORRELATED_TASKS     Tasks share hidden dims via pairwise combinations.
CONCEPT_SHIFT        Each task uses a different hidden subset.
DOMAIN_SHIFT_FACTOR  Per-task input mean shift; 0 = shared input distribution.
DOMAIN_SHIFT         Dimensional domain shift: widen the input vector so each
                     task's X_DIM values occupy their own slice (total width
                     (N_TASK-1)*DOMAIN_SHIFT_STRIDE + X_DIM, zeros elsewhere).
                     The task function itself is unchanged; the model input
                     size follows the widened vector automatically.
DOMAIN_SHIFT_STRIDE  Element-index shift between consecutive tasks' slices.
                     0 = concept shift (all tasks share the same elements);
                     >= X_DIM = fully disjoint (orthogonal) input supports.
DOMAIN_DENSE_PROJECTION
                     Apply a fixed random orthogonal projection (same width)
                     to the final input vector: the sparse domain-shifted
                     inputs become dense (no zero elements, so no trivially
                     zero-gradient input weights) while preserving their
                     information — a rotation of the domain.
DOMAIN_PER_TASK_PROJECTION
                     If True, N_TASK distinct projections (one per task);
                     if False, a single shared projection for all tasks.

Data generator: compose (TASK == "compose", task_comp.DataGenComposition)
-------------------------------------------------------------------------
Each of the N_TASK tasks is a distinct, fixed composition of COMP_STEPS
operators (rot90_cw / flip_lr / shift_right) applied to random binary
COMP_MATRIX_DIM x COMP_MATRIX_DIM matrices; the target is the transformed
matrix, always trained as per-cell binary classification (one sigmoid per
cell). Input/output dim = COMP_MATRIX_DIM**2.
COMP_MATRIX_DIM      Matrix side length N.
COMP_STEPS           Operators per composition (composition depth), e.g. 3 =
                     "rotate -> flip -> shift". Distinct tasks get distinct
                     length-COMP_STEPS operator words.
COMP_P               Bernoulli probability for each input matrix entry being 1.
COMP_SHIFT           Columns used by the shift_right operator.
COMP_REJECT_IDENTITY Skip words whose net transform is the identity.
COMP_REJECT_REDUCIBLE
                     Skip words equivalent to a shorter operator sequence.
COMP_REQUIRE_UNIQUE_TRANSFORM
                     Skip words whose global transform is realized by another
                     length-COMP_STEPS word (keeps every task unambiguous).

FTN backbone
------------
SQUARE_DIM           Mask-grid side length; HIDDEN_SIZE = SQUARE_DIM**2.
LAYERS               ParallelMLP depth.
INNER_DIM            ParallelMLP inner width.
DROPOUT              Dropout on backbone features.
LATERAL_B            Backbone feature-smoothing strength; 0 disables mixing.
LATERAL_STEPS        Backbone feature-smoothing iterations; 0 disables.
KERNEL_SIZE          Backbone lateral conv kernel size.
TORUS                Circular padding for lateral convolutions.

FTN mask / reconfiguration
--------------------------
K                    kWTA budget: number of active mask units.
RECONFIG_STEPS       Gradient steps per mask reconfiguration.
RECONFIG_LR          Adam lr for the mask logits.
RECONFIG_SAMPLE_SIZE Max samples used per reconfiguration.
RECONFIG_NOISE_SCALE annealste only: multiplier on the annealed exploration
                     noise added to the logits inside kwta(logits + noise) at
                     each search step. 1.0 = current behavior; >1 explores more
                     aggressively; 0 = greedy kwta with no exploration noise.
                     Applies to the base FTN, stacked/recurrent, and LoRA
                     annealste searches; the "fast" algorithm has no such noise.
RECONFIG_EVERY       Reconfigure the mask every N training batches (1 = every
                     batch). 0 = reconfigure once at the start of each task
                     block (the base/ CONFIGURE_PER_BATCH=False behavior);
                     backwards eval then also reconfigures once per task
                     instead of once per eval batch.
RECONFIG_REACTIVE    Replace the RECONFIG_EVERY schedule with loss-triggered
                     reconfiguration: reconfig once on the first training
                     batch, then track an EMA of the batch-loss mean and
                     variance and reconfig whenever the loss under the
                     current mask exceeds mean + RECONFIG_REACTIVE_SD * sd.
                     Boundary-free (state never resets at task switches).
                     Training only; eval keeps the reconfig protocol.
RECONFIG_REACTIVE_SD Trigger threshold: standard deviations above the EMA
                     mean the batch loss must reach to trigger a reconfig.
RECONFIG_REACTIVE_EMA
                     Smoothing factor for the loss mean/variance EMA (higher
                     = slower). Triggers are suppressed for ~1/(1-EMA)
                     batches after each reconfig while the statistics re-form.
MASK_KERNEL_SIZE     Mask lateral-smoothing kernel size (SmoothKWTA).
MASK_LATERAL_STEPS   Mask lateral-smoothing iterations; 0 = plain KWTA,
                     >0 = SmoothKWTA (smooth, then re-sparsify with kWTA).
MASK_LATERAL_POLYAK  Polyak mixing of pre/post-smoothing masks per lateral
                     step (0 = pure smoothing; ~0.95 = PolyakKWTA variants;
                     1.0 skips smoothing entirely).
MASK_LATERAL_NOISE_SD
                     Gaussian noise added per mask lateral-smoothing step.
RECONFIG_MATCH_STORED
                     annealste only. After RECONFIG_MATCH_STORED_AFTER gradient
                     steps, replace the mask search's running best with the
                     closest task mask in the replay / noise-probe buffer (max
                     overlap) and make its loss the bar the remaining steps must
                     beat. Small AFTER = warm-start from a stored mask;
                     AFTER >= RECONFIG_STEPS = end anchored to a stored mask.
                     Needs a populated buffer (replay or noise probe) — no-op
                     when empty. Applies to the base FTN, the stacked/recurrent
                     simultaneous search, and FTN-LoRA; not the greedy
                     sequential search or the DualMask second mask.
RECONFIG_MATCH_STORED_AFTER
                     Gradient steps taken before the stored-mask swap.
RECONFIG_MATCH_STORED_TOPK
                     Hopfield recall breadth. 1 (default) snaps onto the single
                     highest-overlap stored mask. k>1 evaluates the k
                     highest-overlap stored masks on the current batch and snaps
                     onto the best-performing one (lowest loss) instead of just
                     the closest match. Clamped to the buffer size.
RECONFIG_MATCH_TIE_STEPS
                     Recall granularity for match-stored, stack/recurrent
                     architectures only. True (default): stored masks span all
                     recurrent steps and are recalled as one concatenated whole —
                     a complete deep N-step solution is recovered as a unit.
                     False: split each stored mask into per-step square_dim**2
                     masks and pool them across steps into one shared bank, then
                     recall each step independently from that bank (a separate
                     best per step, TOPK applied per step). Because the recurrent
                     cell is weight-tied and reused across steps, a subspace
                     stored as one task's step-3 mask can be recalled as this
                     run's step 0/1/2, so masks act as stable, reusable per-step
                     functions that compose rather than fixed full compositions.

Second mask (DualMask)
----------------------
USE_SECOND_MASK      Enable two-stage masking. mask1 is reconfigured normally,
                     then mask2 is reconfigured against mask2*mask1 with mask1
                     frozen; the effective mask applied by the network is
                     mask1*mask2. False = single mask (default).
K_2, RECONFIG_ALGORITHM_2, RECONFIG_STEPS_2, RECONFIG_LR_2,
RECONFIG_SAMPLE_SIZE_2, MASK_KERNEL_SIZE_2, MASK_LATERAL_STEPS_2,
MASK_LATERAL_POLYAK_2, MASK_LATERAL_NOISE_SD_2, TORUS_2
                     The second mask's twin of each mask-1 knob, so mask2 can
                     use its own algorithm ("fast"/"annealste"), budget, K,
                     lateral dynamics, kernel size and torus. Ignored when
                     USE_SECOND_MASK is False.

FTN-LoRA (ARCHITECTURE == "lora")
---------------------------------
A plain feed-forward MLP whose weights are randomly initialized and FROZEN
(never trained). Only the LoRA atom bank - H = SQUARE_DIM**2 atoms on every
linear - is trained; the mask selects K active atoms (K / RECONFIG_* are shared
with the base FTN). The fixed backbone lets you sweep random weight scales while
the FTN-LoRA does all the learning.
LORA_LAYERS          Number of hidden layers in the frozen MLP.
LORA_WIDTH           Width of each hidden layer.
LORA_WEIGHT_SCALE    Frozen backbone init std (N(0, scale/sqrt(fan_in))); the
                     fixed weight scale to sweep.
LORA_RANK            Rank of each LoRA atom.
LORA_ALPHA           LoRA scaling; effective delta scaled by alpha/K.
LORA_DROPOUT         Dropout on the LoRA path.
LORA_BIAS            Per-atom LoRA bias deltas (mask-gated, summed like the
                     weight delta); False = no bias anywhere.
LORA_B0_SCALE        B0 init scale for the atoms' B matrices (a reconfiguration
                     parameter): B must be nonzero so an untrained atom has a
                     nonzero gradient and can be selected by the mask search.
                     0 would make untrained atoms unreachable.
RECONFIG_INIT_SD     Std of the mask-logit init for the search (shared by the
                     LoRA and stacked-mask searches).

Stacked / recurrent-mask FTN (ARCHITECTURE in {stack, stack_residual, rnn,
rnn_residual})
--------------------------------------------------------------------------
Chain STACK_LAYERS ParallelMLP blocks (each the base per-unit backbone, reusing
LAYERS / INNER_DIM / DROPOUT), masking every layer's population:
x -> z0*m0 -> ... -> z_{L-1}*m_{L-1} -> readout. Each layer is square_dim**2
units with its own kWTA-k mask (K / RECONFIG_* / MASK_* shared with the base
FTN). "stack" reads out from the last masked layer only; "stack_residual" adds a
linear skip from every masked layer to the output and sums them. "rnn" /
"rnn_residual" are the same but the STACK_LAYERS blocks are ONE weight-tied
ParallelMLP cell unrolled over STACK_LAYERS steps (a linear input encoder lifts
x into the hidden space; a single shared readout decodes it).
STACK_LAYERS         Chained blocks (recurrent steps when rnn/rnn_residual).
STACK_SEQUENTIAL     Residual variants only: reconfigure the masks greedily one
                     layer at a time (m0, then m1|m0, ...) instead of all at once
                     by a single optimizer.
RNN_STEP_RMSNORM     rnn/rnn_residual only: apply a shared learnable RMSNorm to
                     the input of each recurrent step EXCEPT the first - it
                     normalizes the fed-back hidden state z_{t-1}*m_{t-1} but
                     leaves the initial embed(x) untouched - to keep the
                     weight-tied cell's state from drifting/vanishing across
                     steps. Ignored for the non-recurrent variants.
RNN_RECURRENT_MASK   rnn/rnn_residual only: reconfigure by searching ONLY the
                     first-step mask m_0 (annealed-STE / fast as usual, and it
                     may be recalled from the stored buffer if RECONFIG_MATCH_
                     STORED is on). The remaining steps' masks are produced by a
                     LEARNED recurrent transition m_{t+1} = kWTA(W m_t + b)
                     (straight-through, top-k), whose weights train jointly with
                     the cell/readout rather than being searched. So each
                     batch/eval the search picks an entry point and the mask
                     self-propagates, selecting the operation from the neuron
                     pool at each step. Initialized near identity (start by
                     reusing the same subspace, learn to deviate). Incompatible
                     with STACK_SEQUENTIAL; ignored for the non-recurrent
                     variants. RECONFIG_MATCH_TIE_STEPS still selects the recall
                     bank for m_0 (each task's first step vs. all steps pooled).
STACK_DETACH_MASK_DEEP_PATH
                     Residual variants only: detach each masked population where
                     it feeds the NEXT block, so a mask's RECONFIGURATION gradient
                     reaches the output only through that layer's own linear
                     readout skip, not the deeper nonlinear blocks it also gates.
                     Forward values are identical either way; no-op for the
                     non-residual variants.
STACK_CONCAT_READOUT Residual output topology. False (default) = accumulative:
                     one shared readout summed across steps, out = W*sum_l s_l.
                     True = concatenated (like HML): a distinct readout per step,
                     summed, out = sum_l W_l s_l (== one readout over the
                     concatenation [s_0,...,s_{L-1}]). The stacked-residual case
                     already uses per-layer readouts, so this only changes the
                     recurrent-residual (rnn_residual) case; no-op otherwise.

HML FTN (ARCHITECTURE == "hml")
-------------------------------
A distinct input population z0 = tanh(input_proj(x)) (square_dim**2 neurons)
seeds the recurrence, then a weight-tied recurrent population (an HMLCell) is
unrolled HML_DEPTH steps. Each recurrent neuron reads the whole masked
population s_l and computes a first-order (linear) term plus rank-R multilinear
interactions up to HML_MAX_ORDER, then tanh - no shared cell matrix, every
projection private to a neuron. There are HML_DEPTH+1 masked states s_0..s_depth
(one kWTA-k mask each, K / RECONFIG_* / MASK_* shared with the base FTN); a
distinct linear decoder reads each and the outputs are summed (plus a bias).
HML_DEPTH             Recurrent applications of the HMLCell; masks = depth+1.
HML_RANK              Rank R of the per-neuron multilinear interactions.
HML_MAX_ORDER         Highest interaction order P (2 = one quadratic term; 1
                      collapses the cell to a per-neuron tanh-linear map).

Data sizes / schedule
---------------------
TRAIN_DURATION       Training steps (batches) per task per epoch.
N_EPOCHS             Epochs over the task stream.
N_SAMPLE_PER_TASK    Batches per task retained for replay; feeds REPLAY_MAXLEN.
N_EVAL_EPOCHS        Evaluation epochs.
N_REPLAY_SAMPLES     Stored batches replayed per step (0 disables replay).
REPLAY_BATCH_SIZE    Rows kept per stored replay batch (subsampled once at
                     store time); None keeps the full batch.

Noise probe (replay-free structure preservation; alternative to replay)
-----------------------------------------------------------------------
NOISE_PROBE          Enable the noise-probe strategy. Structurally identical to
                     replay, with two substitutions: the replayed input is
                     random NOISE, and the target is the network's OWN response
                     to that noise, generated and FROZEN at the end of each task
                     block (before any later task can overwrite the shared
                     neurons; dropout is off during generation for a clean
                     target). Each step then selects stored probes whose mask
                     overlaps the current mask in (NOISE_PROBE_OVERLAP_LOW,
                     NOISE_PROBE_OVERLAP_HIGH] and replays them under the stored
                     (full) mask: pred = f(noise; stored_mask), loss =
                     MSE(pred, frozen_output). Because the target was captured
                     while the task's function still lived in the weights, the
                     gradient pulls the shared neurons back toward the old
                     function instead of a target that drifts with the
                     overwrite. Only masks + small frozen noise batches are
                     stored, never task data. Independent of replay; leave
                     replay disabled when using it (see the NoiseProbe config).
NOISE_PROBE_SAMPLE_PER_TASK
                     Frozen noise batches generated per task at block end; feeds
                     NOISE_PROBE_MAXLEN (analogous to N_SAMPLE_PER_TASK).
NOISE_PROBE_SAMPLES  Stored probes replayed (gradient-augmented) per training
                     step.
NOISE_PROBE_OVERLAP_LOW, NOISE_PROBE_OVERLAP_HIGH
                     Keep stored probes whose fractional overlap with the current
                     mask (|stored & current| / |current|) falls in (LOW, HIGH].
NOISE_PROBE_DISTRIBUTION
                     Noise input distribution: "gaussian" (unit normal) or
                     "uniform" (centered on 0).
NOISE_PROBE_SCALE    Noise scale: Gaussian std, or uniform half-width (draws in
                     [-SCALE, SCALE]).
NOISE_PROBE_BATCH    Number of noise rows per frozen probe batch.

Backwards / prospective eval (forgetting monitor)
-------------------------------------------------
EVAL_BACKWARDS       Re-evaluate all previously seen tasks after each task.
EVAL_PROSPECTIVE     Also evaluate not-yet-trained tasks (forward transfer).
N_RETRO_EVAL_BATCHES Batches per task during backwards eval.

Entropy tests (post-hoc out-of-distribution subspace probe)
-----------------------------------------------------------
ENTROPY_TESTS        Enable the post-hoc ENTROPY_TESTS pass that runs once after
                     run_experiments completes. For every ordered task pair
                     (i, j) it reconfigures on a fresh task-i batch to recover
                     task i's mask, then drives task j's inputs through the
                     network UNDER task i's mask and scores the output against
                     task j's target — probing how the subspace trained on task
                     i behaves out of its trained input distribution. Emits three
                     matrices (rows = mask task i, cols = input task j):
                     entropy_acc (mean accuracy), entropy_std (across-trial
                     accuracy std. dev.), and entropy_share (mean fraction of task
                     i's K active neurons also active in task j's mask). Dropout
                     stays ENABLED during output generation (model kept in
                     train() mode) so dropout can be explored as an OOD-entropy
                     lever. Off by default.
ENTROPY_TEST_TRIALS  Trials averaged for the accuracy / std. dev. / sharing
                     matrices. Each trial reconfigures a fresh task-i mask on a
                     new task-i batch and draws fresh task-j data, so the
                     std. dev. matrix reflects mask + data resampling.

Training / plotting
-------------------
TRAIN_LR   Weight-optimizer hyperparameters.
LOSS_EMA_SMOOTHING       EMA factor for the plotted loss curve.
RENDER_STEPS             Render per-training-step mask visualizations (mask
                         GIFs, task vis). False skips those; loss plots and
                         the forgetting (transfer) matrices are always saved.
PLOT_EVERY               Plot/log every N epochs.
GIF_DURATION             Per-frame duration (ms) for GIFs.
GIF_SAVE_EPOCH           Epoch interval at which GIFs are saved.
GIF_EMA_SCALE            Time-constant scale for the blue/purple activity trails
                         in bw_*.gif. The per-pixel EMA decays each step at
                         decay**(1/GIF_EMA_SCALE) on the red and blue channels
                         (the purple mid-trail and blue tail); the fast green
                         channel (bright head) is unaffected. 1.0 = original
                         rates (0.95 red / 0.975 blue), >1 = slower decay and
                         longer trails, <1 = faster decay and shorter trails.
"""


class Default:
    """Default experiment configuration (classification, 6 correlated tasks)."""

    # Architecture
    ARCHITECTURE = "base"
    TASK = "sine"            # "sine" (tasks.DataGenSine) | "compose" (task_comp)

    # Experiment pars
    SEED = 123

    # ---- Data generator: shared (both TASK values) ----
    N_TASK = 8               # number of continual-learning tasks
    T = 256                  # samples generated per task batch

    # ---- Data generator: sine (TASK == "sine") ----
    # CLASSIFICATION=True -> single-label softmax cross-entropy (one-hot);
    # False -> MSE regression. Sine-only; compose is always per-cell binary
    # cross-entropy (see the BINARY property, driven by TASK).
    CLASSIFICATION = True
    X_DIM = 2
    Y_DIM = 2
    TASK_SIZE = 4
    FREQUENCY = 8
    DATA_TEMP = 1e-8
    CORRELATED_TASKS = True
    CONCEPT_SHIFT = True

    DOMAIN_SHIFT_FACTOR = 0
    DOMAIN_SHIFT = False
    DOMAIN_SHIFT_STRIDE = 0
    DOMAIN_DENSE_PROJECTION = False
    DOMAIN_PER_TASK_PROJECTION = False

    # ---- Data generator: compose (TASK == "compose") ----
    # Each task is a distinct fixed length-COMP_STEPS operator composition
    # (rot90_cw / flip_lr / shift_right) on binary COMP_MATRIX_DIM x
    # COMP_MATRIX_DIM matrices; target = transformed matrix, always trained as
    # per-cell binary classification (the BINARY property is True for compose).
    COMP_MATRIX_DIM = 8      # matrix side N; input/output dim = N**2
    COMP_STEPS = 3           # operators per composition ("rotate->flip->shift")
    COMP_P = 0.25            # Bernoulli prob for input matrix entries
    COMP_SHIFT = 1           # columns used by shift_right
    COMP_REJECT_IDENTITY = True
    COMP_REJECT_REDUCIBLE = True
    COMP_REQUIRE_UNIQUE_TRANSFORM = True

    # FTN backbone
    SQUARE_DIM = 64
    LAYERS = 8
    INNER_DIM = 8

    DROPOUT = 0.0

    LATERAL_B = 0.0
    LATERAL_STEPS = 0
    KERNEL_SIZE = 3
    TORUS = True

    # FTN mask / reconfiguration
    K = 256

    RECONFIG_ALGORITHM = "annealste"
    RECONFIG_STEPS = 30
    RECONFIG_LR = 0.3
    RECONFIG_SAMPLE_SIZE = T
    RECONFIG_NOISE_SCALE = 1.0

    RECONFIG_EVERY = 1

    RECONFIG_REACTIVE = False
    RECONFIG_REACTIVE_SD = 5.0
    RECONFIG_REACTIVE_EMA = 0.95

    MASK_KERNEL_SIZE = 3
    MASK_LATERAL_STEPS = 0
    MASK_LATERAL_POLYAK = 0.0
    MASK_LATERAL_NOISE_SD = 0.0

    # Match-stored reconfiguration
    RECONFIG_MATCH_STORED = False
    RECONFIG_MATCH_STORED_AFTER = 15
    RECONFIG_MATCH_STORED_TOPK = 1
    RECONFIG_MATCH_TIE_STEPS = True

    # Second mask (DualMask)
    USE_SECOND_MASK = False
    K_2 = 64
    RECONFIG_ALGORITHM_2 = "annealste"
    RECONFIG_STEPS_2 = 30
    RECONFIG_LR_2 = 0.3
    RECONFIG_SAMPLE_SIZE_2 = T
    MASK_KERNEL_SIZE_2 = 3
    MASK_LATERAL_STEPS_2 = 0
    MASK_LATERAL_POLYAK_2 = 0.0
    MASK_LATERAL_NOISE_SD_2 = 0.0
    TORUS_2 = False

    # FTN-LoRA (ARCHITECTURE == "lora")
    LORA_LAYERS = 4
    LORA_WIDTH = 64
    LORA_WEIGHT_SCALE = 1.0
    LORA_RANK = 4
    LORA_ALPHA = 16.0
    LORA_DROPOUT = 0.0
    LORA_BIAS = True
    LORA_B0_SCALE = 0.01
    RECONFIG_INIT_SD = 1e-5

    # Stacked / recurrent-mask FTN
    STACK_LAYERS = 4
    STACK_SEQUENTIAL = False
    RNN_STEP_RMSNORM = False
    RNN_RECURRENT_MASK = False     # recurrent only; search m_0, propagate the rest
    STACK_DETACH_MASK_DEEP_PATH = False
    STACK_CONCAT_READOUT = False   # residual only; rnn_residual: per-step readouts

    # HML FTN (ARCHITECTURE == "hml")
    HML_DEPTH = 4       # recurrent applications of the HMLCell; masks = depth+1
    HML_RANK = 4        # rank R of the per-neuron multilinear interactions
    HML_MAX_ORDER = 2   # highest interaction order P (2 = one quadratic term)

    # Data sizes
    TRAIN_DURATION = 500
    N_EPOCHS = 1
    N_EVAL_EPOCHS = 1

    N_SAMPLE_PER_TASK = 100
    N_REPLAY_SAMPLES = 4
    REPLAY_BATCH_SIZE = T

    # Noise probe (replay-free structure preservation)
    NOISE_PROBE = False
    NOISE_PROBE_SAMPLE_PER_TASK = 100
    NOISE_PROBE_SAMPLES = 1
    NOISE_PROBE_OVERLAP_LOW = 0.2
    NOISE_PROBE_OVERLAP_HIGH = 0.8
    NOISE_PROBE_DISTRIBUTION = "gaussian"
    NOISE_PROBE_SCALE = 1.0
    NOISE_PROBE_BATCH = T

    # Backwards eval (forgetting monitor)
    EVAL_BACKWARDS = True
    EVAL_PROSPECTIVE = True
    N_RETRO_EVAL_BATCHES = 10

    # Entropy tests (post-hoc OOD subspace probe; see ENTROPY_TESTS in glossary)
    ENTROPY_TESTS = True
    ENTROPY_TEST_TRIALS = 10

    # Training hyperpars
    TRAIN_LR = 3e-4

    # Plotting
    RENDER_STEPS = True
    LOSS_EMA_SMOOTHING = 0.8
    PLOT_EVERY = 1
    GIF_DURATION = 20
    GIF_SAVE_EPOCH = 1
    GIF_EMA_SCALE = 0.0

    @property
    def BINARY(self):
        """Loss mode is per-task: the compose task is always per-cell binary
        cross-entropy; sine uses CLASSIFICATION (softmax CE) or MSE."""
        return self.TASK == "compose"

    @property
    def HIDDEN_SIZE(self):
        return self.SQUARE_DIM**2

    @property
    def REPLAY_MAXLEN(self):
        return self.N_SAMPLE_PER_TASK * self.N_TASK

    @property
    def NOISE_PROBE_MAXLEN(self):
        return self.NOISE_PROBE_SAMPLE_PER_TASK * self.N_TASK

    @property
    def FTN_PARS(self):
        """Everything the FTN class takes besides input/output/hidden sizes."""
        return {
            "square_dim": self.SQUARE_DIM,
            "classification": self.CLASSIFICATION,
            "binary": self.BINARY,
            # backbone
            "layers": self.LAYERS,
            "inner_size": self.INNER_DIM,
            "dropout_p": self.DROPOUT,
            "lateral_b": self.LATERAL_B,
            "lateral_steps": self.LATERAL_STEPS,
            "lateral_kernel_size": self.KERNEL_SIZE,
            # mask / reconfiguration
            "k": self.K,
            "reconfig_steps": self.RECONFIG_STEPS,
            "reconfig_lr": self.RECONFIG_LR,
            "reconfig_sample_size": self.RECONFIG_SAMPLE_SIZE,
            "reconfig_algorithm": self.RECONFIG_ALGORITHM,
            "reconfig_noise_scale": self.RECONFIG_NOISE_SCALE,
            "mask_kernel_size": self.MASK_KERNEL_SIZE,
            "mask_lateral_steps": self.MASK_LATERAL_STEPS,
            "mask_lateral_polyak": self.MASK_LATERAL_POLYAK,
            "mask_lateral_noise_sd": self.MASK_LATERAL_NOISE_SD,
            "torus": self.TORUS,
            "match_stored": self.RECONFIG_MATCH_STORED,
            "match_stored_after": self.RECONFIG_MATCH_STORED_AFTER,
            "match_stored_topk": self.RECONFIG_MATCH_STORED_TOPK,
            # second mask (DualMask)
            "use_second_mask": self.USE_SECOND_MASK,
            "k2": self.K_2,
            "reconfig_steps2": self.RECONFIG_STEPS_2,
            "reconfig_lr2": self.RECONFIG_LR_2,
            "reconfig_sample_size2": self.RECONFIG_SAMPLE_SIZE_2,
            "reconfig_algorithm2": self.RECONFIG_ALGORITHM_2,
            "mask_kernel_size2": self.MASK_KERNEL_SIZE_2,
            "mask_lateral_steps2": self.MASK_LATERAL_STEPS_2,
            "mask_lateral_polyak2": self.MASK_LATERAL_POLYAK_2,
            "mask_lateral_noise_sd2": self.MASK_LATERAL_NOISE_SD_2,
            "torus2": self.TORUS_2,
        }

    @property
    def LORA_PARS(self):
        """Everything LoRAFTN takes besides input/output/hidden sizes.
        Reuses the shared K / RECONFIG_* / MASK_* knobs so the LoRA mask search
        is configured exactly like the base FTN's."""
        return {
            "square_dim": self.SQUARE_DIM,
            "classification": self.CLASSIFICATION,
            "binary": self.BINARY,
            # frozen backbone MLP
            "n_layers": self.LORA_LAYERS,
            "width": self.LORA_WIDTH,
            "weight_scale": self.LORA_WEIGHT_SCALE,
            # LoRA atoms
            "rank": self.LORA_RANK,
            "alpha": self.LORA_ALPHA,
            "dropout_p": self.LORA_DROPOUT,
            "b0_scale": self.LORA_B0_SCALE,
            "use_bias": self.LORA_BIAS,
            # reconfiguration (mask search)
            "k": self.K,
            "reconfig_steps": self.RECONFIG_STEPS,
            "reconfig_lr": self.RECONFIG_LR,
            "reconfig_sample_size": self.RECONFIG_SAMPLE_SIZE,
            "reconfig_algorithm": self.RECONFIG_ALGORITHM,
            "reconfig_noise_scale": self.RECONFIG_NOISE_SCALE,
            "reconfig_init_sd": self.RECONFIG_INIT_SD,
            "match_stored": self.RECONFIG_MATCH_STORED,
            "match_stored_after": self.RECONFIG_MATCH_STORED_AFTER,
            "match_stored_topk": self.RECONFIG_MATCH_STORED_TOPK,
            "mask_kernel_size": self.MASK_KERNEL_SIZE,
            "mask_lateral_steps": self.MASK_LATERAL_STEPS,
            "mask_lateral_polyak": self.MASK_LATERAL_POLYAK,
            "mask_lateral_noise_sd": self.MASK_LATERAL_NOISE_SD,
            "torus": self.TORUS,
        }

    @property
    def STACK_PARS(self):
        """Everything StackFTN takes besides input/output/hidden sizes and the
        residual flag (which run_experiments derives from ARCHITECTURE). Reuses
        the base backbone knobs (LAYERS/INNER_DIM/DROPOUT) per block and the
        shared K / RECONFIG_* / MASK_* mask-search knobs."""
        return {
            "square_dim": self.SQUARE_DIM,
            "classification": self.CLASSIFICATION,
            "binary": self.BINARY,
            # topology
            "n_stack": self.STACK_LAYERS,
            "sequential": self.STACK_SEQUENTIAL,
            "rnn_step_rmsnorm": self.RNN_STEP_RMSNORM,
            "rnn_recurrent_mask": self.RNN_RECURRENT_MASK,
            "detach_mask_deep_path": self.STACK_DETACH_MASK_DEEP_PATH,
            "concat_readout": self.STACK_CONCAT_READOUT,
            # per-block backbone (reused base knobs)
            "layers": self.LAYERS,
            "inner_size": self.INNER_DIM,
            "dropout_p": self.DROPOUT,
            # reconfiguration (mask search)
            "k": self.K,
            "reconfig_steps": self.RECONFIG_STEPS,
            "reconfig_lr": self.RECONFIG_LR,
            "reconfig_sample_size": self.RECONFIG_SAMPLE_SIZE,
            "reconfig_algorithm": self.RECONFIG_ALGORITHM,
            "reconfig_noise_scale": self.RECONFIG_NOISE_SCALE,
            "reconfig_init_sd": self.RECONFIG_INIT_SD,
            "match_stored": self.RECONFIG_MATCH_STORED,
            "match_stored_after": self.RECONFIG_MATCH_STORED_AFTER,
            "match_stored_topk": self.RECONFIG_MATCH_STORED_TOPK,
            "match_tie_steps": self.RECONFIG_MATCH_TIE_STEPS,
            "mask_kernel_size": self.MASK_KERNEL_SIZE,
            "mask_lateral_steps": self.MASK_LATERAL_STEPS,
            "mask_lateral_polyak": self.MASK_LATERAL_POLYAK,
            "mask_lateral_noise_sd": self.MASK_LATERAL_NOISE_SD,
            "torus": self.TORUS,
        }

    @property
    def HML_PARS(self):
        """Everything HMLFTN takes besides input/output/hidden sizes. Reuses the
        shared K / RECONFIG_* / MASK_* mask-search knobs so the HML mask search
        is configured exactly like the base FTN's."""
        return {
            "square_dim": self.SQUARE_DIM,
            "classification": self.CLASSIFICATION,
            "binary": self.BINARY,
            # topology
            "depth": self.HML_DEPTH,
            "rank": self.HML_RANK,
            "max_order": self.HML_MAX_ORDER,
            "dropout_p": self.DROPOUT,
            # reconfiguration (mask search)
            "k": self.K,
            "reconfig_steps": self.RECONFIG_STEPS,
            "reconfig_lr": self.RECONFIG_LR,
            "reconfig_sample_size": self.RECONFIG_SAMPLE_SIZE,
            "reconfig_algorithm": self.RECONFIG_ALGORITHM,
            "reconfig_noise_scale": self.RECONFIG_NOISE_SCALE,
            "reconfig_init_sd": self.RECONFIG_INIT_SD,
            "match_stored": self.RECONFIG_MATCH_STORED,
            "match_stored_after": self.RECONFIG_MATCH_STORED_AFTER,
            "match_stored_topk": self.RECONFIG_MATCH_STORED_TOPK,
            "mask_kernel_size": self.MASK_KERNEL_SIZE,
            "mask_lateral_steps": self.MASK_LATERAL_STEPS,
            "mask_lateral_polyak": self.MASK_LATERAL_POLYAK,
            "mask_lateral_noise_sd": self.MASK_LATERAL_NOISE_SD,
            "torus": self.TORUS,
        }

    @property
    def COMP_PARS(self):
        """Everything task_comp.DataGenComposition takes besides the shared
        N_TASK / T / rng / seed (which run_experiments passes directly)."""
        return {
            "matrix_dim": self.COMP_MATRIX_DIM,
            "steps": self.COMP_STEPS,
            "p": self.COMP_P,
            "shift": self.COMP_SHIFT,
            "classify": self.BINARY,
            "reject_identity": self.COMP_REJECT_IDENTITY,
            "reject_reducible": self.COMP_REJECT_REDUCIBLE,
            "require_unique_transform": self.COMP_REQUIRE_UNIQUE_TRANSFORM,
        }


class Compose(Default):
    """Continual-learning function-composition task (task_comp.DataGenComposition):
    N_TASK distinct fixed operator compositions on binary matrices. The target
    is the transformed binary matrix, always trained as per-cell binary
    classification (the BINARY property is True whenever TASK=="compose", so one
    sigmoid per cell). Sine-specific knobs (CLASSIFICATION, X_DIM, ...) unused."""
    TASK = "compose"
    N_TASK = 8
    COMP_MATRIX_DIM = 8
    COMP_STEPS = 3


class Smoke(Default):
    """Tiny end-to-end configuration for fast sanity checks."""
    N_TASK = 3
    SQUARE_DIM = 8
    LAYERS = 2
    K = 16
    RECONFIG_STEPS = 5
    T = 64
    TRAIN_DURATION = 5
    N_SAMPLE_PER_TASK = 2
    N_RETRO_EVAL_BATCHES = 2


class EntropyTests(Default):
    """Mixin: enable the post-hoc ENTROPY_TESTS OOD subspace probe. Compose onto
    any run, e.g. --config Default EntropyTests, or --config Smoke EntropyTests."""
    ENTROPY_TESTS = True
    ENTROPY_TEST_TRIALS = 10

class ConceptShift(Default):
    DOMAIN_SHIFT = False



class DomainShift(Default):
    DOMAIN_SHIFT_FACTOR = 0
    DOMAIN_SHIFT = True
    DOMAIN_SHIFT_STRIDE = 1
    DOMAIN_DENSE_PROJECTION = True
    DOMAIN_PER_TASK_PROJECTION = True

class Replay(Default):
    N_SAMPLE_PER_TASK = 100
    N_REPLAY_SAMPLES = 4

class NoReplay(Default):
    N_SAMPLE_PER_TASK = 1
    N_REPLAY_SAMPLES = 0
    REPLAY_BATCH_SIZE = 1


class MatchStored(Default):
    """Match-stored annealste reconfiguration on top of the default replay buffer.

    After RECONFIG_MATCH_STORED_AFTER search steps, snap the running best onto
    the closest previous-task mask in the buffer and let the rest try to beat it.
    Small AFTER warm-starts from a stored mask; AFTER >= RECONFIG_STEPS ends
    anchored to one. Compose with NoiseProbe to source the masks from the
    noise-probe buffer instead of replay."""
    RECONFIG_MATCH_STORED = True
    RECONFIG_MATCH_STORED_AFTER = 15

class Reactive_Low(Default):
    RECONFIG_EVERY = 0
    RECONFIG_REACTIVE = True
    RECONFIG_REACTIVE_SD = 5.0
    RECONFIG_REACTIVE_EMA = 0.9

class Reactive_Med(Default):
    RECONFIG_EVERY = 0
    RECONFIG_REACTIVE = True
    RECONFIG_REACTIVE_SD = 3.0
    RECONFIG_REACTIVE_EMA = 0.9

class Reactive_High(Default):
    RECONFIG_EVERY = 0
    RECONFIG_REACTIVE = True
    RECONFIG_REACTIVE_SD = 1.0
    RECONFIG_REACTIVE_EMA = 0.9


class PerBatch(Default):
    RECONFIG_EVERY = 1
    RECONFIG_REACTIVE = False

class PerTask(Default):
    RECONFIG_EVERY = 0
    RECONFIG_REACTIVE = False

class PolyakSmooth(Default):
    MASK_LATERAL_STEPS = 2  # 0 = plain KWTA; >0 = SmoothKWTA
    MASK_LATERAL_POLYAK = 0.9
    MASK_LATERAL_NOISE_SD = 0.0
    MASK_KERNEL_SIZE = 5
    RECONFIG_STEPS = 30
    RECONFIG_LR = 0.3
    DROPOUT = 0.0
    RECONFIG_ALGORITHM = "annealste"
    TORUS = True

class Smooth(Default):
    MASK_LATERAL_STEPS = 2  # 0 = plain KWTA; >0 = SmoothKWTA
    MASK_LATERAL_POLYAK = 0.0
    MASK_LATERAL_NOISE_SD = 0.0
    MASK_KERNEL_SIZE = 17
    RECONFIG_STEPS = 1
    RECONFIG_LR = 1.0
    DROPOUT = 0.0
    RECONFIG_ALGORITHM = "fast"
    TORUS = True

class Fast(Default):
    RECONFIG_STEPS = 4
    RECONFIG_LR = 0.3
    DROPOUT = 0.0
    RECONFIG_ALGORITHM = "fast"

class Flash(Default):
    RECONFIG_STEPS = 1
    RECONFIG_LR = 1.0
    # DROPOUT = 0.0
    RECONFIG_ALGORITHM = "fast"


class Large(Default):
    SQUARE_DIM = 96
    K = 1024

class PolyakSmoothLarge(Default):
    MASK_LATERAL_STEPS = 1  # 0 = plain KWTA; >0 = SmoothKWTA
    MASK_LATERAL_POLYAK = 0.9
    MASK_LATERAL_NOISE_SD = 0.0
    MASK_KERNEL_SIZE = 17
    RECONFIG_STEPS = 30
    RECONFIG_LR = 0.3
    DROPOUT = 0.0
    RECONFIG_ALGORITHM = "annealste"
    TORUS = True


class DualMask(Default):
    TORUS = True
    K = 1024
    RECONFIG_ALGORITHM = "fast"
    RECONFIG_STEPS = 1
    RECONFIG_LR = 1
    RECONFIG_EVERY = 1

    # RECONFIG_REACTIVE = False
    # RECONFIG_REACTIVE_SD = 5.0
    # RECONFIG_REACTIVE_EMA = 0.95

    MASK_KERNEL_SIZE = 21
    MASK_LATERAL_STEPS = 3  # 0 = plain KWTA; >0 = SmoothKWTA
    MASK_LATERAL_POLYAK = 0.0
    MASK_LATERAL_NOISE_SD = 0.0

    USE_SECOND_MASK = True
    K_2 = 64
    RECONFIG_ALGORITHM_2 = "annealste"  # annealste or fast
    RECONFIG_STEPS_2 = 60
    RECONFIG_LR_2 = 0.3
    # RECONFIG_SAMPLE_SIZE_2 = T
    MASK_KERNEL_SIZE_2 = 3
    MASK_LATERAL_STEPS_2 = 0  # 0 = plain KWTA; >0 = SmoothKWTA
    MASK_LATERAL_POLYAK_2 = 0.0
    MASK_LATERAL_NOISE_SD_2 = 0.0
    TORUS_2 = True

class LoRA(Default):
    ARCHITECTURE = "lora"
    # FTN-LoRA (used only when ARCHITECTURE == "lora")
    # ------------------------------------------------
    # A plain feed-forward MLP whose weights are randomly initialized and FROZEN
    # (never trained). Only the LoRA atom bank — H = SQUARE_DIM**2 atoms on every
    # linear — is trained; the mask selects K active atoms (K/RECONFIG_* above are
    # shared with the base FTN). The point of the fixed backbone is to experiment
    # with different random weight scales while the FTN-LoRA does all the learning.
    LORA_LAYERS = 8           # number of hidden layers in the frozen MLP
    LORA_WIDTH = 256           # width of each hidden layer
    LORA_WEIGHT_SCALE = 1.0  # frozen backbone init std (N(0, scale/sqrt(fan_in)));
                              # the fixed weight scale to sweep
    LORA_RANK = 1             # rank of each LoRA atom
    LORA_ALPHA = 256.0         # LoRA scaling; effective delta scaled by alpha/K
    LORA_DROPOUT = 0.0
    # B0 init scale for the atoms' B matrices (a reconfiguration parameter):
    # B must be nonzero so an untrained atom has a nonzero gradient and can be
    # selected by the mask search. 0 would make untrained atoms unreachable.
    LORA_B0_SCALE = 0.01
    RECONFIG_INIT_SD = 1e-5   # std of the mask-logit init for the search

    N_EPOCHS = 1

    RECONFIG_ALGORITHM = "annealste"
    RECONFIG_STEPS = 30
    RECONFIG_LR = 0.3

class Base(Default):
    ARCHITECTURE="base"


class Stack(Default):
    """Stacked-mask FTN: chain STACK_LAYERS ParallelMLP blocks, one mask per
    layer, readout from the last masked layer only."""
    ARCHITECTURE = "stack"
    STACK_LAYERS = 4


class StackResidual(Default):
    """Stacked-mask FTN with a linear skip from every masked layer to the output
    (summed). Simultaneous reconfiguration (all masks by one optimizer)."""
    ARCHITECTURE = "stack_residual"
    STACK_LAYERS = 4
    STACK_SEQUENTIAL = False


class StackResidualSequential(StackResidual):
    """stack_residual with greedy per-layer reconfiguration (m0, then m1|m0, ...)."""
    STACK_SEQUENTIAL = True


class RNN(Default):
    """Recurrent-mask FTN: one weight-tied ParallelMLP cell unrolled STACK_LAYERS
    steps, a different mask per step, readout from the last step only."""
    ARCHITECTURE = "rnn"
    STACK_LAYERS = 4


class RNNResidual(Default):
    """Recurrent-mask FTN with a shared skip readout summed over every step.
    Simultaneous reconfiguration (all per-step masks by one optimizer)."""
    ARCHITECTURE = "rnn_residual"
    STACK_LAYERS = 4
    STACK_SEQUENTIAL = False


class RNNResidualSequential(RNNResidual):
    """rnn_residual with greedy per-step reconfiguration (m0, then m1|m0, ...)."""
    STACK_SEQUENTIAL = True


class RNNResidualConcat(RNNResidual):
    """rnn_residual with HML-style concatenated output: a distinct readout per
    step summed (out = sum_l W_l s_l), instead of one shared readout summed."""
    STACK_CONCAT_READOUT = True



class Atomic(Default):
    # FTN backbone
    SQUARE_DIM = 96
    K = 1024
    LAYERS = 8
    INNER_DIM = 8
    DROPOUT = 0.0

class Thorough(Default):
    RECONFIG_ALGORITHM = "annealste"
    RECONFIG_STEPS = 60
    RECONFIG_LR = 0.3

class Thorough2(Default):
    RECONFIG_ALGORITHM = "annealste"
    RECONFIG_STEPS = 120
    RECONFIG_LR = 0.3

class Thorough3(Default):
    RECONFIG_ALGORITHM = "annealste"
    RECONFIG_STEPS = 240
    RECONFIG_LR = 0.3

class FastAnneal(Default):
    RECONFIG_ALGORITHM = "annealste"
    RECONFIG_STEPS = 10
    RECONFIG_LR = 0.3
    RECONFIG_NOISE_SCALE=0.1


class Stack_Resid(Default):
    ARCHITECTURE="stack_residual"
    LAYERS = 4
    STACK_LAYERS = 2          # number of chained ParallelMLP+mask blocks
    STACK_SEQUENTIAL = False  # stack_residual only: greedy per-layer reconfig

class Stack(Default):
    ARCHITECTURE="stack"
    LAYERS = 2
    STACK_LAYERS = 4          # number of chained ParallelMLP+mask blocks
    STACK_SEQUENTIAL = False  # stack_residual only: greedy per-layer reconfig


class RNN_Resid(Default):
    ARCHITECTURE="rnn_residual"
    LAYERS = 2
    STACK_LAYERS = 4          # number of chained ParallelMLP+mask blocks
    STACK_SEQUENTIAL = True  # stack_residual only: greedy per-layer reconfig

class RNN(Default):
    ARCHITECTURE="rnn"
    LAYERS = 4
    STACK_LAYERS = 2          # number of chained ParallelMLP+mask blocks
    STACK_SEQUENTIAL = False  # stack_residual only: greedy per-layer reconfig



class RNN_Resid_1x8_seq(Default):
    ARCHITECTURE="rnn_residual"
    LAYERS = 1
    STACK_LAYERS = 8          # number of chained ParallelMLP+mask blocks
    STACK_SEQUENTIAL = True  # stack_residual only: greedy per-layer reconfig

class RNN_Resid_1x8(Default):
    ARCHITECTURE="rnn_residual"
    LAYERS = 1
    STACK_LAYERS = 8          # number of chained ParallelMLP+mask blocks
    STACK_SEQUENTIAL = False  # stack_residual only: greedy per-layer reconfig

class RNN_Resid_2x4_seq(Default):
    ARCHITECTURE="rnn_residual"
    LAYERS = 2
    STACK_LAYERS = 4          # number of chained ParallelMLP+mask blocks
    STACK_SEQUENTIAL = True  # stack_residual only: greedy per-layer reconfig

class RNN_Resid_2x4(Default):
    ARCHITECTURE="rnn_residual"
    LAYERS = 2
    STACK_LAYERS = 4          # number of chained ParallelMLP+mask blocks
    STACK_SEQUENTIAL = False  # stack_residual only: greedy per-layer reconfig



class RecurrentMask(Default):
    """Mixin (compose onto an rnn/rnn_residual config): search only the first
    mask and derive the rest from a learned recurrent transition. E.g.
      --config RNN_Resid_1x8 RecurrentMask Reactive_Low ..."""
    RNN_RECURRENT_MASK = True

class TieStepsOff(Default):
    """Mixin: pool stored per-step masks and recall each step independently in
    the match-stored search (RECONFIG_MATCH_TIE_STEPS=False). Compose after a
    Hopfield/match-stored config, e.g. --config RNN_Resid_2x4 Hopfield2 TieStepsOff."""
    RECONFIG_MATCH_TIE_STEPS = False


class RNN_1x8(Default):
    ARCHITECTURE="rnn"
    LAYERS = 1
    STACK_LAYERS = 8          # number of chained ParallelMLP+mask blocks
    STACK_SEQUENTIAL = False  # stack_residual only: greedy per-layer reconfig

class RNN_2x4(Default):
    ARCHITECTURE="rnn"
    LAYERS = 2
    STACK_LAYERS = 4          # number of chained ParallelMLP+mask blocks
    STACK_SEQUENTIAL = False  # stack_residual only: greedy per-layer reconfig

class DropoutVeryHigh(Default):
    DROPOUT = 0.9
class DropoutHigh(Default):
    DROPOUT = 0.5
class DropoutMed(Default):
    DROPOUT = 0.3
class DropoutLow(Default):
    DROPOUT = 0.1
class DropoutZero(Default):
    DROPOUT = 0.0

class Detach(Default):
    STACK_DETACH_MASK_DEEP_PATH = True

class LongTrain(Default):
    TRAIN_DURATION = 3000


class NoiseProbe(Default):
    """Replay-free noise-probe structure preservation (replaces replay).

    Replay is disabled; at each task's end the network's response to random
    noise is frozen under that task's mask, and later steps replay those probes
    (under the stored full mask) whenever it overlaps the current mask. See
    NOISE_PROBE_* in the glossary and snapshot_noise_probes()/noise_probe_step()."""
    # Replay off (noise probe is the alternative).
    N_SAMPLE_PER_TASK = 0
    N_REPLAY_SAMPLES = 0
    REPLAY_BATCH_SIZE = 0
    # Noise probe on.
    NOISE_PROBE = True
    NOISE_PROBE_SAMPLE_PER_TASK = 4
    NOISE_PROBE_SAMPLES = 4
    NOISE_PROBE_OVERLAP_LOW = 0.2
    NOISE_PROBE_OVERLAP_HIGH = 0.8
    NOISE_PROBE_DISTRIBUTION = "gaussian"
    NOISE_PROBE_SCALE = 4.0
    NOISE_PROBE_BATCH = 256

class NoiseProbe_Uniform(NoiseProbe):
    NOISE_PROBE_DISTRIBUTION = "uniform"
    NOISE_PROBE_SCALE = 1.0


class Hopfield0(Default):
    RECONFIG_MATCH_STORED = True
    RECONFIG_MATCH_STORED_AFTER = 3
    RECONFIG_MATCH_STORED_TOPK = 3
    RECONFIG_MATCH_TIE_STEPS = True

    RECONFIG_ALGORITHM = "fast"
    RECONFIG_STEPS = 6
    RECONFIG_LR = 0.3
    RECONFIG_NOISE_SCALE=0.0

class Hopfield1(Default):
    RECONFIG_MATCH_STORED = True
    RECONFIG_MATCH_STORED_AFTER = 10
    RECONFIG_MATCH_STORED_TOPK = 5
    RECONFIG_MATCH_TIE_STEPS = True

    RECONFIG_ALGORITHM = "annealste"
    RECONFIG_STEPS = 30
    RECONFIG_LR = 0.1
    RECONFIG_NOISE_SCALE=0.01

class Hopfield2(Default):
    RECONFIG_MATCH_STORED = True
    RECONFIG_MATCH_STORED_AFTER = 30
    RECONFIG_MATCH_STORED_TOPK = 5
    RECONFIG_MATCH_TIE_STEPS = True

    RECONFIG_ALGORITHM = "fast"
    RECONFIG_STEPS = 35
    RECONFIG_LR = 0.3
    RECONFIG_NOISE_SCALE=0.0

class Hopfield3(Default):
    RECONFIG_MATCH_STORED = True
    RECONFIG_MATCH_STORED_AFTER = 30
    RECONFIG_MATCH_STORED_TOPK = 10000
    RECONFIG_MATCH_TIE_STEPS = True

    RECONFIG_ALGORITHM = "annealste"
    RECONFIG_STEPS = 40
    RECONFIG_LR = 0.1
    RECONFIG_NOISE_SCALE=0.01


class Hopfield_Flash(Default):
    RECONFIG_MATCH_STORED = True
    RECONFIG_MATCH_STORED_AFTER = 1
    RECONFIG_MATCH_STORED_TOPK = 3
    RECONFIG_MATCH_TIE_STEPS = True

    RECONFIG_ALGORITHM = "fast"
    RECONFIG_STEPS = 2
    RECONFIG_LR = 1.0
    RECONFIG_NOISE_SCALE=0.00

class Compose(Default):
    TASK = "compose"            # "sine" (tasks.DataGenSine) | "compose" (task_comp)
    COMP_MATRIX_DIM = 6      # matrix side N; input/output dim = N**2
    COMP_STEPS = 4           # operators per composition ("rotate->flip->shift")
    COMP_P = 0.25            # Bernoulli prob for input matrix entries
    COMP_SHIFT = 1           # columns used by shift_right
    COMP_REJECT_IDENTITY = True
    COMP_REJECT_REDUCIBLE = True
    COMP_REQUIRE_UNIQUE_TRANSFORM = True


class HML(Default):
    """Higher-order multilinear FTN: a distinct input population (tanh(Wx)), a
    weight-tied recurrent population whose neurons use per-neuron low-rank
    multilinear interactions instead of feed-forward depth, and a linear decoder
    off every masked recurrence level summed to the output."""
    ARCHITECTURE = "hml"
    HML_DEPTH = 4
    HML_RANK = 4
    HML_MAX_ORDER = 2
    STACK_DETACH_MASK_DEEP_PATH=False

class Small(Default):
    SQUARE_DIM = 8
    K = 4



class HML_Flat(Default):
    """Higher-order multilinear FTN: a distinct input population (tanh(Wx)), a
    weight-tied recurrent population whose neurons use per-neuron low-rank
    multilinear interactions instead of feed-forward depth, and a linear decoder
    off every masked recurrence level summed to the output."""
    ARCHITECTURE = "hml"
    HML_DEPTH = 1
    HML_RANK = 1
    HML_MAX_ORDER = 2
    STACK_DETACH_MASK_DEEP_PATH=False

