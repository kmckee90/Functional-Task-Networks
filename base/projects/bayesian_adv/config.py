"""
config.py - Experiment configurations for the bayesian FTN subproject.

Pared down from base/config.py to exactly this subproject's scope:
  - base FTN, independent-mask LoRA FTN, and persistent-logit BayesFTN,
  - one task: the sine task (tasks.DataGenSine), classification only,
  - a single mask, plain kWTA (no lateral smoothing of features or mask),
  - replay buffer (LoRA can optionally recall stored masks during search).

Each class is a self-contained config read as `Config.<NAME>`. Compose classes
on the command line, left to right, later overriding earlier:
    uv run python run_experiments.py --config Default NoReplay Reactive_Low
The @property fields derive values; constructor and evaluation parameters are
bundled in FTN_PARS, BAYES_FTN_PARS, and EVAL_FTN_PARS.


Parameter glossary
==================

Experiment
----------
SEED                 Base RNG seed; local torch.Generators are seeded off it.
ARCHITECTURE         Which backbone + mask mechanism to build:
                       "base"  - the original FTN (plain per-batch mask search),
                       "lora"  - a frozen MLP with independently searched LoRA
                                 atom masks,
                       "bayes" - the ParallelMLP backbone with persistent
                                 approximate Bayesian mask inference,
                       "bayes_lora" - a frozen MLP + LoRA atom bank under the
                                 same Bayesian mask (LORA_* knobs),
                       "bayes_rnn_residual" - a weight-tied recurrent-residual
                                 cell with a per-step Bayesian mask (N_STACK /
                                 RNN_STEP_RMSNORM knobs),
                       "bayes_lora_rnn" - a weight-tied recurrent LoRA cell whose
                                 per-step Bayesian mask re-uses / re-sequences a
                                 single shared pool of low-rank atoms (LORA_* +
                                 N_STACK knobs).
                     The "bayes*" variants share the BAYES_* posterior
                     hyperparameters and differ only in the backbone.

Data generator (tasks.DataGenSine)
----------------------------------
N_TASK               Number of continual-learning tasks in the stream.
T                    Samples generated per task batch.
CLASSIFICATION       Single-label softmax cross-entropy on one-hot targets
                     (always True in this subproject).
X_DIM, Y_DIM         Input / output dimensions.
TASK_SIZE            Hidden dims allocated per task (controls task overlap).
FREQUENCY            Sine frequency; higher = harder targets.
DATA_TEMP            Label-sampling softmax temperature (classification).
CORRELATED_TASKS     Tasks share hidden dims via pairwise combinations.
CONCEPT_SHIFT        Each task uses a different hidden subset.
DOMAIN_SHIFT         Dimensional domain shift: widen the input so each task's
                     X_DIM values occupy their own slice of a wider vector
                     (width (N_TASK-1)*STRIDE + X_DIM, zeros elsewhere). The
                     task function itself is unchanged; the model input size
                     follows the widened vector automatically.
DOMAIN_SHIFT_FACTOR  Per-task input mean shift; 0 = shared input distribution.
DOMAIN_SHIFT_STRIDE  Element-index shift between consecutive tasks' slices.
                     0 = concept shift (all tasks share the same elements);
                     >= X_DIM = fully disjoint (orthogonal) input supports.
DOMAIN_DENSE_PROJECTION
                     Apply a fixed random orthogonal projection (same width) to
                     the final input vector: the sparse domain-shifted inputs
                     become dense while preserving information (a rotation).
DOMAIN_PER_TASK_PROJECTION
                     True = N_TASK distinct projections (one per task);
                     False = a single shared projection for all tasks.

FTN backbone
------------
SQUARE_DIM           Mask-grid side length; HIDDEN_SIZE = SQUARE_DIM**2.
LAYERS               ParallelMLP depth.
INNER_DIM            ParallelMLP inner width.
DROPOUT              Dropout on backbone features.

FTN mask / reconfiguration
--------------------------
K                    kWTA budget: number of active mask units.
RECONFIG_ALGORITHM   Mask search: "annealste" (annealed straight-through kWTA)
                     or "fast" (RMSProp on the relaxed mask, hard-kWTA at end).
RECONFIG_STEPS       Gradient steps per mask reconfiguration.
RECONFIG_LR          Learning rate for the mask logits.
RECONFIG_SAMPLE_SIZE Max samples used per reconfiguration.
RECONFIG_NOISE_SCALE annealste only: multiplier on the annealed exploration
                     noise added inside kwta(logits + noise). 1.0 = default;
                     >1 explores more; 0 = greedy kWTA, no exploration noise.
RECONFIG_EVERY       Reconfigure the mask every N training batches (1 = every
                     batch). 0 = reconfigure once at the start of each task
                     block; backwards eval then also reconfigures once per task.
RECONFIG_REACTIVE    Replace the RECONFIG_EVERY schedule with loss-triggered
                     change-point handling. Base FTN reconfigures only on
                     triggers. BayesFTN updates on stable batches and resets on
                     triggers. LoRA can also update on stable batches when
                     LORA_REACTIVE_CONTINUOUS_UPDATES is enabled. Task
                     switches are never disclosed to the detector.
RECONFIG_REACTIVE_SD Trigger threshold: standard deviations above the EMA mean
                     the batch loss must reach to trigger a reconfig.
RECONFIG_REACTIVE_EMA
                     Smoothing factor for the loss mean/variance EMA (higher =
                     slower). Triggers are suppressed for ~1/(1-EMA) batches
                     after each detected change point while statistics re-form.
LORA_WARM_START_RECONFIG
                     Carry the selected mask-search logits into the next LoRA
                     training batch. Evaluation still starts from zero logits.
LORA_REACTIVE_CONTINUOUS_UPDATES
                     Run LoRA's ordinary search on stable reactive batches.
LORA_CHANGEPOINT_RECONFIG_STEPS
                     Cold-search steps when the reactive detector fires.

Bayesian mask distribution (ARCHITECTURE="bayes")
-------------------------------------------------
BAYES_RECONFIG_STEPS Variational posterior steps per reconfiguration. Use at
                     least 2 for the KL gradient to act within the same call.
BAYES_RECONFIG_LR    Learning rate for the persistent mask logits.
BAYES_KL_WEIGHT      Strength of mean Bernoulli KL(q_new || q_previous).
BAYES_TEMPERATURE    Temperature for sigmoid relaxation and Gumbel top-k.
BAYES_PRIOR_DECAY    Prior diffusion rho: prior_logits=rho*previous_logits.
                     1.0 strictly carries the preceding posterior forward.
BAYES_STOCHASTIC_MASK
                     True samples exact-k masks with Gumbel top-k; False uses
                     deterministic top-k for a controlled ablation.
BAYES_CHANGEPOINT_RECONFIG_STEPS
BAYES_CHANGEPOINT_RECONFIG_LR
BAYES_CHANGEPOINT_RECONFIG_SAMPLE_SIZE
BAYES_CHANGEPOINT_KL_WEIGHT
BAYES_CHANGEPOINT_TEMPERATURE
BAYES_CHANGEPOINT_PRIOR_DECAY
BAYES_CHANGEPOINT_STOCHASTIC_MASK
                     Alternate posterior-update parameters used only when the
                     reactive loss detector fires. Defaults perform a 30-step
                     reset from zero logits with no KL, then install the result
                     as the new persistent posterior.

Data sizes / schedule
---------------------
TRAIN_DURATION       Training batches per task per epoch.
N_EPOCHS             Epochs over the task stream.
N_EVAL_EPOCHS        Evaluation epochs.

Replay
------
N_SAMPLE_PER_TASK    Batches per task retained for replay; feeds REPLAY_MAXLEN.
N_REPLAY_SAMPLES     Stored batches replayed per step (0 disables replay).
REPLAY_BATCH_SIZE    Rows kept per stored replay batch (subsampled once at store
                     time); None keeps the full batch.
REPLAY_OVERLAP_LOW, REPLAY_OVERLAP_HIGH
                     Replay a stored mask when its overlap with the current
                     mask, |stored & current| / |current|, lies in (LOW, HIGH].
                     Shared by basic FTN and Bayesian FTN variants.

Backwards / prospective eval (forgetting monitor)
-------------------------------------------------
EVAL_BACKWARDS       Re-evaluate all previously seen tasks after each task.
EVAL_PROSPECTIVE     Also evaluate not-yet-trained tasks (forward transfer).
N_RETRO_EVAL_BATCHES Batches per task during backwards eval.
EVAL_RECONFIG_STEPS  Stateless mask-search steps per support set.
EVAL_RECONFIG_LR     Evaluation-only mask-logit learning rate.
EVAL_RECONFIG_SAMPLE_SIZE
                     Maximum support rows used; None uses the entire batch.
EVAL_RECONFIG_TEMPERATURE
                     Temperature for the sigmoid surrogate and Gumbel top-k.
EVAL_RECONFIG_RESTARTS
                     Independent zero-logit searches; the deterministic top-k
                     result with the lowest support loss is retained.
                     Evaluation always uses fresh logits/optimizer, no KL, and
                     never reads or updates BayesFTN's persistent logits.

Entropy tests (post-hoc OOD subspace probe)
-------------------------------------------
ENTROPY_TESTS        Enable the post-hoc OOD subspace probe that runs once after
                     training (see entropy_tests in run_experiments.py).
ENTROPY_TEST_TRIALS  Trials averaged for the accuracy / std / sharing matrices.

Training
--------
TRAIN_LR             Weight-optimizer learning rate.

Plotting / logging
------------------
RENDER_STEPS         Render per-training-step mask visualizations (mask GIFs,
                     task vis). False skips those; loss plots and the forgetting
                     (transfer) matrices are always saved.
LOSS_EMA_SMOOTHING   EMA factor for the plotted loss curve.
PLOT_EVERY           Plot/log every N epochs.
GIF_DURATION         Per-frame duration (ms) for GIFs.
GIF_SAVE_EPOCH       Epoch interval at which GIFs are saved.
GIF_EMA_SCALE        Time-constant scale for the activity trails in bw_*.gif
                     (>1 = slower decay / longer trails, <1 = shorter).

Derived (@property)
-------------------
HIDDEN_SIZE          SQUARE_DIM**2.
REPLAY_MAXLEN        N_SAMPLE_PER_TASK * N_TASK.
FTN_PARS             Shared/base FTN constructor parameters.
BAYES_FTN_PARS       BayesFTN-only constructor parameters.
EVAL_FTN_PARS        Stateless evaluation reconfigurer parameters.


Sub-configs (compose onto Default)
==================================
Base / Bayes          Select the original or persistent-distribution FTN.
BayesLoRA             Bayesian mask over a frozen-MLP + LoRA atom-bank backbone.
BayesRNNResidual      Bayesian per-step mask over a recurrent-residual cell.
BayesLoRARNN          Recurrent LoRA: N per-step masks re-using a shared atom pool.
Smoke                Tiny end-to-end configuration for fast sanity checks.
EntropyTests         Mixin: enable the OOD subspace probe (on by default here).
DomainShift          Dimensional domain shift with a per-task dense rotation.
InputConditioned     Append-only input reconstruction router with replay mask
                     repair; compose after Bayes, BayesLoRA, or BayesLoRARNN.
InputConditionedSmoke  Small replay-search budget for a Smoke path check.
NoConceptShift       Keep the target function shared across input domains.
Replay / NoReplay    Replay amount (some replay / replay disabled).
PerTask              Reconfigure once per task block (schedule, not reactive).
Reactive_Low/Med/High  Loss-triggered reconfiguration at SD 5 / 3 / 1.
Fast / Flash         Fast RMSProp mask search (4 steps / 1 step).
Thorough             annealste with 60 mask-search steps.
Large / Small        Larger / smaller network (SQUARE_DIM, K).
LongTrain            3000 training batches per task.
"""


class Default:
    """Default: 8 correlated sine classification tasks, per-batch reconfig."""

    # Experiment
    SEED = 123
    ARCHITECTURE = "base"

    # ---- Data generator (tasks.DataGenSine) ----
    N_TASK = 8
    T = 256

    CLASSIFICATION = True
    X_DIM = 2
    Y_DIM = 2
    TASK_SIZE = 4
    FREQUENCY = 8
    DATA_TEMP = 1e-8
    CORRELATED_TASKS = True
    CONCEPT_SHIFT = True

    DOMAIN_SHIFT = False
    DOMAIN_SHIFT_FACTOR = 0
    DOMAIN_SHIFT_STRIDE = 0
    DOMAIN_DENSE_PROJECTION = False
    DOMAIN_PER_TASK_PROJECTION = False

    # ---- FTN backbone ----
    SQUARE_DIM = 64
    LAYERS = 8
    INNER_DIM = 8
    DROPOUT = 0.0

    # ---- FTN mask / reconfiguration ----
    K = 256
    RECONFIG_ALGORITHM = "annealste"
    RECONFIG_STEPS = 5
    RECONFIG_LR = 0.3
    RECONFIG_SAMPLE_SIZE = T
    RECONFIG_NOISE_SCALE = 1.0
    # Training-search cost against the frozen mask at search entry. Cold eval
    # remains stateless. lambda=0 preserves the original search exactly.
    RECONFIG_OVERLAP_MODE = "bump"  # "bump" or "gated"
    RECONFIG_OVERLAP_LAMBDA = 0.0
    RECONFIG_OVERLAP_THRESHOLD = 0.5
    RECONFIG_OVERLAP_TEMPERATURE = 0.2

    RECONFIG_EVERY = 1
    RECONFIG_REACTIVE = False
    RECONFIG_REACTIVE_SD = 5.0
    RECONFIG_REACTIVE_EMA = 0.95
    LORA_WARM_START_RECONFIG = False
    LORA_REACTIVE_CONTINUOUS_UPDATES = False
    LORA_CHANGEPOINT_RECONFIG_STEPS = 30

    # ---- Bayesian mask distribution (used only by ARCHITECTURE="bayes") ----
    BAYES_RECONFIG_STEPS = 3
    BAYES_RECONFIG_LR = 0.3
    BAYES_RECONFIG_TAKE_BEST = True
    BAYES_KL_WEIGHT = 1.0
    BAYES_TEMPERATURE = 1.0
    BAYES_PRIOR_DECAY = 1.0
    BAYES_STOCHASTIC_MASK = True

    BAYES_CHANGEPOINT_RECONFIG_STEPS = 30
    BAYES_CHANGEPOINT_RECONFIG_LR = 0.3
    BAYES_CHANGEPOINT_RECONFIG_SAMPLE_SIZE = None
    BAYES_CHANGEPOINT_KL_WEIGHT = 0.0
    BAYES_CHANGEPOINT_TEMPERATURE = 1.0
    BAYES_CHANGEPOINT_PRIOR_DECAY = 0.0
    BAYES_CHANGEPOINT_STOCHASTIC_MASK = True

    # ---- FTN-LoRA backbone (ARCHITECTURE="lora" or "bayes_lora") ----
    # A frozen random MLP plus a bank of HIDDEN_SIZE LoRA atoms on every linear;
    # the Bayesian mask selects K of the atoms. Reuses the shared K / RECONFIG_*
    # mask-search knobs so the atom mask is configured like the base FTN's.
    LORA_LAYERS = 8          # frozen backbone MLP depth
    LORA_WIDTH = 256          # frozen backbone MLP width
    LORA_WEIGHT_SCALE = 1.0  # fixed scale of the frozen backbone weights
    LORA_RANK = 1            # rank r of each LoRA atom
    LORA_ALPHA = 256.0        # LoRA scaling (alpha/K on the summed delta)
    LORA_DROPOUT = 0.0       # dropout inside each LoRA linear
    LORA_BIAS = True         # per-atom bias bank
    LORA_B0_SCALE = 0.01     # atom-reachability init scale (B / bias bank)
    RECONFIG_INIT_SD = 1e-5
    RECONFIG_MATCH_STORED = False
    RECONFIG_MATCH_STORED_AFTER = 0
    RECONFIG_MATCH_STORED_TOPK = 1
    MASK_KERNEL_SIZE = 3
    MASK_LATERAL_STEPS = 0
    MASK_LATERAL_POLYAK = 0.0
    MASK_LATERAL_NOISE_SD = 0.0
    TORUS = False

    # ---- Recurrent-residual FTN backbone (ARCHITECTURE="bayes_rnn_residual") ----
    # One weight-tied ParallelMLP cell unrolled N_STACK steps with a per-step
    # kWTA mask and residual skips; the per-step masks are concatenated in
    # self.m (total HIDDEN_SIZE*N_STACK). K is a per-step budget. The cell reuses
    # the base backbone knobs (LAYERS / INNER_DIM / DROPOUT).
    N_STACK = 4              # recurrent steps / per-step masks
    RNN_STEP_RMSNORM = False # shared RMSNorm on the fed-back state (steps > 0)

    # Recurrent-LoRA read-in / read-out treatment (ARCHITECTURE="bayes_lora_rnn").
    # "fixed": frozen random encoder + readout, no atoms (all mask-selectable
    #          capacity lives in the shared recurrent atom pool).
    # "ftn":   add a maskable LoRA atom bank to the encoder and the readout too,
    #          gated by the step-0 / per-step masks, so a task also selects its
    #          read-in / read-out edits.
    LORARNN_IO = "fixed"

    # ---- Data sizes / schedule ----
    TRAIN_DURATION = 500
    N_EPOCHS = 1
    N_EVAL_EPOCHS = 1

    # ---- Replay ----
    N_SAMPLE_PER_TASK = 100
    N_REPLAY_SAMPLES = 1
    REPLAY_BATCH_SIZE = T
    REPLAY_OVERLAP_LOW = 0.2
    REPLAY_OVERLAP_HIGH = 0.8

    # ---- Backwards / prospective eval (forgetting monitor) ----
    EVAL_BACKWARDS = True
    EVAL_PROSPECTIVE = True
    N_RETRO_EVAL_BATCHES = 30
    EVAL_RECONFIG_STEPS = 30
    EVAL_RECONFIG_LR = 0.3
    EVAL_RECONFIG_SAMPLE_SIZE = None
    EVAL_RECONFIG_TEMPERATURE = 1.0
    EVAL_RECONFIG_RESTARTS = 1

    # ---- Entropy tests (post-hoc OOD subspace probe) ----
    ENTROPY_TESTS = False
    ENTROPY_TEST_TRIALS = 10

    # ---- Training ----
    TRAIN_LR = 3e-4

    # ---- Plotting / logging ----
    RENDER_STEPS = True
    LOSS_EMA_SMOOTHING = 0.8    
    PLOT_EVERY = 1
    GIF_DURATION = 20
    GIF_SAVE_EPOCH = 1
    GIF_EMA_SCALE = 0.2

    @property
    def HIDDEN_SIZE(self):
        return self.SQUARE_DIM**2

    @property
    def REPLAY_MAXLEN(self):
        return self.N_SAMPLE_PER_TASK * self.N_TASK

    @property
    def OVERLAP_PARS(self):
        return {
            "overlap_mode": self.RECONFIG_OVERLAP_MODE,
            "overlap_lambda": self.RECONFIG_OVERLAP_LAMBDA,
            "overlap_threshold": self.RECONFIG_OVERLAP_THRESHOLD,
            "overlap_temperature": self.RECONFIG_OVERLAP_TEMPERATURE,
        }

    @property
    def FTN_PARS(self):
        return {
            "square_dim": self.SQUARE_DIM,
            "classification": self.CLASSIFICATION,
            "layers": self.LAYERS,
            "inner_size": self.INNER_DIM,
            "dropout_p": self.DROPOUT,
            "k": self.K,
            "reconfig_steps": self.RECONFIG_STEPS,
            "reconfig_lr": self.RECONFIG_LR,
            "reconfig_sample_size": self.RECONFIG_SAMPLE_SIZE,
            "reconfig_algorithm": self.RECONFIG_ALGORITHM,
            "reconfig_noise_scale": self.RECONFIG_NOISE_SCALE,
            **self.OVERLAP_PARS,
        }

    @property
    def BAYES_FTN_PARS(self):
        return {
            **self.OVERLAP_PARS,
            "bayes_reconfig_steps": self.BAYES_RECONFIG_STEPS,
            "bayes_reconfig_lr": self.BAYES_RECONFIG_LR,
            "bayes_kl_weight": self.BAYES_KL_WEIGHT,
            "bayes_temperature": self.BAYES_TEMPERATURE,
            "bayes_prior_decay": self.BAYES_PRIOR_DECAY,
            "bayes_stochastic_mask": self.BAYES_STOCHASTIC_MASK,
            "bayes_reconfig_take_best": self.BAYES_RECONFIG_TAKE_BEST,
            "bayes_changepoint_reconfig_steps": (
                self.BAYES_CHANGEPOINT_RECONFIG_STEPS
            ),
            "bayes_changepoint_reconfig_lr": self.BAYES_CHANGEPOINT_RECONFIG_LR,
            "bayes_changepoint_reconfig_sample_size": (
                self.BAYES_CHANGEPOINT_RECONFIG_SAMPLE_SIZE
            ),
            "bayes_changepoint_kl_weight": self.BAYES_CHANGEPOINT_KL_WEIGHT,
            "bayes_changepoint_temperature": self.BAYES_CHANGEPOINT_TEMPERATURE,
            "bayes_changepoint_prior_decay": self.BAYES_CHANGEPOINT_PRIOR_DECAY,
            "bayes_changepoint_stochastic_mask": (
                self.BAYES_CHANGEPOINT_STOCHASTIC_MASK
            ),
        }

    @property
    def LORA_FTN_PARS(self):
        """Everything BayesLoRAFTN takes besides input/output/hidden sizes and
        the BAYES_FTN_PARS posterior bundle. Reuses the shared K / RECONFIG_*
        mask-search knobs so the LoRA atom mask is configured like the base
        FTN's."""
        return {
            "square_dim": self.SQUARE_DIM,
            "classification": self.CLASSIFICATION,
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
            # reconfiguration / search knobs reused by the Bayesian mixin
            "k": self.K,
            "reconfig_steps": self.RECONFIG_STEPS,
            "reconfig_sample_size": self.RECONFIG_SAMPLE_SIZE,
            "reconfig_noise_scale": self.RECONFIG_NOISE_SCALE,
        }

    @property
    def LORA_PARS(self):
        """Independent-mask LoRAFTN parameters, matching base/config.py."""
        return {
            **self.LORA_FTN_PARS,
            **self.OVERLAP_PARS,
            "binary": False,
            "reconfig_lr": self.RECONFIG_LR,
            "reconfig_algorithm": self.RECONFIG_ALGORITHM,
            "reconfig_init_sd": self.RECONFIG_INIT_SD,
            "warm_start_reconfig": self.LORA_WARM_START_RECONFIG,
            "reactive_continuous_updates": self.LORA_REACTIVE_CONTINUOUS_UPDATES,
            "changepoint_reconfig_steps": self.LORA_CHANGEPOINT_RECONFIG_STEPS,
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
    def RNN_FTN_PARS(self):
        """Everything BayesRNNResidualFTN takes besides input/output/hidden sizes
        and the BAYES_FTN_PARS posterior bundle. Reuses the base backbone knobs
        (LAYERS / INNER_DIM / DROPOUT) for the shared cell and the shared K /
        RECONFIG_* mask-search knobs (K is a per-step budget here)."""
        return {
            "square_dim": self.SQUARE_DIM,
            "classification": self.CLASSIFICATION,
            # topology
            "n_stack": self.N_STACK,
            "rnn_step_rmsnorm": self.RNN_STEP_RMSNORM,
            # shared cell backbone (reused base knobs)
            "layers": self.LAYERS,
            "inner_size": self.INNER_DIM,
            "dropout_p": self.DROPOUT,
            # reconfiguration / search knobs reused by the Bayesian mixin
            "k": self.K,
            "reconfig_steps": self.RECONFIG_STEPS,
            "reconfig_sample_size": self.RECONFIG_SAMPLE_SIZE,
            "reconfig_noise_scale": self.RECONFIG_NOISE_SCALE,
        }

    @property
    def LORA_RNN_FTN_PARS(self):
        """Everything BayesLoRARNNFTN takes besides input/output/hidden sizes and
        the BAYES_FTN_PARS posterior bundle. A weight-tied recurrent LoRA cell:
        combines the LoRA atom knobs (LORA_*) with the recurrent knobs
        (N_STACK / RNN_STEP_RMSNORM) and the shared K / RECONFIG_* mask-search
        knobs (K is a per-step budget over the shared atom pool)."""
        return {
            "square_dim": self.SQUARE_DIM,
            "classification": self.CLASSIFICATION,
            # recurrent topology
            "n_stack": self.N_STACK,
            "width": self.LORA_WIDTH,
            "rnn_step_rmsnorm": self.RNN_STEP_RMSNORM,
            "weight_scale": self.LORA_WEIGHT_SCALE,
            "io": self.LORARNN_IO,
            # shared LoRA atom pool
            "rank": self.LORA_RANK,
            "alpha": self.LORA_ALPHA,
            "dropout_p": self.LORA_DROPOUT,
            "b0_scale": self.LORA_B0_SCALE,
            "use_bias": self.LORA_BIAS,
            # reconfiguration / search knobs reused by the Bayesian mixin
            "k": self.K,
            "reconfig_steps": self.RECONFIG_STEPS,
            "reconfig_sample_size": self.RECONFIG_SAMPLE_SIZE,
            "reconfig_noise_scale": self.RECONFIG_NOISE_SCALE,
        }

    @property
    def EVAL_FTN_PARS(self):
        return {
            "steps": self.EVAL_RECONFIG_STEPS,
            "lr": self.EVAL_RECONFIG_LR,
            "sample_size": self.EVAL_RECONFIG_SAMPLE_SIZE,
            "temperature": self.EVAL_RECONFIG_TEMPERATURE,
            "restarts": self.EVAL_RECONFIG_RESTARTS,
            "take_best": getattr(self, "BAYES_RECONFIG_TAKE_BEST", False),
        }


# =============================================================================
# Sub-configs (compose onto Default). Each is a small set of overrides.
# =============================================================================

'''
Bayesian mask distribution (ARCHITECTURE="bayes")
-------------------------------------------------
BAYES_RECONFIG_STEPS Variational posterior steps per reconfiguration. Use at
                     least 2 for the KL gradient to act within the same call.
BAYES_RECONFIG_LR    Learning rate for the persistent mask logits.
BAYES_KL_WEIGHT      Strength of mean Bernoulli KL(q_new || q_previous).
BAYES_TEMPERATURE    Temperature for sigmoid relaxation and Gumbel top-k.
BAYES_PRIOR_DECAY    Prior diffusion rho: prior_logits=rho*previous_logits.
                     1.0 strictly carries the preceding posterior forward.
BAYES_STOCHASTIC_MASK
                     True samples exact-k masks with Gumbel top-k; False uses
                     deterministic top-k for a controlled ablation.
BAYES_RECONFIG_TAKE_BEST
                     Keep the lowest support-loss mask seen during Bayesian
                     training updates and cold evaluation searches.


Backwards / prospective eval (forgetting monitor)
-------------------------------------------------
EVAL_RECONFIG_STEPS  Stateless mask-search steps per support set.
EVAL_RECONFIG_LR     Evaluation-only mask-logit learning rate.
EVAL_RECONFIG_SAMPLE_SIZE
                     Maximum support rows used; None uses the entire batch.
EVAL_RECONFIG_TEMPERATURE
                     Temperature for the sigmoid surrogate and Gumbel top-k.
EVAL_RECONFIG_RESTARTS
                     Independent zero-logit searches; the deterministic top-k
                     result with the lowest support loss is retained.
                     Evaluation always uses fresh logits/optimizer, no KL, and
                     never reads or updates BayesFTN's persistent logits.
'''
class Bayes(Default):
    ARCHITECTURE="bayes"
    BAYES_RECONFIG_TAKE_BEST = True

    TRAIN_LR = 3e-4

    RECONFIG_EVERY = 0
    RECONFIG_REACTIVE = True
    RECONFIG_REACTIVE_SD = 5.0
    RECONFIG_REACTIVE_EMA = 0.9

    GIF_EMA_SCALE = 1.0

    # ---- Bayesian mask distribution (used only by ARCHITECTURE="bayes") ----
    BAYES_RECONFIG_STEPS = 2
    BAYES_RECONFIG_LR = 0.3
    BAYES_KL_WEIGHT = 1.0
    BAYES_TEMPERATURE = 3.0
    BAYES_PRIOR_DECAY = 1.0
    BAYES_STOCHASTIC_MASK = True

    BAYES_CHANGEPOINT_RECONFIG_STEPS = 30
    BAYES_CHANGEPOINT_RECONFIG_LR = 0.3
    BAYES_CHANGEPOINT_RECONFIG_SAMPLE_SIZE = None
    BAYES_CHANGEPOINT_KL_WEIGHT = 0.0
    BAYES_CHANGEPOINT_TEMPERATURE = 1.0
    BAYES_CHANGEPOINT_PRIOR_DECAY = 0.0
    BAYES_CHANGEPOINT_STOCHASTIC_MASK = True

    # ---- Backwards / prospective eval (forgetting monitor) ----
    EVAL_BACKWARDS = True
    EVAL_PROSPECTIVE = True
    N_RETRO_EVAL_BATCHES = 10
    EVAL_RECONFIG_STEPS = 30
    EVAL_RECONFIG_LR = 0.3
    EVAL_RECONFIG_SAMPLE_SIZE = None
    EVAL_RECONFIG_TEMPERATURE = 1.0
    EVAL_RECONFIG_RESTARTS = 1

class BayesReactiveLow(Bayes):
    BAYES_RECONFIG_STEPS = 1
    BAYES_RECONFIG_LR = 0.0
    BAYES_STOCHASTIC_MASK = False
    REPLAY_BATCH_SIZE=0


class Bayes_RS30_Eval(Bayes):
    BAYES_CHANGEPOINT_RECONFIG_STEPS = 30
    EVAL_RECONFIG_STEPS = 30

class Base(Default):
    ARCHITECTURE = "base"


class LoRA(Default):
    """Independent mask search over the frozen MLP's LoRA atom bank."""
    ARCHITECTURE = "lora"

    TRAIN_LR = 1e-4

    LORA_LAYERS = 8
    LORA_WIDTH = 256
    LORA_WEIGHT_SCALE = 1.0
    LORA_RANK = 1
    LORA_ALPHA = 256.0
    LORA_DROPOUT = 0.0
    LORA_BIAS = True
    LORA_B0_SCALE = 0.01

    RECONFIG_INIT_SD = 1e-5
    N_EPOCHS = 1
    RECONFIG_ALGORITHM = "annealste"
    RECONFIG_STEPS = 2
    RECONFIG_LR = 0.3


class LoRAWarmReactive(LoRA):
    """Carry LoRA search logits; cold-refit on reactive changepoints."""
    RECONFIG_REACTIVE = True
    RECONFIG_REACTIVE_SD = 5.0
    RECONFIG_REACTIVE_EMA = 0.9
    LORA_WARM_START_RECONFIG = True
    LORA_REACTIVE_CONTINUOUS_UPDATES = True
    LORA_CHANGEPOINT_RECONFIG_STEPS = 30


# ---- Bayesian re-implementations of the two alternative backbones ----
# Both inherit the Bayes reactive / posterior settings and swap only the
# backbone (and its hyperpars) for the LoRA atom bank or the recurrent-residual
# cell. Compose the LoRA_* / RNN / N_STACK knobs on top as usual.
class BayesLoRA(Bayes):
    ARCHITECTURE = "bayes_lora"

    # RECONFIG_REACTIVE_SD=7
    TRAIN_DURATION=1500
    SQUARE_DIM=64
    K=256

    LORA_LAYERS = 8          # frozen backbone MLP depth
    LORA_WIDTH = 256          # frozen backbone MLP width
    LORA_WEIGHT_SCALE = 1.0  # fixed scale of the frozen backbone weights
    LORA_RANK = 4            # rank r of each LoRA atom
    LORA_ALPHA = 64.0        # LoRA scaling (alpha/K on the summed delta)
    LORA_DROPOUT = 0.0       # dropout inside each LoRA linear
    LORA_BIAS = True         # per-atom bias bank
    LORA_B0_SCALE = 0.01     # atom-reachability init scale (B / bias bank)

    EVAL_RECONFIG_STEPS = 30

    BAYES_RECONFIG_STEPS = 2
    BAYES_RECONFIG_LR = 0.3
    BAYES_RECONFIG_TAKE_BEST = False
    BAYES_KL_WEIGHT = 1.0
    BAYES_TEMPERATURE = 3.0
    BAYES_PRIOR_DECAY = 1.0
    BAYES_STOCHASTIC_MASK = True

    BAYES_CHANGEPOINT_RECONFIG_STEPS = 30
    BAYES_CHANGEPOINT_RECONFIG_LR = 0.3
    BAYES_CHANGEPOINT_RECONFIG_SAMPLE_SIZE = None
    BAYES_CHANGEPOINT_KL_WEIGHT = 0.0
    BAYES_CHANGEPOINT_TEMPERATURE = 1.0
    BAYES_CHANGEPOINT_PRIOR_DECAY = 0.0
    BAYES_CHANGEPOINT_STOCHASTIC_MASK = True

    # ---- Replay ----
    # N_SAMPLE_PER_TASK = 100
    # N_REPLAY_SAMPLES = 4
    # REPLAY_BATCH_SIZE = 16




class BayesRNNResidual(Bayes):
    ARCHITECTURE = "bayes_rnn_residual"
    LAYERS = 2
    N_STACK = 4              # recurrent steps / per-step masks
    RNN_STEP_RMSNORM = False # shared RMSNorm on the fed-back state (steps > 0)

    EVAL_RECONFIG_STEPS = 240
    BAYES_TEMPERATURE = 0.1


class BayesLoRARNN(Bayes):
    # Weight-tied recurrent LoRA: N_STACK per-step masks re-using / re-sequencing
    # a single shared pool of low-rank atoms (LORA_* + N_STACK knobs).
    ARCHITECTURE = "bayes_lora_rnn"
    TRAIN_DURATION=1500
    SQUARE_DIM=64
    K=1024
    
    EVAL_RECONFIG_STEPS = 30

    GIF_EMA_SCALE = 1.0

    BAYES_RECONFIG_STEPS = 2
    BAYES_RECONFIG_LR = 0.3
    BAYES_RECONFIG_TAKE_BEST = False
    BAYES_KL_WEIGHT = 1.0
    BAYES_TEMPERATURE = 3.0
    BAYES_PRIOR_DECAY = 1.0
    BAYES_STOCHASTIC_MASK = True


    BAYES_CHANGEPOINT_RECONFIG_STEPS = 30
    BAYES_CHANGEPOINT_RECONFIG_LR = 0.3
    BAYES_CHANGEPOINT_RECONFIG_SAMPLE_SIZE = None
    BAYES_CHANGEPOINT_KL_WEIGHT = 0.0
    BAYES_CHANGEPOINT_TEMPERATURE = 1.0
    BAYES_CHANGEPOINT_PRIOR_DECAY = 0.0
    BAYES_CHANGEPOINT_STOCHASTIC_MASK = True

    # A frozen random MLP plus a bank of HIDDEN_SIZE LoRA atoms on every linear;
    # the Bayesian mask selects K of the atoms. Reuses the shared K / RECONFIG_*
    # mask-search knobs so the atom mask is configured like the base FTN's.
    LORA_LAYERS = 2          # frozen backbone MLP depth
    LORA_WIDTH = 256          # frozen backbone MLP width
    LORA_WEIGHT_SCALE = 1.0  # fixed scale of the frozen backbone weights
    LORA_RANK = 1            # rank r of each LoRA atom
    LORA_ALPHA = 256.0        # LoRA scaling (alpha/K on the summed delta)
    LORA_DROPOUT = 0.0       # dropout inside each LoRA linear
    LORA_BIAS = True         # per-atom bias bank
    LORA_B0_SCALE = 0.01     # atom-reachability init scale (B / bias bank)

    # ---- Recurrent-residual FTN backbone (ARCHITECTURE="bayes_rnn_residual") ----
    # One weight-tied ParallelMLP cell unrolled N_STACK steps with a per-step
    # kWTA mask and residual skips; the per-step masks are concatenated in
    # self.m (total HIDDEN_SIZE*N_STACK). K is a per-step budget. The cell reuses
    # the base backbone knobs (LAYERS / INNER_DIM / DROPOUT).
    N_STACK = 4              # recurrent steps / per-step masks
    RNN_STEP_RMSNORM = False # shared RMSNorm on the fed-back state (steps > 0)

    # Read-in / read-out treatment: "fixed" (frozen random I/O, atoms only in the
    # shared recurrent pool) or "ftn" (maskable LoRA atoms on the encoder + readout
    # too). See Default.LORARNN_IO.
    LORARNN_IO = "fixed"


class BayesLoRARNN_Fixed(BayesLoRARNN):
    # Frozen random encoder + readout; all mask-selectable capacity is the shared
    # recurrent atom pool.
    LORARNN_IO = "fixed"


class BayesLoRARNN_Ftn(BayesLoRARNN):
    # The encoder and readout also carry maskable LoRA atom banks, so a task
    # selects its read-in / read-out edits alongside the recurrent pool.
    LORARNN_IO = "ftn"


class ORS2(Bayes):
    BAYES_RECONFIG_STEPS = 2
class ORS3(Bayes):
    BAYES_RECONFIG_STEPS = 3
class ORS4(Bayes):
    BAYES_RECONFIG_STEPS = 4
class ORS6(Bayes):
    BAYES_RECONFIG_STEPS = 6
class ORS8(Bayes):
    BAYES_RECONFIG_STEPS = 8



class Smoke(Default):
    N_TASK = 3
    SQUARE_DIM = 8
    LAYERS = 2
    K = 16
    RECONFIG_STEPS = 5
    EVAL_RECONFIG_STEPS = 5
    BAYES_CHANGEPOINT_RECONFIG_STEPS = 5
    T = 64
    TRAIN_DURATION = 5
    N_SAMPLE_PER_TASK = 2
    N_RETRO_EVAL_BATCHES = 2


class EntropyTests(Default):
    ENTROPY_TESTS = True
    ENTROPY_TEST_TRIALS = 10


class DomainShift(Default):
    DOMAIN_SHIFT = True
    DOMAIN_SHIFT_STRIDE = 1
    DOMAIN_DENSE_PROJECTION = True
    DOMAIN_PER_TASK_PROJECTION = True


class InputConditioned(Default):
    """Route evaluation from inputs; repair archived masks with training replay.

    Compose after Bayes, BayesLoRA, or BayesLoRARNN. The demonstrated regime
    uses DomainShift with CONCEPT_SHIFT=False; input-only routing cannot
    identify tasks that have the same input distribution. The replay buffer
    must hold at least FIT_ROWS + VALIDATION_ROWS per old domain for repair.
    All knobs are here so a later config class can override any of them.
    """
    # EVAL_RECONFIG_SAMPLE_SIZE = 4  # also bounds fallback search on router miss

    IC_ENABLED = True
    IC_SUPPORT_SIZE = 4            # unlabeled inputs used to select a route
    IC_AE_RANK = 2                 # rank of each linear reconstruction model
    IC_RECON_THRESHOLD = 0.10      # maximum normalized reconstruction error
    IC_MATCH_TOLERANCE = 1e-3      # prefer latest mask within this error of best
    IC_SAME_DOMAIN_DISTANCE = 0.05 # projector distance for replay deduplication
    IC_NOVELTY_TRIGGER = False     # add input novelty to loss change detector

    IC_REPLAY_REPAIR = True
    IC_REPAIR_EVERY_BLOCKS = 1
    IC_REPAIR_FIT_ROWS = 800
    IC_REPAIR_VALIDATION_ROWS = 800
    IC_REPAIR_STEPS = 100
    IC_REPAIR_LR = 0.3
    IC_REPAIR_TEMPERATURE = 1.0
    IC_REPAIR_RESTARTS = 3
    IC_REPAIR_TAKE_BEST = True
    IC_REPAIR_MIN_IMPROVEMENT = 0.0
    IC_REPAIR_LOGIT_MAGNITUDE = 4.0

    IC_TRAIN_PRIOR = False         # apply retrieved logits at change points
    IC_PRIOR_STEPS = 30            # posterior steps if train/eval refinement used

    IC_PRIOR_KL = 1.0              # KL to retrieved logits if refined
    IC_EVAL_STRATEGY = "refine"    # direct hard mask, or "refine" with support y


class InputConditionedSmoke(Default):
    """Small replay budget for exercising the routing and repair path."""
    TRAIN_DURATION = 8
    IC_NOVELTY_TRIGGER = True
    IC_REPAIR_FIT_ROWS = 32
    IC_REPAIR_VALIDATION_ROWS = 32
    IC_REPAIR_STEPS = 2
    IC_REPAIR_RESTARTS = 1
    REPLAY_BATCH_SIZE = 32


class NoConceptShift(Default):
    """Keep the target function shared while varying input domains."""
    CONCEPT_SHIFT = False


# ---- Replay amount ----
class Replay(Default):
    N_SAMPLE_PER_TASK = 10
    N_REPLAY_SAMPLES = 1
    REPLAY_BATCH_SIZE = 256


class NoReplay(Default):
    N_SAMPLE_PER_TASK = 1
    N_REPLAY_SAMPLES = 0
    REPLAY_BATCH_SIZE = 1


# ---- When to reconfigure ----
class PerTask(Default):
    RECONFIG_EVERY = 0
    RECONFIG_REACTIVE = False


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


# ---- Mask-search budget / algorithm ----
class Fast(Default):
    RECONFIG_ALGORITHM = "fast"
    RECONFIG_STEPS = 4
    RECONFIG_LR = 0.3


class Flash(Default):
    RECONFIG_ALGORITHM = "fast"
    RECONFIG_STEPS = 1
    RECONFIG_LR = 1.0


class Thorough(Default):
    RECONFIG_ALGORITHM = "annealste"
    RECONFIG_STEPS = 60
    RECONFIG_LR = 0.3


# ---- Network size ----
class Large(Default):
    SQUARE_DIM = 96
    K = 1024


class Small(Default):
    SQUARE_DIM = 8
    K = 4


# ---- Schedule ----
class Train1000(Default):
    TRAIN_DURATION = 1000
class Train2000(Default):
    TRAIN_DURATION = 2000
class Train4000(Default):
    TRAIN_DURATION = 4000

class RS_3(Bayes):
    EVAL_RECONFIG_STEPS = 3
class RS_10(Bayes):
    EVAL_RECONFIG_STEPS = 10
class RS_20(Bayes):
    EVAL_RECONFIG_STEPS = 20
class RS_30(Bayes):
    EVAL_RECONFIG_STEPS = 30
class RS_40(Bayes):
    EV_RECONFIG_STEPS = 40
class RS_50(Bayes):
    EVAL_RECONFIG_STEPS = 50
class RS_60(Bayes):
    EVAL_RECONFIG_STEPS = 60


class DxKL0(Bayes):
    BAYES_KL_WEIGHT = 1.0
    BAYES_PRIOR_DECAY = 1.0

class DxKL1(Bayes):
    BAYES_KL_WEIGHT = 0.3
    BAYES_PRIOR_DECAY = 1.0

class DxKL2(Bayes):
    BAYES_KL_WEIGHT = 1.0
    BAYES_PRIOR_DECAY = 0.8

class DxKL3(Bayes):
    BAYES_KL_WEIGHT = 0.3
    BAYES_PRIOR_DECAY = 0.8


class NoSample(Bayes):
    BAYES_STOCHASTIC_MASK = False
class Sample(Bayes):
    BAYES_STOCHASTIC_MASK = True



class KL0(Bayes):
    BAYES_KL_WEIGHT = 10.0

class KL1(Bayes):
    BAYES_KL_WEIGHT = 1.0

class KL2(Bayes):
    BAYES_KL_WEIGHT = 0.1

class KL3(Bayes):
    BAYES_KL_WEIGHT = 0.01


class Supp16(Default):
    RECONFIG_SAMPLE_SIZE=16

class Supp32(Default):
    RECONFIG_SAMPLE_SIZE=32

class Supp64(Default):
    RECONFIG_SAMPLE_SIZE=64

class Supp256(Default):
    RECONFIG_SAMPLE_SIZE=256

class Overlap_05(Default):
    REPLAY_OVERLAP_LOW=0.5

class Overlap_04(Default):
    REPLAY_OVERLAP_LOW=0.4

class Overlap_03(Default):
    REPLAY_OVERLAP_LOW=0.3

class Overlap_02(Default):
    REPLAY_OVERLAP_LOW=0.2

class Orth(Default):
    CORRELATED_TASKS=False

class Sparse(Default):
    K = 16
    SQUARE_DIM=32
    N_TASK=6
    TRAIN_DURATION = 500
    TRAIN_LR = 3e-4
    INNER_DIM=64

class BayesLoRAReactiveLow(BayesLoRA):
    N_TASK=4
    BAYES_RECONFIG_STEPS = 1
    BAYES_RECONFIG_LR = 0.0
    BAYES_STOCHASTIC_MASK = False
    REPLAY_BATCH_SIZE=0

class BayesLoRAReactiveLowSmooth(BayesLoRA):
    N_TASK=4
    BAYES_RECONFIG_STEPS = 1
    BAYES_RECONFIG_LR = 0.0
    BAYES_STOCHASTIC_MASK = False
    REPLAY_BATCH_SIZE=0

    SQUARE_DIM=32
    K=256

    MASK_LATERAL_POLYAK=0.0
    MASK_LATERAL_STEPS=2
    MASK_KERNEL_SIZE=17

class LowSupport(Default):
    # RECONFIG_SAMPLE_SIZE=4
    # BAYES_CHANGEPOINT_RECONFIG_SAMPLE_SIZE=4
    EVAL_RECONFIG_SAMPLE_SIZE=4


class Overlap(Default):
    RECONFIG_OVERLAP_MODE = "gated"  # "bump" or "gated"
    RECONFIG_OVERLAP_LAMBDA = 0.5
    RECONFIG_OVERLAP_THRESHOLD = 0.5
    RECONFIG_OVERLAP_TEMPERATURE = 1.0