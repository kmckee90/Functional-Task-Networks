"""
config.py - Experiment configurations for the bayesian FTN subproject.

Pared down from base/config.py to exactly this subproject's scope:
  - two comparable architectures: the base FTN and persistent-logit BayesFTN,
  - one task: the sine task (tasks.DataGenSine), classification only,
  - a single mask, plain kWTA (no lateral smoothing of features or mask),
  - replay buffer (no noise probe, no Hopfield/match-stored, no dual mask).

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
ARCHITECTURE         "base" for the original FTN or "bayes" for persistent
                     approximate Bayesian mask inference.

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
                     change-point handling. Base FTN retains trigger-only
                     reconfiguration. BayesFTN makes its normal cheap posterior
                     update on stable batches and uses the separate change-point
                     parameter set below when triggered. Boundary-free: task
                     switches are never disclosed to the detector.
RECONFIG_REACTIVE_SD Trigger threshold: standard deviations above the EMA mean
                     the batch loss must reach to trigger a reconfig.
RECONFIG_REACTIVE_EMA
                     Smoothing factor for the loss mean/variance EMA (higher =
                     slower). Triggers are suppressed for ~1/(1-EMA) batches
                     after each detected change point while statistics re-form.

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
Smoke                Tiny end-to-end configuration for fast sanity checks.
EntropyTests         Mixin: enable the OOD subspace probe (on by default here).
DomainShift          Dimensional domain shift with a per-task dense rotation.
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
    N_TASK = 6
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

    RECONFIG_EVERY = 1
    RECONFIG_REACTIVE = False
    RECONFIG_REACTIVE_SD = 5.0
    RECONFIG_REACTIVE_EMA = 0.95

    # ---- Data sizes / schedule ----
    TRAIN_DURATION = 1500
    N_EPOCHS = 1
    N_EVAL_EPOCHS = 1

    # ---- Replay ----
    N_SAMPLE_PER_TASK = 100
    N_REPLAY_SAMPLES = 4
    REPLAY_BATCH_SIZE = T

    # ---- Backwards / prospective eval (forgetting monitor) ----
    EVAL_BACKWARDS = True
    EVAL_PROSPECTIVE = True
    N_RETRO_EVAL_BATCHES = 10
    EVAL_RECONFIG_STEPS = 60
    EVAL_RECONFIG_LR = 0.3
    EVAL_RECONFIG_SAMPLE_SIZE = None
    EVAL_RECONFIG_TEMPERATURE = 1.0
    EVAL_RECONFIG_RESTARTS = 1

    # ---- Bayesian mask distribution (used only by ARCHITECTURE="bayes") ----
    BAYES_RECONFIG_STEPS = 3
    BAYES_RECONFIG_LR = 0.3
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

    # ---- Entropy tests (post-hoc OOD subspace probe) ----
    ENTROPY_TESTS = True
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
        }

    @property
    def BAYES_FTN_PARS(self):
        return {
            "bayes_reconfig_steps": self.BAYES_RECONFIG_STEPS,
            "bayes_reconfig_lr": self.BAYES_RECONFIG_LR,
            "bayes_kl_weight": self.BAYES_KL_WEIGHT,
            "bayes_temperature": self.BAYES_TEMPERATURE,
            "bayes_prior_decay": self.BAYES_PRIOR_DECAY,
            "bayes_stochastic_mask": self.BAYES_STOCHASTIC_MASK,
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
    def EVAL_FTN_PARS(self):
        return {
            "steps": self.EVAL_RECONFIG_STEPS,
            "lr": self.EVAL_RECONFIG_LR,
            "sample_size": self.EVAL_RECONFIG_SAMPLE_SIZE,
            "temperature": self.EVAL_RECONFIG_TEMPERATURE,
            "restarts": self.EVAL_RECONFIG_RESTARTS,
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

    RECONFIG_EVERY = 0
    RECONFIG_REACTIVE = True
    RECONFIG_REACTIVE_SD = 5.0
    RECONFIG_REACTIVE_EMA = 0.9

    GIF_EMA_SCALE = 1.0

    # ---- Bayesian mask distribution (used only by ARCHITECTURE="bayes") ----
    BAYES_RECONFIG_STEPS = 2
    BAYES_RECONFIG_LR = 0.3
    BAYES_KL_WEIGHT = 1.0
    BAYES_TEMPERATURE = 1.0
    BAYES_PRIOR_DECAY = 1.0
    BAYES_STOCHASTIC_MASK = True

    BAYES_CHANGEPOINT_RECONFIG_STEPS = 60
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
    EVAL_RECONFIG_STEPS = 60
    EVAL_RECONFIG_LR = 0.3
    EVAL_RECONFIG_SAMPLE_SIZE = None
    EVAL_RECONFIG_TEMPERATURE = 1.0
    EVAL_RECONFIG_RESTARTS = 1

class Bayes_RS30_Eval(Bayes):
    BAYES_CHANGEPOINT_RECONFIG_STEPS = 30
    EVAL_RECONFIG_STEPS = 30

class Base(Default):
    ARCHITECTURE = "base"
    SEED = 123

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
    RECONFIG_LR = 0.3


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
class LongTrain(Default):
    TRAIN_DURATION = 3000


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
