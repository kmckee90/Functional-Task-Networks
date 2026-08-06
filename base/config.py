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

Experiment
----------
SEED                 Base RNG seed; local torch.Generators are seeded off it.
N_TASK               Number of distinct tasks in the task stream.

Data generator (tasks.DataGenSine)
----------------------------------
CLASSIFICATION       One-hot classification target vs scalar regression.
X_DIM, Y_DIM         Input / output dimensions.
TASK_SIZE            Hidden dims allocated per task (controls task overlap).
FREQUENCY            Sine frequency; higher = harder targets.
T                    Samples per generated batch.
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

Data sizes / schedule
---------------------
TRAIN_DURATION       Training steps (batches) per task per epoch.
N_EPOCHS             Epochs over the task stream.
N_SAMPLE_PER_TASK    Batches per task retained for replay; feeds REPLAY_MAXLEN.
N_EVAL_EPOCHS        Evaluation epochs.
N_REPLAY_SAMPLES     Stored batches replayed per step (0 disables replay).
REPLAY_BATCH_SIZE    Rows kept per stored replay batch (subsampled once at
                     store time); None keeps the full batch.

Backwards / prospective eval (forgetting monitor)
-------------------------------------------------
EVAL_BACKWARDS       Re-evaluate all previously seen tasks after each task.
EVAL_PROSPECTIVE     Also evaluate not-yet-trained tasks (forward transfer).
N_RETRO_EVAL_BATCHES Batches per task during backwards eval.

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

    # Experiment pars
    SEED = 123
    N_TASK = 3

    # Data generator
    CLASSIFICATION = True
    X_DIM = 2
    Y_DIM = 2
    TASK_SIZE = 4
    FREQUENCY = 8
    T = 256
    DATA_TEMP = 1e-8

    CORRELATED_TASKS = True
    CONCEPT_SHIFT = True

    DOMAIN_SHIFT_FACTOR = 0
    DOMAIN_SHIFT = False
    DOMAIN_SHIFT_STRIDE = 0
    DOMAIN_DENSE_PROJECTION = False
    DOMAIN_PER_TASK_PROJECTION = False

    # FTN backbone
    SQUARE_DIM = 96
    LAYERS = 8
    INNER_DIM = 8

    DROPOUT = 0.0

    LATERAL_B = 0.0
    LATERAL_STEPS = 0
    KERNEL_SIZE = 3
    TORUS = True

    # FTN mask / reconfiguration
    K = 1024

    RECONFIG_ALGORITHM = "fast"
    RECONFIG_STEPS = 1
    RECONFIG_LR = 1
    RECONFIG_SAMPLE_SIZE = T

    RECONFIG_EVERY = 1

    RECONFIG_REACTIVE = False
    RECONFIG_REACTIVE_SD = 5.0
    RECONFIG_REACTIVE_EMA = 0.95

    MASK_KERNEL_SIZE = 21
    MASK_LATERAL_STEPS = 3  # 0 = plain KWTA; >0 = SmoothKWTA
    MASK_LATERAL_POLYAK = 0.0
    MASK_LATERAL_NOISE_SD = 0.0

    # Second mask (DualMask). When USE_SECOND_MASK is True the mask is
    # reconfigured in two stages: mask1 is reconfigured normally, then mask2 is
    # reconfigured against mask2*mask1 with mask1 frozen. The effective mask is
    # mask1*mask2. Every mask-1 knob above has a *_2 twin so the second mask can
    # use its own algorithm ("fast"/"annealste"), budget, K, lateral dynamics,
    # kernel size and torus. Ignored when USE_SECOND_MASK is False.
    USE_SECOND_MASK = True
    K_2 = 64
    RECONFIG_ALGORITHM_2 = "annealste"  # annealste or fast
    RECONFIG_STEPS_2 = 30
    RECONFIG_LR_2 = 0.3
    RECONFIG_SAMPLE_SIZE_2 = T
    MASK_KERNEL_SIZE_2 = 3
    MASK_LATERAL_STEPS_2 = 0  # 0 = plain KWTA; >0 = SmoothKWTA
    MASK_LATERAL_POLYAK_2 = 0.0
    MASK_LATERAL_NOISE_SD_2 = 0.0
    TORUS_2 = False

    # Data sizes
    TRAIN_DURATION = 500
    N_EPOCHS = 3
    N_EVAL_EPOCHS = 1

    N_SAMPLE_PER_TASK = 10
    N_REPLAY_SAMPLES = 1
    REPLAY_BATCH_SIZE = T

    # Backwards eval (forgetting monitor)
    EVAL_BACKWARDS = True
    EVAL_PROSPECTIVE = True
    N_RETRO_EVAL_BATCHES = 10

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
    def HIDDEN_SIZE(self):
        return self.SQUARE_DIM**2

    @property
    def REPLAY_MAXLEN(self):
        return self.N_SAMPLE_PER_TASK * self.N_TASK

    @property
    def FTN_PARS(self):
        """Everything the FTN class takes besides input/output/hidden sizes."""
        return {
            "square_dim": self.SQUARE_DIM,
            "classification": self.CLASSIFICATION,
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
            "mask_kernel_size": self.MASK_KERNEL_SIZE,
            "mask_lateral_steps": self.MASK_LATERAL_STEPS,
            "mask_lateral_polyak": self.MASK_LATERAL_POLYAK,
            "mask_lateral_noise_sd": self.MASK_LATERAL_NOISE_SD,
            "torus": self.TORUS,
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


class DomainShift(Default):
    DOMAIN_SHIFT_FACTOR = 0
    DOMAIN_SHIFT = True
    DOMAIN_SHIFT_STRIDE = 1
    DOMAIN_DENSE_PROJECTION = True
    DOMAIN_PER_TASK_PROJECTION = True

class Replay(Default):
    N_SAMPLE_PER_TASK = 10
    N_REPLAY_SAMPLES = 1
    REPLAY_BATCH_SIZE = 256

class NoReplay(Default):
    N_SAMPLE_PER_TASK = 0
    N_REPLAY_SAMPLES = 0
    REPLAY_BATCH_SIZE = 0

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
    RECONFIG_STEPS = 10
    RECONFIG_LR = 0.3
    DROPOUT = 0.0
    RECONFIG_ALGORITHM = "fast"

class DualMask(Default):
    """Two-stage masking: reconfigure mask1, then mask2 against mask2*mask1
    with mask1 frozen; the effective mask is mask1*mask2. The second mask
    mirrors the first here (same annealed-STE algorithm and budget); override
    the *_2 knobs to give it a different algorithm/K/lateral dynamics/torus."""
    USE_SECOND_MASK = True
    K_2 = 256
    RECONFIG_ALGORITHM_2 = "annealste"
    RECONFIG_STEPS_2 = 30
    RECONFIG_LR_2 = 0.3
    MASK_KERNEL_SIZE_2 = 3
    MASK_LATERAL_STEPS_2 = 0
    MASK_LATERAL_POLYAK_2 = 0.0
    MASK_LATERAL_NOISE_SD_2 = 0.0
    TORUS_2 = True


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
    RECONFIG_STEPS_2 = 30
    RECONFIG_LR_2 = 0.3
    # RECONFIG_SAMPLE_SIZE_2 = T
    MASK_KERNEL_SIZE_2 = 3
    MASK_LATERAL_STEPS_2 = 0  # 0 = plain KWTA; >0 = SmoothKWTA
    MASK_LATERAL_POLYAK_2 = 0.0
    MASK_LATERAL_NOISE_SD_2 = 0.0
    TORUS_2 = True

