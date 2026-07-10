"""
config.py - Experiment configurations for FTN experiments.

Each class is a self-contained experiment config consumed by
run_experiments.py. Attributes are read as `Config.<NAME>`; the `@property`
fields at the bottom derive values from them. FTN_PARS collects everything
the FTN class needs (backbone + mask/reconfiguration hyperparameters).

Compared to base/config.py this drops:
  - MASK_CONFIGURER      (FTN's annealed-STE KWTA is the only algorithm;
                          MASK_LATERAL_STEPS=0 gives plain KWTA,
                          >0 gives SmoothKWTA behavior)
  - CONFIGURE_PER_BATCH  (reconfiguration always happens per batch, inside
                          FTN.forward(x, y))
  - SHUFFLE_ON_REPLAY    (replay always subsamples REPLAY_BATCH_SIZE rows once
                          at store time)
  - PHASES               (the experiment is exactly the old phase 1:
                          multi-task learning over the task stream)

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
MASK_KERNEL_SIZE     Mask lateral-smoothing kernel size.
MASK_LATERAL_STEPS   Mask lateral-smoothing iterations; 0 = plain KWTA.

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
TRAIN_LR, WEIGHT_DECAY   Weight-optimizer hyperparameters.
LOSS_EMA_SMOOTHING       EMA factor for the plotted loss curve.
PLOT_EVERY               Plot/log every N epochs.
GIF_DURATION             Per-frame duration (ms) for GIFs.
GIF_SAVE_EPOCH           Epoch interval at which GIFs are saved.
"""


class Default:
    """Default experiment configuration (classification, 6 correlated tasks)."""

    # Experiment pars
    SEED = 123
    N_TASK = 4

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

    # FTN backbone
    SQUARE_DIM = 48
    LAYERS = 8
    INNER_DIM = 8
    DROPOUT = 0.2
    LATERAL_B = 0.0
    LATERAL_STEPS = 0
    KERNEL_SIZE = 5
    TORUS = False

    # FTN mask / reconfiguration
    K = 256
    RECONFIG_STEPS = 30
    RECONFIG_LR = 0.3
    RECONFIG_SAMPLE_SIZE = 256
    MASK_KERNEL_SIZE = 3
    MASK_LATERAL_STEPS = 0  # 0 = plain KWTA; >0 = SmoothKWTA

    # Data sizes
    TRAIN_DURATION = 1000
    N_EPOCHS = 1
    N_SAMPLE_PER_TASK = 10
    N_EVAL_EPOCHS = 1
    N_REPLAY_SAMPLES = 1
    REPLAY_BATCH_SIZE = 32

    # Backwards eval (forgetting monitor)
    EVAL_BACKWARDS = True
    EVAL_PROSPECTIVE = True
    N_RETRO_EVAL_BATCHES = 10

    # Training hyperpars
    TRAIN_LR = 3e-4
    WEIGHT_DECAY = 0.0

    # Plotting
    LOSS_EMA_SMOOTHING = 0.8
    PLOT_EVERY = 1
    GIF_DURATION = 20
    GIF_SAVE_EPOCH = 1

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
            "mask_kernel_size": self.MASK_KERNEL_SIZE,
            "mask_lateral_steps": self.MASK_LATERAL_STEPS,
            "torus": self.TORUS,
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
