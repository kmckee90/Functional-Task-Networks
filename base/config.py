"""
config.py - Experiment configurations for masked network (FTN) experiments.

Each class below is a self-contained experiment config consumed by
run_experiments.py. Attributes are read as `Config.<NAME>`; the `@property`
fields at the bottom derive values from them.

MASK_CONFIGURER      The mask-configurer class (from masks.py) used to select
                     active units per task (e.g. masks.KWTA).

Parameter glossary
==================

Experiment
----------
SEED                 Base RNG seed. Local torch.Generators are seeded off this
                     (e.g. SEED + offset) for reproducibility.
N_TASK               Number of distinct tasks in the continual-learning stream.
PHASES               Which experiment phases to run; subset of [0,1,2,3,4]
                     (see run_experiments.py). Phases select what gets trained
                     and evaluated.

Data generator (see tasks.py DataGen / DataGenSine)
---------------------------------------------------
CLASSIFICATION       If True, generate a classification target (one-hot / sign
                     of hidden sum); if False, a scalar regression target.
X_DIM                Input feature dimension (input_dim).
Y_DIM                Output dimension (n classes for classification, else 1).
TASK_SIZE            Base number of hidden dims allocated per task; controls
                     task overlap and total generator hidden_size.
FREQUENCY            Frequency of the sine nonlinearity in DataGenSine
                     (y ~ sin(FREQUENCY * z)); higher = more oscillatory,
                     harder targets.
T                    Samples generated per data batch (DataGen n_samples).
DATA_TEMP            Softmax temperature for label sampling in the classifier
                     decoder; near-zero => near-deterministic argmax labels.
CORRELATED_TASKS     If True, tasks share hidden dims via pairwise combinations;
                     if False, tasks use disjoint dimensions.
CONCEPT_SHIFT        If True, each task uses a different hidden subset (task
                     identity changes the function); if False, all tasks share
                     the same dims.
DOMAIN_SHIFT_FACTOR  Scale of per-task input mean shift; 0 => all tasks share
                     the same input distribution.

Model hyperparameters (see models.py, masks.py)
------------------------------------------------
K                    k-winner-take-all budget: number of active (=1) units the
                     mask keeps after sparsification (MASK_PARS["k"]).
SQUARE_DIM           Side length of the square grid the mask is reshaped into
                     for 2D lateral dynamics; HIDDEN_SIZE = SQUARE_DIM**2.
LAYERS               Number of hidden MLP layers in the masked model.
DROPOUT              Dropout probability applied in the model.
INNER_DIM            Inner/bottleneck size passed to the model (MODEL_PARS
                     "inner_size").
LATERAL_B            Strength ("b") of lateral dynamics in the model's lateral
                     conv; 0 disables lateral mixing.
LATERAL_STEPS        Number of lateral dynamics iterations ("steps") in the
                     model; 0 disables.
KERNEL_SIZE          Conv kernel size for lateral smoothing (odd; padded to
                     preserve size).
TORUS                If True, lateral conv uses circular padding (wrap-around
                     topology); if False, zero padding.
CONFIGURE_PER_BATCH  If True, reconfigure the mask every batch; if False,
                     reconfigure once per task.

Data sizes / schedule
---------------------
TRAIN_DURATION       Number of training steps (batches) per task per epoch.
N_EPOCHS             Number of epochs over the task stream.
N_SAMPLE_PER_TASK    Samples per task retained for replay; feeds REPLAY_MAXLEN.
N_EVAL_EPOCHS        Number of evaluation epochs.
N_REPLAY_SAMPLES     Number of stored batches drawn per step from the buffer
                     (0 disables replay).
REPLAY_BATCH_SIZE    Rows used per replay minibatch. If < the observed batch
                     size (T), that many rows are randomly sampled; if >= T
                     (or None), the full batch is used.
SHUFFLE_ON_REPLAY    If True, store the full observed batch and draw a fresh
                     random REPLAY_BATCH_SIZE rows on every replay; if False,
                     subsample REPLAY_BATCH_SIZE rows once when storing and
                     reuse that fixed subset.

Backwards / prospective eval (forgetting monitor)
--------------------------------------------------
EVAL_BACKWARDS       If True, re-evaluate on all previously seen tasks to
                     measure forgetting.
EVAL_PROSPECTIVE     If True, also evaluate on tasks not yet trained (forward
                     transfer).
N_RETRO_EVAL_BATCHES Number of batches used per prior task during backwards
                     eval.

Training hyperparameters
------------------------
TRAIN_LR             Learning rate for the main model optimizer.
WEIGHT_DECAY         Weight decay (L2) for the optimizer.

Plotting
--------
LOSS_EMA_SMOOTHING   EMA smoothing factor for the plotted loss curve.
PLOT_EVERY           Plot/log every N epochs.
GIF_DURATION         Per-frame duration (ms) for saved GIF animations.
GIF_SAVE_EPOCH       Epoch interval at which GIFs are saved.

Derived properties
------------------
HIDDEN_SIZE          SQUARE_DIM**2 - hidden width of the masked model.
N_OOD_TASKS          Number of out-of-distribution tasks (= N_TASK).
REPLAY_MAXLEN        Replay buffer capacity (N_SAMPLE_PER_TASK * N_TASK).
MODEL_PARS           Dict of model kwargs assembled from the above.
MASK_PARS            Dict of mask-configurer kwargs, incl. reconfig_steps,
                     reconfig_lr, reconfig_sample_size (mask-reconfiguration
                     optimization budget).
"""

import masks


class Default:
    """Default experiment configuration."""

    # Experiment pars
    SEED = 123  # random.randint(0, 9999)
    N_TASK = 6
    PHASES = [1]

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

    # Model hyperpars
    K = 256
    SQUARE_DIM = 48 #int((K * (4)) ** 0.5)
    LAYERS = 8
    DROPOUT = 0.2
    INNER_DIM = 8

    MASK_CONFIGURER = masks.KWTA
    LATERAL_B = 0.0
    LATERAL_STEPS = 0
    KERNEL_SIZE = 5
    TORUS = False

    CONFIGURE_PER_BATCH = True

    # Data sizes
    TRAIN_DURATION = 100
    N_EPOCHS = 1
    N_SAMPLE_PER_TASK = 10
    N_EVAL_EPOCHS = 1
    N_REPLAY_SAMPLES = 1
    REPLAY_BATCH_SIZE = 32
    SHUFFLE_ON_REPLAY = False

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
    def N_OOD_TASKS(self):
        return self.N_TASK

    @property
    def REPLAY_MAXLEN(self):
        return self.N_SAMPLE_PER_TASK * self.N_TASK

    @property
    def MODEL_PARS(self):
        return {
            "b": self.LATERAL_B,
            "steps": self.LATERAL_STEPS,
            "layers": self.LAYERS,
            "kernel_size": self.KERNEL_SIZE,
            "dropout_p": self.DROPOUT,
            "inner_size": self.INNER_DIM,
        }

    @property
    def MASK_PARS(self):
        return {
            "k": self.K,
            "torus": self.TORUS,
            "reconfig_steps": 30,
            "reconfig_lr": 0.3,
            "reconfig_sample_size": 256,
        }


class Regression:
    """Default experiment configuration."""

    # Experiment pars
    SEED = 123  # random.randint(0, 9999)
    N_TASK = 3
    PHASES = [1]
    MASK_CONFIGURER = masks.KWTA

    # Data generator
    CLASSIFICATION = False
    X_DIM = 2
    Y_DIM = 1
    TASK_SIZE = 4
    FREQUENCY = 12
    T = 256
    DATA_TEMP = 1e-8
    CORRELATED_TASKS = False
    CONCEPT_SHIFT = True
    DOMAIN_SHIFT_FACTOR = 0

    # Model hyperpars
    SQUARE_DIM = 32
    K = 192
    LAYERS = 3
    DROPOUT = 0.0

    LATERAL_B = 0.0
    LATERAL_STEPS = 0
    KERNEL_SIZE = 3
    TORUS = True

    CONFIGURE_PER_BATCH = False

    # Data sizes
    TRAIN_DURATION = 100
    N_EPOCHS = 50  # 1000 // TRAIN_DURATION
    N_SAMPLE_PER_TASK = 10
    N_EVAL_EPOCHS = 1

    # Training hyperpars
    TRAIN_LR = 3e-4
    WEIGHT_DECAY = 0.0

    # Plotting
    LOSS_EMA_SMOOTHING = 0.8
    PLOT_EVERY = 1
    GIF_DURATION = 50
    GIF_SAVE_EPOCH = 5

    @property
    def HIDDEN_SIZE(self):
        return self.SQUARE_DIM**2

    @property
    def N_OOD_TASKS(self):
        return self.N_TASK

    @property
    def REPLAY_MAXLEN(self):
        return self.N_SAMPLE_PER_TASK * self.N_TASK

    @property
    def MODEL_PARS(self):
        return {
            "b": self.LATERAL_B,
            "steps": self.LATERAL_STEPS,
            "layers": self.LAYERS,
            "kernel_size": self.KERNEL_SIZE,
            "dropout": self.DROPOUT,
        }

    @property
    def MASK_PARS(self):
        return {
            "k": self.K,
            "torus": self.TORUS,
            "reconfig_steps": 10,
            "reconfig_lr": 0.2,
        }


class Regression_debug:
    """Default experiment configuration."""

    # Experiment pars
    SEED = 123  # random.randint(0, 9999)
    N_TASK = 1
    PHASES = [1]
    MASK_CONFIGURER = masks.KWTA

    # Data generator
    CLASSIFICATION = False
    X_DIM = 2
    Y_DIM = 1
    TASK_SIZE = 4
    FREQUENCY = 12
    T = 2048
    DATA_TEMP = 1e-8
    CORRELATED_TASKS = False
    CONCEPT_SHIFT = True
    DOMAIN_SHIFT_FACTOR = 0

    # Model hyperpars
    SQUARE_DIM = 32
    K = 192
    LAYERS = 3
    DROPOUT = 0.0

    LATERAL_B = 0.0
    LATERAL_STEPS = 0
    KERNEL_SIZE = 3
    TORUS = True

    # Data sizes
    TRAIN_DURATION = 500
    N_EPOCHS = 50  # 1000 // TRAIN_DURATION
    N_SAMPLE_PER_TASK = 10
    N_EVAL_EPOCHS = 1

    # Training hyperpars
    TRAIN_LR = 3e-4
    WEIGHT_DECAY = 0.0

    # Plotting
    LOSS_EMA_SMOOTHING = 0.8
    PLOT_EVERY = 1
    GIF_DURATION = 50
    GIF_SAVE_EPOCH = 5

    @property
    def HIDDEN_SIZE(self):
        return self.SQUARE_DIM**2

    @property
    def N_OOD_TASKS(self):
        return self.N_TASK

    @property
    def REPLAY_MAXLEN(self):
        return self.N_SAMPLE_PER_TASK * self.N_TASK

    @property
    def MODEL_PARS(self):
        return {
            "b": self.LATERAL_B,
            "steps": self.LATERAL_STEPS,
            "layers": self.LAYERS,
            "kernel_size": self.KERNEL_SIZE,
            "dropout": self.DROPOUT,
        }

    @property
    def MASK_PARS(self):
        return {
            "k": self.K,
            "torus": self.TORUS,
            "reconfig_steps": 1,
            "reconfig_lr": 1,
        }
