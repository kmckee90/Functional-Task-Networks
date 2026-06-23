class Default:
    """Default experiment configuration."""

    # Experiment pars
    SEED = 123  # random.randint(0, 9999)
    N_TASK = 48
    PHASES = [1]

    # Data generator
    CLASSIFICATION = True
    X_DIM = 2
    Y_DIM = 2
    TASK_SIZE = 4
    FREQUENCY = 8
    T = 256
    DATA_TEMP = 1e-8
    CORRELATED_TASKS = False
    CONCEPT_SHIFT = True
    DOMAIN_SHIFT_FACTOR = 0

    # Model hyperpars
    K = 512
    SQUARE_DIM = int((K * (4)) ** 0.5)
    LAYERS = 8
    DROPOUT = 0.2
    INNER_DIM = 8

    LATERAL_B = 0.0
    LATERAL_STEPS = 0
    KERNEL_SIZE = 3
    TORUS = True

    CONFIGURE_PER_BATCH = True

    # Data sizes
    TRAIN_DURATION = 1000
    N_EPOCHS = 1
    N_SAMPLE_PER_TASK = 200
    N_EVAL_EPOCHS = 1
    N_REPLAY_SAMPLES = 1

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
