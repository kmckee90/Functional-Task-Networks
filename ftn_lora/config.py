"""BayesLoRA settings. Compose presets left to right; later overrides win."""

class Default:
    SEED = 123
    ARCHITECTURE = 'bayes_lora'
    N_TASK = 4
    T = 256
    CLASSIFICATION = True
    X_DIM = 2
    Y_DIM = 2
    TASK_SIZE = 4
    FREQUENCY = 8
    DATA_TEMP = 1e-08
    CORRELATED_TASKS = True
    CONCEPT_SHIFT = True

    DOMAIN_SHIFT = False
    DOMAIN_SHIFT_FACTOR = 0
    DOMAIN_SHIFT_STRIDE = 0
    DOMAIN_DENSE_PROJECTION = False
    DOMAIN_PER_TASK_PROJECTION = False

    TRAIN_DURATION = 500
    N_EPOCHS = 1

    N_EVAL_EPOCHS = 1
    EVAL_BACKWARDS = True
    EVAL_PROSPECTIVE = True
    N_RETRO_EVAL_BATCHES = 10
    EVAL_RECONFIG_STEPS = 30
    EVAL_RECONFIG_LR = 0.3
    EVAL_RECONFIG_SAMPLE_SIZE = None
    EVAL_RECONFIG_TEMPERATURE = 1.0
    EVAL_RECONFIG_RESTARTS = 1
    EVAL_RECONFIG_ENTROPY = False
    TRAIN_RECONFIG_ENTROPY = False
    # False: fresh query batch. True: score exactly the support rows searched.
    EVAL_SAME_SUPPORT = False

    ENTROPY_TESTS = True
    ENTROPY_TEST_TRIALS = 10

    TRAIN_LR = 0.0003

    SQUARE_DIM = 48
    K = 256

    RECONFIG_STEPS = 5
    RECONFIG_SAMPLE_SIZE = T
    RECONFIG_NOISE_SCALE = 1.0
    RECONFIG_EVERY = 0
    RECONFIG_REACTIVE = True
    RECONFIG_REACTIVE_SD = 5.0
    RECONFIG_REACTIVE_EMA = 0.9

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

    LORA_LAYERS = 4
    LORA_WIDTH = 256
    LORA_WEIGHT_SCALE = 1.0
    LORA_RANK = 4
    LORA_ALPHA = 64.0
    LORA_DROPOUT = 0.0
    LORA_BIAS = True
    LORA_B0_SCALE = 0.01

    N_SAMPLE_PER_TASK = 100
    N_REPLAY_SAMPLES = 1
    REPLAY_BATCH_SIZE = T
    REPLAY_OVERLAP_LOW = 0.2
    REPLAY_OVERLAP_HIGH = 0.8

    RENDER_STEPS = True
    LOSS_EMA_SMOOTHING = 0.8
    PLOT_EVERY = 1
    GIF_DURATION = 20
    GIF_SAVE_EPOCH = 1
    GIF_EMA_SCALE = 1.0
    BW_FRAME_SKIP = 1
    
    SPATIAL_PERMUTE = True
    SPATIAL_PERMUTE_INIT = 'random_blobs'
    SPATIAL_PERMUTE_SEED = 0
    SPATIAL_PERMUTE_ROUNDNESS = 2.0
    SPATIAL_PERMUTE_CENTER = 0.25
    SPATIAL_PERMUTE_RADIUS = 7
    SPATIAL_PERMUTE_SWEEPS = 8
    SPATIAL_PERMUTE_TEMPERATURE = 0.05
    
    IC_ENABLED = False

    @property
    def HIDDEN_SIZE(self):
        return self.SQUARE_DIM ** 2

    @property
    def REPLAY_MAXLEN(self):
        return self.N_SAMPLE_PER_TASK * self.N_TASK

    @property
    def BAYES_FTN_PARS(self):
        return {
            'train_reconfig_entropy': self.TRAIN_RECONFIG_ENTROPY,
            'bayes_reconfig_steps': self.BAYES_RECONFIG_STEPS,
            'bayes_reconfig_lr': self.BAYES_RECONFIG_LR,
            'bayes_kl_weight': self.BAYES_KL_WEIGHT,
            'bayes_temperature': self.BAYES_TEMPERATURE,
            'bayes_prior_decay': self.BAYES_PRIOR_DECAY,
            'bayes_stochastic_mask': self.BAYES_STOCHASTIC_MASK,
            'bayes_reconfig_take_best': self.BAYES_RECONFIG_TAKE_BEST,
            'bayes_changepoint_reconfig_steps': self.BAYES_CHANGEPOINT_RECONFIG_STEPS,
            'bayes_changepoint_reconfig_lr': self.BAYES_CHANGEPOINT_RECONFIG_LR,
            'bayes_changepoint_reconfig_sample_size': self.BAYES_CHANGEPOINT_RECONFIG_SAMPLE_SIZE,
            'bayes_changepoint_kl_weight': self.BAYES_CHANGEPOINT_KL_WEIGHT,
            'bayes_changepoint_temperature': self.BAYES_CHANGEPOINT_TEMPERATURE,
            'bayes_changepoint_prior_decay': self.BAYES_CHANGEPOINT_PRIOR_DECAY,
            'bayes_changepoint_stochastic_mask': self.BAYES_CHANGEPOINT_STOCHASTIC_MASK,
        }

    @property
    def LORA_FTN_PARS(self):
        return {
            'square_dim': self.SQUARE_DIM,
            'classification': self.CLASSIFICATION,
            'n_layers': self.LORA_LAYERS,
            'width': self.LORA_WIDTH,
            'weight_scale': self.LORA_WEIGHT_SCALE,
            'rank': self.LORA_RANK,
            'alpha': self.LORA_ALPHA,
            'dropout_p': self.LORA_DROPOUT,
            'b0_scale': self.LORA_B0_SCALE,
            'use_bias': self.LORA_BIAS,
            'k': self.K,
            'reconfig_steps': self.RECONFIG_STEPS,
            'reconfig_sample_size': self.RECONFIG_SAMPLE_SIZE,
            'reconfig_noise_scale': self.RECONFIG_NOISE_SCALE,
        }

    @property
    def EVAL_FTN_PARS(self):
        return {
            'objective': 'entropy' if self.EVAL_RECONFIG_ENTROPY else 'supervised',
            'steps': self.EVAL_RECONFIG_STEPS,
            'lr': self.EVAL_RECONFIG_LR,
            'sample_size': self.EVAL_RECONFIG_SAMPLE_SIZE,
            'temperature': self.EVAL_RECONFIG_TEMPERATURE,
            'restarts': self.EVAL_RECONFIG_RESTARTS,
            'take_best': getattr(self, 'BAYES_RECONFIG_TAKE_BEST', False),
        }

class BayesLoRA(Default):
    """Alias for the default Bayesian LoRA experiment."""

class EntropyTests(Default):
    ENTROPY_TESTS = True
    ENTROPY_TEST_TRIALS = 10


class EntropyReconfig(Default):
    """Infer evaluation masks from inputs by minimizing mean output entropy."""
    EVAL_RECONFIG_ENTROPY = True
    EVAL_SAME_SUPPORT = True
    TRAIN_RECONFIG_ENTROPY = False

class EntropyReconfigTrain(Default):
    """Infer evaluation masks from inputs by minimizing mean output entropy."""
    EVAL_RECONFIG_ENTROPY = True
    EVAL_SAME_SUPPORT = True
    TRAIN_RECONFIG_ENTROPY = True


class DomainShift(Default):
    DOMAIN_SHIFT = True
    DOMAIN_SHIFT_STRIDE = 1
    DOMAIN_DENSE_PROJECTION = True
    DOMAIN_PER_TASK_PROJECTION = True

class InputConditioned(Default):
    """Route evaluation from inputs and repair archived masks using replay. Combine with DomainShift NoConceptShift for identifiable input domains."""
    IC_ENABLED = True
    IC_SUPPORT_SIZE = 256
    IC_AE_RANK = 2
    IC_RECON_THRESHOLD = 0.1
    IC_MATCH_TOLERANCE = 0.001
    IC_SAME_DOMAIN_DISTANCE = 0.05
    IC_NOVELTY_TRIGGER = False
    IC_REPLAY_REPAIR = True
    IC_REPAIR_EVERY_BLOCKS = 1
    IC_REPAIR_FIT_ROWS = 800
    IC_REPAIR_VALIDATION_ROWS = 800
    IC_REPAIR_STEPS = 100
    IC_REPAIR_LR = 0.3
    IC_REPAIR_TEMPERATURE = 1.0
    IC_REPAIR_RESTARTS = 1
    IC_REPAIR_TAKE_BEST = True
    IC_REPAIR_MIN_IMPROVEMENT = 0.0
    IC_REPAIR_LOGIT_MAGNITUDE = 4.0
    IC_TRAIN_PRIOR = False
    IC_PRIOR_STEPS = 10
    IC_PRIOR_KL = 1.0
    IC_EVAL_STRATEGY = 'refine'

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

class Replay(Default):
    N_SAMPLE_PER_TASK = 10
    N_REPLAY_SAMPLES = 1
    REPLAY_BATCH_SIZE = 256

class NoReplay(Default):
    N_SAMPLE_PER_TASK = 1
    N_REPLAY_SAMPLES = 0
    REPLAY_BATCH_SIZE = 1

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

class LowSupport(Default):
    EVAL_RECONFIG_SAMPLE_SIZE = 4

class Orth(Default):
    CORRELATED_TASKS = False

class BayesLoRAReactiveLow(BayesLoRA):
    N_TASK = 4
    BAYES_RECONFIG_STEPS = 1
    BAYES_RECONFIG_LR = 0.0
    BAYES_STOCHASTIC_MASK = False
    REPLAY_BATCH_SIZE = 0

class SpatialPermuteRandomCenter(Default):
    """Random blob layout, refined from its previous state after each task block."""
    SPATIAL_PERMUTE = True
    SPATIAL_PERMUTE_INIT = 'random_blobs'
    SPATIAL_PERMUTE_CENTER = 0.25


class Smoke(Default):
    """Small end-to-end run retaining every output and entropy probe."""
    N_TASK = 3
    SQUARE_DIM = 8
    K = 16
    LORA_LAYERS = 1
    LORA_WIDTH = 16
    LORA_RANK = 2
    LORA_ALPHA = 16.0
    T = 64
    TRAIN_DURATION = 5
    RECONFIG_SAMPLE_SIZE = 64
    EVAL_RECONFIG_STEPS = 3
    BAYES_CHANGEPOINT_RECONFIG_STEPS = 3
    N_SAMPLE_PER_TASK = 2
    N_RETRO_EVAL_BATCHES = 2
    ENTROPY_TEST_TRIALS = 2
    REPLAY_BATCH_SIZE = 64
