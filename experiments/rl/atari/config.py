"""
config.py - Configuration for Atari continual learning with masked DQN.

All masking parameters mirror those in the parent project (func_net_2/config.py).
Atari/DQN-specific parameters are added for the RL loop.
"""

# fmt: off

# ── Default game sequence ──────────────────────────────────────────────
# A curated subset that covers diverse game mechanics.
# Override via command-line or by editing GAME_SEQUENCE below.
DEFAULT_GAMES = [
    "Pong",
    "Breakout",
    "SpaceInvaders",
    "Seaquest",
    "Qbert",
    "BeamRider",
    "Enduro",
    "MsPacman",
    "Asterix",
    "Freeway",
]

DEFAULT_MINATAR_GAMES = [
    "space_invaders",
    "breakout",
    # "seaquest",
    # "asterix",
    # "freeway",
]


class AtariConfig:
    """Configuration for masked-DQN continual learning on Atari."""

    SEED = 42

    # ── MinAtar mode ──────────────────────────────────────────────────
    USE_MINATAR = True               # set True (or --minatar) to use MinAtar

    # ── Game / curriculum ──────────────────────────────────────────────
    GAME_SEQUENCE = DEFAULT_GAMES     # ordered list of game names
    MAX_STEPS_PER_GAME = 1_000_000    # env steps budget per game
    EVAL_EVERY_N_GAMES = 1            # eval on all learned games every N games
    EVAL_EPISODES = 50                # episodes per game during eval
    WARMUP_STEPS = 0             # random actions before learning starts (per game)
    CYCLES = 10
    
    # ── Environment ────────────────────────────────────────────────────
    FRAME_STACK = 4
    FRAME_SKIP = 4          # handled by ALE frameskip=4 default in v5
    IMG_SIZE = 84
    GRAYSCALE = True
    REWARD_CLIP = True       # clip rewards to {-1, 0, +1}
    EPISODIC_LIFE = True     # treat life loss as episode end during training

    # ── DQN hyperparameters ────────────────────────────────────────────
    BATCH_SIZE = 256
    GAMMA = 0.90
    LR = 1e-4
    WEIGHT_DECAY = 0.0
    TARGET_UPDATE_FREQ = 10       # gradient steps between target net syncs
    TRAIN_FREQ = 1                    # env steps between training bursts
    GRADIENT_STEPS = 1                 # gradient steps (batches) per training burst
    GRAD_CLIP = 1.0                  # max gradient norm

    # Epsilon-greedy schedule (per game)
    EPS_START = 0.1
    EPS_END = 0.1
    EPS_DECAY_STEPS = 20_000        # linear decay over this many steps

    # ── Replay buffer ──────────────────────────────────────────────────
    REPLAY_CAPACITY = 10_000         # transitions (kept modest for memory)
    MIN_REPLAY_SIZE = 1_000          # min transitions before training starts

    # Mask-correlation selective replay (from parent project)
    MASK_REPLAY_ENABLED = True
    MASK_REPLAY_CORR_LOW = 0.15       # min mask correlation for replay sample
    MASK_REPLAY_CORR_HIGH = 1.0       # max mask correlation for replay sample
    N_REPLAY_SAMPLES = 100              # extra replay samples per training step
    N_SAMPLE_PER_GAME = 10_000         # how many transitions per game go into masked replay

    # ── Model / masking ────────────────────────────────────────────────
    # Hidden layer is a 2D cortical grid, same as parent project
    SQUARE_DIM = 16                   # side length → hidden_size = 32² = 1024
    K = 32                           # KWTA winners (active neurons per game)
    FC_LAYERS = 2                     # number of masked FC layers after CNN
    DROPOUT = 0.0

    # Lateral dynamics (applied in model forward pass)
    LATERAL_B = 0.0
    LATERAL_STEPS = 0
    KERNEL_SIZE = 3
    TORUS = False

    # Mask reconfiguration (per-game mask discovery)
    DISABLE_MASK = False
    CONFIGURE_PER_BATCH = False        # if True, reconfigure mask every batch
    RECONFIG_STEPS = 10                # gradient steps for mask optimization
    RECONFIG_LR = 0.2                 # RMSprop lr for mask optimization
    RECONFIG_SAMPLE_SIZE = 256        # max samples used for mask reconfig
    EVAL_CYCLES = 8

    # ── Logging / plotting ─────────────────────────────────────────────
    LOG_EVERY_STEPS = 5_000
    PLOT_EVERY_GAMES = 1
    CHECKPOINT_EVERY_GAMES = 5
    LOSS_EMA_SMOOTHING = 0.99
    GIF_DURATION = 50
    GIF_SAVE_EPOCH = 5
    PLOT_EVERY = 1                    # for mask vis compatibility

    # ── Derived ────────────────────────────────────────────────────────
    @property
    def HIDDEN_SIZE(self):
        return self.SQUARE_DIM ** 2

    @property
    def N_GAMES(self):
        return len(self.GAME_SEQUENCE)

    @property
    def MODEL_PARS(self):
        return {
            "b": self.LATERAL_B,
            "steps": self.LATERAL_STEPS,
            "layers": self.FC_LAYERS,
            "kernel_size": self.KERNEL_SIZE,
            "dropout": self.DROPOUT,
        }

    @property
    def MASK_PARS(self):
        return {
            "k": self.K,
            "torus": self.TORUS,
            "reconfig_steps": self.RECONFIG_STEPS,
            "reconfig_lr": self.RECONFIG_LR,
            "reconfig_sample_size": self.RECONFIG_SAMPLE_SIZE,
        }

    def enable_minatar(self):
        """Switch configuration to MinAtar defaults."""
        self.USE_MINATAR = True
        self.GAME_SEQUENCE = DEFAULT_MINATAR_GAMES
        self.FRAME_STACK = 1               # MinAtar state is already informative
        self.FRAME_SKIP = 1                # MinAtar handles its own timing
        self.IMG_SIZE = 10
        self.GRAYSCALE = False             # MinAtar uses multi-channel binary obs
        self.REWARD_CLIP = False           # MinAtar rewards are already simple
        self.EPISODIC_LIFE = False         # no lives mechanic in MinAtar
        self.MAX_STEPS_PER_GAME = 5000
        self.EPS_DECAY_STEPS = 1000
        self.WARMUP_STEPS = 0
        self.REPLAY_CAPACITY = 10_000
        self.RECONFIG_SAMPLE_SIZE = 256
