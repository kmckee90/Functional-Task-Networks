"""compose_config.py - consolidated config for the composition subproject.

One file, matching the ftn/config.py convention: it holds BOTH the experiment
parameters (task / difficulty / support-size sweep / capacity gate) AND the model
parameters (FTN backbone via the inherited *_PARS, plus the baseline sizes).

It subclasses base config.Default, so every base architecture and its *_PARS
(FTN, LoRA, stack/rnn variants, HML, quad) is available unchanged; this file only
ADDS the GF(2) task, the baseline knobs, and the sweep/gate schedule, and defines
the concrete experiment classes.

Selectors:
    COMPOSE_FAMILY = "gf2" | "perm"   picks the data generator (GF(2) linear+S-box
                                      vs the geometric rot/flip/shift permutation
                                      task). TASK stays "compose" either way so the
                                      loss is per-cell binary cross-entropy.
"""

import _paths  # noqa: F401  (adds ftn/ to sys.path)
import config as base_config


class Default(base_config.Default):
    """Base for the composition project: base defaults + GF(2) task + baselines
    + sweep/gate schedule. Difficulty knobs (GF2_M_DENSITY / GF2_NONLINEAR /
    GF2_SBOX_BITS / GF2_MATRIX_DIM) default to the discussed hardest setting
    (dense linear + interleaved S-box); dial them down to make the task easier."""

    # ---- Which composition generator ----
    COMPOSE_FAMILY = "gf2"        # "gf2" | "perm"

    # ---- GF(2) linear + S-box task (COMPOSE_FAMILY == "gf2") ----
    GF2_MATRIX_DIM = 5           # N; state dim d = N**2  (5 -> 25-bit state)
    GF2_STEPS = 3                # operators per composition (word length)
    GF2_N_OPS = 3                # number of primitive operators (generators)
    GF2_P = 0.5                  # input Bernoulli density
    GF2_M_DENSITY = 0.5          # target nonzero fraction of each M_i (0.5 = dense;
                                 #   lower = sparser = easier). The sparsity knob.
    GF2_M_MIN_ROW_WEIGHT = 2     # guarantee each output bit is an XOR of >= this
                                 #   many input bits (>=2 => strictly lookup-proof;
                                 #   no output bit is a copy of one input bit)
    GF2_NONLINEAR = True         # interleave a fixed reversible S-box after each
                                 #   linear step (defeats algebraic solvers). The
                                 #   nonlinearity knob; False = pure GF(2)-linear.
    GF2_SBOX_BITS = 4            # S-box block width in bits (S-box difficulty knob)
    # Intermediate (primitive) pretraining: train the weight-tied cell directly on
    # single operators (length-1 compositions, k=1 unroll via forward_k), then the
    # full length-GF2_STEPS compositions are reached by reconfiguring k steps. The
    # cleanest test of whether reconfiguration composes learned primitives.
    COMPOSE_INTERMEDIATE = False
    COMPOSE_INTERMEDIATE_STEPS = 1
    COMPOSE_INTERMEDIATE_FRACTION = 0.5   # fraction of SHUFFLE_STEPS spent on
                                          #   primitives before switching to
                                          #   full compositions (1.0 = primitives
                                          #   only, eval still on compositions)

    GF2_REJECT_IDENTITY = True
    GF2_REJECT_REDUCIBLE = True  # drop words equal to a shorter word's transform
    GF2_DEDUPE_UNIQUE = True     # keep only distinct-transform words
    GF2_N_PROBE = 256            # probe batch for distinct-transform fingerprinting

    # ---- Baseline models (composition/models_baseline.py) ----
    # Param-matching to the FTN is done at sweep time; these are the shape knobs.
    BASE_MLP_WIDTH = 256
    BASE_MLP_LAYERS = 8
    BASE_RNN_WIDTH = 128
    BASE_RNN_STEPS = 4           # recurrent steps used as processing depth
    DEEPSET_WIDTH = 256          # in-context DeepSet hidden width
    DEEPSET_LAYERS = 3
    # In-context baseline training (run_baselines.py): trained across the training
    # tasks with random support sizes so it can be evaluated at any sweep B.
    BASE_TRAIN_STEPS = 4000
    BASE_TRAIN_LR = 3e-4
    BASE_B_MIN = 2               # min support size seen in in-context training
    BASE_B_MAX = 256             # max support size seen in in-context training
    BASE_QUERY = 256             # query batch size during in-context training
    # Per-task test-time-trained baselines (bench.py --ttt): fit a fresh net on the
    # B support samples at eval (matched to the FTN's test-time adaptation).
    TTT_STEPS = 400
    TTT_LR = 3e-3

    # ---- Support-size sweep (bench.py) ----
    # Accuracy vs support size B: FTN vs oracle ceiling vs baselines vs floors.
    SWEEP_SUPPORT_SIZES = [4, 8, 16, 25, 32, 64, 128, 256, 512]
    SWEEP_QUERY_SIZE = 512       # fresh query batch scored per (task, B)
    SWEEP_EVAL_TASKS = None      # tasks per split (None = all train / all holdout)
    SWEEP_SEEDS = [0, 1, 2]      # repeats for error bars

    # ---- Capacity gate (capacity_gate.py) ----
    # Can the architecture even REPRESENT a single primitive f_i at the intended
    # 1-step-per-operator budget? Trains weights to fit ONE primitive with
    # unlimited fresh data at each depth, reports best accuracy.
    GATE_DEPTHS = [1, 3, 4]      # per-operator layer budget n to fit ONE primitive
                                 #   (FTN deep cell at k=1; MLP n layers; RNN n steps)
    GATE_STEPS = 2500            # training steps for the gate fit
    GATE_LR = 1e-3               # baseline optimizer lr
    GATE_FTN_LR = 1e-3           # FTN gate-fit lr (a fair capacity probe, decoupled
                                 #   from the low TRAIN_LR used in the real sweep)
    GATE_RECONFIG_EVERY = 50     # FTN gate: re-search the mask every N weight steps
                                 #   (train weights under the fixed mask in between).
                                 #   Representability probe -- ~N x faster than a
                                 #   full mask search every step.
    GATE_EVAL_BATCH = 4096       # fresh batch to score the fitted primitive
    GATE_TARGET_ACC = 0.99       # per-cell accuracy considered "represented"

    @property
    def GF2_PARS(self):
        """Everything DataGenGF2 takes besides n_tasks / n_samples / rng / seed
        (which the drivers pass directly, mirroring COMP_PARS)."""
        return {
            "matrix_dim": self.GF2_MATRIX_DIM,
            "steps": self.GF2_STEPS,
            "n_ops": self.GF2_N_OPS,
            "p": self.GF2_P,
            "m_density": self.GF2_M_DENSITY,
            "m_min_row_weight": self.GF2_M_MIN_ROW_WEIGHT,
            "nonlinear": self.GF2_NONLINEAR,
            "sbox_bits": self.GF2_SBOX_BITS,
            "reject_identity": self.GF2_REJECT_IDENTITY,
            "reject_reducible": self.GF2_REJECT_REDUCIBLE,
            "dedupe_unique": self.GF2_DEDUPE_UNIQUE,
            "n_probe": self.GF2_N_PROBE,
            "classify": self.BINARY,
        }


class GF2Compose(Default):
    """Main GF(2) composition experiment on the FTN-LoRA-RNN backbone.

    OPERATING REGIME (chosen after the capacity gate, 2026-09-10): 16-bit state
    (4x4), n_ops sparse invertible GF(2)-LINEAR primitives (~2 XORs/bit), no
    S-box, length-3 compositions. This is cleanly learnable (FTN fits a primitive
    to ~1.0), lookup-proof (each output bit is an XOR of >1 input bit), and
    zero-redundancy. Dropping the S-box loses closed-form-solver resistance, but
    the linear solver becomes the exact Bayes-optimal baseline (rank cap with a
    sharp phase transition at support size B ~= d). The dense / S-box settings in
    Default remain available as harder ablations.

    Model: k = STACK_LAYERS = 3 recurrent steps (one per operator); each step is
    an n = RNN_CELL_LAYERS = 3-layer deep LoRA MLP with ReLU."""
    TASK = "compose"
    COMPOSE_FAMILY = "gf2"

    GF2_MATRIX_DIM = 4          # d = 16
    GF2_STEPS = 3
    GF2_N_OPS = 3
    GF2_P = 0.5
    GF2_M_DENSITY = 0.06        # very sparse; the min-row-weight floor then sets
    GF2_M_MIN_ROW_WEIGHT = 2    #   ~2 XOR terms per output bit (strictly lookup-proof)
    GF2_NONLINEAR = False       # linear (learnable; still lookup-proof)

    ARCHITECTURE = "lorarnn"
    FTNLORA_IO = "ftn"
    STACK_LAYERS = 3            # k: recurrent steps == composition depth (3)
    RNN_CELL_LAYERS = 3        # n: feed-forward layers per step (deep ReLU cell)
    RNN_CELL_INNER_ACT = "relu"
    RNN_STEP_ACT = "tanh"
    SQUARE_DIM = 8              # atom grid; atom count = 64
    K = 16                      # kWTA budget (active atoms per mask)
    LORA_WIDTH = 64            # recurrent hidden width
    LORA_RANK = 1
    LORA_ALPHA = 16.0
    LORA_B0_SCALE = 0.01

    RECONFIG_ALGORITHM = "annealste"
    RECONFIG_STEPS = 30
    RECONFIG_LR = 0.3
    SHUFFLE_EVAL_RECONFIG_STEPS = 120

    T = 512
    TRAIN_LR = 3e-4
    SHUFFLE_N_TRAIN = 12
    SHUFFLE_N_HOLDOUT = 4
    SHUFFLE_TASKS_PER_STEP = 8
    SHUFFLE_STEPS = 4000
    SHUFFLE_EVAL_EVERY = 200
    SHUFFLE_EVAL_BATCHES = 1


class GF2ComposeRNNResid(GF2Compose):
    """Same GF(2) task on the RNN-residual FTN (weight-tied cell + per-step
    masks) instead of the LoRA-RNN."""
    ARCHITECTURE = "rnn_residual"


class Big(Default):
    """Capacity+optimization mixin (compose AFTER a GF2 config): more atoms, a
    larger kWTA budget, wider cell, higher LR, more steps + more reconfig. For
    ruling out under-fitting. e.g. --config GF2Compose Big."""
    SQUARE_DIM = 12             # 144 atoms
    K = 32
    LORA_WIDTH = 96
    LORA_RANK = 8
    TRAIN_LR = 1e-3
    RECONFIG_STEPS = 60
    SHUFFLE_EVAL_RECONFIG_STEPS = 200
    SHUFFLE_STEPS = 6000


class Intermediate(Default):
    """Enable primitive pretraining (compose AFTER a GF2 config):
    --config GF2Compose Intermediate."""
    COMPOSE_INTERMEDIATE = True
    COMPOSE_INTERMEDIATE_STEPS = 1
    COMPOSE_INTERMEDIATE_FRACTION = 0.5


class GF2Smoke(Default):
    """Size-only mixin for a fast smoke / gate check. Compose AFTER a GF2 config,
    e.g. --config GF2Compose GF2Smoke."""
    GF2_MATRIX_DIM = 4
    SQUARE_DIM = 6
    K = 6
    LORA_WIDTH = 32
    STACK_LAYERS = 3
    RECONFIG_STEPS = 6
    T = 128
    SHUFFLE_N_TRAIN = 4
    SHUFFLE_N_HOLDOUT = 2
    SHUFFLE_TASKS_PER_STEP = 3
    SHUFFLE_STEPS = 20
    SHUFFLE_EVAL_EVERY = 10
    SHUFFLE_EVAL_BATCHES = 1
    GATE_DEPTHS = [1, 2]
    GATE_STEPS = 200
    SWEEP_SUPPORT_SIZES = [8, 32, 128]
    SWEEP_SEEDS = [0]


def resolve(names):
    """Build a Config instance from one or more class names (left-to-right, later
    overrides earlier), looking them up in this module first, then base config."""
    bases = []
    for n in names:
        cls = getattr(__import__("compose_config"), n, None) or getattr(base_config, n, None)
        if cls is None:
            raise SystemExit(f"Unknown config class: {n!r}")
        bases.append(cls)
    return type("+".join(names), tuple(reversed(bases)), {})()
