"""
config.py - Configuration for the piecewise continual-learning experiment.

Mirrors ftn/config.py conventions. This experiment differs from base ONLY in
what a "task" is:

  base:      each task_idx is a DIFFERENT latent function (concept shift).
  piecewise: each task_idx is a contiguous SUB-DOMAIN of ONE shared, highly
             nonlinear latent-sinusoid function. The FTN must cover the whole
             function by learning its sub-domains block-sequentially, and at
             inference reconfigure its mask from a small sample of whichever
             sub-domain it is shown.

Piecewise task knobs
--------------------
N_TASK        Number of sub-domains the input range is split into (= number of
              block-sequentially presented tasks).
N_COMPONENTS  Number of latent sinusoid components summed to form the function
              (function complexity -- the "number of components").
FREQUENCY     Sine frequency (function hardness), as in ftn/config.py.
DOMAIN_HALF   Half-width of the bounded input box [-DOMAIN_HALF, DOMAIN_HALF];
              the split dimension is partitioned into N_TASK equal slabs of it.
SPLIT_DIM     Input axis along which the domain is partitioned.

Every other field is an FTN model / mask / schedule hyperparameter, identical
in meaning to ftn/config.py -- see that file's glossary for the full list. The
FTN model and mask machinery are imported unmodified from ../base.
"""

import ftn.projects.piecewise.masks as masks  # snapshot of the pre-v2 ftn/masks.py, this directory


class Default:
    """Default piecewise experiment configuration."""

    # Experiment pars
    SEED = 123
    N_TASK = 3  # number of sub-domains (block-sequential tasks)
    # 1: piecewise FTN fit; 2: simultaneous baseline (one fixed mask, whole
    # domain, TRAIN_DURATION*N_TASK steps) for a piecewise-vs-simultaneous
    # comparison. Run both to populate the comparison summary.
    PHASES = [1, 2]

    # Piecewise function / domain
    #
    # DOMAIN_HALF is kept small so inputs stay near zero: the base FTN model has
    # no input normalization, so far-from-origin inputs (which the outer
    # sub-domains would otherwise carry) can't be fit -- masking selects units
    # but can't add capacity for off-center inputs. FREQUENCY is raised to keep
    # the function highly nonlinear (~several sine periods) across the smaller
    # box. See git history / the piecewise README for the diagnosis.
    N_COMPONENTS = 24  # latent sinusoid components (function complexity)
    FREQUENCY = 128  # sine frequency (function hardness)
    DOMAIN_HALF = 0.5  # input box half-width (kept small for model stability)
    SPLIT_DIM = 0  # axis partitioned into sub-domains

    # Data generator
    CLASSIFICATION = True
    X_DIM = 2
    Y_DIM = 2
    T = 256
    DATA_TEMP = 1e-8

    # Model hyperpars (identical to base)
    K = 256
    SQUARE_DIM = 48
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
    # Off-center (outer) sub-domains converge more slowly (input-mean
    # ill-conditioning, ~12x here) but do converge -- they reach ~0.11 loss by
    # ~3000 steps in isolation, vs looking "stalled" at ~500 steps. Give enough
    # steps for the outer slabs, not just the center one.
    TRAIN_DURATION = 500
    N_EPOCHS = 1
    N_SAMPLE_PER_TASK = 10
    N_EVAL_EPOCHS = 1
    N_REPLAY_SAMPLES = 1
    REPLAY_BATCH_SIZE = 256
    SHUFFLE_ON_REPLAY = False

    # Backwards eval (forgetting monitor)
    EVAL_BACKWARDS = True
    EVAL_PROSPECTIVE = True
    N_RETRO_EVAL_BATCHES = 10

    # Training hyperpars
    TRAIN_LR = 1e-3
    WEIGHT_DECAY = 0.0

    # Plotting
    # SAVE_TRAINING_VIS gates the per-step training visualizations produced in
    # run_phase -- the bw_*/rgb_* mask GIFs+PNGs and the loss_* curve. Set False
    # to skip them (and the per-step frame building) for a faster run. The
    # func_* reconstruction and xfer_* forgetting matrix are unaffected.
    SAVE_TRAINING_VIS = False
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
