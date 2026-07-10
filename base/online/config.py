"""
config.py - Configuration for the online (point-wise) FTN experiment.

Mirrors base/config.py conventions. This experiment differs from base in what
a "task" is:

  base:   each task_idx is a different latent function; batches of samples.
  online: there are no tasks. A single fixed set of N_POINTS individual
          (x, y) pairs is drawn once from ONE function. Phase 1 presents the
          points SEQUENTIALLY, overfitting the model (to convergence, each)
          on one pair at a time with per-batch mask reconfiguration -- each
          point gets its own mask-routed subspace. Phase 2 then reconfigures
          on BATCHES of those same points (all together, or EVAL_BATCH_SIZE
          at a time) under different reconfig settings, asking whether
          reconfiguration composes the point-wise subspaces into the function.

Train-time and eval-time reconfiguration parameters are deliberately separate:
TRAIN_* feeds TRAIN_MASK_PARS (phase 1, single-point reconfig), EVAL_* feeds
EVAL_MASK_PARS (phase 2, batched reconfig). EVAL_SETTINGS then lets one run
sweep several eval-reconfig variants.

Every field not listed below is an FTN model / data hyperparameter identical
in meaning to base/config.py -- see that file's glossary.

Point set
---------
N_POINTS                Number of individual (x, y) pairs drawn (once) from the
                        function. These are BOTH the sequential training stream
                        (phase 1) and the eval batch pool (phase 2).

Phase 1: point-wise overfitting
-------------------------------
MAX_STEPS_PER_POINT     Gradient-step budget per point; training on a point
                        stops early once CONVERGENCE_LOSS is reached.
CONVERGENCE_LOSS        Loss threshold declaring a point "overfit".
TRAIN_CONFIGURE_PER_STEP  If True, reconfigure the mask at every gradient step
                        (the per-batch analog for a batch of one point); if
                        False, reconfigure once when the point is first shown
                        and keep that mask for the whole overfit.
RESET_OPTIMIZER_PER_POINT  If True, clear Adam state between points (matches
                        base's per-task optimizer reset).
TRAIN_K                 kwta budget for train-time (single-point) reconfig.
TRAIN_RECONFIG_STEPS    Reconfig optimization steps (train).
TRAIN_RECONFIG_LR       Reconfig optimizer lr (train).
TRAIN_RECONFIG_SAMPLE_SIZE  Rows used by the configurer (train). NOTE: values
                        < 2 break ParallelMLP's squeeze; single points are
                        already replicated to 2 rows by the runner.
XFER_MATRIX             If True, build the retro+forward transfer matrix over
                        individual points (xfer_{tag}.png, the point analog of
                        base's forgetting matrix): after every XFER_EVERY-th
                        point's overfit, EVERY point is scored under a fresh
                        single-point reconfig (TRAIN_MASK_PARS). The reconfig
                        support for a point IS that point (deliberately: this
                        asks whether the stored overfit solution is
                        retrievable). Costs an extra N_POINTS reconfigs per
                        row -- O(N_POINTS^2 / XFER_EVERY) total.
XFER_EVERY              Add a matrix row after every k-th point (1 = after
                        every point, as in base).

Phase 2: batched reconfiguration eval
-------------------------------------
EVAL_BATCH_SIZE         Points per eval batch. None (or >= N_POINTS) = one
                        batch containing every stored point; smaller values
                        partition the point set into sequential chunks and the
                        model must compose each chunk with one mask.
N_EVAL_RECONFIGS        Independent reconfigurations per batch (reconfig is
                        stochastic); scores are averaged across them.
EVAL_K                  kwta budget for eval-time (batched) reconfig.
EVAL_RECONFIG_STEPS     Reconfig optimization steps (eval).
EVAL_RECONFIG_LR        Reconfig optimizer lr (eval).
EVAL_RECONFIG_SAMPLE_SIZE  Rows used by the configurer (eval).
EVAL_SETTINGS           {name: overrides} -- one full eval pass per entry, with
                        the overrides merged on top of EVAL_MASK_PARS. This is
                        where eval-reconfig variants are swept, e.g.
                        {"default": {}, "wide_k": {"k": 512},
                         "more_steps": {"reconfig_steps": 100}}. The special
                        key "eval_batch_size" overrides EVAL_BATCH_SIZE for
                        that setting only (it is not a configurer kwarg).

Phase 3: reference feed-forward baseline
----------------------------------------
REFERENCE_FF            If True, also train a FRESH copy of the same model with
                        the mask fixed to the first EVAL_K units, JOINTLY on
                        the full N_POINTS batch (it will overfit them -- that
                        is the intended reference), and report its batched
                        loss as "reference_ff" in the summary.
REF_TRAIN_STEPS         Gradient-step budget for the reference fit; stops early
                        once the batch loss reaches CONVERGENCE_LOSS.
"""

import os
import sys

_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BASE not in sys.path:
    sys.path.append(_BASE)

import masks  # imported from ../base


class Default:
    """Default online (point-wise) experiment configuration."""

    # Experiment pars
    SEED = 123

    # Data generator (one function). Y_DIM is dictated by CLASSIFICATION:
    # True -> Y_DIM=2 (DataGenSine's classify decoder always emits 2-column
    # one-hot), False -> Y_DIM=1 (regression decoder always emits 1 column).
    # The runner validates the pairing at startup.
    CLASSIFICATION = False
    X_DIM = 2
    Y_DIM = 1
    TASK_SIZE = 4
    FREQUENCY = 8
    DATA_TEMP = 1e-8

    # Point set
    N_POINTS = 32

    # Model hyperpars (identical to base)
    SQUARE_DIM = 48
    LAYERS = 8
    DROPOUT = 0.2
    INNER_DIM = 8

    MASK_CONFIGURER = masks.KWTA
    LATERAL_B = 0.0
    LATERAL_STEPS = 0
    KERNEL_SIZE = 5
    TORUS = False

    # Phase 1: point-wise overfitting
    MAX_STEPS_PER_POINT = 500
    CONVERGENCE_LOSS = 1e-3
    TRAIN_CONFIGURE_PER_STEP = True
    RESET_OPTIMIZER_PER_POINT = True
    TRAIN_LR = 3e-4
    WEIGHT_DECAY = 0.0

    # Train-time (single-point) reconfig params
    TRAIN_K = 16
    TRAIN_RECONFIG_STEPS = 30
    TRAIN_RECONFIG_LR = 0.3
    TRAIN_RECONFIG_SAMPLE_SIZE = 256

    # Retro + forward transfer matrix over points (optional; costs
    # N_POINTS/XFER_EVERY x N_POINTS extra reconfigs).
    XFER_MATRIX = True
    XFER_EVERY = 1

    # Phase 2: batched reconfiguration eval
    EVAL_BATCH_SIZE = None  # None = all N_POINTS in one batch
    N_EVAL_RECONFIGS = 3

    # Eval-time (batched) reconfig params
    EVAL_K = 256
    EVAL_RECONFIG_STEPS = 30
    EVAL_RECONFIG_LR = 0.3
    EVAL_RECONFIG_SAMPLE_SIZE = 256

    # One eval pass per entry; values override EVAL_MASK_PARS per-key.
    # "eval_batch_size" is special-cased to override EVAL_BATCH_SIZE.
    EVAL_SETTINGS = {
        "default": {},
        # "wide_k": {"k": 512},
        # "more_steps": {"reconfig_steps": 100},
        # "slow_lr": {"reconfig_lr": 0.05},
        # "pairs": {"eval_batch_size": 2},
    }

    # Phase 3: reference feed-forward baseline (fixed first-EVAL_K-units mask,
    # trained jointly on the full N_POINTS batch; overfitting is expected).
    REFERENCE_FF = True
    REF_TRAIN_STEPS = 5000

    # Plotting
    LOSS_EMA_SMOOTHING = 0.8
    # Eval-fit figure (fit_{setting}_b{batch}.png): 3 panels -- true function,
    # model prediction surface under the eval mask (+ eval batch points), and
    # the eval mask raster. Adapted from base/vis_ftn.py's render_panel.
    GRID_RES = 120  # prediction-surface grid resolution (GRID_RES x GRID_RES)
    PLOT_RANGE = 2.0  # input domain shown: [-PLOT_RANGE, PLOT_RANGE]^2

    @property
    def HIDDEN_SIZE(self):
        return self.SQUARE_DIM**2

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
    def TRAIN_MASK_PARS(self):
        return {
            "k": self.TRAIN_K,
            "torus": self.TORUS,
            "reconfig_steps": self.TRAIN_RECONFIG_STEPS,
            "reconfig_lr": self.TRAIN_RECONFIG_LR,
            "reconfig_sample_size": self.TRAIN_RECONFIG_SAMPLE_SIZE,
        }

    @property
    def EVAL_MASK_PARS(self):
        return {
            "k": self.EVAL_K,
            "torus": self.TORUS,
            "reconfig_steps": self.EVAL_RECONFIG_STEPS,
            "reconfig_lr": self.EVAL_RECONFIG_LR,
            "reconfig_sample_size": self.EVAL_RECONFIG_SAMPLE_SIZE,
        }


class Smoke(Default):
    """Tiny configuration for a fast CPU smoke test of the pipeline."""

    SQUARE_DIM = 16
    LAYERS = 2
    INNER_DIM = 4

    N_POINTS = 4
    MAX_STEPS_PER_POINT = 50

    TRAIN_K = 32
    TRAIN_RECONFIG_STEPS = 5

    EVAL_K = 32
    EVAL_RECONFIG_STEPS = 5
    N_EVAL_RECONFIGS = 2

    EVAL_SETTINGS = {
        "default": {},
        "pairs": {"eval_batch_size": 2},  # exercises the chunking path
    }


    XFER_MATRIX = True  # cheap at N_POINTS=4; exercises the matrix path


class SmokeClassification(Smoke):
    """Smoke test of the classification path."""

    CLASSIFICATION = True
    Y_DIM = 2
