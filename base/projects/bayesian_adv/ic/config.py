"""Local experiment overrides; the parent project's configs remain untouched."""

import importlib.util
from pathlib import Path


_source = Path(__file__).resolve().parent.parent / "config.py"
_spec = importlib.util.spec_from_file_location("bayesian_adv_parent_config", _source)
_parent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parent)

for _name in dir(_parent):
    if _name[0].isupper():
        globals()[_name] = getattr(_parent, _name)


class ICExperiment:
    """Low-support DomainShift experiment with the same training pipeline."""

    # Run both modes with this configuration; only IC_MODE differs.
    N_TASK = 4
    CONCEPT_SHIFT = False
    N_EPOCHS = 2
    T = 64
    TRAIN_DURATION = 120
    N_SAMPLE_PER_TASK = 8
    REPLAY_BATCH_SIZE = 16
    N_RETRO_EVAL_BATCHES = 4
    N_EVAL_EPOCHS = 1
    EVAL_PROSPECTIVE = False
    RENDER_STEPS = False
    ENTROPY_TESTS = False

    SQUARE_DIM = 16
    K = 64
    LORA_WIDTH = 96
    LORA_LAYERS = 4
    N_STACK = 4
    LORA_ALPHA = 64.0

    BAYES_CHANGEPOINT_RECONFIG_SAMPLE_SIZE = 4
    EVAL_RECONFIG_SAMPLE_SIZE = 4
    BAYES_CHANGEPOINT_RECONFIG_STEPS = 12
    EVAL_RECONFIG_STEPS = 12
    IC_SUPPORT_SIZE = 4
    IC_AE_RANK = 2
    IC_RECON_THRESHOLD = 0.10
    IC_TRIGGER = True
    IC_PRIOR_STEPS = 12
    IC_PRIOR_KL = 0.1
    IC_EVAL_STRATEGY = "refine"
    IC_MODE = "none"
    IC_TRAIN_PRIOR = True
    IC_EVAL_PRIOR = True
    IC_REFRESH = False
    IC_REFRESH_ROWS = 256
    IC_REFRESH_STEPS = 30
    IC_REFRESH_KL = 0.1
    IC_REFRESH_METHOD = "warm"
    IC_REPLAY_REPAIR = False
    IC_REPAIR_FIT_ROWS = 800
    IC_REPAIR_STEPS = 100
    IC_REPAIR_RESTARTS = 3


class ICPrior:
    IC_MODE = "reconstruction"


class ICNoPrior:
    IC_MODE = "none"


class ICTrainOnly:
    IC_MODE = "reconstruction"
    IC_TRAIN_PRIOR = True
    IC_EVAL_PRIOR = False


class ICEvalOnly:
    IC_MODE = "reconstruction"
    IC_TRAIN_PRIOR = False
    IC_EVAL_PRIOR = True


class ICSmoke:
    N_TASK = 2
    N_EPOCHS = 1
    TRAIN_DURATION = 4
    N_RETRO_EVAL_BATCHES = 1
    SQUARE_DIM = 4
    K = 4
    LORA_WIDTH = 16
    LORA_LAYERS = 2
    N_STACK = 2
    LORA_ALPHA = 4.0
    BAYES_CHANGEPOINT_RECONFIG_STEPS = 2
    EVAL_RECONFIG_STEPS = 2
    IC_PRIOR_STEPS = 1


class ICFull:
    """Parent-scale training; only retrospective mask support stays at four."""
    N_TASK = 6
    N_EPOCHS = 1
    T = 256
    TRAIN_DURATION = 1500
    N_SAMPLE_PER_TASK = 100
    SQUARE_DIM = 64
    K = 1024
    LORA_WIDTH = 256
    LORA_LAYERS = 8
    LORA_ALPHA = 256.0
    BAYES_CHANGEPOINT_RECONFIG_SAMPLE_SIZE = None
    BAYES_CHANGEPOINT_RECONFIG_STEPS = 30
    IC_TRIGGER = False
    EVAL_RECONFIG_STEPS = 30
    IC_PRIOR_STEPS = 30
    IC_DIAGNOSTIC_TRIALS = 4


class ICRNNFull:
    """Use the parent recurrent training duration with ICFull capacity."""
    TRAIN_DURATION = 500


class ICRefresh:
    """Refresh old mask values from labeled replay, preserving input keys."""
    IC_REFRESH = True


class ICColdRefresh:
    """Re-search old masks from zero on cached labeled training examples."""
    IC_REFRESH = True
    IC_REFRESH_METHOD = "cold"


class ICDirect:
    """Use the retrieved hard mask as-is at evaluation."""
    IC_EVAL_STRATEGY = "direct"


class ICReplayRepair:
    """Relearn archived mask values from routed replay; use them directly."""
    IC_REPLAY_REPAIR = True
    IC_EVAL_STRATEGY = "direct"


class ICChangepointReplay:
    """Refresh replay slots with recent data under the outgoing hard mask."""
    IC_CP_REPLAY_REFRESH = True
    IC_CP_REPLAY_TAIL_BATCHES = 100


class ICChangepointReplayPriority:
    """Spend the usual one replay batch on an old late-mask snapshot."""
    IC_CP_REPLAY_PRIORITY = True
    IC_CP_REPLAY_PRIORITY_EVERY = 1


class ICChangepointReplayPriorityHalf:
    """Reserve every other replay step for old late-mask snapshots."""
    IC_CP_REPLAY_PRIORITY = True
    IC_CP_REPLAY_PRIORITY_EVERY = 2


class ICChangepointReplaySmoke:
    """Small forced-change check for the storage and repair paths."""
    RECONFIG_REACTIVE_SD = -100.0
    RECONFIG_REACTIVE_EMA = 0.0
    N_SAMPLE_PER_TASK = 4
    IC_CP_REPLAY_TAIL_BATCHES = 4
    IC_REPAIR_FIT_ROWS = 16
    IC_REPAIR_STEPS = 2
    IC_REPAIR_RESTARTS = 1


class ICNoRepairDirect:
    """Direct input retrieval of the archived mask, without replay repair."""
    IC_REPLAY_REPAIR = False
    IC_EVAL_STRATEGY = "direct"


class ICGradReplay0:
    """Retain the full replay buffer but use no replay gradients."""
    N_REPLAY_SAMPLES = 0


class ICGradReplay2:
    N_REPLAY_SAMPLES = 2


class ICGradReplay4:
    N_REPLAY_SAMPLES = 4


class ICGradReplay8:
    N_REPLAY_SAMPLES = 8


class ICGradReplay8Normalized:
    """Eight replay batches, with their total loss weight held at one."""
    N_REPLAY_SAMPLES = 8
    IC_REPLAY_LOSS_SCALE = 0.125


class ICRepairHeavy:
    """Extra search budget for the no-gradient-replay repair ablation."""
    IC_REPAIR_STEPS = 200
    IC_REPAIR_RESTARTS = 5


class ICReplayLogits:
    """Retrieve train-time posterior logits from ordinary replay inputs."""
    REPLAY_LOGIT_PRIOR_ENABLED = True
    IC_MODE = "none"
    IC_TRIGGER = False
    IC_REPLAY_REPAIR = False
    IC_REFRESH = False
    IC_DIAGNOSTIC_TRIALS = 0
    IC_SUPPORT_SIZE = 4
    BAYES_CHANGEPOINT_RECONFIG_SAMPLE_SIZE = None
    EVAL_RECONFIG_SAMPLE_SIZE = 4
    RP_MATCH = "nearest"  # nearest sample or replay-batch subspace
    RP_SUBSPACE_RANK = 2
    RP_TIE_TOLERANCE = 0.0
    RP_NEIGHBORS = 1
    RP_EVAL_STRATEGY = "direct"  # direct, refine, or zero
    RP_PRIOR_STEPS = 30
    RP_PRIOR_KL = 0.1
    RP_PRIOR_TEMPERATURE = 1.0


class ICReplayLogitsSubspace:
    RP_MATCH = "subspace"
    RP_TIE_TOLERANCE = 1e-3


class ICReplayLogitsAverage:
    """Average logits from eight nearest replay batches."""
    RP_NEIGHBORS = 8


class ICReplayLogitsRefine:
    RP_EVAL_STRATEGY = "refine"


class ICReplayLogitsZero:
    RP_EVAL_STRATEGY = "zero"


class ICAutoencoder:
    """Base x->x mask gates a separate x->prior-logits FTN; eval-only prior."""
    AE_PRIOR_ENABLED = True
    IC_MODE = "none"
    IC_TRIGGER = False
    IC_REPLAY_REPAIR = False
    IC_REFRESH = False
    IC_DIAGNOSTIC_TRIALS = 0
    IC_SUPPORT_SIZE = 4
    BAYES_CHANGEPOINT_RECONFIG_SAMPLE_SIZE = None
    EVAL_RECONFIG_SAMPLE_SIZE = 4

    AE_SQUARE_DIM = 8
    AE_K = 16
    AE_LAYERS = 2
    AE_INNER_DIM = 8
    AE_RECONFIG_ALGORITHM = "annealste"
    AE_RECONFIG_STEPS = 5
    AE_RECONFIG_LR = 0.3
    AE_TRAIN_LR = 1e-3
    AE_REACTIVE_SD = 5.0
    AE_REACTIVE_EMA = 0.9
    AE_EVAL_RECONFIG_STEPS = 30
    AE_EVAL_RECONFIG_LR = 0.3
    AE_EVAL_TEMPERATURE = 1.0
    AE_EVAL_RESTARTS = 2

    AE_LOGIT_CACHE_ROWS = 256
    AE_LOGIT_TRAIN_STEPS = 100
    AE_LOGIT_BATCH_SIZE = 64
    AE_LOGIT_LR = 1e-3
    AE_LOGIT_CLIP = 3.0
    AE_LOGIT_SCALE = 2.0
    AE_LOGIT_FREEZE_OLD = False
    AE_TEACHER_MODE = "posterior"  # posterior or cold_mask
    AE_TEACHER_STEPS = 100
    AE_TEACHER_LR = 0.3
    AE_TEACHER_RESTARTS = 3
    AE_LOGIT_POSITIVE_WEIGHT = 1.0
    AE_EVAL_STRATEGY = "direct"  # zero, direct, or refine
    AE_PRIOR_STEPS = 30
    AE_PRIOR_KL = 0.1
    AE_PRIOR_TEMPERATURE = 1.0


class ICAutoencoderZero:
    """Train both auxiliary FTNs, but use four-label cold search at evaluation."""
    AE_EVAL_STRATEGY = "zero"


class ICAutoencoderDirect:
    """Use the predicted logits' hard top-k mask without label refinement."""
    AE_EVAL_STRATEGY = "direct"


class ICAutoencoderRefine:
    """Update the predicted prior using four labeled evaluation examples."""
    AE_EVAL_STRATEGY = "refine"


class ICAutoencoderFreeze:
    """Freeze x->L neurons used by older domain masks during new blocks."""
    AE_LOGIT_FREEZE_OLD = True


class ICAutoencoderColdTeacher:
    """Distill a full-support current-task mask into x->L, once per block."""
    AE_TEACHER_MODE = "cold_mask"
    AE_LOGIT_POSITIVE_WEIGHT = 3.0


class ICAutoencoderSmoke:
    AE_SQUARE_DIM = 4
    AE_K = 4
    AE_RECONFIG_STEPS = 2
    AE_EVAL_RECONFIG_STEPS = 3
    AE_EVAL_RESTARTS = 1
    AE_LOGIT_CACHE_ROWS = 16
    AE_LOGIT_TRAIN_STEPS = 3
    AE_LOGIT_BATCH_SIZE = 8
    AE_PRIOR_STEPS = 2
