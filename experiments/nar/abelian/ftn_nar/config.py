"""Experiment configuration for the FTN abelian-grid experiment."""

from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ---- Task family (abelian group Z_p x Z_L acting on one-hot sequences) ----
    p: int = 10               # token alphabet size (value shifts live in Z_p)
    L: int = 4                # sequence length (rotations live in Z_L)
    input_train_frac: float = 0.5  # fraction of the p^L input space trained on

    # ---- Grid split ----
    n_train_cells: int = 20   # cells (a, r) trained sequentially; rest held out
    min_per_a: int = 1        # coverage constraints for identifiability
    min_per_r: int = 3

    # ---- FTN^2 backbone ----
    H1: int = 2048            # bank-1 neurons
    H2: int = 2048            # bank-2 neurons
    k1: int = 256             # active budget per bank (single-cell sweep: large k
    k2: int = 256             # is the decisive lever; 0.62 -> 1.0 pos acc)
    neuron_depth: int = 4     # hidden layers inside each neuron-MLP
    neuron_width: int = 8
    d_mid: int = 64           # interface code between the two stages
    dropout: float = 0.2

    # ---- Configurer (AnnealSTE) ----
    s_train: int = 15         # inner steps for per-batch reconfiguration
    s_eval: int = 60          # inner steps for stored/reconfig-protocol masks
    configurer_lr: float = 0.5
    support_size: int = 128

    # ---- Block-sequential training ----
    steps_per_task: int = 300
    cycles: int = 3           # passes through the block-sequential stream
                              # (each visit = steps_per_task steps; >1 is a
                              # recurring stream, still never joint training)
    curriculum: bool = True   # axis cells (primitives) first within each cycle
    batch_size: int = 256
    lr: float = 1e-3          # 3e-3 learned faster but drift wrecked retention
    replay_capacity: int = 200
    replay_batch: int = 64    # examples stored per replay slot
    replay_samples: int = 8   # eligible slots replayed per step
    final_restarts: int = 5   # post-stream configurer restarts per trained cell
                              # (consensus scores = synthesis basis; removes the
                              # staleness of block-end masks under weight drift)
    rho_lo: float = 0.2       # mask-overlap replay window
    rho_hi: float = 0.8
    mask_ema: float = 0.99    # EMA of per-batch masks per cell (for factor analysis)

    # ---- Joint task-conditioned baseline ----
    joint_hidden: int = 512
    joint_layers: int = 3
    joint_steps: int = 6000

    seed: int = 0
    device: str = "cuda"

    def to_dict(self):
        return asdict(self)


def small() -> Config:
    """Fast smoke-test configuration."""
    return Config(
        H1=512, H2=512, k1=128, k2=128,
        s_train=10, s_eval=30, steps_per_task=400,
        n_train_cells=6, min_per_a=0, min_per_r=1,
        joint_steps=1000,
    )


def full() -> Config:
    return Config()
