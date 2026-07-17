"""Configuration for the temporal (recurrent FTN) experiment."""

from dataclasses import dataclass, asdict


@dataclass
class Config:
    # ---- Task family: affine group Aff(Z_p), words over two generators ----
    p: int = 11                    # prime; 2 is a primitive root -> {T1, M2} generate all 110 maps
    L: int = 4                     # tokens per sequence (ops act elementwise)
    generators: tuple = (("T1", (1, 1)), ("M2", (2, 0)))   # (name, (c, a)): x -> c*x + a
    train_max_len: int = 3         # train on all maps with shortest word <= this (plus identity)
    eval_max_len: int = 9          # hold out everything with shortest word in (train_max_len, this]
    T_max: int = 10                # trace-length ceiling at evaluation
    input_train_frac: float = 0.5  # of the p^L input space

    # ---- Recurrent FTN backbone (one MaskedStage, weight-tied across steps) ----
    H: int = 2048
    k: int = 256
    neuron_depth: int = 4
    neuron_width: int = 8
    dropout: float = 0.2

    # ---- Configurers ----
    s_train: int = 10              # inner steps, per-batch atom reconfig (V1 atoms)
    s_eval: int = 40               # inner steps, eval-time configuration
    configurer_lr: float = 0.5
    support_size: int = 128
    search_support: int = 32       # smaller support used inside BFS trace search
    lib_restarts: int = 5          # consensus restarts when consolidating atom masks
    v2_T_train: int = 4            # V2: trace-length sweep ceiling during training
    v2_s_train: int = 8            # V2: inner steps per candidate length during training

    # ---- Block-sequential training ----
    steps_per_task: int = 300
    cycles: int = 3
    trace_refresh: int = 25        # V1: re-run BFS trace search every this many batches
    batch_size: int = 256
    lr: float = 1e-3
    replay_capacity: int = 200
    replay_batch: int = 64
    replay_samples: int = 8
    rho_lo: float = 0.2
    rho_hi: float = 0.8

    # ---- Joint task-conditioned MLP baseline ----
    joint_hidden: int = 512
    joint_layers: int = 3
    joint_steps: int = 6000

    seed: int = 0
    device: str = "cuda"

    def to_dict(self):
        return asdict(self)


def small() -> Config:
    return Config(H=512, k=128, steps_per_task=120, cycles=2, s_eval=25,
                  lib_restarts=3, joint_steps=1000, eval_max_len=6)


def full() -> Config:
    return Config()
