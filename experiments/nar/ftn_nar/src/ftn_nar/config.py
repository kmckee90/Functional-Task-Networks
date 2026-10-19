"""Configuration dataclasses (defaults = design doc section 24) + YAML overrides."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

METHODS = [
    "FTN_FULL",
    "ORACLE_COMPOSITION",
    "FTN_RANDOM_MASK_TRAIN",
    "FTN_NO_REPLAY",
    "DENSE_ACTIVE_MATCHED",
    "SOFT_MIXTURE",
    "DISJOINT_TASK_MODULES",
    "JOINT_MULTITASK_FTN",
    "SINGLE_TASK",  # stage-B specialist: one algorithm, oracle-fixed mask
    "FTN_PER_TASK_MASK",  # ablation 12.6 / tiny-bank scaffolding comparison
]


@dataclass
class ModelCfg:
    latent_dim: int = 128
    edge_embed_dim: int = 16
    extender_atoms: int = 64
    reducer_atoms: int = 64
    active_extender_atoms: int = 8
    active_reducer_atoms: int = 8
    atom_width: int = 32
    encoder_hidden: int = 128
    edge_temperature: float = 0.1
    hard_attention_warmup_steps: int = 250
    dense_hidden: int = 256  # DENSE_ACTIVE_MATCHED hidden width (param-matched to active path)
    round_register: bool = True  # STE integer rounding of the scalar register each step
    op_latent: bool = True  # feed latent h_u,h_v to operator scalar paths (False = lean)
    extender_residual: bool = False  # candidate = r_u + delta ablation (didn't fix depth OOD)


@dataclass
class ConfigurerCfg:
    inner_steps: int = 12
    learning_rate: float = 1.0
    tau_start: float = 2.0
    tau_end: float = 0.2
    gradient_clip: float = 5.0
    train_restarts: int = 1
    eval_restarts: int = 4
    logit_init_std: float = 0.01
    soft_entropy_weight: float = 0.1  # SOFT_MIXTURE entropy penalty coefficient


@dataclass
class ReplayCfg:
    capacity: int = 2048
    insertion_period: int = 10
    insertion_count: int = 4
    overlap_low: float = 0.20
    overlap_high: float = 0.80
    replay_weight: float = 1.0
    max_batch: int = 16
    uniform: bool = False  # ORACLE/DISJOINT/DENSE: uniform sampling (no mask-overlap filter)


@dataclass
class OptimCfg:
    learning_rate: float = 3e-4
    minimum_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8
    gradient_clip: float = 1.0


@dataclass
class DataCfg:
    support_size: int = 16
    query_batch_size: int = 32
    support_candidate_pool: int = 256
    id_val_count: int = 512
    id_test_count: int = 1024
    ood_count: int = 512
    eval_node_chunk: int = 20000  # max nodes per evaluation forward chunk


@dataclass
class LossCfg:
    final_value: float = 1.0
    trace_value: float = 0.5
    reached: float = 0.1
    final_pointer: float = 0.5
    trace_pointer: float = 0.25
    support_value: float = 1.0
    support_pointer: float = 0.25
    support_logit_l2: float = 0.01


@dataclass
class ExperimentCfg:
    method: str = "FTN_FULL"
    held_out: int = 0  # algorithm id; SINGLE_TASK trains this algorithm instead
    seed: int = 0
    steps_per_task: int = 2000
    trace_supervision: bool = True
    n_eval_support_sets: int = 10
    checkpoint_interval: int = 500
    log_interval: int = 10
    boundary_eval_count: int = 512  # graphs for R-matrix rows
    eval_id_test: bool = True
    eval_ood: bool = True
    device: str = "cuda"
    out_dir: str = "outputs/runs"

    model: ModelCfg = field(default_factory=ModelCfg)
    configurer: ConfigurerCfg = field(default_factory=ConfigurerCfg)
    replay: ReplayCfg = field(default_factory=ReplayCfg)
    optimizer: OptimCfg = field(default_factory=OptimCfg)
    data: DataCfg = field(default_factory=DataCfg)
    loss: LossCfg = field(default_factory=LossCfg)


def tiny_model() -> ModelCfg:
    """Tiny-bank exhaustive diagnostic model (design 13.2)."""
    return ModelCfg(latent_dim=64, extender_atoms=12, reducer_atoms=12,
                    active_extender_atoms=2, active_reducer_atoms=2, atom_width=16,
                    encoder_hidden=64)


def _update(dc, d: dict):
    for k, v in d.items():
        if not hasattr(dc, k):
            raise KeyError(f"unknown config key {k!r} for {type(dc).__name__}")
        cur = getattr(dc, k)
        if dataclasses.is_dataclass(cur):
            _update(cur, v)
        else:
            setattr(dc, k, type(cur)(v) if cur is not None and not isinstance(cur, tuple) else v)
    return dc


def load_config(yaml_path: str | None = None, overrides: dict | None = None) -> ExperimentCfg:
    cfg = ExperimentCfg()
    if yaml_path:
        _update(cfg, yaml.safe_load(Path(yaml_path).read_text()) or {})
    if overrides:
        _update(cfg, overrides)
    return cfg


def to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)


def config_hash(cfg) -> str:
    js = json.dumps(to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha1(js.encode()).hexdigest()[:10]


def save_config(cfg, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(to_dict(cfg), sort_keys=False))
