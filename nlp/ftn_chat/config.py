"""Central configuration for the FTN-Chat live demo.

All knobs live here so train / routing_check / app stay consistent. See
CONTINUATION.md for the rationale behind each default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import registry

HERE = Path(__file__).resolve().parent
REPO_KT = HERE.parent  # experiments/func_net_kt
EXPERIMENTS = REPO_KT.parent  # experiments/


@dataclass
class Config:
    # --- Model ---
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    bf16: bool = True

    # --- LoRA bank (atoms laid out on a grid_dim x grid_dim square) ---
    grid_dim: int = 48  # H = grid_dim**2 atoms (48 -> 2304)
    k: int = 256  # active atoms per forward pass
    rank: int = 4  # 2->4: more per-atom capacity for verbatim recall (scaling=alpha/k is rank-independent)
    alpha: float = (
        16.0  # 8->16: more authority to override the frozen base prior (recall)
    )
    dropout: float = 0.05
    init_b_zero: bool = False  # MUST be False: see CONTINUATION.md §6

    # --- Prompt-perplexity reconfiguration (the core mechanism) ---
    # Stick to the paper's documented AnnealSTE budget: 30 steps, train + infer.
    reconfig_steps: int = 1  # AnnealSTE inner steps at INFERENCE
    train_reconfig_steps: int = 1  # per-batch reconfig during training
    reconfig_lr: float = 1.0
    lateral_steps: int = 2  # lateral smoothing iterations (blobs)
    lateral_kernel: int = 9
    lateral_polyak: float = 0.0  # 0..1; lower = stronger smoothing pull
    torus: bool = False  # circular padding for the lateral conv

    # --- Data + balancing ---
    chunk_tokens: int = (
        256  # token window per training chunk (small single-file domains)
    )
    token_ceiling: int = 400_000  # per-domain cap on training tokens
    heldout_per_domain: int = 4  # chunks held out for routing_check
    support_per_domain: int = 8  # chunks used as reconfig support pool

    # --- Training (deeper defaults; override per-run via train.py CLI) ---
    epochs: int = 1
    steps_per_domain: int = 200
    # Examples per training batch. Each step reconfigures on one random member's
    # question and applies that single FTN mask to the whole padded batch.
    batch_size: int = 16

    # --- Q&A synthesis + answer-masked SFT (mixed with raw text) ---
    qa_gen_model: str = "Qwen/Qwen2.5-7B-Instruct"  # local generator (no egress)
    qa_per_chunk: int = 16  # Q&A pairs synthesized per chunk
    qa_max_len: int = 1024
    qa_frac: float = 0.0  # 0.0 => pure text-prediction (no Q&A)
    qa_sys: str = "You are a helpful assistant. Answer concisely and accurately."

    # --- General-instruction rehearsal (anti-forgetting; preserves fluency) ---
    rehearsal_dataset: str = "tatsu-lab/alpaca"
    rehearsal_n: int = 400  # general examples mixed in
    rehearsal_frac: float = 0.0  # 0.0 => no general-instruction rehearsal
    batch_tokens: int = 512
    lr: float = 1e-4
    replay_frac: float = 0.25  # prob of a replay step per train step
    seed: int = 0

    # --- Serving ---
    port: int = 7860

    @property
    def H(self) -> int:
        return self.grid_dim**2

    # --- Domains: default block order comes from the task registry ---
    domains: tuple = field(default_factory=lambda: tuple(registry.DEFAULT_ORDER))

    # --- Paths ---
    data_cache: Path = field(default_factory=lambda: HERE / "data_cache")
    # FTN_CKPT_DIR lets the app serve from a snapshot dir while train.py keeps
    # writing checkpoints/ (per-process env, set on the app launch only).
    ckpt_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get("FTN_CKPT_DIR", str(HERE / "checkpoints"))
        )
    )

    def __post_init__(self):
        self.data_cache.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)


# Labels come straight from the registry so new tasks show up everywhere.
DOMAIN_LABELS = registry.labels()

CFG = Config()
