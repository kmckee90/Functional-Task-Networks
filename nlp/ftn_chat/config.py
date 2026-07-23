"""Central configuration for the FTN-Chat live demo.

All knobs live here so qa_synth / train / routing_check / app stay consistent.

The training design mirrors the base/ experiment invoked as
    uv run python run_experiments.py --config Large NoReplay Reactive_Low
i.e. composed base/config.py classes:
  NoReplay     -> replay disabled entirely
  Reactive_Low -> loss-triggered reconfiguration during training
                  (RECONFIG_REACTIVE_SD=5.0, RECONFIG_REACTIVE_EMA=0.9)
  Default      -> annealSTE reconfig (30 steps, lr 0.3), plain KWTA
                  (MASK_LATERAL_STEPS=0; SmoothKWTA options kept but OFF),
                  MASK_KERNEL_SIZE=3, TORUS=True.
Bank size is set independently of the Large class (Kevin, 2026-07-19):
H=32**2=1024 atoms, k=128, rank-1 atoms (~0.65B atom params over the banked
layers instead of Large's 96**2 sizing).

At inference the app still reconfigures on EVERY user prompt (reactive
scheduling is a training-time mechanism only).

Post-training is context distillation (Cartridges-style self-study):
teacher = the same frozen backbone WITH the source excerpt in context and the
atom mask zeroed; student = the masked bank WITHOUT the context. The student
is trained with KL on the answer tokens. No raw-text LM training.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import registry

HERE = Path(__file__).resolve().parent


@dataclass
class Config:
    # --- Model ---
    # Qwen3.6-27B: dense-FFN hybrid (Gated DeltaNet + full attention), 64
    # layers, thinking toggle. Dense FFNs keep the full bank design; the
    # last-25%-layers evidence came from its dense sibling Qwen3-32B.
    model_name: str = "Qwen/Qwen3.6-27B"
    bf16: bool = True
    # Banks attach ONLY to the last quarter of the decoder stack: the LoCoMo
    # layer-localization sweep found extractable semantic knowledge lives in
    # the last ~25% of blocks and mid-stack deltas are dead weight or harmful.
    lora_layer_frac: float = 0.25

    # --- LoRA bank (atoms laid out on a grid_dim x grid_dim square) ---
    grid_dim: int = 32  # H = grid_dim**2 atoms (32 -> 1024)
    k: int = 128  # active atoms per forward pass
    rank: int = 1  # rank-1 atoms (scaling=alpha/k is rank-independent)
    alpha: float = 16.0  # authority to override the frozen base prior
    dropout: float = 0.05
    init_b_zero: bool = False  # MUST be False: see CONTINUATION.md §6

    # --- Prompt-perplexity reconfiguration (annealSTE; base Default budget) ---
    reconfig_steps: int = 30  # annealSTE inner steps at INFERENCE  [RECONFIG_STEPS]
    train_reconfig_steps: int = 30  # per-reconfig steps during training
    reconfig_lr: float = 0.3  # Adam lr for the mask logits          [RECONFIG_LR]
    reconfig_init_sd: float = 1e-5  # mask-logit init noise sd (base: randn*1e-5)
    # SmoothKWTA lateral options — AVAILABLE but OFF by default (plain KWTA,
    # matching Default.MASK_LATERAL_STEPS=0 in the base config).
    lateral_steps: int = 0  # 0 = plain KWTA; >0 = SmoothKWTA   [MASK_LATERAL_STEPS]
    lateral_kernel: int = 3  # smoothing kernel size              [MASK_KERNEL_SIZE]
    lateral_polyak: float = 0.0  # pre/post smoothing mix         [MASK_LATERAL_POLYAK]
    lateral_noise_sd: float = 0.0  # per-step mask noise          [MASK_LATERAL_NOISE_SD]
    torus: bool = True  # circular padding for the lateral conv   [TORUS]

    # --- Reactive (loss-triggered) reconfiguration during training ---
    # Reconfigure once on the first training batch, then track an EMA of the
    # loss mean/variance and reconfigure only when the batch loss under the
    # current mask exceeds mean + reactive_sd * sd. Boundary-free: the EMA
    # state persists across task blocks AND across train.py invocations
    # (stored in state.json). [Reactive_Low]
    reactive: bool = True
    reactive_sd: float = 5.0  # trigger threshold in EMA sds  [RECONFIG_REACTIVE_SD]
    reactive_ema: float = 0.9  # EMA smoothing; warmup ~1/(1-ema)=10 batches

    # --- Data ---
    chunk_tokens: int = 512  # token window per source chunk (distillation context)
    token_ceiling: int = 400_000  # per-task cap on ingested tokens
    heldout_per_domain: int = 4  # chunks held out for routing_check
    support_per_domain: int = 8  # chunks used as reconfig support pool

    # --- Training (deeper defaults; override per-run via train.py CLI) ---
    epochs: int = 1
    steps_per_domain: int = 400
    # Examples per training batch. Each step reconfigures (when triggered) on
    # one random member's question and applies that single FTN mask to the
    # whole padded batch.
    batch_size: int = 16

    # --- Context distillation (teacher-with-context -> student-closed-book) ---
    # Synthesis: qa_synth.py generates diverse questions per source chunk with
    # the LOCAL backbone model, then the teacher answers each question WITH the
    # chunk in context. Training: the student (masked bank, no context) matches
    # the teacher's answer-token distributions (KL), recomputed on the fly —
    # the teacher is the same backbone with the mask zeroed, so no second model.
    qa_gen_model: str = "Qwen/Qwen3.6-27B"  # local generator (no egress)
    qa_per_chunk: int = 40  # questions synthesized per chunk (diversity > depth)
    # Thinking OFF for synthesis: our tasks are factual Q&A about code/docs with
    # the source IN the teacher's context — extraction, not multi-step
    # reasoning — so a think trace barely changes the answer while dominating
    # generation cost (~70-80% of tokens) and inflating training sequences.
    # See CONTINUATION.md. Flip to True for a reasoning-heavy corpus (then also
    # raise synth_answer_tokens / qa_max_len for the trace).
    synth_think_answers: bool = False
    synth_answer_tokens: int = 512  # teacher generation budget (answer only)
    # Generation batch size for synthesis. Both stages (question-gen and
    # answer-gen) are pooled across chunks and generated in batches of this
    # many sequences, so the GPUs stay saturated instead of decoding one
    # stream at a time (single-stream was the synthesis bottleneck).
    synth_gen_batch: int = 24
    qa_max_len: int = 1024  # student sequence cap (tokens)
    qa_ctx_max_len: int = 1664  # teacher sequence cap (context + student seq)
    qa_sys: str = "You are a helpful assistant. Answer concisely and accurately."
    kl_temp: float = 1.0  # distillation softmax temperature
    kl_clip: float = 10.0  # per-token KL clamp (OPSD); 0 disables
    ce_weight: float = 0.0  # optional CE-on-teacher-answer mixed into the KL loss

    # --- General-instruction rehearsal (optional; off by default) ---
    rehearsal_dataset: str = "tatsu-lab/alpaca"
    rehearsal_n: int = 400
    rehearsal_frac: float = 0.0  # 0.0 => no general-instruction rehearsal

    lr: float = 1e-4
    seed: int = 0

    # --- Serving ---
    port: int = 7860

    @property
    def H(self) -> int:
        return self.grid_dim**2

    @property
    def reactive_warmup(self) -> int:
        """Batches to suppress triggers after each reconfig (~1/(1-EMA))."""
        import math

        return int(math.ceil(1.0 / max(1.0 - self.reactive_ema, 1e-8)))

    # --- Domains: default block order comes from the task registry ---
    domains: tuple = field(default_factory=lambda: tuple(registry.DEFAULT_ORDER))

    # --- Paths ---
    data_cache: Path = field(default_factory=lambda: HERE / "data_cache")
    # Checkpoints live OUTSIDE the synced workspace (large objects like bank.pt
    # must stay on the pod's home PVC and never enter the DevSpace sync tree).
    # On the pod this resolves to /home/dev/ftn_chat_state/checkpoints.
    # FTN_CKPT_DIR overrides (e.g. so the app serves from a snapshot dir while
    # train.py keeps writing the default; per-process env on the app launch).
    ckpt_dir: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "FTN_CKPT_DIR", str(Path.home() / "ftn_chat_state/checkpoints")
            )
        )
    )

    def __post_init__(self):
        self.data_cache.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)


# Labels come straight from the registry so new tasks show up everywhere.
DOMAIN_LABELS = registry.labels()

CFG = Config()
