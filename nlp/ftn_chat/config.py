"""Central configuration for the FTN-Chat live demo.

The transformer bank has 1024 rank-1 atoms and one shared exact-k mask with
128 active entries. Training and chat use the same BayesLoRA posterior update
on prompt NLL: short KL-regularized steps on stable prompts, longer reset/refit
steps after reactive change detection. The teacher/student training objective
remains context distillation on synthesized Q&A.
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
    # model_name: str = "Qwen/Qwen3.6-27B"
    model_name: str = "Qwen/Qwen3.6-27B"
    # model_name: str = "Qwen/Qwen3-8B"

    bf16: bool = True
    attn_implementation: str = "eager"
    # Qwen3-8B + this bank fits on one H200, which benchmarks faster than
    # layer sharding for the stateful batch-16 trainer. Set True for larger
    # backbones that need all visible GPUs for memory.
    model_parallel: bool = False
    # Banks attach ONLY to the last quarter of the decoder stack: the LoCoMo
    # layer-localization sweep found extractable semantic knowledge lives in
    # the last ~25% of blocks and mid-stack deltas are dead weight or harmful.
    lora_layer_frac: float = 0.25

    # --- LoRA bank (atoms laid out on a grid_dim x grid_dim square) ---
    grid_dim: int = 48  # H = grid_dim**2 atoms (32 -> 1024)
    k: int = 256  # active atoms per forward pass
    rank: int = 1  # rank-1 atoms (scaling=alpha/k is rank-independent)
    alpha: float = 64.0  # authority to override the frozen base prior
    dropout: float = 0.00
    init_b_zero: bool = False  # MUST be False: see CONTINUATION.md §6

    # --- BayesLoRA posterior update on prompt NLL ---
    # Shared mask/logits persist across training batches and chat prompts.
    # Stable inputs get a cheap KL-regularized update; a detected change gets
    # a longer reset/refit. These are the BayesLoRA defaults from bayesian_adv.
    bayes_reconfig_steps: int = 2
    bayes_reconfig_lr: float = 0.3
    bayes_kl_weight: float = 1.0
    bayes_temperature: float = 3.0
    bayes_prior_decay: float = 1.0
    bayes_changepoint_reconfig_steps: int = 30
    bayes_changepoint_reconfig_lr: float = 0.3
    bayes_changepoint_kl_weight: float = 0.0
    bayes_changepoint_temperature: float = 3.0
    bayes_changepoint_prior_decay: float = 0.0
    
    # Legacy cold-start configurer retained for explicit comparisons.
    reconfig_steps: int = 240
    train_reconfig_steps: int = 30
    reconfig_lr: float = 0.03
    reconfig_init_sd: float = 1e-5

    # SmoothKWTA on Bayes candidate/final masks and routing repair candidates.
    # Disabled entirely by lateral_steps=0 or lateral_polyak=1.0.
    lateral_steps: int = 2  # 0 = plain KWTA; >0 = SmoothKWTA   [MASK_LATERAL_STEPS]
    lateral_kernel: int = 9  # smoothing kernel size              [MASK_KERNEL_SIZE]
    lateral_polyak: float = 0.93  # pre/post smoothing mix         [MASK_LATERAL_POLYAK]
    lateral_noise_sd: float = 0.0  # per-step mask noise          [MASK_LATERAL_NOISE_SD]
    torus: bool = False  # circular padding for the lateral conv   [TORUS]

    # --- Reactive changepoint detection on pre-update prompt NLL ---
    # The same detector is used by training and live chat. The training state
    # persists across task blocks and train.py invocations.
    reactive: bool = True
    reactive_sd: float = 0.5  # BayesLoRA default, in EMA standard deviations
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
    # 27B + H=2304: batch 4 passed a max-length source/replay stress test on
    # one H200 (116.1 GB allocated); batch 8 exceeded memory on update two.
    batch_size: int = 8
    # Optional extra source-text next-token CE on this fraction of Q&A steps.
    # It uses the question-selected mask from that step and only train chunks.
    source_ce_prob: float = 0.25
    source_ce_weight: float = 0.1

    # --- Mask-filtered replay (base FTN style) ---
    # A sparse random proposal and minimum gap spread stored examples across
    # the whole task rather than taking its first few adjacent batches.
    replay_store_prob: float = 0.2
    replay_min_gap: int = 1
    replay_per_task: int = 64
    replay_max_entries: int = 128*2
    # Only partially overlapping masks can interfere, so those are replayed.
    # k/H=0.125 here: 0.15 selects above-random overlap more often than the
    # base project's 0.20 threshold at its larger k/H.
    replay_overlap_min: float = 0.05
    replay_overlap_max: float = 0.90
    replay_use_prob: float = 1.0
    replay_weight: float = 1.0

    # --- Optional input-conditioned routing ---
    # Training keeps its existing posterior updates. After a task block, route
    # masks are fit on cached Q&A and accepted on disjoint Q&A validation rows.
    # The app can retrieve these masks from a question without answer labels.
    input_route_archive_questions: int = 128
    input_route_semantic_examples: int = 32
    input_route_semantic_weight: float = 2.0
    input_route_semantic_floor: float = 0.30
    input_route_semantic_gap: float = 0.25
    input_route_anchor_min_docs: int = 2
    input_route_margin: float = 0.5
    input_route_refine_steps: int = 30
    input_route_refine_kl: float = 1.0
    input_route_fit_rows: int = 8
    input_route_validation_rows: int = 8
    input_route_steps: int = 30
    input_route_restarts: int = 2
    input_route_lr: float = 0.3
    input_route_temperature: float = 3.0
    input_route_min_improvement: float = 0.0
    input_route_logit_magnitude: float = 4.0

    # --- Context distillation (teacher-with-context -> student-closed-book) ---
    # Synthesis: qa_synth.py generates diverse questions per source chunk with
    # the LOCAL backbone model, then the teacher answers each question WITH the
    # chunk in context. Training: the student (masked bank, no context) matches
    # the teacher's answer-token distributions (KL), recomputed on the fly —
    # the teacher is the same backbone with the mask zeroed, so no second model.
    qa_gen_model: str = "Qwen/Qwen3.6-27B"  # local generator (no egress)
    qa_per_chunk: int = 40  # questions synthesized per chunk (diversity > depth)
    # Accepted Q&A totals for `qa_synth.py --resume`. Set a task's desired
    # total here; later resume runs continue from its existing cache.
    qa_target_counts: dict[str, int] = field(default_factory=lambda: {
        "base_ftn": 1077,
        "explorer": 2019,
    })
    qa_max_rounds: int = 30  # stop and report if the fixed source is exhausted
    qa_stall_rounds: int = 3  # consecutive rounds with no accepted new pairs
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
    ce_weight: float = 0.1  # optional CE-on-teacher-answer mixed into the KL loss

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
    # Synthesized Q&A can grow substantially. Keep it on the pod's home PVC,
    # outside the bidirectionally synced source workspace.
    data_cache: Path = field(
        default_factory=lambda: Path(
            os.environ.get(
                "FTN_DATA_CACHE_DIR", str(Path.home() / "ftn_chat_state/data_cache")
            )
        )
    )
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
