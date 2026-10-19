"""Central configuration for FTN-Agent.

All knobs live here so capture / synth / train / app stay consistent. See
design.md for the rationale behind each default.

Storage roots (design.md §7):
  - state_dir  : PERSISTENT, on the pod home PVC, OUTSIDE the synced workspace.
                 Holds bank/refs/replay/state checkpoints + the processed
                 activity store. Never synced to the laptop.
  - work_dir   : on the PVC too, but PRUNED after each block trains (large
                 synthesized/tokenized datasets live here only transiently).
  - inbox_dir  : SMALL synced bridge under the workspace. Laptop-side capture
                 drops normalized activity records here (small text is fine to
                 sync); the pod daemon drains it into state_dir/activity.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _pvc_root(name: str) -> Path:
    """Prefer a PVC path OUTSIDE the synced workspace when on the pod
    (/home/dev exists + writable); fall back to a local dir off-pod."""
    pod_home = Path("/home/dev")
    if pod_home.is_dir() and os.access(pod_home, os.W_OK):
        return pod_home / name
    return HERE / name


@dataclass
class Config:
    # --- Model (design.md §4) ---
    model_name: str = "Qwen/Qwen3-32B"      # dense hybrid-thinking reasoning model
    bf16: bool = True

    # --- LoRA bank (atoms on a grid_dim x grid_dim square; design.md §4.4) ---
    grid_dim: int = 32                       # H = grid_dim**2 (24 -> 576)
    k: int = 32                             # active atoms per forward pass
    rank: int = 1
    alpha: float = 16.0
    dropout: float = 0.05
    init_b_zero: bool = False                # MUST be False (design.md, configurer needs to SELECT)

    # --- Prompt-perplexity reconfiguration (the core mechanism) ---
    reconfig_steps: int = 1                  # AnnealSTE inner steps at INFERENCE
    train_reconfig_steps: int = 1            # per-batch reconfig during training
    reconfig_lr: float = 1.0
    lateral_steps: int = 2                   # lateral smoothing iterations (blobs)
    lateral_kernel: int = 9
    lateral_polyak: float = 0.0              # 0..1; lower = stronger smoothing pull
    torus: bool = False

    # --- Data + chunking ---
    chunk_tokens: int = 512                  # token window per raw training chunk
    token_ceiling: int = 400_000             # per-block cap on raw training tokens

    # --- Synthesis (reuses the SERVING model; no second load — design.md §5.2) ---
    qa_per_episode: int = 16                  # Q&A pairs synthesized per episode
    insights_per_episode: int = 8            # postmortem/insight records per episode
    qa_max_len: int = 1024
    qa_frac: float = 0.7                     # prob a training example is Q&A/insight vs raw
    synth_max_new_tokens: int = 900          # generation budget for a synthesis call
    synth_use_thinking: bool = False         # let the synthesizer emit <think> (slower)
    qa_sys: str = "You are a helpful assistant. Answer concisely and accurately."

    # --- General-instruction rehearsal (anti-forgetting; preserves fluency) ---
    rehearsal_dataset: str = "tatsu-lab/alpaca"
    rehearsal_n: int = 200
    rehearsal_frac: float = 0.0
    lr: float = 1e-4
    replay_frac: float = 0.0                # prob of a replay step per train step
    use_8bit_adam: bool = False              # scale-up lever (design.md §4.4)
    seed: int = 0

    # --- Training schedule (per block; override via train.py CLI) ---
    epochs: int = 1
    steps_per_block: int = 300
    batch_size: int = 16

    # --- Continual-learning retention bounds (design.md §7.3) ---
    replay_per_block: int = 24               # cap examples kept per block for replay
    max_blocks_retained: int = 64            # cap total blocks kept in replay
    keep_checkpoints: int = 2                # rolling bank checkpoints (latest + prev)

    # --- Capture sources (LAPTOP side — where Claude/shell/git live) ---
    claude_projects_dir: Path = field(
        default_factory=lambda: Path.home() / ".claude" / "projects"
    )
    # Which project transcript folder to tail (laptop path slug); empty => all.
    claude_project_slug: str = "-home-klmckee-Desktop-test-actl-functional-task-networks"
    terminal_log: Path = field(
        default_factory=lambda: Path.home() / ".ftn_agent" / "shell.jsonl"
    )
    repo_dir: Path = field(default_factory=lambda: HERE)  # git repo to watch (S3)
    stdout_tail_chars: int = 4000            # how much command/tool output to keep

    # --- Serving ---
    port: int = 7860
    api_port: int = 7861                     # (reserved) FastAPI query endpoint if enabled

    # --- Storage roots (design.md §7) ---
    state_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("FTN_AGENT_STATE", str(_pvc_root("ftn_agent_state"))))
    )
    work_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("FTN_AGENT_WORK", str(_pvc_root("ftn_agent_work"))))
    )
    inbox_dir: Path = field(default_factory=lambda: HERE / "_inbox")  # synced bridge

    @property
    def H(self) -> int:
        return self.grid_dim**2

    @property
    def activity_dir(self) -> Path:
        return self.state_dir / "activity"

    @property
    def qa_gen_model(self) -> str:
        # Synthesis reuses the serving backbone (design.md §5.2) — same weights.
        return self.model_name

    def __post_init__(self):
        for d in (self.state_dir, self.work_dir, self.inbox_dir, self.activity_dir):
            Path(d).mkdir(parents=True, exist_ok=True)


CFG = Config()
