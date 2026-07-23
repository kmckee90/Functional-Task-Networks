"""Task registry for FTN-Chat continual learning.

A *task* is a named block of text learned as one CL stage. To add a new dataset,
call `register(key, label, globs, color=...)` — that's the whole extension point.
New tasks are appended to DEFAULT_ORDER and will be picked up by `train.py` as
the next block on top of whatever is already trained.

All task sources live under `corpus/` INSIDE this project so the pod (which
syncs only nlp/ftn_chat) always has them. `make_corpus.py` snapshots them from
the surrounding functional_task_networks repo — run it locally after upstream
files change, then retrain the affected block.

Dependency-light (no torch / no network at import) so config, data, app, and
train can all import it without cycles.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus"

# A small, well-separated palette for tasks registered without an explicit color.
PALETTE = [
    (0.90, 0.25, 0.25), (0.25, 0.80, 0.35), (0.30, 0.45, 0.95),
    (0.95, 0.70, 0.20), (0.70, 0.35, 0.85), (0.20, 0.80, 0.80),
    (0.95, 0.45, 0.70), (0.60, 0.60, 0.60),
]


@dataclass
class Task:
    key: str
    label: str
    color: tuple            # RGB in 0..1 (for the mask viz)
    globs: list             # absolute glob strings; recursive ** supported


TASKS: dict[str, Task] = {}
DEFAULT_ORDER: list[str] = []


def register(key, label, globs, color=None, default=True):
    """Register a task. `globs` is a list of absolute glob patterns. Returns the
    Task. If color is None, one is auto-assigned from PALETTE."""
    if color is None:
        color = PALETTE[len(TASKS) % len(PALETTE)]
    TASKS[key] = Task(key, label, tuple(color), list(globs))
    if default and key not in DEFAULT_ORDER:
        DEFAULT_ORDER.append(key)
    return TASKS[key]


def labels() -> dict:
    return {k: t.label for k, t in TASKS.items()}


def colors() -> dict:
    return {k: t.color for k, t in TASKS.items()}


# --- Currently registered tasks (block order = registration order) -----------
# Sources are snapshotted into corpus/ by make_corpus.py (see that file for the
# upstream repo paths each snapshot comes from).

register("base_ftn", "Base FTN (ftn.py + config.py)",
         [str(CORPUS / "base_ftn/ftn.py"),
          str(CORPUS / "base_ftn/config.py")],
         color=(0.90, 0.25, 0.25))

register("explorer", "Explorer agent (RDDPG_Dual + RECODE)",
         [str(CORPUS / "explorer/agent_recode.py"),
          str(CORPUS / "explorer/recode.py"),
          str(CORPUS / "explorer/reservoir_local_connectivity.py"),
          str(CORPUS / "explorer/density_estimator.py"),
          str(CORPUS / "explorer/observation_buffer.py")],
         color=(0.25, 0.80, 0.35))

register("rl_mt10", "FTN robotics MT10 (BC + DDPG)",
         [str(CORPUS / "rl_mt10/train_mt10_bc.py"),
          str(CORPUS / "rl_mt10/train_mt10_ddpg.py"),
          str(CORPUS / "rl_mt10/ftn.py")],
         color=(0.30, 0.45, 0.95))

# --- To add a dataset, copy the pattern above, e.g.:
# register("mydata", "My Dataset", ["/abs/path/**/*.py", "/abs/path/*.md"])
# then:  uv run python qa_synth.py mydata && uv run python train.py --tasks mydata
