"""Task registry for FTN-Chat continual learning.

A *task* is a named block of text learned as one CL stage. To add a new dataset,
call `register(key, label, color, globs)` — that's the whole extension point.
Set `needs_sandbox=True` if the files live inside the astera-org/sandbox clone
(data.ensure_sandbox() runs first). New tasks are appended to DEFAULT_ORDER and
will be picked up by `train.py` as the next block on top of whatever is already
trained.

Dependency-light (no torch / no network at import) so config, data, app, and
train can all import it without cycles.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_KT = HERE.parent                       # experiments/func_net_kt
EXPERIMENTS = REPO_KT.parent                # experiments/
SANDBOX_CACHE = HERE / "data_cache" / "sandbox"

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
    needs_sandbox: bool = False


TASKS: dict[str, Task] = {}
DEFAULT_ORDER: list[str] = []


def register(key, label, globs, color=None, needs_sandbox=False, default=True):
    """Register a task. `globs` is a list of absolute glob patterns. Returns the
    Task. If color is None, one is auto-assigned from PALETTE."""
    if color is None:
        color = PALETTE[len(TASKS) % len(PALETTE)]
    TASKS[key] = Task(key, label, tuple(color), list(globs), needs_sandbox)
    if default and key not in DEFAULT_ORDER:
        DEFAULT_ORDER.append(key)
    return TASKS[key]


def labels() -> dict:
    return {k: t.label for k, t in TASKS.items()}


def colors() -> dict:
    return {k: t.color for k, t in TASKS.items()}


# --- Currently registered tasks (block order = registration order) -----------
# Pure text-prediction over one source file per domain (no Q&A synthesis).
register("sudoku", "Sudoku (JAX gradient search)",
         [str(SANDBOX_CACHE / "team/miguel/gso_vs_mp/sudoku/sudoku_gs.py"),
          str(HERE / "domain_docs/sudoku_gs_doc.md")],   # code + NL annotation
         color=(0.25, 0.80, 0.35), needs_sandbox=True)

register("genetic_programming", "Genetic programming (JAX)",
         [str(EXPERIMENTS / "genetic_programming/jax_gp.py"),
          str(EXPERIMENTS / "genetic_programming/jax_gp_doc.md")],   # code + NL annotation
         color=(0.95, 0.70, 0.20))

register("robotics", "FTN robotics / RL (DDPG)",
         [str(EXPERIMENTS / "robotics/train.py"),
          str(EXPERIMENTS / "robotics/train_doc.md")],               # code + NL annotation
         color=(0.30, 0.45, 0.95))

register("cscg", "CSCG model (code)",
         [str(SANDBOX_CACHE / "packages/algorithms/src/algorithms/cscg/cscg_he.py")],
         color=(0.90, 0.25, 0.25), needs_sandbox=True)

# --- To add a dataset, copy the pattern above, e.g.:
# register("mydata", "My Dataset", ["/abs/path/**/*.py", "/abs/path/*.md"])
# then:  uv run python train.py --tasks mydata     (trains it as the next block)
