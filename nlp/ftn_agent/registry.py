"""Dynamic stream-block registry for FTN-Agent continual learning.

Unlike ftn_chat's static task registry (fixed domains registered in code), blocks
here are created at runtime — one per synthesis/train run, keyed by session/date
(design.md §5.3). The manifest (blocks.json) persists in the state dir alongside
the checkpoints, so the app can build the legend and color the mask.

Coloring: a time gradient (older -> newer) so a continuous stream still reads
legibly, since there are no clean topic delineations (design.md §5.3, §8.2).

Dependency-light (no torch / no network at import).
"""
from __future__ import annotations

import colorsys
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Block:
    key: str            # e.g. "sess_2026-07-06_1430"
    label: str          # human-readable
    created_ts: float   # unix time (passed in; we never call time at import)
    n_examples: int = 0 # how many synthesized examples fed this block


def _manifest_path(state_dir: Path) -> Path:
    return Path(state_dir) / "blocks.json"


def load_blocks(state_dir: Path) -> list[Block]:
    p = _manifest_path(state_dir)
    if not p.exists():
        return []
    try:
        raw = json.loads(p.read_text())
    except Exception:
        return []
    return [Block(**d) for d in raw]


def save_blocks(state_dir: Path, blocks: list[Block]) -> None:
    _manifest_path(state_dir).write_text(
        json.dumps([asdict(b) for b in blocks], indent=2)
    )


def register_block(state_dir: Path, key: str, label: str, created_ts: float,
                   n_examples: int = 0) -> list[Block]:
    """Append (or update) a block and persist. Returns the full block list."""
    blocks = load_blocks(state_dir)
    for b in blocks:
        if b.key == key:
            b.label, b.n_examples = label, n_examples
            save_blocks(state_dir, blocks)
            return blocks
    blocks.append(Block(key=key, label=label, created_ts=created_ts, n_examples=n_examples))
    save_blocks(state_dir, blocks)
    return blocks


def colors(blocks: list[Block]) -> dict:
    """Time-gradient RGB per block: oldest = cool (blue), newest = warm (red),
    walking the hue circle. Deterministic in block order (== recency order)."""
    n = max(len(blocks), 1)
    out = {}
    for i, b in enumerate(blocks):
        # hue from 0.6 (blue, old) down to 0.0 (red, new)
        hue = 0.6 * (1.0 - (i / max(n - 1, 1)))
        r, g, bl = colorsys.hsv_to_rgb(hue, 0.75, 0.95)
        out[b.key] = (r, g, bl)
    return out


def labels(blocks: list[Block]) -> dict:
    return {b.key: b.label for b in blocks}
