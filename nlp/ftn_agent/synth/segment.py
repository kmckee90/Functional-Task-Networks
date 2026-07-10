"""Segment the raw activity stream into EPISODES (design.md §5.1).

An episode is a coherent work span — a debugging arc, a feature, an experiment.
Cheap heuristics: split on session boundaries and large time gaps; flag episodes
that contain a failure (any error record) as high-value for insight synthesis.
Also renders each episode to compact text for the synthesizer prompt.

Stdlib only (operates on capture.records.Record objects).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from capture.records import Record

GAP_SECONDS = 45 * 60          # new episode after this idle gap within a session
MAX_RECORDS = 80              # cap an episode's size
MAX_RENDER_CHARS = 6000        # cap the text handed to the synthesizer


@dataclass
class Episode:
    key: str
    session_id: str
    start_ts: float
    end_ts: float
    records: list = field(default_factory=list)
    has_failure: bool = False
    cwd: str = ""


def _is_error(r: Record) -> bool:
    return bool(r.payload.get("is_error"))


def segment(records: list[Record]) -> list[Episode]:
    recs = sorted([r for r in records if r.ts], key=lambda r: (r.session_id, r.ts))
    episodes: list[Episode] = []
    cur: Episode | None = None
    for r in recs:
        new = (
            cur is None
            or r.session_id != cur.session_id
            or (r.ts - cur.end_ts) > GAP_SECONDS
            or len(cur.records) >= MAX_RECORDS
        )
        if new:
            cur = Episode(
                key=f"{r.session_id[:8]}_{int(r.ts)}",
                session_id=r.session_id, start_ts=r.ts, end_ts=r.ts, cwd=r.cwd,
            )
            episodes.append(cur)
        cur.records.append(r)
        cur.end_ts = r.ts
        if _is_error(r):
            cur.has_failure = True
    # Order episodes by time.
    episodes.sort(key=lambda e: e.start_ts)
    return episodes


def _one_line(s: str, n: int = 500) -> str:
    s = " ".join(str(s).split())
    return s[:n]


def render(ep: Episode) -> str:
    """Compact human-readable narrative of an episode for the synth prompt."""
    lines: list[str] = []
    for r in ep.records:
        p = r.payload
        if r.kind == "user_msg":
            lines.append(f"USER: {_one_line(p.get('text',''), 600)}")
        elif r.kind == "assistant_msg":
            lines.append(f"CLAUDE: {_one_line(p.get('text',''), 600)}")
        elif r.kind == "tool_exec":
            inp = p.get("input", {})
            arg = inp.get("command") or inp.get("file_path") or inp.get("skill") \
                or inp.get("pattern") or _one_line(str(inp), 120)
            tag = " -> ERROR" if p.get("is_error") else ""
            out = _one_line(p.get("output_tail", ""), 300)
            lines.append(f"[{p.get('tool','tool')}] {_one_line(arg,200)}{tag}"
                         + (f"  out: {out}" if out else ""))
        elif r.kind == "command":
            tag = f" (exit {p.get('exit_code')})" if p.get("exit_code") not in (0, None) else ""
            out = _one_line(p.get("output_tail", ""), 300)
            lines.append(f"$ {_one_line(p.get('command',''),200)}{tag}"
                         + (f"  {out}" if out else ""))
        elif r.kind == "diff":
            lines.append(f"COMMIT {p.get('sha','')} {_one_line(p.get('subject',''),160)}\n"
                         f"{_one_line(p.get('diffstat',''), 400)}")
    text = "\n".join(lines)
    return text[:MAX_RENDER_CHARS]
