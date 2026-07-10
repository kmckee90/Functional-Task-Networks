"""S1: Claude Code session transcripts -> normalized activity records.

The richest source (design.md §3): ~/.claude/projects/<slug>/*.jsonl contain
every user turn, assistant reasoning, tool call, and tool result (with is_error).
We join each tool_use to its tool_result by tool_use_id so a single record
carries "command -> outcome (ok/error)" — exactly the failure->resolution signal
episode/insight synthesis needs.

Runs on the LAPTOP (where the transcripts live); stdlib only.
"""
from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from pathlib import Path

from .records import Record

# Keep these tool inputs compact but informative.
_INPUT_KEYS = ("command", "file_path", "path", "pattern", "query", "description",
               "skill", "prompt", "url")


def _ts(o) -> float:
    t = o.get("timestamp")
    if isinstance(t, (int, float)):
        return float(t)
    if isinstance(t, str):
        try:
            return datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0
    return 0.0


def _text_of(content) -> str:
    """Flatten a message.content (str or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for b in content:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            parts.append(str(b.get("text", "")))
    return "\n".join(p for p in parts if p)


def _result_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                out.append(str(b.get("text", "")))
            elif isinstance(b, str):
                out.append(b)
        return "\n".join(out)
    return ""


def _slim_input(inp: dict) -> dict:
    if not isinstance(inp, dict):
        return {}
    return {k: inp[k] for k in _INPUT_KEYS if k in inp}


def _collect_file(path: str, cfg) -> list[Record]:
    tail = cfg.stdout_tail_chars
    # First pass: index tool_result blocks by tool_use_id.
    results: dict[str, dict] = {}
    lines = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            try:
                o = json.loads(line)
            except Exception:
                continue
            lines.append(o)
            if o.get("type") == "user":
                for b in (o.get("message", {}).get("content") or []):
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        results[b.get("tool_use_id", "")] = b

    out: list[Record] = []
    for o in lines:
        t = o.get("type")
        if t not in ("user", "assistant"):
            continue
        ts, cwd, sid = _ts(o), o.get("cwd", ""), o.get("sessionId", "")
        msg = o.get("message", {})
        content = msg.get("content")
        if t == "user":
            # Skip pure tool_result-carrier user turns (captured via tool_exec).
            txt = _text_of(content)
            if txt.strip():
                out.append(Record("user_msg", "claude", ts, sid, cwd,
                                   {"_seed": o.get("uuid", ""), "text": txt[:tail]}))
        else:  # assistant
            txt = _text_of(content)
            if txt.strip():
                out.append(Record("assistant_msg", "claude", ts, sid, cwd,
                                   {"_seed": o.get("uuid", ""), "text": txt[:tail]}))
            for b in (content or []):
                if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                    continue
                tid = b.get("id", "")
                res = results.get(tid, {})
                res_txt = _result_text(res.get("content"))
                is_err = bool(res.get("is_error"))
                # Heuristic error sniff even when is_error is unset.
                low = res_txt.lower()
                if not is_err and any(s in low for s in
                                      ("traceback", "error:", "exit code 1", "command not found",
                                       "no such file", "exception", "failed")):
                    is_err = True
                out.append(Record(
                    "tool_exec", "claude", ts, sid, cwd,
                    {"_seed": tid,
                     "tool": b.get("name", ""),
                     "input": _slim_input(b.get("input", {})),
                     "is_error": is_err,
                     "output_tail": res_txt[-tail:] if res_txt else ""},
                ))
    return out


def collect(cfg) -> list[Record]:
    """Layout: ~/.claude/projects/<project-slug>/<session>.jsonl. With a slug set
    we read just that project; otherwise every project under the base dir."""
    base = Path(cfg.claude_projects_dir)
    if not base.exists():
        return []
    if cfg.claude_project_slug:
        files = glob.glob(str(base / cfg.claude_project_slug / "*.jsonl"))
    else:
        files = glob.glob(str(base / "*" / "*.jsonl"))
    out: list[Record] = []
    for f in sorted(set(files), key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0):
        try:
            out += _collect_file(f, cfg)
        except Exception:
            continue
    return out
