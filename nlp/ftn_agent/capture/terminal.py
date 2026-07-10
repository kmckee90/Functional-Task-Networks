"""S2: terminal command log -> activity records (design.md §3).

A shell hook appends one JSON object per command to cfg.terminal_log:
    {"ts": <unix>, "command": str, "cwd": str, "exit_code": int, "output": str?}
See README for the bash snippet. Output capture is optional (hard from a
PROMPT_COMMAND hook); command + exit_code + cwd alone are already high-value,
especially nonzero exits. Stdlib only; runs wherever the shell runs.
"""
from __future__ import annotations

import json
from pathlib import Path

from .records import Record


def collect(cfg) -> list[Record]:
    log = Path(cfg.terminal_log)
    if not log.exists():
        return []
    out: list[Record] = []
    for i, line in enumerate(log.read_text(errors="ignore").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        cmd = str(o.get("command", "")).strip()
        if not cmd:
            continue
        exit_code = o.get("exit_code")
        out_tail = str(o.get("output", ""))[-cfg.stdout_tail_chars:]
        out.append(Record(
            "command", "terminal", float(o.get("ts", 0.0)),
            session_id=str(o.get("session_id", "shell")),
            cwd=str(o.get("cwd", "")),
            payload={"_seed": f"{o.get('ts','')}:{cmd}",
                     "command": cmd,
                     "exit_code": exit_code,
                     "is_error": bool(exit_code not in (0, None)),
                     "output_tail": out_tail},
        ))
    return out
