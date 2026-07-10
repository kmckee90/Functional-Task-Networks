"""S3: code evolution via git -> activity records (design.md §3).

Reads commits newer than a stored marker in cfg.repo_dir and emits one record
per commit (message + diffstat + a diff tail). The "what changed"; S1/S2 supply
the "why / whether it worked". Stdlib + git only; runs where the repo lives.
"""
from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path

from .records import Record

_MARKER = "git_seen.txt"          # stored under inbox_dir, laptop-side


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, check=True).stdout
    except Exception:
        return ""


def collect(cfg) -> list[Record]:
    repo = Path(cfg.repo_dir)
    if not (repo / ".git").exists() and not _git(repo, "rev-parse", "--git-dir").strip():
        return []
    marker = Path(cfg.inbox_dir) / _MARKER
    last = marker.read_text().strip() if marker.exists() else ""
    rng = f"{last}..HEAD" if last else "-n20"       # first run: last 20 commits
    fmt = "%H%x1f%ct%x1f%an%x1f%s"
    log = _git(repo, "log", rng, f"--pretty=format:{fmt}")
    out: list[Record] = []
    newest = last
    for line in log.splitlines():
        parts = line.split("\x1f")
        if len(parts) != 4:
            continue
        sha, ct, author, subject = parts
        newest = newest or sha
        if sha == last:
            continue
        stat = _git(repo, "show", "--stat", "--oneline", "-s", sha) or \
            _git(repo, "diff", "--stat", f"{sha}~1", sha)
        diff = _git(repo, "diff", f"{sha}~1", sha)
        out.append(Record(
            "diff", "git", float(ct or 0.0),
            session_id="git",
            cwd=str(repo),
            payload={"_seed": sha, "sha": sha[:12], "author": author,
                     "subject": subject, "diffstat": stat[:1500],
                     "diff_tail": diff[:cfg.stdout_tail_chars]},
        ))
    # advance marker to HEAD
    head = _git(repo, "rev-parse", "HEAD").strip()
    if head:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(head)
    return out
