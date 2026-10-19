"""FTN-Agent activity capture (design.md §3).

Collectors (stdlib-only, run laptop-side):
  transcripts.collect  — S1 Claude Code session transcripts (richest)
  terminal.collect     — S2 shell command log
  gitwatch.collect     — S3 git commits + diffs

records.py holds the shared Record schema + inbox/store I/O (drain_inbox,
load_activity are used pod-side).
"""
from . import gitwatch, records, terminal, transcripts

__all__ = ["transcripts", "terminal", "gitwatch", "records"]
