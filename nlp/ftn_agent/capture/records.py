"""Shared activity-record schema + append/dedup I/O for FTN-Agent capture.

A Record is the normalized unit every collector emits (S1/S2/S3, design.md §3).
Deliberately stdlib-only (json, hashlib, dataclasses) so the collectors run on
the LAPTOP in a plain python3 — no torch, no deps. The pod side drains the synced
inbox into the persistent (unsynced) activity store, deduping by id.

Flow (design.md §3, §7):
  laptop:  collectors -> write_records(inbox/<source>.jsonl)   [small, synced]
  pod:     drain_inbox(cfg) -> state_dir/activity/activity.jsonl [persistent]
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Record:
    kind: str                    # user_msg | assistant_msg | tool_exec | command | diff
    source: str                  # claude | terminal | git
    ts: float                    # unix time
    session_id: str = ""
    cwd: str = ""
    payload: dict = field(default_factory=dict)
    id: str = ""                 # stable dedup id (filled by finalize())

    def finalize(self) -> "Record":
        if not self.id:
            h = hashlib.sha1()
            # A record is identified by source+session+kind+a stable seed the
            # collector supplies in payload["_seed"] (a uuid / line-hash / sha).
            seed = str(self.payload.get("_seed", ""))
            h.update("|".join([self.source, self.session_id, self.kind, seed]).encode())
            if not seed:  # fall back to full-content hash when no seed given
                h.update(json.dumps(self.payload, sort_keys=True, default=str).encode())
            self.id = h.hexdigest()[:16]
        return self


def write_records(path: Path, records: list[Record]) -> int:
    """Append records as JSONL. Returns count written. Creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a") as fh:
        for r in records:
            r.finalize()
            fh.write(json.dumps(dataclasses.asdict(r), default=str) + "\n")
            n += 1
    return n


def read_records(path: Path) -> list[Record]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(Record(**json.loads(line)))
        except Exception:
            continue
    return out


def drain_inbox(cfg) -> int:
    """POD side: merge every inbox JSONL into the persistent activity store,
    deduping by id. Truncates the inbox files it consumes. Returns #new records.

    The inbox is the small synced bridge; the activity store is on the PVC,
    outside the synced workspace (design.md §7)."""
    inbox = Path(cfg.inbox_dir)
    store = Path(cfg.activity_dir) / "activity.jsonl"
    seen = {r.id for r in read_records(store)}
    new: list[Record] = []
    consumed: list[Path] = []
    for f in sorted(inbox.glob("*.jsonl")):
        for r in read_records(f):
            r.finalize()
            if r.id not in seen:
                seen.add(r.id)
                new.append(r)
        consumed.append(f)
    if new:
        write_records(store, new)
    for f in consumed:            # clear the bridge so it stays small
        try:
            f.write_text("")
        except Exception:
            pass
    return len(new)


def load_activity(cfg) -> list[Record]:
    """POD side: the full persistent activity store, time-ordered."""
    recs = read_records(Path(cfg.activity_dir) / "activity.jsonl")
    recs.sort(key=lambda r: r.ts)
    return recs
