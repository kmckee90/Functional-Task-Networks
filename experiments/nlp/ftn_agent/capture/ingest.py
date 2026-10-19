"""LAPTOP-side capture runner: collect S1/S2/S3 -> write to the synced inbox.

Run this wherever Claude / your shell / the git repo live (i.e. your laptop):

    python capture/ingest.py                # one-shot: append new records to inbox
    python capture/ingest.py --watch 60     # re-run every 60s

It writes small JSONL to cfg.inbox_dir (under the synced workspace), which
DevSpace carries to the pod; the pod daemon/app drains it into the persistent
activity store (design.md §3, §7). Stdlib only — no torch, no pod needed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as `python capture/ingest.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CFG                     # noqa: E402
from capture import gitwatch, terminal, transcripts   # noqa: E402
from capture.records import write_records   # noqa: E402

_SOURCES = {"claude": transcripts, "terminal": terminal, "git": gitwatch}


def run_once(cfg) -> dict:
    counts = {}
    for name, mod in _SOURCES.items():
        try:
            recs = mod.collect(cfg)
        except Exception as e:
            print(f"  [{name}] collector error: {e}", file=sys.stderr)
            recs = []
        # Snapshot semantics: overwrite this source's inbox file with the current
        # full snapshot (drain dedups by id anyway). But NEVER blank it when the
        # collector found nothing — otherwise running capture where a source is
        # absent (e.g. transcripts on the pod) would wipe records synced from
        # elsewhere. Only rewrite when we actually have records.
        path = Path(cfg.inbox_dir) / f"{name}.jsonl"
        if recs:
            path.write_text("")
            n = write_records(path, recs)
        else:
            n = 0
        counts[name] = n
    return counts


def main():
    ap = argparse.ArgumentParser(description="FTN-Agent laptop-side capture")
    ap.add_argument("--watch", type=int, default=0,
                    help="seconds between re-runs (0 = one-shot)")
    args = ap.parse_args()
    while True:
        counts = run_once(CFG)
        total = sum(counts.values())
        print(f"  inbox <- {counts}  (total {total})  -> {CFG.inbox_dir}", flush=True)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
