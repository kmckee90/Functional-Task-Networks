"""Async 'stay-current' orchestrator for FTN-Agent (design.md §5.4).

Pipeline (all pod-side; reuses the one loaded model for synthesis):
    drain inbox -> load activity -> segment into episodes ->
    synthesize NEW episodes -> save a dated block dataset to work_dir ->
    train pending blocks -> checkpoint to state_dir -> prune work_dir.

Exposes functions the app's Memory panel calls (consolidate / train_pending),
plus a __main__ autonomous loop. v1 default is MANUAL-FIRST (design.md D6): the
loop is opt-in via --watch; the app buttons drive the same functions by hand.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from capture.records import drain_inbox, load_activity
from config import CFG
from data import save_block_dataset
from synth.segment import render, segment
from synth.synthesize import synthesize_blocks
from train import make_optimizer, pending_blocks, train_pending


def _consolidated_path(cfg) -> Path:
    return Path(cfg.state_dir) / "consolidated.json"


def _load_consolidated(cfg) -> set:
    p = _consolidated_path(cfg)
    if p.exists():
        try:
            return set(json.loads(p.read_text()))
        except Exception:
            return set()
    return set()


def _save_consolidated(cfg, keys: set):
    _consolidated_path(cfg).write_text(json.dumps(sorted(keys)))


def consolidate(cfg, h, log=print) -> dict:
    """Drain inbox, segment, synthesize NEW episodes into one dated block dataset.
    Returns a summary dict. Does NOT train (call train_pending for that)."""
    n_new = drain_inbox(cfg)
    activity = load_activity(cfg)
    log(f"  ingested {n_new} new activity records; {len(activity)} total in store")
    episodes = segment(activity)
    done = _load_consolidated(cfg)
    fresh = [e for e in episodes if e.key not in done]
    log(f"  {len(episodes)} episodes ({len(fresh)} new, "
        f"{sum(e.has_failure for e in fresh)} with failures)")
    if not fresh:
        return {"records": n_new, "episodes": len(episodes), "new_episodes": 0,
                "pairs": 0, "block": None}

    pairs = synthesize_blocks(h.model, h.tok, h.ctrl, cfg, fresh, log=log)
    if not pairs:
        # still mark episodes seen so we don't re-synthesize empties forever
        done.update(e.key for e in fresh)
        _save_consolidated(cfg, done)
        return {"records": n_new, "episodes": len(episodes),
                "new_episodes": len(fresh), "pairs": 0, "block": None}

    block_key = "sess_" + datetime.now().strftime("%Y-%m-%d_%H%M%S")
    raw_text = "\n\n".join(render(e) for e in fresh)
    save_block_dataset(cfg, block_key, pairs, raw_text)
    done.update(e.key for e in fresh)
    _save_consolidated(cfg, done)
    n_ins = sum(p.get("kind") == "insight" for p in pairs)
    log(f"  saved block '{block_key}': {len(pairs)} examples ({n_ins} insights) -> work_dir")
    return {"records": n_new, "episodes": len(episodes), "new_episodes": len(fresh),
            "pairs": len(pairs), "insights": n_ins, "block": block_key}


def run_once(cfg, h, opt=None, log=print) -> dict:
    """One full cycle: consolidate + train pending."""
    summ = consolidate(cfg, h, log=log)
    n = train_pending(cfg, h, opt=opt, log=log)
    summ["blocks_trained"] = n
    return summ


def status(cfg) -> dict:
    """Lightweight status for the app's capture panel."""
    from train import load_state
    inbox = sum(1 for f in Path(cfg.inbox_dir).glob("*.jsonl") for _ in f.open())
    act = Path(cfg.activity_dir) / "activity.jsonl"
    n_act = sum(1 for _ in act.open()) if act.exists() else 0
    trained, _, _ = load_state(cfg) if (cfg.state_dir / "state.json").exists() else ([], {}, {})
    return {"inbox_lines": inbox, "activity_records": n_act,
            "trained_blocks": len(trained), "pending_datasets": len(pending_blocks(cfg))}


def main():
    ap = argparse.ArgumentParser(description="FTN-Agent daemon")
    ap.add_argument("--watch", type=int, default=0,
                    help="seconds between cycles (0 = one-shot)")
    ap.add_argument("--consolidate-only", action="store_true")
    args = ap.parse_args()

    from model import build_handles
    h = build_handles(CFG)
    opt = make_optimizer(CFG, h.wrappers)
    while True:
        t0 = time.time()
        if args.consolidate_only:
            summ = consolidate(CFG, h)
        else:
            summ = run_once(CFG, h, opt=opt)
        print(f"  cycle done in {time.time() - t0:.0f}s: {summ}", flush=True)
        if args.watch <= 0:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
