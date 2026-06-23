"""Rebuild per-task references as USAGE-WEIGHTED masks for the neuron display.

The original refs.pt holds one binary top-k mask per task (kwta over 16 support
reconfigs), so render() blends a shared neuron equally across the tasks that
claim it. This script instead colors each neuron by HOW OFTEN each task used it.

For every trained task we reconfigure the mask on that task's training-chunk
pool (the same `raw` examples used during training) under the FINAL saved bank,
and accumulate, per neuron, the number of reconfigs that selected it. Each task's
usage vector is then normalized to unit sum so tasks contribute on equal footing
(they each got the same `steps_per_domain` budget during training). The result is
saved to checkpoints/refs_usage.pt as {task: float tensor of length H}.

render() already does proportional color mixing: it normalizes the summed
per-task colors by the per-neuron total, so storing usage WEIGHTS (not a binary
mask) makes a neuron used 70%/30% by two tasks come out as a 70/30 color blend.

Caveat: this recomputes under the final bank over each task's training pool. It
is a faithful, reproducible measure of which neurons each task uses and how
often -- NOT the literal per-step masks from training (those were never logged,
and were produced by the bank as it evolved block-by-block).

    uv run python rebuild_refs_usage.py                 # all trained tasks
    uv run python rebuild_refs_usage.py --per-task 256  # cap reconfigs/task
    uv run python rebuild_refs_usage.py --tasks robotics cscg
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from config import CFG
from data import build_examples, build_task
from model import build, load_bank
from perplexity_configurer import PerplexityConfigurer


def overlap(a, b, k):
    return float(((a > 0).float() * (b > 0).float()).sum() / k)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="subset of trained tasks (default: all in state.json)",
    )
    p.add_argument(
        "--per-task",
        type=int,
        default=CFG.steps_per_domain,
        help="max raw chunks reconfigured per task (matches the "
        "per-task training step budget by default)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=CFG.train_reconfig_steps,
        help="reconfig inner steps per chunk (default: training value)",
    )
    p.add_argument("--out", default="refs_usage.pt", help="filename under ckpt_dir")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(CFG.seed)

    state_p = CFG.ckpt_dir / "state.json"
    trained = json.loads(state_p.read_text())["trained"]
    keys = args.tasks or trained
    unknown = [k for k in keys if k not in trained]
    if unknown:
        raise SystemExit(f"not in checkpoint: {unknown}; trained: {trained}")
    print(f"  rebuilding usage refs for: {keys}", flush=True)

    model, tok, ctrl, wrappers, _ = build(CFG)
    load_bank(wrappers, CFG.ckpt_dir / "bank.pt")
    model.gradient_checkpointing_disable()
    model.config.use_cache = False  # reconfig keeps its own small graph
    model.eval()
    dev = next(model.parameters()).device
    cfgr = PerplexityConfigurer(CFG, ctrl)

    refs = {}
    t0 = time.time()
    for ti, key in enumerate(keys):
        data = build_task(CFG, tok, key)
        raw = build_examples(CFG, tok, key, data)["raw"]
        n = min(len(raw), args.per_task)
        gen = torch.Generator(device="cpu").manual_seed(CFG.seed + ti)
        idx = torch.randperm(len(raw), generator=gen, device="cpu")[:n].tolist()
        if n < len(raw):
            print(f"  {key}: sampling {n}/{len(raw)} raw chunks", flush=True)
        acc = torch.zeros(CFG.H, device="cpu")
        for j, i in enumerate(idx):
            mm, _, _ = cfgr(
                model, raw[i]["recfg"].to(dev).unsqueeze(0), steps=args.steps
            )
            acc += mm.float().cpu()
            if j % 50 == 0:
                print(f"  {key:<18} {j:>4}/{n}  ({time.time() - t0:.0f}s)", flush=True)
        total = float(acc.sum())
        refs[key] = (acc / total) if total > 0 else acc  # per-task unit sum
        used = int((acc > 0).sum())
        print(
            f"  ✓ {key}: {used} distinct neurons used over {n} reconfigs (k={CFG.k})",
            flush=True,
        )

    out_p = CFG.ckpt_dir / args.out
    torch.save({k: v.detach().cpu() for k, v in refs.items()}, out_p)
    print(f"\n  saved usage refs -> {out_p}", flush=True)

    # Shared-neuron / overlap summary (union membership).
    keys = list(refs.keys())
    if len(keys) > 1:
        print("\nunion overlap matrix (|used_i ∩ used_j| / k):")
        print(f"{'':<18}" + "".join(f"{k:>18}" for k in keys))
        for ki in keys:
            print(
                f"{ki:<18}"
                + "".join(f"{overlap(refs[ki], refs[kj], CFG.k):>18.2f}" for kj in keys)
            )


if __name__ == "__main__":
    main()
