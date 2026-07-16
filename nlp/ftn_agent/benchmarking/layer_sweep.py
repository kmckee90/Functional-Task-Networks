"""Layer-localization sweep: where in the stack does a trained bank store
usable knowledge?

Loads a trained run's checkpoint ONCE (default p2-mono, the single-LoRA arm,
evaluated mask=all-ones like eval_ftn's monolithic path), then re-runs the
closed-book eval on one conversation under different per-layer bank configs:

  full          all layers active (parity check vs the run's final eval)
  none          bank fully disabled (~= closed-book floor)
  only-A-B      bank active ONLY in decoder layers [A..B]  (sufficiency)
  drop-A-B      bank active everywhere EXCEPT [A..B]       (necessity)

plus a refinement of the best only-window: 8 -> 4 -> 2 layers.

A wrapper's delta is zeroed exactly by giving it a private LoRABankConfig with
alpha=0 (BankedLinear.scaling = alpha/k) — weights, mask, and everything else
stay identical to the official eval harness, dropout quirks included, so the
"full" row should reproduce the run's conv-26 numbers.

Token-F1 only (no cloud judge; the pod stays offline). Per-category means in
the summary. Resume-aware per config via the per-config .jsonl.

    uv run python layer_sweep.py --run p2-mono --conv conv-26
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval_ftn import answer_closed_book, make_run_config
from locomo_data import apply_corrections, load_locomo
from lora_bank import BankedLinear
from run_baselines import RESULTS_DIR, question_records
from score import token_f1


def wrappers_by_layer(model) -> dict[int, list]:
    by_layer: dict[int, list] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, BankedLinear):
            m = re.search(r"\.layers\.(\d+)\.", name)
            if m:
                by_layer.setdefault(int(m.group(1)), []).append(mod)
    return by_layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="p2-mono")
    ap.add_argument("--conv", default="conv-26")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--configs", default=None,
                    help="comma-separated tags (full, none, only-A-B, drop-A-B);"
                         " default: the full coarse+refine+drop sweep")
    args = ap.parse_args()

    cfg, meta = make_run_config(args.run)
    monolithic = bool(meta.get("monolithic"))
    from model import build_handles
    h = build_handles(cfg)
    refs = None
    if not monolithic:
        from eval_ftn import route
        from train import load_state
        _, refs, _ = load_state(cfg)
        print(f"non-monolithic run: reconfig + routing over {len(refs)} "
              f"trained blocks", flush=True)

    by_layer = wrappers_by_layer(h.model)
    n_layers = max(by_layer) + 1
    print(f"bank spans {n_layers} decoder layers, "
          f"{sum(len(v) for v in by_layer.values())} wrapped linears", flush=True)

    live_cfg = by_layer[0][0].cfg
    dead_cfg = dataclasses.replace(live_cfg, alpha=0.0)

    def set_active(layers: set[int]):
        for l, ws in by_layer.items():
            c = live_cfg if l in layers else dead_cfg
            for w in ws:
                w.cfg = c

    conv = next(c for c in load_locomo() if c.sample_id == args.conv)
    apply_corrections([conv], log=lambda *a: None)
    records = list(question_records([conv]))
    print(f"{args.conv}: {len(records)} questions", flush=True)

    out_dir = RESULTS_DIR / "sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.run}.{args.conv}.summary.json"
    summary: dict[str, dict] = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    def run_config(tag: str, layers: set[int]) -> float:
        if tag in summary:
            print(f"[{tag:12s}] cached F1={summary[tag]['f1']:.4f}", flush=True)
            return summary[tag]["f1"]
        path = out_dir / f"{args.run}.{args.conv}.{tag}.jsonl"
        done: dict[tuple, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    done[(r["sample_id"], r["q_idx"])] = r
        set_active(layers)
        t0 = time.time()
        with open(path, "a") as f:
            for r in records:
                if (r["sample_id"], r["q_idx"]) in done:
                    continue
                pred, mask = answer_closed_book(
                    h, cfg, r["question"], (r["speaker_a"], r["speaker_b"]),
                    monolithic=monolithic)
                rec = {**r, "mode": f"sweep:{tag}", "prediction": pred,
                       "f1": round(token_f1(pred, r["gold"]), 4)}
                if not monolithic:
                    ranked = route(mask, refs, cfg.k)
                    rec["route_top"] = ranked[0][1]
                    rec["route_conv_hit"] = (
                        ranked[0][1].split("/s")[0] == r["sample_id"])
                done[(r["sample_id"], r["q_idx"])] = rec
                f.write(json.dumps(rec) + "\n")
                f.flush()
        rows = list(done.values())
        f1 = sum(r["f1"] for r in rows) / len(rows)
        by_cat: dict[str, list[float]] = {}
        for r in rows:
            by_cat.setdefault(r["category_name"], []).append(r["f1"])
        cats = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_cat.items())}
        summary[tag] = {"f1": round(f1, 4), "n": len(rows), "cats": cats,
                        "layers": sorted(layers)}
        routed = [r for r in rows if "route_conv_hit" in r]
        extra = ""
        if routed:
            hit = sum(r["route_conv_hit"] for r in routed) / len(routed)
            summary[tag]["conv_hit"] = round(hit, 4)
            extra = f"  conv_hit={hit:.3f}"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"[{tag:12s}] F1={f1:.4f}{extra}  {cats}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        return f1

    if args.configs:
        for tag in args.configs.split(","):
            tag = tag.strip()
            if tag == "full":
                run_config("full", set(range(n_layers)))
            elif tag == "none":
                run_config("none", set())
            else:
                kind, a, b = tag.split("-")
                win = set(range(int(a), int(b) + 1))
                run_config(tag, win if kind == "only"
                           else set(range(n_layers)) - win)
        print("\n=== SWEEP SUMMARY (mean token-F1, "
              f"{args.run} / {args.conv}) ===")
        for tag, s in summary.items():
            print(f"  {tag:14s} {s['f1']:.4f}"
                  + (f"  conv_hit={s['conv_hit']:.3f}" if "conv_hit" in s else ""))
        print(f"-> {summary_path}")
        return

    W = args.window
    run_config("full", set(range(n_layers)))
    run_config("none", set())

    windows = [(a, min(a + W, n_layers) - 1) for a in range(0, n_layers, W)]
    for a, b in windows:
        run_config(f"only-{a:02d}-{b:02d}", set(range(a, b + 1)))

    # Refine the best coarse window: split in half twice (8 -> 4 -> 2 layers).
    def split(a: int, b: int) -> tuple[str, str]:
        half = (b - a + 1) // 2
        tags = []
        for s, e in ((a, a + half - 1), (a + half, b)):
            tags.append(f"only-{s:02d}-{e:02d}")
            run_config(tags[-1], set(range(s, e + 1)))
        return tags[0], tags[1]

    best8 = max((t for t in summary if re.fullmatch(r"only-\d+-\d+", t)),
                key=lambda t: summary[t]["f1"])
    a, b = map(int, best8.split("-")[1:])
    while b - a + 1 > 2:
        t1, t2 = split(a, b)
        best = t1 if summary[t1]["f1"] >= summary[t2]["f1"] else t2
        a, b = map(int, best.split("-")[1:])

    for a, b in windows:
        run_config(f"drop-{a:02d}-{b:02d}",
                   set(range(n_layers)) - set(range(a, b + 1)))

    print("\n=== SWEEP SUMMARY (mean token-F1, "
          f"{args.run} / {args.conv}) ===")
    for tag, s in summary.items():
        print(f"  {tag:14s} {s['f1']:.4f}")
    print(f"-> {summary_path}")


if __name__ == "__main__":
    main()
