"""Command-line entry points.

  python -m ftn_nar.cli train --method FTN_FULL --held-out MIN_ADD --seed 0 [--config x.yaml]
  python -m ftn_nar.cli evaluate --run-dir outputs/runs/<name> [--local-swap]
  python -m ftn_nar.cli exhaustive --run-dir outputs/runs/<tiny-run>
  python -m ftn_nar.cli aggregate --runs-dir outputs/runs --out outputs/reports
  python -m ftn_nar.cli selftest       # generator + model + configurer + ckpt round-trip
"""

from __future__ import annotations

import argparse
import json
import sys

from . import algorithms as A
from .config import load_config, tiny_model


def _held_out_arg(s: str) -> int:
    if s.isdigit():
        return int(s)
    return A.alg_id(s)


def cmd_train(args):
    overrides = {"method": args.method, "held_out": _held_out_arg(args.held_out),
                 "seed": args.seed}
    if args.steps:
        overrides["steps_per_task"] = args.steps
    if args.out:
        overrides["out_dir"] = args.out
    if args.device:
        overrides["device"] = args.device
    cfg = load_config(args.config, overrides)
    if args.tiny:
        cfg.model = tiny_model()
    from .trainer import Trainer, run_name
    tr = Trainer(cfg, resume=args.resume)
    print(f"run: {run_name(cfg)}  params={sum(p.numel() for p in tr.model.parameters())}",
          flush=True)
    tr.train()
    if not args.no_eval:
        from .evalrun import evaluate_run
        out = evaluate_run(tr.run_dir, do_local_swap=args.local_swap)
        _print_summary(out)


def _print_summary(out: dict):
    for mode, agg in out.get("held_out_eval", {}).items():
        keys = ["id_test/exact_mean", "ood_width_64/exact_mean", "ood_width_128/exact_mean",
                "ood_depth_16/exact_mean"]
        vals = {k.split("/")[0]: round(agg[k], 4) for k in keys if k in agg}
        print(f"[{out['method']}|{mode}] held-out {out['held_out']}: {vals}", flush=True)


def cmd_evaluate(args):
    from .evalrun import evaluate_run
    out = evaluate_run(args.run_dir, do_local_swap=args.local_swap,
                       skip_retention=args.skip_retention)
    _print_summary(out)


def cmd_exhaustive(args):
    from .exhaustive import exhaustive_run
    out = exhaustive_run(args.run_dir)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("exact_mask_exists_id", "best_id_exact", "n_near_exact_95",
                               "best_support_loss", "spearman_support_vs_screen")}, indent=1))


def cmd_aggregate(args):
    from .aggregate import build_report
    rep = build_report(args.runs_dir, args.out)
    print(f"aggregated {rep['n_runs']} runs -> {args.out}")
    if not args.no_figures:
        from .plots import build_figures
        build_figures(args.runs_dir, args.out)


def cmd_selftest(args):
    """Container startup self-test (design 20.9): generator, model forward, one
    configurer step, one training step, checkpoint round-trip."""
    import torch
    from . import generator as G
    from .batch import build_batch
    from .config import ExperimentCfg
    from .configurer import configure
    from .model import DPModel, Program
    from .masks import random_topk_mask
    import numpy as np

    cfg = ExperimentCfg()
    cfg.model = tiny_model()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs = [G.generate("train", 0, "selftest", i) for i in range(4)]
    b = build_batch(graphs, 0, with_trace=True).to(dev)
    model = DPModel(cfg.model).to(dev)
    rng = np.random.default_rng(0)
    prog = Program(gate_ext=random_topk_mask(cfg.model.extender_atoms,
                                             cfg.model.active_extender_atoms, rng, dev),
                   gate_red=random_topk_mask(cfg.model.reducer_atoms,
                                             cfg.model.active_reducer_atoms, rng, dev))
    out = model(b, prog, edge_mode="hard", return_trace=True)
    assert torch.isfinite(out["final_values"]).all()
    res = configure(model, b, cfg, (0, "selftest"))
    from .losses import model_training_loss
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    out = model(b, res.program, edge_mode="ste", return_trace=True)
    loss = model_training_loss(out, b, cfg.loss)["loss"]
    loss.backward()
    opt.step()
    import io
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    model.load_state_dict(torch.load(buf, weights_only=True))
    print(f"selftest OK on {dev} (loss={float(loss):.4f})")


def main():
    p = argparse.ArgumentParser(prog="ftn_nar")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--method", required=True)
    t.add_argument("--held-out", required=True, help="algorithm name or id 0-7")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--config", default=None)
    t.add_argument("--steps", type=int, default=None, help="steps per task override")
    t.add_argument("--out", default=None)
    t.add_argument("--device", default=None)
    t.add_argument("--tiny", action="store_true", help="tiny-bank model (design 13.2)")
    t.add_argument("--resume", action="store_true")
    t.add_argument("--no-eval", action="store_true")
    t.add_argument("--local-swap", action="store_true")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("evaluate")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--local-swap", action="store_true")
    e.add_argument("--skip-retention", action="store_true")
    e.set_defaults(fn=cmd_evaluate)

    x = sub.add_parser("exhaustive")
    x.add_argument("--run-dir", required=True)
    x.set_defaults(fn=cmd_exhaustive)

    a = sub.add_parser("aggregate")
    a.add_argument("--runs-dir", default="outputs/runs")
    a.add_argument("--out", default="outputs/reports")
    a.add_argument("--no-figures", action="store_true")
    a.set_defaults(fn=cmd_aggregate)

    s = sub.add_parser("selftest")
    s.set_defaults(fn=cmd_selftest)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
