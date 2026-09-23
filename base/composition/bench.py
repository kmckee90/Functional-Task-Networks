"""bench.py - the support-size sweep (the money plot).

Loads a trained FTN checkpoint (run_compose.py) and an in-context DeepSet
checkpoint (run_baselines.py) and, at each support size B, compares -- on IDENTICAL
support/query draws -- for the train and held-out task splits:

  FTN            reconfigure on B support samples, score on the query
  DeepSet        in-context prediction from the B support pairs
  catalog cap    Bayes ceiling knowing the primitive set (perfect composition)
  linear cap     Bayes ceiling knowing only that the task is GF(2)-linear
                 (rank-based; sharp phase transition at B ~= d)
  memorization   1-NN over support (floor; ~chance on fresh queries)
  identity       predict y = x (floor)
  [--ttt]        per-task MLP / RNN trained from scratch on the B support samples
                 (matched to the FTN's test-time adaptation)

Answers "is the FTN hitting the cap, and how sample-efficiently does it compose
(train vs held-out)?" Everything is per-cell accuracy, averaged over tasks and
SWEEP_SEEDS. Outputs a CSV and a two-panel PNG.

Usage (pod base):
    uv run python bench.py --ftn artifacts/GF2Compose_s123/checkpoint.pt \
        --deepset artifacts/GF2Compose_s123/deepset.pt
    uv run python bench.py --ftn <ckpt> --deepset <ds> --ttt
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
import compose_config
from run_compose import make_gen
from models_ftn import build_ftn
from models_baseline import DeepSetInContext, MLP, MultiStepRNN
from oracle import (catalog_oracle, linear_agnostic_cap, identity_floor,
                    memorization_floor, _percell_and_exact)


def cfg_from_dict(d):
    return type("Ck", (compose_config.Default,), dict(d))()


def percell(logits, y):
    return ((logits > 0) == (y > 0.5)).float().mean().item()


@torch.no_grad()
def ftn_predict(ftn, xs, ys, xq, reconfig_steps):
    saved = ftn.reconfig_steps
    ftn.reconfig_steps = int(reconfig_steps)
    ftn.eval()
    ftn._reconfig(xs, ys)
    pred = ftn(xq)
    ftn.reconfig_steps = saved
    return pred


def ttt_fit(model, xs, ys, xq, steps, lr):
    """Test-time train a fresh baseline on the support set, predict the query."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(steps):
        opt.zero_grad(); lossf(model(xs), ys).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return model(xq)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ftn", required=True, help="run_compose checkpoint.pt")
    ap.add_argument("--deepset", default=None, help="run_baselines deepset.pt")
    ap.add_argument("--ttt", action="store_true", help="add per-task TTT MLP/RNN")
    ap.add_argument("--out", default=None, help="output dir (default: FTN's dir)")
    args = ap.parse_args()

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ftn, map_location="cuda" if torch.cuda.is_available()
                    else "cpu", weights_only=False)
    C = cfg_from_dict(ck["config_dict"])
    J, K = int(ck["n_train_tasks"]), int(ck["n_holdout_tasks"])
    ftn = build_ftn(C, ck["input_size"], ck["output_size"])
    ftn.load_state_dict(ck["model_state"]); ftn.eval()

    deepset = None
    if args.deepset:
        dck = torch.load(args.deepset, map_location="cuda" if torch.cuda.is_available()
                         else "cpu", weights_only=False)
        deepset = DeepSetInContext(dck["input_size"], C.DEEPSET_WIDTH, C.DEEPSET_LAYERS)
        deepset.load_state_dict(dck["model_state"]); deepset.eval()

    gen = make_gen(C, J + K, C.SEED + 100)
    splits = {"train": list(range(J)), "holdout": list(range(J, J + K))}
    Bs = list(C.SWEEP_SUPPORT_SIZES)
    Q = int(C.SWEEP_QUERY_SIZE)
    seeds = list(C.SWEEP_SEEDS)
    rc = int(getattr(C, "SHUFFLE_EVAL_RECONFIG_STEPS", None) or ftn.reconfig_steps)
    d = ck["output_size"]

    print(f"bench: d={d} k={C.STACK_LAYERS} n={getattr(C,'RNN_CELL_LAYERS',1)} "
          f"B={Bs} Q={Q} seeds={seeds} reconfig_steps(eval)={rc} ttt={args.ttt}")

    methods = ["ftn", "catalog_cap", "linear_cap", "memorization", "identity"]
    if deepset is not None:
        methods.insert(1, "deepset")
    if args.ttt:
        methods += ["ttt_mlp", "ttt_rnn"]

    rows = []
    for split, task_ids in splits.items():
        for B in Bs:
            acc = {m: [] for m in methods}
            for s in seeds:
                gen.rng.manual_seed(C.SEED + 100 + s)
                for t in task_ids:
                    xs, ys = gen.sample(t, B)
                    xq, yq = gen.sample(t, Q)
                    acc["ftn"].append(percell(ftn_predict(ftn, xs, ys, xq, rc), yq))
                    if deepset is not None:
                        with torch.no_grad():
                            acc["deepset"].append(percell(deepset(xs, ys, xq), yq))
                    cat_pc, _, _ = catalog_oracle(gen, t, xs, ys, xq, yq)
                    acc["catalog_cap"].append(cat_pc)
                    lin_pc, _, _ = linear_agnostic_cap(xs)
                    acc["linear_cap"].append(lin_pc)
                    acc["memorization"].append(memorization_floor(xs, ys, xq, yq)[0])
                    acc["identity"].append(identity_floor(xq, yq)[0])
                    if args.ttt:
                        acc["ttt_mlp"].append(percell(ttt_fit(
                            MLP(d, C.BASE_MLP_WIDTH, C.BASE_MLP_LAYERS),
                            xs, ys, xq, C.TTT_STEPS, C.TTT_LR), yq))
                        acc["ttt_rnn"].append(percell(ttt_fit(
                            MultiStepRNN(d, C.BASE_RNN_WIDTH, C.BASE_RNN_STEPS),
                            xs, ys, xq, C.TTT_STEPS, C.TTT_LR), yq))
            row = {"split": split, "B": B}
            row.update({m: float(np.mean(acc[m])) for m in methods})
            rows.append(row)
            print(f"  [{split:>7}] B={B:>4}: " +
                  " ".join(f"{m}={row[m]:.3f}" for m in methods))

    out_dir = args.out or os.path.dirname(os.path.abspath(args.ftn))
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "B"] + methods)
        w.writeheader(); w.writerows(rows)
    print(f"Saved {csv_path}")

    styles = {"ftn": ("tab:blue", "-o"), "deepset": ("tab:orange", "-o"),
              "catalog_cap": ("black", "--"), "linear_cap": ("tab:green", "--"),
              "memorization": ("tab:red", ":"), "identity": ("tab:gray", ":"),
              "ttt_mlp": ("tab:purple", "-s"), "ttt_rnn": ("tab:brown", "-s")}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, split in zip(axes, ["train", "holdout"]):
        srows = [r for r in rows if r["split"] == split]
        xs_ = [r["B"] for r in srows]
        for m in methods:
            c, ls = styles.get(m, ("k", "-"))
            ax.plot(xs_, [r[m] for r in srows], ls, color=c, ms=4, label=m)
        ax.axvline(d, color="k", alpha=0.2, lw=1)
        ax.set_xscale("log", base=2); ax.set_ylim(0.45, 1.02)
        ax.set_xlabel("support size B"); ax.set_title(f"{split} tasks")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("per-cell accuracy"); axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Composition support-size sweep (d={d}, k={C.STACK_LAYERS}, "
                 f"n={getattr(C,'RNN_CELL_LAYERS',1)}); vline = d")
    fig.tight_layout()
    png = os.path.join(out_dir, "bench_sweep.png")
    fig.savefig(png, dpi=120); print(f"Saved {png}")


if __name__ == "__main__":
    main()
