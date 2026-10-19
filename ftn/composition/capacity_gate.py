"""capacity_gate.py - can the architecture even REPRESENT one primitive?

Before the composition experiment is meaningful, each architecture must be able
to fit a SINGLE primitive f_i(v) = S(M_i v) with unlimited data. The intended
mapping is one recurrent step per operator, so a dense GF(2) primitive + S-box
must be representable in that budget -- if not, the FTN will sit at chance for a
representational reason, not a compositional one, and the experiment is moot.

This gate trains weights to fit primitive 0 (task index 0 of a steps=1 generator)
with fresh data every step, at several depths (net recurrent steps allocated to
the one operator), and reports the best per-cell accuracy + exact-match reached.
Use it to find the easiest difficulty (M density x S-box size x matrix_dim) that
the architecture can represent, then run the sweep there.

Usage (on pod base):
    uv run python capacity_gate.py --config GF2Compose
    uv run python capacity_gate.py --config GF2Compose --m-density 0.2 --no-nonlinear
    uv run python capacity_gate.py --config GF2Compose GF2Smoke   # fast smoke
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
import compose_config
import tasks
from task_gf2 import DataGenGF2
from models_ftn import build_ftn
from models_baseline import MLP, MultiStepRNN, count_trainable


def binacc(logits, y):
    """(per_cell_acc, exact_match) for logits vs {0,1} targets."""
    correct = (logits > 0) == (y > 0.5)
    return correct.float().mean().item(), correct.all(dim=1).float().mean().item()


def make_primitive_gen(Config, seed):
    """A steps=1 GF(2) generator: its n_ops tasks are the primitives f_0..f_{n-1}."""
    pars = {**Config.GF2_PARS, "steps": 1, "reject_reducible": False,
            "reject_identity": False, "dedupe_unique": False}
    return DataGenGF2(n_tasks=Config.GF2_N_OPS, n_samples=Config.T,
                      rng=tasks.make_rng(seed), seed=Config.SEED, **pars)


# --------------------------------------------------------------------- fits

def fit_ftn(Config, gen, depth, task, steps, eval_every=200):
    # One primitive == one operator == ONE recurrent step (k=1); `depth` is the
    # number of feed-forward layers n INSIDE that step (the deep-cell budget).
    cfg = type("G", (type(Config),),
               {"STACK_LAYERS": 1, "RNN_CELL_LAYERS": int(depth)})()
    ftn = build_ftn(cfg, gen.d, gen.d)
    opt = torch.optim.Adam([p for p in ftn.parameters() if p.requires_grad],
                           lr=getattr(Config, "GATE_FTN_LR", Config.TRAIN_LR))
    reconfig_every = int(getattr(Config, "GATE_RECONFIG_EVERY", 1))
    best = (0.0, 0.0)
    ftn.train()
    for step in range(steps):
        x, y = gen(task)
        # Representability probe: re-search the mask periodically (adapting it to
        # the improving weights), and train the weights under the fixed mask in
        # between -- far cheaper than a full mask search every step.
        if step % reconfig_every == 0:
            with torch.no_grad():
                ftn._reconfig(x, y)
        pred = ftn(x)                          # predict under current self.m
        loss = ftn.loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0 or step == steps - 1:
            acc = eval_ftn(ftn, gen, task, Config)
            best = max(best, acc, key=lambda a: a[0])
    return best, count_trainable(ftn)


@torch.no_grad()
def eval_ftn(ftn, gen, task, Config):
    ftn.eval()
    xs, ys = gen.sample(task, Config.T)
    ftn._reconfig(xs, ys)
    xq, yq = gen.sample(task, Config.GATE_EVAL_BATCH)
    acc = binacc(ftn(xq), yq)
    ftn.train()
    return acc


def fit_baseline(model, gen, task, steps, lr, eval_every=200):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    best = (0.0, 0.0)
    model.train()
    for step in range(steps):
        x, y = gen(task)
        loss = lossf(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % eval_every == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                xq, yq = gen.sample(task, gen.n_samples * 4)
                best = max(best, binacc(model(xq), yq), key=lambda a: a[0])
            model.train()
    return best, count_trainable(model)


# --------------------------------------------------------------------- main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["GF2Compose"])
    ap.add_argument("--task", type=int, default=0, help="primitive index to fit")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--depths", default=None, help="CSV, e.g. 1,2,4")
    ap.add_argument("--matrix-dim", type=int, default=None)
    ap.add_argument("--m-density", type=float, default=None)
    ap.add_argument("--sbox-bits", type=int, default=None)
    ap.add_argument("--no-nonlinear", action="store_true")
    ap.add_argument("--out-root", default="artifacts")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    Config = compose_config.resolve(args.config)
    # Difficulty overrides.
    if args.matrix_dim is not None: Config.GF2_MATRIX_DIM = args.matrix_dim
    if args.m_density is not None:  Config.GF2_M_DENSITY = args.m_density
    if args.sbox_bits is not None:  Config.GF2_SBOX_BITS = args.sbox_bits
    if args.no_nonlinear:           Config.GF2_NONLINEAR = False
    if Config.SEED is not None:
        torch.manual_seed(Config.SEED)

    steps = args.steps or Config.GATE_STEPS
    depths = ([int(d) for d in args.depths.split(",")] if args.depths
              else list(Config.GATE_DEPTHS))

    gen = make_primitive_gen(Config, Config.SEED + 100)
    print(f"GF(2) primitive gate: d={gen.d} (N={Config.GF2_MATRIX_DIM}), "
          f"n_ops={Config.GF2_N_OPS}, m_density={Config.GF2_M_DENSITY}, "
          f"nonlinear={Config.GF2_NONLINEAR} (sbox_bits={Config.GF2_SBOX_BITS})")
    print(f"Fitting primitive f{args.task} for {steps} steps at per-operator "
          f"layer budgets n={depths} (FTN: k=1 recurrent step, n-deep cell).\n")
    print(f"Actual M density: "
          f"{[round(M.mean().item(), 3) for M in gen.Ms]}")

    # `depth` is the per-operator layer budget n: FTN uses an n-deep cell at k=1;
    # the MLP baseline uses n layers; the RNN baseline uses n recurrent steps.
    rows = []
    arch = getattr(Config, "ARCHITECTURE", "base")
    for depth in depths:
        (pc, ex), n = fit_ftn(Config, gen, depth, args.task, steps)
        rows.append((f"FTN[{arch}]", depth, n, pc, ex))
        print(f"  FTN[{arch}] n={depth:>2} params={n:>7}: "
              f"per_cell={pc:.3f} exact={ex:.3f}")
        mlp = MLP(gen.d, Config.BASE_MLP_WIDTH, depth)
        (pc, ex), n = fit_baseline(mlp, gen, args.task, steps, Config.GATE_LR)
        rows.append(("MLP", depth, n, pc, ex))
        print(f"  MLP         n={depth:>2} params={n:>7}: "
              f"per_cell={pc:.3f} exact={ex:.3f}")
        rnn = MultiStepRNN(gen.d, Config.BASE_RNN_WIDTH, depth)
        (pc, ex), n = fit_baseline(rnn, gen, args.task, steps, Config.GATE_LR)
        rows.append(("MultiStepRNN", depth, n, pc, ex))
        print(f"  MultiStepRNN n={depth:>1} params={n:>7}: "
              f"per_cell={pc:.3f} exact={ex:.3f}")

    target = Config.GATE_TARGET_ACC
    best_pc = max(r[3] for r in rows)
    verdict = ("PASS" if best_pc >= target else "FAIL")
    print(f"\nGate target per-cell acc = {target}; best reached = {best_pc:.3f} "
          f"=> {verdict}")
    if verdict == "FAIL":
        print("  -> difficulty too high for this architecture. Lower "
              "--m-density, --sbox-bits, --matrix-dim, or --no-nonlinear.")

    # Save table + plot.
    tag = args.tag or "+".join(args.config)
    out_dir = os.path.join(args.out_root, f"gate_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gate_results.txt"), "w") as f:
        f.write(f"# gate config={args.config} matrix_dim={Config.GF2_MATRIX_DIM} "
                f"m_density={Config.GF2_M_DENSITY} nonlinear={Config.GF2_NONLINEAR} "
                f"sbox_bits={Config.GF2_SBOX_BITS} steps={steps}\n")
        f.write("model\tdepth\tparams\tper_cell_acc\texact_match\n")
        for name, depth, n, pc, ex in rows:
            f.write(f"{name}\t{depth}\t{n}\t{pc:.4f}\t{ex:.4f}\n")
        f.write(f"# best_per_cell={best_pc:.4f} target={target} verdict={verdict}\n")

    labels = [f"{name}\nn={depth}" for name, depth, _, _, _ in rows]
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(rows)), 4.5))
    ax.bar(range(len(rows)), [r[3] for r in rows], color="tab:blue")
    ax.axhline(target, color="tab:red", ls="--", label=f"target {target}")
    ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1); ax.set_ylabel("best per-cell accuracy")
    ax.set_title(f"Capacity gate: fit one GF(2) primitive "
                 f"(dense={Config.GF2_M_DENSITY}, nonlinear={Config.GF2_NONLINEAR})")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "gate_results.png"), dpi=120)
    print(f"\nSaved {out_dir}/gate_results.txt and gate_results.png")
