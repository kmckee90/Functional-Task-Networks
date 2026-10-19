"""run_compose.py - train an FTN on the composition task (GF(2) or permutation).

Shuffled multi-task training: J training + K held-out composition tasks; each
gradient step accumulates SHUFFLE_TASKS_PER_STEP cold-start-reconfigured task
batches (a unique mask per batch), one optimizer step over the shared weights.
Held-out tasks are never trained -- they probe whether reconfiguration composes
the learned primitives into an unseen function.

Writes a checkpoint (model + effective config + task metadata) that bench.py and
eval_compose.py load to run the support-size sweep / length-generalization eval.

Usage (pod base):
    uv run python run_compose.py --config GF2Compose
    uv run python run_compose.py --config GF2Compose GF2Smoke   # fast smoke
"""

import argparse
import os
import time

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


def make_gen(C, n_tasks, seed):
    """Dispatch the data generator on COMPOSE_FAMILY (gf2 | perm)."""
    if getattr(C, "COMPOSE_FAMILY", "gf2") == "gf2":
        return DataGenGF2(n_tasks=n_tasks, n_samples=C.T,
                          rng=tasks.make_rng(seed), seed=C.SEED, **C.GF2_PARS)
    import task_comp
    return task_comp.DataGenComposition(n_tasks=n_tasks, n_samples=C.T,
                                        rng=tasks.make_rng(seed), seed=C.SEED,
                                        **C.COMP_PARS)


def make_primitive_gen(C, seed):
    """A length-1 GF(2) generator whose tasks ARE the n_ops primitives (same
    operators as the full-composition gen, since operators are seeded from
    C.SEED). Used for intermediate (primitive) pretraining via forward_k."""
    pars = {**C.GF2_PARS, "steps": int(C.COMPOSE_INTERMEDIATE_STEPS),
            "reject_reducible": False, "reject_identity": False,
            "dedupe_unique": False}
    return DataGenGF2(n_tasks=C.GF2_N_OPS, n_samples=C.T,
                      rng=tasks.make_rng(seed), seed=C.SEED, **pars)


def binacc(pred, y):
    return ((pred > 0) == (y > 0.5)).float().mean().item()


def effective_config_dict(C):
    out = {}
    for k in dir(C):
        if not k.isupper():
            continue
        if isinstance(getattr(type(C), k, None), property):
            continue
        out[k] = getattr(C, k)
    return out


def eval_split(ftn, gen, task_indices, C):
    """Cold-start reconfigure on a fresh batch, score on another; mean over tasks
    and SHUFFLE_EVAL_BATCHES. Restores ftn.m / train state."""
    m_save = ftn.m.clone()
    was_training = ftn.training
    ftn.eval()
    saved_rc = getattr(ftn, "reconfig_steps", None)
    n_rc = getattr(C, "SHUFFLE_EVAL_RECONFIG_STEPS", None)
    if n_rc is not None and saved_rc is not None:
        ftn.reconfig_steps = int(n_rc)
    accs, losses = [], []
    for t in task_indices:
        a, l = [], []
        for _ in range(int(C.SHUFFLE_EVAL_BATCHES)):
            xr, yr = gen(t)
            ftn._reconfig(xr, yr)
            xe, ye = gen(t)
            with torch.no_grad():
                pred = ftn(xe)
                a.append(binacc(pred, ye))
                l.append(ftn.loss(pred, ye).item())
        accs.append(np.mean(a)); losses.append(np.mean(l))
    if saved_rc is not None:
        ftn.reconfig_steps = saved_rc
    ftn.m.copy_(m_save)
    if was_training:
        ftn.train()
    return float(np.mean(accs)), float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=["GF2Compose"])
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out-root", default="artifacts")
    args = ap.parse_args()

    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    C = compose_config.resolve(args.config)
    if args.seed is not None:
        C.SEED = args.seed
    if args.steps is not None:
        C.SHUFFLE_STEPS = args.steps
    if C.SEED is not None:
        torch.manual_seed(C.SEED)

    name = args.name or ("+".join(args.config) + f"_s{C.SEED}")
    out_dir = os.path.join(args.out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    J, K = int(C.SHUFFLE_N_TRAIN), int(C.SHUFFLE_N_HOLDOUT)
    n_tasks = J + K
    gen = make_gen(C, n_tasks, C.SEED + 100)
    train_tasks, holdout_tasks = list(range(J)), list(range(J, n_tasks))

    print(f"run_compose: {name} | family={C.COMPOSE_FAMILY} arch={C.ARCHITECTURE} "
          f"d={gen.d if hasattr(gen,'d') else gen.input_dim} "
          f"k(steps)={C.STACK_LAYERS} n(cell layers)={getattr(C,'RNN_CELL_LAYERS',1)}")
    print(f"tasks: {J} train + {K} holdout; distinct catalog="
          f"{len(gen.enumerate_words()) if hasattr(gen,'enumerate_words') else 'n/a'}")

    ftn = build_ftn(C, gen.total_input_dim, gen.output_dim)
    opt = torch.optim.Adam([p for p in ftn.parameters() if p.requires_grad],
                           lr=C.TRAIN_LR)
    print(f"params(trainable)={sum(p.numel() for p in ftn.parameters() if p.requires_grad)}")

    N = int(C.SHUFFLE_TASKS_PER_STEP)
    n_steps = int(C.SHUFFLE_STEPS)
    rng = np.random.default_rng(C.SEED)
    curves = {"eval_x": [], "train_acc": [], "hold_acc": [], "loss_ema": []}
    ema = None

    # Intermediate (primitive) pretraining: for the first FRACTION of steps, train
    # the weight-tied cell directly on single operators via forward_k (k=1); then
    # switch to the full compositions. Eval is always on the full compositions.
    intermediate = bool(getattr(C, "COMPOSE_INTERMEDIATE", False))
    if intermediate:
        if not hasattr(ftn, "forward_k"):
            raise SystemExit("COMPOSE_INTERMEDIATE needs a recurrent arch with "
                             "forward_k (lorarnn / rnn / rnn_residual).")
        prim_gen = make_primitive_gen(C, C.SEED + 300)
        prim_pool = list(range(int(C.GF2_N_OPS)))
        k_prim = int(C.COMPOSE_INTERMEDIATE_STEPS)
        switch = int(C.COMPOSE_INTERMEDIATE_FRACTION * n_steps)
        print(f"intermediate: primitives 0..{n_steps and switch-1} "
              f"({len(prim_pool)} primitive tasks, k={k_prim}), then compositions")

    t0 = time.perf_counter()
    ftn.train()
    for step in range(n_steps):
        use_prim = intermediate and step < switch
        src_gen = prim_gen if use_prim else gen
        pool = prim_pool if use_prim else train_tasks
        picks = rng.choice(pool, size=N, replace=(N > len(pool)))
        opt.zero_grad()
        step_loss = 0.0
        for task in picks:
            x, y = src_gen(int(task))
            if use_prim:
                pred = ftn.forward_k(x, y, k_prim)   # reconfig k_prim masks + predict
            else:
                pred = ftn(x, y)                     # cold-start reconfig + predict
            loss = ftn.loss(pred, y) / N
            loss.backward()
            step_loss += loss.item()
        opt.step()
        ema = step_loss if ema is None else 0.98 * ema + 0.02 * step_loss
        if step % int(C.SHUFFLE_EVAL_EVERY) == 0 or step == n_steps - 1:
            tr_acc, _ = eval_split(ftn, gen, train_tasks, C)
            ho_acc, _ = eval_split(ftn, gen, holdout_tasks, C)
            curves["eval_x"].append(step)
            curves["train_acc"].append(tr_acc)
            curves["hold_acc"].append(ho_acc)
            curves["loss_ema"].append(ema)
            print(f"  step {step:>5}: loss_ema={ema:.4f} | train_acc={tr_acc:.3f} "
                  f"| holdout_acc={ho_acc:.3f} | {time.perf_counter()-t0:.0f}s")

    # Checkpoint for bench.py / eval_compose.py.
    ckpt = {
        "model_state": ftn.state_dict(),
        "config_dict": effective_config_dict(C),
        "config_names": args.config,
        "architecture": C.ARCHITECTURE,
        "seed": C.SEED,
        "input_size": gen.total_input_dim,
        "output_size": gen.output_dim,
        "n_train_tasks": J,
        "n_holdout_tasks": K,
        "step": n_steps,
        "curves": curves,
    }
    ckpt_path = os.path.join(out_dir, "checkpoint.pt")
    tmp = ckpt_path + ".tmp"
    torch.save(ckpt, tmp); os.replace(tmp, ckpt_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curves["eval_x"], curves["train_acc"], "-o", ms=3, label="train acc")
    ax.plot(curves["eval_x"], curves["hold_acc"], "-o", ms=3, label="holdout acc")
    ax.set_ylim(0, 1); ax.set_xlabel("gradient step"); ax.set_ylabel("per-cell acc")
    ax.set_title(f"{name}: train vs holdout"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "curves.png"), dpi=120)
    print(f"\nSaved checkpoint + curves to {out_dir}/")
    print(f"Final: train_acc={curves['train_acc'][-1]:.3f} "
          f"holdout_acc={curves['hold_acc'][-1]:.3f}")


if __name__ == "__main__":
    main()
