"""run_baselines.py - train the in-context (few-shot) DeepSet baseline.

The honest matched baseline for the FTN: it ingests a support set of (x, y) pairs
as a permutation-invariant context and predicts the query, WITHOUT any weight
update at test time (like the FTN's mask reconfiguration). Trained across the
TRAINING composition tasks with random support sizes, so bench.py can evaluate it
at any sweep support size B; held-out tasks probe compositional generalization.

Saves a checkpoint bench.py loads. Per-task test-time-trained MLP/RNN baselines
are handled inside bench.py (--ttt), since they fit fresh at each eval support B.

Usage (pod base):
    uv run python run_baselines.py --config GF2Compose
"""

import argparse
import os

import numpy as np
import torch

import _paths  # noqa: F401
import compose_config
import tasks
from run_compose import make_gen, binacc, effective_config_dict
from models_baseline import DeepSetInContext, count_trainable


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
    if C.SEED is not None:
        torch.manual_seed(C.SEED)
    steps = args.steps or C.BASE_TRAIN_STEPS

    name = args.name or ("+".join(args.config) + f"_s{C.SEED}")
    out_dir = os.path.join(args.out_root, name)
    os.makedirs(out_dir, exist_ok=True)

    J, K = int(C.SHUFFLE_N_TRAIN), int(C.SHUFFLE_N_HOLDOUT)
    gen = make_gen(C, J + K, C.SEED + 100)
    train_tasks, holdout_tasks = list(range(J)), list(range(J, J + K))
    d = gen.output_dim

    model = DeepSetInContext(d, C.DEEPSET_WIDTH, C.DEEPSET_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=C.BASE_TRAIN_LR)
    lossf = torch.nn.BCEWithLogitsLoss()
    print(f"DeepSet in-context baseline: d={d} params={count_trainable(model)} "
          f"B in [{C.BASE_B_MIN},{C.BASE_B_MAX}] steps={steps}")

    rng = np.random.default_rng(C.SEED)
    model.train()
    for step in range(steps):
        task = int(rng.choice(train_tasks))
        B = int(rng.integers(C.BASE_B_MIN, C.BASE_B_MAX + 1))
        sx, sy = gen.sample(task, B)
        qx, qy = gen.sample(task, C.BASE_QUERY)
        loss = lossf(model(sx, sy, qx), qy)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                def split_acc(ts):
                    accs = []
                    for t in ts:
                        sx, sy = gen.sample(t, 256)
                        qx, qy = gen.sample(t, C.BASE_QUERY)
                        accs.append(binacc(model(sx, sy, qx), qy))
                    return float(np.mean(accs))
                tr, ho = split_acc(train_tasks), split_acc(holdout_tasks)
            model.train()
            print(f"  step {step:>5}: loss={loss.item():.4f} "
                  f"train_acc(B=256)={tr:.3f} holdout_acc(B=256)={ho:.3f}")

    ckpt = {
        "kind": "deepset",
        "model_state": model.state_dict(),
        "config_dict": effective_config_dict(C),
        "config_names": args.config,
        "input_size": d, "output_size": d,
        "n_train_tasks": J, "n_holdout_tasks": K,
    }
    path = os.path.join(out_dir, "deepset.pt")
    tmp = path + ".tmp"; torch.save(ckpt, tmp); os.replace(tmp, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
