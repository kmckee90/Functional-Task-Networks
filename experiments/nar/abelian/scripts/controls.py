"""Controls for the headline reconfig-protocol result (eval-only, loads a
trained checkpoint).

1. Random-network control: same architecture, untrained weights — does the
   configurer solve held-out cells anyway? (If yes, the result is search
   expressivity, not learned composition.)
2. Support-size sweep at held-out cells: how few labeled examples does
   composition need?
3. Configurer-steps sweep: how much search does composition need?
"""

import argparse
import json
from pathlib import Path

import torch

from ftn_nar.config import full
from ftn_nar.configurer import AnnealSTEConfigurer
from ftn_nar.data import TaskGrid
from ftn_nar.model import FTN2, accuracy


def reconfig_acc(model, grid, cells, cfg, support_n, steps, restarts=1):
    configurer = AnnealSTEConfigurer(cfg.k1, cfg.k2, cfg.configurer_lr, cfg.L, cfg.p)
    accs = []
    for cell in cells:
        x, y = grid.eval_tensors(cell, "test")
        best = 0.0
        for _ in range(restarts):
            xs, ys = grid.batch(cell, support_n, "train")
            m = configurer(model, (xs, ys), steps)
            best = max(best, accuracy(model, m, x, y, cfg.L, cfg.p)[0])
        accs.append(best)
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="main2")
    args = ap.parse_args()

    cfg = full()
    torch.manual_seed(123)
    out = Path("results") / args.run
    ckpt = torch.load(out / "checkpoint.pt", weights_only=False)
    cfg_d = ckpt["cfg"]
    grid = TaskGrid(cfg.p, cfg.L, cfg.input_train_frac, cfg_d["seed"], cfg.device)
    train_cells = [tuple(c) for c in ckpt["train_cells"]]
    held_cells = [tuple(c) for c in ckpt["held_cells"]]

    model = FTN2(cfg.L * cfg.p, cfg.L * cfg.p, cfg.H1, cfg.H2, cfg.d_mid,
                 cfg.neuron_depth, cfg.neuron_width, cfg.dropout).to(cfg.device)
    model.load_state_dict(ckpt["model"])

    results = {}

    # 1. Random-network control (same arch, fresh init, never trained)
    rand_model = FTN2(cfg.L * cfg.p, cfg.L * cfg.p, cfg.H1, cfg.H2, cfg.d_mid,
                      cfg.neuron_depth, cfg.neuron_width, cfg.dropout).to(cfg.device)
    accs = reconfig_acc(rand_model, grid, held_cells, cfg,
                        cfg.support_size, cfg.s_eval)
    results["random_net_heldout"] = accs
    print(f"random-net control (held-out, n_s={cfg.support_size}, S={cfg.s_eval}): "
          f"mean={sum(accs)/len(accs):.3f}", flush=True)

    # 2. Support-size sweep on the trained network (held-out cells)
    results["support_sweep"] = {}
    for n_s in (4, 8, 16, 32, 64, 128):
        accs = reconfig_acc(model, grid, held_cells, cfg, n_s, cfg.s_eval)
        results["support_sweep"][n_s] = accs
        print(f"support n_s={n_s}: mean={sum(accs)/len(accs):.3f}", flush=True)

    # 3. Configurer-steps sweep (held-out cells)
    results["steps_sweep"] = {}
    for S in (5, 10, 20, 40, 60, 120):
        accs = reconfig_acc(model, grid, held_cells, cfg, cfg.support_size, S)
        results["steps_sweep"][S] = accs
        print(f"steps S={S}: mean={sum(accs)/len(accs):.3f}", flush=True)

    with open(out / "controls.json", "w") as f:
        json.dump(results, f, indent=2)
    print("controls written")


if __name__ == "__main__":
    main()
