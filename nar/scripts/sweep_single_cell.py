"""Single-cell hyperparameter sweep: which knobs make FTN^2 learn one task well?"""

import time

import torch

from ftn_nar.config import small
from ftn_nar.configurer import AnnealSTEConfigurer
from ftn_nar.data import TaskGrid, split_grid
from ftn_nar.model import FTN2, accuracy, sequence_ce

VARIANTS = [
    dict(name="base", steps=1000),
    dict(name="s30", steps=1000, s_train=30),
    dict(name="lr3e3", steps=1000, lr=3e-3),
    dict(name="k128", steps=1000, k1=128, k2=128),
    dict(name="nodrop", steps=1000, dropout=0.0),
    dict(name="wide16", steps=1000, neuron_width=16),
    dict(name="s30_k128", steps=1000, s_train=30, k1=128, k2=128),
    dict(name="s30_k128_lr3e3", steps=1000, s_train=30, k1=128, k2=128, lr=3e-3),
]


def run(variant):
    cfg = small()
    steps = variant.pop("steps")
    name = variant.pop("name")
    for k, v in variant.items():
        setattr(cfg, k, v)
    torch.manual_seed(0)
    grid = TaskGrid(cfg.p, cfg.L, cfg.input_train_frac, cfg.seed, cfg.device)
    tc, _ = split_grid(cfg.p, cfg.L, cfg.n_train_cells, cfg.min_per_a,
                       cfg.min_per_r, cfg.seed)
    model = FTN2(cfg.L * cfg.p, cfg.L * cfg.p, cfg.H1, cfg.H2, cfg.d_mid,
                 cfg.neuron_depth, cfg.neuron_width, cfg.dropout).to(cfg.device)
    c = AnnealSTEConfigurer(cfg.k1, cfg.k2, cfg.configurer_lr, cfg.L, cfg.p)
    cell = tc[0]
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    t0 = time.time()
    curve = []
    for step in range(steps):
        xs, ys = grid.batch(cell, cfg.support_size, "train")
        m = c(model, (xs, ys), cfg.s_train)
        x, y = grid.batch(cell, cfg.batch_size, "train")
        loss = sequence_ce(model(x, m), y, cfg.L, cfg.p)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 250 == 0:
            pa, ex = accuracy(model, m, *grid.eval_tensors(cell, "test"),
                              cfg.L, cfg.p)
            curve.append(round(pa, 3))
    print(f"{name:>18}: {curve} exact={ex:.3f} ({time.time() - t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    for v in VARIANTS:
        run(v)
