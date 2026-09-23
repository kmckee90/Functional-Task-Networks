"""
hml_test.py - Standalone tinkering / debugging harness for the HML core idea.

A plain feed-forward network built from *factorized multilinear* layers. Each
unit computes the same thing as one HML-FTN neuron:

    y_n = tanh( w_n . s                                   # linear term
                + sum_{p=2..P} sum_{r=1..R}               # rank-R order-P
                    mix[n,p,r] * prod_{q=1..p} (U[n,p,q,r] . s) )   interactions

The layers are stacked with UNIQUE weights per layer (NOT weight-tied, NOT
recurrent) and there is NO mask / FTN reconfiguration machinery - just a normal
network trained with full-batch Adam on the synthetic DataGenSine regression
target. Everything you'd want to poke at (width, depth, rank, order, lr, ...)
is in CONFIG, and the model (HMLLayer / HMLNet) is defined inline so you can
hack on it directly.

Run on the pod:
    uv run python hml_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import tasks


# ------------------------------------------------------------------- config
class CONFIG:
    seed = 0

    # Data: DataGenSine, one task, scalar regression.
    x_dim = 2
    task_size = 4
    frequency = 8
    n_tasks = 4
    task_idx = 0
    n_train = 4096
    n_val = 4096

    # Model: stack of `n_layers` factorized-multilinear layers of width `hidden`.
    hidden = 64        # units per layer (population size); make as large as you like
    n_layers = 2       # feed-forward depth (unique weights per layer)
    rank = 4           # R: rank of each unit's multilinear interaction
    max_order = 4      # P: highest interaction order (2 = one quadratic term)

    # Training.
    lr = 1e-4
    steps = 5000
    log_every = 250



import torch
import torch.nn as nn



class Quad0(nn.Module):
    def __init__(self, in_dim, out_dim, quad_dim):
        super().__init__()
        self.sigx = nn.Linear(in_dim, quad_dim)
        self.out = nn.Linear(in_dim + quad_dim + quad_dim*quad_dim, out_dim)

    def forward(self, x):
        q = torch.tanh(self.sigx(x))
        q2 = (q.unsqueeze(-1) @ q.unsqueeze(-2))
        q2 = q2.reshape(q2.shape[0], -1)
        z = torch.cat((x, q, q2), -1)
        return self.out(z)

class Quad1(nn.Module):
    def __init__(self, in_dim, out_dim, quad_dim):
        super().__init__()
        self.sigx = nn.Linear(in_dim, quad_dim)
        self.out = nn.Linear(in_dim + quad_dim + quad_dim*quad_dim, out_dim)

    def forward(self, x):
        q = torch.tanh(self.sigx(x))
        q2 = (q.unsqueeze(-1) @ q.unsqueeze(-2)).reshape(q.shape[0], -1)        
        z = torch.cat((x, q, q2), -1)
        return self.out(z)


class Gated(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.g = nn.Linear(in_dim, out_dim)
        self.out = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        gate = torch.sigmoid(self.g(x))
        out = torch.relu(self.out(x))
        return gate*out


class Cube0(nn.Module):
    def __init__(self, in_dim, out_dim, quad_dim):
        super().__init__()
        self.sigx = nn.Linear(in_dim, quad_dim)
        self.out = nn.Linear(in_dim + quad_dim + quad_dim*quad_dim + quad_dim*quad_dim*quad_dim, out_dim)

    def forward(self, x):
        q = torch.tanh(self.sigx(x))
        q2 = (q.unsqueeze(-1) @ q.unsqueeze(-2))
        q2 = q2.reshape(q2.shape[0], -1)
        q3 = (q2.unsqueeze(-1) @ q.unsqueeze(-2))
        q3 = q3.reshape(q3.shape[0], -1)
        z = torch.cat((x, q, q2, q3), -1)
        return self.out(z)


class Poly(nn.Module):
    def __init__(self, in_dim, out_dim, quad_dim, order=2):
        super().__init__()
        self.sigx = nn.Linear(in_dim, quad_dim)
        self.order = order
        h = quad_dim
        for i in range(2, order+1):
            h = h + quad_dim**i
        self.out = nn.Linear(in_dim + h, out_dim)

    def forward(self, x):
        q0 = torch.tanh(self.sigx(x))
        q = torch.clone(q0)
        qa = [q0]
        for _ in range(self.order-1):
            q = (q.unsqueeze(-1) @ q0.unsqueeze(-2)).reshape(q0.shape[0], -1)
            qa.append(q)
        qa = torch.cat(qa, -1)
        z = torch.cat((x, qa), -1)
        return self.out(z)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(in_dim, hidden*2),
                                   nn.ReLU(),
                                   nn.Linear(hidden*2, hidden*2),
                                   nn.ReLU(),
                                   nn.Linear(hidden*2, hidden*2),
                                   nn.ReLU(),
                                   nn.Linear(hidden*2, hidden),
                                   nn.ReLU(),
                                   nn.Linear(hidden, out_dim),
                                   )
    def forward(self, x):
        return self.model(x)

    

class GatedMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden):
        super().__init__()
        self.model = nn.Sequential(Gated(in_dim, hidden*2),
                                   Gated(hidden*2, hidden*2),
                                   Gated(hidden*2, hidden*2),
                                   Gated(hidden*2, out_dim),
                                   )
    def forward(self, x):
        return self.model(x)

class QuadMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, quad):
        super().__init__()
        self.model = nn.Sequential(Quad0(in_dim, hidden, quad),
                                   Quad0(hidden, hidden, quad),
                                   Quad0(hidden, out_dim, quad),
                                   )
    def forward(self, x):
        return self.model(x)

class CubeMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden, quad):
        super().__init__()
        self.model = nn.Sequential(Cube0(in_dim, hidden, quad),
                                   Cube0(hidden, out_dim, quad),
                                   )
    def forward(self, x):
        return self.model(x)
# --------------------------------------------------------------------- data
def make_data(cfg):
    """Fixed train / val batches for one DataGenSine task (regression)."""
    dg = tasks.DataGenSine(
        input_dim=cfg.x_dim,
        output_dim=1,
        n_tasks=cfg.n_tasks,
        task_size=cfg.task_size,
        frequency=cfg.frequency,
        classify=False,
        n_samples=cfg.n_train,
        rng=tasks.make_rng(cfg.seed + 100),
    )
    x_tr, y_tr = dg(cfg.task_idx)
    dg.n_samples = cfg.n_val
    x_va, y_va = dg(cfg.task_idx)
    return (x_tr, y_tr), (x_va, y_va)


# ----------------------------------------------------------------- training
def main():
    import time

    cfg = CONFIG
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    (x_tr, y_tr), (x_va, y_va) = make_data(cfg)

    # model = HMLNet(cfg.x_dim, 1, cfg.hidden, cfg.n_layers, cfg.rank, cfg.max_order)

    # model = MLP(cfg.x_dim, 1, cfg.hidden) #4 layer
    # model = GatedMLP(cfg.x_dim, 1, cfg.hidden) #4 layer
    model = QuadMLP(cfg.x_dim, 1, cfg.hidden//2, 20)
    # model = CubeMLP(cfg.x_dim, 1, cfg.hidden//8, 16) #2 layer
    model = Poly(cfg.x_dim, 1, 20, order=4)

    # model = CrossMultiplicativeNet(
    #     in_dim=cfg.x_dim,
    #     out_dim=1,
    #     hidden=cfg.hidden,
    #     n_layers=4,
    #     mult_dim=cfg.hidden,
    #     alpha_init=0.05,
    #     deep_supervision=True,
    # )

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"{type(model).__name__}: hidden={cfg.hidden} layers={cfg.n_layers} rank={cfg.rank} "
        f"order={cfg.max_order} -> {n_params} params"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-1)

    def r2(pred, y):  # 1 - MSE/var, the "accuracy" run_experiments reports
        mse = F.mse_loss(pred.squeeze(), y.squeeze()).item()
        var = float(y.float().var().item())
        return max(0.0, 1.0 - mse / max(var, 1e-8))

    for step in range(cfg.steps + 1):
        t0 = time.time()
        model.train()
        loss = F.mse_loss(model(x_tr), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % cfg.log_every == 0:
            model.eval()
            with torch.no_grad():
                vp = model(x_va)
                print(
                    f"step {step:5d} | train_mse {loss.item():.5f} "
                    f"| val_mse {F.mse_loss(vp, y_va).item():.5f} "
                    f"| val_r2 {r2(vp, y_va):.4f}"
                    f"| {time.time()-t0:.4f}s"

                )


if __name__ == "__main__":
    main()
