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
    lr = 3e-4
    steps = 5000
    log_every = 250



import torch
import torch.nn as nn


class MultiplicativeResidualLayer(nn.Module):
    """
    h -> h + alpha * W[(U norm(h)) * (V norm(h))]

    Quadratic multiplicative residual block with:
      - RMSNorm before multiplication
      - fused U/V projection
      - residual connection
      - explicit branch scale alpha
    """

    def __init__(self, dim, mult_dim=None, alpha_init=0.1):
        super().__init__()

        mult_dim = mult_dim or dim

        self.norm = nn.RMSNorm(dim)

        # Compute U h and V h in one matmul.
        self.factor_proj = nn.Linear(
            dim,
            2 * mult_dim,
            bias=False,
        )

        # Project multiplicative features back to hidden dimension.
        self.out_proj = nn.Linear(
            mult_dim,
            dim,
            bias=False,
        )

        # Explicitly controls strength of multiplicative branch.
        self.alpha = nn.Parameter(
            torch.tensor(float(alpha_init))
        )

        self.reset_parameters()

    def reset_parameters(self):
        # Standard variance-preserving initialization.
        nn.init.xavier_uniform_(self.factor_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, h):
        z = self.norm(h)

        u, v = self.factor_proj(z).chunk(2, dim=-1)

        interaction = u * v

        delta = self.out_proj(interaction)

        return h + self.alpha * delta


class MultiplicativeResidualNet(nn.Module):
    """
    Input projection -> multiplicative residual stack -> readout.

    If deep_supervision=True:
        y = sum_l W_l h_l

    Otherwise:
        y = W h_L
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        hidden,
        n_layers,
        mult_dim=None,
        alpha_init=0.1,
        deep_supervision=True,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden)

        self.layers = nn.ModuleList([
            MultiplicativeResidualLayer(
                dim=hidden,
                mult_dim=mult_dim,
                alpha_init=alpha_init,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.RMSNorm(hidden)
        self.deep_supervision = deep_supervision

        if deep_supervision:
            self.readouts = nn.ModuleList([
                nn.Linear(hidden, out_dim)
                for _ in range(n_layers + 1)
            ])

            # Learn how much each depth contributes.
            self.readout_logits = nn.Parameter(
                torch.zeros(n_layers + 1)
            )
        else:
            self.readout = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.input_proj(x)

        if self.deep_supervision:
            states = [h]

            for layer in self.layers:
                h = layer(h)
                states.append(h)

            weights = torch.softmax(
                self.readout_logits,
                dim=0,
            )

            y = 0.0
            for w, readout, state in zip(
                weights,
                self.readouts,
                states,
            ):
                y = y + w * readout(
                    self.final_norm(state)
                )

            return y

        for layer in self.layers:
            h = layer(h)

        return self.readout(
            self.final_norm(h)
        )

class CrossMultiplicativeLayer(nn.Module):
    """
    h_{l+1} =
        h_l
        + alpha * W[
            (U x) * (V norm(h_l))
        ]

    If h_0 is linear in x, polynomial degree increases by
    approximately one per layer rather than doubling.
    """

    def __init__(
        self,
        input_dim,
        hidden,
        mult_dim=None,
        alpha_init=0.1,
    ):
        super().__init__()

        mult_dim = mult_dim or hidden

        self.norm = nn.RMSNorm(hidden)

        self.x_proj = nn.Linear(
            input_dim,
            mult_dim,
            bias=False,
        )

        self.h_proj = nn.Linear(
            hidden,
            mult_dim,
            bias=False,
        )

        self.out_proj = nn.Linear(
            mult_dim,
            hidden,
            bias=False,
        )

        self.alpha = nn.Parameter(
            torch.tensor(float(alpha_init))
        )

        nn.init.xavier_uniform_(self.x_proj.weight)
        nn.init.xavier_uniform_(self.h_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, x, h):
        a = self.x_proj(x)
        b = self.h_proj(self.norm(h))

        delta = self.out_proj(a * b)

        return h + self.alpha * delta


class CrossMultiplicativeNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden,
        n_layers,
        mult_dim=None,
        alpha_init=0.1,
        deep_supervision=True,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_dim, hidden)

        self.layers = nn.ModuleList([
            CrossMultiplicativeLayer(
                input_dim=in_dim,
                hidden=hidden,
                mult_dim=mult_dim,
                alpha_init=alpha_init,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.RMSNorm(hidden)
        self.deep_supervision = deep_supervision

        if deep_supervision:
            self.readouts = nn.ModuleList([
                nn.Linear(hidden, out_dim)
                for _ in range(n_layers + 1)
            ])

            self.readout_logits = nn.Parameter(
                torch.zeros(n_layers + 1)
            )
        else:
            self.readout = nn.Linear(hidden, out_dim)

    def forward(self, x):
        h = self.input_proj(x)

        if self.deep_supervision:
            states = [h]

            for layer in self.layers:
                h = layer(x, h)
                states.append(h)

            weights = torch.softmax(
                self.readout_logits,
                dim=0,
            )

            y = 0.0
            for w, readout, state in zip(
                weights,
                self.readouts,
                states,
            ):
                y = y + w * readout(
                    self.final_norm(state)
                )

            return y

        for layer in self.layers:
            h = layer(x, h)

        return self.readout(
            self.final_norm(h)
        )

# ---------------------------------------------------------------- the model
class HMLLayer(nn.Module):
    """One factorized-multilinear layer, in_dim -> out_dim.

    Each of the out_dim units reads the whole input `s` and computes a linear
    term plus rank-R interactions up to order P, then tanh. Every unit's weights
    are private (rows of dense nn.Linear layers); the whole population is
    evaluated in one matmul per factor - no loop over units.
    """

    def __init__(self, in_dim, out_dim, rank=1, max_order=2):
        super().__init__()
        assert max_order >= 1, "max_order must be >= 1"
        self.out_dim, self.R, self.P = out_dim, rank, max_order

        # First-order (linear) path.
        self.linear = nn.Linear(in_dim, out_dim)

        # Factor projections: factors[p-2][q] is in_dim -> out_dim*R; each unit
        # owns R private projection vectors per factor. Orders >= 2 only.
        self.factors = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(in_dim, out_dim * rank, bias=False) for _ in range(p)
            ])
            for p in range(2, max_order + 1)
        ])

        # Mix the R rank components back to one scalar per unit, per order.
        self.rank_mix = nn.ParameterList([
            nn.Parameter(torch.randn(out_dim, rank) * 0.01)
            for _ in range(2, max_order + 1)
        ])

    def forward(self, s):
        B = s.shape[0]
        h = self.linear(s)
        for oi, _order in enumerate(range(2, self.P + 1)):
            projs = [
                layer(s).view(B, self.out_dim, self.R)
                for layer in self.factors[oi]
            ]
            inter = projs[0]
            for q in projs[1:]:
                inter = inter * q                      # elementwise -> order p
            h = h + (inter * self.rank_mix[oi]).sum(-1)  # (B,out,R) -> (B,out)
        return h


class HMLNet(nn.Module):
    """Feed-forward stack of HMLLayers + a plain linear readout."""

    def __init__(self, in_dim, out_dim, hidden, n_layers, rank=1, max_order=2):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers
        self.layers = nn.ModuleList([
            HMLLayer(dims[i], dims[i + 1], rank, max_order)
            for i in range(n_layers)
        ])
        self.readout = nn.Linear(hidden, out_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.readout(x)

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
    cfg = CONFIG
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    (x_tr, y_tr), (x_va, y_va) = make_data(cfg)

    # model = HMLNet(cfg.x_dim, 1, cfg.hidden, cfg.n_layers, cfg.rank, cfg.max_order)
    # model = MLP(cfg.x_dim, 1, cfg.hidden)
    model = CrossMultiplicativeNet(
        in_dim=cfg.x_dim,
        out_dim=1,
        hidden=cfg.hidden,
        n_layers=4,
        mult_dim=cfg.hidden,
        alpha_init=0.05,
        deep_supervision=True,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"{type(model).__name__}: hidden={cfg.hidden} layers={cfg.n_layers} rank={cfg.rank} "
        f"order={cfg.max_order} -> {n_params} params"
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    def r2(pred, y):  # 1 - MSE/var, the "accuracy" run_experiments reports
        mse = F.mse_loss(pred.squeeze(), y.squeeze()).item()
        var = float(y.float().var().item())
        return max(0.0, 1.0 - mse / max(var, 1e-8))

    for step in range(cfg.steps + 1):
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
                )


if __name__ == "__main__":
    main()
