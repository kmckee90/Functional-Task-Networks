"""The 'well-solved' ceiling: a task-conditioned MLP trained JOINTLY.

Input = [one-hot sequence | one-hot a | one-hot r]; standard MLP; i.i.d.
minibatches over all training cells simultaneously. This is the setting in
which combinatorial generalization is known to work; it is the reference
point FTN must be compared against, not a competitor -- it gets task
identity as input and joint gradients, both of which FTN is denied.
"""

import torch
import torch.nn as nn

from .data import TaskGrid
from .model import sequence_ce


class JointConditionedMLP(nn.Module):
    def __init__(self, p: int, L: int, hidden: int, layers: int):
        super().__init__()
        self.p, self.L = p, L
        dims = [L * p + p + L] + [hidden] * layers + [L * p]
        mods = []
        for i in range(len(dims) - 1):
            mods.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                mods.append(nn.ReLU())
        self.net = nn.Sequential(*mods)

    def forward(self, x, a, r):
        cond_a = nn.functional.one_hot(a, self.p).float()
        cond_r = nn.functional.one_hot(r, self.L).float()
        return self.net(torch.cat([x, cond_a, cond_r], dim=-1))


def train_joint_baseline(grid: TaskGrid, train_cells, held_cells, cfg, log_fn=print):
    device = cfg.device
    model = JointConditionedMLP(cfg.p, cfg.L, cfg.joint_hidden, cfg.joint_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    cells_t = torch.tensor(train_cells, device=device)

    for step in range(cfg.joint_steps):
        which = torch.randint(0, len(train_cells), (cfg.batch_size,), device=device)
        a, r = cells_t[which, 0], cells_t[which, 1]
        idx = torch.randint(0, grid.train_seqs.shape[0], (cfg.batch_size,), device=device)
        seqs = grid.train_seqs[idx]
        # Apply per-example (a, r) via gather-free arithmetic.
        pos = (torch.arange(cfg.L, device=device).unsqueeze(0) - r.unsqueeze(1)) % cfg.L
        y = (torch.gather(seqs, 1, pos) + a.unsqueeze(1)) % cfg.p
        loss = sequence_ce(model(grid.encode(seqs), a, r), y, cfg.L, cfg.p)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 1000 == 0:
            log_fn(f"[joint baseline] step {step + 1}: loss={loss.item():.4f}")

    @torch.no_grad()
    def eval_cells(cells):
        rows = []
        for a, r in cells:
            x, y = grid.eval_tensors((a, r), "test")
            aa = torch.full((x.shape[0],), a, device=device, dtype=torch.long)
            rr = torch.full((x.shape[0],), r, device=device, dtype=torch.long)
            pred = model(x, aa, rr).reshape(-1, cfg.L, cfg.p).argmax(-1)
            rows.append(dict(cell=(a, r),
                             pos=(pred == y).float().mean().item(),
                             exact=(pred == y).all(-1).float().mean().item()))
        return rows

    return eval_cells(train_cells), eval_cells(held_cells)
