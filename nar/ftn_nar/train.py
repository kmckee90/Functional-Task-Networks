"""Block-sequential FTN training on the task grid (Algorithm 1).

Per minibatch: draw a support batch from the current cell, run the
configurer with theta frozen to get m_cur, take one gradient step on
(current loss + mask-overlap-filtered replay loss), insert the batch into
the replay buffer. Optimizer state resets at task boundaries. Task identity
is never given to the model; it exists only in the mask.
"""

import random

import torch

from .configurer import AnnealSTEConfigurer
from .data import TaskGrid
from .model import FTN2, accuracy, sequence_ce


class ReplayBuffer:
    """Mask-overlap-filtered replay (rho in (rho_lo, rho_hi])."""

    def __init__(self, capacity: int, rho_lo: float, rho_hi: float, k_total: int,
                 n_samples: int = 1):
        self.capacity, self.rho_lo, self.rho_hi = capacity, rho_lo, rho_hi
        self.k_total, self.n_samples = k_total, n_samples
        self.slots = []
        self.stats = dict(calls=0, eligible=0, rho_sum=0.0, rho_n=0)

    def insert(self, x, y, m):
        self.slots.append((x, y, m))
        if len(self.slots) > self.capacity:
            self.slots.pop(0)

    def sample(self, m_cur, rng: random.Random):
        eligible = []
        self.stats["calls"] += 1
        for x, y, m in self.slots:
            rho = (m * m_cur).sum().item() / self.k_total
            self.stats["rho_sum"] += rho
            self.stats["rho_n"] += 1
            if self.rho_lo < rho <= self.rho_hi:
                eligible.append((x, y, m))
        if not eligible:
            return []
        self.stats["eligible"] += 1
        n = min(self.n_samples, len(eligible))
        return rng.sample(eligible, n)

    def pop_stats(self):
        s = self.stats
        out = dict(
            eligible_frac=s["eligible"] / max(s["calls"], 1),
            mean_rho=s["rho_sum"] / max(s["rho_n"], 1),
        )
        self.stats = dict(calls=0, eligible=0, rho_sum=0.0, rho_n=0)
        return out


def train_sequential(model: FTN2, grid: TaskGrid, train_cells, cfg, log_fn=print):
    device = cfg.device
    configurer = AnnealSTEConfigurer(cfg.k1, cfg.k2, cfg.configurer_lr, cfg.L, cfg.p)
    buffer = ReplayBuffer(cfg.replay_capacity, cfg.rho_lo, cfg.rho_hi,
                          cfg.k1 + cfg.k2, n_samples=cfg.replay_samples)
    rng = random.Random(cfg.seed + 2)

    order = list(train_cells)
    random.Random(cfg.seed + 3).shuffle(order)
    if getattr(cfg, "curriculum", False):
        # Primitives-first: axis cells (a, 0) and (0, r) early, composites after.
        axis = [c for c in order if c[0] == 0 or c[1] == 0]
        rest = [c for c in order if c not in axis]
        order = axis + rest
    visits = []
    for cyc in range(getattr(cfg, "cycles", 1)):
        cyc_order = list(order)
        if cyc > 0:
            random.Random(cfg.seed + 100 + cyc).shuffle(cyc_order)
        visits.extend(cyc_order)

    stored_masks, mask_emas, history = {}, {}, []
    model.train()

    for block, cell in enumerate(visits):
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)  # reset per block
        ema = torch.zeros(model.H1 + model.H2, device=device)

        for step in range(cfg.steps_per_task):
            xs, ys = grid.batch(cell, cfg.support_size, "train")
            m_cur = configurer(model, (xs, ys), cfg.s_train)
            ema = cfg.mask_ema * ema + (1 - cfg.mask_ema) * m_cur

            x, y = grid.batch(cell, cfg.batch_size, "train")
            loss = sequence_ce(model(x, m_cur), y, cfg.L, cfg.p)

            for xr, yr, mr in buffer.sample(m_cur, rng):
                loss = loss + sequence_ce(model(xr, mr), yr, cfg.L, cfg.p)

            opt.zero_grad()
            loss.backward()
            opt.step()

            xb, yb = grid.batch(cell, cfg.replay_batch, "train")
            buffer.insert(xb, yb, m_cur)

            if (step + 1) % 100 == 0:
                pos_acc, exact = accuracy(
                    model, m_cur, *grid.eval_tensors(cell, "test"), cfg.L, cfg.p)
                history.append(dict(block=block, cell=cell, step=step + 1,
                                    loss=loss.item(), pos_acc=pos_acc,
                                    exact=exact))
                log_fn(f"[block {block:02d} cell {cell}] step {step + 1}: "
                       f"loss={loss.item():.4f} pos_acc={pos_acc:.3f} "
                       f"exact={exact:.3f}")

        # Stored mask: a final high-fidelity configuration with theta frozen.
        xs, ys = grid.batch(cell, cfg.support_size * 2, "train")
        stored_masks[cell] = configurer(model, (xs, ys), cfg.s_eval).cpu()
        mask_emas[cell] = ema.cpu()
        log_fn(f"[block {block:02d} cell {cell}] replay stats: "
               f"{buffer.pop_stats()}")

    return stored_masks, mask_emas, history, order
