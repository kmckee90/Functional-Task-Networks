"""AnnealSTE support-conditioned mask configurer (Algorithm 1 of the paper),
extended to the two-bank mask (per-bank top-k on a joint logit vector).

Given a labeled support batch and frozen weights, S inner Adam steps optimize
mask logits; each step samples annealed noise, takes a hard per-bank KWTA
mask, and backpropagates through a straight-through sigmoid relaxation
(the AnnealSTE of base/masks.py::KWTA_Search). Returns the best-loss hard
mask seen during the search.
"""

import torch
import torch.nn as nn

from .model import FTN2, sequence_ce


def kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    out = torch.zeros_like(x)
    out[torch.topk(x, k).indices] = 1
    return out


class AnnealSTEConfigurer:
    def __init__(self, k1: int, k2: int, lr: float, L: int, p: int):
        self.k1, self.k2, self.lr, self.L, self.p = k1, k2, lr, L, p

    def __call__(self, model: FTN2, support, steps: int,
                 generator: torch.Generator | None = None) -> torch.Tensor:
        x, y = support
        H1, H2 = model.H1, model.H2
        was_training = model.training
        model.eval()  # no dropout inside the configurer

        with torch.no_grad():
            z1 = model.bank1(x)

        s = nn.Parameter(torch.randn(H1 + H2, device=x.device) * 1e-4)
        opt = torch.optim.Adam([s], lr=self.lr)

        best, best_loss = None, float("inf")
        for i in range(steps):
            tau = s.detach().std() * torch.exp(
                torch.tensor(-i / max(steps / 3, 1.0), device=x.device))
            noisy = s + torch.randn_like(s) * tau
            hard = torch.cat(
                [kwta(noisy[:H1], self.k1), kwta(noisy[H1:], self.k2)])
            sg = torch.sigmoid(s)
            m = hard + sg - sg.detach()  # straight-through

            loss = sequence_ce(model.forward_from_z1(z1, m), y, self.L, self.p)
            if loss.item() < best_loss:
                best, best_loss = hard.detach().clone(), loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()

        if was_training:
            model.train()
        return best
