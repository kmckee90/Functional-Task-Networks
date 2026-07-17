"""Recurrent FTN: ONE canonical FTN stage, weight-tied across time.

s_{t+1} = stage(s_t; m_t) — the mask sequence (m_1..m_T) IS the program, and
time replaces depth: composition of operators happens by iterating the stage,
so a single additive-readout bank suffices per step (each affine generator is
a permutation of the L*p one-hot bits, expressible in one stage), and an atom
learned once is reusable at every position in every program.

Two execution modes:
  hard — argmax re-encode between steps (deployment; the discrete bottleneck
         controls error accumulation, as in the abelian temporal-route eval)
  soft — per-position softmax passed between steps (training; lets gradients
         flow through the whole trace for BPTT)
"""

import torch
import torch.nn as nn

from ftn_core.model import MaskedStage


class RecurrentFTN(nn.Module):
    def __init__(self, L: int, p: int, H: int, neuron_depth: int,
                 neuron_width: int, dropout: float):
        super().__init__()
        self.L, self.p, self.H = L, p, H
        self.stage = MaskedStage(L * p, L * p, H, neuron_depth, neuron_width,
                                 dropout)

    def step(self, s: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.stage(s, m)

    def _harden(self, logits: torch.Tensor) -> torch.Tensor:
        idx = logits.reshape(-1, self.L, self.p).argmax(-1)
        oh = nn.functional.one_hot(idx, num_classes=self.p).float()
        return oh.reshape(-1, self.L * self.p)

    def _soften(self, logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
        sm = torch.softmax(logits.reshape(-1, self.L, self.p) / tau, dim=-1)
        return sm.reshape(-1, self.L * self.p)

    def execute_hard(self, x: torch.Tensor, masks) -> torch.Tensor:
        """Returns final logits; intermediate states re-encoded hard."""
        s = x
        logits = None
        for m in masks:
            logits = self.step(s, m)
            s = self._harden(logits)
        return logits if logits is not None else x

    def execute_hard_states(self, x: torch.Tensor, masks):
        """Hard execution returning every intermediate state (as one-hot)."""
        s, states = x, [x]
        for m in masks:
            s = self._harden(self.step(s, m))
            states.append(s)
        return states

    def execute_soft(self, x: torch.Tensor, masks, tau: float = 1.0) -> torch.Tensor:
        """Returns final logits; gradients flow through all steps."""
        s = x
        logits = None
        for m in masks:
            logits = self.step(s, m)
            s = self._soften(logits, tau)
        return logits if logits is not None else x


@torch.no_grad()
def trace_accuracy(model: RecurrentFTN, masks, x, y, batch: int = 8192):
    """Per-position / exact-match of hard execution over an eval split."""
    model.eval()
    pos, exact, n = 0, 0, x.shape[0]
    for i in range(0, n, batch):
        logits = model.execute_hard(x[i:i + batch], masks)
        pred = logits.reshape(-1, model.L, model.p).argmax(-1)
        pos += (pred == y[i:i + batch]).sum().item()
        exact += (pred == y[i:i + batch]).all(-1).sum().item()
    model.train()
    return pos / (n * model.L), exact / n
