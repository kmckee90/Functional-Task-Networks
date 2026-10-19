"""FTN^2: two canonical FTN stages in series.

Each stage is exactly the FTN of the paper -- a bank of H independent deep
scalar-output neuron-MLPs z = f(x), a binary top-k mask m, and a linear
readout of Dropout(z) * m. Stage 1's readout produces a d_mid interface code
that is stage 2's input; stage 2's readout produces the task logits.

The stack is the one deliberate deviation from the single-bank FTN: the
composite tasks of the abelian grid are not additively decomposable in
output space, so composition must happen *through depth* while selection
remains purely spatial (a mask over 2 banks). Commutativity of the task
group is what makes order-free spatial selection sufficient.
"""

import math

import torch
import torch.nn as nn


class NeuronBank(nn.Module):
    """H independent deep MLPs with scalar output, evaluated in parallel.

    Follows the ParallelMLP of ftn/models.py (einsum over a (H, d_in, d_out)
    weight stack per layer; kaiming-ish init).
    """

    def __init__(self, in_dim: int, H: int, depth: int, width: int):
        super().__init__()
        self.depth = depth
        sizes = [in_dim] + [width] * depth + [1]
        self.weights = nn.ParameterList(
            [nn.Parameter(torch.empty(H, sizes[i], sizes[i + 1]))
             for i in range(len(sizes) - 1)]
        )
        self.biases = nn.ParameterList(
            [nn.Parameter(torch.empty(H, sizes[i + 1]))
             for i in range(len(sizes) - 1)]
        )
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            fan_in, fan_out = sizes[i], sizes[i + 1]
            if i < depth:
                bound = math.sqrt(6.0 / fan_in)
            else:
                bound = math.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(w, -bound, bound)
            nn.init.uniform_(b, -0.1 / math.sqrt(fan_in), 0.1 / math.sqrt(fan_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(-2)  # (B, 1, d_in) broadcasts over H
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.einsum("bki,kij->bkj", h, W) + b
            if i < self.depth:
                h = torch.relu(h)
        return h.squeeze(-1)  # (B, H)


class MaskedStage(nn.Module):
    """One canonical FTN stage: bank of deep scalar-output neurons + linear
    readout of Dropout(z) * m. The building block of both the FTN^2 stack
    (abelian experiment) and the recurrent FTN (temporal experiment)."""

    def __init__(self, in_dim: int, out_dim: int, H: int,
                 neuron_depth: int, neuron_width: int, dropout: float):
        super().__init__()
        self.H = H
        self.bank = NeuronBank(in_dim, H, neuron_depth, neuron_width)
        self.out = nn.Linear(H, out_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.out(self.drop(self.bank(x)) * m)

    def forward_from_z(self, z: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Readout from precomputed (mask-independent) bank activations —
        the configurer's fast path with theta frozen."""
        return self.out(z * m)


class FTN2(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, H1: int, H2: int,
                 d_mid: int, neuron_depth: int, neuron_width: int,
                 dropout: float):
        super().__init__()
        self.H1, self.H2 = H1, H2
        self.bank1 = NeuronBank(in_dim, H1, neuron_depth, neuron_width)
        self.mid = nn.Linear(H1, d_mid, bias=False)
        self.bank2 = NeuronBank(d_mid, H2, neuron_depth, neuron_width)
        self.out = nn.Linear(H2, out_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def split_mask(self, m: torch.Tensor):
        return m[..., : self.H1], m[..., self.H1:]

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """m: (H1+H2,) binary (or STE-relaxed) mask."""
        m1, m2 = self.split_mask(m)
        h = self.mid(self.drop(self.bank1(x)) * m1)
        return self.out(self.drop(self.bank2(h)) * m2)

    def forward_from_z1(self, z1: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """Forward with a precomputed (mask-independent) bank-1 activation --
        used by the configurer, which freezes theta and iterates only on m."""
        m1, m2 = self.split_mask(m)
        h = self.mid(z1 * m1)
        return self.out(self.bank2(h) * m2)


def sequence_ce(logits: torch.Tensor, targets: torch.Tensor, L: int, p: int):
    """Cross-entropy summed over the L position heads."""
    return nn.functional.cross_entropy(
        logits.reshape(-1, L, p).transpose(1, 2), targets
    )


@torch.no_grad()
def accuracy(model: FTN2, m: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
             L: int, p: int, batch: int = 4096):
    """Per-position accuracy and exact-match over an eval split."""
    model.eval()
    pos_correct, exact, n = 0, 0, x.shape[0]
    for i in range(0, n, batch):
        logits = model(x[i:i + batch], m)
        pred = logits.reshape(-1, L, p).argmax(-1)
        pos_correct += (pred == y[i:i + batch]).sum().item()
        exact += (pred == y[i:i + batch]).all(-1).sum().item()
    model.train()
    return pos_correct / (n * L), exact / n
