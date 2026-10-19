"""models_baseline.py - parameter-matched baselines for the composition task.

All operate on the d-bit state (input_size == output_size == d) and emit logits
(train with BCE-with-logits; score with logit > 0). Three families:

  MLP           : plain feed-forward net. Per-task (trained on one function) it is
                  an architecture-capacity reference; wrapped in-context it is a
                  matched few-shot baseline.
  MultiStepRNN  : a vanilla recurrent cell applied `n_steps` times as processing
                  DEPTH (the "RNN using multiple steps as processing depth" ask).
  DeepSetInContext : the honest matched few-shot baseline -- ingests the support
                  set (x, y) pairs as a permutation-invariant context, then
                  predicts the query. Handles variable composition length because
                  the support set is of that length.

`match_width_for_params` sizes a baseline to a target trainable-parameter count so
comparisons against the FTN are parameter-matched.
"""

import torch
import torch.nn as nn


def _mlp(sizes, act=nn.ReLU):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(self, d, width=256, n_layers=4):
        super().__init__()
        self.net = _mlp([d] + [width] * n_layers + [d])

    def forward(self, x):
        return self.net(x)          # logits (B, d)


class MultiStepRNN(nn.Module):
    """Encode x -> h0, apply one tanh recurrent cell n_steps times (depth), read
    out logits. Depth (not width) is the compute knob, mirroring the FTN's unroll."""

    def __init__(self, d, width=128, n_steps=4):
        super().__init__()
        self.n_steps = int(n_steps)
        self.enc = nn.Linear(d, width)
        self.cell = nn.Linear(width, width)
        self.dec = nn.Linear(width, d)

    def forward(self, x):
        h = torch.tanh(self.enc(x))
        for _ in range(self.n_steps):
            h = torch.tanh(self.cell(h))
        return self.dec(h)


class DeepSetInContext(nn.Module):
    """In-context few-shot baseline. Encodes each support (x_b, y_b) pair, pools
    them permutation-invariantly into a task embedding, then predicts the query
    y from [query_x, task_embedding]."""

    def __init__(self, d, width=256, n_layers=3):
        super().__init__()
        self.pair_enc = _mlp([2 * d] + [width] * n_layers + [width])
        self.query_net = _mlp([d + width] + [width] * n_layers + [d])

    def forward(self, support_x, support_y, query_x):
        # support_x/y: (B, d); query_x: (Q, d)
        pairs = torch.cat([support_x, support_y], dim=1)     # (B, 2d)
        ctx = self.pair_enc(pairs).mean(dim=0, keepdim=True)  # (1, width)
        ctx = ctx.expand(query_x.shape[0], -1)                # (Q, width)
        return self.query_net(torch.cat([query_x, ctx], dim=1))


def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_width_for_params(builder, target_params, lo=8, hi=2048):
    """Binary-search a width so builder(width) has ~target_params trainable params.
    builder: width -> nn.Module. Returns the chosen width."""
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        n = count_trainable(builder(mid))
        if n <= target_params:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best
