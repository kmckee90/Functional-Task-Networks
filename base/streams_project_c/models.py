"""
models.py — Neural network models for project_c_4.

The main architecture is NeuroModel: a bank of `hidden_size` independent
parallel MLPs ("neurons"), each mapping the shared input x to a scalar
pre-activation.  The per-neuron scalar outputs are then linearly read
out by `self.linear`; before the readout the vector of neuron outputs is
multiplied element-wise by the routing mask m, so only the active
neurons contribute to the forward pass and receive gradient updates.

This is a direct copy of `func_net_spatial/models.py::NeuroModel` with
its ParallelMLP dependency, preserving the original init scheme and
default init parameters.
"""

import math

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# ParallelMLP — bank of independent small MLPs, one per neuron slot
# ---------------------------------------------------------------------------


class ParallelMLP(nn.Module):
    """
    K parallel MLPs sharing the input x.  Weights: (K, in, out) tensors;
    biases: (K, out).  Forward is an einsum that broadcasts the input x
    across the K neuron slots.  Each slot has an identical topology
    (n_layers hidden layers of `inner_size`, then a scalar readout).

    Init: Uniform Kaiming-style bounds scaled by s=6 / s_b=3 on weights and
    biases respectively (see `_init_weights`).
    """

    def __init__(self, input_size, output_size, hidden_size, **kwargs):
        super().__init__()
        self.n_layers = kwargs["layers"]
        self.inner_size = kwargs["inner_size"]
        sizes = [input_size] + [self.inner_size] * self.n_layers + [output_size]
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(hidden_size, sizes[i], sizes[i + 1]))
                for i in range(len(sizes) - 1)
            ]
        )
        self.biases = nn.ParameterList(
            [
                nn.Parameter(torch.empty(hidden_size, sizes[i + 1]))
                for i in range(len(sizes) - 1)
            ]
        )
        self._init_weights(sizes)

    def _init_weights(self, sizes):
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            fan_in = sizes[i]
            fan_out = sizes[i + 1]
            s = 6.0
            s_b = 0.1
            if i < self.n_layers:
                # Kaiming uniform for ReLU hidden layers
                bound = math.sqrt(s / fan_in)
            else:
                # Xavier uniform for the linear output layer
                bound = math.sqrt(s / (fan_in + fan_out))
            nn.init.uniform_(w, -bound, bound)
            # Bias: uniform in [-s_b/sqrt(fan_in), s_b/sqrt(fan_in)]
            nn.init.uniform_(b, -s_b / math.sqrt(fan_in), s_b / math.sqrt(fan_in))

    def forward(self, x):
        h = x.squeeze().unsqueeze(-2)  # (B, 1, D) — broadcasts over K
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.einsum("bki,kij->bkj", h, W) + b  # (B, K, out_i)
            if i < self.n_layers:  # relu on hidden layers only, not the output
                h = torch.relu(h)
        out = h.squeeze(-1)
        return out  # (B, K)


# ---------------------------------------------------------------------------
# NeuroModel — parallel neuron bank + mask-gated linear readout
# ---------------------------------------------------------------------------


class NeuroModel(nn.Module):
    """
    Parallel-neuron network whose per-neuron outputs are gated by the
    routing mask m and linearly read out.

    Forward:
        z  = ParallelMLP(x)              # (B, hidden_size)
        z  = Dropout(z)
        y  = linear(z * m)               # (B, output_size)

    Args:
        input_size:   Dim of input x.
        output_size:  Dim of output y.
        hidden_size:  Number of parallel neuron slots (= grid_dim^2).
        layers:       Hidden layers inside each parallel MLP.
        inner_size:   Hidden width inside each parallel MLP.
        dropout_p:    Dropout probability on the per-neuron output vector.
    """

    def __init__(self, input_size, output_size, hidden_size, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.parallel = ParallelMLP(input_size, 1, hidden_size, **kwargs)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.dropout = nn.Dropout(p=kwargs["dropout_p"])

    def forward(self, x, m):
        z = self.parallel(x)  # (B, K)
        z = self.dropout(z)
        if hasattr(self, "linear"):
            return self.linear(z * m.reshape(1, -1))
        else:
            return (z * m.reshape(1, -1, 1)).mean(-2)


# ---------------------------------------------------------------------------
# Aliases for trainer/experiment compatibility
# ---------------------------------------------------------------------------

# The trainer and experiment scripts reference MaskedMLP / NoMaskMLP from
# the project_c_3 scaffold.  In project_c_4 both point at NeuroModel; the
# NoMask baseline still uses the NoMask configurer (all-ones fixed mask),
# so the underlying network is the same.
MaskedMLP = NeuroModel
NoMaskMLP = NeuroModel
