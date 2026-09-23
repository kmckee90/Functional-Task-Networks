"""
bayes_stack_ftn.py - BayesFTN RNN-Residual: a weight-tied recurrent FTN cell
with a per-step mask and residual skips, driven by the persistent
approximate-posterior mask inference of the bayesian subproject.

This is the "rnn_residual" topology of base/stack_ftn.py (StackFTN with
recurrent=True, residual=True) re-implemented on the new BayesFTN design
(ftn.BayesFTN / bayes_reconfig.BayesMaskReconfig) instead of the plain per-batch
StackFTN mask search it originally used.

Backbone (unchanged in spirit from StackFTN's rnn_residual)
-----------------------------------------------------------
One shared ParallelMLP cell is unrolled over n_stack steps. A linear encoder
lifts the raw input into the recurrent hidden space as the initial state, a
distinct kWTA mask gates the cell output at every step, and a single shared
readout runs from every masked step straight to the output; the output is their
sum (the residual topology):

    h = embed(x)
    for t in range(n_stack):
        h = rmsnorm(h)           # optional, steps after the first
        z = cell(h)
        h = z * m_t
        out += readout(h)

An optional shared RMSNorm (rnn_step_rmsnorm) normalizes the fed-back state
before every step but the first, keeping the weight-tied cell from drifting.

Mask (the BayesFTN part)
------------------------
The per-step masks are held concatenated in ``self.m`` (size
n_stack * square_dim**2), so ``self.hidden_size`` reports that total and the mask
GIF lays the per-step grids out side by side. k is a *per-step* budget: each
step selects k of its square_dim**2 units. ``self.mask_logits`` (from the mixin)
is the persistent posterior over the whole concatenated mask; every kWTA sample
is taken per step (``_topk_mask`` reshapes to (n_stack, layer_size) and top-k's
each row). Reconfiguration is BayesFTN's KL-regularized posterior update carried
across batches, with a separate change-point reset/refit (see bayes_reconfig).

Pared down from base/stack_ftn.py to the bayesian subproject's scope: only the
recurrent-residual topology, one shared accumulative readout, no sequential /
recurrent-mask search, no lateral mask smoothing, no Hopfield / match-stored
recall, no dual mask.

Interface mirrors ftn.BayesFTN so run_experiments.py drives it unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayes_reconfig import BayesMaskReconfig
from ftn import ParallelMLP, kwta, task_loss


class BayesRNNResidualFTN(BayesMaskReconfig, nn.Module):
    """Weight-tied recurrent residual FTN with a per-step persistent Bayes mask.

    forward(x) predicts under the current per-step masks, forward(x, y)
    posterior-updates them first, forward(x, m=...) uses an explicit
    (concatenated) mask.

    Args:
        input_size, output_size:  Input / output dimensions.
        hidden_size:              Units *per step* == square_dim**2. The total
                                  maskable size (n_stack * hidden_size) is
                                  reported as ``self.hidden_size``.
        square_dim:               Side of each per-step mask grid.
        classification:           Cross-entropy (one-hot) vs MSE loss.
        n_stack:                  Number of recurrent steps / per-step masks.
        rnn_step_rmsnorm:         Shared learnable RMSNorm on the fed-back state
                                  before every step but the first.
        layers, inner_size:       Shared ParallelMLP cell depth / inner width.
        dropout_p:                Dropout on each step's population (forward
                                  only; the mask search sees no dropout).
        k:                        Per-step kWTA budget.
        reconfig_steps, reconfig_sample_size, reconfig_noise_scale:
                                  Search knobs the mixin reuses (see BayesLoRAFTN).
        bayes_* / sample_type:    Variational hyperparameters forwarded to
                                  BayesMaskReconfig._init_bayes.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,          # units per step == square_dim**2
        square_dim: int,
        classification: bool = True,
        # Topology
        n_stack: int = 4,
        rnn_step_rmsnorm: bool = False,
        # Shared cell backbone
        layers: int = 8,
        inner_size: int = 8,
        dropout_p: float = 0.0,
        # Reconfiguration / search knobs reused by the Bayesian mixin
        k: int = 256,
        reconfig_steps: int = 30,
        reconfig_sample_size: int = 256,
        reconfig_noise_scale: float = 1.0,
        # Bayesian posterior (forwarded to _init_bayes)
        **bayes_pars,
    ):
        super().__init__()
        assert hidden_size == square_dim**2, (
            f"hidden_size (units per step) {hidden_size} must equal square_dim**2 "
            f"{square_dim**2} so each step's mask shares the FTN grid / viz."
        )
        assert n_stack >= 1, "n_stack must be >= 1"

        self.layer_size = hidden_size          # units in one step
        self.n_stack = n_stack
        self.square_dim = square_dim
        self.classification = classification
        self.rnn_step_rmsnorm = bool(rnn_step_rmsnorm)
        self.k = k
        self.reconfig_steps = reconfig_steps
        self.reconfig_sample_size = reconfig_sample_size
        self.reconfig_noise_scale = reconfig_noise_scale

        # Total maskable size reported to viz/replay: one square_dim**2 block per
        # step, laid out side by side by mask_to_grid.
        self.hidden_size = n_stack * hidden_size

        # One shared ParallelMLP cell (hidden_size -> hidden_size) unrolled over
        # n_stack steps, plus a linear encoder lifting the raw input into the
        # recurrent hidden space as the initial state, plus one shared readout.
        self.embed = nn.Linear(input_size, hidden_size)
        self.cell = ParallelMLP(hidden_size, 1, hidden_size, layers, inner_size)
        self.readout = nn.Linear(hidden_size, output_size, bias=False)
        if self.rnn_step_rmsnorm:
            self.rms_gain = nn.Parameter(torch.ones(hidden_size))
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        # Concatenated per-step masks (single source of truth for viz/replay).
        self.register_buffer("m", torch.zeros(self.hidden_size))

        self._init_bayes(**bayes_pars)

    # ---------------------------------------------------------------- packing

    def _split(self, m: torch.Tensor) -> list:
        """Concatenated (n_stack*layer_size,) mask -> list of per-step masks."""
        return list(m.reshape(self.n_stack, self.layer_size))

    def _topk_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Hard exact-k mask *per step*: top-k within each square_dim**2 block."""
        rows = scores.reshape(self.n_stack, self.layer_size)
        return torch.cat([kwta(r, self.k) for r in rows])

    # -------------------------------------------------- encode / cell / readout

    def _encode(self, x: torch.Tensor, detach: bool) -> torch.Tensor:
        """Initial hidden state fed to the first step: the linear encoder."""
        w = self.embed.weight.detach() if detach else self.embed.weight
        b = self.embed.bias.detach() if detach else self.embed.bias
        return F.linear(x, w, b)

    def _cell(self, h: torch.Tensor, detach: bool) -> torch.Tensor:
        """Pre-mask population z from the shared ParallelMLP cell. Run manually
        (rather than self.cell(h)) so the weights can be detached during the mask
        search - the subproject's pared ParallelMLP.forward takes no detach flag,
        unlike base/ftn.py's. Mirrors that forward exactly (ReLU on hidden layers,
        sigmoid on the output)."""
        z = h.unsqueeze(-2)  # (B, 1, D) - broadcasts over the units
        for i, (W, b) in enumerate(zip(self.cell.weights, self.cell.biases)):
            if detach:
                W, b = W.detach(), b.detach()
            z = torch.einsum("bki,kij->bkj", z, W) + b
            if i < self.cell.n_layers:  # ReLU on hidden layers only
                z = torch.relu(z)
        return torch.sigmoid(z.squeeze(-1))

    def _rmsnorm(self, h: torch.Tensor, detach: bool) -> torch.Tensor:
        """Shared RMSNorm on the fed-back state before a recurrent step."""
        g = self.rms_gain.detach() if detach else self.rms_gain
        rms = h.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        return h * rms * g

    def _readout(self, h: torch.Tensor, detach: bool) -> torch.Tensor:
        w = self.readout.weight.detach() if detach else self.readout.weight
        return F.linear(h, w)

    # ---------------------------------------------------------------- forward

    def _forward_stack(self, x, masks, detach: bool) -> torch.Tensor:
        """Unroll the shared cell under `masks` (a list of per-step masks).

        detach=True detaches every weight so gradients reach only the masks
        (the mask search); detach=False is the trainable pass and also applies
        dropout to each step's population.
        """
        h = self._encode(x, detach)
        out = 0.0
        for l in range(self.n_stack):
            # Normalize the fed-back state before every step but the first (the
            # first input is embed(x) - the data - left untouched).
            if l > 0 and self.rnn_step_rmsnorm:
                h = self._rmsnorm(h, detach)
            z = self._cell(h, detach=detach)
            if not detach:
                z = self.dropout(z)
            h = z * masks[l].reshape(1, -1)
            out = out + self._readout(h, detach)   # residual: skip from every step
        return out

    def _run(self, x, mask, detach: bool):
        """Predictions under a (differentiable) concatenated mask."""
        return self._forward_stack(x, self._split(mask), detach=detach)

    def forward(self, x, y=None, m=None):
        if m is not None:
            # Explicit mask (replay / eval): use the concatenated masks as given.
            return self._forward_stack(x, self._split(m), detach=False)
        if y is not None:
            self._reconfig(x, y)
        return self._forward_stack(x, self._split(self.m), detach=False)

    def loss(self, pred, y):
        return task_loss(pred, y, self.classification)
