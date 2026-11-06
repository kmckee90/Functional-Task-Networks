"""Frozen MLP with trainable LoRA atoms and persistent Bayesian exact-k routing.

One shared flat mask selects atoms on every linear layer. Its square grid is
used only for visualization; the model contains no lateral smoothing.
"""

import math

import torch
import torch.nn as nn

if __package__:
    from .bayes_reconfig import BayesMaskReconfig
else:
    from bayes_reconfig import BayesMaskReconfig
if __package__:
    from .masks import kwta, task_loss
else:
    from masks import kwta, task_loss


class MaskController:
    """Shared holder for the active mask + a reconfiguration flag.

    Every LoRALinear reads the SAME controller, so one selection drives the
    whole backbone. ``detach_atoms`` is flipped True during a mask search so the
    atom weights leave the graph and only the mask logits receive gradient.
    """

    def __init__(self, hidden_size: int, k: int):
        self.H = hidden_size
        self.k = k
        self.detach_atoms = False
        self._mask = torch.zeros(hidden_size)

    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    def set_mask(self, m: torch.Tensor):
        self._mask = m


class LoRALinear(nn.Module):
    """Frozen base linear + a bank of H LoRA atoms, gated by the shared mask.

        y = base(x) + (alpha / k) * sum_j m_j (x @ A_j @ B_j)

    Every atom's delta is computed and gated by its mask entry m_j (a dense
    elementwise multiply), so every atom is on the graph and every logit gets a
    gradient during the search - inactive atoms (m_j == 0) contribute zero value
    but a nonzero straight-through gradient, matching the base FTN's
    compute-all-then-select gradient.
    """

    def __init__(self, in_f: int, out_f: int, controller: MaskController,
                 rank: int, alpha: float, weight_scale: float,
                 b0_scale: float, dropout_p: float, use_bias: bool = True):
        super().__init__()
        self.controller = controller
        self.alpha = alpha

        # Frozen random backbone weight: N(0, weight_scale / sqrt(fan_in)). No
        # base bias - the bias is provided entirely by the LoRA bias bank below
        # (or absent when use_bias is False), so nothing about the bias is static.
        base = nn.Linear(in_f, out_f, bias=False)
        with torch.no_grad():
            base.weight.normal_(0.0, weight_scale / math.sqrt(in_f))
        base.weight.requires_grad_(False)
        self.base = base

        H = controller.H
        A = torch.empty(H, in_f, rank)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        # B (and the bias bank) are small-but-nonzero (NOT zero): a zero delta
        # gives an untrained atom a zero gradient, so the mask search could never
        # select it. b0_scale is the atom-reachability knob.
        B = torch.empty(H, rank, out_f).normal_(0.0, b0_scale)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        # Per-atom bias delta: one (out_f,) vector per atom, gated and summed by
        # the same mask and scaled by the same alpha/k as the weight delta.
        self.bias_bank = (
            nn.Parameter(torch.empty(H, out_f).normal_(0.0, b0_scale))
            if use_bias else None
        )
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        m = self.controller.mask
        A, B = self.A, self.B
        if self.controller.detach_atoms:
            A, B = A.detach(), B.detach()
        xd = self.dropout(x)
        proj = torch.einsum("bi,hir->bhr", xd, A)
        proj = proj * m.view(1, -1, 1)
        delta = torch.einsum("bhr,hro->bo", proj, B)
        scaling = self.alpha / max(1, self.controller.k)
        out = y + scaling * delta
        if self.bias_bank is not None:
            bb = self.bias_bank
            if self.controller.detach_atoms:
                bb = bb.detach()
            bias_delta = torch.einsum("h,ho->o", m, bb)   # (out_f,), sum over atoms
            out = out + scaling * bias_delta
        return out


class BayesLoRAFTN(BayesMaskReconfig, nn.Module):
    """Frozen random MLP + trainable LoRA atom bank + persistent Bayesian mask.

    The single mask selects k of the H atoms; ``self.m`` (a hard (H,) buffer) is
    the current selection and ``self.mask_logits`` (from the mixin) is the
    persistent posterior over it. forward(x) predicts under the current mask,
    forward(x, y) posterior-updates first, forward(x, m=...) uses an explicit
    mask.

    Args:
        input_size, output_size:  Input / output dimensions.
        hidden_size:              H = number of LoRA atoms == square_dim**2.
        square_dim:               Side of the mask grid (viz only); H == side**2.
        classification:           Cross-entropy (one-hot) vs MSE loss.
        n_layers, width, weight_scale:
                                  Frozen backbone MLP depth / width / init scale.
        rank, alpha, dropout_p, b0_scale, use_bias:
                                  LoRA atom knobs (see LoRALinear).
        k:                        kWTA budget - number of active atoms.
        reconfig_steps, reconfig_sample_size, reconfig_noise_scale:
                                  Search knobs the mixin reuses (reconfig_steps /
                                  reconfig_noise_scale drive the annealste tau
                                  schedule; reconfig_sample_size is the cheap
                                  posterior update's default support size).
        bayes_* / sample_type:    Variational hyperparameters forwarded to
                                  BayesMaskReconfig._init_bayes.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        square_dim: int,
        classification: bool = True,
        # Frozen backbone MLP
        n_layers: int = 4,
        width: int = 64,
        weight_scale: float = 1.0,
        # LoRA atoms
        rank: int = 4,
        alpha: float = 16.0,
        dropout_p: float = 0.0,
        b0_scale: float = 0.01,
        use_bias: bool = True,
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
            f"hidden_size (atom count) {hidden_size} must equal square_dim**2 "
            f"{square_dim**2} so the mask shares the FTN grid / visualization."
        )
        self.hidden_size = hidden_size
        self.square_dim = square_dim
        self.classification = classification
        self.k = k
        self.reconfig_steps = reconfig_steps
        self.reconfig_sample_size = reconfig_sample_size
        self.reconfig_noise_scale = reconfig_noise_scale

        self.controller = MaskController(hidden_size, k)

        # Frozen random MLP: input -> width -> ... -> output, LoRA on every
        # linear, ReLU between hidden layers.
        sizes = [input_size] + [width] * n_layers + [output_size]
        self.layers = nn.ModuleList(
            LoRALinear(sizes[i], sizes[i + 1], self.controller,
                       rank=rank, alpha=alpha, weight_scale=weight_scale,
                       b0_scale=b0_scale, dropout_p=dropout_p, use_bias=use_bias)
            for i in range(len(sizes) - 1)
        )

        # Persistent hard task mask (single source of truth); the controller mask
        # is a scratch that forward()/_run() set right before running.
        self.register_buffer("m", torch.zeros(hidden_size))

        self._init_bayes(**bayes_pars)

    # ---------------------------------------------------------------- backbone

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LoRA MLP under whatever mask the controller currently holds."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
        return h

    def _run(self, x, mask, detach: bool):
        """Predictions under a (differentiable) flat atom mask. detach=True
        detaches the atom bank so a search leaves gradient only on the mask."""
        self.controller.set_mask(mask)
        self.controller.detach_atoms = detach
        try:
            return self._backbone(x)
        finally:
            self.controller.detach_atoms = False

    def _topk_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Hard exact-k mask over the flat atom bank."""
        return kwta(scores, self.k)

    # ---------------------------------------------------------------- forward

    def forward(self, x, y=None, m=None):
        if m is None:
            if y is not None:
                self._reconfig(x, y)
            m = self.m
        return self._run(x, m, detach=False)

    def loss(self, pred, y):
        return task_loss(pred, y, self.classification)
