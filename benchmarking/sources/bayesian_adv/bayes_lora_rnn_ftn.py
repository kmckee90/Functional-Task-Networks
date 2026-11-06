"""
bayes_lora_rnn_ftn.py - BayesFTN LoRA-RNN: a weight-tied *recurrent* LoRA cell
whose per-step mask selects low-rank atoms from a single shared pool, driven by
the bayesian subproject's persistent approximate-posterior mask inference.

Relationship to the other two LoRA / RNN variants
-------------------------------------------------
BayesLoRAFTN (bayes_lora_ftn.py): every LoRALinear owns its *own* bank of H
    atoms, and ONE shared mask gates atom j in *every* layer at once - a
    maskable unit is a whole depth-spanning column of layer-wise low-rank
    matrices, all switched on/off together.

BayesRNNResidualFTN (bayes_stack_ftn.py): one weight-tied ParallelMLP cell
    unrolled n_stack steps with a *per-step* mask that gates activations, plus
    residual skips.

This model is the cross of the two. There is ONE shared pool of H rank-r atoms
(a single low-rank weight matrix each - a "single layer's worth"), and a frozen
recurrent backbone linear applied at every step. Unrolling that cell over n_stack
steps with a *distinct mask per step* gives n_stack masks over the SAME pool, so
a step selects k of the H atoms and different steps can re-use the same atoms or
apply them in a different order - re-sequencing a shared library of low-rank
weight edits.

    h = embed(x)
    out = 0
    for t in range(n_stack):
        h = rmsnorm(h)                                   # optional, steps > 0
        z = base(h) + (alpha/k) * sum_j m_t[j] (h @ A_j @ B_j)   # shared pool
        h = relu(z)
        out += readout(h)                                # residual skip

Read-in / read-out weights (the ``io`` knob)
--------------------------------------------
The recurrent atom pool is square (width -> width), so it cannot itself contain
the dimension-changing input encoder (input -> width) or the readout
(width -> output); those are handled separately, and ``io`` picks how:

  io="fixed": the encoder and readout are FROZEN random projections - no atoms,
      not trained. Every learned, mask-selectable degree of freedom then lives in
      the shared recurrent atom pool (a reservoir-style read-in / read-out).

  io="ftn":  the encoder and readout each get their OWN bank of H LoRA atoms on
      top of the frozen random base, gated by a per-step mask block just like the
      recurrent pool - the encoder edit by the step-0 mask (embed runs once,
      before step 0), each step's readout edit by that step's mask. Atom j's
      read-in, recurrent, and read-out edits are therefore co-selected by mask
      bit j, so a maskable unit spans all three roles (yet is still re-used /
      re-sequenced across steps). The I/O banks are non-shared (their own
      parameters); only the recurrent pool is shared across steps.

In BOTH modes the encoder/readout *base* weights are frozen random (FTN-LoRA
style); io only decides whether selectable atoms are added on top. Only the atom
banks, the optional RMSNorm gain, and (io="ftn") the I/O atoms are trained; the
frozen backbones stay at their random init (weight_scale controls that scale).

Mask (the BayesFTN part)
------------------------
The per-step masks are held concatenated in ``self.m`` (size n_stack * H), so
``self.hidden_size`` reports that total and each step's H-atom grid renders side
by side. k is a per-step budget. ``self.mask_logits`` (from the mixin) is the
persistent posterior over the whole concatenated mask; every kWTA sample is taken
per step. Reconfiguration is BayesFTN's KL-regularized posterior update, with a
separate change-point reset/refit (see bayes_reconfig).

Interface mirrors ftn.BayesFTN so run_experiments.py drives it unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bayes_reconfig import BayesMaskReconfig
from ftn import kwta, task_loss


def _frozen_linear(in_f: int, out_f: int, weight_scale: float) -> nn.Linear:
    """A bias-free linear frozen at a scaled random init (FTN-LoRA base weight)."""
    lin = nn.Linear(in_f, out_f, bias=False)
    with torch.no_grad():
        lin.weight.normal_(0.0, weight_scale / math.sqrt(in_f))
    lin.weight.requires_grad_(False)
    return lin


def _atom_bank(H: int, in_f: int, out_f: int, rank: int, b0_scale: float,
               use_bias: bool):
    """A bank of H rank-r LoRA atoms (A, B[, bias]) mapping in_f -> out_f.

    A is Kaiming-uniform; B and the bias bank are small-but-nonzero so an
    unselected atom still carries a gradient the mask search can find. Returns
    (A, B, bias) as nn.Parameters (bias is None when use_bias is False).
    """
    A = torch.empty(H, in_f, rank)
    nn.init.kaiming_uniform_(A, a=math.sqrt(5))
    B = torch.empty(H, rank, out_f).normal_(0.0, b0_scale)
    bias = (
        torch.empty(H, out_f).normal_(0.0, b0_scale) if use_bias else None
    )
    return (
        nn.Parameter(A),
        nn.Parameter(B),
        nn.Parameter(bias) if bias is not None else None,
    )


class BayesLoRARNNFTN(BayesMaskReconfig, nn.Module):
    """Weight-tied recurrent LoRA cell + per-step mask over a shared atom pool.

    forward(x) predicts under the current per-step masks, forward(x, y)
    posterior-updates them first, forward(x, m=...) uses an explicit
    (concatenated) mask.

    Args:
        input_size, output_size:  Input / output dimensions.
        hidden_size:              H = size of the shared LoRA atom pool ==
                                  square_dim**2. The total maskable size
                                  (n_stack * H) is reported as ``self.hidden_size``.
        square_dim:               Side of each per-step atom grid.
        classification:           Cross-entropy (one-hot) vs MSE loss.
        n_stack:                  Number of recurrent steps / per-step masks.
        width:                    Recurrent hidden dimension (the space the atoms
                                  and the frozen backbone act on). Independent of
                                  the pool size H.
        rnn_step_rmsnorm:         Shared learnable RMSNorm on the fed-back state
                                  before every step but the first.
        weight_scale:             Fixed scale of the frozen backbones (recurrent
                                  cell and the frozen I/O projections).
        io:                       Read-in / read-out treatment: "fixed" (frozen
                                  random encoder + readout, no atoms) or "ftn"
                                  (add a maskable LoRA atom bank to each, gated by
                                  the step-0 / per-step masks). See module docstring.
        rank, alpha, dropout_p, b0_scale, use_bias:
                                  LoRA atom knobs (rank r, alpha/k scaling,
                                  activation dropout, atom-reachability init, and
                                  per-atom bias banks). Shared by the recurrent
                                  pool and, when io="ftn", the I/O banks.
        k:                        Per-step kWTA budget (atoms active per step).
        reconfig_steps, reconfig_sample_size, reconfig_noise_scale:
                                  Search knobs the mixin reuses (see BayesLoRAFTN).
        bayes_* / sample_type:    Variational hyperparameters forwarded to
                                  BayesMaskReconfig._init_bayes.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,          # H = shared atom-pool size == square_dim**2
        square_dim: int,
        classification: bool = True,
        # Recurrent topology
        n_stack: int = 4,
        width: int = 64,
        rnn_step_rmsnorm: bool = False,
        weight_scale: float = 1.0,
        io: str = "fixed",         # "fixed" | "ftn"
        # LoRA atoms (shared pool + optional I/O banks)
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
            f"hidden_size (atom-pool size) {hidden_size} must equal square_dim**2 "
            f"{square_dim**2} so each step's atom mask shares the FTN grid / viz."
        )
        assert n_stack >= 1, "n_stack must be >= 1"
        if io not in ("fixed", "ftn"):
            raise ValueError(f"io must be 'fixed' or 'ftn', got {io!r}")

        self.layer_size = hidden_size          # atoms in the shared pool (H)
        self.n_stack = n_stack
        self.width = width
        self.square_dim = square_dim
        self.classification = classification
        self.rnn_step_rmsnorm = bool(rnn_step_rmsnorm)
        self.io = io
        self.alpha = alpha
        self.k = k
        self.reconfig_steps = reconfig_steps
        self.reconfig_sample_size = reconfig_sample_size
        self.reconfig_noise_scale = reconfig_noise_scale

        # Total maskable size reported to viz/replay: one H-atom block per step.
        self.hidden_size = n_stack * hidden_size

        H = hidden_size

        # Frozen random I/O projections (both modes) + optional per-atom I/O banks
        # (io="ftn" only). The encoder lifts the input into the recurrent space;
        # the readout is summed across steps (residual). Their base weights are
        # frozen; io decides whether selectable atoms ride on top.
        self.embed = _frozen_linear(input_size, width, weight_scale)
        self.readout = _frozen_linear(width, output_size, weight_scale)
        if io == "ftn":
            self.A_in, self.B_in, self.bias_in = _atom_bank(
                H, input_size, width, rank, b0_scale, use_bias
            )
            self.A_out, self.B_out, self.bias_out = _atom_bank(
                H, width, output_size, rank, b0_scale, use_bias
            )

        if self.rnn_step_rmsnorm:
            self.rms_gain = nn.Parameter(torch.ones(width))
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()

        # Frozen random recurrent backbone (width -> width), applied every step;
        # only the shared atom pool below corrects it (FTN-LoRA style).
        self.base = _frozen_linear(width, width, weight_scale)

        # The SHARED pool of H rank-r atoms (one low-rank matrix each), reused
        # across all n_stack steps.
        self.A, self.B, self.bias_bank = _atom_bank(
            H, width, width, rank, b0_scale, use_bias
        )

        # Concatenated per-step masks (single source of truth for viz/replay).
        self.register_buffer("m", torch.zeros(self.hidden_size))

        self._init_bayes(**bayes_pars)

    # ---------------------------------------------------------------- packing

    def _split(self, m: torch.Tensor) -> list:
        """Concatenated (n_stack*H,) mask -> list of per-step atom masks."""
        return list(m.reshape(self.n_stack, self.layer_size))

    def _topk_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Hard exact-k mask *per step*: top-k atoms within each H-atom block."""
        rows = scores.reshape(self.n_stack, self.layer_size)
        return torch.cat([kwta(r, self.k) for r in rows])

    # ------------------------------------------------------------ LoRA helpers

    def _lora_delta(self, x, A, B, bias, m, detach):
        """(alpha/k) * sum_j m_j (x @ A_j @ B_j) [+ (alpha/k) * sum_j m_j bias_j].

        Every atom's delta is computed and gated by its mask entry (a dense
        multiply), so every atom logit gets a gradient during the search -
        inactive atoms contribute zero value but a nonzero straight-through
        gradient (FTN-LoRA style). detach=True detaches the atoms so the search
        leaves gradient only on the mask.
        """
        if detach:
            A, B = A.detach(), B.detach()
        proj = torch.einsum("bi,hir->bhr", x, A)   # (B, H, rank)
        proj = proj * m.view(1, -1, 1)             # gate atoms by this mask
        delta = torch.einsum("bhr,hro->bo", proj, B)
        scaling = self.alpha / max(1, self.k)
        out = scaling * delta
        if bias is not None:
            bb = bias.detach() if detach else bias
            out = out + scaling * torch.einsum("h,ho->o", m, bb)   # (out_f,)
        return out

    # -------------------------------------------------- encode / cell / readout

    def _encode(self, x: torch.Tensor, m: torch.Tensor, detach: bool) -> torch.Tensor:
        """Initial recurrent state: frozen random encoder, plus (io="ftn") a
        mask-gated read-in atom edit selected by the step-0 mask ``m``."""
        w = self.embed.weight.detach() if detach else self.embed.weight
        h = F.linear(x, w)
        if self.io == "ftn":
            h = h + self._lora_delta(x, self.A_in, self.B_in, self.bias_in, m, detach)
        return h

    def _rmsnorm(self, h: torch.Tensor, detach: bool) -> torch.Tensor:
        """Shared RMSNorm on the fed-back state before a recurrent step."""
        g = self.rms_gain.detach() if detach else self.rms_gain
        rms = h.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        return h * rms * g

    def _cell(self, h: torch.Tensor, m: torch.Tensor, detach: bool) -> torch.Tensor:
        """One recurrent LoRA step under the per-step atom mask ``m``:
        h_next = relu(base(h) + shared-pool delta gated by m)."""
        wb = self.base.weight.detach() if detach else self.base.weight
        z = F.linear(h, wb) + self._lora_delta(
            h, self.A, self.B, self.bias_bank, m, detach
        )
        return torch.relu(z)

    def _readout(self, h: torch.Tensor, m: torch.Tensor, detach: bool) -> torch.Tensor:
        """Frozen random readout, plus (io="ftn") a mask-gated read-out atom edit
        selected by this step's mask ``m``. Summed across steps by the caller."""
        w = self.readout.weight.detach() if detach else self.readout.weight
        out = F.linear(h, w)
        if self.io == "ftn":
            out = out + self._lora_delta(
                h, self.A_out, self.B_out, self.bias_out, m, detach
            )
        return out

    # ---------------------------------------------------------------- forward

    def _forward_recurrent(self, x, masks, detach: bool) -> torch.Tensor:
        """Unroll the shared recurrent LoRA cell under `masks` (per-step atom
        masks). detach=True detaches every weight so gradients reach only the
        masks; detach=False is the trainable pass and applies activation dropout.
        """
        h = self._encode(x, masks[0], detach)
        out = 0.0
        for l in range(self.n_stack):
            # Normalize the fed-back state before every step but the first (the
            # first input is embed(x) - the data - left untouched).
            if l > 0 and self.rnn_step_rmsnorm:
                h = self._rmsnorm(h, detach)
            h = self._cell(h, masks[l], detach)
            if not detach:
                h = self.dropout(h)
            out = out + self._readout(h, masks[l], detach)   # residual skip
        return out

    def _run(self, x, mask, detach: bool):
        """Predictions under a (differentiable) concatenated per-step mask."""
        return self._forward_recurrent(x, self._split(mask), detach=detach)

    def forward(self, x, y=None, m=None):
        if m is not None:
            # Explicit mask (replay / eval): use the concatenated masks as given.
            return self._forward_recurrent(x, self._split(m), detach=False)
        if y is not None:
            self._reconfig(x, y)
        return self._forward_recurrent(x, self._split(self.m), detach=False)

    def loss(self, pred, y):
        return task_loss(pred, y, self.classification)
