"""
lora_rnn_ftn.py - FTN-LoRA-RNN: a recurrent FTN whose per-step "neurons" are
low-rank weight atoms (FTN-LoRA) selected fresh by the mask at every recurrent
step (RNN FTN).

Selected with ARCHITECTURE="lorarnn" in config.py. Presents the same interface
as ftn.FTN (`ftn(x, y)` reconfigures + predicts, `ftn(x)` predicts,
`ftn(x, m=...)` predicts under an explicit mask, `ftn._reconfig(x, y)`,
`ftn.loss`, `ftn.m`, `ftn.hidden_size`), so run_experiments.py drives it
unchanged.

What it is
==========
This architecture recycles the parameters of BOTH the RNN FTN and FTN-LoRA:

  * RNN FTN (stack_ftn.py, recurrent) is ONE weight-tied cell unrolled over N
    steps, activating a *different mask at each step* m_0..m_{N-1}. There is one
    layer of weights reused every step; the mask is what makes each step do a
    different thing.

  * FTN-LoRA (lora_ftn.py) has a frozen random backbone plus a bank of H = k-of
    low-rank atoms (A_j @ B_j); the mask selects k active atoms whose rank-r
    deltas are added to the backbone weight.

LoRA-RNN is a SINGLE recurrent layer whose weight matrix is an FTN-LoRA bank:

    W_eff(m_t) = W_base + (alpha / k) * sum_j m_t[j] * (A_j @ B_j)
    h_{t+1}    = tanh( W_eff(m_t) @ h_t + b_eff(m_t) )

so there is ONE frozen random recurrent backbone W_base and ONE trained atom
bank (H = square_dim**2 atoms, each a rank-r weight matrix -- "each FTN neuron
is a low-rank weight matrix for that single layer"). The reconfiguration selects
a *distinct* mask m_t for every recurrent step, so the effective recurrent
weight matrix -- the operation applied at that step -- is chosen uniquely per
step from the shared atom pool. The parameters of the layer (the atom bank) are
recycled across all steps; only the selection changes.

Input / output weights (FTNLORA_IO)
===================================
The recurrent bank is shared across every mask selection, and so, in the
simplest form, would the input encoder and output readout be. Both are legit
ways to keep the input/output weights from being a single trainable tensor that
every task/step overwrites:

  io="fixed"  The input encoder (x -> width) and output readout (width -> out)
              are FROZEN random linear maps. Nothing about them is trained or
              selected, so there is no shared trainable parameter to interfere
              across masks (the FTN-LoRA "frozen random backbone" philosophy
              applied to I/O). Reconfiguration selects ONLY the N per-step
              recurrent masks.

  io="ftn"    The input encoder and the output readout are each their own
              FTN-LoRA atom bank with a DEDICATED mask (m_in, m_out), searched
              jointly with the per-step recurrent masks. Reconfiguration then
              selects the input weights, the per-step recurrent weights, and the
              output weights independently -- every weight the datum flows
              through is a mask-selected subspace, nothing trainable is shared
              across tasks.

Mask layout (self.m)
====================
All masks are concatenated into self.m (size n_masks * square_dim**2), one
square_dim**2 block per mask, so viz.MaskAnimator / replay see them side by side
exactly like the stacked/recurrent FTN. Order:

    io="fixed":  [ m_0, m_1, ..., m_{N-1} ]                      (N masks)
    io="ftn":    [ m_in, m_0, ..., m_{N-1}, m_out ]              (N+2 masks)

Reconfiguration
===============
Simultaneous annealed-STE (or "fast" soft-sigmoid) search over every mask's
logits at once, minimizing the final task loss with all weights detached so only
the masks move -- the same search as the recurrent StackFTN, generalized to the
input/recurrent/output mask set. Match-stored Hopfield recall (whole-mask or
per-slot pooled) is supported just like StackFTN.
"""

import math

import torch
from reconfig_optimizer import make_reconfig_optimizer, validate_reconfig_optimizer
import torch.nn as nn
import torch.nn.functional as F

# Reuse the backbone + mask primitives so the LoRA-RNN masks live in the same
# grid / viz world as the base FTN, FTN-LoRA and the stacked/recurrent FTN.
from ftn import (
    _uniform_conv,
    best_stored,
    kwta,
    mask_to_grid,
    mask_to_vec,
    task_loss,
)


class LoRABank(nn.Module):
    """A frozen random base linear + a bank of H rank-r LoRA atoms, gated by a
    per-call mask.

        y = base(x) + (alpha / k) * sum_j m_j (x @ A_j @ B_j) [+ bias term]

    Every atom's delta is computed and gated by its mask entry m_j (a dense
    elementwise multiply), so every atom is on the graph and every logit gets a
    gradient during the mask search -- inactive atoms (m_j == 0) contribute zero
    value but a nonzero STE gradient, exactly like FTN-LoRA's LoRALinear. The
    base weight is frozen random (never trained); `detach` (set during the mask
    search) detaches the trainable atoms so only the mask logits receive
    gradient.
    """

    def __init__(self, in_f: int, out_f: int, n_atoms: int, rank: int,
                 alpha: float, weight_scale: float, b0_scale: float,
                 dropout_p: float, use_bias: bool = True):
        super().__init__()
        self.alpha = alpha

        # Frozen random backbone weight: N(0, weight_scale / sqrt(fan_in)).
        base = nn.Linear(in_f, out_f, bias=False)
        with torch.no_grad():
            base.weight.normal_(0.0, weight_scale / math.sqrt(in_f))
        base.weight.requires_grad_(False)
        self.base = base

        A = torch.empty(n_atoms, in_f, rank)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        # B (and the bias bank) are small-but-nonzero so an untrained atom has a
        # nonzero gradient and can be selected by the mask search (b0_scale is
        # the atom-reachability knob, grouped with the reconfiguration params).
        B = torch.empty(n_atoms, rank, out_f).normal_(0.0, b0_scale)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)
        self.bias_bank = (
            nn.Parameter(torch.empty(n_atoms, out_f).normal_(0.0, b0_scale))
            if use_bias else None
        )
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor, k: int,
                detach: bool) -> torch.Tensor:
        y = self.base(x)                       # base is frozen (always static)
        A, B = self.A, self.B
        if detach:
            A, B = A.detach(), B.detach()
        xd = self.dropout(x) if not detach else x
        proj = torch.einsum("bi,hir->bhr", xd, A)
        proj = proj * mask.view(1, -1, 1)
        delta = torch.einsum("bhr,hro->bo", proj, B)
        scaling = self.alpha / max(1, k)
        out = y + scaling * delta
        if self.bias_bank is not None:
            bb = self.bias_bank.detach() if detach else self.bias_bank
            bias_delta = torch.einsum("h,ho->o", mask, bb)   # sum over atoms
            out = out + scaling * bias_delta
        return out


_ACTIVATIONS = {
    "relu": F.relu,
    "tanh": torch.tanh,
    "gelu": F.gelu,
    "identity": lambda x: x,
    "none": lambda x: x,
}


class DeepLoRABank(nn.Module):
    """An n-layer deep feed-forward LoRA MLP that is gated by a SINGLE per-call
    mask shared across all n layers.

    Each of the n layers (width -> width) is its own LoRABank (frozen random base
    + a bank of H rank-r atoms). The same mask selects the active atoms in every
    layer, so one atom ("neuron") supplies a coordinated LoRA adaptation across
    the whole n-layer block, callable by mask reconfiguration. An inner activation
    (ReLU by default) is applied BETWEEN the layers (n-1 times); the final layer's
    output is returned raw so the caller can apply the recurrent step activation.

    n_layers == 1 is exactly a single LoRABank (no inner activation), so the
    weight-tied recurrent cell reduces to the original LoRA-RNN.
    """

    def __init__(self, width: int, n_layers: int, n_atoms: int, rank: int,
                 alpha: float, weight_scale: float, b0_scale: float,
                 dropout_p: float, use_bias: bool, inner_activation: str = "relu"):
        super().__init__()
        self.n_layers = int(n_layers)
        self.inner_act = _ACTIVATIONS[str(inner_activation).lower()]
        self.layers = nn.ModuleList([
            LoRABank(width, width, n_atoms, rank, alpha, weight_scale,
                     b0_scale, dropout_p, use_bias)
            for _ in range(self.n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor, k: int,
                detach: bool) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, mask, k, detach)
            if i < self.n_layers - 1:
                h = self.inner_act(h)
        return h


from overlap_regularizer import OverlapRegularizer


class LoRARNNFTN(OverlapRegularizer, nn.Module):
    """One weight-tied recurrent CELL unrolled n_steps steps, a distinct mask per
    step (plus optional dedicated input/output masks). The cell is an
    ``cell_layers``-deep LoRA MLP (DeepLoRABank): each step is a deep feed-forward
    block with inner ReLU nonlinearities whose weights are adapted, across all
    ``cell_layers`` layers at once, by that step's mask (cell_layers == 1 is the
    original single-layer recurrent cell).

    Interface mirrors ftn.FTN: forward(x) predicts under the current masks,
    forward(x, y) reconfigures first, forward(x, m=...) uses an explicit
    (concatenated) mask.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,          # H = number of LoRA atoms == square_dim**2
        square_dim: int,
        classification: bool = False,
        binary: bool = False,
        # Topology
        io: str = "fixed",         # "fixed" | "ftn" (dedicated I/O atom banks)
        n_steps: int = 4,          # recurrent steps k (== distinct per-step masks)
        rnn_step_rmsnorm: bool = False,
        cell_layers: int = 1,      # feed-forward layers n PER recurrent step (the
                                   #   cell is an n-layer deep LoRA MLP; 1 = single
                                   #   layer, the original LoRA-RNN cell)
        inner_activation: str = "relu",   # activation BETWEEN the n cell layers
        step_activation: str = "tanh",    # activation on each recurrent step output
        # LoRA atoms / recurrent hidden width
        width: int = 64,           # recurrent hidden-state width
        rank: int = 4,
        alpha: float = 16.0,
        weight_scale: float = 1.0,
        b0_scale: float = 0.01,
        use_bias: bool = True,
        dropout_p: float = 0.0,
        # Reconfiguration (mask search), shared across masks
        k: int = 64,
        reconfig_steps: int = 30,
        reconfig_lr: float = 0.3,
        reconfig_optimizer: str = "Adam",
        reconfig_sample_size: int = 2048,
        reconfig_algorithm: str = "annealste",   # "annealste" or "fast"
        reconfig_init_sd: float = 1e-5,
        reconfig_noise_scale: float = 1.0,
        # Match-stored reconfiguration
        match_stored: bool = False,
        match_stored_after: int = 0,
        match_stored_topk: int = 1,
        match_tie_steps: bool = True,
        mask_kernel_size: int = 3,
        mask_lateral_steps: int = 0,
        mask_lateral_polyak: float = 0.0,
        mask_lateral_noise_sd: float = 0.0,
        torus: bool = False,
        overlap_mode: str = "seen_logits",
        overlap_lambda: float = 0.0,
        overlap_threshold: float = 0.5,
        overlap_temperature: float = 0.2,
        reconfig_entropy_coef: float = 0.0,
    ):
        super().__init__()
        self.configure_overlap(overlap_mode, overlap_lambda, overlap_threshold, overlap_temperature)
        self.configure_entropy(reconfig_entropy_coef)
        assert hidden_size == square_dim**2, (
            f"hidden_size (atom count) {hidden_size} must equal square_dim**2 "
            f"{square_dim**2} so each mask shares the FTN grid / visualization."
        )
        assert n_steps >= 1, "n_steps must be >= 1"
        io = str(io).lower()
        assert io in ("fixed", "ftn"), f"FTNLORA_IO must be 'fixed' or 'ftn', got {io!r}"

        self.layer_size = hidden_size          # atoms per bank == square_dim**2
        self.square_dim = square_dim
        self.classification = classification
        self.binary = binary
        self.io = io
        self.n_steps = n_steps
        self.rnn_step_rmsnorm = bool(rnn_step_rmsnorm)
        self.cell_layers = int(cell_layers)
        self.step_act = _ACTIVATIONS[str(step_activation).lower()]
        self.width = width

        self.k = k
        self.reconfig_steps = reconfig_steps
        self.reconfig_lr = reconfig_lr
        self.reconfig_optimizer = validate_reconfig_optimizer(reconfig_optimizer)
        self.reconfig_sample_size = reconfig_sample_size
        self.reconfig_algorithm = reconfig_algorithm
        self.reconfig_init_sd = reconfig_init_sd
        self.reconfig_noise_scale = reconfig_noise_scale
        # Match-stored reconfiguration option (see _search_simultaneous). Live
        # reference to the replay / noise-probe mask buffer, wired by
        # run_experiments; None disables regardless of match_stored.
        self.match_stored = match_stored
        self.match_stored_after = match_stored_after
        self.match_stored_topk = match_stored_topk
        self.match_tie_steps = bool(match_tie_steps)
        self._stored_masks_ref = None
        self.mask_lateral_steps = mask_lateral_steps
        self.mask_lateral_polyak = mask_lateral_polyak
        self.mask_lateral_noise_sd = mask_lateral_noise_sd

        # Number of masks: N recurrent-step masks, plus (io="ftn") a dedicated
        # input and output mask. Order in self.m:
        #   fixed: [m_0..m_{N-1}]        ftn: [m_in, m_0..m_{N-1}, m_out]
        self.n_masks = n_steps + (2 if io == "ftn" else 0)
        # Total maskable size reported to viz/replay: one square_dim**2 block per
        # mask, laid out side by side by mask_to_grid.
        self.hidden_size = self.n_masks * hidden_size

        # ---- Input encoder (x -> width) ----
        if io == "ftn":
            self.input_bank = LoRABank(
                input_size, width, hidden_size, rank, alpha, weight_scale,
                b0_scale, dropout_p, use_bias,
            )
            self.fixed_input = None
        else:
            enc = nn.Linear(input_size, width, bias=True)
            with torch.no_grad():
                enc.weight.normal_(0.0, weight_scale / math.sqrt(input_size))
                enc.bias.zero_()
            enc.requires_grad_(False)
            self.fixed_input = enc
            self.input_bank = None

        # ---- Recurrent cell (width -> width): one shared bank, a distinct mask
        # each step selects a distinct effective weight. cell_layers == 1 keeps a
        # single LoRABank (identical state_dict keys / behaviour as before);
        # cell_layers > 1 makes each step an n-layer deep LoRA MLP whose n layers
        # are all gated by that step's single mask. ----
        if self.cell_layers == 1:
            self.cell = LoRABank(
                width, width, hidden_size, rank, alpha, weight_scale,
                b0_scale, dropout_p, use_bias,
            )
        else:
            self.cell = DeepLoRABank(
                width, self.cell_layers, hidden_size, rank, alpha, weight_scale,
                b0_scale, dropout_p, use_bias, inner_activation=inner_activation,
            )

        # ---- Output readout (width -> output) ----
        if io == "ftn":
            self.output_bank = LoRABank(
                width, output_size, hidden_size, rank, alpha, weight_scale,
                b0_scale, dropout_p, use_bias,
            )
            self.fixed_output = None
        else:
            ro = nn.Linear(width, output_size, bias=False)
            with torch.no_grad():
                ro.weight.normal_(0.0, weight_scale / math.sqrt(width))
            ro.requires_grad_(False)
            self.fixed_output = ro
            self.output_bank = None

        # Shared learnable per-unit gain for the per-step RMSNorm on the fed-back
        # recurrent state (tied across steps, like the cell). Only allocated when
        # the norm is enabled.
        if self.rnn_step_rmsnorm:
            self.rms_gain = nn.Parameter(torch.ones(width))

        self.mask_lateral = _uniform_conv(mask_kernel_size, 1.0, torus)

        # Concatenated per-mask selections (single source of truth for viz/replay).
        self.register_buffer("m", torch.zeros(self.hidden_size))

        # Dispatch target used by run_experiments (ftn._reconfig).
        self._reconfig = (
            self.fast_reconfigure if reconfig_algorithm == "fast"
            else self.reconfigure
        )

    # ------------------------------------------------------------ mask packing

    def set_stored_masks_ref(self, masks_ref):
        """Wire in the live buffer of stored task masks (a deque, e.g.
        ReplayBuffer.masks / NoiseProbeBuffer.masks) used by the match-stored
        annealste option. run_experiments sets this; None disables it."""
        self._stored_masks_ref = masks_ref

    def _split(self, m: torch.Tensor) -> list:
        """Concatenated (n_masks*layer_size,) mask -> list of per-mask masks, in
        self.m order (input?, steps, output?)."""
        return list(m.reshape(self.n_masks, self.layer_size))

    # -------------------------------------------------- forward

    def _rmsnorm(self, h: torch.Tensor, detach: bool) -> torch.Tensor:
        """Shared RMSNorm applied to the fed-back state before a recurrent step.
        Detaches the gain in the mask search so only the masks carry gradient."""
        g = self.rms_gain.detach() if detach else self.rms_gain
        rms = h.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        return h * rms * g

    def _forward_masks(self, x, masks, detach: bool) -> torch.Tensor:
        """Run the encode -> N recurrent steps -> readout under `masks` (a list
        of per-mask selections in self.m order).

        detach=True detaches every trainable atom bank / gain so gradients reach
        only the masks (used by the mask search); detach=False is the trainable
        pass and applies dropout inside each bank.

        The number of recurrent steps run is inferred from len(masks) (minus the
        two dedicated I/O masks when io="ftn"), so this also serves the shorter
        unroll used by intermediate (single-operator) training: the shared cell
        and I/O banks are reused for 1..n_steps steps.
        """
        io_extra = 2 if self.io == "ftn" else 0
        k = len(masks) - io_extra
        idx = 0
        # Input encoder.
        if self.io == "ftn":
            h = self.input_bank(x, masks[idx], self.k, detach)
            idx += 1
        else:
            h = self.fixed_input(x)

        # Recurrent unroll: one shared cell, a distinct mask each step.
        for t in range(k):
            if t > 0 and self.rnn_step_rmsnorm:
                h = self._rmsnorm(h, detach)
            h = self.cell(h, masks[idx + t], self.k, detach)
            h = self.step_act(h)
        idx += k

        # Output readout.
        if self.io == "ftn":
            out = self.output_bank(h, masks[idx], self.k, detach)
        else:
            out = self.fixed_output(h)
        return out

    def forward(self, x, y=None, m=None):
        # Explicit mask (replay / eval): use the concatenated masks exactly as
        # given without re-deriving anything.
        if m is not None:
            return self._forward_masks(x, self._split(m), detach=False)
        if y is not None:
            self._reconfig(x, y)
        return self._forward_masks(x, self._split(self.m), detach=False)

    def loss(self, pred, y):
        return task_loss(pred, y, self.classification, self.binary)

    # ------------------------------------------------------ reconfiguration
    # Simultaneous mask search: minimize task loss on the support batch over
    # every mask's logits at once, with all weights detached so only the masks
    # move. Mirrors the recurrent StackFTN search, generalized to the
    # input/recurrent/output mask set.

    def _step_mask(self, logits, i, anneal: bool):
        """One mask's forward selection for search step i.

        anneal=True  -> hard kWTA(logits + decaying noise) with a sigmoid STE
                        gradient; the noise pulls new atoms into the active set.
        anneal=False -> plain soft sigmoid ("fast"); a single kWTA is applied at
                        the end of the search instead.
        """
        if not anneal:
            return torch.sigmoid(logits)
        sm = torch.sigmoid(logits)
        tau = logits.detach().std() * math.exp(
            -i / max(1.0, self.reconfig_steps / 4)
        )
        noise = torch.randn_like(logits) * tau * self.reconfig_noise_scale
        return kwta(logits + noise, self.k) + sm - sm.detach()

    @torch.enable_grad()
    def _search_simultaneous(self, x, y, anneal: bool, n_masks=None) -> list:
        """Jointly optimize the mask logits against the final loss.

        n_masks (default self.n_masks) is how many masks to search. Passing a
        smaller count searches a shorter unroll (fewer recurrent-step masks) —
        used by intermediate (single-operator) training on the weight-tied cell.
        Match-stored recall is only defined for the full mask set (it recalls
        whole stored task masks), so it is disabled when n_masks != self.n_masks.
        """
        nm = self.n_masks if n_masks is None else int(n_masks)
        reference = self._overlap_reference(nm * self.layer_size)
        n = min(self.reconfig_sample_size, x.shape[0])
        x, y = x[:n], y[:n]

        logits = [
            (torch.randn(self.layer_size) * self.reconfig_init_sd)
            .detach()
            .requires_grad_(True)
            for _ in range(nm)
        ]
        opt = make_reconfig_optimizer(logits, lr=self.reconfig_lr, name=self.reconfig_optimizer)

        # Match-stored option: after match_stored_after steps, snap the running
        # best masks onto the closest previous-task masks in the buffer, then let
        # the rest of the steps try to beat it.
        do_match = (
            self.match_stored
            and nm == self.n_masks
            and self._stored_masks_ref is not None
            and len(self._stored_masks_ref) > 0
        )
        stored = torch.stack(list(self._stored_masks_ref)) if do_match else None
        swapped = False

        def _cand_loss(cand: torch.Tensor) -> float:
            with torch.no_grad():
                return self._with_overlap(self._reconfig_task_loss(
                    self._forward_masks(x, self._split(cand), detach=True), y
                ), torch.cat(logits), reference).item()

        best, best_loss = None, math.inf
        for i in range(self.reconfig_steps):
            masks = [self._step_mask(lg, i, anneal) for lg in logits]
            loss = self._with_overlap(self._reconfig_task_loss(self._forward_masks(x, masks, detach=True), y), torch.cat(logits), reference)
            scored = [m.detach().clone() for m in masks] if self._regularized_search(reference) and anneal else None
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best = scored if scored is not None else [kwta(lg.detach(), self.k) for lg in logits]
                if do_match and not swapped and (i + 1) >= self.match_stored_after:
                    cur = (
                        best if best is not None
                        else [kwta(lg.detach(), self.k) for lg in logits]
                    )
                    best = self._recall_best(cur, stored, _cand_loss)
                    best_loss = _cand_loss(torch.cat(best))
                    swapped = True

        with torch.no_grad():
            if best is None:
                best = [kwta(lg.detach(), self.k) for lg in logits]
            if do_match and not swapped:
                best = self._recall_best(best, stored, _cand_loss)
            return [self._smooth(m) for m in best]

    def _recall_best(self, best_masks: list, stored: torch.Tensor, cand_loss) -> list:
        """Snap the running-best per-mask selections onto stored task masks.

        match_tie_steps=True: one Hopfield recall over the whole concatenated
        mask -- a complete stored solution (input + steps + output) is recovered
        as a unit.

        match_tie_steps=False: recall each mask slot independently. Because the
        recurrent cell is weight-tied across steps, the per-step masks all gate
        the SAME weights, so they pool into one shared bank and any step may
        recall any stored step's subspace -- per-step operations compose rather
        than each being a fixed N-step solution (exactly the recurrent StackFTN
        behavior). The dedicated input/output masks (io="ftn") gate their OWN
        atom banks, which are NOT tied to the cell or to each other; each is
        therefore recalled only from the matching stored slot, never pooled with
        the recurrent steps or the other I/O bank. In io="fixed" every slot is a
        recurrent step, so this reduces to pooling all slots -- identical to the
        recurrent StackFTN.
        """
        if self.match_tie_steps:
            m = best_stored(
                torch.cat(best_masks), stored, self.match_stored_topk, cand_loss
            )
            return self._split(m)
        S = self.layer_size

        def col(t):
            """Stored column bank for slot t: (N, S)."""
            return stored[:, t * S:(t + 1) * S].contiguous()

        # Recurrent-step slots (all slots for io="fixed"; the middle n_steps for
        # io="ftn") share the weight-tied cell, so they pool into one bank.
        step_start = 1 if self.io == "ftn" else 0
        step_slots = range(step_start, step_start + self.n_steps)
        step_pool = torch.cat([col(t) for t in step_slots], dim=0)

        new = [m.clone() for m in best_masks]
        for t in range(self.n_masks):
            # Recurrent steps recall from the pooled bank; the dedicated
            # input/output masks recall only from their own stored slot.
            bank = step_pool if t in step_slots else col(t)

            def slot_loss(cand, t=t):
                trial = list(new)
                trial[t] = cand
                return cand_loss(torch.cat(trial))
            new[t] = best_stored(new[t], bank, self.match_stored_topk, slot_loss)
        return new

    def _smooth(self, m: torch.Tensor) -> torch.Tensor:
        """Optional SmoothKWTA lateral dynamics on one mask; no-op when
        mask_lateral_steps == 0."""
        if self.mask_lateral_steps == 0 or self.mask_lateral_polyak >= 1.0:
            return m
        mf = m.float()
        for _ in range(self.mask_lateral_steps):
            g = mask_to_grid(mf, self.square_dim).unsqueeze(0).unsqueeze(0)
            m_lat = mask_to_vec(
                self.mask_lateral(g).squeeze(0).squeeze(0), self.square_dim
            )
            if self.mask_lateral_noise_sd > 0:
                m_lat = m_lat + torch.randn_like(m_lat) * self.mask_lateral_noise_sd
            mf = self.mask_lateral_polyak * mf + (1 - self.mask_lateral_polyak) * m_lat
        return kwta(mf, self.k)

    # ------------------------------------------------------ reconfig entry points

    def reconfigure(self, x, y) -> torch.Tensor:
        """Simultaneous annealed-STE mask search; writes self.m."""
        masks = self._search_simultaneous(x, y, anneal=True)
        self.m.copy_(torch.cat(masks))
        return self.m

    def fast_reconfigure(self, x, y) -> torch.Tensor:
        """Simultaneous fast (soft-sigmoid) mask search; writes self.m."""
        masks = self._search_simultaneous(x, y, anneal=False)
        self.m.copy_(torch.cat(masks))
        return self.m

    # ------------------------------------------------ short-unroll (intermediate)

    def forward_k(self, x, y=None, k_steps=None) -> torch.Tensor:
        """Run the recurrence for k_steps steps (default n_steps), returning the
        prediction. With y given, first cold-start reconfigure the masks (input
        mask if io="ftn", k_steps recurrent-step masks, output mask if io="ftn")
        via the simultaneous search; with y None, reuse the corresponding masks
        currently held in self.m.

        Intended for INTERMEDIATE training (k_steps < n_steps, e.g. k_steps=1 on
        a single-operator dataset): the shared recurrent atom bank and the I/O
        maps are reused for any step count, so a short unroll trains exactly the
        weights the full-length eval unroll uses. It does NOT persist the searched
        masks into self.m, so it never disturbs eval / viz / replay.
        """
        k = self.n_steps if k_steps is None else int(k_steps)
        if not (1 <= k <= self.n_steps):
            raise ValueError(f"k_steps must be in [1, n_steps={self.n_steps}], got {k}")
        io_extra = 2 if self.io == "ftn" else 0
        if y is not None:
            anneal = self.reconfig_algorithm != "fast"
            masks = self._search_simultaneous(x, y, anneal=anneal, n_masks=k + io_extra)
        else:
            full = self._split(self.m)   # [m_in?, m_0..m_{n-1}, m_out?]
            if self.io == "ftn":
                masks = [full[0]] + full[1:1 + k] + [full[-1]]
            else:
                masks = full[:k]
        return self._forward_masks(x, masks, detach=False)
