"""
stack_ftn.py - Stacked / recurrent-mask FTN: apply a mask on every layer of a
chain of FTN blocks (distinct weights) or of one weight-tied FTN cell unrolled
over steps.

Selected with ARCHITECTURE in {"stack", "stack_residual", "rnn", "rnn_residual"}
in config.py. Presents the same interface as ftn.FTN (`ftn(x, y)` reconfigures +
predicts, `ftn(x)` predicts, `ftn(x, m=...)` predicts under an explicit mask,
`ftn._reconfig(x, y)`, `ftn.loss`, `ftn.m`, `ftn.hidden_size`), so
run_experiments.py drives it unchanged.

How it differs from the base ParallelMLP FTN
============================================
The base FTN has ONE ParallelMLP backbone: it computes every unit's activation
`z` in a single pass and a single mask *selects a subspace of z* at the very end
(`linear(z * m)`).

A stacked FTN chains N of those blocks. Each block is a ParallelMLP (the same
per-unit backbone), its masked population output feeding as the input to the
next block:

    x -> z0*m0 -> z1*m1 -> z2*m2 -> ... -> z_{L-1}*m_{L-1} -> readout -> output

so there is one mask *per layer*, each a kWTA selection of k of the layer's
square_dim**2 units. The masks are held concatenated in `self.m`
(size n_stack * square_dim**2); `self.hidden_size` reports that total so the
mask GIF / RGB rasters lay the per-layer grids out side by side (one block per
layer), reusing viz.MaskAnimator unchanged.

Stacked vs recurrent (recurrent flag)
-------------------------------------
recurrent=False ("stack"/"stack_residual"): the N blocks have *independent*
weights (an nn.ModuleList of N ParallelMLPs), and block 0 consumes the raw
input directly.

recurrent=True ("rnn"/"rnn_residual"): the N "blocks" are the SAME ParallelMLP
cell unrolled over N steps — it is one FTN that sequentially activates a
different mask at each recurrent step. Because the shared cell maps the hidden
space to itself, a small linear encoder lifts the raw input into that space as
the initial state, and a single shared linear readout decodes it:

    h = embed(x); for t: z_t = cell(h); h = z_t * m_t; ... ; readout(h)

An optional shared RMSNorm (rnn_step_rmsnorm) normalizes the fed-back state h
before every step but the first (embed(x), the data, is left untouched),
keeping the weight-tied cell's state from drifting/vanishing across steps.

The per-step masks, self.m layout, viz, and the whole mask search below are
identical to the stacked case; only the block weights are tied and the input is
encoded.

Two output topologies (residual flag)
-------------------------------------
residual=False ("stack"/"rnn"): the readout reads only the last masked layer,
                  `readout(z_{L-1} * m_{L-1})`. Information must pass
                  sequentially through every masked bottleneck.
residual=True  ("stack_residual"/"rnn_residual"): a linear skip connection runs
                  from *every* masked layer straight to the output, and the
                  output is their sum, `sum_l readout_l(z_l * m_l)`. This is the
                  only difference from the non-residual variant — an isolated
                  test of whether the skips improve reconfiguration. (The
                  stacked case uses one readout per layer; the recurrent case
                  shares a single readout across steps, so its residual output
                  is `readout(sum_t z_t * m_t)`.)

Reconfiguration
---------------
Default (all topologies) is *simultaneous*: run the full stack/unroll with all
masks applied and optimize the logits for every layer jointly with a single
optimizer against the final task loss (annealed-STE kWTA per layer, or the
soft-sigmoid "fast" variant). Weights are detached, so only the mask logits
move.

The residual topologies additionally offer a *sequential* (greedy) option
(sequential=True): reconfigure m0 first, then m1 given m0, then m2 given m1,m0,
and so on. Each stage optimizes one layer's mask with all earlier masks frozen,
scoring against the prefix output sum_{j<=l} readout_j(z_j * m_j) — layers past
the current one are not yet computed. This is well-defined only because each
residual layer has a direct path to the output, which is why it is offered for
the residual topologies alone.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the backbone + mask primitives so the stacked masks live in the same
# grid / viz world as the base FTN and FTN-LoRA.
from ftn import (
    ParallelMLP,
    _uniform_conv,
    best_stored,
    kwta,
    mask_to_grid,
    mask_to_vec,
    task_loss,
)


class StackFTN(nn.Module):
    """Chained (or weight-tied recurrent) ParallelMLP blocks with a per-layer
    kWTA mask.

    Interface mirrors ftn.FTN: forward(x) predicts under the current masks,
    forward(x, y) reconfigures first, forward(x, m=...) uses an explicit
    (concatenated) mask.

    Args:
        input_size, output_size:  Input / output dimensions.
        hidden_size:              Units *per stacked layer*; must equal
                                  square_dim**2 so each layer's mask is a
                                  square_dim x square_dim grid.
        square_dim:               Side length of each per-layer mask grid.
        classification:           Cross-entropy vs MSE (reconfig + self.loss).
        n_stack:                  Number of chained ParallelMLP+mask blocks
                                  (recurrent steps when recurrent=True).
        recurrent:                False = independent per-layer block weights
                                  ("stack"/"stack_residual"). True = one shared
                                  ParallelMLP cell unrolled over n_stack steps
                                  with a per-step mask, plus a linear input
                                  encoder and a single shared readout
                                  ("rnn"/"rnn_residual").
        residual:                 False (readout reads the last masked layer
                                  only) vs True (a skip from every masked layer
                                  to the output, summed).
        sequential:               Residual-only greedy reconfiguration (m0, then
                                  m1|m0, ...). Ignored/forbidden when residual is
                                  False.
        detach_mask_deep_path:    Residual-only. When True, each masked population
                                  is fed to the NEXT block with its mask detached,
                                  so a mask's reconfiguration gradient reaches the
                                  output only through that layer's own linear
                                  readout skip, never through the deeper nonlinear
                                  blocks it also gates. False (default) = exactly
                                  as before (gradient flows through both paths).
                                  A no-op for non-residual topologies.
        rnn_step_rmsnorm:         Recurrent-only: apply a shared learnable RMSNorm
                                  to the input of every recurrent step except the
                                  first (i.e. normalize the fed-back state
                                  z_{t-1}*m_{t-1}, but leave the initial embed(x)
                                  untouched). Ignored when recurrent is False.
        layers, inner_size:       Per-block ParallelMLP depth / inner width
                                  (the base backbone knobs, reused per layer /
                                  by the shared cell).
        dropout_p:                Dropout on each block's population (forward
                                  only; the mask search sees no dropout).
        k, reconfig_steps, reconfig_lr, reconfig_sample_size,
        reconfig_algorithm, reconfig_init_sd:
                                  Mask-search budget / algorithm, shared by all
                                  layers (same knobs as the base FTN's mask).
        match_stored, match_stored_after, match_stored_topk:
                                  Match-stored recall knobs (simultaneous search
                                  only): after match_stored_after steps, snap the
                                  running-best masks onto stored task masks; topk
                                  is the Hopfield recall breadth.
        match_tie_steps:          Recall granularity for match-stored. True
                                  (default): recall the full concatenated
                                  all-steps mask as one whole. False: split
                                  stored masks into per-step square_dim**2 masks,
                                  pool them across steps, and recall each step
                                  independently from the shared pool (mix-and-
                                  match per-step functions). See _recall_best.
        mask_kernel_size, mask_lateral_steps, mask_lateral_polyak,
        mask_lateral_noise_sd, torus:
                                  SmoothKWTA lateral dynamics applied to each
                                  layer's final mask; mask_lateral_steps=0 =
                                  plain kWTA.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,          # units per layer == square_dim**2
        square_dim: int,
        classification: bool = False,
        binary: bool = False,
        # Topology
        n_stack: int = 4,
        recurrent: bool = False,
        residual: bool = False,
        sequential: bool = False,
        rnn_step_rmsnorm: bool = False,
        rnn_recurrent_mask: bool = False,
        detach_mask_deep_path: bool = False,
        concat_readout: bool = False,
        # Per-block backbone (reused ParallelMLP knobs)
        layers: int = 8,
        inner_size: int = 8,
        dropout_p: float = 0.0,
        # Reconfiguration (mask search), shared across layers
        k: int = 128,
        reconfig_steps: int = 30,
        reconfig_lr: float = 0.3,
        reconfig_sample_size: int = 2048,
        reconfig_algorithm: str = "annealste",   # "annealste" or "fast"
        reconfig_init_sd: float = 1e-5,
        reconfig_noise_scale: float = 1.0,
        # Match-stored reconfiguration (annealste simultaneous search only)
        match_stored: bool = False,
        match_stored_after: int = 0,
        match_stored_topk: int = 1,
        match_tie_steps: bool = True,
        mask_kernel_size: int = 3,
        mask_lateral_steps: int = 0,
        mask_lateral_polyak: float = 0.0,
        mask_lateral_noise_sd: float = 0.0,
        torus: bool = False,
    ):
        super().__init__()
        assert hidden_size == square_dim**2, (
            f"hidden_size (units per layer) {hidden_size} must equal square_dim**2 "
            f"{square_dim**2} so each layer's mask shares the FTN grid / viz."
        )
        assert n_stack >= 1, "n_stack must be >= 1"
        if sequential and not residual:
            raise ValueError(
                "sequential reconfiguration requires residual=True "
                '(ARCHITECTURE="stack_residual"/"rnn_residual"): a non-residual '
                "stack has no direct per-layer output path to score a layer "
                "prefix against."
            )
        if rnn_recurrent_mask and sequential:
            raise ValueError(
                "rnn_recurrent_mask is incompatible with sequential: it searches "
                "only the first mask and derives the rest from the learned "
                "recurrent transition, so there is no per-layer greedy stage."
            )

        self.layer_size = hidden_size          # units in one layer
        self.n_stack = n_stack
        self.square_dim = square_dim
        self.classification = classification
        self.recurrent = recurrent
        self.binary = binary
        self.residual = residual
        self.sequential = sequential
        # Isolate each mask's reconfiguration gradient to its own linear readout
        # skip (residual only; ignored otherwise — non-residual has no per-layer
        # linear path, so detaching would starve every mask but the last).
        self.detach_mask_deep_path = bool(detach_mask_deep_path) and residual
        # RMSNorm on the fed-back state before each recurrent step after the
        # first; recurrent-only (the non-recurrent stack has no shared cell to
        # stabilize, and its blocks take their own inputs directly).
        self.rnn_step_rmsnorm = bool(rnn_step_rmsnorm) and recurrent
        # Recurrent-mask reconfiguration (recurrent-only). When True, the mask
        # search finds ONLY the first-step mask m_0 (optionally recalled from the
        # stored buffer); the remaining steps' masks are produced by a *learned*
        # recurrent transition m_{t+1} = kWTA(W m_t + b) (straight-through), whose
        # weights are trained jointly with the cell/readout — not searched. So
        # reconfiguration selects an entry point into a learned, self-propagating
        # sequence of per-step operation selections. See _propagate_mask /
        # _search_first_mask. Requires recurrent (a weight-tied cell to reuse).
        self.rnn_recurrent_mask = bool(rnn_recurrent_mask) and recurrent
        # Output topology (residual only). Accumulative (default): one shared
        # readout summed across steps, out = W * sum_l s_l. Concatenated
        # (concat_readout, like HML): a distinct readout per step, summed,
        # out = sum_l W_l s_l == one readout over [s_0, ..., s_{L-1}]. The
        # stacked-residual case already uses per-step readouts, so this flag only
        # changes the recurrent-residual (rnn_residual) case.
        self.concat_readout = bool(concat_readout) and residual
        self._per_step_readouts = residual and (not recurrent or self.concat_readout)

        self.k = k
        self.reconfig_steps = reconfig_steps
        self.reconfig_lr = reconfig_lr
        self.reconfig_sample_size = reconfig_sample_size
        self.reconfig_algorithm = reconfig_algorithm
        self.reconfig_init_sd = reconfig_init_sd
        self.reconfig_noise_scale = reconfig_noise_scale
        # Match-stored reconfiguration option (see _search_simultaneous). Live
        # reference to the replay / noise-probe mask buffer, wired by
        # run_experiments; None disables regardless of match_stored.
        self.match_stored = match_stored
        self.match_stored_after = match_stored_after
        # Hopfield recall breadth: consider the top-k highest-overlap stored
        # masks and keep the best-performing one (by loss). 1 = closest match.
        self.match_stored_topk = match_stored_topk
        # Recall granularity for match-stored. True (default): the stored masks
        # span all n_stack steps and are recalled as one concatenated whole — a
        # complete deep solution is recovered or nothing is. False: the stored
        # masks are split into per-step square_dim**2 masks and pooled across all
        # steps, and each step runs its own recall against the shared pool. Since
        # the cell is weight-tied and reused across steps, a subspace stored as
        # (say) a step-3 mask can then be recalled as this run's step 0/1/2 —
        # masks become stable, reusable per-step functions that compose rather
        # than each being one fixed N-step composition. See _recall_best.
        self.match_tie_steps = bool(match_tie_steps)
        self._stored_masks_ref = None
        self.mask_lateral_steps = mask_lateral_steps
        self.mask_lateral_polyak = mask_lateral_polyak
        self.mask_lateral_noise_sd = mask_lateral_noise_sd

        # Total maskable size reported to viz/replay: one square_dim**2 block per
        # layer/step, laid out side by side by mask_to_grid.
        self.hidden_size = n_stack * hidden_size

        if recurrent:
            # One shared ParallelMLP cell (hidden_size -> hidden_size units)
            # unrolled over n_stack steps, plus a linear encoder that lifts the
            # raw input into the recurrent hidden space as the initial state.
            self.embed = nn.Linear(input_size, hidden_size)
            self.cell = ParallelMLP(hidden_size, 1, hidden_size, layers, inner_size)
            # Shared learnable per-unit gain for the per-step RMSNorm (tied across
            # steps, like the cell). Only allocated when the norm is enabled.
            if self.rnn_step_rmsnorm:
                self.rms_gain = nn.Parameter(torch.ones(hidden_size))
            # Learned recurrent mask transition m_{t+1} = kWTA(W m_t + b), tied
            # across steps like the cell. Initialized near identity so it starts
            # by reusing the same subspace each step and learns to deviate; its
            # weights train with the rest of the network, not the mask search.
            if self.rnn_recurrent_mask:
                self.mask_transition = nn.Linear(hidden_size, hidden_size)
                with torch.no_grad():
                    self.mask_transition.weight.copy_(torch.eye(hidden_size))
                    self.mask_transition.bias.zero_()
        else:
            # One ParallelMLP block per layer. Block 0 takes the raw input;
            # every later block takes the previous layer's masked population.
            self.blocks = nn.ModuleList(
                ParallelMLP(
                    input_size if l == 0 else hidden_size,
                    1,
                    hidden_size,
                    layers,
                    inner_size,
                )
                for l in range(n_stack)
            )

        # Readout(s). Non-residual: a single readout on the last masked layer.
        # Per-step distinct readouts (summed) for the stacked-residual case and
        # for concat_readout recurrent-residual; otherwise one shared readout
        # (accumulative: non-residual last-layer, or recurrent-residual summing
        # one shared readout across steps).
        if self._per_step_readouts:
            self.readouts = nn.ModuleList(
                nn.Linear(hidden_size, output_size, bias=False)
                for _ in range(n_stack)
            )
        else:
            self.readout = nn.Linear(hidden_size, output_size, bias=False)

        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.mask_lateral = _uniform_conv(mask_kernel_size, 1.0, torus)

        # Concatenated per-layer masks (single source of truth for viz/replay).
        self.register_buffer("m", torch.zeros(self.hidden_size))

        # Dispatch target used by run_experiments (ftn._reconfig).
        if self.rnn_recurrent_mask:
            self._reconfig = self.recurrent_mask_reconfigure
        elif sequential:
            self._reconfig = self.sequential_reconfigure
        elif reconfig_algorithm == "fast":
            self._reconfig = self.fast_reconfigure
        else:
            self._reconfig = self.reconfigure

    # ------------------------------------------------------------ mask packing

    def set_stored_masks_ref(self, masks_ref):
        """Wire in the live buffer of stored task masks (a deque, e.g.
        ReplayBuffer.masks / NoiseProbeBuffer.masks) used by the match-stored
        annealste option. run_experiments sets this; None disables it."""
        self._stored_masks_ref = masks_ref

    def _split(self, m: torch.Tensor) -> list:
        """Concatenated (n_stack*layer_size,) mask -> list of per-layer masks."""
        return list(m.reshape(self.n_stack, self.layer_size))

    # -------------------------------------------------- encode / block / readout

    def _encode(self, x: torch.Tensor, detach: bool) -> torch.Tensor:
        """Initial hidden state fed to the first block. Recurrent: the linear
        encoder embed(x); stacked: the raw input passed straight to block 0."""
        if not self.recurrent:
            return x
        w = self.embed.weight.detach() if detach else self.embed.weight
        b = self.embed.bias.detach() if detach else self.embed.bias
        return F.linear(x, w, b)

    def _block(self, l: int, h: torch.Tensor, detach: bool) -> torch.Tensor:
        """Pre-mask population z at layer/step l. Recurrent shares one cell."""
        blk = self.cell if self.recurrent else self.blocks[l]
        return blk(h, detach=detach)

    def _rmsnorm(self, h: torch.Tensor, detach: bool) -> torch.Tensor:
        """Shared RMSNorm applied to the fed-back state before a recurrent step.
        Detaches the gain in the mask search so only the masks carry gradient."""
        g = self.rms_gain.detach() if detach else self.rms_gain
        rms = h.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
        return h * rms * g

    def _readout(self, l: int, h: torch.Tensor, detach: bool) -> torch.Tensor:
        # Per-step distinct readout when using the concatenated output topology
        # (stacked-residual, or concat_readout recurrent-residual); otherwise the
        # single shared readout (accumulative / non-residual last layer).
        ro = self.readouts[l] if self._per_step_readouts else self.readout
        w = ro.weight.detach() if detach else ro.weight
        return F.linear(h, w)

    # ---------------------------------------------------------------- forward

    def _forward_stack(self, x, masks, detach: bool) -> torch.Tensor:
        """Run the full chain / unroll under `masks` (a list of per-layer masks).

        detach=True detaches every block/readout weight so gradients reach only
        the masks (used by the mask search); detach=False is the trainable pass
        and also applies dropout to each population.
        """
        h = self._encode(x, detach)
        out = 0.0
        for l in range(self.n_stack):
            # Normalize the fed-back state before every step but the first (the
            # first input is embed(x) — the data — which is left untouched).
            if l > 0 and self.rnn_step_rmsnorm:
                h = self._rmsnorm(h, detach)
            z = self._block(l, h, detach=detach)
            if not detach:
                z = self.dropout(z)
            h = z * masks[l].reshape(1, -1)
            if self.residual:
                out = out + self._readout(l, h, detach)
                # Optionally block this mask's gradient on the deep path, so
                # reconfiguration gradient reaches m_l only via the readout above
                # (the linear skip). Forward values are unchanged — detach only
                # cuts the gradient; a no-op when masks carry no grad (training).
                if self.detach_mask_deep_path:
                    h = z * masks[l].detach().reshape(1, -1)
        if not self.residual:
            out = self._readout(self.n_stack - 1, h, detach)  # last masked layer
        return out

    def forward(self, x, y=None, m=None):
        # Explicit mask (replay / eval): use the concatenated masks exactly as
        # given, all steps included, without re-deriving anything.
        if m is not None:
            return self._forward_stack(x, self._split(m), detach=False)
        if y is not None:
            self._reconfig(x, y)
        if self.rnn_recurrent_mask:
            # Re-propagate m_0 (the searched first mask stored in self.m) through
            # the CURRENT learned transition so its weights receive training
            # gradient; the reconfigured self.m holds the same m_0 in its first
            # block, and the transition is deterministic, so this reproduces the
            # stored per-step masks while keeping them differentiable in W.
            masks = self._propagate_mask(self.m[: self.layer_size], detach=False)
        else:
            masks = self._split(self.m)
        return self._forward_stack(x, masks, detach=False)

    def loss(self, pred, y):
        return task_loss(pred, y, self.classification, self.binary)

    # ------------------------------------------------------ reconfiguration
    # The mask search: minimize task loss on the support batch over the per-layer
    # logits, with the network weights detached so only the masks move.

    def _step_mask(self, logits, i, anneal: bool):
        """One layer's forward mask for search step i.

        anneal=True  -> hard kWTA(logits + decaying noise) with a sigmoid STE
                        gradient; the noise pulls new units into the active set.
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

    def _propagate_mask(self, m0: torch.Tensor, detach: bool) -> list:
        """Recurrent-mask option: derive all n_stack per-step masks from the
        first mask via the learned transition m_{t+1} = kWTA(W m_t + b), with a
        sigmoid straight-through estimator so gradient flows to W (detach=False,
        the trainable pass) or back to m_0 (detach=True, the first-mask search).

        detach=True detaches the transition weights so only m_0 carries gradient
        (used inside the search); detach=False leaves them trainable so the
        transition learns alongside the cell/readout. Under torch.no_grad the STE
        term vanishes and each returned mask is a hard k-hot vector.
        """
        w = self.mask_transition.weight
        b = self.mask_transition.bias
        if detach:
            w, b = w.detach(), b.detach()
        masks = [m0]
        m = m0
        for _ in range(self.n_stack - 1):
            logits = F.linear(m, w, b)
            sm = torch.sigmoid(logits)
            m = kwta(logits, self.k) + sm - sm.detach()
            masks.append(m)
        return masks

    @torch.enable_grad()
    def _search_simultaneous(self, x, y, anneal: bool) -> list:
        """Jointly optimize all n_stack layers' logits against the final loss."""
        n = min(self.reconfig_sample_size, x.shape[0])
        x, y = x[:n], y[:n]

        logits = [
            (torch.randn(self.layer_size) * self.reconfig_init_sd)
            .detach()
            .requires_grad_(True)
            for _ in range(self.n_stack)
        ]
        opt = torch.optim.Adam(logits, lr=self.reconfig_lr)

        # Match-stored option (annealste only): after match_stored_after steps,
        # snap the running best per-layer masks to the closest previous-task mask
        # in the buffer, then let the rest of the steps try to beat it.
        do_match = (
            # anneal
            self.match_stored
            and self._stored_masks_ref is not None
            and len(self._stored_masks_ref) > 0
        )
        stored = torch.stack(list(self._stored_masks_ref)) if do_match else None
        swapped = False

        def _cand_loss(cand: torch.Tensor) -> float:
            with torch.no_grad():
                return self.loss(
                    self._forward_stack(x, self._split(cand), detach=True), y
                ).item()

        best, best_loss = None, math.inf
        for i in range(self.reconfig_steps):
            masks = [self._step_mask(lg, i, anneal) for lg in logits]
            loss = self.loss(self._forward_stack(x, masks, detach=True), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best = [kwta(lg.detach(), self.k) for lg in logits]
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
            # match_stored_after >= steps never triggered above: anchor now.
            if do_match and not swapped:
                best = self._recall_best(best, stored, _cand_loss)
            return [self._smooth(m) for m in best]

    def _recall_best(self, best_masks: list, stored: torch.Tensor, cand_loss) -> list:
        """Snap the running-best per-step masks onto stored task masks.

        `best_masks` is the current list of n_stack per-step masks; `stored` is
        the (N, n_stack*layer_size) stack of concatenated stored task masks;
        `cand_loss(full_mask)` scores a concatenated candidate on the batch.

        match_tie_steps=True: one Hopfield recall over the whole concatenated
        mask — stored masks span every step, so a complete deep solution is
        recovered as a unit (the original behaviour).

        match_tie_steps=False: the stored masks are split into per-step
        square_dim**2 masks and pooled across steps into one shared bank; each
        step then runs its own recall against that bank, keeping a separate best
        per step. Because the recurrent cell is weight-tied, a subspace stored as
        one task's step-t mask can be recalled as any step here, letting masks
        compose step by step rather than each being a fixed N-step solution.
        Steps are recalled in order, each scored against the masks already chosen
        for earlier steps and the running best for later ones.
        """
        if self.match_tie_steps:
            m = best_stored(
                torch.cat(best_masks), stored, self.match_stored_topk, cand_loss
            )
            return self._split(m)
        # Pool every stored mask's per-step blocks into one bank of layer-sized
        # masks (row i,step t -> row i*n_stack + t), then recall each step from it.
        pool = stored.reshape(-1, self.layer_size)
        new = [m.clone() for m in best_masks]
        for t in range(self.n_stack):
            def step_loss(cand, t=t):
                trial = list(new)
                trial[t] = cand
                return cand_loss(torch.cat(trial))
            new[t] = best_stored(new[t], pool, self.match_stored_topk, step_loss)
        return new

    def _recall_first(self, ref: torch.Tensor, stored: torch.Tensor, cand_loss) -> torch.Tensor:
        """Recurrent-mask option: recall the first-step mask m_0 from the stored
        buffer. `ref` is a layer_size mask; `stored` is (N, n_stack*layer_size);
        `cand_loss(m0)` scores a candidate first mask (propagated through the
        transition and run over the batch).

        Honors match_tie_steps: True recalls m_0 from each stored task's own
        first-step slice (re-enter a stored solution at its start); False pools
        every stored step into one bank so m_0 can re-enter at any stored step.
        """
        if self.match_tie_steps:
            bank = stored[:, : self.layer_size].contiguous()
        else:
            bank = stored.reshape(-1, self.layer_size)
        return best_stored(ref, bank, self.match_stored_topk, cand_loss)

    @torch.enable_grad()
    def _search_first_mask(self, x, y, anneal: bool) -> list:
        """Recurrent-mask search: optimize ONLY the first-step logits against the
        final loss, deriving m_1..m_{n-1} from the (detached) learned transition
        each step. Returns the full per-step mask list (m_0 propagated forward).

        The transition weights are detached here so the search moves only m_0;
        they are trained in the normal forward/backward pass. The match-stored
        recall, if enabled, snaps m_0 onto a stored first mask (see _recall_first).
        """
        n = min(self.reconfig_sample_size, x.shape[0])
        x, y = x[:n], y[:n]

        logits = (
            (torch.randn(self.layer_size) * self.reconfig_init_sd)
            .detach()
            .requires_grad_(True)
        )
        opt = torch.optim.Adam([logits], lr=self.reconfig_lr)

        do_match = (
            self.match_stored
            and self._stored_masks_ref is not None
            and len(self._stored_masks_ref) > 0
        )
        stored = torch.stack(list(self._stored_masks_ref)) if do_match else None
        swapped = False

        def _cand_loss(m0: torch.Tensor) -> float:
            with torch.no_grad():
                masks = self._propagate_mask(m0, detach=True)
                return self.loss(self._forward_stack(x, masks, detach=True), y).item()

        best, best_loss = None, math.inf
        for i in range(self.reconfig_steps):
            m0 = self._step_mask(logits, i, anneal)
            masks = self._propagate_mask(m0, detach=True)
            loss = self.loss(self._forward_stack(x, masks, detach=True), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best = kwta(logits.detach(), self.k)
                if do_match and not swapped and (i + 1) >= self.match_stored_after:
                    ref = best if best is not None else kwta(logits.detach(), self.k)
                    best = self._recall_first(ref, stored, _cand_loss)
                    best_loss = _cand_loss(best)
                    swapped = True

        with torch.no_grad():
            if best is None:
                best = kwta(logits.detach(), self.k)
            if do_match and not swapped:
                best = self._recall_first(best, stored, _cand_loss)
            return self._propagate_mask(self._smooth(best), detach=True)

    @torch.enable_grad()
    def _search_sequential(self, x, y, anneal: bool) -> list:
        """Greedy per-layer search (residual only): m0, then m1|m0, then ....

        Each stage optimizes one layer's logits with all earlier masks frozen,
        scoring against the prefix output sum_{j<=l} readout_j(z_j * m_j).
        """
        n = min(self.reconfig_sample_size, x.shape[0])
        x, y = x[:n], y[:n]

        chosen = []  # frozen hard masks for layers already reconfigured
        for _ in range(self.n_stack):
            logits = (
                (torch.randn(self.layer_size) * self.reconfig_init_sd)
                .detach()
                .requires_grad_(True)
            )
            opt = torch.optim.Adam([logits], lr=self.reconfig_lr)

            best, best_loss = None, math.inf
            for i in range(self.reconfig_steps):
                mask_l = self._step_mask(logits, i, anneal)
                loss = self.loss(self._forward_prefix(x, chosen, mask_l), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best = kwta(logits.detach(), self.k)

            with torch.no_grad():
                if best is None:
                    best = kwta(logits.detach(), self.k)
                chosen.append(self._smooth(best))
        return chosen

    def _forward_prefix(self, x, chosen, mask_l) -> torch.Tensor:
        """Residual prefix output through the first len(chosen)+1 layers.

        Layers < len(chosen) use their frozen `chosen` masks; the current layer
        uses `mask_l` (the only one carrying gradient). Weights are detached.
        """
        L = len(chosen)
        h = self._encode(x, detach=True)
        out = 0.0
        for l in range(L + 1):
            if l > 0 and self.rnn_step_rmsnorm:
                h = self._rmsnorm(h, detach=True)
            z = self._block(l, h, detach=True)
            m = (chosen[l] if l < L else mask_l).reshape(1, -1)
            h = z * m
            out = out + self._readout(l, h, detach=True)
        return out

    def _smooth(self, m: torch.Tensor) -> torch.Tensor:
        """Optional SmoothKWTA lateral dynamics on one layer's mask; no-op when
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

    def sequential_reconfigure(self, x, y) -> torch.Tensor:
        """Greedy per-layer mask search (residual only); writes self.m. Uses the
        annealed-STE search unless reconfig_algorithm == 'fast'."""
        anneal = self.reconfig_algorithm != "fast"
        masks = self._search_sequential(x, y, anneal=anneal)
        self.m.copy_(torch.cat(masks))
        return self.m

    def recurrent_mask_reconfigure(self, x, y) -> torch.Tensor:
        """Recurrent-mask search (recurrent-only): search only the first mask and
        propagate the rest through the learned transition; writes self.m. Uses
        the annealed-STE search unless reconfig_algorithm == 'fast'."""
        anneal = self.reconfig_algorithm != "fast"
        masks = self._search_first_mask(x, y, anneal=anneal)
        self.m.copy_(torch.cat(masks))
        return self.m

