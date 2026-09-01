"""
hml_ftn.py - Higher-order MultiLinear FTN ("hml"): a recurrent FTN whose
recurrent population replaces feed-forward depth with per-neuron low-rank
multilinear interactions.

Selected with ARCHITECTURE == "hml" in config.py. Presents the same interface as
ftn.FTN / stack_ftn.StackFTN (`ftn(x, y)` reconfigures + predicts, `ftn(x)`
predicts, `ftn(x, m=...)` predicts under an explicit mask, `ftn._reconfig(x, y)`,
`ftn.loss`, `ftn.m`, `ftn.hidden_size`, `ftn.set_stored_masks_ref`), so
run_experiments.py drives it unchanged.

Relationship to the recurrent-residual StackFTN
================================================
Structurally this is the "rnn_residual" model (a weight-tied cell unrolled over
steps, a mask per step, a linear skip from every masked step summed to the
output), with three substitutions:

  1. Input population instead of a bare linear encoder. The initial state is a
     distinct population of square_dim**2 neurons, z0 = tanh(W x); each neuron
     reads the whole input and produces one bounded activation.

  2. Low-rank multilinear interactions instead of feed-forward depth. The tied
     recurrent population is an HMLCell: each of the square_dim**2 neurons
     computes, from the whole masked population s_l, a first-order (linear) term
     plus rank-R multilinear interactions up to max_order, then tanh. Every
     projection / mix vector is private to a neuron - nothing is shared across
     neurons, and there is no shared "cell" matrix beyond each neuron's own row.

  3. Concatenated outputs over recurrence levels. A distinct linear decoder reads
     every masked state s_0, ..., s_depth and their outputs are summed (plus a
     bias) - algebraically one readout over the concatenation
     [s_0, ..., s_depth]. (rnn_residual instead shares one readout over the
     summed states.)

Forward pass
------------
    z0 = tanh(input_proj(x))          # input population (distinct weights)
    s0 = z0 * m0
    for l in 0 .. depth-1:            # tied recurrent population (HMLCell)
        z_{l+1} = cell(s_l)           # tanh(linear + multilinear interactions)
        s_{l+1} = z_{l+1} * m_{l+1}
    y = output_bias + sum_l readout_l(s_l)

There are depth+1 masks (one per state s_0 .. s_depth), each a kWTA-k selection
of the square_dim**2 units of that state. The masks are held concatenated in
self.m (size (depth+1) * square_dim**2); self.hidden_size reports that total so
the mask GIF / RGB rasters lay the per-level grids out side by side, reusing
viz.MaskAnimator unchanged - exactly as StackFTN does for its per-layer masks.

Reconfiguration
---------------
The mask search is the same simultaneous scheme as StackFTN: run the full unroll
with all depth+1 masks applied and jointly optimize their logits against the
final task loss (annealed-STE kWTA per level, or the soft-sigmoid "fast"
variant), with every weight (input_proj, cell, readouts, bias) detached so only
the mask logits move. The optional match-stored (Hopfield) snap onto a previous
task's mask carries over unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the mask primitives so the HML masks live in the same grid / viz world
# as the base FTN, FTN-LoRA and the stacked/recurrent FTN.
from ftn import _uniform_conv, best_stored, kwta, mask_to_grid, mask_to_vec, task_loss


def _linear(layer: nn.Linear, x: torch.Tensor, detach: bool) -> torch.Tensor:
    """F.linear through an nn.Linear, optionally detaching its parameters so a
    caller (the mask search) gets gradients on the input/masks only, never on
    these weights. Handles bias=False layers (layer.bias is None)."""
    w = layer.weight
    b = layer.bias
    if detach:
        w = w.detach()
        b = None if b is None else b.detach()
    return F.linear(x, w, b)


class HMLCell(nn.Module):
    """One recurrent HML population of H = hidden_dim neurons.

    Each neuron computes, from the whole (masked) population s (dim H):

        linear term            w_n . s
        + rank-R multilinear   sum_{p=2..P} sum_{r=1..R} mix_{n,p,r}
          interactions                       * prod_{q=1..p} (U_{n,p,q,r} . s)
        -> tanh

    Every weight is private to a neuron. The first-order path is one nn.Linear
    (row n is neuron n's linear weights); each factor projection is an
    H -> (H*R) nn.Linear so output neuron n owns R private projection vectors,
    and rank_mix maps the R rank components back to one scalar per neuron.
    """

    def __init__(self, hidden_dim: int, rank: int = 4, max_order: int = 2):
        super().__init__()
        assert max_order >= 1, "max_order must be >= 1"
        H, R, P = hidden_dim, rank, max_order
        self.H, self.R, self.P = H, R, P

        # First-order path (per-neuron linear term over the whole population).
        self.linear = nn.Linear(H, H)

        # Independent factor projections per interaction order/factor:
        # factors[p-2][q] is an H -> (H*R) projection; each output neuron owns R
        # private projection vectors. Only allocated for orders >= 2.
        self.factors = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(H, H * R, bias=False) for _ in range(p)]
                )
                for p in range(2, P + 1)
            ]
        )

        # Mix the R rank components back to one scalar per neuron, per order.
        self.rank_mix = nn.ParameterList(
            [nn.Parameter(torch.randn(H, R) * 0.01) for _ in range(2, P + 1)]
        )

    def forward(self, s: torch.Tensor, detach: bool = False) -> torch.Tensor:
        B = s.shape[0]

        h = _linear(self.linear, s, detach)

        for order_idx, _order in enumerate(range(2, self.P + 1)):
            # [B, H, R] for every factor.
            projections = [
                _linear(layer, s, detach).view(B, self.H, self.R)
                for layer in self.factors[order_idx]
            ]

            # Elementwise product gives the rank-R order-p interactions.
            interaction = projections[0]
            for q in projections[1:]:
                interaction = interaction * q

            mix = self.rank_mix[order_idx]
            mix = mix.detach() if detach else mix
            # [B, H, R] -> [B, H]
            interaction = (interaction * mix).sum(dim=-1)

            h = h + interaction

        return torch.relu(h)


class HMLFTN(nn.Module):
    """Recurrent multilinear FTN with a per-level kWTA mask.

    Interface mirrors ftn.FTN / stack_ftn.StackFTN: forward(x) predicts under
    the current masks, forward(x, y) reconfigures first, forward(x, m=...) uses
    an explicit (concatenated) mask.

    Args:
        input_size, output_size:  Input / output dimensions.
        hidden_size:              Units *per population*; must equal square_dim**2
                                  so each level's mask is a square_dim x
                                  square_dim grid.
        square_dim:               Side length of each per-level mask grid.
        classification:           Cross-entropy vs MSE (reconfig + self.loss).
        depth:                    Number of recurrent applications of the HMLCell.
                                  There are depth+1 masked states / masks
                                  (s_0 from the input population, s_1..s_depth
                                  from the tied recurrent population).
        rank, max_order:          HMLCell low-rank interaction rank R and highest
                                  interaction order P.
        dropout_p:                Dropout on each population (forward only; the
                                  mask search sees no dropout).
        k, reconfig_steps, reconfig_lr, reconfig_sample_size,
        reconfig_algorithm, reconfig_init_sd, reconfig_noise_scale:
                                  Mask-search budget / algorithm, shared by all
                                  levels (same knobs as the base FTN's mask).
        match_stored, match_stored_after, match_stored_topk:
                                  Match-stored (Hopfield) reconfiguration option
                                  (annealste simultaneous search only).
        mask_kernel_size, mask_lateral_steps, mask_lateral_polyak,
        mask_lateral_noise_sd, torus:
                                  SmoothKWTA lateral dynamics applied to each
                                  level's final mask; mask_lateral_steps=0 =
                                  plain kWTA.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,          # units per population == square_dim**2
        square_dim: int,
        classification: bool = False,
        binary: bool = False,
        # Topology
        depth: int = 4,
        rank: int = 4,
        max_order: int = 2,
        dropout_p: float = 0.0,
        # Reconfiguration (mask search), shared across levels
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
        mask_kernel_size: int = 3,
        mask_lateral_steps: int = 0,
        mask_lateral_polyak: float = 0.0,
        mask_lateral_noise_sd: float = 0.0,
        torus: bool = False,
    ):
        super().__init__()
        assert hidden_size == square_dim**2, (
            f"hidden_size (units per population) {hidden_size} must equal "
            f"square_dim**2 {square_dim**2} so each level's mask shares the FTN "
            f"grid / viz."
        )
        assert depth >= 1, "depth must be >= 1"

        self.layer_size = hidden_size          # units in one population
        self.depth = depth
        # Number of masked states / per-level masks (s_0 .. s_depth). Named
        # n_stack to mirror StackFTN's per-layer mask machinery.
        self.n_stack = depth + 1
        self.square_dim = square_dim
        self.classification = classification
        self.binary = binary

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
        self.match_stored_topk = match_stored_topk
        self._stored_masks_ref = None
        self.mask_lateral_steps = mask_lateral_steps
        self.mask_lateral_polyak = mask_lateral_polyak
        self.mask_lateral_noise_sd = mask_lateral_noise_sd

        # Total maskable size reported to viz/replay: one square_dim**2 block per
        # level, laid out side by side by mask_to_grid.
        self.hidden_size = self.n_stack * hidden_size

        # Input population: distinct weights, z0 = tanh(input_proj(x)). Each of
        # the square_dim**2 neurons reads the whole input.
        self.input_proj = nn.Linear(input_size, hidden_size)

        # Tied recurrent population reused for every recurrent step.
        self.cell = HMLCell(hidden_size, rank=rank, max_order=max_order)

        # One distinct linear decoder per masked state, summed (== one readout
        # over the concatenation [s_0, ..., s_depth]) plus an output bias.
        self.readouts = nn.ModuleList(
            nn.Linear(hidden_size, output_size, bias=False)
            for _ in range(self.n_stack)
        )
        self.output_bias = nn.Parameter(torch.zeros(output_size))

        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.mask_lateral = _uniform_conv(mask_kernel_size, 1.0, torus)

        # Concatenated per-level masks (single source of truth for viz/replay).
        self.register_buffer("m", torch.zeros(self.hidden_size))

        # Dispatch target used by run_experiments (ftn._reconfig).
        if reconfig_algorithm == "fast":
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
        """Concatenated (n_stack*layer_size,) mask -> list of per-level masks."""
        return list(m.reshape(self.n_stack, self.layer_size))

    # ---------------------------------------------------------------- forward

    def _readout(self, l: int, s: torch.Tensor, detach: bool) -> torch.Tensor:
        return _linear(self.readouts[l], s, detach)

    def _forward_stack(self, x, masks, detach: bool) -> torch.Tensor:
        """Run the full unroll under `masks` (a list of depth+1 per-level masks).

        detach=True detaches every input_proj / cell / readout / bias weight so
        gradients reach only the masks (used by the mask search); detach=False is
        the trainable pass and also applies dropout to each population.
        """
        # Input population: z0 = tanh(input_proj(x)); s0 = z0 * m0.
        z = torch.tanh(_linear(self.input_proj, x, detach))
        out = self.output_bias.detach() if detach else self.output_bias
        for l in range(self.n_stack):
            if not detach:
                z = self.dropout(z)
            s = z * masks[l].reshape(1, -1)
            out = out + self._readout(l, s, detach)
            # Recurrent population builds the next state from the masked one.
            if l < self.n_stack - 1:
                z = self.cell(s, detach=detach)
        return out

    def forward(self, x, y=None, m=None):
        if m is None:
            if y is not None:
                self._reconfig(x, y)
            m = self.m
        masks = self._split(m)
        return self._forward_stack(x, masks, detach=False)

    def loss(self, pred, y):
        return task_loss(pred, y, self.classification, self.binary)

    # ------------------------------------------------------ reconfiguration
    # The mask search: minimize task loss on the support batch over the per-level
    # logits, with the network weights detached so only the masks move.

    def _step_mask(self, logits, i, anneal: bool):
        """One level's forward mask for search step i.

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

    @torch.enable_grad()
    def _search_simultaneous(self, x, y, anneal: bool) -> list:
        """Jointly optimize all depth+1 levels' logits against the final loss."""
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
        # snap the running best per-level masks to the closest previous-task mask
        # in the buffer, then let the rest of the steps try to beat it.
        do_match = (
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
                    ref = torch.cat(
                        best if best is not None
                        else [kwta(lg.detach(), self.k) for lg in logits]
                    )
                    m_stored = best_stored(
                        ref, stored, self.match_stored_topk, _cand_loss
                    )
                    best = self._split(m_stored)
                    best_loss = _cand_loss(m_stored)
                    swapped = True

        with torch.no_grad():
            if best is None:
                best = [kwta(lg.detach(), self.k) for lg in logits]
            # match_stored_after >= steps never triggered above: anchor now.
            if do_match and not swapped:
                best = self._split(
                    best_stored(
                        torch.cat(best), stored, self.match_stored_topk, _cand_loss
                    )
                )
            return [self._smooth(m) for m in best]

    def _smooth(self, m: torch.Tensor) -> torch.Tensor:
        """Optional SmoothKWTA lateral dynamics on one level's mask; no-op when
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
