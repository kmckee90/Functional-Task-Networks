"""
lora_ftn.py - FTN-LoRA: the LoRA-bank alternative to the ParallelMLP FTN.

Selected with ARCHITECTURE="lora" in config.py. Presents the same interface as
ftn.FTN (`ftn(x, y)` reconfigures + predicts, `ftn(x)` predicts, `ftn(x, m=...)`
predicts under an explicit mask, `ftn._reconfig(x, y)`, `ftn.loss`, `ftn.m`,
`ftn.hidden_size`), so run_experiments.py drives it unchanged.

How it differs from the ParallelMLP FTN
========================================
The base FTN computes every unit's activation `z` in one forward pass and the
mask *selects a subspace of z* — so a reconfiguration step gets a gradient on
every unit at once.

FTN-LoRA has ONE frozen backbone (a plain random MLP whose weights are never
trained) plus a bank of H LoRA atoms on every linear. A single shared binary
mask selects k of the H atoms; the active atoms' rank-r deltas are added to the
backbone weights. There is no "compute all z then select": an atom only affects
the output through the backbone computation it modifies, so reconfiguration must
run a real forward pass per candidate mask. LoRALinear.forward computes *every*
atom's delta and gates it by the full mask (a dense elementwise multiply), so —
like the base FTN — the reconfiguration step gets a gradient on every atom at
once, not just the k active ones. Inactive atoms contribute zero value but a
nonzero STE gradient, and the annealed exploration noise still drives which
atoms enter the active set — see reconfigure() below.

Only the atom bank is trained; the backbone weights stay at their random init
(LORA_WEIGHT_SCALE controls that fixed scale).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the mask primitives so the LoRA mask lives in the same grid / viz world.
from ftn import (
    _uniform_conv,
    best_stored,
    closest_stored,
    kwta,
    mask_to_grid,
    mask_to_vec,
    task_loss,
)


class MaskController:
    """Shared holder for the active mask + a reconfiguration flag.

    Every LoRALinear reads the SAME controller, so one selection drives the
    whole backbone. `detach_atoms` is flipped True during reconfiguration so the
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

    Every atom's delta is computed and gated by its mask entry m_j (a full
    elementwise multiply), so every atom is on the graph and every logit gets a
    gradient during the search — inactive atoms (m_j == 0) contribute zero value
    but a nonzero STE gradient. Swap this dense multiply for a top-k gather over
    (m_j > 0) to make the search's gradient sparse (only the k active atoms) —
    the one lever that changes the search's gradient sparsity, at the cost of
    computing all H atoms every forward pass.
    """

    def __init__(self, in_f: int, out_f: int, controller: MaskController,
                 rank: int, alpha: float, weight_scale: float,
                 b0_scale: float, dropout_p: float, use_bias: bool = True):
        super().__init__()
        self.controller = controller
        self.alpha = alpha

        # Frozen random backbone weight: N(0, weight_scale / sqrt(fan_in)), so
        # weight_scale == 1 is a sanely-scaled init and cranking it up/down is a
        # clean knob on the fixed backbone the atoms have to correct. No base
        # bias — the bias is provided entirely by the LoRA bias bank below (or
        # absent when use_bias is False), so nothing about the bias is static.
        base = nn.Linear(in_f, out_f, bias=False)
        with torch.no_grad():
            base.weight.normal_(0.0, weight_scale / math.sqrt(in_f))
        base.weight.requires_grad_(False)
        self.base = base

        H = controller.H
        A = torch.empty(H, in_f, rank)
        nn.init.kaiming_uniform_(A, a=math.sqrt(5))
        # B (and the bias bank) are small-but-nonzero (NOT zero): a zero delta
        # gives an untrained atom a zero gradient, so the mask search could
        # never select it. b0_scale is the atom-reachability knob grouped with
        # the reconfiguration params.
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
        # Dense: every atom's delta is computed and gated by its mask entry, so
        # every m_h has a path to the loss and receives a gradient. Inactive
        # atoms (m_h == 0) contribute zero value but a nonzero gradient via the
        # STE, matching the base FTN's "compute-all-then-select" gradient.
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


class LoRAFTN(nn.Module):
    """Frozen random MLP + trainable LoRA atom bank + task mask.

    Interface mirrors ftn.FTN: forward(x) predicts under the current mask,
    forward(x, y) reconfigures first, forward(x, m=...) uses an explicit mask.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,          # H = number of LoRA atoms == square_dim**2
        square_dim: int,
        classification: bool = False,
        binary: bool = False,
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
        # Reconfiguration (mask search)
        k: int = 64,
        reconfig_steps: int = 30,
        reconfig_lr: float = 0.3,
        reconfig_sample_size: int = 2048,
        reconfig_algorithm: str = "annealste",   # "annealste" or "fast"
        reconfig_init_sd: float = 1e-5,
        reconfig_noise_scale: float = 1.0,
        # Match-stored reconfiguration (annealste only)
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
            f"hidden_size (atom count) {hidden_size} must equal square_dim**2 "
            f"{square_dim**2} so the mask shares the FTN grid / visualization."
        )
        self.hidden_size = hidden_size
        self.square_dim = square_dim
        self.classification = classification
        self.binary = binary
        self.k = k
        self.reconfig_steps = reconfig_steps
        self.reconfig_lr = reconfig_lr
        self.reconfig_sample_size = reconfig_sample_size
        self.reconfig_init_sd = reconfig_init_sd
        self.reconfig_noise_scale = reconfig_noise_scale
        self.mask_lateral_steps = mask_lateral_steps
        self.mask_lateral_polyak = mask_lateral_polyak
        self.mask_lateral_noise_sd = mask_lateral_noise_sd
        # Match-stored reconfiguration option (see _search). Live reference to the
        # replay / noise-probe mask buffer, wired by run_experiments; None
        # disables regardless of match_stored.
        self.match_stored = match_stored
        self.match_stored_after = match_stored_after
        # Hopfield recall breadth: consider the top-k highest-overlap stored
        # masks and keep the best-performing one (by loss). 1 = closest match.
        self.match_stored_topk = match_stored_topk
        self._stored_masks_ref = None

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

        self.mask_lateral = _uniform_conv(mask_kernel_size, 1.0, torus)

        # Persistent task mask (single source of truth); the controller mask is
        # a scratch that forward()/reconfigure() set right before running.
        self.register_buffer("m", torch.zeros(hidden_size))

        # Dispatch target used by run_experiments (ftn._reconfig).
        self._reconfig = (
            self.fast_reconfigure if reconfig_algorithm == "fast"
            else self.reconfigure
        )

    # ---------------------------------------------------------------- forward

    def set_stored_masks_ref(self, masks_ref):
        """Wire in the live buffer of stored task masks (a deque, e.g.
        ReplayBuffer.masks / NoiseProbeBuffer.masks) used by the match-stored
        annealste option. run_experiments sets this; None disables it."""
        self._stored_masks_ref = masks_ref

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run the LoRA MLP under whatever mask the controller currently holds."""
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
        return h

    def forward(self, x, y=None, m=None):
        if m is None:
            if y is not None:
                self._reconfig(x, y)
            m = self.m
        self.controller.set_mask(m)
        return self._backbone(x)

    def loss(self, pred, y):
        return task_loss(pred, y, self.classification, self.binary)

    # ------------------------------------------------------ reconfiguration
    # This is the experimental surface: a self-contained mask search over the
    # atom bank that minimizes task loss on the support batch. Swap the body to
    # try different protocols (noise schedule, optimizer, dense vs. gathered
    # gradient, second mask, ...). Everything above stays fixed.

    @torch.enable_grad()
    def _search(self, x, y, anneal: bool) -> torch.Tensor:
        """Optimize H mask logits to minimize task loss under the current atoms.

        anneal=True  -> annealed straight-through kWTA (hard top-k forward,
                        sigmoid STE gradient, decaying exploration noise); the
                        noise is what pulls new atoms into the active set.
        anneal=False -> "fast": plain sigmoid soft mask, no noise/STE, then a
                        single kWTA at the end.
        The atoms are detached, so only the logits move here.
        """
        n = min(self.reconfig_sample_size, x.shape[0])
        x, y = x[:n], y[:n]

        logits = (torch.randn(self.hidden_size) * self.reconfig_init_sd) 
        # logits = (torch.randn(self.hidden_size) * self.reconfig_init_sd)

        logits = logits.detach().requires_grad_(True)
        opt = torch.optim.Adam([logits], lr=self.reconfig_lr)

        # Match-stored option (annealste only): after match_stored_after steps,
        # snap the running best to the closest previous-task mask in the buffer,
        # then let the rest of the steps try to beat it.
        do_match = (
            anneal
            and self.match_stored
            and self._stored_masks_ref is not None
            and len(self._stored_masks_ref) > 0
        )
        stored = torch.stack(list(self._stored_masks_ref)) if do_match else None
        swapped = False

        def _cand_loss(cand: torch.Tensor) -> float:
            with torch.no_grad():
                self.controller.set_mask(cand)
                return self.loss(self._backbone(x), y).item()

        self.controller.detach_atoms = True
        best, best_loss = None, math.inf
        try:
            for i in range(self.reconfig_steps):
                if anneal:
                    sm = torch.sigmoid(logits)
                    tau = logits.detach().std() * math.exp(
                        -i / max(1.0, self.reconfig_steps / 4)
                    )
                    noise = torch.randn_like(logits) * tau * self.reconfig_noise_scale
                    mask = kwta(logits + noise, self.k) + sm - sm.detach()
                    # mask.retain_grad()
                    # sm.retain_grad()
                else:
                    mask = torch.sigmoid(logits)
                self.controller.set_mask(mask)

                loss = self.loss(self._backbone(x), y)
                opt.zero_grad()
                loss.backward()
                # breakpoint()
                opt.step()

                with torch.no_grad():
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best = kwta(logits.detach(), self.k)
                    if do_match and not swapped and (i + 1) >= self.match_stored_after:
                        ref = best if best is not None else kwta(logits.detach(), self.k)
                        m_stored = best_stored(ref, stored, self.match_stored_topk, _cand_loss)
                        best_loss = _cand_loss(m_stored)
                        best = m_stored
                        swapped = True
        finally:
            self.controller.detach_atoms = False

        with torch.no_grad():
            if do_match and not swapped:
                ref = best if best is not None else kwta(logits.detach(), self.k)
                best = best_stored(ref, stored, self.match_stored_topk, _cand_loss)
            m = best if best is not None else kwta(logits.detach(), self.k)
            m = self._smooth(m)
        return m

    def _smooth(self, m: torch.Tensor) -> torch.Tensor:
        """Optional SmoothKWTA lateral dynamics; no-op when steps==0."""
        if self.mask_lateral_steps == 0 or self.mask_lateral_polyak >= 1.0:
            return m
        mf = m.float()
        for _ in range(self.mask_lateral_steps):
            g = mask_to_grid(mf, self.square_dim).unsqueeze(0).unsqueeze(0)
            m_lat = mask_to_vec(self.mask_lateral(g).squeeze(0).squeeze(0),
                                self.square_dim)
            if self.mask_lateral_noise_sd > 0:
                m_lat = m_lat + torch.randn_like(m_lat) * self.mask_lateral_noise_sd
            mf = self.mask_lateral_polyak * mf + (1 - self.mask_lateral_polyak) * m_lat
        return kwta(mf, self.k)

    def reconfigure(self, x, y) -> torch.Tensor:
        """Annealed-STE mask search; writes the result into self.m."""
        best = self._search(x, y, anneal=True)
        self.m.copy_(best)
        return self.m

    def fast_reconfigure(self, x, y) -> torch.Tensor:
        """Fast (soft-sigmoid) mask search; writes the result into self.m."""
        best = self._search(x, y, anneal=False)
        self.m.copy_(best)
        return self.m
