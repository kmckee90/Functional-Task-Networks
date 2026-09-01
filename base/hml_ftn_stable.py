"""
hml_ftn_stable.py - Stabilized Higher-order MultiLinear FTN ("hml").

This keeps the original architectural premise intact:

  * one recurrent population is reused at every composition step;
  * every output neuron owns its own linear row and its own multilinear factors;
  * a separate kWTA mask selects a sparse subspace at every recurrent level;
  * the mask is optimized on a support set while network weights are detached;
  * readouts from all selected recurrent states are combined at the output.

The changes relative to the preceding implementation are numerical and
optimization safeguards rather than a change to the compositional architecture:

  1. The recurrent cell actually uses tanh, as described by the model, so the
     state passed into higher-order products is bounded.
  2. Multilinear factor vectors are row-normalized during the forward pass. This
     removes the exact factor-scale degeneracy (c*u_1, u_2/c) while leaving the
     rank-mixing coefficient responsible for interaction amplitude.
  3. The expected gain of a factor projection is adjusted for k-sparse states,
     and recurrent linear weights get a sparsity-aware orthogonal init.
  4. Rank mixing is rank-normalized and optionally bounded with tanh, preventing
     one multilinear branch from driving the recurrent tanh permanently into
     saturation. Products accumulate in fp32 under fp16/bfloat16 execution.
  5. The summed depth readout is scaled by 1/sqrt(depth+1), so output/gradient
     scale does not grow merely because more recurrence levels are present.
  6. Mask-search logits are allocated on the correct device and optimized in
     fp32. The soft relaxation has exactly k units of total mask mass, avoiding
     the dense-vs-sparse mismatch of an unconstrained sigmoid relaxation.
  7. The STE exploration noise has a floor and a schedule that decays to zero;
     mask-logit gradients are clipped and logits are gauge-centered/clamped.
  8. "Best mask" bookkeeping evaluates the actual deterministic hard mask after
     an optimizer step instead of associating a pre-step surrogate loss with a
     different post-step mask.
  9. Non-finite mask-search losses/gradients are skipped safely rather than
     corrupting the search state.

The public interface mirrors ftn.FTN / stack_ftn.StackFTN.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ftn import _uniform_conv, best_stored, kwta, mask_to_grid, mask_to_vec, task_loss


def _linear(layer: nn.Linear, x: torch.Tensor, detach: bool) -> torch.Tensor:
    """F.linear through ``layer`` with optional parameter detachment."""
    w = layer.weight.detach() if detach else layer.weight
    b = layer.bias
    if b is not None and detach:
        b = b.detach()
    return F.linear(x, w, b)


def _row_normalized_linear(
    layer: nn.Linear,
    x: torch.Tensor,
    detach: bool,
    target_row_norm: float,
    eps: float,
) -> torch.Tensor:
    """Linear projection with a fixed norm for every output-row weight vector.

    The raw row parameters still learn their directions. Fixing their norms is
    especially useful for a CP-style multilinear factorization because factor
    magnitudes are otherwise non-identifiable: multiplying one factor by c and
    another by 1/c leaves the represented multilinear function unchanged while
    making its optimization arbitrarily ill-conditioned.
    """
    w = layer.weight.detach() if detach else layer.weight

    # Model parameters are normally fp32 even under autocast. If a model itself
    # has been converted to half precision, compute the norm in fp32 and cast the
    # effective weight back; this prevents tiny row norms from underflowing.
    norm_dtype = torch.float32 if w.dtype in (torch.float16, torch.bfloat16) else w.dtype
    w_norm = w.to(norm_dtype).norm(dim=1, keepdim=True).clamp_min(eps)
    w_eff = (w.to(norm_dtype) / w_norm) * target_row_norm
    w_eff = w_eff.to(w.dtype)
    return F.linear(x, w_eff, None)


class HMLCell(nn.Module):
    """One tied recurrent population of independent multilinear neurons.

    For output neuron n,

        a_n(s) = w_n . s
                 + sum_p sum_r alpha[n,p,r]
                       prod_q (u[n,p,q,r] . s)

        z_n = tanh(a_n(s))

    Every w/u/alpha associated with neuron n is private to that neuron. There is
    no shared feature projection between neurons; recurrence only reuses this
    same population at successive composition steps.
    """

    def __init__(
        self,
        hidden_dim: int,
        rank: int = 4,
        max_order: int = 2,
        active_k: Optional[int] = None,
        multilinear_init_scale: float = 0.05,
        multilinear_mix_bound: Optional[float] = 1.0,
        linear_gain_bound: float = 2.0,
        linear_bias_bound: Optional[float] = 1.0,
        factor_norm_eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if max_order < 1:
            raise ValueError("max_order must be >= 1")

        H, R, P = int(hidden_dim), int(rank), int(max_order)
        k = H if active_k is None else int(active_k)
        if not 1 <= k <= H:
            raise ValueError(f"active_k must be in [1, {H}], got {k}")
        if multilinear_init_scale < 0:
            raise ValueError("multilinear_init_scale must be >= 0")
        if multilinear_mix_bound is not None and multilinear_mix_bound <= 0:
            raise ValueError("multilinear_mix_bound must be > 0 or None")
        if linear_gain_bound <= 0:
            raise ValueError("linear_gain_bound must be > 0")
        if linear_bias_bound is not None and linear_bias_bound <= 0:
            raise ValueError("linear_bias_bound must be > 0 or None")

        self.H, self.R, self.P = H, R, P
        self.active_k = k
        self.factor_norm_eps = float(factor_norm_eps)
        self.multilinear_mix_bound = multilinear_mix_bound
        self.linear_gain_bound = float(linear_gain_bound)
        self.linear_bias_bound = linear_bias_bound

        # A k-sparse state retains k/H of the coordinates. Compensating by
        # sqrt(H/k) makes a random masked recurrence roughly norm-preserving at
        # initialization rather than progressively vanishing with depth.
        self.sparse_gain = math.sqrt(H / k)

        # The linear row direction and amplitude are parameterized separately.
        # This prevents recurrent row norms from running away while still giving
        # every neuron an independent learned amplitude in [0, bound*sparse_gain].
        self.linear = nn.Linear(H, H)
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.zeros_(self.linear.bias)
        self.linear_gain_logits = nn.Parameter(torch.zeros(H))

        # Independent factor projections per interaction order and factor.
        # A layer has H*R output rows: each (neuron, rank) pair owns one private
        # projection vector. Their effective row norms are fixed in prepare().
        self.factors = nn.ModuleList(
            [
                nn.ModuleList(
                    [nn.Linear(H, H * R, bias=False) for _ in range(order)]
                )
                for order in range(2, P + 1)
            ]
        )
        for order_layers in self.factors:
            for layer in order_layers:
                nn.init.normal_(layer.weight, mean=0.0, std=1.0 / math.sqrt(H))

        # Raw rank-mixing parameters. Effective coefficients are rank-normalized
        # and optionally bounded, so interaction amplitude cannot explode merely
        # by increasing rank or by unconstrained coefficient growth.
        self.rank_mix = nn.ParameterList(
            [nn.Parameter(torch.empty(H, R)) for _ in range(2, P + 1)]
        )
        for mix in self.rank_mix:
            nn.init.normal_(mix, mean=0.0, std=multilinear_init_scale)

    def _normalized_rows(
        self,
        w: torch.Tensor,
        target_norm: torch.Tensor,
    ) -> torch.Tensor:
        norm_dtype = torch.float32 if w.dtype in (torch.float16, torch.bfloat16) else w.dtype
        wf = w.to(norm_dtype)
        denom = wf.norm(dim=1, keepdim=True).clamp_min(self.factor_norm_eps)
        eff = wf / denom
        eff = eff * target_norm.to(norm_dtype)
        return eff.to(w.dtype)

    def _effective_mix(self, raw_mix: torch.Tensor, detach: bool) -> torch.Tensor:
        mix = raw_mix.detach() if detach else raw_mix
        if self.multilinear_mix_bound is not None:
            mix = self.multilinear_mix_bound * torch.tanh(mix)
        return mix / math.sqrt(self.R)

    def prepare(self, detach: bool = False):
        """Prepare bounded effective recurrent parameters once per full unroll.

        The same tied cell is applied at every recurrence level, so recomputing
        row normalizations at every level is wasted work. Reusing these tensors
        also preserves exactly the same gradients as recomputing them each call.
        """
        raw_w = self.linear.weight.detach() if detach else self.linear.weight
        gain_logits = self.linear_gain_logits.detach() if detach else self.linear_gain_logits
        gain = (
            self.sparse_gain
            * self.linear_gain_bound
            * torch.sigmoid(gain_logits)
        ).reshape(-1, 1)
        linear_w = self._normalized_rows(raw_w, gain)

        raw_b = self.linear.bias.detach() if detach else self.linear.bias
        if self.linear_bias_bound is None:
            linear_b = raw_b
        else:
            linear_b = self.linear_bias_bound * torch.tanh(raw_b)

        factor_weights = []
        factor_target = torch.as_tensor(
            self.sparse_gain,
            device=raw_w.device,
            dtype=raw_w.dtype,
        ).reshape(1, 1)
        for order_layers in self.factors:
            prepared_order = []
            for layer in order_layers:
                w = layer.weight.detach() if detach else layer.weight
                prepared_order.append(self._normalized_rows(w, factor_target))
            factor_weights.append(prepared_order)

        mixes = [self._effective_mix(mix, detach) for mix in self.rank_mix]
        return linear_w, linear_b, factor_weights, mixes

    def forward(self, s: torch.Tensor, detach: bool = False, prepared=None) -> torch.Tensor:
        if s.shape[-1] != self.H:
            raise ValueError(f"Expected final dimension {self.H}, got {s.shape[-1]}")

        if prepared is None:
            prepared = self.prepare(detach=detach)
        linear_w, linear_b, factor_weights, mixes = prepared

        B = s.shape[0]
        linear_term = F.linear(s, linear_w, linear_b)

        # Higher-order products are substantially more fragile in fp16 than a
        # linear layer. Accumulate in fp32 under low precision, apply tanh before
        # casting back, and keep the recurrent state bounded at every step.
        accum_dtype = (
            torch.float32
            if linear_term.dtype in (torch.float16, torch.bfloat16)
            else linear_term.dtype
        )
        h = linear_term.to(accum_dtype)

        for order_idx, _order in enumerate(range(2, self.P + 1)):
            projections = [
                F.linear(s, weight, None).reshape(B, self.H, self.R)
                for weight in factor_weights[order_idx]
            ]

            interaction = projections[0].to(accum_dtype)
            for projection in projections[1:]:
                interaction = interaction * projection.to(accum_dtype)

            mix = mixes[order_idx].to(accum_dtype)
            h = h + (interaction * mix.unsqueeze(0)).sum(dim=-1)

        return torch.tanh(h).to(linear_term.dtype)


class HMLFTN(nn.Module):
    """Recurrent multilinear FTN with one kWTA mask per recurrent level."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        square_dim: int,
        classification: bool = False,
        binary: bool = False,
        # Topology
        depth: int = 4,
        rank: int = 4,
        max_order: int = 2,
        dropout_p: float = 0.0,
        # Reconfiguration
        k: int = 128,
        reconfig_steps: int = 30,
        reconfig_lr: float = 0.3,
        reconfig_sample_size: int = 2048,
        reconfig_algorithm: str = "annealste",
        reconfig_init_sd: float = 1e-5,
        reconfig_noise_scale: float = 1.0,
        # Match-stored reconfiguration
        match_stored: bool = False,
        match_stored_after: int = 0,
        match_stored_topk: int = 1,
        mask_kernel_size: int = 3,
        mask_lateral_steps: int = 0,
        mask_lateral_polyak: float = 0.0,
        mask_lateral_noise_sd: float = 0.0,
        torus: bool = False,
        # Stabilization controls. Appended so existing positional callers keep
        # the same meaning as the preceding implementation.
        multilinear_init_scale: float = 0.05,
        multilinear_mix_bound: Optional[float] = 1.0,
        linear_gain_bound: float = 2.0,
        linear_bias_bound: Optional[float] = 1.0,
        reconfig_temperature: float = 1.0,
        reconfig_noise_floor: float = 0.05,
        reconfig_grad_clip: Optional[float] = 5.0,
        reconfig_logit_clip: Optional[float] = 8.0,
        soft_mask_solve_steps: int = 6,
        reconfig_eval_interval: int = 1,
    ):
        super().__init__()

        if hidden_size != square_dim**2:
            raise ValueError(
                f"hidden_size (units per population) {hidden_size} must equal "
                f"square_dim**2 {square_dim**2}"
            )
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if not 1 <= k <= hidden_size:
            raise ValueError(f"k must be in [1, {hidden_size}], got {k}")
        if reconfig_steps < 1:
            raise ValueError("reconfig_steps must be >= 1")
        if reconfig_lr <= 0:
            raise ValueError("reconfig_lr must be > 0")
        if reconfig_sample_size < 1:
            raise ValueError("reconfig_sample_size must be >= 1")
        if reconfig_algorithm not in {"annealste", "fast"}:
            raise ValueError("reconfig_algorithm must be 'annealste' or 'fast'")
        if reconfig_temperature <= 0:
            raise ValueError("reconfig_temperature must be > 0")
        if reconfig_noise_floor < 0:
            raise ValueError("reconfig_noise_floor must be >= 0")
        if soft_mask_solve_steps < 1:
            raise ValueError("soft_mask_solve_steps must be >= 1")
        if reconfig_eval_interval < 1:
            raise ValueError("reconfig_eval_interval must be >= 1")

        self.layer_size = int(hidden_size)
        self.depth = int(depth)
        self.n_stack = self.depth + 1
        self.square_dim = int(square_dim)
        self.classification = classification
        self.binary = binary

        self.k = int(k)
        self.reconfig_steps = int(reconfig_steps)
        self.reconfig_lr = float(reconfig_lr)
        self.reconfig_sample_size = int(reconfig_sample_size)
        self.reconfig_algorithm = reconfig_algorithm
        self.reconfig_init_sd = float(reconfig_init_sd)
        self.reconfig_noise_scale = float(reconfig_noise_scale)
        self.reconfig_temperature = float(reconfig_temperature)
        self.reconfig_noise_floor = float(reconfig_noise_floor)
        self.reconfig_grad_clip = reconfig_grad_clip
        self.reconfig_logit_clip = reconfig_logit_clip
        self.soft_mask_solve_steps = int(soft_mask_solve_steps)
        self.reconfig_eval_interval = int(reconfig_eval_interval)

        self.match_stored = match_stored
        self.match_stored_after = int(match_stored_after)
        self.match_stored_topk = int(match_stored_topk)
        self._stored_masks_ref = None
        self.mask_lateral_steps = int(mask_lateral_steps)
        self.mask_lateral_polyak = float(mask_lateral_polyak)
        self.mask_lateral_noise_sd = float(mask_lateral_noise_sd)

        # Exposed mask size is still all per-level masks concatenated.
        self.hidden_size = self.n_stack * self.layer_size

        self.input_proj = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        self.cell = HMLCell(
            hidden_size,
            rank=rank,
            max_order=max_order,
            active_k=k,
            multilinear_init_scale=multilinear_init_scale,
            multilinear_mix_bound=multilinear_mix_bound,
            linear_gain_bound=linear_gain_bound,
            linear_bias_bound=linear_bias_bound,
        )

        self.readouts = nn.ModuleList(
            nn.Linear(hidden_size, output_size, bias=False)
            for _ in range(self.n_stack)
        )
        for readout in self.readouts:
            nn.init.xavier_uniform_(readout.weight)
        self.output_bias = nn.Parameter(torch.zeros(output_size))

        # Summing L independent readouts otherwise increases initial output and
        # gradient RMS by sqrt(L). This constant does not reduce expressivity;
        # readout weights can absorb it.
        self.readout_scale = 1.0 / math.sqrt(self.n_stack)

        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.mask_lateral = _uniform_conv(mask_kernel_size, 1.0, torus)

        self.register_buffer("m", torch.zeros(self.hidden_size))

        self._reconfig = (
            self.fast_reconfigure if reconfig_algorithm == "fast" else self.reconfigure
        )

    # ------------------------------------------------------------ mask packing

    def set_stored_masks_ref(self, masks_ref):
        self._stored_masks_ref = masks_ref

    def _split(self, m: torch.Tensor) -> List[torch.Tensor]:
        if m.numel() != self.hidden_size:
            raise ValueError(
                f"Mask has {m.numel()} entries; expected {self.hidden_size}"
            )
        return list(m.reshape(self.n_stack, self.layer_size))

    # ---------------------------------------------------------------- forward

    def _readout(self, level: int, s: torch.Tensor, detach: bool) -> torch.Tensor:
        return _linear(self.readouts[level], s, detach)

    def _forward_stack(
        self,
        x: torch.Tensor,
        masks: List[torch.Tensor],
        detach: bool,
    ) -> torch.Tensor:
        if len(masks) != self.n_stack:
            raise ValueError(f"Expected {self.n_stack} masks, got {len(masks)}")

        z = torch.tanh(_linear(self.input_proj, x, detach))
        out = self.output_bias.detach() if detach else self.output_bias
        prepared_cell = self.cell.prepare(detach=detach)

        for level in range(self.n_stack):
            if not detach:
                z = self.dropout(z)

            # Search logits are deliberately fp32; cast only the mask value into
            # the state dtype here. Autograd still propagates through the cast.
            mask = masks[level].reshape(1, -1).to(device=z.device, dtype=z.dtype)
            s = z * mask
            out = out + self.readout_scale * self._readout(level, s, detach)

            if level < self.n_stack - 1:
                z = self.cell(s, detach=detach, prepared=prepared_cell)

        return out

    def forward(self, x, y=None, m=None):
        if m is None:
            if y is not None:
                self._reconfig(x, y)
            m = self.m
        return self._forward_stack(x, self._split(m), detach=False)

    def loss(self, pred, y):
        return task_loss(pred, y, self.classification, self.binary)

    # ------------------------------------------------------ reconfiguration

    def _soft_cardinality_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Smooth mask in [0,1] with total forward mass approximately k.

        A plain sigmoid starts with H/2 active mass regardless of k. Here a
        detached scalar threshold is solved by a few Newton iterations so the
        soft network seen by the mask optimizer has the same total mask mass as
        the final kWTA network. The solve stays entirely on device.
        """
        if self.k == self.layer_size:
            return torch.ones_like(logits)

        temperature = self.reconfig_temperature
        with torch.no_grad():
            z = logits.detach()
            q = self.k / self.layer_size
            logit_q = math.log(q) - math.log1p(-q)
            threshold = z.mean() - temperature * logit_q

            for _ in range(self.soft_mask_solve_steps):
                soft = torch.sigmoid((z - threshold) / temperature)
                error = soft.sum() - self.k
                derivative = -(soft * (1.0 - soft)).sum() / temperature
                derivative = torch.minimum(
                    derivative,
                    torch.as_tensor(-1e-6, device=z.device, dtype=z.dtype),
                )
                threshold = threshold - error / derivative

        return torch.sigmoid((logits - threshold) / temperature)

    def _step_mask(self, logits: torch.Tensor, step: int, anneal: bool) -> torch.Tensor:
        soft = self._soft_cardinality_mask(logits)
        if not anneal:
            return soft

        if self.reconfig_steps <= 1:
            progress = 1.0
        else:
            progress = step / (self.reconfig_steps - 1)

        # Exploration is meaningful even when logits are initialized almost
        # identically; unlike std(logits) alone, the floor does not vanish at step 0.
        spread = logits.detach().std().clamp_min(self.reconfig_noise_floor)
        noise_sd = self.reconfig_noise_scale * spread * (1.0 - progress) ** 2
        noisy_logits = logits + torch.randn_like(logits) * noise_sd
        hard = kwta(noisy_logits, self.k)

        # Hard kWTA in the forward pass, cardinality-calibrated soft gradient.
        return hard + soft - soft.detach()

    def _hard_masks_from_logits(self, logits: List[torch.Tensor]) -> List[torch.Tensor]:
        return [kwta(lg.detach(), self.k) for lg in logits]

    def _sanitize_logits(self, logits: List[torch.Tensor]) -> None:
        """Fix the top-k common-offset gauge and avoid useless extreme logits."""
        with torch.no_grad():
            for lg in logits:
                lg.sub_(lg.mean())
                if self.reconfig_logit_clip is not None:
                    lg.clamp_(-self.reconfig_logit_clip, self.reconfig_logit_clip)

    @torch.enable_grad()
    def _search_simultaneous(self, x, y, anneal: bool) -> List[torch.Tensor]:
        n = min(self.reconfig_sample_size, x.shape[0])
        x, y = x[:n], y[:n]

        # Keep the small optimization problem in fp32 even under mixed precision.
        # Masks are cast to the recurrent state dtype inside _forward_stack().
        logits = [
            (
                torch.randn(
                    self.layer_size,
                    device=x.device,
                    dtype=torch.float32,
                )
                * self.reconfig_init_sd
            ).requires_grad_(True)
            for _ in range(self.n_stack)
        ]
        self._sanitize_logits(logits)
        opt = torch.optim.Adam(logits, lr=self.reconfig_lr)

        do_match = (
            anneal
            and self.match_stored
            and self._stored_masks_ref is not None
            and len(self._stored_masks_ref) > 0
        )
        stored = None
        if do_match:
            stored = torch.stack(list(self._stored_masks_ref)).to(
                device=x.device, dtype=self.m.dtype
            )
        swapped = False

        def _cand_loss(cand: torch.Tensor) -> float:
            with torch.no_grad():
                pred = self._forward_stack(x, self._split(cand), detach=True)
                value = self.loss(pred, y)
                return float(value.detach().float().item())

        # Track the actual deterministic hard-mask objective from the beginning.
        best = self._hard_masks_from_logits(logits)
        best_loss = _cand_loss(torch.cat(best))

        for step in range(self.reconfig_steps):
            opt.zero_grad(set_to_none=True)
            masks = [self._step_mask(lg, step, anneal) for lg in logits]
            pred = self._forward_stack(x, masks, detach=True)
            loss = self.loss(pred, y)

            if not torch.isfinite(loss.detach()):
                # Do not let one pathological support pass poison Adam state.
                with torch.no_grad():
                    for lg in logits:
                        lg.mul_(0.5)
                        if self.reconfig_init_sd > 0:
                            lg.add_(torch.randn_like(lg) * self.reconfig_init_sd)
                self._sanitize_logits(logits)
                continue

            loss.backward()

            if self.reconfig_grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    logits, max_norm=self.reconfig_grad_clip
                )
                if not torch.isfinite(torch.as_tensor(grad_norm)).item():
                    opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        for lg in logits:
                            lg.mul_(0.5)
                    self._sanitize_logits(logits)
                    continue

            opt.step()
            self._sanitize_logits(logits)

            # Evaluate the mask we would actually deploy, not the noisy/soft STE
            # forward mask and not a post-update mask paired with a pre-update loss.
            should_eval = (
                (step + 1) % self.reconfig_eval_interval == 0
                or step + 1 == self.reconfig_steps
            )
            if should_eval:
                candidate = self._hard_masks_from_logits(logits)
                candidate_loss = _cand_loss(torch.cat(candidate))
                if math.isfinite(candidate_loss) and candidate_loss < best_loss:
                    best_loss = candidate_loss
                    best = [m.clone() for m in candidate]

            if do_match and not swapped and (step + 1) >= self.match_stored_after:
                ref = torch.cat(best)
                m_stored = best_stored(
                    ref, stored, self.match_stored_topk, _cand_loss
                ).to(device=x.device, dtype=self.m.dtype)
                stored_loss = _cand_loss(m_stored)

                # Preserve the original match-stored semantics (the stored mask
                # becomes the anchor candidate), but seed logits around that mask
                # as well; otherwise the alleged "snap" never affects the search.
                best = [m.clone() for m in self._split(m_stored)]
                best_loss = stored_loss
                with torch.no_grad():
                    for lg, target in zip(logits, best):
                        target = target.to(device=lg.device, dtype=lg.dtype)
                        lg.copy_(2.0 * target - 1.0)
                opt.state.clear()
                self._sanitize_logits(logits)
                swapped = True

        with torch.no_grad():
            if do_match and not swapped:
                m_stored = best_stored(
                    torch.cat(best), stored, self.match_stored_topk, _cand_loss
                ).to(device=x.device, dtype=self.m.dtype)
                stored_loss = _cand_loss(m_stored)
                if math.isfinite(stored_loss) and stored_loss < best_loss:
                    best = [m.clone() for m in self._split(m_stored)]

            return [self._smooth(m) for m in best]

    def _smooth(self, m: torch.Tensor) -> torch.Tensor:
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
            mf = (
                self.mask_lateral_polyak * mf
                + (1.0 - self.mask_lateral_polyak) * m_lat
            )
        return kwta(mf, self.k)

    # ------------------------------------------------------ reconfig entry points

    def reconfigure(self, x, y) -> torch.Tensor:
        masks = self._search_simultaneous(x, y, anneal=True)
        self.m.copy_(torch.cat(masks).to(device=self.m.device, dtype=self.m.dtype))
        return self.m

    def fast_reconfigure(self, x, y) -> torch.Tensor:
        masks = self._search_simultaneous(x, y, anneal=False)
        self.m.copy_(torch.cat(masks).to(device=self.m.device, dtype=self.m.dtype))
        return self.m
