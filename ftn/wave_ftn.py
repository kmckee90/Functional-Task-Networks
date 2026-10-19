"""FTN with a fixed WaveTanh reservoir in place of learned recurrence.

For each independent sample, r starts at zero. At each processing step:
    z_t = cell(concat(x_t, local_patches(r[:, 0]))) * m_t
    r[:, 0] += input_scale * (z_t * (abs(z_t) > input_threshold)).reshape(grid)
    r = WaveTanh(r)                         # substeps times
Thresholding only gates new injections, before input scaling; existing ripples
continue propagating. Retained values keep their original sign and magnitude.
Zero threshold disables the gate, including its effect on search gradients.
Final output: y = readout(z_t). Cell/readout weights are shared at every step;
the wave kernel is a buffer with no learned parameters. Only the primary wave
channel is visible to neurons. The secondary channel remains internal memory.

step_masks=True stores independently selectable (not necessarily disjoint)
masks per step. False stores one mask reused throughout the unroll.
reconfig_bptt=False cuts the history before the final step DURING MASK SEARCH
ONLY: earlier independent masks stay fixed and only the final mask is searched.
For a shared mask, its final use gets gradient; the earlier uses are detached.
reconfig_bptt=True optimizes all uses through the neurons and fixed reservoir.
Weight training always uses the full recurrence graph in either mode.
"""

import math

import torch
from reconfig_optimizer import make_reconfig_optimizer, validate_reconfig_optimizer
from torch import nn
from torch.nn import functional as F

from ftn import ParallelMLP, _uniform_conv, best_stored, kwta, task_loss
from stack_ftn import StackFTN
from projects.wave.wave_tanh import WaveTanh


from overlap_regularizer import OverlapRegularizer


class WaveFTN(OverlapRegularizer, nn.Module):
    def __init__(self, input_size, output_size, hidden_size, square_dim,
                 classification=False, binary=False, n_stack=6,
                 step_masks=True, reconfig_bptt=True,
                 wave_rho=.99, wave_c=.65, wave_omega=.15,
                 wave_padding="circular", wave_substeps=4, wave_input_scale=.01,
                 wave_input_threshold=0., wave_input_first_step_only=False,
                 wave_input_kernel_size=3,
                 layers=8, inner_size=8, dropout_p=0., k=128,
                 reconfig_steps=30, reconfig_lr=.3, reconfig_sample_size=2048,
                 reconfig_optimizer="Adam",
                 reconfig_algorithm="annealste", reconfig_init_sd=1e-5,
                 reconfig_noise_scale=1., match_stored=False,
                 match_stored_after=0, match_stored_topk=1, match_tie_steps=True,
                 mask_kernel_size=3, mask_lateral_steps=0,
                 mask_lateral_polyak=0., mask_lateral_noise_sd=0., torus=False,
                 overlap_mode="seen_logits", overlap_lambda=0.,
                 overlap_threshold=.5, overlap_temperature=.2, reconfig_entropy_coef=0.):
        super().__init__()
        self.configure_overlap(overlap_mode, overlap_lambda, overlap_threshold, overlap_temperature)
        self.configure_entropy(reconfig_entropy_coef)
        if hidden_size != square_dim**2 or square_dim < 2:
            raise ValueError("hidden_size must equal square_dim², with square_dim >= 2")
        if n_stack < 1 or wave_substeps < 1 or not 1 <= k <= hidden_size:
            raise ValueError("require positive step counts and 1 <= k <= hidden_size")
        if not (math.isfinite(wave_rho) and 0 < wave_rho <= 1
                and wave_c > 0 and wave_omega > 0
                and wave_omega**2 + (16/3)*wave_c**2 < 4):
            raise ValueError("require 0 < rho <= 1, c, omega > 0 and omega²+(16/3)c² < 4")
        if not math.isfinite(wave_input_scale) or wave_input_scale <= 0:
            raise ValueError("wave_input_scale must be positive and finite")
        if not math.isfinite(wave_input_threshold) or wave_input_threshold < 0:
            raise ValueError("wave_input_threshold must be nonnegative and finite")
        if reconfig_algorithm not in ("annealste", "fast"):
            raise ValueError("reconfig_algorithm must be annealste or fast")
        if reconfig_steps < 0 or reconfig_sample_size < 1:
            raise ValueError("require reconfig_steps >= 0 and reconfig_sample_size >= 1")
        if (isinstance(wave_input_kernel_size, bool)
                or not isinstance(wave_input_kernel_size, int)
                or wave_input_kernel_size < 1 or wave_input_kernel_size % 2 == 0):
            raise ValueError("wave_input_kernel_size must be a positive odd integer")
        if wave_input_kernel_size // 2 >= square_dim:
            raise ValueError("wave input patch radius must be smaller than square_dim")
        self.input_size, self.output_size = input_size, output_size
        self.layer_size, self.square_dim = hidden_size, square_dim
        self.n_stack, self.step_masks = int(n_stack), bool(step_masks)
        self.n_masks = self.n_stack if self.step_masks else 1
        self.hidden_size = self.n_masks * hidden_size  # replay/viz mask layout
        self.classification, self.binary = classification, binary
        self.reconfig_bptt = bool(reconfig_bptt)
        self.wave_substeps, self.wave_input_scale = int(wave_substeps), wave_input_scale
        self.wave_input_threshold = wave_input_threshold
        self.wave_input_first_step_only = bool(wave_input_first_step_only)
        self.wave_input_kernel_size = wave_input_kernel_size
        self.wave_padding = wave_padding
        self.k, self.reconfig_steps = k, reconfig_steps
        self.reconfig_lr, self.reconfig_sample_size = reconfig_lr, reconfig_sample_size
        self.reconfig_optimizer = validate_reconfig_optimizer(reconfig_optimizer)
        self.reconfig_algorithm, self.reconfig_init_sd = reconfig_algorithm, reconfig_init_sd
        self.reconfig_noise_scale = reconfig_noise_scale
        self.match_stored, self.match_stored_after = match_stored, match_stored_after
        self.match_stored_topk, self.match_tie_steps = match_stored_topk, match_tie_steps
        self._stored_masks_ref = None
        self.mask_lateral_steps = mask_lateral_steps
        self.mask_lateral_polyak = mask_lateral_polyak
        self.mask_lateral_noise_sd = mask_lateral_noise_sd
        self.mask_lateral = _uniform_conv(mask_kernel_size, 1., torus)
        self.cell = ParallelMLP(input_size+wave_input_kernel_size**2, 1, hidden_size, layers, inner_size)
        self.readout = nn.Linear(hidden_size, output_size, bias=False)
        self.dropout = nn.Dropout(dropout_p) if dropout_p else nn.Identity()
        self.waves = WaveTanh(wave_rho, wave_c, wave_omega, wave_padding)
        # Nonempty initial masks also make detach-last meaningful before a
        # previous task has supplied the fixed prefix masks.
        self.register_buffer("m", torch.cat([
            kwta(torch.rand(hidden_size), k) for _ in range(self.n_masks)]))
        self._reconfig = self.fast_reconfigure if reconfig_algorithm == "fast" else self.reconfigure

    # Exact existing FTN mask exploration and smoothing, without allocating an
    # extra StackFTN model or duplicating its trainable backbone.
    _step_mask = StackFTN._step_mask
    _smooth = StackFTN._smooth

    def set_stored_masks_ref(self, masks_ref):
        self._stored_masks_ref = masks_ref

    def _split(self, m):
        if m.ndim != 1 or m.numel() != self.hidden_size:
            raise ValueError(f"mask must be a flat vector of length {self.hidden_size}")
        return list(m.reshape(self.n_masks, self.layer_size)) if self.step_masks else [m]*self.n_stack

    def _cell_input(self, x, r):
        """Task input plus each neuron's centered primary-field patch (B,N,D+J²)."""
        radius = self.wave_input_kernel_size // 2
        field = r[:, :1]
        if radius:
            mode = "constant" if self.wave_padding == "zeros" else self.wave_padding
            field = F.pad(field, (radius,)*4, mode=mode)
        patches = F.unfold(field, kernel_size=self.wave_input_kernel_size).transpose(1, 2)
        return torch.cat((x[:, None, :].expand(-1, self.layer_size, -1), patches), dim=-1)

    def _forward_stack(self, x, masks, detach=False, trace=None):
        """`detach` freezes weights for search, not the reservoir Jacobian.

        trace, when provided, receives detached snapshots of the actual unroll.
        x is (B,D), reused at each step, or (B,steps,D) for explicit step inputs.
        wave_input_first_step_only zeros the task-input slice after step zero
        for either input shape, without zeroing the reservoir-input slice.
        There is never any state carried between forward calls or samples.
        """
        if not masks:
            raise ValueError("at least one processing step is required")
        if x.ndim not in (2, 3) or x.shape[-1] != self.input_size:
            raise ValueError("x must have shape (batch, input_size) or (batch, steps, input_size)")
        if x.ndim == 3 and x.shape[1] != len(masks):
            raise ValueError("step-input count must match the number of processing steps")
        r = x.new_zeros(x.shape[0], 2, self.square_dim, self.square_dim)
        if trace is not None:
            trace.append(dict(step=-1, substep=-1, phase="reset", r=r.detach().clone(),
                              mask=torch.zeros_like(masks[0])))
        for step, mask in enumerate(masks):
            if detach and not self.reconfig_bptt and step == len(masks)-1:
                r = r.detach()
            task_input = x if x.ndim == 2 else x[:, step]
            if self.wave_input_first_step_only and step > 0:
                task_input = torch.zeros_like(task_input)
            z = self.cell(self._cell_input(task_input, r), detach=detach)
            if not detach:
                z = self.dropout(z)
            z = z * mask.reshape(1, -1)
            # Hard admission gate: rejected values have zero injection gradient;
            # admitted values retain their amplitude. Bypass at zero to preserve
            # the original STE mask gradients at inactive (zero-valued) units.
            admitted = z if self.wave_input_threshold == 0 else torch.where(
                z.abs() > self.wave_input_threshold, z, torch.zeros_like(z))
            injection = admitted.reshape(-1, 1, self.square_dim, self.square_dim)*self.wave_input_scale
            r = r + torch.cat((injection, torch.zeros_like(injection)), dim=1)
            if trace is not None:
                trace.append(dict(step=step, substep=0, phase="inject", r=r.detach().clone(),
                                  mask=mask.detach().clone(), z=z.detach().clone()))
            for substep in range(self.wave_substeps):
                r = self.waves(r)
                if trace is not None:
                    trace.append(dict(step=step, substep=substep+1, phase="propagate",
                                      r=r.detach().clone(), mask=mask.detach().clone()))
        weight = self.readout.weight.detach() if detach else self.readout.weight
        return F.linear(z, weight)

    def forward(self, x, y=None, m=None):
        if m is None:
            if y is not None:
                self._reconfig(x, y)
            m = self.m
        return self._forward_stack(x, self._split(m))

    @torch.no_grad()
    def trace(self, x, m=None):
        """Trace real injection/wave substeps, with dropout disabled; restore mode."""
        training = self.training
        self.eval()
        frames = []
        try:
            output = self._forward_stack(x, self._split(self.m if m is None else m), trace=frames)
        finally:
            self.train(training)
        return output, frames

    def loss(self, pred, y):
        return task_loss(pred, y, self.classification, self.binary)

    @torch.enable_grad()
    def _search_simultaneous(self, x, y, anneal, n_steps=None):
        steps = self.n_stack if n_steps is None else int(n_steps)
        if steps < 1:
            raise ValueError("steps must be positive")
        n = min(self.reconfig_sample_size, len(x))
        x, y = x[:n].detach(), y[:n].detach()
        count = steps if self.step_masks else 1
        reference = self._overlap_reference(count * self.layer_size)
        # Detach-last cannot learn prefix masks; preserve the currently held
        # prefix rather than randomly resampling it on every reconfiguration.
        fixed = [m.detach().clone() for m in self._split(self.m)[:steps]]
        while len(fixed) < steps:
            fixed.append(kwta(torch.rand_like(self.m[:self.layer_size]), self.k))
        search_indices = list(range(count)) if self.reconfig_bptt or not self.step_masks else [steps-1]
        logits = [(torch.randn(self.layer_size, device=x.device, dtype=x.dtype)*self.reconfig_init_sd)
                  .requires_grad_() for _ in search_indices]
        opt = make_reconfig_optimizer(logits, lr=self.reconfig_lr, name=self.reconfig_optimizer)

        def expand(values):
            if not self.step_masks:
                return [values[0]]*steps
            masks = list(fixed)
            for index, value in zip(search_indices, values):
                masks[index] = value
            return masks

        def hard_masks():
            return expand([kwta(lg.detach(), self.k) for lg in logits])

        def packed_logits():
            if not self.step_masks:
                return logits[0]
            values = [torch.zeros_like(fixed[0]) for _ in range(steps)]
            for index, value in zip(search_indices, logits):
                values[index] = value
            return torch.cat(values)

        def score(masks):
            packed = packed_logits()
            return self._with_overlap(self._reconfig_task_loss(self._forward_stack(x, masks, detach=True), y), packed, reference).item()

        do_match = (self.match_stored and steps == self.n_stack
                    and self._stored_masks_ref is not None and len(self._stored_masks_ref)>0)
        stored = torch.stack(list(self._stored_masks_ref)).to(x) if do_match else None

        def recall(masks):
            # Tied recall restores a whole stored solution. Untied recall only
            # replaces the optimized masks, holding the prefix fixed in detach mode.
            packed = torch.cat(masks) if self.step_masks else masks[0]
            if self.step_masks and not self.reconfig_bptt:
                pool = stored[:, -self.layer_size:] if self.match_tie_steps else stored.reshape(-1,self.layer_size)
                result = list(masks)
                def last_loss(candidate):
                    return score(result[:-1]+[candidate])
                result[-1] = best_stored(result[-1],pool,self.match_stored_topk,last_loss)
                return result
            if self.match_tie_steps or not self.step_masks:
                return self._split(best_stored(packed, stored, self.match_stored_topk,
                                               lambda cand: score(self._split(cand))))
            pool = stored.reshape(-1, self.layer_size)
            result = list(masks)
            for index in search_indices:
                def candidate_loss(candidate):
                    trial = list(result)
                    trial[index] = candidate
                    return score(trial)
                result[index] = best_stored(result[index], pool, self.match_stored_topk, candidate_loss)
            return result

        with torch.no_grad():
            best = hard_masks()
            best_loss = score(best)
        swapped = False
        for iteration in range(self.reconfig_steps):
            masks = expand([self._step_mask(lg, iteration, anneal) for lg in logits])
            packed = packed_logits()
            loss = self._with_overlap(self._reconfig_task_loss(self._forward_stack(x, masks, detach=True), y), packed, reference)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            with torch.no_grad():
                # Evaluate the hard candidate actually saved, including for fast
                # searches; a soft-mask loss is not its hard-mask performance.
                candidate = hard_masks()
                candidate_loss = score(candidate)
                if candidate_loss < best_loss:
                    best, best_loss = candidate, candidate_loss
                if do_match and not swapped and iteration+1 >= self.match_stored_after:
                    best = recall(best)
                    best_loss = score(best)
                    swapped = True
        with torch.no_grad():
            if do_match and not swapped:
                best = recall(best)
            if not self.step_masks:
                return [self._smooth(best[0])]*steps
            return [self._smooth(m) if i in search_indices else m for i,m in enumerate(best)]

    def _configure(self, x, y, anneal):
        masks = self._search_simultaneous(x, y, anneal)
        self.m.copy_(torch.cat(masks) if self.step_masks else masks[0])
        return self.m

    def reconfigure(self, x, y):
        return self._configure(x, y, True)

    def fast_reconfigure(self, x, y):
        return self._configure(x, y, False)

    def forward_k(self, x, y=None, k_steps=None):
        steps = self.n_stack if k_steps is None else int(k_steps)
        if not 1 <= steps <= self.n_stack:
            raise ValueError(f"k_steps must be in [1, {self.n_stack}]")
        masks = (self._search_simultaneous(x, y, self.reconfig_algorithm!="fast", steps)
                 if y is not None else self._split(self.m)[:steps])
        return self._forward_stack(x, masks)
