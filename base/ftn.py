"""
ftn.py - Self-contained Functional Task Network (FTN).

An FTN is a masked network that owns its task mask and the loop that
reconfigures it. Everything mask-related lives on this class:

  - backbone: parallel per-unit MLPs (ParallelMLP) + linear readout,
  - mask state: `self.m`, a persistent (hidden_size,) binary buffer,
  - reconfiguration: annealed straight-through-estimator k-winner-take-all
    (the base/masks.KWTA algorithm), with optional lateral smoothing of the
    mask (the base/masks.SmoothKWTA behavior). mask_lateral_steps=0 recovers
    plain KWTA.

Usage:
    pred = ftn(x, y)          # reconfigure mask on this batch, then predict
    pred = ftn(x)             # predict with the current mask (holdout eval)
    pred = ftn(x, m=m_stored) # predict with an explicit mask (e.g. replay)
    ftn.reconfigure(x, y)     # reconfigure only; result lands in ftn.m

The optimizer for the weights stays outside the class. reconfigure() detaches
all weights, so it never leaves gradients on FTN parameters and never needs a
zero_grad() from the caller.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Mask utilities (module-level so visualization code can reuse them)
# =============================================================================


def kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    """K-winner-take-all: binary mask with 1s at the top-k values."""
    idx = torch.topk(x, k).indices
    out = torch.zeros_like(x)
    out[idx] = 1
    return out


def closest_stored(best: torch.Tensor, stored: torch.Tensor) -> torch.Tensor:
    """The stored mask most similar to `best` (largest overlap / intersection).

    `best` is a flat (H,) mask; `stored` is (N, H). Returns a clone of the
    highest-overlap row. Used by the match-stored annealste reconfiguration
    option to snap the running best onto a previous task's mask.
    """
    overlap = (stored * best.reshape(1, -1)).sum(-1)
    return stored[overlap.argmax()].clone()


def best_stored(
    ref: torch.Tensor,
    stored: torch.Tensor,
    topk: int = 1,
    loss_fn=None,
) -> torch.Tensor:
    """Recall a stored mask to snap the running best onto (Hopfield-style).

    `ref` is a flat mask; `stored` is (N, H). Candidates are the `topk` stored
    masks with highest overlap with `ref`. With topk <= 1 (or no loss_fn) this
    is the single closest match, identical to `closest_stored`. Otherwise each
    of the top-k candidates is scored by `loss_fn(candidate_row) -> float` and
    the lowest-loss (best-performing) one is returned. Returns a clone of the
    chosen row.
    """
    overlap = (stored * ref.reshape(1, -1)).sum(-1)
    topk = max(1, min(int(topk), stored.shape[0]))
    if topk == 1 or loss_fn is None:
        return stored[overlap.argmax()].clone()
    idx = torch.topk(overlap, topk).indices
    losses = torch.tensor([float(loss_fn(stored[j])) for j in idx])
    return stored[idx[int(losses.argmin())]].clone()


def normalize_mask(m: torch.Tensor) -> torch.Tensor:
    """Normalize mask to [0, 1] range."""
    m = m - m.min()
    m = m / m.max()
    return m


def mask_to_grid(m: torch.Tensor, block_dim: int) -> torch.Tensor:
    """
    Reshape a flat mask into 2D by arranging block_dim x block_dim blocks
    side by side. For a single block this is just reshape(block_dim, block_dim).
    For multiple blocks (e.g. after neurogenesis) it produces
    (block_dim, n_blocks * block_dim).
    """
    block_size = block_dim * block_dim
    blocks = m.split(block_size)
    return torch.cat([b.reshape(block_dim, block_dim) for b in blocks], dim=-1)


def mask_to_vec(grid: torch.Tensor, block_dim: int) -> torch.Tensor:
    """Inverse of mask_to_grid."""
    blocks = grid.split(block_dim, dim=-1)
    return torch.cat([b.flatten() for b in blocks])


def _uniform_conv(kernel_size: int, fill: float, torus: bool) -> nn.Conv2d:
    """Single-channel, frozen, uniform 2D convolution used for lateral dynamics."""
    conv = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        bias=False,
        padding_mode="circular" if torus else "zeros",
    ).requires_grad_(False)
    conv.weight.data.fill_(fill)
    return conv


# =============================================================================
# Backbone
# =============================================================================


class ParallelMLP(nn.Module):
    """hidden_size independent small MLPs, each mapping input -> a scalar unit."""

    def __init__(self, input_size, output_size, hidden_size, layers, inner_size):
        super().__init__()
        self.n_layers = layers
        sizes = [input_size] + [inner_size] * layers + [output_size]
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
            nn.init.uniform_(b, -s_b / math.sqrt(fan_in), s_b / math.sqrt(fan_in))

    def forward(self, x, detach: bool = False):
        # detach=True runs the block with its weights detached from the graph,
        # so a caller (e.g. a stacked mask search) gets gradients on the masks
        # only, never on these parameters — mirroring the reconfig algorithms'
        # `self.parallel(x).detach()`, but usable mid-stack where the input
        # itself carries mask gradients.
        # Shared inputs (B,D), or per-neuron inputs (B,K,D) for local wave patches.
        h = x.unsqueeze(-2) if x.ndim == 2 else x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            if detach:
                W, b = W.detach(), b.detach()
            h = torch.einsum("bki,kij->bkj", h, W) + b  # (B, K, out_i)
            if i < self.n_layers:  # relu on hidden layers only, not the output
                h = torch.relu(h)
        out = h.squeeze(-1)
        out = torch.tanh(out)
        # out = torch.bernoulli(out_sm) + out_sm - out_sm.detach()
        return out  # (B, K)


class ParallelQuadMLP(nn.Module):
    """hidden_size independent per-neuron *quadratic* MLPs, each mapping the
    shared input -> a scalar unit, vectorized across units with einsum (the
    parallel-MLP trick applied to a Quad0 block).

    Each block projects its input h to quad_dim features q = tanh(W_sig h), forms
    the full quadratic feature set [h, q, outer(q, q)] (the linear term plus every
    pairwise product of the q features), then linearly maps that to the next
    block's input. `layers` inner blocks of width `inner_size` are stacked
    (blocks = layers + 1, matching ParallelMLP's depth convention); the final
    block maps to output_size. Every parameter's leading dim is hidden_size, so
    each unit is a private MLP evaluated in parallel — no weights shared across
    units. Mirrors ParallelMLP's interface: forward(x, detach) -> (B, hidden_size)
    in [0, 1], plus .weights / .biases lists for FTN.expand()."""

    def __init__(self, input_size, output_size, hidden_size, layers,
                 inner_size, quad_dim):
        super().__init__()
        self.n_layers = layers
        self.quad_dim = quad_dim
        sizes = [input_size] + [inner_size] * layers + [output_size]
        # Per block: a sigma projection (in_b -> quad_dim, feeding tanh) and an
        # output map over the quadratic feature vector [h, q, q⊗q] (length
        # in_b + quad_dim + quad_dim**2) -> out_b.
        self.sig_w = nn.ParameterList()
        self.sig_b = nn.ParameterList()
        self.out_w = nn.ParameterList()
        self.out_b = nn.ParameterList()
        for i in range(len(sizes) - 1):
            in_b, out_b = sizes[i], sizes[i + 1]
            z_b = in_b + quad_dim + quad_dim * quad_dim
            self.sig_w.append(nn.Parameter(torch.empty(hidden_size, in_b, quad_dim)))
            self.sig_b.append(nn.Parameter(torch.empty(hidden_size, quad_dim)))
            self.out_w.append(nn.Parameter(torch.empty(hidden_size, z_b, out_b)))
            self.out_b.append(nn.Parameter(torch.empty(hidden_size, out_b)))
        self._init_weights(sizes)

    def _init_weights(self, sizes):
        for i in range(len(self.sig_w)):
            in_b = sizes[i]
            z_b = self.out_w[i].shape[1]
            out_b = sizes[i + 1]
            s, s_b = 6.0, 0.1
            # sigma projection: Kaiming-uniform on the fan-in (feeds a tanh)
            bound = math.sqrt(s / in_b)
            nn.init.uniform_(self.sig_w[i], -bound, bound)
            nn.init.uniform_(self.sig_b[i], -s_b / math.sqrt(in_b), s_b / math.sqrt(in_b))
            # output map: Xavier-uniform over the quadratic feature fan-in
            bound = math.sqrt(s / (z_b + out_b))
            nn.init.uniform_(self.out_w[i], -bound, bound)
            nn.init.uniform_(self.out_b[i], -s_b / math.sqrt(z_b), s_b / math.sqrt(z_b))

    # expand() (neurogenesis) copies backbone params by leading (unit) dim; expose
    # every per-neuron tensor the same way ParallelMLP exposes .weights / .biases.
    @property
    def weights(self):
        return list(self.sig_w) + list(self.out_w)

    @property
    def biases(self):
        return list(self.sig_b) + list(self.out_b)

    def forward(self, x, detach: bool = False):
        # Broadcast the shared input across the K units -> (B, K, in), so the
        # first block is a per-unit einsum just like the deeper blocks (which
        # already carry a private per-unit state). expand() is a view; einsum
        # consumes it without materializing the copy.
        K = self.sig_w[0].shape[0]
        h = x.unsqueeze(-2).expand(x.shape[0], K, x.shape[-1])
        for i in range(len(self.sig_w)):
            sw, sb = self.sig_w[i], self.sig_b[i]
            ow, ob = self.out_w[i], self.out_b[i]
            if detach:
                sw, sb, ow, ob = sw.detach(), sb.detach(), ow.detach(), ob.detach()
            q = torch.tanh(torch.einsum("bki,kij->bkj", h, sw) + sb)   # (B,K,quad)
            q2 = torch.einsum("bkp,bkq->bkpq", q, q)                   # (B,K,quad,quad)
            q2 = q2.reshape(q2.shape[0], q2.shape[1], -1)              # (B,K,quad**2)
            z = torch.cat((h, q, q2), dim=-1)                         # (B,K,z_b)
            h = torch.einsum("bkz,kzo->bko", z, ow) + ob              # (B,K,out_b)
        out = h.squeeze(-1)          # (B, K)  (output_size == 1)
        return torch.sigmoid(out)


def task_loss(pred, y, classification: bool = False, binary: bool = False):
    """Task loss shared by all FTN variants (training and the mask search).

    binary=True:         per-cell binary cross-entropy with logits. Targets are
                         independent bits in {0, 1} of the same shape as pred
                         (e.g. the composition task's binary matrices); every
                         output unit is its own sigmoid classifier.
    classification=True: single-label softmax cross-entropy on one-hot targets
                         (exactly one 1 per row = pick one of output_size classes).
    otherwise:           mean squared error (scalar/vector regression).

    binary takes precedence over classification.
    """
    if binary:
        return F.binary_cross_entropy_with_logits(pred, y.float())
    if classification:
        target = torch.where(y == 1)[1]
        return F.cross_entropy(pred, target)
    return F.mse_loss(pred.squeeze(), y.squeeze())


# =============================================================================
# FTN
# =============================================================================


class FTN(nn.Module):
    """
    Functional Task Network: masked backbone + task mask + reconfiguration.

    forward(x) predicts with the current mask; forward(x, y) first
    reconfigures the mask on that batch (per-batch reconfiguration),
    forward(x, m=...) uses an explicit mask.

    Args:
        input_size:          Input feature dimension.
        output_size:         Output dimension (n classes or 1).
        hidden_size:         Number of maskable units (K of ParallelMLP).
        square_dim:          Side length of the square grid each mask block is
                             reshaped into for 2D lateral dynamics. hidden_size
                             must be a multiple of square_dim**2.
        classification:      Cross-entropy (one-hot targets) vs MSE, used for
                             both reconfiguration and self.loss().
        layers, inner_size:  ParallelMLP depth / inner width.
        dropout_p:           Dropout on backbone features (forward only).
        lateral_b, lateral_steps, lateral_kernel_size:
                             Optional lateral smoothing of backbone features;
                             lateral_steps=0 disables (default).
        k:                   kWTA budget - number of active mask units.
        reconfig_steps, reconfig_lr, reconfig_sample_size:
                             Mask-optimization budget per reconfiguration.
        mask_kernel_size, mask_lateral_steps, mask_lateral_polyak,
        mask_lateral_noise_sd:
                             SmoothKWTA lateral dynamics on the mask after the
                             gradient loop; mask_lateral_steps=0 = plain KWTA.
        torus:               Circular (wrap-around) padding for both lateral
                             convolutions.
        use_second_mask:     Enable the DualMask two-stage reconfiguration. The
                             first mask is reconfigured normally; then a second
                             mask is reconfigured against mask2*mask1 with mask1
                             frozen, and the effective mask becomes mask1*mask2.
        k2, reconfig_steps2, reconfig_lr2, reconfig_sample_size2,
        reconfig_algorithm2, mask_kernel_size2, mask_lateral_steps2,
        mask_lateral_polyak2, mask_lateral_noise_sd2, torus2:
                             The second mask's twin of every mask-1 knob above,
                             so the second mask can use a different algorithm
                             ("fast"/"annealste"), budget, K, lateral dynamics
                             and torus. Ignored when use_second_mask is False.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        square_dim: int,
        classification: bool = False,
        binary: bool = False,
        # Backbone
        layers: int = 3,
        inner_size: int = 8,
        backbone: str = "mlp",   # "mlp" (ParallelMLP) | "quad" (ParallelQuadMLP)
        quad_dim: int = 8,       # per-unit quad feature width (backbone == "quad")
        dropout_p: float = 0.2,
        lateral_b: float = 0.0,
        lateral_steps: int = 0,
        lateral_kernel_size: int = 3,
        # Mask / reconfiguration (first mask)
        k: int = 128,
        reconfig_steps: int = 30,
        reconfig_lr: float = 0.3,
        reconfig_sample_size: int = 2048,
        reconfig_algorithm = "annealste", #annealste (default) or fast
        reconfig_noise_scale: float = 1.0,
        mask_kernel_size: int = 3,
        mask_lateral_steps: int = 0,
        mask_lateral_polyak: float = 0.0,
        mask_lateral_noise_sd: float = 0.0,
        torus: bool = False,
        # Match-stored reconfiguration (annealste only)
        match_stored: bool = False,
        match_stored_after: int = 0,
        match_stored_topk: int = 1,
        # Second mask (DualMask); ignored when use_second_mask is False
        use_second_mask: bool = False,
        k2: int = 128,
        reconfig_steps2: int = 30,
        reconfig_lr2: float = 0.3,
        reconfig_sample_size2: int = 2048,
        reconfig_algorithm2 = "annealste",
        mask_kernel_size2: int = 3,
        mask_lateral_steps2: int = 0,
        mask_lateral_polyak2: float = 0.0,
        mask_lateral_noise_sd2: float = 0.0,
        torus2: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.square_dim = square_dim
        self.classification = classification
        self.binary = binary
        self.use_second_mask = use_second_mask

        self.k = k
        self.reconfig_steps = reconfig_steps
        self.reconfig_lr = reconfig_lr
        self.reconfig_sample_size = reconfig_sample_size
        self.reconfig_noise_scale = reconfig_noise_scale
        self.mask_lateral_steps = mask_lateral_steps
        self.mask_lateral_polyak = mask_lateral_polyak
        self.mask_lateral_noise_sd = mask_lateral_noise_sd

        # Match-stored reconfiguration option (see _annealste_alg). A reference to
        # the live replay / noise-probe mask buffer (a deque), wired in by
        # run_experiments; None disables regardless of match_stored.
        self.match_stored = match_stored
        self.match_stored_after = match_stored_after
        # Hopfield recall breadth: consider the top-k highest-overlap stored
        # masks and keep the best-performing one (by loss). 1 = closest match.
        self.match_stored_topk = match_stored_topk
        self._stored_masks_ref = None

        self.lateral_b = lateral_b
        self.lateral_steps = lateral_steps

        # Backbone: per-unit MLP (default) or per-unit quadratic MLP. Both expose
        # forward(x)->(B, hidden_size) plus .weights/.biases, so everything
        # downstream (mask, reconfiguration, readout, expand) is unchanged.
        if backbone == "quad":
            self.parallel = ParallelQuadMLP(
                input_size, 1, hidden_size, layers, inner_size, quad_dim
            )
        else:
            self.parallel = ParallelMLP(input_size, 1, hidden_size, layers, inner_size)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)
        # Feature smoothing (averaging kernel); no-op when lateral_steps == 0
        self.lateral = _uniform_conv(
            lateral_kernel_size, 1 / lateral_kernel_size**2, torus
        )
        # Mask smoothing (summing kernel) for SmoothKWTA lateral dynamics.
        # One conv per mask so each mask has its own kernel size / torus.
        self.mask_lateral = _uniform_conv(mask_kernel_size, 1.0, torus)
        self.mask_lateral2 = _uniform_conv(mask_kernel_size2, 1.0, torus2)

        # Current effective task mask (mask1*mask2 in dual mode, else mask1),
        # plus the two component masks (m2 stays zero in single-mask mode).
        self.register_buffer("m", torch.zeros(hidden_size))
        self.register_buffer("m1", torch.zeros(hidden_size))
        self.register_buffer("m2", torch.zeros(hidden_size))

        # Per-mask hyperparameter bundles consumed by the reconfig algorithms.
        self.spec1 = SimpleNamespace(
            k=k,
            steps=reconfig_steps,
            lr=reconfig_lr,
            sample_size=reconfig_sample_size,
            algorithm=reconfig_algorithm,
            lateral_conv=self.mask_lateral,
            lateral_steps=mask_lateral_steps,
            polyak=mask_lateral_polyak,
            noise_sd=mask_lateral_noise_sd,
        )
        self.spec2 = SimpleNamespace(
            k=k2,
            steps=reconfig_steps2,
            lr=reconfig_lr2,
            sample_size=reconfig_sample_size2,
            algorithm=reconfig_algorithm2,
            lateral_conv=self.mask_lateral2,
            lateral_steps=mask_lateral_steps2,
            polyak=mask_lateral_polyak2,
            noise_sd=mask_lateral_noise_sd2,
        )

        # Which reconfiguration routine forward()/callers dispatch through.
        if use_second_mask:
            self._reconfig = self.dual_reconfigure
        elif reconfig_algorithm == "fast":
            self._reconfig = self.fast_reconfigure
        else:
            self._reconfig = self.reconfigure

        # Kept for expand(): everything needed to rebuild this FTN
        self._config = dict(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            square_dim=square_dim,
            classification=classification,
            binary=binary,
            layers=layers,
            inner_size=inner_size,
            backbone=backbone,
            quad_dim=quad_dim,
            dropout_p=dropout_p,
            lateral_b=lateral_b,
            lateral_steps=lateral_steps,
            lateral_kernel_size=lateral_kernel_size,
            k=k,
            reconfig_steps=reconfig_steps,
            reconfig_lr=reconfig_lr,
            reconfig_sample_size=reconfig_sample_size,
            reconfig_algorithm=reconfig_algorithm,
            reconfig_noise_scale=reconfig_noise_scale,
            mask_kernel_size=mask_kernel_size,
            mask_lateral_steps=mask_lateral_steps,
            mask_lateral_polyak=mask_lateral_polyak,
            mask_lateral_noise_sd=mask_lateral_noise_sd,
            torus=torus,
            match_stored=match_stored,
            match_stored_after=match_stored_after,
            match_stored_topk=match_stored_topk,
            use_second_mask=use_second_mask,
            k2=k2,
            reconfig_steps2=reconfig_steps2,
            reconfig_lr2=reconfig_lr2,
            reconfig_sample_size2=reconfig_sample_size2,
            reconfig_algorithm2=reconfig_algorithm2,
            mask_kernel_size2=mask_kernel_size2,
            mask_lateral_steps2=mask_lateral_steps2,
            mask_lateral_polyak2=mask_lateral_polyak2,
            mask_lateral_noise_sd2=mask_lateral_noise_sd2,
            torus2=torus2,
        )



    # ---------------------------------------------------------------- forward

    def set_stored_masks_ref(self, masks_ref):
        """Wire in the live buffer of stored task masks (a deque, e.g.
        ReplayBuffer.masks / NoiseProbeBuffer.masks) used by the match-stored
        annealste option. run_experiments sets this; None disables it."""
        self._stored_masks_ref = masks_ref

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone features before masking: (B, hidden_size)."""
        z = self.parallel(x)
        return self.dropout(z)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = None, m: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Predict. If y is given (and no explicit mask), the mask is first
        reconfigured on this batch - per-batch reconfiguration lives here.

        Args:
            x: Inputs (B, input_size)
            y: Optional targets used only to reconfigure the mask
            m: Optional explicit mask overriding the internal one (no reconfig)

        Returns:
            Predictions (B, output_size)
        """
        if m is None:
            if y is not None:
                self._reconfig(x, y)
            m = self.m
        m = m.reshape(1, -1)
        z = self.features(x)
        # z = self._step_feature_lateral(z*m)*m # This is kind of useless.
        return self.linear(z*m)

    def loss(self, pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Task loss: per-cell BCE (binary), one-hot cross-entropy, or MSE."""
        return task_loss(pred, y, self.classification, self.binary)

    # ---------------------------------------------------------- reconfiguration

    def _run_alg(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        spec: SimpleNamespace,
        gate: torch.Tensor = None,
    ) -> torch.Tensor:
        """Dispatch to this spec's reconfiguration algorithm and return its
        (unmultiplied) kWTA mask. `gate` is an optional frozen mask that gates
        the features before selection (used to freeze mask1 while mask2 is
        reconfigured against mask2*mask1)."""
        if spec.algorithm == "fast":
            return self._fast_alg(x, y, spec, gate)
        return self._annealste_alg(x, y, spec, gate)

    @torch.enable_grad()
    def _annealste_alg(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        spec: SimpleNamespace,
        gate: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Annealed straight-through KWTA reconfiguration for one mask.

        Optimize a real-valued mask logit with Adam, where the forward mask is
        kwta(logits + noise) with an annealed noise temperature and a sigmoid
        STE for gradients. Keeps the best mask by loss, then (optionally)
        applies lateral smoothing + re-sparsifies. All weights are detached:
        gradients flow only into the mask logits. Returns the kWTA mask.
        """
        n = min(spec.sample_size, x.shape[0])
        x, y = x[:n], y[:n]

        m = nn.Parameter(torch.randn_like(self.m) * 0.00001)
        optimizer = torch.optim.Adam([m], lr=spec.lr)

        z = self.parallel(x).detach()
        w = self.linear.weight.detach()
        if gate is not None:
            z = z * gate.reshape(1, -1)

        # Match-stored option: after match_stored_after steps, snap the running
        # best to the closest previous-task mask in the buffer and let the rest
        # of the steps try to beat it. Gated to the primary mask (gate is None)
        # and to a populated buffer.
        do_match = (
            self.match_stored
            and gate is None
            and self._stored_masks_ref is not None
            and len(self._stored_masks_ref) > 0
        )
        stored = torch.stack(list(self._stored_masks_ref)) if do_match else None
        swapped = False

        def _cand_loss(cand: torch.Tensor) -> float:
            with torch.no_grad():
                return self.loss(F.linear(z * cand.reshape(1, -1), w), y).item()

        lowest_loss = torch.inf
        best = None
        for i in range(spec.steps):
            sm = torch.sigmoid(m)
            tau = m.detach().std() * math.exp(-i / (spec.steps / 4))
            noise = torch.randn_like(m) * tau * self.reconfig_noise_scale
            mask = kwta(m + noise, k=spec.k) + sm - sm.detach()

            y_pred = F.linear(z * mask, w)
            loss = self.loss(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if loss.item() < lowest_loss:
                best = mask.clone().detach()
                lowest_loss = loss.item()

            # Replace best with the closest stored mask, and make its loss the
            # bar the remaining steps must beat.
            if do_match and not swapped and (i + 1) >= self.match_stored_after:
                ref = best if best is not None else kwta(m.detach(), spec.k)
                m_stored = best_stored(ref, stored, self.match_stored_topk, _cand_loss)
                lowest_loss = _cand_loss(m_stored)
                best = m_stored
                swapped = True

        # match_stored_after >= steps never triggers in the loop: anchor to the
        # best stored mask now (end with a stored mask).
        if do_match and not swapped:
            ref = best if best is not None else kwta(m.detach(), spec.k)
            best = best_stored(ref, stored, self.match_stored_topk, _cand_loss)

        # Lateral dynamics (spatial smoothing) + re-sparsification.
        # With lateral_steps == 0 this leaves the KWTA mask unchanged.
        if spec.polyak < 1.0:
            for _ in range(spec.lateral_steps):
                m_lat = self._step_mask_lateral(best, spec.lateral_conv, spec.noise_sd)
                best = spec.polyak * best + (1 - spec.polyak) * m_lat
            best = kwta(best, k=spec.k)

        return best

    @torch.enable_grad()
    def _fast_alg(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        spec: SimpleNamespace,
        gate: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Fast (RMSprop, no annealing/STE) reconfiguration for one mask.

        All weights are detached: gradients flow only into the mask logits.
        Returns the kWTA mask.
        """
        n = min(spec.sample_size, x.shape[0])
        x, y = x[:n], y[:n]

        m = nn.Parameter(torch.randn_like(self.m) * 0.00001)
        optimizer = torch.optim.RMSprop([m], lr=spec.lr)

        z = self.parallel(x).detach()
        w = self.linear.weight.detach()
        if gate is not None:
            z = z * gate.reshape(1, -1)

        for i in range(spec.steps):
            y_pred = F.linear(z * torch.sigmoid(m), w)
            loss = self.loss(y_pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            # Lateral dynamics (spatial smoothing) + re-sparsification.
            # With lateral_steps == 0 this leaves the KWTA mask unchanged.
            if spec.polyak < 1.0:
                for _ in range(spec.lateral_steps):
                    m_lat = self._step_mask_lateral(m, spec.lateral_conv, spec.noise_sd)
                    m = spec.polyak * m + (1 - spec.polyak) * m_lat

        return kwta(m, k=spec.k)

    def reconfigure(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Single-mask annealed-STE reconfiguration; updates self.m in place."""
        best = self._annealste_alg(x, y, self.spec1)
        self.m1.copy_(best)
        self.m.copy_(best)
        return self.m

    # ---------------------------------------------------------- Fast reconfig

    def fast_reconfigure(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Single-mask fast reconfiguration; updates self.m in place."""
        best = self._fast_alg(x, y, self.spec1)
        self.m1.copy_(best)
        self.m.copy_(best)
        return self.m

    # ---------------------------------------------------------- Dual reconfig

    def dual_reconfigure(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Two-stage DualMask reconfiguration.

        Reconfigure the first mask normally (mask1), then reconfigure the
        second mask against mask2*mask1 with mask1 frozen, and set the
        effective mask to mask1*mask2. Each mask uses its own spec (algorithm,
        K, lateral dynamics, torus, ...). Updates self.m in place and returns
        it; self.m1/self.m2 hold the two components.
        """
        m1 = self._run_alg(x, y, self.spec1, gate=None)
        self.m1.copy_(m1)
        m2 = self._run_alg(x, y, self.spec2, gate=m1)
        self.m2.copy_(m2)
        self.m.copy_(m1 * m2)
        return self.m


    # ------------------------------------------------------------- neurogenesis

    def expand(self, new_hidden_size: int) -> "FTN":
        """
        Neurogenesis: return a larger FTN with this one's weights copied into
        the corresponding sub-tensors. New units are freshly initialized.
        """
        cfg = dict(self._config, hidden_size=new_hidden_size)
        new = FTN(**cfg)
        with torch.no_grad():
            for w_old, w_new in zip(self.parallel.weights, new.parallel.weights):
                w_new[: w_old.shape[0]] = w_old
            for b_old, b_new in zip(self.parallel.biases, new.parallel.biases):
                b_new[: b_old.shape[0]] = b_old
            new.linear.weight[:, : self.hidden_size] = self.linear.weight
            new.m[: self.hidden_size] = self.m
        return new

    # ----------------------------------------------------------------- lateral

    def _step_feature_lateral(self, h: torch.Tensor) -> torch.Tensor:
        """Optional lateral smoothing of backbone features (no-op by default)."""
        if self.lateral_steps == 0 or self.lateral_b == 1.0:
            return h
        hshape = h.shape
        h = h.reshape(-1, 1, self.square_dim, self.square_dim)
        for _ in range(self.lateral_steps):
            h = self.lateral_b * h + (1 - self.lateral_b) * self.lateral(h)
        return h.reshape(hshape)

    def _step_mask_lateral(
        self, m: torch.Tensor, conv: nn.Module = None, noise_sd: float = None
    ) -> torch.Tensor:
        """
        One lateral-smoothing step on the mask. Blocks of square_dim**2 units
        are smoothed independently, so each block (e.g. pre/post neurogenesis
        populations) keeps its own spatial topology. `conv`/`noise_sd` default
        to the first mask's smoothing conv / noise (the second mask passes its
        own via its spec).
        """
        if conv is None:
            conv = self.mask_lateral
        if noise_sd is None:
            noise_sd = self.mask_lateral_noise_sd
        g = mask_to_grid(m, self.square_dim).unsqueeze(0).unsqueeze(0)
        g = conv(g).squeeze(0).squeeze(0)
        if noise_sd > 0:
            g = g + torch.randn_like(g) * noise_sd
        return mask_to_vec(g, self.square_dim)
