"""
ftn.py - Plain-kWTA Functional Task Network (bayesian subproject).

A deliberately pared-down, single-mask FTN. Compared to the full base/ FTN this
drops everything not needed here: no lateral smoothing (features or mask), no
dual mask, no Hopfield / match-stored recall, no neurogenesis, no architecture
switching. What remains is just:

  - a ParallelMLP backbone (hidden_size independent per-unit MLPs) + linear
    readout,
  - a persistent binary task mask self.m (K active units, kWTA),
  - the original independent per-batch reconfiguration (FTN), and
  - a Bayesian persistent-logit approximation (BayesFTN) that samples a hard
    kWTA mask and regularizes each posterior toward the preceding distribution.

The reconfiguration methods are short and self-contained on purpose - a new
reconfig protocol can be written by hand right here (and in the run loop in
run_experiments.py) without fighting any abstraction. All weights are detached
inside them, so gradients flow only into the mask logits and the caller never
needs a zero_grad().

Usage:
    pred = ftn(x, y)          # reconfigure the mask on this batch, then predict
    pred = ftn(x)             # predict with the current mask (holdout eval)
    pred = ftn(x, m=m_stored) # predict under an explicit mask (e.g. replay)
    ftn.reconfigure(x, y)     # reconfigure only; result lands in ftn.m
    ftn.cold_reconfigure(x, y) # stateless support inference used by evaluation
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from overlap_regularizer import OverlapRegularizer


# =============================================================================
# Mask utilities (module-level so viz.py can reuse them)
# =============================================================================


def kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    """K-winner-take-all: binary mask with 1s at the top-k values of x."""
    idx = torch.topk(x, k).indices
    out = torch.zeros_like(x)
    out[idx] = 1
    return out


def gumbel_kwta(
    logits: torch.Tensor, k: int, temperature: float = 1.0
) -> torch.Tensor:
    """Sample an exact-k mask from a Gumbel-perturbed logit ranking."""
    u = torch.rand_like(logits).clamp_(1e-7, 1 - 1e-7)
    scores = logits / temperature - torch.log(-torch.log(u))
    return kwta(scores, k)


def normalize_mask(m: torch.Tensor) -> torch.Tensor:
    """Normalize a tensor to [0, 1] (used by the RGB mask visualization)."""
    m = m - m.min()
    return m / m.max()


def mask_to_grid(m: torch.Tensor, block_dim: int) -> torch.Tensor:
    """Reshape a flat mask into 2D for visualization by arranging block_dim x
    block_dim blocks side by side. A single block (hidden_size == block_dim**2)
    is just reshape(block_dim, block_dim); a multi-block mask (e.g. the stacked /
    recurrent-residual FTN's n_stack per-step masks) yields
    (block_dim, n_blocks * block_dim)."""
    block_size = block_dim * block_dim
    blocks = m.split(block_size)
    return torch.cat([b.reshape(block_dim, block_dim) for b in blocks], dim=-1)


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
            if i < self.n_layers:
                bound = math.sqrt(6.0 / fan_in)          # Kaiming (ReLU hidden)
            else:
                bound = math.sqrt(6.0 / (fan_in + fan_out))  # Xavier (linear out)
            nn.init.uniform_(w, -bound, bound)
            nn.init.uniform_(b, -0.1 / math.sqrt(fan_in), 0.1 / math.sqrt(fan_in))

    def forward(self, x):
        h = x.unsqueeze(-2)  # (B, 1, D) - broadcasts over the K units
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.einsum("bki,kij->bkj", h, W) + b  # (B, K, out_i)
            if i < self.n_layers:  # ReLU on hidden layers only
                h = torch.relu(h)
        return torch.tanh(h.squeeze(-1))  # (B, K)


def task_loss(pred, y, classification: bool = True):
    """Single-label softmax cross-entropy on one-hot targets (classification),
    else MSE. Shared by training and the mask search."""
    if classification:
        target = torch.where(y == 1)[1]
        return F.cross_entropy(pred, target)
    return F.mse_loss(pred.squeeze(), y.squeeze())


# =============================================================================
# FTN
# =============================================================================


class FTN(OverlapRegularizer, nn.Module):
    """Plain-kWTA masked backbone + task mask + reconfiguration.

    forward(x) predicts with the current mask; forward(x, y) reconfigures the
    mask on that batch first; forward(x, m=...) uses an explicit mask.

    Args:
        input_size, output_size, hidden_size: dimensions.
        square_dim:   side length of the square mask grid (hidden_size ==
                      square_dim**2); used only for visualization.
        classification: cross-entropy (one-hot) vs MSE loss.
        layers, inner_size, dropout_p: ParallelMLP depth / width / dropout.
        k:            kWTA budget - number of active mask units.
        reconfig_steps, reconfig_lr, reconfig_sample_size: mask-search budget.
        reconfig_algorithm: "annealste" (default) or "fast".
        reconfig_noise_scale: multiplier on the annealed exploration noise
                      (annealste only; 0 = greedy kWTA, no exploration noise).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        square_dim: int,
        classification: bool = True,
        layers: int = 3,
        inner_size: int = 8,
        dropout_p: float = 0.0,
        k: int = 128,
        reconfig_steps: int = 30,
        reconfig_lr: float = 0.3,
        reconfig_sample_size: int = 2048,
        reconfig_algorithm: str = "annealste",
        reconfig_noise_scale: float = 1.0,
        overlap_mode: str = "bump",
        overlap_lambda: float = 0.0,
        overlap_threshold: float = 0.5,
        overlap_temperature: float = 0.2,
    ):
        super().__init__()
        self.configure_overlap(overlap_mode, overlap_lambda, overlap_threshold, overlap_temperature)
        self.hidden_size = hidden_size
        self.square_dim = square_dim
        self.classification = classification
        self.k = k
        self.reconfig_steps = reconfig_steps
        self.reconfig_lr = reconfig_lr
        self.reconfig_sample_size = reconfig_sample_size
        self.reconfig_algorithm = reconfig_algorithm
        self.reconfig_noise_scale = reconfig_noise_scale

        self.parallel = ParallelMLP(input_size, 1, hidden_size, layers, inner_size)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)

        # The persistent task mask (a binary (hidden_size,) buffer).
        self.register_buffer("m", torch.zeros(hidden_size))

        # forward()/callers dispatch reconfiguration through self._reconfig.
        self._reconfig = (
            self.fast_reconfigure
            if reconfig_algorithm == "fast"
            else self.reconfigure
        )

    # ---------------------------------------------------------------- forward

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone features before masking: (B, hidden_size)."""
        return self.dropout(self.parallel(x))

    def forward(self, x, y=None, m=None):
        """Predict. If y is given (and no explicit mask), reconfigure first."""
        if m is None:
            if y is not None:
                self._reconfig(x, y)
            m = self.m
        z = self.features(x)
        return self.linear(z * m.reshape(1, -1))

    def loss(self, pred, y):
        return task_loss(pred, y, self.classification)

    def get_reconfig_state(self):
        """Clone mutable mask-inference state for side-effect-free evaluation."""
        return {"m": self.m.clone()}

    def set_reconfig_state(self, state):
        """Restore a state returned by get_reconfig_state()."""
        self.m.copy_(state["m"])

    @property
    def continuous_reactive_updates(self):
        """Whether reactive mode should also reconfigure non-change batches."""
        return False

    def reconfigure_changepoint(self, x, y):
        """Base FTN has no separate posterior; use its configured search."""
        return self._reconfig(x, y)

    # ---------------------------------------------------------- reconfiguration

    @torch.enable_grad()
    def cold_reconfigure(
        self,
        x,
        y,
        steps: int = 30,
        lr: float = 0.3,
        sample_size=None,
        temperature: float = 1.0,
        restarts: int = 1,
        take_best: bool | None = None,
    ):
        """Stateless support-conditioned mask inference for evaluation.

        This routine is shared unchanged by FTN and BayesFTN. Every restart
        creates zero logits and a fresh optimizer, uses no KL or stored mask
        state, and returns the deterministic top-k mask with the lowest support
        loss. Gumbel top-k is used only to explore during optimization.
        """
        if steps < 1:
            raise ValueError("evaluation reconfiguration steps must be at least 1")
        if lr <= 0:
            raise ValueError("evaluation reconfiguration lr must be positive")
        if temperature <= 0:
            raise ValueError("evaluation temperature must be positive")
        if restarts < 1:
            raise ValueError("evaluation restarts must be at least 1")
        if sample_size is not None and sample_size < 1:
            raise ValueError("evaluation sample_size must be positive or None")
        if take_best is None:
            take_best = getattr(self, "bayes_reconfig_take_best", False)

        n = x.shape[0] if sample_size is None else min(sample_size, x.shape[0])
        x, y = x[:n], y[:n]
        z = self.parallel(x).detach()
        w = self.linear.weight.detach()

        best_mask, best_loss = None, float("inf")
        for _ in range(restarts):
            logits = nn.Parameter(torch.zeros_like(self.m))
            opt = torch.optim.Adam([logits], lr=lr)
            for _ in range(steps):
                soft = torch.sigmoid(logits / temperature)
                hard = gumbel_kwta(logits, self.k, temperature)
                mask = hard + soft - soft.detach()
                loss = self.loss(F.linear(z * mask, w), y)
                if take_best and loss.item() < best_loss:
                    best_mask = hard.detach().clone()
                    best_loss = loss.item()
                opt.zero_grad()
                loss.backward()
                opt.step()

            candidate = kwta(logits.detach(), self.k)
            with torch.no_grad():
                candidate_loss = self.loss(F.linear(z * candidate, w), y).item()
            if candidate_loss < best_loss:
                best_mask = candidate.clone()
                best_loss = candidate_loss

        self.m.copy_(best_mask)
        self.last_cold_reconfig_stats = {
            "support_loss": best_loss,
            "restarts": restarts,
        }
        return self.m

    @torch.enable_grad()
    def reconfigure(self, x, y):
        """Annealed straight-through kWTA reconfiguration; updates self.m.

        Optimize a real-valued mask logit with Adam, where the forward mask is
        kwta(logits + annealed_noise) with a sigmoid straight-through estimator
        for gradients. The backbone is detached, so gradients flow only into the
        mask logits. Keeps the lowest-loss mask seen.
        """
        n = min(self.reconfig_sample_size, x.shape[0])
        x, y = x[:n], y[:n]
        z = self.parallel(x).detach()      # (n, hidden_size) backbone features
        w = self.linear.weight.detach()    # readout weights

        logits = nn.Parameter(torch.randn_like(self.m) * 1e-5)
        opt = torch.optim.Adam([logits], lr=self.reconfig_lr)

        best, best_loss = None, torch.inf
        reference = self._overlap_reference()
        for i in range(self.reconfig_steps):
            sm = torch.sigmoid(logits)
            tau = logits.detach().std() * math.exp(-i / (self.reconfig_steps / 4))
            noise = torch.randn_like(logits) * tau * self.reconfig_noise_scale
            mask = kwta(logits + noise, self.k) + sm - sm.detach()  # kWTA + STE
            # mask = kwta(logits + noise, self.k) + logits - logits.detach()  # kWTA + STE

            pred = F.linear(z * mask, w)
            loss = self._with_overlap(self.loss(pred, y), mask, reference)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if loss.item() < best_loss:
                best, best_loss = mask.detach().clone(), loss.item()

        self.m.copy_(best)
        return self.m

    @torch.enable_grad()
    def fast_reconfigure(self, x, y):
        """Fast (RMSProp, no annealing/STE) reconfiguration; updates self.m.

        Relaxes the mask to sigmoid(logits) during the search and hard-kWTAs at
        the end. Backbone detached; gradients flow only into the mask logits.
        """
        n = min(self.reconfig_sample_size, x.shape[0])
        x, y = x[:n], y[:n]
        z = self.parallel(x).detach()
        w = self.linear.weight.detach()

        logits = nn.Parameter(torch.randn_like(self.m) * 1e-5)
        opt = torch.optim.RMSprop([logits], lr=self.reconfig_lr)
        reference = self._overlap_reference()
        for _ in range(self.reconfig_steps):
            mask = torch.sigmoid(logits)
            pred = F.linear(z * mask, w)
            loss = self._with_overlap(self.loss(pred, y), mask, reference)
            opt.zero_grad()
            loss.backward()
            opt.step()

        self.m.copy_(kwta(logits, self.k))
        return self.m


class BayesFTN(FTN):
    """FTN with a persistent approximate posterior over kWTA masks.

    ``mask_logits`` are the variational state.  A reconfiguration starts from a
    detached copy of the preceding logits (the prior), takes a small number of
    support-loss steps, and penalizes movement with a factorized-Bernoulli KL.
    The forward pass is always a hard k-of-H mask sampled by Gumbel top-k; the
    sigmoid mask is used only as a straight-through gradient surrogate.

    The Bernoulli KL is a tractable proxy: conditioning independent Bernoulli
    variables on exactly k active entries has an expensive normalizer, while
    Gumbel top-k enforces the budget exactly in every actual forward pass.
    """

    def __init__(
        self,
        *args,
        bayes_reconfig_steps: int = 3,
        bayes_reconfig_lr: float = 0.3,
        bayes_kl_weight: float = 1.0,
        bayes_temperature: float = 1.0,
        bayes_prior_decay: float = 1.0,
        bayes_stochastic_mask: bool = True,
        bayes_reconfig_take_best: bool = False,
        bayes_changepoint_reconfig_steps: int = 30,
        bayes_changepoint_reconfig_lr: float = 0.3,
        bayes_changepoint_reconfig_sample_size=None,
        bayes_changepoint_kl_weight: float = 0.0,
        bayes_changepoint_temperature: float = 1.0,
        bayes_changepoint_prior_decay: float = 0.0,
        bayes_changepoint_stochastic_mask: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if bayes_reconfig_steps < 1:
            raise ValueError("bayes_reconfig_steps must be at least 1")
        if bayes_temperature <= 0:
            raise ValueError("bayes_temperature must be positive")
        if bayes_kl_weight < 0:
            raise ValueError("bayes_kl_weight must be non-negative")
        if not 0 <= bayes_prior_decay <= 1:
            raise ValueError("bayes_prior_decay must be in [0, 1]")
        if bayes_changepoint_reconfig_steps < 1:
            raise ValueError("bayes_changepoint_reconfig_steps must be at least 1")
        if bayes_changepoint_reconfig_lr <= 0:
            raise ValueError("bayes_changepoint_reconfig_lr must be positive")
        if (
            bayes_changepoint_reconfig_sample_size is not None
            and bayes_changepoint_reconfig_sample_size < 1
        ):
            raise ValueError(
                "bayes_changepoint_reconfig_sample_size must be positive or None"
            )
        if bayes_changepoint_kl_weight < 0:
            raise ValueError("bayes_changepoint_kl_weight must be non-negative")
        if bayes_changepoint_temperature <= 0:
            raise ValueError("bayes_changepoint_temperature must be positive")
        if not 0 <= bayes_changepoint_prior_decay <= 1:
            raise ValueError("bayes_changepoint_prior_decay must be in [0, 1]")

        self.bayes_reconfig_steps = bayes_reconfig_steps
        self.bayes_reconfig_lr = bayes_reconfig_lr
        self.bayes_kl_weight = bayes_kl_weight
        self.bayes_temperature = bayes_temperature
        self.bayes_prior_decay = bayes_prior_decay
        self.bayes_stochastic_mask = bayes_stochastic_mask
        self.bayes_reconfig_take_best = bayes_reconfig_take_best
        self.bayes_changepoint_reconfig_steps = bayes_changepoint_reconfig_steps
        self.bayes_changepoint_reconfig_lr = bayes_changepoint_reconfig_lr
        self.bayes_changepoint_reconfig_sample_size = (
            bayes_changepoint_reconfig_sample_size
        )
        self.bayes_changepoint_kl_weight = bayes_changepoint_kl_weight
        self.bayes_changepoint_temperature = bayes_changepoint_temperature
        self.bayes_changepoint_prior_decay = bayes_changepoint_prior_decay
        self.bayes_changepoint_stochastic_mask = bayes_changepoint_stochastic_mask

        # A buffer, not a Parameter: the weight optimizer must never update the
        # variational state. Reconfiguration updates it explicitly.
        self.register_buffer("mask_logits", torch.zeros(self.hidden_size))
        self._reconfig = self.reconfigure
        self.last_reconfig_stats = {}

        self.sample_type = "gumbel" #vs "annealste"

    @property
    def continuous_reactive_updates(self):
        """Stable batches keep making cheap posterior updates in reactive mode."""
        return True

    def get_reconfig_state(self):
        state = super().get_reconfig_state()
        state["mask_logits"] = self.mask_logits.clone()
        return state

    def set_reconfig_state(self, state):
        super().set_reconfig_state(state)
        self.mask_logits.copy_(state["mask_logits"])

    def _sample_mask(self, logits, temperature, stochastic):
        """Sample exactly k entries from the logit-induced ranking."""
        if stochastic:
            return gumbel_kwta(logits, self.k, temperature)
        return kwta(logits, self.k)

    def _kl_to_prior(self, logits, prior_logits, temperature):
        """Mean factorized-Bernoulli KL(q_logits || q_prior)."""
        eps = torch.finfo(logits.dtype).eps
        q = torch.sigmoid(logits / temperature).clamp(eps, 1 - eps)
        p = torch.sigmoid(prior_logits / temperature).clamp(eps, 1 - eps)
        kl = q * (q.log() - p.log())
        kl = kl + (1 - q) * ((1 - q).log() - (1 - p ).log())
        return kl.mean()

    @torch.enable_grad()
    def _posterior_update(
        self,
        x,
        y,
        *,
        steps,
        lr,
        sample_size,
        kl_weight,
        temperature,
        prior_decay,
        stochastic_mask,
        changepoint,
    ):
        """Run one persistent posterior update under an explicit parameter set."""
        n = x.shape[0] if sample_size is None else min(sample_size, x.shape[0])
        x, y = x[:n], y[:n]
        z = self.parallel(x).detach()
        w = self.linear.weight.detach()

        # Diffusion toward zero logits is optional; rho=1 is strict persistence.
        prior_logits = (prior_decay * self.mask_logits).detach().clone()
        reference = self._overlap_reference()
        logits = nn.Parameter(prior_logits.clone())
        opt = torch.optim.Adam([logits], lr=lr)

        task_value = kl_value = objective_value = float("nan")
        best_loss, best_mask, best_logits = float("inf"), None, None
        best_kl = best_objective = float("nan")
        best_selection = float("inf")
        overlap_value = best_overlap = 0.0
        for i in range(steps):

            if self.sample_type == "annealste":
                sm = torch.sigmoid(logits)
                tau = logits.detach().std() * math.exp(-i / (self.reconfig_steps / 4))
                noise = temperature*torch.randn_like(logits) * tau * self.reconfig_noise_scale
                mask = kwta(logits + noise, self.k) + sm - sm.detach()  # kWTA + STE
                # mask = kwta(logits + noise, self.k) + logits - logits.detach()  # kWTA + STE
            else:
                soft = torch.sigmoid(logits / temperature)
                anneal = math.exp(-i/max((steps-1)/3, 10))
                hard = self._sample_mask(logits, temperature*anneal, stochastic_mask)
                mask = hard + soft - soft.detach()

            pred = F.linear(z * mask, w)
            task = self.loss(pred, y)
            kl = self._kl_to_prior(logits, prior_logits, temperature)
            objective = task + kl_weight * kl
            penalty = self._overlap_penalty(mask, reference)
            objective = self._with_overlap(objective, mask, reference)
            selection = self._with_overlap(task, mask, reference)

            if self.bayes_reconfig_take_best and selection.item() < best_selection:
                best_selection = selection.item()
                best_overlap = penalty.detach().item()
                best_loss = task.item()
                best_mask = mask.detach().clone()
                best_logits = logits.detach().clone()
                best_kl = kl.item()
                best_objective = objective.item()

            opt.zero_grad()
            objective.backward()
            opt.step()
            task_value = task.detach().item()
            kl_value = kl.detach().item()
            objective_value = objective.detach().item()
            overlap_value = penalty.detach().item()

        if self.bayes_reconfig_take_best:
            self.mask_logits.copy_(best_logits)
            self.m.copy_(best_mask)
            task_value = best_loss
            kl_value = best_kl
            objective_value = best_objective
            overlap_value = best_overlap
        else:
            self.mask_logits.copy_(logits.detach())
            self.m.copy_(
                self._sample_mask(self.mask_logits, temperature, stochastic_mask)
            )
        self.last_reconfig_stats = {
            "changepoint": changepoint,
            "task_loss": task_value,
            "kl": kl_value,
            "objective": objective_value,
            "overlap_penalty": overlap_value,
            "prior_decay": prior_decay,
            "steps": steps,
        }
        return self.m

    def reconfigure(self, x, y):
        """Cheap persistent posterior update for an ordinary stable batch."""
        return self._posterior_update(
            x,
            y,
            steps=self.bayes_reconfig_steps,
            lr=self.bayes_reconfig_lr,
            sample_size=self.reconfig_sample_size,
            kl_weight=self.bayes_kl_weight,
            temperature=self.bayes_temperature,
            prior_decay=self.bayes_prior_decay,
            stochastic_mask=self.bayes_stochastic_mask,
            changepoint=False,
        )

    def reconfigure_changepoint(self, x, y):
        """Reset/refit the posterior using change-point-specific parameters."""
        return self._posterior_update(
            x,
            y,
            steps=self.bayes_changepoint_reconfig_steps,
            lr=self.bayes_changepoint_reconfig_lr,
            sample_size=self.bayes_changepoint_reconfig_sample_size,
            kl_weight=self.bayes_changepoint_kl_weight,
            temperature=self.bayes_changepoint_temperature,
            prior_decay=self.bayes_changepoint_prior_decay,
            stochastic_mask=self.bayes_changepoint_stochastic_mask,
            changepoint=True,
        )
