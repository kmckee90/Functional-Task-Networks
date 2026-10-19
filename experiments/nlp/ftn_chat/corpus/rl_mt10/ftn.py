"""FTN backbone + AnnealedSTE mask configurer + mask-overlap-filtered replay.

Ported and adapted from experiments/func_net_kt for use as a DDPG actor/critic
backbone. Each network keeps its own ParallelMLP bank and its own per-task mask.

Forward pass (NeuroModel-style):
    z = ParallelMLP(x)             # (B, H)  -- H independent small MLPs
    z = Dropout(z)
    y = Linear(z * m)              # (B, out) with binary top-k mask m

The actor wraps this with `tanh` on the output, the critic uses `concat(s, a)`
as input and produces a scalar Q.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# ParallelMLP -- bank of H independent small MLPs (port of func_net_kt models.py)
# ---------------------------------------------------------------------------


class ParallelMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        layers: int,
        inner_size: int,
    ):
        super().__init__()
        self.n_layers = layers
        self.inner_size = inner_size
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
        s = 6.0
        s_b = 0.1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            fan_in = sizes[i]
            fan_out = sizes[i + 1]
            if i < self.n_layers:
                bound = math.sqrt(s / fan_in)
            else:
                bound = math.sqrt(s / (fan_in + fan_out))
            nn.init.uniform_(w, -bound, bound)
            nn.init.uniform_(b, -s_b / math.sqrt(fan_in), s_b / math.sqrt(fan_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        h = x.unsqueeze(-2)  # (B, 1, D) -- broadcasts over H
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = torch.einsum("bki,kij->bkj", h, W) + b
            if i < self.n_layers:
                h = torch.relu(h)
        return h.squeeze(-1)  # (B, H)


# ---------------------------------------------------------------------------
# Actor + Critic
# ---------------------------------------------------------------------------


class FTNActor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        H: int,
        layers: int,
        inner_size: int,
        dropout_p: float,
    ):
        super().__init__()
        self.H = H
        self.parallel = ParallelMLP(obs_dim, 1, H, layers, inner_size)
        self.linear = nn.Linear(H, act_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = self.parallel(obs)  # (B, H)
        z = self.dropout(z)
        return torch.tanh(self.linear(z * mask.reshape(1, -1)))


class FTNCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        H: int,
        layers: int,
        inner_size: int,
        dropout_p: float,
    ):
        super().__init__()
        self.H = H
        self.parallel = ParallelMLP(obs_dim + act_dim, 1, H, layers, inner_size)
        self.linear = nn.Linear(H, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self, obs: torch.Tensor, act: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        z = self.parallel(x)
        z = self.dropout(z)
        return self.linear(z * mask.reshape(1, -1)).squeeze(-1)


# ---------------------------------------------------------------------------
# KWTA / AnnealedSTE configurer
# ---------------------------------------------------------------------------


def kwta(x: torch.Tensor, k: int) -> torch.Tensor:
    """Binary top-k mask over a 1D tensor."""
    idx = torch.topk(x, k).indices
    out = torch.zeros_like(x)
    out[idx] = 1.0
    return out


# ---------------------------------------------------------------------------
# Square-grid layout + lateral spatial smoothing
# (ported from func_net_spatial/masks.py: mask_to_grid / mask_to_vec / SmoothKWTA)
# ---------------------------------------------------------------------------


def mask_to_grid(m: torch.Tensor, block_dim: int) -> torch.Tensor:
    """Reshape a flat mask into 2D by laying out block_dim x block_dim blocks
    side by side. For a single block this is reshape(block_dim, block_dim); for
    n_blocks blocks it produces (block_dim, n_blocks * block_dim). H must equal
    n_blocks * block_dim**2."""
    block_size = block_dim * block_dim
    blocks = m.split(block_size)
    return torch.cat([b.reshape(block_dim, block_dim) for b in blocks], dim=-1)


def mask_to_vec(grid: torch.Tensor, block_dim: int) -> torch.Tensor:
    """Inverse of mask_to_grid: split a (block_dim, n_blocks*block_dim) grid into
    block_dim x block_dim blocks and flatten each back into the flat mask."""
    blocks = grid.split(block_dim, dim=-1)
    return torch.cat([b.flatten() for b in blocks])


class LateralSmoother:
    """Lateral spatial-smoothing step from func_net_spatial's SmoothKWTA.

    Lays the flat k-hot mask onto a square grid (square_dim x square_dim blocks),
    runs `lateral_steps` of uniform-kernel convolution mixed back into the mask by
    a Polyak coefficient, then re-sparsifies with top-k. Spatially clusters the
    active units so a task's mask becomes a contiguous blob rather than a random
    scatter. Smoothing is a no-op (returns the input mask untouched) when
    `lateral_steps <= 0` or `lateral_polyak >= 1.0`."""

    def __init__(
        self,
        hidden_size: int,
        k: int,
        square_dim: int = 16,
        kernel_size: int = 17,
        lateral_steps: int = 0,
        lateral_polyak: float = 1.0,
        lateral_noise_sd: float = 0.0,
        torus: bool = False,
    ):
        if lateral_steps > 0 and lateral_polyak < 1.0 and (
            hidden_size % (square_dim * square_dim) != 0
        ):
            raise ValueError(
                f"hidden_size={hidden_size} must be a multiple of square_dim**2="
                f"{square_dim * square_dim} for the square-grid layout."
            )
        self.hidden_size = hidden_size
        self.k = k
        self.square_dim = square_dim
        self.lateral_steps = int(lateral_steps)
        self.lateral_polyak = float(lateral_polyak)
        self.lateral_noise_sd = float(lateral_noise_sd)
        # Single-channel uniform averaging kernel (weights = 1; the post-step
        # top-k only cares about relative ordering, so normalization is moot).
        self.lateral = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
            padding_mode="circular" if torus else "zeros",
        ).requires_grad_(False)
        self.lateral.weight.data.fill_(1.0)

    @property
    def enabled(self) -> bool:
        return self.lateral_steps > 0 and self.lateral_polyak < 1.0

    def step_lateral(self, m: torch.Tensor) -> torch.Tensor:
        """One uniform-kernel smoothing pass over the square-grid layout."""
        g = mask_to_grid(m, self.square_dim).unsqueeze(0).unsqueeze(0)
        g = self.lateral(g).squeeze(0).squeeze(0)
        if self.lateral_noise_sd > 0:
            g = g + torch.randn_like(g) * self.lateral_noise_sd
        return mask_to_vec(g, self.square_dim)

    @torch.no_grad()
    def smooth(self, mask: torch.Tensor) -> torch.Tensor:
        """Polyak-mix `lateral_steps` smoothing passes into the (binary) mask and
        re-sparsify to top-k. Returns a fresh k-hot mask; identity when disabled."""
        if not self.enabled:
            return mask
        best = mask.float().clone()
        for _ in range(self.lateral_steps):
            m_lat = self.step_lateral(best)
            best = self.lateral_polyak * best + (1.0 - self.lateral_polyak) * m_lat
        return kwta(best, k=self.k)


def anneal_ste_mask(
    H: int,
    k: int,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    n_steps: int,
    lr: float,
    init_logits: torch.Tensor | None = None,
    noise_scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Run AnnealedSTE mask search.

    `loss_fn(soft_mask)` -- callable that returns a scalar tensor under the
    given soft (STE'd) mask. Caller is responsible for any precomputation
    (e.g. detached activations from the support batch).

    `init_logits` -- if given, warm-starts m_param from these values (cloned).
    Otherwise cold-starts from noise_scale * randn(H).

    Returns (best_hard_mask, final_logits, initial_loss, final_loss).
    """
    if init_logits is None:
        m_param = nn.Parameter(noise_scale * torch.randn(H))
    else:
        m_param = nn.Parameter(noise_scale * init_logits.detach().clone())
    opt = torch.optim.SGD([m_param], lr=lr)

    best: torch.Tensor | None = None
    lowest = float("inf")
    first: float | None = None
    denom = max(1.0, n_steps / 4)

    for step in range(n_steps):
        std = m_param.detach().std().item()
        scale = std if std > 1e-6 else 0.1
        tau = scale * math.exp(-step / denom)
        noise = torch.randn_like(m_param) * tau
        hard = kwta((m_param + noise).detach(), k=k)
        mask_ste = hard + m_param - m_param.detach()  # straight-through estimator

        loss = loss_fn(mask_ste)
        v = loss.item()
        if first is None:
            first = v
        if v < lowest:
            lowest = v
            best = hard.detach().clone()

        opt.zero_grad()
        loss.backward()
        opt.step()

    if best is None:
        best = kwta(m_param.detach(), k=k)
        lowest = float(loss_fn(best).item())
    return best, m_param.detach().clone(), float(first or lowest), float(lowest)


def anneal_ste_mask_batched(
    H: int,
    k: int,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    n_steps: int,
    lr: float,
    n_search: int,
    init_logits: torch.Tensor | None = None,
    noise_scale: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Batched AnnealedSTE: run `n_search` independent cold-started searches
    in a single backward graph.

    `loss_fn(soft_mask)` -- callable taking a (n_search, H) soft mask and
    returning a (n_search,) per-row loss tensor. The sum is what gets
    backpropped; per-row tracking of "best so far" is done internally so each
    row can keep its own best hard mask.
    """
    if init_logits is None:
        m_param = nn.Parameter(noise_scale * torch.randn(n_search, H))
    else:
        if init_logits.dim() == 1:
            base = init_logits.detach().clone().unsqueeze(0).expand(n_search, -1).contiguous()
        else:
            base = init_logits.detach().clone()
        m_param = nn.Parameter(noise_scale * base)
    opt = torch.optim.SGD([m_param], lr=lr)

    best_mask = torch.zeros(n_search, H)
    lowest = torch.full((n_search,), float("inf"))
    first_loss: torch.Tensor | None = None
    denom = max(1.0, n_steps / 4)

    for step in range(n_steps):
        std = m_param.detach().std(dim=-1, keepdim=True)  # (E, 1)
        scale = torch.where(std > 1e-6, std, torch.full_like(std, 0.1))
        tau = scale * math.exp(-step / denom)
        noise = torch.randn_like(m_param) * tau
        x = (m_param + noise).detach()
        _, top_idx = torch.topk(x, k, dim=-1)
        hard = torch.zeros_like(x)
        hard.scatter_(-1, top_idx, 1.0)
        mask_ste = hard + m_param - m_param.detach()

        losses = loss_fn(mask_ste)  # (E,)
        with torch.no_grad():
            losses_d = losses.detach()
            if first_loss is None:
                first_loss = losses_d.clone()
            improved = losses_d < lowest
            lowest = torch.where(improved, losses_d, lowest)
            best_mask = torch.where(improved.unsqueeze(-1), hard, best_mask)

        opt.zero_grad()
        losses.sum().backward()
        opt.step()

    # Single sync at the very end (vs per-step in the non-batched variant).
    return (
        best_mask,
        m_param.detach().clone(),
        float(first_loss.mean().item()) if first_loss is not None else float(lowest.mean().item()),
        float(lowest.mean().item()),
    )


# ---------------------------------------------------------------------------
# Mask-overlap-filtered replay buffer
# ---------------------------------------------------------------------------


@dataclass
class _Transition:
    obs: np.ndarray
    act: np.ndarray
    rew: float
    next_obs: np.ndarray
    done: float
    task_idx: int  # which training phase this transition came from


class MaskedReplayBuffer:
    """Stores (transition, m_a_at_collection, m_c_at_collection) and supports
    overlap-filtered sampling.

    Current-phase storage is a fixed-shape numpy ringbuffer (fast batch sample
    for the regular DDPG update). Past-phase transitions are stashed per-phase
    along with that phase's actor and critic masks; on-demand we materialize
    a filtered batch by enumerating past phases whose mask overlap falls in
    (rho_low, rho_high].
    """

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

        # Per-completed-phase archive of transitions + masks.
        # past[i] = dict(obs=..., act=..., rew=..., next_obs=..., done=...,
        #                m_a=tensor (H,), m_c=tensor (H,))
        self.past: list[dict] = []

    # ---- current-phase API (mirrors the simple ReplayBuffer in train.py) ----
    def add(self, obs, act, rew, next_obs, done):
        i = self.ptr
        self.obs[i] = obs
        self.act[i] = act
        self.rew[i] = rew
        self.next_obs[i] = next_obs
        self.done[i] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch):
        idx = np.random.randint(0, self.size, size=batch)
        return (
            torch.tensor(self.obs[idx]),
            torch.tensor(self.act[idx]),
            torch.tensor(self.rew[idx]),
            torch.tensor(self.next_obs[idx]),
            torch.tensor(self.done[idx]),
        )

    def clear_current(self):
        self.ptr = 0
        self.size = 0

    # ---- phase-archive API ----
    def archive_phase(self, m_a: torch.Tensor, m_c: torch.Tensor):
        """Copy current-phase buffer into the past archive, tagged with the
        actor and critic masks that were active during the phase."""
        if self.size == 0:
            return
        self.past.append(
            {
                "obs": self.obs[: self.size].copy(),
                "act": self.act[: self.size].copy(),
                "rew": self.rew[: self.size].copy(),
                "next_obs": self.next_obs[: self.size].copy(),
                "done": self.done[: self.size].copy(),
                # Masks stay on the default device so they can be fed directly to
                # the networks during filtered replay forward passes.
                "m_a": m_a.detach().clone(),
                "m_c": m_c.detach().clone(),
            }
        )

    @staticmethod
    def _overlap(m_cur: torch.Tensor, m_old: torch.Tensor) -> float:
        denom = float(m_cur.sum().clamp(min=1).item())
        return float((m_cur * m_old).sum().item()) / denom

    def sample_filtered(
        self,
        m_a_cur: torch.Tensor,
        m_c_cur: torch.Tensor,
        n: int,
        rho_low: float,
        rho_high: float,
    ):
        """Pick past phases whose actor OR critic mask overlap with the current
        masks within (rho_low, rho_high], then sample n transitions uniformly
        across the union of those phases. Returns (batch_tuple, m_a_old, m_c_old)
        where the masks are the per-phase masks the transitions were collected
        under. Returns None if no phases qualify."""
        if not self.past:
            return None
        candidates: list[int] = []
        for i, ph in enumerate(self.past):
            rho_a = self._overlap(m_a_cur, ph["m_a"])
            rho_c = self._overlap(m_c_cur, ph["m_c"])
            if rho_low < rho_a <= rho_high or rho_low < rho_c <= rho_high:
                candidates.append(i)
        if not candidates:
            return None
        # Sample one past phase, then n transitions from it. We use one phase
        # per replay call so the (m_a_old, m_c_old) pair is unambiguous.
        ph_idx = int(np.random.choice(candidates))
        ph = self.past[ph_idx]
        N = ph["obs"].shape[0]
        idx = np.random.randint(0, N, size=n)
        batch = (
            torch.tensor(ph["obs"][idx]),
            torch.tensor(ph["act"][idx]),
            torch.tensor(ph["rew"][idx]),
            torch.tensor(ph["next_obs"][idx]),
            torch.tensor(ph["done"][idx]),
        )
        return batch, ph["m_a"].clone(), ph["m_c"].clone()
