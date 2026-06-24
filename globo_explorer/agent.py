"""Explorer RECODE agent for the globo vision environment.

Adapts the explorer agent (/explorer/agents/agent_recode.py, an RDDPG_Dual DDPG
agent with RECODE intrinsic reward) to:
  * a 3D first-person VISION environment, by replacing the 1D observation
    reservoir with the Fluid_1S conv2d reservoir + deep conv2d decoder, and
  * a DISCRETE action space (globo via craftium is key/mouse based), by using
    DQN instead of DDPG.

RECODE clusters/densities are computed over the IMAGES THEMSELVES (downscaled
pixels), not over any latent state of the model -- exactly as requested.

The reservoir (and the fixed feedback->channel projection) are frozen, so a
transition's reservoir feature is a stationary function of the trajectory; we
therefore store the reservoir hidden features directly in replay and train only
the conv decoder (a standard fixed-reservoir / trained-readout setup).
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Conv2dReservoir, DeepConvDecoder, FeedbackToChannel, downscale_embedding
from recode.recode import RECODE, adaptive_knn_reward, knn_reward


class ReplayBuffer:
    """Stores reservoir hidden features (frozen reservoir => features are stable)."""

    def __init__(self, capacity: int, device):
        self.capacity = capacity
        self.device = device
        self.z = None      # (N, C, h, w)  reservoir hidden at t
        self.z_next = None
        self.action = torch.zeros(capacity, dtype=torch.long, device=device)
        self.reward = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.done = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.size = 0
        self.ptr = 0

    def _lazy(self, z):
        if self.z is None:
            c, h, w = z.shape
            self.z = torch.zeros(self.capacity, c, h, w, device=self.device)
            self.z_next = torch.zeros(self.capacity, c, h, w, device=self.device)

    def add(self, z, action, reward, z_next, done):
        self._lazy(z)
        i = self.ptr
        self.z[i] = z
        self.z_next[i] = z_next
        self.action[i] = int(action)
        self.reward[i] = float(reward)
        self.done[i] = bool(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.z[idx], self.action[idx], self.reward[idx], self.z_next[idx], self.done[idx]


class GloboExplorerDQN:
    """Conv2d-reservoir RECODE DQN explorer."""

    def __init__(
        self,
        num_actions: int,
        in_h: int,
        in_w: int,
        *,
        reservoir_channels: int = 64,
        decoder_layer_size: int = 256,
        # RECODE over downscaled images
        recode_down_h: int = 16,
        recode_down_w: int = 32,
        recode_grayscale: bool = True,
        recode_memory_size: int = 2000,
        recode_k: int = 10,
        recode_knn_k: int = 10,
        num_novelty_bins: int = 8,
        # RL
        gamma: float = 0.99,
        lr: float = 3e-4,
        memory_length: int = 20000,
        minibatch_size: int = 64,
        target_update_tau: float = 0.01,
        epsilon: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20000,
        env_reward_coef: float = 0.0,
        exploration_coef: float = 1.0,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ):
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.in_h, self.in_w = in_h, in_w
        self.gamma = gamma
        self.minibatch_size = minibatch_size
        self.tau = target_update_tau
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.env_reward_coef = env_reward_coef
        self.exploration_coef = exploration_coef
        self.num_novelty_bins = num_novelty_bins
        self.recode_down = (recode_down_h, recode_down_w)
        self.recode_grayscale = recode_grayscale
        self.cur_step = 0

        # feedback vector = one-hot(prev action) + prev reward + novelty bins
        self.feedback_size = num_actions + 1 + num_novelty_bins

        # ----- frozen reservoir front-end -----
        self.feedback_to_channel = FeedbackToChannel(self.feedback_size, in_h, in_w, seed=seed + 1).to(self.device)
        self.reservoir = Conv2dReservoir(in_channels=4, num_channels=reservoir_channels, seed=seed).to(self.device)
        c, hh, hw = self.reservoir.hidden_shape(in_h, in_w)
        self.hidden_shape = (c, hh, hw)

        # ----- trained conv decoder (Q-net) + target -----
        self.q_net = DeepConvDecoder(c, hh, hw, num_actions, layer_size=decoder_layer_size).to(self.device)
        self.q_target = deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # ----- RECODE density over downscaled images -----
        chans = 1 if recode_grayscale else 3
        obs_emb_size = recode_down_h * recode_down_w * chans
        reward_func = partial(knn_reward, k=recode_knn_k) if recode_knn_k > 0 else adaptive_knn_reward
        self.density = RECODE(
            observation_size=obs_emb_size,
            memory_size=recode_memory_size,
            k=recode_k,
            device=self.device,
            reward_func=reward_func,
        )

        self.memory = ReplayBuffer(memory_length, self.device)

        # rollout state
        self._hidden = self.reservoir.init_hidden(1, in_h, in_w, device=self.device)
        self._prev_action = 0
        self._prev_reward = 0.0

    # ---------- reservoir front-end ----------
    def reset_hidden(self):
        self._hidden = self.reservoir.init_hidden(1, self.in_h, self.in_w, device=self.device)
        self._prev_action = 0
        self._prev_reward = 0.0

    def _feedback_vector(self, novelty_bins: torch.Tensor) -> torch.Tensor:
        oh = F.one_hot(torch.tensor([self._prev_action], device=self.device), self.num_actions).float()
        rew = torch.tensor([[self._prev_reward]], device=self.device, dtype=torch.float32)
        return torch.cat([oh, rew, novelty_bins.reshape(1, -1)], dim=1)

    def _bin_novelty(self, novelty: torch.Tensor) -> torch.Tensor:
        # map a scalar novelty into a one-hot over num_novelty_bins via tanh squashing
        v = torch.tanh(novelty).clamp(0, 1 - 1e-6)
        idx = (v * self.num_novelty_bins).long()
        return F.one_hot(idx.reshape(()), self.num_novelty_bins).float()

    @torch.no_grad()
    def encode(self, img_chw: torch.Tensor, novelty: torch.Tensor) -> torch.Tensor:
        """Step the reservoir with image+feedback, return hidden features (C,h,w)."""
        img = img_chw.unsqueeze(0).to(self.device)
        fb = self._feedback_vector(self._bin_novelty(novelty))
        fb_chan = self.feedback_to_channel(fb)
        x = torch.cat([img, fb_chan], dim=1)
        self._hidden = self.reservoir(x, self._hidden)
        return self._hidden.squeeze(0)

    # ---------- RECODE over images ----------
    def _embed(self, img_chw: torch.Tensor) -> torch.Tensor:
        return downscale_embedding(img_chw.to(self.device), *self.recode_down, grayscale=self.recode_grayscale)

    @torch.no_grad()
    def intrinsic_reward(self, img_chw: torch.Tensor) -> torch.Tensor:
        emb = self._embed(img_chw).unsqueeze(0)
        return self.density.get_reward(emb).squeeze()

    def update_density(self, img_chw: torch.Tensor):
        self.density.update(self._embed(img_chw).unsqueeze(0))

    # ---------- acting ----------
    @torch.no_grad()
    def act(self, hidden_feat: torch.Tensor, greedy: bool = False) -> int:
        if (not greedy) and torch.rand(1).item() < self.epsilon:
            return int(torch.randint(0, self.num_actions, (1,)).item())
        q = self.q_net(hidden_feat.unsqueeze(0))
        return int(q.argmax(dim=1).item())

    def remember(self, z, action, reward, z_next, done):
        self.memory.add(z, action, reward, z_next, done)
        self._prev_action = int(action)
        self._prev_reward = float(reward)

    # ---------- learning ----------
    def replay(self) -> float | None:
        if self.memory.size < self.minibatch_size:
            return None
        z, a, r, z_next, done = self.memory.sample(self.minibatch_size)
        q = self.q_net(z).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.q_target(z_next).max(dim=1).values
            target = r + self.gamma * next_q * (~done)
        loss = F.mse_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # soft target update
        with torch.no_grad():
            for tp, p in zip(self.q_target.parameters(), self.q_net.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)
        return float(loss.item())

    def step_schedule(self):
        self.cur_step += 1
        frac = min(1.0, self.cur_step / max(1, self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
        self.density.step()

    @property
    def metrics(self) -> dict:
        return dict(self.density.metrics)
