"""Explorer RECODE agent for the globo vision environment.

Adapts the explorer agent (/explorer/agents/agent_recode.py) to a 3D first-person
VISION environment and a DISCRETE action space:
  * conv2d echo-state reservoir + deep conv2d decoder (Fluid_1S) instead of a 1-D
    reservoir; the reservoir + feedback projection are FROZEN, only the decoder
    (Q-heads) is trained;
  * DQN (single head) or branching DQN (factored heads) instead of DDPG; the
    dual obs/feedback critic of the original is intentionally not ported.

RECODE clusters/densities are computed over the IMAGES THEMSELVES (downscaled
pixels), giving both:
  * an ONLINE novelty reward (computed at collection time, stored), and
  * an OFFLINE novelty reward (recomputed during training against the evolving
    cluster memory).

Goal / prioritized replay (matching agent_recode.replay semantics): each training
round the lowest-density (= highest-novelty) state in the buffer is found -- the
exploration FRONTIER -- given a sparse bonus, and the cohesive trajectory leading
to it is moved into a goal buffer and over-sampled, so Bellman backups propagate
the frontier bonus back along that trajectory (long-range planning to the frontier).
"""

from __future__ import annotations

import math
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Conv2dReservoir, DeepConvDecoder, FeedbackToChannel, downscale_embedding
from recode.recode import RECODE, adaptive_knn_reward, knn_reward


class ReplayBuffer:
    """Circular buffer of transitions. Stores frozen-reservoir features (z, z_next),
    the RECODE image embedding (emb, of next state) so offline novelty can be
    recomputed, the env reward and online novelty, and terminal/boundary flags.
    `terminal` marks true episode ends (for bootstrapping); `boundary` marks any
    episode reset (death OR truncation) for trajectory segmentation."""

    def __init__(self, capacity: int, device, n_heads: int = 1, emb_dim: int = 1):
        self.capacity = capacity
        self.device = device
        self.emb_dim = emb_dim
        self.z = None
        self.z_next = None
        self.emb = torch.zeros(capacity, emb_dim, device=device)
        self.action = torch.zeros(capacity, n_heads, dtype=torch.long, device=device)
        self.env_r = torch.zeros(capacity, device=device)
        self.online_nov = torch.zeros(capacity, device=device)
        self.terminal = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.boundary = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.size = 0
        self.ptr = 0

    def _lazy(self, z):
        if self.z is None:
            c, h, w = z.shape
            self.z = torch.zeros(self.capacity, c, h, w, device=self.device)
            self.z_next = torch.zeros(self.capacity, c, h, w, device=self.device)

    def add(self, z, action, env_r, online_nov, emb, z_next, terminal, boundary):
        self._lazy(z)
        i = self.ptr
        self.z[i] = z
        self.z_next[i] = z_next
        self.emb[i] = emb
        self.action[i] = torch.as_tensor(action, dtype=torch.long, device=self.device)
        self.env_r[i] = float(env_r)
        self.online_nov[i] = float(online_nov)
        self.terminal[i] = bool(terminal)
        self.boundary[i] = bool(boundary)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def gather(self, idx):
        return dict(z=self.z[idx], z_next=self.z_next[idx], emb=self.emb[idx],
                    action=self.action[idx], env_r=self.env_r[idx],
                    online_nov=self.online_nov[idx], terminal=self.terminal[idx],
                    boundary=self.boundary[idx])

    def insert(self, data):
        """Append a batch of transitions (e.g. a copied trajectory)."""
        for j in range(data["z"].shape[0]):
            self.add(data["z"][j], data["action"][j], data["env_r"][j], data["online_nov"][j],
                     data["emb"][j], data["z_next"][j], data["terminal"][j], data["boundary"][j])

    def segments(self):
        """List of (start, end_exclusive) episode spans over [0:size], split after
        each boundary flag. Time-contiguous while the buffer hasn't wrapped."""
        b = self.boundary[:self.size]
        ends = torch.nonzero(b, as_tuple=False).flatten().tolist()
        segs, start = [], 0
        for e in ends:
            segs.append((start, e + 1))
            start = e + 1
        if start < self.size:
            segs.append((start, self.size))
        return segs if segs else [(0, self.size)]


class GloboExplorerDQN:
    """Conv2d-reservoir RECODE DQN explorer with offline reward + goal/frontier
    prioritized trajectory replay."""

    def __init__(
        self,
        num_actions: int | None = None,
        in_h: int = 32,
        in_w: int = 64,
        *,
        action_groups: list[int] | None = None,
        # reservoir (frozen) architecture
        reservoir_channels: int = 64,
        reservoir_kernel: int = 7,
        reservoir_stride: int = 7,
        reservoir_recur_kernel: int = 3,
        reservoir_input_scale: float = 23.75,
        reservoir_recur_scale: float = 1.49,
        reservoir_tc: float = 1.0,
        reservoir_sparsity: float = 0.5,
        decoder_layer_size: int = 256,
        num_novelty_bins: int = 8,
        # RECODE over downscaled images
        recode_down_h: int = 16,
        recode_down_w: int = 32,
        recode_grayscale: bool = True,
        recode_memory_size: int = 2000,
        recode_k: int = 10,
        recode_knn_k: int = 10,
        recode_kappa: float = 0.2,
        recode_decay: float = 0.9999,
        recode_density_gamma: float = 1.0,
        recode_insertion_probability: float = 0.1,
        # reward shaping
        use_online_novelty: bool = False,
        use_offline_novelty: bool = True,
        online_coef: float = 1.0,
        offline_coef: float = 1.0,
        env_reward_coef: float = 0.0,
        main_sparse_reward_scale: float = 5.0,
        goal_sparse_reward_scale: float = 5.0,
        # goal / prioritized replay
        goal_memory_length: int = 20000,
        prioritized_replay: bool = True,
        offline_scan_max: int = 8192,
        # RL
        gamma: float = 0.99,
        lr: float = 3e-4,
        grad_clip: float = 0.0,            # max grad-norm; 0 disables
        memory_length: int = 20000,
        minibatch_size: int = 64,
        target_update_tau: float = 0.01,
        epsilon: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20000,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ):
        self.device = torch.device(device)
        if action_groups is not None:
            self.head_sizes = list(int(n) for n in action_groups)
        elif num_actions is not None:
            self.head_sizes = [int(num_actions)]
        else:
            raise ValueError("Provide either num_actions or action_groups")
        self.n_heads = len(self.head_sizes)
        self.factored = self.n_heads > 1
        self.in_h, self.in_w = in_h, in_w
        self.gamma = gamma
        self.grad_clip = grad_clip
        self.minibatch_size = minibatch_size
        self.tau = target_update_tau
        self.epsilon = epsilon
        self.epsilon_start = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.env_reward_coef = env_reward_coef
        self.online_coef = online_coef
        self.offline_coef = offline_coef
        self.use_online_novelty = use_online_novelty
        self.use_offline_novelty = use_offline_novelty
        self.main_sparse_reward_scale = main_sparse_reward_scale
        self.goal_sparse_reward_scale = goal_sparse_reward_scale
        self.prioritized_replay = prioritized_replay
        self.offline_scan_max = offline_scan_max
        self.num_novelty_bins = num_novelty_bins
        self.recode_down = (recode_down_h, recode_down_w)
        self.recode_grayscale = recode_grayscale
        self.cur_step = 0

        self.feedback_size = sum(self.head_sizes) + 1 + num_novelty_bins

        # ----- frozen reservoir front-end -----
        self.feedback_to_channel = FeedbackToChannel(self.feedback_size, in_h, in_w, seed=seed + 1).to(self.device)
        self.reservoir = Conv2dReservoir(
            in_channels=4, num_channels=reservoir_channels,
            kernel_size=reservoir_kernel, stride=reservoir_stride, recur_kernel=reservoir_recur_kernel,
            input_scale=reservoir_input_scale, recur_scale=reservoir_recur_scale,
            tc=reservoir_tc, in_sparsity=reservoir_sparsity, seed=seed,
        ).to(self.device)
        c, hh, hw = self.reservoir.hidden_shape(in_h, in_w)
        self.hidden_shape = (c, hh, hw)

        # ----- trained conv decoder (Q-heads) + target -----
        self.q_net = DeepConvDecoder(c, hh, hw, self.head_sizes, layer_size=decoder_layer_size).to(self.device)
        self.q_target = deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.RMSprop(self.q_net.parameters(), lr=lr)

        # ----- RECODE density over downscaled images -----
        chans = 1 if recode_grayscale else 3
        obs_emb_size = recode_down_h * recode_down_w * chans
        # Normalize the kNN Euclidean distance by sqrt(embedding dim) -> RMS per-dimension
        # distance, so novelty magnitude is stable regardless of the downscale resolution.
        self.nov_scale = 1.0 / math.sqrt(max(1, obs_emb_size))
        reward_func = partial(knn_reward, k=recode_knn_k) if recode_knn_k > 0 else adaptive_knn_reward
        self.density = RECODE(
            observation_size=obs_emb_size, memory_size=recode_memory_size, k=recode_k,
            device=self.device, reward_func=reward_func, relative_tolerance=recode_kappa,
            decay_rate=recode_decay, gamma=recode_density_gamma,
            insertion_probability=recode_insertion_probability,
        )

        self.memory = ReplayBuffer(memory_length, self.device, n_heads=self.n_heads, emb_dim=obs_emb_size)
        self.goal = ReplayBuffer(goal_memory_length, self.device, n_heads=self.n_heads, emb_dim=obs_emb_size)

        # rollout state
        self._hidden = self.reservoir.init_hidden(1, in_h, in_w, device=self.device)
        self._prev_action = [0] * self.n_heads
        self._prev_reward = 0.0
        self._last_metrics: dict[str, float] = {}

    # ---------- reservoir front-end ----------
    def reset_hidden(self):
        self._hidden = self.reservoir.init_hidden(1, self.in_h, self.in_w, device=self.device)
        self._prev_action = [0] * self.n_heads
        self._prev_reward = 0.0

    def _feedback_vector(self, novelty_bins: torch.Tensor) -> torch.Tensor:
        ohs = [F.one_hot(torch.tensor(self._prev_action[i], device=self.device), n).float()
               for i, n in enumerate(self.head_sizes)]
        oh = torch.cat(ohs).reshape(1, -1)
        rew = torch.tensor([[self._prev_reward]], device=self.device, dtype=torch.float32)
        return torch.cat([oh, rew, novelty_bins.reshape(1, -1)], dim=1)

    def _bin_novelty(self, novelty: torch.Tensor) -> torch.Tensor:
        v = torch.tanh(novelty).clamp(0, 1 - 1e-6)
        idx = (v * self.num_novelty_bins).long()
        return F.one_hot(idx.reshape(()), self.num_novelty_bins).float()

    @torch.no_grad()
    def encode(self, img_chw: torch.Tensor, novelty: torch.Tensor) -> torch.Tensor:
        img = img_chw.unsqueeze(0).to(self.device)
        fb = self._feedback_vector(self._bin_novelty(novelty))
        fb_chan = self.feedback_to_channel(fb)
        x = torch.cat([img, fb_chan], dim=1)
        self._hidden = self.reservoir(x, self._hidden)
        return self._hidden.squeeze(0)

    # ---------- RECODE over images ----------
    def embed(self, img_chw: torch.Tensor) -> torch.Tensor:
        return downscale_embedding(img_chw.to(self.device), *self.recode_down, grayscale=self.recode_grayscale)

    @torch.no_grad()
    def _novelty(self, emb: torch.Tensor) -> torch.Tensor:
        """RECODE kNN distance, dimensionality-normalized (RMS per dim) so the
        reward scale is independent of the embedding size."""
        return self.density.get_reward(emb) * self.nov_scale

    @torch.no_grad()
    def intrinsic_reward(self, img_chw: torch.Tensor) -> torch.Tensor:
        return self._novelty(self.embed(img_chw).unsqueeze(0)).squeeze()

    def update_density(self, img_chw: torch.Tensor):
        self.density.update(self.embed(img_chw).unsqueeze(0))

    # ---------- acting ----------
    @torch.no_grad()
    def act(self, hidden_feat: torch.Tensor, greedy: bool = False):
        qs = self.q_net(hidden_feat.unsqueeze(0))
        out = []
        for q, n in zip(qs, self.head_sizes):
            if (not greedy) and torch.rand(1).item() < self.epsilon:
                out.append(int(torch.randint(0, n, (1,)).item()))
            else:
                out.append(int(q.argmax(dim=1).item()))
        return out

    def note_action(self, action, reward):
        self._prev_action = [int(action)] if isinstance(action, int) else [int(a) for a in action]
        self._prev_reward = float(reward)

    def remember(self, z, action, env_r, online_nov, emb, z_next, terminal, boundary):
        action = [int(action)] if isinstance(action, int) else list(action)
        self.memory.add(z, action, env_r, online_nov, emb, z_next, terminal, boundary)

    # ---------- learning ----------
    def _traj_slice(self, seg, traj_len):
        s, e = seg
        return torch.arange(max(s, e - traj_len), e, device=self.device)

    @staticmethod
    def _seg_containing(segs, idx):
        for s, e in segs:
            if s <= idx < e:
                return (s, e)
        return segs[-1]

    def _td_update(self, B, idx, is_goal):
        if len(idx) == 0:
            return None
        z, a, z_next, terminal = B.z[idx], B.action[idx], B.z_next[idx], B.terminal[idx]
        with torch.no_grad():
            offline_nov = self._novelty(B.emb[idx])   # dimensionality-normalized
        r = self.env_reward_coef * B.env_r[idx]
        if self.use_online_novelty:
            r = r + self.online_coef * B.online_nov[idx]    # online_nov stored already normalized
        if self.use_offline_novelty:
            r = r + self.offline_coef * offline_nov
        # sparse bonus on the frontier-most (highest novelty) state in this trajectory
        scale = self.goal_sparse_reward_scale if is_goal else self.main_sparse_reward_scale
        best = int(offline_nov.argmax())
        r[best] = r[best] * scale
        qs, qs_next = self.q_net(z), self.q_target(z_next)
        total = 0.0
        for i, (q, qn) in enumerate(zip(qs, qs_next)):
            q_sa = q.gather(1, a[:, i:i + 1]).squeeze(1)
            with torch.no_grad():
                tgt = r + self.gamma * qn.max(dim=1).values * (~terminal)
            total = total + F.mse_loss(q_sa, tgt)
        loss = total / self.n_heads
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
        self.optimizer.step()
        with torch.no_grad():
            for tp, p in zip(self.q_target.parameters(), self.q_net.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)
        return float(loss.item())

    def train_round(self, n_updates: int) -> float | None:
        """One training round: recompute the offline frontier, move its trajectory
        into the goal buffer, then run `n_updates` prioritized trajectory updates."""
        M = self.memory
        if M.size < self.minibatch_size:
            return None

        # --- offline frontier: lowest-density (highest-novelty) state ---
        with torch.no_grad():
            if M.size <= self.offline_scan_max:
                scan = torch.arange(M.size, device=self.device)
            else:
                scan = torch.randperm(M.size, device=self.device)[:self.offline_scan_max]
            nov_scan = self._novelty(M.emb[scan])
        fpos = int(nov_scan.argmax())
        frontier = int(scan[fpos])
        main_best = float(nov_scan[fpos]) * self.main_sparse_reward_scale

        goal_best = 0.0
        if self.goal.size >= 1:
            with torch.no_grad():
                gnov = self._novelty(self.goal.emb[:self.goal.size])
            goal_best = float(gnov.max()) * self.goal_sparse_reward_scale

        # --- move the cohesive trajectory leading to the frontier into the goal buffer ---
        if main_best >= goal_best:
            seg = self._seg_containing(M.segments(), frontier)
            self.goal.insert(M.gather(self._traj_slice(seg, self.minibatch_size)))

        # --- prioritized updates ---
        mseg = M.segments()
        gnov_full, gseg = None, None
        if self.goal.size >= self.minibatch_size:
            with torch.no_grad():
                gnov_full = self._novelty(self.goal.emb[:self.goal.size])
            gseg = self.goal.segments()
        modes = ["goal", "last", "random"] if self.prioritized_replay else ["random"]

        losses = []
        for i in range(n_updates):
            mode = modes[i % len(modes)]
            if mode == "goal" and gnov_full is not None:
                seg = self._seg_containing(gseg, int(gnov_full.argmax()))
                loss = self._td_update(self.goal, self._traj_slice(seg, self.minibatch_size), True)
            elif mode == "last":
                loss = self._td_update(M, self._traj_slice(mseg[-1], self.minibatch_size), False)
            else:
                seg = mseg[int(torch.randint(len(mseg), (1,)).item())]
                loss = self._td_update(M, self._traj_slice(seg, self.minibatch_size), False)
            if loss is not None:
                losses.append(loss)

        self._last_metrics = dict(frontier_novelty=main_best / max(1e-9, self.main_sparse_reward_scale),
                                  goal_buffer=float(self.goal.size))
        return float(sum(losses) / len(losses)) if losses else None

    def step_schedule(self):
        self.cur_step += 1
        frac = min(1.0, self.cur_step / max(1, self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)
        self.density.step()

    # ---------- centroid export ----------
    def dump_centroids(self, out_dir: str, upscale: int = 8) -> int:
        """Write each RECODE cluster centroid (a downscaled image) as <i>.png, plus
        masses.csv (per-centroid visitation mass = RECODE count)."""
        import os
        import numpy as np
        import imageio.v2 as imageio
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):
            if f.endswith(".png"):
                os.remove(os.path.join(out_dir, f))
        n = self.density.size
        mem = self.density.memory[:n].detach().cpu().numpy()
        counts = self.density.counts[:n].detach().cpu().numpy()
        h, w = self.recode_down
        ch = 1 if self.recode_grayscale else 3
        for i in range(n):
            v = np.clip(mem[i], 0.0, 1.0)
            img = v.reshape(h, w) if ch == 1 else v.reshape(ch, h, w).transpose(1, 2, 0)
            img = (img * 255).astype(np.uint8)
            if ch == 1:
                img = np.stack([img] * 3, axis=-1)
            img = np.kron(img, np.ones((upscale, upscale, 1), dtype=np.uint8))
            imageio.imwrite(os.path.join(out_dir, f"{i}.png"), img)
        # leading '!' (ASCII 33 < '0') sorts the log above the numbered centroid PNGs
        with open(os.path.join(out_dir, "!masses.csv"), "w") as f:
            f.write("index,mass\n")
            for i, c in enumerate(counts):
                f.write(f"{i},{float(c):.6f}\n")
        return n

    @property
    def metrics(self) -> dict:
        m = dict(self.density.metrics)
        m.update(self._last_metrics)
        return m
