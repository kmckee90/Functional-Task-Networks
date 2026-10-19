"""Explorer RECODE agent for the globo vision environment.

Adapts the explorer agent (/m2e/agents/agent_recode.py) to a 3D first-person
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

from .models import Conv2dReservoir, DeepConvDecoder, FeedbackToChannel, downscale_embedding
from .recode.recode import RECODE, adaptive_knn_reward, knn_reward


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
        self.storage_dtype = torch.float32
        self.z = None
        self.z_next = None
        self.emb = torch.zeros(capacity, emb_dim, device=device)
        self.action = torch.zeros(capacity, n_heads, dtype=torch.long, device=device)
        self.env_r = torch.zeros(capacity, device=device)
        self.online_nov = torch.zeros(capacity, device=device)
        self.model_nov = torch.zeros(capacity, device=device)
        self.model_step = torch.zeros(capacity, dtype=torch.long, device=device)
        self.terminal = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.boundary = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.size = 0
        self.ptr = 0

    def _lazy(self, z):
        if self.z is None:
            c, h, w = z.shape
            # Publish both only after allocation succeeds, so failed allocation
            # cannot leave a half-initialized buffer in a checkpoint.
            current = torch.zeros(self.capacity, c, h, w, device=self.device,dtype=self.storage_dtype)
            following = torch.zeros_like(current)
            self.z, self.z_next = current, following

    def add(self, z, action, env_r, online_nov, emb, z_next, terminal, boundary,
            model_nov=0., model_step=0):
        self._lazy(z)
        i = self.ptr
        self.z[i] = z
        self.z_next[i] = z_next
        self.emb[i] = emb
        self.action[i] = torch.as_tensor(action, dtype=torch.long, device=self.device)
        self.env_r[i] = float(env_r)
        self.online_nov[i] = float(online_nov)
        self.model_nov[i] = float(model_nov)
        self.model_step[i] = int(model_step)
        self.terminal[i] = bool(terminal)
        self.boundary[i] = bool(boundary)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def gather(self, idx):
        return dict(z=self.z[idx].float(), z_next=self.z_next[idx].float(), emb=self.emb[idx],
                    action=self.action[idx], env_r=self.env_r[idx],
                    online_nov=self.online_nov[idx], terminal=self.terminal[idx],
                    boundary=self.boundary[idx],model_nov=self.model_nov[idx],model_step=self.model_step[idx])

    def insert(self, data):
        """Append a batch of transitions (e.g. a copied trajectory)."""
        for j in range(data["z"].shape[0]):
            self.add(data["z"][j], data["action"][j], data["env_r"][j], data["online_nov"][j],
                     data["emb"][j], data["z_next"][j], data["terminal"][j], data["boundary"][j],
                     data['model_nov'][j] if 'model_nov' in data else 0.,
                     data['model_step'][j] if 'model_step' in data else 0)

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
        reservoir_variant: str = 'legacy',
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
        death_penalty: float = 0.0,        # negative env reward applied on death; 0 disables
        main_sparse_reward_scale: float = 5.0,
        goal_sparse_reward_scale: float = 5.0,
        # goal / prioritized replay
        goal_memory_length: int = 20000,
        prioritized_replay: bool = True,
        offline_scan_max: int = 8192,
        # RL
        gamma: float = 0.99,
        gamma_env: float = 0.99,           # separate discount for the environment-reward stream
        lr: float = 3e-4,
        grad_clip: float = 0.0,            # max grad-norm; 0 disables
        memory_length: int = 20000,
        minibatch_size: int = 64,
        target_update_tau: float = 0.01,
        target_update_tau_env: float | None = None,   # env-stream target tau; None -> same as intrinsic
        epsilon: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 20000,
        device: str | torch.device = "cpu",
        seed: int = 0,
        model_feedback: bool = False,
        telemetry_dims: int = 0,
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
        self.gamma_env = gamma_env
        self.death_penalty = death_penalty
        self.grad_clip = grad_clip
        self.minibatch_size = minibatch_size
        self.tau = target_update_tau
        # The env/survival critic uses a high gamma_env, which under hard target updates
        # (large tau) and off-policy prioritized replay is prone to deadly-triad divergence.
        # Give it its own (typically slower) Polyak tau, independent of the exploration tau.
        self.tau_env = target_update_tau if target_update_tau_env is None else target_update_tau_env
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

        self.model_feedback = model_feedback
        self._model_feedback = 0.
        self.telemetry = torch.zeros(telemetry_dims, device=self.device)
        self.feedback_size = sum(self.head_sizes) + 1 + num_novelty_bins + int(model_feedback) + telemetry_dims

        # ----- frozen reservoir front-end -----
        self.feedback_to_channel = FeedbackToChannel(self.feedback_size, in_h, in_w, seed=seed + 1).to(self.device)
        self.reservoir = Conv2dReservoir(
            in_channels=4, num_channels=reservoir_channels,
            kernel_size=reservoir_kernel, stride=reservoir_stride, recur_kernel=reservoir_recur_kernel,
            input_scale=reservoir_input_scale, recur_scale=reservoir_recur_scale,
            tc=reservoir_tc, in_sparsity=reservoir_sparsity, seed=seed,
        ).to(self.device)
        c, hh, hw = self.reservoir.hidden_shape(in_h, in_w)
        if reservoir_variant not in ('legacy', 'tuned_v1'):
            raise ValueError(f'Unknown reservoir variant: {reservoir_variant}')
        self.reservoir_config = ({'version': 'legacy'} if reservoir_variant == 'legacy' else
            self.reservoir.configure_tuned(in_h, in_w, seed))
        self.hidden_shape = (c, hh, hw)

        # ----- trained conv decoders (Q-heads) + targets -----
        # Two value streams with independent discounts: q_net = intrinsic (novelty +
        # frontier bonus, discounted at gamma); q_env = environment/survival (in-game
        # reward + death penalty, discounted at gamma_env). The behavior policy acts on
        # their SUM, and each stream bootstraps off the combined greedy action so both
        # value functions stay consistent with one coherent policy (cf. NGU/Agent57's
        # split intrinsic/extrinsic value with separate discount factors).
        self.q_net = DeepConvDecoder(c, hh, hw, self.head_sizes, layer_size=decoder_layer_size).to(self.device)
        self.q_target = deepcopy(self.q_net)
        self.q_env = DeepConvDecoder(c, hh, hw, self.head_sizes, layer_size=decoder_layer_size).to(self.device)
        self.q_env_target = deepcopy(self.q_env)
        for p in (*self.q_target.parameters(), *self.q_env_target.parameters()):
            p.requires_grad = False
        self._trainable_params = list(self.q_net.parameters()) + list(self.q_env.parameters())
        self.optimizer = torch.optim.RMSprop(self._trainable_params, lr=lr)

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
        self._model_feedback = 0.
        self._hidden = self.reservoir.init_hidden(1, self.in_h, self.in_w, device=self.device)
        self._prev_action = [0] * self.n_heads
        self._prev_reward = 0.0

    def _feedback_vector(self, novelty_bins: torch.Tensor) -> torch.Tensor:
        ohs = [F.one_hot(torch.tensor(self._prev_action[i], device=self.device), n).float()
               for i, n in enumerate(self.head_sizes)]
        oh = torch.cat(ohs).reshape(1, -1)
        rew = torch.tensor([[self._prev_reward]], device=self.device, dtype=torch.float32)
        parts = [oh, rew, novelty_bins.reshape(1, -1)]
        if self.model_feedback:
            parts.append(torch.tensor([[self._model_feedback]], device=self.device))
        if self.telemetry.numel():
            parts.append(self.telemetry.reshape(1, -1))
        return torch.cat(parts, dim=1)

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
        if getattr(self, 'capture_input', False):
            self.last_input = x[0].detach().cpu().numpy().copy()
            self.last_feedback = fb[0].detach().cpu().numpy().copy()
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
        h = hidden_feat.unsqueeze(0)
        qs_i, qs_e = self.q_net(h), self.q_env(h)   # intrinsic + environment streams
        out = []
        for q_i, q_e, n in zip(qs_i, qs_e, self.head_sizes):
            if (not greedy) and torch.rand(1).item() < self.epsilon:
                out.append(int(torch.randint(0, n, (1,)).item()))
            else:
                out.append(int((q_i + q_e).argmax(dim=1).item()))   # act on the combined value
        return out

    def note_action(self, action, reward):
        self._prev_action = [int(action)] if isinstance(action, int) else [int(a) for a in action]
        self._prev_reward = float(reward)

    def remember(self, z, action, env_r, online_nov, emb, z_next, terminal, boundary,
                 model_nov=0., model_step=0):
        action = [int(action)] if isinstance(action, int) else list(action)
        self.memory.add(z, action, env_r, online_nov, emb, z_next, terminal, boundary,model_nov,model_step)

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
        z, a, z_next, terminal = B.z[idx].float(), B.action[idx], B.z_next[idx].float(), B.terminal[idx]
        not_term = (~terminal).float()
        with torch.no_grad():
            offline_nov = self._novelty(B.emb[idx])   # dimensionality-normalized
        # --- intrinsic reward stream (novelty + sparse frontier bonus) ---
        r_intr = torch.zeros(len(idx), device=self.device)
        if self.use_online_novelty:
            r_intr = r_intr + self.online_coef * B.online_nov[idx]    # online_nov stored already normalized
        if self.use_offline_novelty:
            r_intr = r_intr + self.offline_coef * offline_nov
        # sparse bonus on the frontier-most (highest novelty) state in this trajectory
        scale = self.goal_sparse_reward_scale if is_goal else self.main_sparse_reward_scale
        best = int(offline_nov.argmax())
        r_intr[best] = r_intr[best] * scale
        # Model novelty is separate from density/frontier reward. Old uncertainty
        # targets decay instead of making the untrained model permanently rewarding.
        weight=getattr(self,'model_reward_weight',0.)
        if weight:
            age=(self.cur_step-B.model_step[idx]).clamp(min=0).float()
            decay=torch.exp2(-age/getattr(self,'model_reward_half_life',10000.))
            r_intr = r_intr + weight * B.model_nov[idx] * decay
        # --- environment (survival) reward stream: in-game reward + death penalty ---
        # death_penalty is applied on terminal (death) transitions independently of
        # env_reward_coef, so a survival signal exists even with globo's reward off.
        r_env = self.env_reward_coef * B.env_r[idx] - self.death_penalty * terminal.float()

        qs_i, qs_e = self.q_net(z), self.q_env(z)
        with torch.no_grad():
            qn_i, qn_e = self.q_target(z_next), self.q_env_target(z_next)
        total = 0.0
        for i in range(self.n_heads):
            q_i, q_e, qn_i_h, qn_e_h = qs_i[i], qs_e[i], qn_i[i], qn_e[i]
            with torch.no_grad():
                # combined greedy next action -> consistent bootstrap for both streams
                a_next = (qn_i_h + qn_e_h).argmax(dim=1, keepdim=True)
                tgt_i = r_intr + self.gamma * qn_i_h.gather(1, a_next).squeeze(1) * not_term
                tgt_e = r_env + self.gamma_env * qn_e_h.gather(1, a_next).squeeze(1) * not_term
            q_i_sa = q_i.gather(1, a[:, i:i + 1]).squeeze(1)
            q_e_sa = q_e.gather(1, a[:, i:i + 1]).squeeze(1)
            total = total + F.mse_loss(q_i_sa, tgt_i) + F.mse_loss(q_e_sa, tgt_e)
        loss = total / self.n_heads
        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip and self.grad_clip > 0:
            nn.utils.clip_grad_norm_(self._trainable_params, self.grad_clip)
        self.optimizer.step()
        with torch.no_grad():
            for tp, p in zip(self.q_target.parameters(), self.q_net.parameters()):
                tp.mul_(1 - self.tau).add_(self.tau * p)
            for tp, p in zip(self.q_env_target.parameters(), self.q_env.parameters()):
                tp.mul_(1 - self.tau_env).add_(self.tau_env * p)
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

    # ---------- checkpointing ----------
    def state_dict(self) -> dict:
        """Everything needed to reproduce the agent's behaviour: both Q streams +
        their targets, optimizer, the frozen reservoir/feedback modules (saved for
        exactness even though they're seed-deterministic), the RECODE density (so the
        novelty feedback channel matches), and the epsilon schedule position. The
        replay buffers are intentionally NOT saved (not needed for demo/inference)."""
        return {
            "q_net": self.q_net.state_dict(),
            "q_target": self.q_target.state_dict(),
            "q_env": self.q_env.state_dict(),
            "q_env_target": self.q_env_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "reservoir": self.reservoir.state_dict(),
            "reservoir_config": self.reservoir_config,
            "feedback_to_channel": self.feedback_to_channel.state_dict(),
            "density": self.density.state,
            "epsilon": self.epsilon,
            "cur_step": self.cur_step,
        }

    def load_state_dict(self, sd: dict, *, load_optimizer: bool = True) -> None:
        if sd.get('reservoir_config', {'version': 'legacy'}) != self.reservoir_config:
            raise ValueError('Reservoir configuration changed; start a fresh run/replay')
        self.q_net.load_state_dict(sd["q_net"])
        self.q_target.load_state_dict(sd["q_target"])
        self.q_env.load_state_dict(sd["q_env"])
        self.q_env_target.load_state_dict(sd["q_env_target"])
        self.reservoir.load_state_dict(sd["reservoir"])
        self.feedback_to_channel.load_state_dict(sd["feedback_to_channel"])
        if load_optimizer and "optimizer" in sd:
            self.optimizer.load_state_dict(sd["optimizer"])
            # move optimizer state onto this agent's device (state may have been saved/loaded
            # on CPU; RMSprop.step() would otherwise hit a device mismatch with cuda params)
            for st in self.optimizer.state.values():
                for k, v in st.items():
                    if torch.is_tensor(v):
                        st[k] = v.to(self.device)
        # move restored density tensors onto this agent's device
        dens = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in sd["density"].items()}
        self.density.set_state(dens)
        self.epsilon = sd.get("epsilon", self.epsilon)
        self.cur_step = sd.get("cur_step", self.cur_step)

    # ---------- centroid export ----------
    def dump_centroids(self, out_dir: str, upscale: int = 8, prefix: str = "") -> int:
        """Write each RECODE cluster centroid (a downscaled image) as <prefix><i>.png,
        plus <prefix>!masses.csv (per-centroid visitation mass = RECODE count)."""
        import os
        import numpy as np
        import imageio.v2 as imageio
        os.makedirs(out_dir, exist_ok=True)
        for f in os.listdir(out_dir):
            if f.startswith(prefix) and f.endswith(".png"):
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
            imageio.imwrite(os.path.join(out_dir, f"{prefix}{i}.png"), img)
        # leading '!' (ASCII 33 < '0') sorts the log above the numbered centroid PNGs
        with open(os.path.join(out_dir, f"{prefix}!masses.csv"), "w") as f:
            f.write("index,mass\n")
            for i, c in enumerate(counts):
                f.write(f"{i},{float(c):.6f}\n")
        return n

    @property
    def metrics(self) -> dict:
        m = dict(self.density.metrics)
        m.update(self._last_metrics)
        return m
