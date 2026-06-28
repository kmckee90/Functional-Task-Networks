"""Explorer agent that discovers options over RECODE centroids with FTN-DDPG.

Fuses three existing pieces (imported, not reimplemented where practical):

* **RECODE** (``explorer/recode/recode.py``) -- online clustering of egocentric
  observations into atoms, and the intrinsic *novelty* reward that drives
  exploration (``get_reward`` against the evolving atom memory).
* **FTN-DDPG** (``rl/robotics/ftn.py``) -- ``FTNActor`` / ``FTNCritic`` are banks
  of ``H`` parallel MLPs read out through a binary KWTA ``mask``. An **option is a
  mask pair** ``(m_a, m_c)``; ``anneal_ste_mask`` searches a mask for a given
  reward and DDPG then trains that masked subspace. Mask *overlap* across options
  is the cross-task consolidation channel (shared neurons accumulate gradients).
* a thin **CentroidRegistry** giving RECODE's churning atoms stable integer IDs so
  options and the behavioral graph can refer to fixed centroids across rounds.

Design (see options/README.md "Step 2"): observations are PURELY egocentric, so a
centroid is a *sensory situation*. An option's artificial reward is a function of
the observation the policy itself sees -- ``r_C = R_goal*[member(next_obs)==C] -
step_cost`` -- so a reactive FTN policy can perceive exactly what it optimizes.
The true (x, y) pose is used only to give each centroid a render-time footprint.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- import bootstrap: RECODE needs `recode` top-level; ftn is self-contained ---
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_ROOT, "explorer"), os.path.join(_ROOT, "rl", "robotics")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ftn import FTNActor, FTNCritic, LateralSmoother, anneal_ste_mask, kwta  # noqa: E402
from recode.recode import RECODE, adaptive_knn_reward, knn_reward  # noqa: E402

# All tunable parameters live in options_config.py (single source of truth).
from options_config import AgentConfig  # noqa: E402


# ---------------------------------------------------------------------------
# MC returns (port of compute_mc_returns from rl/robotics/train.py)
# ---------------------------------------------------------------------------


def compute_mc_returns(rewards: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
    T = len(rewards)
    ret = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            g = 0.0
        g = float(rewards[t]) + gamma * g
        ret[t] = g
    return ret


# ---------------------------------------------------------------------------
# Episode buffer (contiguous, episode-aware) -- supports DDPG sampling, ordered
# membership streaming, and per-option MC returns.
# ---------------------------------------------------------------------------


class EpisodeBuffer:
    """Contiguous, episode-aware transition store on preallocated tensors.

    Append is O(1) and ``arrays()`` returns zero-copy views (no per-call restack),
    so frequent DDPG updates over a growing buffer stay O(batch), not O(n^2).
    Capacity is a hard cap (no wrap) -- size it above the planned step count;
    once full, further adds are dropped (with a one-time warning)."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 200_000, device: str = "cpu"):
        self.device = device
        self.capacity = capacity
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.act = torch.zeros(capacity, act_dim, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.pos = torch.zeros(capacity, 2, device=device)
        self.next_pos = torch.zeros(capacity, 2, device=device)
        self.boundary = torch.zeros(capacity, device=device)
        self.size = 0
        self._warned = False

    def add(self, obs, act, next_obs, pos, next_pos, boundary):
        i = self.size
        if i >= self.capacity:
            if not self._warned:
                print(f"[EpisodeBuffer] capacity {self.capacity} reached; dropping further transitions")
                self._warned = True
            return
        self.obs[i] = torch.as_tensor(np.asarray(obs, np.float32), device=self.device)
        self.act[i] = torch.as_tensor(np.asarray(act, np.float32), device=self.device)
        self.next_obs[i] = torch.as_tensor(np.asarray(next_obs, np.float32), device=self.device)
        self.pos[i] = torch.as_tensor(np.asarray(pos, np.float32), device=self.device)
        self.next_pos[i] = torch.as_tensor(np.asarray(next_pos, np.float32), device=self.device)
        self.boundary[i] = float(boundary)
        self.size += 1

    def __len__(self):
        return self.size

    def arrays(self) -> dict:
        """Zero-copy views over the filled portion of the buffer."""
        n = self.size
        return {"obs": self.obs[:n], "act": self.act[:n], "next_obs": self.next_obs[:n],
                "pos": self.pos[:n], "next_pos": self.next_pos[:n], "boundary": self.boundary[:n]}

    def sample_idx(self, batch: int) -> torch.Tensor:
        return torch.randint(0, self.size, (min(batch, self.size),), device=self.device)


# ---------------------------------------------------------------------------
# Stable centroid registry on top of RECODE
# ---------------------------------------------------------------------------


class CentroidRegistry:
    """Assigns stable integer IDs to coarsened RECODE atoms.

    Centroid vectors live in observation space (same as RECODE atoms). Each
    centroid also carries a render-only mean physical position, updated from the
    true poses streamed through ``update_footprints``.
    """

    def __init__(self, obs_dim: int, cfg: AgentConfig, device: str = "cpu"):
        self.cfg = cfg
        self.device = device
        self.obs_dim = obs_dim
        self.centroids = torch.zeros(0, obs_dim, device=device)
        self.pos = torch.zeros(0, 2, device=device)       # mean footprint
        self.pos_count = torch.zeros(0, device=device)
        self.counts = torch.zeros(0, device=device)        # membership counts

    @property
    def n(self) -> int:
        return self.centroids.shape[0]

    def sync(self, recode: RECODE) -> int:
        """Fold the current RECODE atoms into the stable centroid set via
        farthest-point sampling, seeded with the existing centroids so their IDs
        are preserved across rounds.

        Existing centroids are first re-anchored (EMA) onto their nearest atom so
        they track the evolving cluster manifold; then new centroids are appended
        by repeatedly taking the atom farthest from the current centroid set until
        `target_centroids` is reached (or atoms are exhausted / too close).
        Returns the number of centroids added."""
        if recode.size == 0:
            return 0
        atoms = recode.all_pos.detach().to(self.device)  # (A, obs_dim)
        cfg = self.cfg

        # Re-anchor existing centroids onto the current manifold (gentle EMA).
        if self.n > 0:
            d = torch.cdist(self.centroids, atoms)
            nearest = atoms[torch.argmin(d, dim=1)]
            e = cfg.centroid_anchor_ema
            self.centroids = (1 - e) * self.centroids + e * nearest

        added = 0
        target = min(cfg.target_centroids, cfg.max_centroids, atoms.shape[0])
        while self.n < target:
            if self.n == 0:
                self._append(atoms[0])
                added += 1
                continue
            d = torch.cdist(atoms, self.centroids).min(dim=1).values  # (A,)
            j = int(torch.argmax(d))
            if float(d[j]) < cfg.centroid_min_sep:
                break
            self._append(atoms[j])
            added += 1
        return added

    def _append(self, vec: torch.Tensor):
        self.centroids = torch.cat([self.centroids, vec[None, :]], dim=0)
        self.pos = torch.cat([self.pos, torch.zeros(1, 2, device=self.device)], dim=0)
        self.pos_count = torch.cat([self.pos_count, torch.zeros(1, device=self.device)], dim=0)
        self.counts = torch.cat([self.counts, torch.zeros(1, device=self.device)], dim=0)

    def membership(self, obs: torch.Tensor) -> torch.Tensor:
        """Nearest-centroid id(s). Accepts (obs_dim,) or (B, obs_dim).
        Returns -1 where no centroids exist."""
        if obs.dim() == 1:
            obs = obs[None, :]
        if self.n == 0:
            return torch.full((obs.shape[0],), -1, dtype=torch.long, device=self.device)
        d = torch.cdist(obs, self.centroids)  # (B, N)
        return torch.argmin(d, dim=1)

    def update_footprints(self, obs: torch.Tensor, pos: torch.Tensor) -> None:
        """Stream observations + true poses to maintain per-centroid mean pos
        and membership counts (render-only)."""
        ids = self.membership(obs)
        for i in range(self.n):
            m = ids == i
            c = int(m.sum())
            if c == 0:
                continue
            self.counts[i] += c
            mean_p = pos[m].mean(dim=0)
            n0 = self.pos_count[i]
            self.pos[i] = (self.pos[i] * n0 + mean_p * c) / (n0 + c)
            self.pos_count[i] += c

    def positions_np(self) -> np.ndarray:
        return self.pos.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------


class OptionsAgent:
    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        dev = cfg.device
        torch.set_default_device(dev)
        H, k = cfg.ftn_hidden_size, cfg.ftn_k
        L, IS, D = cfg.ftn_layers, cfg.ftn_inner_size, cfg.ftn_dropout_p

        self.actor = FTNActor(cfg.obs_dim, cfg.act_dim, H, L, IS, D)
        self.critic = FTNCritic(cfg.obs_dim, cfg.act_dim, H, L, IS, D)
        self.actor_t = FTNActor(cfg.obs_dim, cfg.act_dim, H, L, IS, D)
        self.critic_t = FTNCritic(cfg.obs_dim, cfg.act_dim, H, L, IS, D)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())
        for p in self.actor_t.parameters():
            p.requires_grad_(False)
        for p in self.critic_t.parameters():
            p.requires_grad_(False)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        # SmoothKWTA lateral smoother applied to every option / explore mask. A
        # no-op (identity) unless lateral_steps > 0 and lateral_polyak < 1.0.
        self.smoother = LateralSmoother(
            H, k, square_dim=cfg.lateral_square_dim, kernel_size=cfg.lateral_kernel_size,
            lateral_steps=cfg.lateral_steps, lateral_polyak=cfg.lateral_polyak,
            lateral_noise_sd=cfg.lateral_noise_sd, torus=cfg.lateral_torus,
        )

        # Exploration owns a dedicated fixed mask; options own learned masks.
        self.m_explore_a = self.smoother.smooth(kwta(torch.randn(H), k=k))
        self.m_explore_c = self.smoother.smooth(kwta(torch.randn(H), k=k))
        # option id -> dict(m_a, m_c, m_a_logits, m_c_logits)
        self.options: dict[int, dict] = {}

        # RECODE: clustering source + intrinsic novelty reward.
        reward_func = knn_reward if cfg.recode_k > 0 else adaptive_knn_reward
        self.recode = RECODE(
            observation_size=cfg.obs_dim,
            memory_size=cfg.recode_memory_size,
            k=cfg.recode_k,
            device=dev,
            reward_func=reward_func,
            relative_tolerance=cfg.recode_kappa,
            gamma=cfg.recode_gamma,
            insertion_probability=cfg.recode_insertion_probability,
            decay_rate=cfg.recode_decay,
        )
        self.registry = CentroidRegistry(cfg.obs_dim, cfg, device=dev)
        self.buffer = EpisodeBuffer(cfg.obs_dim, cfg.act_dim, capacity=cfg.buffer_capacity, device=dev)
        self.epsilon = cfg.epsilon
        self.device = dev

    # ---- exploration ---------------------------------------------------

    @torch.no_grad()
    def act_explore(self, obs_np: np.ndarray) -> np.ndarray:
        """Forward-biased epsilon-random / novelty-DDPG explore action."""
        if np.random.rand() < self.epsilon:
            turn = np.random.uniform(-1, 1)
            accel = np.random.uniform(self.cfg.forward_bias - 1.0, 1.0)  # bias forward
            return np.array([turn, accel], dtype=np.float32)
        obs = torch.tensor(obs_np, device=self.device).unsqueeze(0)
        self.actor.eval()
        a = self.actor(obs, self.m_explore_a).squeeze(0).cpu().numpy()
        self.actor.train()
        a = a + np.random.randn(self.cfg.act_dim).astype(np.float32) * self.cfg.noise_std
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def observe(self, obs, action, next_obs, pos, next_pos, boundary):
        """Record a transition and feed RECODE its online clustering update."""
        self.buffer.add(obs, action, next_obs, pos, next_pos, boundary)
        self.recode.update(torch.tensor(np.asarray(next_obs, np.float32), device=self.device))
        self.recode.step()

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_min, self.epsilon * self.cfg.epsilon_decay)

    # ---- generic DDPG step (joint over terms; FTN consolidation) -------

    def _ddpg_step(self, terms: list[dict]):
        """One critic + actor + polyak update over a list of terms, each a dict
        {obs, act, next_obs, reward, done, m_a, m_c}. Summing per-term losses with
        each forwarded under its own mask accumulates gradients on neurons shared
        across masks -- the FTN cross-option consolidation mechanism."""
        gamma, tau = self.cfg.gamma, self.cfg.tau

        critic_total = None
        for t in terms:
            with torch.no_grad():
                a_next = self.actor_t(t["next_obs"], t["m_a"])
                q_next = self.critic_t(t["next_obs"], a_next, t["m_c"])
                target = t["reward"] + gamma * (1.0 - t["done"]) * q_next
            loss = F.mse_loss(self.critic(t["obs"], t["act"], t["m_c"]), target)
            critic_total = loss if critic_total is None else critic_total + loss
        self.critic_opt.zero_grad()
        critic_total.backward()
        self.critic_opt.step()

        actor_total = None
        for t in terms:
            loss = -self.critic(t["obs"], self.actor(t["obs"], t["m_a"]), t["m_c"]).mean()
            actor_total = loss if actor_total is None else actor_total + loss
        self.actor_opt.zero_grad()
        actor_total.backward()
        self.actor_opt.step()

        with torch.no_grad():
            for p, pt in zip(self.actor.parameters(), self.actor_t.parameters()):
                pt.mul_(1 - tau).add_(p, alpha=tau)
            for p, pt in zip(self.critic.parameters(), self.critic_t.parameters()):
                pt.mul_(1 - tau).add_(p, alpha=tau)

    # ---- explorer training (novelty reward) ----------------------------

    def train_explorer(self, n_steps: int):
        if len(self.buffer) < self.cfg.batch_size:
            return
        arr = self.buffer.arrays()
        for _ in range(n_steps):
            idx = self.buffer.sample_idx(self.cfg.batch_size)
            next_obs = arr["next_obs"][idx]
            with torch.no_grad():
                reward = self.recode.get_reward(next_obs).reshape(-1)
            done = arr["boundary"][idx].reshape(-1)
            self._ddpg_step([{
                "obs": arr["obs"][idx], "act": arr["act"][idx], "next_obs": next_obs,
                "reward": reward, "done": done,
                "m_a": self.m_explore_a, "m_c": self.m_explore_c,
            }])

    # ---- option (per-centroid membership reward) -----------------------

    def _option_reward_done(self, next_obs: torch.Tensor, cid: int, boundary: torch.Tensor):
        """Artificial reward/terminal for reaching centroid `cid`.

        reward = R*[member==C] - shaping*dist(next_obs, centroid_C) - step_cost.
        The dense distance term lets the option descend toward the centroid in
        observation space even before exact membership flips."""
        ids = self.registry.membership(next_obs)
        member = (ids == cid).float()
        dist = torch.norm(next_obs - self.registry.centroids[cid][None, :], dim=1)
        reward = (self.cfg.option_goal_reward * member
                  - self.cfg.option_shaping * dist
                  - self.cfg.option_step_cost)
        done = torch.clamp(member + boundary.reshape(-1), max=1.0)  # arrival or ep end
        return reward.reshape(-1), done.reshape(-1), member

    def reconfig_option(self, cid: int) -> dict:
        """Search a fresh mask pair for option `cid` via AnnealedSTE on the
        membership-relabeled exploration data (critic mask correlates Q with MC
        returns-to-C; actor mask maximizes Q under it)."""
        cfg = self.cfg
        arr = self.buffer.arrays()
        n = arr["obs"].shape[0]
        # Subsample support for the mask search.
        n_sup = min(cfg.reconfig_n, n)
        sup_idx = torch.randperm(n, device=self.device)[:n_sup]
        obs = arr["obs"][sup_idx]
        act = arr["act"][sup_idx]
        next_obs = arr["next_obs"][sup_idx]
        boundary = arr["boundary"][sup_idx]
        # MC returns over the ORDERED stream (so reaching C propagates backward),
        # then gathered to the support rows.
        full_r, full_d, _ = self._option_reward_done(arr["next_obs"], cid, arr["boundary"])
        returns_full = compute_mc_returns(
            full_r.cpu().numpy(), full_d.cpu().numpy(), cfg.gamma
        )
        returns = torch.tensor(returns_full, device=self.device)[sup_idx]

        H, k = cfg.ftn_hidden_size, cfg.ftn_k

        def critic_loss(soft_m):  # -Pearson(Q(s,a), G_t)
            q = self.critic(obs, act, soft_m)
            qc = q - q.mean(); rc = returns - returns.mean()
            r = F.cosine_similarity(qc.unsqueeze(0), rc.unsqueeze(0), dim=1).squeeze(0)
            return -r

        m_c, m_c_logits, c0, c1 = anneal_ste_mask(
            H, k, critic_loss, cfg.reconfig_steps, cfg.reconfig_lr,
            noise_scale=cfg.mask_init_noise_scale,
        )
        m_c = self.smoother.smooth(m_c)  # spatial blob (no-op if disabled)

        def actor_loss(soft_m):   # maximize Q under the new critic mask
            return -self.critic(obs, self.actor(obs, soft_m), m_c).mean()

        m_a, m_a_logits, a0, a1 = anneal_ste_mask(
            H, k, actor_loss, cfg.reconfig_steps, cfg.reconfig_lr,
            noise_scale=cfg.mask_init_noise_scale,
        )
        m_a = self.smoother.smooth(m_a)
        self.options[cid] = {"m_a": m_a, "m_c": m_c, "m_a_logits": m_a_logits, "m_c_logits": m_c_logits}
        return {"critic": (c0, c1), "actor": (a0, a1)}

    def _mask_overlap(self, m1: torch.Tensor, m2: torch.Tensor) -> float:
        denom = float(m1.sum().clamp(min=1).item())
        return float((m1 * m2).sum().item()) / denom

    def ensure_option(self, cid: int) -> dict:
        """Make sure option `cid` has a mask pair. When mask reconfig is disabled
        (or before the first reconfig) an option still needs *some* subspace to
        train/act in, so initialize a fresh random KWTA mask pair (the fixed-random
        -subspace regime). No-op if the option already exists."""
        if cid not in self.options:
            H, k = self.cfg.ftn_hidden_size, self.cfg.ftn_k
            self.options[cid] = {"m_a": self.smoother.smooth(kwta(torch.randn(H), k=k)),
                                 "m_c": self.smoother.smooth(kwta(torch.randn(H), k=k)),
                                 "m_a_logits": None, "m_c_logits": None}
        return self.options[cid]

    def train_option(self, cid: int, n_steps: int, consolidate: bool = True):
        """DDPG-train option `cid`'s masked subspace on membership-relabeled data.
        Optionally sum in mask-overlapping sibling options (consolidation)."""
        if len(self.buffer) < self.cfg.batch_size:
            return
        self.ensure_option(cid)  # lazily create a random mask if reconfig is off
        arr = self.buffer.arrays()
        opt = self.options[cid]
        # Sibling options whose actor mask overlaps this one's (consolidation).
        siblings = []
        if consolidate:
            for sid, so in self.options.items():
                if sid == cid:
                    continue
                rho = self._mask_overlap(opt["m_a"], so["m_a"])
                if 0.1 < rho < 0.95:
                    siblings.append(sid)
            siblings = siblings[:3]

        for _ in range(n_steps):
            terms = []
            idx = self.buffer.sample_idx(self.cfg.batch_size)
            r, d, _ = self._option_reward_done(arr["next_obs"][idx], cid, arr["boundary"][idx])
            terms.append({"obs": arr["obs"][idx], "act": arr["act"][idx],
                          "next_obs": arr["next_obs"][idx], "reward": r, "done": d,
                          "m_a": opt["m_a"], "m_c": opt["m_c"]})
            for sid in siblings:
                sidx = self.buffer.sample_idx(self.cfg.batch_size)
                so = self.options[sid]
                sr, sd, _ = self._option_reward_done(arr["next_obs"][sidx], sid, arr["boundary"][sidx])
                terms.append({"obs": arr["obs"][sidx], "act": arr["act"][sidx],
                              "next_obs": arr["next_obs"][sidx], "reward": sr, "done": sd,
                              "m_a": so["m_a"], "m_c": so["m_c"]})
            self._ddpg_step(terms)

    @torch.no_grad()
    def act_option(self, obs_np: np.ndarray, cid: int, noise: float = 0.0) -> np.ndarray:
        """Greedy action of option `cid` (its actor mask)."""
        m_a = self.options[cid]["m_a"]
        obs = torch.tensor(obs_np, device=self.device).unsqueeze(0)
        self.actor.eval()
        a = self.actor(obs, m_a).squeeze(0).cpu().numpy()
        self.actor.train()
        if noise > 0:
            a = a + np.random.randn(self.cfg.act_dim).astype(np.float32) * noise
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    @torch.no_grad()
    def membership_np(self, obs_np: np.ndarray) -> int:
        obs = torch.tensor(np.asarray(obs_np, np.float32), device=self.device)
        return int(self.registry.membership(obs)[0])

    # ---- checkpoint ----------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "cfg": self.cfg,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_t": self.actor_t.state_dict(),
            "critic_t": self.critic_t.state_dict(),
            "m_explore_a": self.m_explore_a, "m_explore_c": self.m_explore_c,
            "options": self.options,
            "centroids": self.registry.centroids,
            "centroid_pos": self.registry.pos,
            "centroid_pos_count": self.registry.pos_count,
            "centroid_counts": self.registry.counts,
            "recode_state": self.recode.state,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, sd: dict):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])
        self.actor_t.load_state_dict(sd["actor_t"])
        self.critic_t.load_state_dict(sd["critic_t"])
        self.m_explore_a = sd["m_explore_a"]
        self.m_explore_c = sd["m_explore_c"]
        self.options = sd["options"]
        self.registry.centroids = sd["centroids"]
        self.registry.pos = sd["centroid_pos"]
        self.registry.pos_count = sd["centroid_pos_count"]
        self.registry.counts = sd["centroid_counts"]
        self.recode.set_state(sd["recode_state"])
        self.epsilon = sd["epsilon"]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "OptionsAgent":
        sd = torch.load(path, map_location=map_location, weights_only=False)
        cfg: AgentConfig = sd["cfg"]
        cfg.device = map_location
        agent = cls(cfg)
        agent.load_state_dict(sd)
        return agent
