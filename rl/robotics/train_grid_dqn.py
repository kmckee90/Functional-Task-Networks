"""Sequential DQN on grid-world reward-location tasks (FTN backbone + masked replay).

Sister file to train_grid.py — same env, same continual-learning protocol,
same eval procedure, but DQN instead of PPO. Because DQN is off-policy, the
FTN mask-overlap-filtered cross-task replay (the piece that was awkward to
port to PPO) is fully wired up here: archived past-task transitions stay
valid as TD-update targets under the mask they were collected under, and
when the current mask shares neurons with an archived mask in (ρ_low, ρ_high]
we run an extra update on that archive so gradients mix through the shared
neurons.

Per task:
    1. Exploration: run up to explore_episodes rollouts; stop on first reward.
    2. Mask reconfig on the rewarded support, **DQN TD target**: optimize
       the Q-mask to minimize MSE(Q(s,a)[mask], r + γ·max_a' Q_target(s',a')[mask]).
    3. Exploitation: K episodes; per-step DQN TD updates from the current
       replay ringbuffer, plus mask-overlap-filtered updates on archived
       past-task transitions.
    4. Archive the phase's transitions tagged with the phase's mask.

Eval matches train_grid.py: explore until first reward → run eval_episodes
fresh cold-started AnnealedSTE reconfigs on the same support, one greedy
episode under each → mean success rate.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ftn import ParallelMLP, anneal_ste_mask, anneal_ste_mask_batched, kwta

from train_grid_ppo import (
    GridEnv,
    build_grid_envs,
    compute_mc_returns,
    plot_matrix,
)

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_REWARDS = (
    (0, 0),
    (6, 0),
    (0, 6),
    (6, 6),
    (0, 4),
    (4, 0),
    (4, 6),
    (6, 4),
)


# ---------------------------------------------------------------------------
# DQNConfig glossary (parameter reference; fields are listed bare below)
# ---------------------------------------------------------------------------
# -- environment (field names must match the GridEnv constructor) --
# grid_h / grid_w         : grid dimensions.
# reward_locations        : per-task reward cells (one phase per location).
# start_pos               : fixed agent start cell each episode.
# step_penalty            : per-step negative reward (encourages short paths).
# terminal_reward         : reward for reaching the goal cell.
# max_episode_steps       : episode truncation horizon.
#
# -- experiment --
# seed                    : RNG seed for python/numpy/torch and env sampling.
# explore_episodes        : cap on exploration episodes per task / per eval cell.
# n_successful_explore    : stop exploration after this many rewarded episodes.
# n_trials_per_task       : K — exploitation episodes run after the boundary reconfig.
# eval_episodes           : greedy rollouts per (phase, task) eval cell.
# output_path             : .npz/.png results path.
# log_every               : print a training summary every this many trials.
#
# -- DQN --
# gamma                   : discount factor for TD targets and MC returns.
# tau                     : Polyak target-network averaging rate.
# q_lr                    : Adam learning rate for the Q-net.
# batch_size              : transitions per gradient step.
# replay_capacity         : ring-buffer capacity (current-task transitions).
# updates_per_step        : DQN gradient steps per env step.
# eps_train               : fixed ε-greedy exploration rate throughout training.
#
# -- FTN backbone (single Q-net mask) --
# ftn_hidden_size         : H — number of parallel sub-networks.
# ftn_inner_size          : width inside each parallel MLP.
# ftn_layers              : hidden layers inside each parallel MLP.
# ftn_dropout_p           : dropout on the per-neuron output vector (train-time only).
# ftn_k                   : active sub-networks selected by the KWTA mask.
#
# -- AnnealedSTE mask reconfiguration --
# reconfig_steps          : inner optimization steps per mask search.
# reconfig_lr             : learning rate for the soft-mask logits.
# reconfig_n              : support-batch cap (subsampled rewarded transitions).
# mask_init_noise_scale   : cold-start logit init = noise_scale * randn(H).
# mask_warm_start         : warm-start logits from the current mask (eval cold-starts
#                           regardless).
# reconfig_objective      : Q-mask objective; one of:
#     "td"          — MSE(Q(s,a)[m], r + γ·(1-done)·max_a' Q_target(s',a')[m])
#     "mse_returns" — MSE(Q(s,a_t)[m], G_t); fixed MC-return target (no bootstrap),
#                     so the mask makes Q match the rewarded-trajectory return.
#     "cos_returns" — −cos(Q(s,a_t)[m], G_t); scale-invariant, uncentered (an
#                     all-positive Q scores high against an all-positive G).
#     "r_returns"   — −Pearson(Q(s,a_t)[m], G_t); centered cosine. Scale- AND
#                     offset-invariant: only the variation of Q across states
#                     needs to match the variation of G.   (default)
# reconfig_gamma          : discount for MC returns G_t; None → fall back to gamma.
#                           Lowering sharpens G_t near the rewarded step; raising
#                           it (toward 1) propagates credit further back.
# reconfig_every_trials   : mid-phase reconfig cadence (trials); 0 disables (mask
#                           only set at the boundary).
# reconfig_support        : mid-phase reconfig support source:
#     "recent"         — the last recent_eps_for_reconfig training episodes
#                        (collected under the ε-greedy policy).
#     "random_explore" — eval protocol: roll out up to explore_episodes random
#                        episodes, stop on the first rewarded one, use those.
# recent_eps_for_reconfig : # recent episodes used when reconfig_support="recent".
#
# -- mask-overlap-filtered cross-task replay --
# mask_replay_n_per_step  : filtered past-task replay updates per gradient step.
# rho_low / rho_high      : keep a past phase only if mask overlap ρ ∈ (rho_low, rho_high].
#
# -- exploration / eval-time mask recovery --
# warmup_policy           : explore policy after task 0 — "random" or "prev_mask".
#
# -- bootstrapped exploration (eval-time support collection) --
# explore_method          : how the eval-time rewarded support is gathered
#                           (`_eval_with_reconfig`); eval then runs the batched
#                           cold-start reconfigs on that support either way.
#                           CLI override: --bootstrap selects "bootstrap".
#     "random"    — collect random / warmup_policy rollouts until
#                   n_successful_explore rewarded episodes. Off-policy: returns
#                   come from random rollouts, so the Q-vs-returns target is noisy.
#     "bootstrap" — on-policy collection. Random actions until the first reward;
#                   thereafter EVERY exploration episode is preceded by a fresh
#                   COLD-START Q-mask reconfig on the *cumulative* rewarded
#                   support (never committed), so each episode draws a fresh
#                   plausible mask and a bad early mask can't trap exploration.
#                   Rewarded episodes accumulate. Stops at n_successful_explore
#                   rewarded episodes or explore_episodes total.
# eps_explore             : ε-greedy rate for the "bootstrap" collection rollouts
#                           (separate from eps_train).
# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    # --- environment (field names must match GridEnv constructor) ---
    grid_h: int = 7
    grid_w: int = 7
    reward_locations: tuple = DEFAULT_REWARDS
    start_pos: tuple = (3, 3)
    step_penalty: float = 0.01
    terminal_reward: float = 1.0
    max_episode_steps: int = 20

    # --- experiment ---
    seed: int = 0
    explore_episodes: int = 200
    n_successful_explore: int = 4
    n_trials_per_task: int = 250
    eval_episodes: int = 40
    output_path: str = "results_grid_dqn.npz"
    log_every: int = 50

    # --- DQN ---
    gamma: float = 0.90
    tau: float = 0.1
    q_lr: float = 5e-4
    batch_size: int = 64
    replay_capacity: int = 100_000
    updates_per_step: int = 1
    eps_train: float = 0.1

    # --- FTN backbone (single Q-net mask) ---
    ftn_hidden_size: int = 256
    ftn_inner_size: int = 8
    ftn_layers: int = 8
    ftn_dropout_p: float = 0.1
    ftn_k: int = 32

    # --- AnnealedSTE ---
    reconfig_steps: int = 50
    reconfig_lr: float = 0.03
    reconfig_n: int = 4096
    mask_init_noise_scale: float = 0.0001
    mask_warm_start: bool = True
    reconfig_objective: str = "r_returns"
    reconfig_gamma: float | None = None
    reconfig_every_trials: int = 10
    reconfig_support: str = "recent"
    recent_eps_for_reconfig: int = 10

    # --- mask-overlap-filtered cross-task replay ---
    mask_replay_n_per_step: int = 1
    rho_low: float = 0.2
    rho_high: float = 0.8

    # --- exploration / eval-time mask recovery ---
    warmup_policy: str = "random"

    # --- bootstrapped exploration (explore_method="bootstrap") ---
    explore_method: str = "bootstrap"
    eps_explore: float = 0.5


def smoke_dqn_config() -> DQNConfig:
    return DQNConfig(
        grid_h=4,
        grid_w=4,
        reward_locations=((3, 3), (0, 3), (3, 0)),
        start_pos=(0, 0),
        max_episode_steps=40,
        explore_episodes=10,
        n_trials_per_task=80,
        eval_episodes=4,
        batch_size=32,
        replay_capacity=2_000,
        ftn_hidden_size=32,
        ftn_inner_size=4,
        ftn_layers=2,
        ftn_k=8,
        reconfig_steps=10,
        reconfig_n=64,
        recent_eps_for_reconfig=4,
        log_every=20,
        output_path="results_grid_dqn_smoke.npz",
    )


# ---------------------------------------------------------------------------
# FTN Q-network
# ---------------------------------------------------------------------------


class FTNQNetwork(nn.Module):
    """ParallelMLP bank → KWTA-masked linear → (B, n_actions) Q-values."""

    def __init__(self, obs_dim, n_actions, H, layers, inner_size, dropout_p):
        super().__init__()
        self.H = H
        self.parallel = ParallelMLP(obs_dim, 1, H, layers, inner_size)
        self.linear = nn.Linear(H, n_actions, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, obs, mask):
        z = self.parallel(obs)
        z = self.dropout(z)
        return self.linear(z * mask.reshape(1, -1))

    def forward_batched(self, obs, masks):
        """Batched forward: obs (B, D), masks (E, H) -> (E, B, n_actions).
        The parallel-MLP forward and dropout run once on the shared support;
        only the per-search mask multiply and final linear are duplicated."""
        z = self.parallel(obs)  # (B, H)
        z = self.dropout(z)
        zm = z.unsqueeze(0) * masks.unsqueeze(1)  # (E, B, H)
        return self.linear(zm)  # (E, B, n_actions)


# ---------------------------------------------------------------------------
# Mask-overlap-filtered replay buffer (single-mask DQN variant)
# ---------------------------------------------------------------------------


class MaskedReplayBufferDQN:
    """Current-phase ringbuffer + per-completed-phase archive tagged with the
    Q-mask the phase was collected under. Adapted from ftn.MaskedReplayBuffer;
    discrete actions stored as int64, single mask m per phase instead of
    (m_a, m_c).

    Storage lives on the default device (GPU when available). Episodes are
    bulk-added so we pay one H2D copy per field per episode rather than per
    transition. `sample` and `sample_filtered` return GPU tensor views with
    no extra copy.
    """

    def __init__(self, capacity, obs_dim):
        self.capacity = capacity
        self.obs = torch.zeros(capacity, obs_dim)
        self.act = torch.zeros(capacity, dtype=torch.long)
        self.rew = torch.zeros(capacity)
        self.next_obs = torch.zeros(capacity, obs_dim)
        self.done = torch.zeros(capacity)
        self.ptr = 0
        self.size = 0
        self.past: list[dict] = []
        # Cached stack of past phase masks for vectorized overlap; refreshed
        # on archive_phase. Shape (P, H) once any phase has been archived.
        self._past_masks: torch.Tensor | None = None

    def add_episode(self, ep: dict):
        """Bulk-add a full episode. One H2D copy per field instead of L."""
        L = ep["length"]
        if L == 0:
            return
        obs_t = torch.from_numpy(ep["obs"])
        act_t = torch.from_numpy(ep["act"])
        rew_t = torch.from_numpy(ep["reward"])
        next_t = torch.from_numpy(ep["next_obs"])
        done_t = torch.from_numpy(ep["done"])
        end = self.ptr + L
        if end <= self.capacity:
            self.obs[self.ptr : end].copy_(obs_t)
            self.act[self.ptr : end].copy_(act_t)
            self.rew[self.ptr : end].copy_(rew_t)
            self.next_obs[self.ptr : end].copy_(next_t)
            self.done[self.ptr : end].copy_(done_t)
        else:
            first = self.capacity - self.ptr
            tail = L - first
            self.obs[self.ptr :].copy_(obs_t[:first])
            self.obs[:tail].copy_(obs_t[first:])
            self.act[self.ptr :].copy_(act_t[:first])
            self.act[:tail].copy_(act_t[first:])
            self.rew[self.ptr :].copy_(rew_t[:first])
            self.rew[:tail].copy_(rew_t[first:])
            self.next_obs[self.ptr :].copy_(next_t[:first])
            self.next_obs[:tail].copy_(next_t[first:])
            self.done[self.ptr :].copy_(done_t[:first])
            self.done[:tail].copy_(done_t[first:])
        self.ptr = end % self.capacity
        self.size = min(self.size + L, self.capacity)

    def sample(self, batch):
        idx = torch.randint(0, self.size, (batch,))
        return (
            self.obs[idx],
            self.act[idx],
            self.rew[idx],
            self.next_obs[idx],
            self.done[idx],
        )

    def clear_current(self):
        self.ptr = 0
        self.size = 0

    def archive_phase(self, m: torch.Tensor):
        if self.size == 0:
            return
        self.past.append(
            {
                "obs": self.obs[: self.size].clone(),
                "act": self.act[: self.size].clone(),
                "rew": self.rew[: self.size].clone(),
                "next_obs": self.next_obs[: self.size].clone(),
                "done": self.done[: self.size].clone(),
                "m": m.detach().clone(),
            }
        )
        self._past_masks = torch.stack([ph["m"] for ph in self.past])

    def sample_filtered(
        self, m_cur: torch.Tensor, n: int, rho_low: float, rho_high: float
    ):
        """Pick one past phase whose mask overlap with m_cur falls in
        (rho_low, rho_high], sample n transitions from it. Returns
        (batch_tuple, m_old) or None if no phase qualifies."""
        if not self.past or self._past_masks is None:
            return None
        # Vectorized overlap across all past phases in one op.
        denom = m_cur.sum().clamp(min=1)
        rho = (self._past_masks * m_cur).sum(dim=-1) / denom  # (P,)
        ok = (rho > rho_low) & (rho <= rho_high)
        candidates = ok.nonzero(as_tuple=True)[0]  # (k,)
        n_cand = int(candidates.numel())  # one sync
        if n_cand == 0:
            return None
        pick_local = int(torch.randint(0, n_cand, ()).item())  # one sync
        ph_idx = int(candidates[pick_local].item())
        ph = self.past[ph_idx]
        N = ph["obs"].shape[0]
        idx = torch.randint(0, N, (n,))
        batch = (
            ph["obs"][idx],
            ph["act"][idx],
            ph["rew"][idx],
            ph["next_obs"][idx],
            ph["done"][idx],
        )
        return batch, ph["m"]


# ---------------------------------------------------------------------------
# FTN DQN agent
# ---------------------------------------------------------------------------


class FTNDQN:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig):
        self.cfg = cfg
        self.n_actions = n_actions
        H, k = cfg.ftn_hidden_size, cfg.ftn_k
        L, IS, D = cfg.ftn_layers, cfg.ftn_inner_size, cfg.ftn_dropout_p

        self.q = FTNQNetwork(obs_dim, n_actions, H, L, IS, D)
        self.q_target = FTNQNetwork(obs_dim, n_actions, H, L, IS, D)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.q_opt = torch.optim.Adam(self.q.parameters(), lr=cfg.q_lr)

        self.m = kwta(torch.randn(H), k=k)
        self.m_logits: torch.Tensor | None = None
        self.stored_m: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def act(self, obs_np, mask=None, eps: float = 0.0):
        """ε-greedy under the given mask. Returns (action, 0.0, 0.0) so the
        signature matches the policy_fn contract used by `_collect_episode_dqn`."""
        m = self.m if mask is None else mask
        if eps > 0.0 and np.random.random() < eps:
            return int(np.random.randint(self.n_actions)), 0.0, 0.0
        obs = torch.tensor(obs_np).unsqueeze(0)
        q = self.q(obs, m).squeeze(0)
        return int(q.argmax().item()), 0.0, 0.0

    def reconfig_masks(self, support, commit: bool = True):
        """Mask reconfig (single mask for the Q-net). Objective selected by
        cfg.reconfig_objective:

        - "td": MSE(Q(s,a)[m], r + γ·(1-done)·max_a' Q_target(s',a')[m]).
          Bootstrap target uses target net under the same soft mask, in
          no_grad so gradients only flow through the prediction.
        - "cos_returns": −cosine(Q(s,a_t)[m], G_t) over the support. Scale-
          invariant — picks a mask whose Q ranks (s_t, a_t) the same way
          MC returns do, without requiring Q to share magnitude with G.

        support = (obs, act, rew, next_obs, done, returns).
        """
        obs, act, rew, next_obs, done, returns = support
        cfg = self.cfg
        H, k = cfg.ftn_hidden_size, cfg.ftn_k
        warm = cfg.mask_warm_start
        init = self.m_logits if warm else None
        objective = cfg.reconfig_objective

        def q_loss(soft_m):
            q_all = self.q(obs, soft_m)
            # q_sa = q_all.gather(1, act.unsqueeze(-1).long()).squeeze(-1)
            # breakpoint()
            q_sa = q_all.max(-1).values
            if objective == "td":
                with torch.no_grad():
                    q_next = self.q_target(next_obs, soft_m).max(dim=-1).values
                    target = rew + cfg.gamma * (1.0 - done) * q_next
                return F.mse_loss(q_sa, target)
            elif objective == "mse_returns":
                return F.mse_loss(q_sa, returns)
            elif objective == "cos_returns":
                cos = F.cosine_similarity(
                    q_sa.unsqueeze(0), returns.unsqueeze(0), dim=1
                ).squeeze(0)
                return -cos
            elif objective == "r_returns":
                q_c = q_sa - q_sa.mean()
                r_c = returns - returns.mean()
                r = F.cosine_similarity(
                    q_c.unsqueeze(0), r_c.unsqueeze(0), dim=1
                ).squeeze(0)
                return -r
            else:
                raise ValueError(f"unknown reconfig_objective={objective!r}")

        m_new, m_logits_new, l_init, l_final = anneal_ste_mask(
            H,
            k,
            q_loss,
            cfg.reconfig_steps,
            cfg.reconfig_lr,
            init_logits=init,
            noise_scale=cfg.mask_init_noise_scale,
        )
        stats = {"q_init": l_init, "q_final": l_final}
        if commit:
            self.m = m_new
            self.m_logits = m_logits_new
        return m_new, stats

    def reconfig_masks_batched(self, support, n_masks: int):
        """Run `n_masks` independent cold-started STE searches in a single
        batched pass. All searches share the parallel-MLP forward on the
        support; only the per-search mask multiply and final linear are
        duplicated. Returns (n_masks, H) hard masks and stats."""
        obs, act, rew, next_obs, done, returns = support
        cfg = self.cfg
        H, k = cfg.ftn_hidden_size, cfg.ftn_k
        objective = cfg.reconfig_objective
        E = n_masks

        def q_loss_batched(soft_m):
            # soft_m: (E, H). Returns (E,).
            q_all = self.q.forward_batched(obs, soft_m)  # (E, B, n_actions)
            act_e = act.long().unsqueeze(0).expand(E, -1).unsqueeze(-1)
            q_sa = q_all.gather(2, act_e).squeeze(-1)  # (E, B)
            if objective == "td":
                with torch.no_grad():
                    q_next_all = self.q_target.forward_batched(next_obs, soft_m)
                    q_next = q_next_all.max(dim=-1).values  # (E, B)
                    target = (
                        rew.unsqueeze(0)
                        + cfg.gamma * (1.0 - done.unsqueeze(0)) * q_next
                    )
                return F.mse_loss(q_sa, target, reduction="none").mean(dim=1)
            elif objective == "mse_returns":
                r_e = returns.unsqueeze(0).expand_as(q_sa)
                return F.mse_loss(q_sa, r_e, reduction="none").mean(dim=1)
            elif objective == "cos_returns":
                r_e = returns.unsqueeze(0).expand_as(q_sa)
                cos = F.cosine_similarity(q_sa, r_e, dim=1)  # (E,)
                return -cos
            elif objective == "r_returns":
                q_c = q_sa - q_sa.mean(dim=1, keepdim=True)  # (E, B)
                r_c = (returns - returns.mean()).unsqueeze(0).expand_as(q_sa)
                r = F.cosine_similarity(q_c, r_c, dim=1)  # (E,)
                return -r
            else:
                raise ValueError(f"unknown reconfig_objective={objective!r}")

        masks, _, l_init, l_final = anneal_ste_mask_batched(
            H,
            k,
            q_loss_batched,
            cfg.reconfig_steps,
            cfg.reconfig_lr,
            n_search=E,
            noise_scale=cfg.mask_init_noise_scale,
        )
        return masks, {"q_init": l_init, "q_final": l_final}

    def update(self, batch, mask: torch.Tensor | None = None):
        """One DQN TD step under the given mask. For mask-overlap-filtered
        cross-task replay, pass the mask the transitions were collected under
        — gradients only flow through neurons active under that mask, so
        shared neurons with the current mask receive accumulated updates.

        Returns nothing: the loss scalar is not read by callers, so we skip
        the `.item()` sync (this runs ~96K times per full sweep).
        """
        m = self.m if mask is None else mask
        obs, act, rew, next_obs, done = batch
        with torch.no_grad():
            q_next = self.q_target(next_obs, m).max(dim=-1).values
            target = rew + self.cfg.gamma * (1.0 - done) * q_next
        q_all = self.q(obs, m)
        q_sa = q_all.gather(1, act.unsqueeze(-1).long()).squeeze(-1)
        loss = F.mse_loss(q_sa, target)
        self.q_opt.zero_grad()
        loss.backward()
        self.q_opt.step()
        with torch.no_grad():
            tau = self.cfg.tau
            for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                pt.mul_(1 - tau).add_(p, alpha=tau)


# ---------------------------------------------------------------------------
# Episode collection (DQN-specific: needs next_obs in the dict)
# ---------------------------------------------------------------------------


def _collect_episode_dqn(env: GridEnv, policy_fn, max_steps: int | None = None) -> dict:
    """Run one episode using policy_fn(obs) -> (action, _, _). Returns dict
    of numpy arrays — includes next_obs for replay-buffer pushes."""
    if max_steps is None:
        max_steps = env.max_steps
    obs_buf, act_buf, r_buf, next_buf, d_buf = [], [], [], [], []
    obs, _ = env.reset()
    success = 0
    for _ in range(max_steps):
        a, _, _ = policy_fn(obs)
        ns, r, term, trunc, info = env.step(a)
        obs_buf.append(obs)
        act_buf.append(a)
        r_buf.append(r)
        next_buf.append(ns)
        d_buf.append(float(term or trunc))
        obs = ns
        if float(info.get("success", 0.0)) > 0.5:
            success = 1
        if term or trunc:
            break
    return {
        "obs": np.stack(obs_buf).astype(np.float32),
        "act": np.asarray(act_buf, dtype=np.int64),
        "next_obs": np.stack(next_buf).astype(np.float32),
        "reward": np.asarray(r_buf, dtype=np.float32),
        "done": np.asarray(d_buf, dtype=np.float32),
        "success": success,
        "length": len(obs_buf),
    }


def _push_episode_to_replay(replay: MaskedReplayBufferDQN, ep: dict):
    replay.add_episode(ep)


def _support_from_episodes_dqn(
    episodes: list[dict],
    gamma: float,
    n_max: int,
) -> tuple[torch.Tensor, ...]:
    """Concatenate episode transitions and subsample ≤ n_max rows. MC returns
    are computed on the full concatenation (so episode boundaries are respected
    via `done`) before subsampling. Returns (obs, act, rew, next_obs, done,
    returns) — TD objective uses (rew, next_obs, done); cos_returns uses
    (returns)."""
    obs = np.concatenate([e["obs"] for e in episodes], axis=0)
    act = np.concatenate([e["act"] for e in episodes], axis=0)
    rew = np.concatenate([e["reward"] for e in episodes], axis=0)
    next_obs = np.concatenate([e["next_obs"] for e in episodes], axis=0)
    done = np.concatenate([e["done"] for e in episodes], axis=0)
    returns = compute_mc_returns(rew, done, gamma)
    N = obs.shape[0]
    if N > n_max:
        idx = np.random.choice(N, size=n_max, replace=False)
    else:
        idx = np.arange(N)
    return (
        torch.tensor(obs[idx]),
        torch.tensor(act[idx], dtype=torch.long),
        torch.tensor(rew[idx]),
        torch.tensor(next_obs[idx]),
        torch.tensor(done[idx]),
        torch.tensor(returns[idx]),
    )


# ---------------------------------------------------------------------------
# Exploration + unified eval
# ---------------------------------------------------------------------------


def _explore_for_reward(
    env: GridEnv,
    agent: FTNDQN,
    max_episodes: int,
    policy: str,
    prev_m: torch.Tensor | None = None,
    n_successful: int = 1,
) -> tuple[list[dict], int]:
    """Run up to max_episodes rollouts; stop after `n_successful` rewarded ones."""
    episodes = []
    n_success = 0
    n_actions = env.n_actions
    for _ in range(max_episodes):
        if policy == "random":

            def policy_fn(obs):
                return int(np.random.randint(n_actions)), 0.0, 0.0
        else:  # "prev_mask"

            def policy_fn(obs):
                return agent.act(obs, mask=prev_m, eps=0.0)

        ep = _collect_episode_dqn(env, policy_fn)
        episodes.append(ep)
        if ep["success"]:
            n_success += 1
            if n_success >= n_successful:
                break
    return episodes, n_success


def _bootstrap_collect_rewarded(
    env: GridEnv,
    agent: FTNDQN,
    cfg: DQNConfig,
    gamma_rc: float,
) -> list[dict]:
    """Collect rewarded episodes for eval-time recovery (`explore_method='bootstrap'`).

    Until the first reward is found there is no support, so exploration uses
    random actions. Once any rewarded episode exists, EVERY subsequent
    exploration episode is preceded by a fresh COLD-START Q-mask reconfiguration
    on the full cumulative rewarded support (commit=False — no single mask is
    committed). With low support the recovered masks are unreliable, so we
    deliberately avoid committing to any one of them: each episode re-derives
    the policy from scratch, drawing a fresh mask (cold-start init noise) from
    the same support, so a bad early mask can't trap exploration when it is
    still failing to find reward. Rewarded episodes accumulate into the support.
    Stops at cfg.n_successful_explore rewarded episodes or cfg.explore_episodes
    total episodes. The caller clears/restores the warm-start logits so every
    reconfig here cold-starts. Returns the rewarded episodes."""
    rewarded: list[dict] = []
    n_actions = env.n_actions
    for _ in range(cfg.explore_episodes):
        if len(rewarded) >= cfg.n_successful_explore:
            break
        if not rewarded:
            # No support yet: uniform random actions.
            def policy_fn(_obs):
                return int(np.random.randint(n_actions)), 0.0, 0.0
        else:
            # Cold-start reconfig on the cumulative support before every episode.
            support = _support_from_episodes_dqn(rewarded, gamma_rc, cfg.reconfig_n)
            m_cur = agent.reconfig_masks(support, commit=False)[0]

            def policy_fn(obs, _m=m_cur):
                return agent.act(obs, mask=_m, eps=cfg.eps_explore)

        ep = _collect_episode_dqn(env, policy_fn)
        if ep["success"]:
            rewarded.append(ep)
    return rewarded


def _eval_with_reconfig(env: GridEnv, agent: FTNDQN, cfg: DQNConfig) -> float:
    """Gather a rewarded support set, then run `cfg.eval_episodes` fresh
    cold-started mask reconfigs on the rewarded-only support, one greedy episode
    each; mean success is the cell score. Support collection depends on
    `cfg.explore_method`:
      "random"    — random / warmup_policy rollouts until cfg.n_successful_explore
                    rewarded episodes.
      "bootstrap" — on-policy: each exploration episode after the first reward
                    acts under a fresh cold-start reconfig on the cumulative
                    rewarded support (`_bootstrap_collect_rewarded`).

    The eval_episodes cold-started reconfigs are batched into one STE search
    (each row of the batched search is an independent reconfig sharing the
    parallel-MLP forward on the support)."""
    gamma_rc = cfg.reconfig_gamma if cfg.reconfig_gamma is not None else cfg.gamma

    # Clear warm-start logits for the whole eval so every reconfig below
    # cold-starts and is not contaminated by whichever task was trained last;
    # restore at the end. No mask is committed during eval (commit=False), so
    # agent.m needs no save/restore.
    saved_logits = agent.m_logits
    agent.m_logits = None

    if cfg.explore_method == "bootstrap":
        rewarded = _bootstrap_collect_rewarded(env, agent, cfg, gamma_rc)
    else:
        eps, _ = _explore_for_reward(
            env,
            agent,
            cfg.explore_episodes,
            cfg.warmup_policy,
            prev_m=agent.m.clone(),
            n_successful=cfg.n_successful_explore,
        )
        rewarded = [e for e in eps if e["success"]]

    if not rewarded:
        agent.m_logits = saved_logits
        return 0.0

    support = _support_from_episodes_dqn(rewarded, gamma_rc, cfg.reconfig_n)
    masks_E, _ = agent.reconfig_masks_batched(support, n_masks=cfg.eval_episodes)

    n_eval_succ = 0
    for e in range(cfg.eval_episodes):
        m_e = masks_E[e]

        def policy_fn(obs, _m=m_e):
            return agent.act(obs, mask=_m, eps=0.0)

        ep = _collect_episode_dqn(env, policy_fn)
        if ep["success"]:
            n_eval_succ += 1
    agent.m_logits = saved_logits
    return n_eval_succ / cfg.eval_episodes


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train_grid_dqn(cfg: DQNConfig, args):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    envs = build_grid_envs(cfg)
    obs_dim = envs[0].obs_dim
    n_actions = envs[0].n_actions

    agent = FTNDQN(obs_dim, n_actions, cfg)
    replay = MaskedReplayBufferDQN(cfg.replay_capacity, obs_dim)

    N = len(envs)
    matrix = np.zeros((N, N), dtype=np.float32)

    gamma_rc = cfg.reconfig_gamma if cfg.reconfig_gamma is not None else cfg.gamma

    t_start = time.time()
    for ti, env in enumerate(envs):
        print(f"\n=== Phase {ti}/{N - 1}: {env.name} (reward {env.reward_pos}) ===")
        replay.clear_current()
        recent_eps: deque = deque(maxlen=cfg.recent_eps_for_reconfig)

        # 1) Exploration ---------------------------------------------------
        prev_m = agent.m.clone()
        policy = "random" if ti == 0 else cfg.warmup_policy
        explore_eps, n_succ = _explore_for_reward(
            env,
            agent,
            cfg.explore_episodes,
            policy,
            prev_m=prev_m,
        )
        n_steps_exp = sum(e["length"] for e in explore_eps)
        print(
            f"  exploration ({policy}): {len(explore_eps)} eps, "
            f"{n_steps_exp} steps, {n_succ} successful"
        )

        # 2) Mask reconfig (objective = cfg.reconfig_objective) -----------
        if n_succ > 0:
            support = _support_from_episodes_dqn(
                explore_eps,
                gamma_rc,
                cfg.reconfig_n,
            )
            _, stats = agent.reconfig_masks(support, commit=True)
            print(f"  reconfig: Q-loss {stats['q_init']:.3f} -> {stats['q_final']:.3f}")
        else:
            print("  WARN: no reward found during exploration; keeping prior mask")
        agent.stored_m[ti] = agent.m.detach().clone()

        # Push exploration trajectories into the replay ringbuffer so the
        # first DQN updates have something to chew on.
        for ep in explore_eps:
            _push_episode_to_replay(replay, ep)
            recent_eps.append(ep)

        # 3) Exploitation: K episodes, per-step DQN + cross-task replay ----
        train_returns, train_lengths, train_succs = [], [], []
        for trial in range(cfg.n_trials_per_task):

            def policy_fn(obs):
                return agent.act(obs, eps=cfg.eps_train)

            ep = _collect_episode_dqn(env, policy_fn)
            _push_episode_to_replay(replay, ep)
            recent_eps.append(ep)
            train_returns.append(float(ep["reward"].sum()))
            train_lengths.append(ep["length"])
            train_succs.append(ep["success"])

            # Gradient updates: one DQN update per env step within the episode.
            if replay.size >= cfg.batch_size:
                n_updates = ep["length"] * cfg.updates_per_step
                for _ in range(n_updates):
                    agent.update(replay.sample(cfg.batch_size))
                    for _ in range(cfg.mask_replay_n_per_step):
                        sf = replay.sample_filtered(
                            agent.m,
                            cfg.batch_size,
                            cfg.rho_low,
                            cfg.rho_high,
                        )
                        if sf is None:
                            break
                        old_batch, m_old = sf
                        agent.update(old_batch, mask=m_old)

            # Periodic mid-phase reconfig (DQN-TD target).
            if (
                cfg.reconfig_every_trials > 0
                and (trial + 1) % cfg.reconfig_every_trials == 0
            ):
                if cfg.reconfig_support == "random_explore":
                    rc_eps, rc_succ = _explore_for_reward(
                        env, agent, cfg.explore_episodes, "random"
                    )
                    support_eps = rc_eps if rc_succ > 0 else []
                elif cfg.reconfig_support == "recent":
                    support_eps = list(recent_eps)
                else:
                    raise ValueError(
                        f"unknown reconfig_support={cfg.reconfig_support!r}"
                    )
                if len(support_eps) > 0:
                    support = _support_from_episodes_dqn(
                        support_eps,
                        gamma_rc,
                        cfg.reconfig_n,
                    )
                    agent.reconfig_masks(support, commit=True)

            if (trial + 1) % cfg.log_every == 0:
                k = min(cfg.log_every, len(train_returns))
                mean_r = float(np.mean(train_returns[-k:]))
                mean_sr = float(np.mean(train_succs[-k:]))
                mean_l = float(np.mean(train_lengths[-k:]))
                elapsed = time.time() - t_start
                print(
                    f"  trial {trial + 1:>4d}  ep_ret={mean_r:6.2f}  "
                    f"sr={mean_sr:.2f}  len={mean_l:5.1f}  ({elapsed:.0f}s)"
                )

        # 4) Archive phase for cross-task replay --------------------------
        agent.stored_m[ti] = agent.m.detach().clone()
        replay.archive_phase(agent.m)

        # 5) Eval all tasks (unified procedure) ---------------------------
        print(f"  --- Eval after phase {ti} ---")
        for tj in range(N):
            sr = _eval_with_reconfig(envs[tj], agent, cfg)
            tag = "<-" if tj == ti else "  "
            matrix[ti, tj] = sr
            print(
                f"    [{tag}] {envs[tj].name:10s} reward {envs[tj].reward_pos} "
                f"sr={sr:.2f}"
            )

        # np.savez(
        #     cfg.output_path,
        #     matrix=matrix,
        #     reward_locations=np.array(cfg.reward_locations),
        #     phases_done=ti + 1,
        # )

    print("\n=== Eval matrix (rows: training phase, cols: eval task) ===")
    for ti in range(N):
        row = " ".join(f"{matrix[ti, tj]:.2f}" for tj in range(N))
        print(f"  phase {ti} ({envs[ti].name}): {row}")

    # np.savez(
    #     cfg.output_path,
    #     matrix=matrix,
    #     reward_locations=np.array(cfg.reward_locations),
    #     phases_done=N,
    # )
    plot_matrix(
        matrix,
        [e.name for e in envs],
        cfg.output_path.replace(".npz", ".png"),
    )
    return matrix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test config")
    bootstrap_group = parser.add_mutually_exclusive_group()
    bootstrap_group.add_argument(
        "--bootstrap",
        dest="bootstrap",
        action="store_true",
        default=None,
        help="Use bootstrapped eval-time exploration (explore_method='bootstrap'): "
        "after the first reward, every exploration episode cold-start-reconfigs "
        "on the cumulative rewarded support.",
    )
    bootstrap_group.add_argument(
        "--no-bootstrap",
        dest="bootstrap",
        action="store_false",
        help="Force random eval-time exploration (explore_method='random').",
    )
    parser.add_argument("--name", default="")
    parser.add_argument("--seed", default="0")

    args = parser.parse_args()

    cfg = smoke_dqn_config() if args.smoke else DQNConfig()
    if args.bootstrap is not None:
        cfg.explore_method = "bootstrap" if args.bootstrap else "random"
    cfg.output_path = cfg.output_path.replace(".", f"_{args.name}_{args.seed}.")
    cfg.seed = int(args.seed)
    print("Config:", asdict(cfg))
    train_grid_dqn(cfg, args)
