"""Sequential PPO on grid-world reward-location tasks (FTN backbone).

Each task is just a different reward cell in an N×M grid; the reward is NOT
visible to the agent. Observation is the flattened grid with a 1 at the
agent's cell and 0 elsewhere. Four discrete actions (up/down/left/right).
Reward is -step_penalty per step plus +terminal_reward on landing on the
goal cell; episode terminates on goal or on max_episode_steps.

Per task:
    1. Exploration: run episodes (cap explore_episodes) using either uniform
       random actions or the prev-task policy, until the reward is found at
       least once. The resulting trajectories are the support set.
    2. Mask reconfig: optimize critic mask via MSE(V, MC-return) on the
       support batch; then actor mask via REINFORCE-style loss with
       return - V_baseline as advantage. Reuses anneal_ste_mask from ftn.py.
    3. Exploitation: K trials starting from start_pos, standard PPO updates.
    4. Archive the per-task mask for retrospective and head-start eval.

The whole point of the experiment is to test whether a *rewarded trajectory*
(not the observation alone) can identify the right task mask.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ftn import ParallelMLP, anneal_ste_mask, kwta
from torch.distributions import Categorical

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Default sequence of reward cells (row, col). Designed to be far from
# start_pos = (0, 0) and from each other, so masks are forced to differ.
DEFAULT_REWARDS = (
    (0, 0),
    (6, 0),
    (0, 6),
    (6, 6),
    (0, 4),
    # (4, 0),
    # (4, 6),
    # (6, 4),
)


@dataclass
class GridConfig:
    # --- environment ---
    grid_h: int = 7
    grid_w: int = 7
    reward_locations: tuple = DEFAULT_REWARDS
    start_pos: tuple = (3, 3)
    step_penalty: float = 0.01
    terminal_reward: float = 1.0
    max_episode_steps: int = 50

    # --- experiment ---
    seed: int = 0
    explore_episodes: int = 20  # max exploration episodes per task boundary
    n_trials_per_task: int = 500  # K: exploitation trials after reconfig
    eval_episodes: int = 20
    output_path: str = "results_grid.npz"
    log_every: int = 50

    # --- PPO ---
    gamma: float = 0.90
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    minibatch_size: int = 64
    rollout_length: int = 256  # collect ≥ this many steps before PPO update
    entropy_coef: float = 0.01
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3

    # --- FTN backbone ---
    ftn_hidden_size: int = 256
    ftn_inner_size: int = 8
    ftn_layers: int = 4
    ftn_dropout_p: float = 0.2
    ftn_k: int = 64

    # --- AnnealedSTE ---
    reconfig_steps: int = 30
    reconfig_lr: float = 0.3
    reconfig_n: int = 256
    mask_init_noise_scale: float = 0.001
    mask_warm_start: bool = False
    # Periodic reconfig during exploitation (every N trials). 0 = disabled.
    reconfig_every_trials: int = 1

    # --- exploration / head-start ---
    warmup_policy: str = "random"  # "random" or "prev_mask"


def smoke_grid_config() -> GridConfig:
    return GridConfig(
        grid_h=4,
        grid_w=4,
        reward_locations=((3, 3), (0, 3), (3, 0)),
        max_episode_steps=40,
        explore_episodes=10,
        n_trials_per_task=40,
        eval_episodes=4,
        rollout_length=64,
        minibatch_size=32,
        ftn_hidden_size=32,
        ftn_inner_size=4,
        ftn_layers=2,
        ftn_k=8,
        reconfig_steps=10,
        reconfig_n=64,
        head_start_explore_episodes=10,
        log_every=20,
        output_path="results_grid_smoke.npz",
    )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class GridEnv:
    """Discrete N×M grid with 4 actions; reward only on one (invisible) cell."""

    ACTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # up, down, left, right

    def __init__(
        self,
        H: int,
        W: int,
        reward_pos: tuple,
        start_pos: tuple,
        step_penalty: float,
        terminal_reward: float,
        max_steps: int,
        name: str = "",
    ):
        self.H = H
        self.W = W
        self.reward_pos = tuple(reward_pos)
        self.start_pos = tuple(start_pos)
        self.step_penalty = step_penalty
        self.terminal_reward = terminal_reward
        self.max_steps = max_steps
        self.name = name
        self.obs_dim = H * W
        self.n_actions = 4
        self.pos = self.start_pos
        self.step_count = 0

    def reset(self):
        self.pos = self.start_pos
        self.step_count = 0
        return self._obs(), {}

    def step(self, action: int):
        dr, dc = self.ACTIONS[action]
        nr = max(0, min(self.H - 1, self.pos[0] + dr))
        nc = max(0, min(self.W - 1, self.pos[1] + dc))
        self.pos = (nr, nc)
        self.step_count += 1
        on_goal = self.pos == self.reward_pos
        r = self.terminal_reward if on_goal else -self.step_penalty
        term = bool(on_goal)
        trunc = self.step_count >= self.max_steps
        return self._obs(), float(r), term, trunc, {"success": float(on_goal)}

    def _obs(self):
        o = np.zeros(self.H * self.W, dtype=np.float32)
        o[self.pos[0] * self.W + self.pos[1]] = 1.0
        return o


def build_grid_envs(cfg: GridConfig):
    return [
        GridEnv(
            cfg.grid_h,
            cfg.grid_w,
            rp,
            cfg.start_pos,
            cfg.step_penalty,
            cfg.terminal_reward,
            cfg.max_episode_steps,
            name=f"r{rp[0]}c{rp[1]}",
        )
        for rp in cfg.reward_locations
    ]


# ---------------------------------------------------------------------------
# FTN networks for discrete-action PPO
# ---------------------------------------------------------------------------


class FTNDiscreteActor(nn.Module):
    """Policy network: outputs logits over n_actions."""

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


class FTNValueCritic(nn.Module):
    """Value network: outputs scalar V(s) (state-only, no action input)."""

    def __init__(self, obs_dim, H, layers, inner_size, dropout_p):
        super().__init__()
        self.H = H
        self.parallel = ParallelMLP(obs_dim, 1, H, layers, inner_size)
        self.linear = nn.Linear(H, 1, bias=False)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, obs, mask):
        z = self.parallel(obs)
        z = self.dropout(z)
        return self.linear(z * mask.reshape(1, -1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Returns / GAE
# ---------------------------------------------------------------------------


def compute_mc_returns(
    rewards: np.ndarray, dones: np.ndarray, gamma: float
) -> np.ndarray:
    """Discounted MC returns. dones[t]==1 resets the running future-sum."""
    T = len(rewards)
    ret = np.zeros(T, dtype=np.float32)
    g = 0.0
    for t in reversed(range(T)):
        if dones[t]:
            g = 0.0
        g = float(rewards[t]) + gamma * g
        ret[t] = g
    return ret


def compute_gae(
    rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float
) -> tuple[np.ndarray, np.ndarray]:
    """GAE-λ advantages and returns. Truncated/terminal episodes bootstrap with 0."""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    last_adv = 0.0
    next_v = 0.0
    for t in reversed(range(T)):
        nonterm = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_v * nonterm - values[t]
        adv[t] = last_adv = delta + gamma * lam * nonterm * last_adv
        next_v = values[t]
    ret = adv + values
    return adv.astype(np.float32), ret.astype(np.float32)


# ---------------------------------------------------------------------------
# Rollout collection
# ---------------------------------------------------------------------------


def _collect_episode(env: GridEnv, policy_fn, max_steps: int | None = None) -> dict:
    """Run one episode using policy_fn(obs) -> (action, log_prob, value). Returns
    dict of numpy arrays for the whole episode."""
    if max_steps is None:
        max_steps = env.max_steps
    obs_buf, act_buf, lp_buf, v_buf, r_buf, d_buf = [], [], [], [], [], []
    obs, _ = env.reset()
    success = 0
    for _ in range(max_steps):
        a, lp, v = policy_fn(obs)
        ns, r, term, trunc, info = env.step(a)
        obs_buf.append(obs)
        act_buf.append(a)
        lp_buf.append(lp)
        v_buf.append(v)
        r_buf.append(r)
        d_buf.append(float(term or trunc))  # episode-end boundary for GAE/MC
        obs = ns
        if float(info.get("success", 0.0)) > 0.5:
            success = 1
        if term or trunc:
            break
    return {
        "obs": np.stack(obs_buf).astype(np.float32),
        "act": np.asarray(act_buf, dtype=np.int64),
        "log_prob": np.asarray(lp_buf, dtype=np.float32),
        "value": np.asarray(v_buf, dtype=np.float32),
        "reward": np.asarray(r_buf, dtype=np.float32),
        "done": np.asarray(d_buf, dtype=np.float32),
        "success": success,
        "length": len(obs_buf),
    }


def _concat_episodes(episodes: list[dict]) -> dict:
    return {
        "obs": np.concatenate([e["obs"] for e in episodes], axis=0),
        "act": np.concatenate([e["act"] for e in episodes], axis=0),
        "log_prob": np.concatenate([e["log_prob"] for e in episodes], axis=0),
        "value": np.concatenate([e["value"] for e in episodes], axis=0),
        "reward": np.concatenate([e["reward"] for e in episodes], axis=0),
        "done": np.concatenate([e["done"] for e in episodes], axis=0),
    }


def _support_from_episodes(
    episodes: list[dict],
    gamma: float,
    n_max: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate, compute MC returns, subsample to ≤ n_max rows."""
    cat = _concat_episodes(episodes)
    returns = compute_mc_returns(cat["reward"], cat["done"], gamma)
    N = cat["obs"].shape[0]
    if N > n_max:
        idx = np.random.choice(N, size=n_max, replace=False)
    else:
        idx = np.arange(N)
    obs = torch.tensor(cat["obs"][idx])
    act = torch.tensor(cat["act"][idx], dtype=torch.long)
    ret = torch.tensor(returns[idx])
    return obs, act, ret


# ---------------------------------------------------------------------------
# FTN PPO agent
# ---------------------------------------------------------------------------


class FTNPPO:
    def __init__(self, obs_dim: int, n_actions: int, cfg: GridConfig):
        self.cfg = cfg
        self.n_actions = n_actions
        H, k = cfg.ftn_hidden_size, cfg.ftn_k
        L, IS, D = cfg.ftn_layers, cfg.ftn_inner_size, cfg.ftn_dropout_p

        self.actor = FTNDiscreteActor(obs_dim, n_actions, H, L, IS, D)
        self.critic = FTNValueCritic(obs_dim, H, L, IS, D)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.m_a = kwta(torch.randn(H), k=k)
        self.m_c = kwta(torch.randn(H), k=k)
        self.m_a_logits: torch.Tensor | None = None
        self.m_c_logits: torch.Tensor | None = None

        self.stored_m_a: dict[int, torch.Tensor] = {}
        self.stored_m_c: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def act(self, obs_np, mask_a=None, mask_c=None, greedy=False):
        m_a = self.m_a if mask_a is None else mask_a
        m_c = self.m_c if mask_c is None else mask_c
        obs = torch.tensor(obs_np).unsqueeze(0)
        logits = self.actor(obs, m_a).squeeze(0)
        if greedy:
            a = int(logits.argmax().item())
            lp = float(F.log_softmax(logits, dim=-1)[a].item())
        else:
            dist = Categorical(logits=logits)
            at = dist.sample()
            a = int(at.item())
            lp = float(dist.log_prob(at).item())
        v = float(self.critic(obs, m_c).squeeze(0).item())
        return a, lp, v

    def reconfig_masks(self, support, commit: bool = True):
        """Optimize critic mask (MSE V/return) then actor mask (REINFORCE-style).

        support = (obs, act, returns), all tensors on the default device.
        """
        obs, act, returns = support
        cfg = self.cfg
        H, k = cfg.ftn_hidden_size, cfg.ftn_k
        warm = cfg.mask_warm_start
        init_a = self.m_a_logits if warm else None
        init_c = self.m_c_logits if warm else None

        def critic_loss(soft_m):
            v = self.critic(obs, soft_m)
            return F.mse_loss(v, returns)

        m_c_new, m_c_logits_new, c_init, c_final = anneal_ste_mask(
            H,
            k,
            critic_loss,
            cfg.reconfig_steps,
            cfg.reconfig_lr,
            init_logits=init_c,
            noise_scale=cfg.mask_init_noise_scale,
        )

        # Advantage = returns - V_baseline under the freshly chosen critic mask.
        with torch.no_grad():
            v_baseline = self.critic(obs, m_c_new)
        adv = returns - v_baseline
        if adv.numel() > 1 and adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        def actor_loss(soft_m):
            logits = self.actor(obs, soft_m)
            log_probs = F.log_softmax(logits, dim=-1)
            log_pi = log_probs.gather(1, act.unsqueeze(-1).long()).squeeze(-1)
            return -(log_pi * adv).mean()

        m_a_new, m_a_logits_new, a_init, a_final = anneal_ste_mask(
            H,
            k,
            actor_loss,
            cfg.reconfig_steps,
            cfg.reconfig_lr,
            init_logits=init_a,
            noise_scale=cfg.mask_init_noise_scale,
        )

        stats = {
            "critic_init": c_init,
            "critic_final": c_final,
            "actor_init": a_init,
            "actor_final": a_final,
        }
        if commit:
            self.m_a = m_a_new
            self.m_c = m_c_new
            self.m_a_logits = m_a_logits_new
            self.m_c_logits = m_c_logits_new
        return m_a_new, m_c_new, stats

    def update(self, rollout):
        """Standard PPO update on a rollout (clip + entropy bonus + MSE value)."""
        cfg = self.cfg
        obs, act = rollout["obs"], rollout["act"]
        lp_old, ret, adv = rollout["log_prob_old"], rollout["ret"], rollout["adv"]
        N = obs.shape[0]
        if N > 1 and adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.arange(N)
        a_losses, c_losses = [], []
        for _ in range(cfg.ppo_epochs):
            np.random.shuffle(idx)
            for s in range(0, N, cfg.minibatch_size):
                mb = idx[s : s + cfg.minibatch_size]
                if len(mb) < 2:
                    continue
                mb_t = torch.tensor(mb, dtype=torch.long)
                mb_obs = obs[mb_t]
                mb_act = act[mb_t]
                mb_lp_old = lp_old[mb_t]
                mb_ret = ret[mb_t]
                mb_adv = adv[mb_t]

                logits = self.actor(mb_obs, self.m_a)
                dist = Categorical(logits=logits)
                lp_new = dist.log_prob(mb_act)
                ent = dist.entropy().mean()
                ratio = (lp_new - mb_lp_old).exp()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - cfg.ppo_clip, 1 + cfg.ppo_clip) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean() - cfg.entropy_coef * ent
                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                v = self.critic(mb_obs, self.m_c)
                critic_loss = F.mse_loss(v, mb_ret)
                self.critic_opt.zero_grad()
                critic_loss.backward()
                self.critic_opt.step()

                a_losses.append(actor_loss.item())
                c_losses.append(critic_loss.item())
        return (
            float(np.mean(a_losses)) if a_losses else 0.0,
            float(np.mean(c_losses)) if c_losses else 0.0,
        )


# ---------------------------------------------------------------------------
# Exploration phase
# ---------------------------------------------------------------------------


def explore_for_reward(
    env: GridEnv,
    agent: FTNPPO,
    max_episodes: int,
    policy: str,
    prev_m_a: torch.Tensor | None = None,
) -> tuple[list[dict], int]:
    """Collect exploration episodes until the first reward is observed (cap at
    max_episodes). Returns (episodes, n_successful)."""
    episodes = []
    n_success = 0
    n_actions = env.n_actions
    log_unif = -math.log(n_actions)
    for _ in range(max_episodes):
        if policy == "random":

            def policy_fn(obs):
                a = int(np.random.randint(n_actions))
                return a, log_unif, 0.0
        else:  # "prev_mask"

            def policy_fn(obs):
                return agent.act(obs, mask_a=prev_m_a)

        ep = _collect_episode(env, policy_fn)
        episodes.append(ep)
        if ep["success"]:
            n_success += 1
            break
    return episodes, n_success


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def _eval_with_reconfig(env: GridEnv, agent: FTNPPO, cfg: GridConfig) -> float:
    """Unified eval used for every cell of the matrix (retrospective AND
    head-start). What we're measuring: given one rewarded exploration
    trajectory, how reliably does the AnnealedSTE mask-search recover a mask
    that solves the task?

    Procedure:
        1. Run up to cfg.explore_episodes exploration rollouts (warmup_policy);
           stop on the first rewarded one. If no reward seen → score = 0.
        2. Otherwise, for each of cfg.eval_episodes trials: call
           reconfig_masks(support, commit=False) — each call cold-starts
           AnnealedSTE on the same exploration support, landing on a
           different mask — then run one greedy episode under that mask.
        3. Mean success across the cfg.eval_episodes trials is the score.
    """
    eps, n_succ = explore_for_reward(
        env,
        agent,
        cfg.explore_episodes,
        cfg.warmup_policy,
        prev_m_a=agent.m_a.clone(),
    )
    if n_succ == 0:
        return 0.0
    support = _support_from_episodes(eps, cfg.gamma, cfg.reconfig_n)

    n_eval_succ = 0
    for _ in range(cfg.eval_episodes):
        m_a, m_c, _ = agent.reconfig_masks(support, commit=False)

        def policy_fn(obs):
            return agent.act(obs, mask_a=m_a, mask_c=m_c, greedy=True)

        ep = _collect_episode(env, policy_fn)
        if ep["success"]:
            n_eval_succ += 1
    return n_eval_succ / cfg.eval_episodes


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def train_grid_ftn(cfg: GridConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    envs = build_grid_envs(cfg)
    obs_dim = envs[0].obs_dim
    n_actions = envs[0].n_actions

    agent = FTNPPO(obs_dim, n_actions, cfg)
    N = len(envs)
    matrix = np.zeros((N, N), dtype=np.float32)

    t_start = time.time()
    for ti, env in enumerate(envs):
        print(f"\n=== Phase {ti}/{N - 1}: {env.name} (reward {env.reward_pos}) ===")

        # 1) Exploration ---------------------------------------------------
        prev_m_a = agent.m_a.clone()
        # Task 0 forces random exploration since the agent has no prior policy.
        policy = "random" if ti == 0 else cfg.warmup_policy
        explore_eps, n_succ = explore_for_reward(
            env,
            agent,
            cfg.explore_episodes,
            policy,
            prev_m_a=prev_m_a,
        )
        n_steps = sum(e["length"] for e in explore_eps)
        print(
            f"  exploration ({policy}): {len(explore_eps)} eps, "
            f"{n_steps} steps, {n_succ} successful"
        )

        # 2) Mask reconfig -------------------------------------------------
        if n_succ > 0:
            support = _support_from_episodes(explore_eps, cfg.gamma, cfg.reconfig_n)
            _, _, stats = agent.reconfig_masks(support, commit=True)
            print(
                f"  reconfig: critic {stats['critic_init']:.3f}->"
                f"{stats['critic_final']:.3f}  actor {stats['actor_init']:.3f}->"
                f"{stats['actor_final']:.3f}"
            )
        else:
            print("  WARN: no reward found during exploration; keeping prior mask")
        agent.stored_m_a[ti] = agent.m_a.detach().clone()
        agent.stored_m_c[ti] = agent.m_c.detach().clone()

        # 3) Exploitation: K trials of PPO ---------------------------------
        rollout_eps: list[dict] = []
        rollout_steps = 0
        trials = 0
        train_returns, train_lengths, train_succs = [], [], []

        while trials < cfg.n_trials_per_task:
            ep = _collect_episode(env, agent.act)
            rollout_eps.append(ep)
            rollout_steps += ep["length"]
            trials += 1
            train_returns.append(float(ep["reward"].sum()))
            train_lengths.append(ep["length"])
            train_succs.append(ep["success"])

            if rollout_steps >= cfg.rollout_length or trials >= cfg.n_trials_per_task:
                cat = _concat_episodes(rollout_eps)
                adv, ret = compute_gae(
                    cat["reward"],
                    cat["value"],
                    cat["done"],
                    cfg.gamma,
                    cfg.gae_lambda,
                )
                rollout = {
                    "obs": torch.tensor(cat["obs"]),
                    "act": torch.tensor(cat["act"], dtype=torch.long),
                    "log_prob_old": torch.tensor(cat["log_prob"]),
                    "ret": torch.tensor(ret),
                    "adv": torch.tensor(adv),
                }
                agent.update(rollout)
                rollout_eps, rollout_steps = [], 0

            # Optional periodic mid-phase reconfig.
            if (
                cfg.reconfig_every_trials > 0
                and trials % cfg.reconfig_every_trials == 0
            ):
                # Use the most recent few trials as support.
                recent = rollout_eps[-3:] if rollout_eps else [ep]
                support = _support_from_episodes(recent, cfg.gamma, cfg.reconfig_n)
                agent.reconfig_masks(support, commit=True)

            if trials % cfg.log_every == 0:
                k = min(cfg.log_every, len(train_returns))
                mean_r = float(np.mean(train_returns[-k:]))
                mean_sr = float(np.mean(train_succs[-k:]))
                mean_l = float(np.mean(train_lengths[-k:]))
                elapsed = time.time() - t_start
                print(
                    f"  trial {trials:>4d}  ep_ret={mean_r:6.2f}  "
                    f"sr={mean_sr:.2f}  len={mean_l:5.1f}  ({elapsed:.0f}s)"
                )

        # 4) Archive final mask --------------------------------------------
        agent.stored_m_a[ti] = agent.m_a.detach().clone()
        agent.stored_m_c[ti] = agent.m_c.detach().clone()

        # 5) Eval all tasks (same mechanism for retrospective and head-start) -
        print(f"  --- Eval after phase {ti} ---")
        for tj in range(N):
            sr = _eval_with_reconfig(envs[tj], agent, cfg)
            tag = "<-" if tj == ti else "  "
            matrix[ti, tj] = sr
            print(
                f"    [{tag}] {envs[tj].name:10s} reward {envs[tj].reward_pos} "
                f"sr={sr:.2f}"
            )

        np.savez(
            cfg.output_path,
            matrix=matrix,
            reward_locations=np.array(cfg.reward_locations),
            phases_done=ti + 1,
        )

    print("\n=== Eval matrix (rows: training phase, cols: eval task) ===")
    for ti in range(N):
        row = " ".join(f"{matrix[ti, tj]:.2f}" for tj in range(N))
        print(f"  phase {ti} ({envs[ti].name}): {row}")

    np.savez(
        cfg.output_path,
        matrix=matrix,
        reward_locations=np.array(cfg.reward_locations),
        phases_done=N,
    )
    plot_matrix(matrix, [e.name for e in envs], cfg.output_path.replace(".npz", ".png"))
    return matrix


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_matrix(matrix: np.ndarray, names, out_path: str):
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n + 2), max(5, 0.6 * n + 1.5)))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel("Eval task")
    ax.set_ylabel("Trained through task")
    ax.set_title("Sequential grid-world PPO: success rate matrix")
    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i, j] < 0.5 else "black"
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="success rate")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test config")
    args = parser.parse_args()

    cfg = smoke_grid_config() if args.smoke else GridConfig()
    print("Config:", asdict(cfg))
    train_grid_ftn(cfg)
