"""Sequential behavior cloning on Meta-World MT10 from *expert* trajectories.

This is the supervised counterpart to train_mt10.py. Instead of cheat-init +
random exploration + DDPG, it learns each task by **behavior cloning** the
Meta-World scripted expert policies (the hand-written `ENV_POLICY_MAP` solvers
that ship with the benchmark and solve every task at ~100%). Standard resets,
no cheat init.

Pipeline (one block per task, learned strictly in sequence — mirrors train.py):

    1. Data: roll out the task's scripted expert on the standard env, collect
       (obs, action) pairs, truncate each demo shortly after first success.
       Cached to disk so the four baselines reuse the same demonstrations.
    2. Mask reconfig (FTN only): pick the actor mask that minimizes BC loss on
       this task's expert support set (AnnealedSTE search over the k-hot mask).
       The expert data *is* the support set.
    3. Train: behavior-clone the FTN actor under the committed mask (MSE on the
       expert actions), optionally summing a BC loss over mask-overlapping
       archived chunks from past tasks (the cross-task consolidation mechanism).
    4. Store the phase's mask; archive the phase's support for filtered replay.
    5. Eval every task (greedy rollouts) → success-rate matrix.

Four conditions (identical to train.py / train_mt10.py), select with a flag:
    --no-mask     all-ones mask, full shared net  → catastrophic forgetting ref
    --fixed-mask  oracle disjoint allocation: task i owns neurons [i*k,(i+1)*k)
    --stored-mask FTN training; eval recalls the per-task last mask
    (default)     full FTN: reconfig the mask at eval from each task's expert data

There is no critic: BC is pure supervised regression, so only the FTN *actor*
backbone and the actor mask are used. The reconfig objective is the BC loss
itself (not −Q), evaluated on the expert support set.
"""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict, dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import metaworld
import numpy as np
import torch
import torch.nn.functional as F
from ftn import FTNActor, LateralSmoother, anneal_ste_mask, kwta
from metaworld.policies import ENV_POLICY_MAP

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")


# The same 5-task subset train_mt10.py currently runs. Every task here has a
# scripted expert in ENV_POLICY_MAP that solves it at ~100% from standard resets.
DEFAULT_TASKS = (
    # "reach-v3",
    # "push-v3",
    "door-open-v3",
    "drawer-open-v3",
    "drawer-close-v3",
    "button-press-topdown-v3",
    "window-open-v3",
    "window-close-v3",
    "peg-insert-side-v3",  # hard: needs a sustained grasp + insertion
    "pick-place-v3",  # hard: needs a sustained grasp
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ExpertBCConfig:
    # --- tasks / experiment ---
    tasks: tuple = DEFAULT_TASKS
    seed: int = 0
    output_path: str = "results_expert.npz"
    data_dir: str = "expert_data"  # demo cache (shared across the 4 conditions)
    log_every: int = 5

    # --- expert demonstrations ---
    n_demos: int = 50  # expert episodes collected per task
    demo_max_steps: int = 500  # rollout horizon for data generation
    demo_tail: int = 5  # frames kept after the first success step

    # --- behavior cloning ---
    bc_epochs: int = 50
    bc_batch_size: int = 256
    bc_lr: float = 3e-4
    weight_decay: float = 0.0

    # --- eval ---
    eval_episodes: int = 1
    eval_max_steps: int = 10

    # --- FTN backbone ---
    # H/N must hold k <= H/len(tasks) for the fixed-mask disjoint allocation:
    # 2048 / 5 ~= 409, so k=256 leaves room for all 5 disjoint slots.
    ftn_hidden_size: int = 2304
    ftn_k: int = 256
    ftn_inner_size: int = 8
    ftn_layers: int = 8
    ftn_dropout_p: float = 0.2

    # --- square mask + lateral spatial smoothing (SmoothKWTA-style) ---
    # The k-hot actor mask is laid onto a square grid of square_dim x square_dim
    # blocks (H must be a multiple of square_dim**2: 2048 = 2 * 32**2) and, after
    # each AnnealedSTE search, smoothed by `lateral_steps` Polyak-mixed uniform
    # convolutions before a final top-k. This spatially clusters each task's mask.
    # Disable with lateral_steps=0 OR lateral_polyak=1.0 (default: disabled).
    ftn_square_dim: int = 48
    lateral_steps: int = 3
    lateral_polyak: float = 0.800  # 1.0 -> no smoothing; <1 mixes smoothed grid in
    lateral_kernel: int = 5
    lateral_noise_sd: float = 0.0
    lateral_torus: bool = False

    # --- checkpointing ---
    # If set, the FTN model (actor weights, normalizer, per-task masks) is saved
    # to this single file after each task finishes, overwriting the previous one.
    ckpt_path: str = ""

    # --- AnnealedSTE mask reconfig (objective = BC loss on expert support) ---
    reconfig_steps: int = 60
    reconfig_lr: float = 0.03
    reconfig_n: int = 4096  # support transitions subsampled for the mask search
    mask_init_noise_scale: float = 0.00001
    mask_warm_start: bool = True
    # Reconfig cadence (FTN / stored-mask only). Both are integers:
    #   reconfig_train (n batches): a reconfig always runs once at each phase
    #       start. n=0 -> no further reconfig (frozen mask for the phase).
    #       n>=1 -> additionally re-search the mask every n'th BC minibatch,
    #       warm-started from the running logits, on that minibatch as support.
    #   reconfig_eval (n episodes): full-FTN eval recovers task tj's mask from
    #       its expert support. n=0 -> reconfig once per eval cell, reuse for all
    #       episodes. n>=1 -> cold-start a fresh reconfig every n'th episode
    #       (n=1 == every episode, mirrors train.py's _eval_with_reconfig).
    reconfig_train: int = 10
    reconfig_eval: int = 0

    # --- cross-task consolidation (mask-overlap-filtered BC replay) ---
    consolidate: bool = True
    replay_max_chunks: int = 10
    replay_max_tuples: int = 256
    rho_low: float = 0.2
    rho_high: float = 0.8

    # --- baselines ---
    fixed_mask: bool = False
    no_mask: bool = False
    stored_mask: bool = False


def smoke_config() -> ExpertBCConfig:
    return ExpertBCConfig(
        tasks=("drawer-open-v3", "drawer-close-v3"),
        n_demos=5,
        demo_max_steps=200,
        bc_epochs=5,
        eval_episodes=3,
        eval_max_steps=200,
        ftn_hidden_size=64,
        ftn_k=16,
        ftn_inner_size=4,
        ftn_layers=2,
        ftn_square_dim=8,  # 8**2 = 64 = ftn_hidden_size (one block)
        reconfig_steps=10,
        reconfig_n=128,
        output_path="results_expert_smoke.npz",
        data_dir="expert_data_smoke",
    )


# ---------------------------------------------------------------------------
# Standard (non-cheat) MT10 env wrapper
# ---------------------------------------------------------------------------


class StandardTaskEnv:
    """One MT10 env class + its 50 goal variants. Standard reset (random goal
    variant, default initial state) — no cheat init."""

    def __init__(self, name: str, env_cls, tasks, seed: int):
        self.name = name
        self.env_cls = env_cls
        self.env = env_cls()
        self.tasks = tasks
        self.rng = np.random.default_rng(seed)
        self.env.set_task(self.tasks[0])
        obs, _ = self.env.reset()
        self.obs_dim = int(obs.shape[0])
        self.act_dim = int(self.env.action_space.shape[0])

    def reset(self):
        task = self.tasks[self.rng.integers(len(self.tasks))]
        self.env.set_task(task)
        obs, info = self.env.reset()
        return obs.astype(np.float32), info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action.astype(np.float32))
        return obs.astype(np.float32), float(r), bool(term), bool(trunc), info


def build_envs(task_names, seed: int) -> list[StandardTaskEnv]:
    mt10 = metaworld.MT10(seed=seed)
    out = []
    for i, name in enumerate(task_names):
        tasks = [t for t in mt10.train_tasks if t.env_name == name]
        out.append(StandardTaskEnv(name, mt10.train_classes[name], tasks, seed + i + 1))
    return out


# ---------------------------------------------------------------------------
# Expert demonstration generation (+ disk cache)
# ---------------------------------------------------------------------------


def _rollout_expert(env: StandardTaskEnv, policy, cfg: ExpertBCConfig):
    """One scripted-expert episode. Returns (obs[T,D], act[T,A], success) where
    T is truncated to `first_success + demo_tail` (full horizon if no success)."""
    obs_buf, act_buf = [], []
    obs, _ = env.reset()
    first_succ = -1
    for t in range(cfg.demo_max_steps):
        a = np.clip(policy.get_action(obs), -1.0, 1.0).astype(np.float32)
        obs_buf.append(obs)
        act_buf.append(a)
        obs, _, term, trunc, info = env.step(a)
        if float(info.get("success", 0.0)) > 0.5 and first_succ < 0:
            first_succ = t
        if term or trunc:
            break
    success = int(first_succ >= 0)
    cut = (
        len(obs_buf)
        if first_succ < 0
        else min(len(obs_buf), first_succ + 1 + cfg.demo_tail)
    )
    return (
        np.asarray(obs_buf[:cut], dtype=np.float32),
        np.asarray(act_buf[:cut], dtype=np.float32),
        success,
    )


def generate_or_load_demos(envs, cfg: ExpertBCConfig) -> dict[str, dict]:
    """Per task: a cached dict {obs:[N,D], act:[N,A]} of concatenated expert
    transitions. Cache key includes the demo-generation params + seed so the
    four baselines reuse identical data."""
    os.makedirs(cfg.data_dir, exist_ok=True)
    demos: dict[str, dict] = {}
    for env in envs:
        key = f"{env.name}_n{cfg.n_demos}_s{cfg.seed}_t{cfg.demo_tail}_m{cfg.demo_max_steps}.npz"
        path = os.path.join(cfg.data_dir, key)
        if os.path.exists(path):
            d = np.load(path)
            demos[env.name] = {"obs": d["obs"], "act": d["act"]}
            print(f"  [cache] {env.name:25s} {d['obs'].shape[0]} transitions")
            continue
        policy = ENV_POLICY_MAP[env.name]()
        obs_all, act_all, n_succ = [], [], 0
        for _ in range(cfg.n_demos):
            o, a, s = _rollout_expert(env, policy, cfg)
            obs_all.append(o)
            act_all.append(a)
            n_succ += s
        obs_cat = np.concatenate(obs_all, axis=0)
        act_cat = np.concatenate(act_all, axis=0)
        np.savez(path, obs=obs_cat, act=act_cat)
        demos[env.name] = {"obs": obs_cat, "act": act_cat}
        print(
            f"  [gen]   {env.name:25s} {obs_cat.shape[0]} transitions "
            f"({n_succ}/{cfg.n_demos} expert demos succeeded)"
        )
    return demos


def fit_normalizer(demos: dict[str, dict]) -> tuple[torch.Tensor, torch.Tensor]:
    """Global observation standardizer over all tasks' expert data (frozen,
    shared across every task and condition so comparisons are fair). Actions
    are already in [-1, 1] and left unscaled."""
    all_obs = np.concatenate([d["obs"] for d in demos.values()], axis=0)
    mean = torch.tensor(all_obs.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(all_obs.std(axis=0), dtype=torch.float32).clamp_min(1e-6)
    return mean, std


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def disjoint_task_mask(ti: int, H: int, k: int) -> torch.Tensor:
    """k-hot mask for the fixed-mask baseline: task ti owns neurons
    [ti*k, (ti+1)*k). Caller must ensure (ti+1)*k <= H."""
    m = torch.zeros(H)
    m[ti * k : (ti + 1) * k] = 1.0
    return m


# ---------------------------------------------------------------------------
# BC agent (FTN actor only — no critic, no targets)
# ---------------------------------------------------------------------------


class BCAgent:
    def __init__(self, obs_dim: int, act_dim: int, cfg: ExpertBCConfig):
        self.cfg = cfg
        self.act_dim = act_dim
        H, k = cfg.ftn_hidden_size, cfg.ftn_k
        self.actor = FTNActor(
            obs_dim, act_dim, H, cfg.ftn_layers, cfg.ftn_inner_size, cfg.ftn_dropout_p
        )
        self.opt = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.bc_lr, weight_decay=cfg.weight_decay
        )
        if cfg.no_mask:
            self.m_a = torch.ones(H)
        elif cfg.fixed_mask:
            self.m_a = disjoint_task_mask(0, H, k)
        else:
            self.m_a = kwta(torch.randn(H), k=k)
        self.m_a_logits: torch.Tensor | None = None
        self.stored_m_a: dict[int, torch.Tensor] = {}
        # Lateral spatial-smoothing step applied to every reconfigured mask.
        self.smoother = LateralSmoother(
            H,
            k,
            square_dim=cfg.ftn_square_dim,
            kernel_size=cfg.lateral_kernel,
            lateral_steps=cfg.lateral_steps,
            lateral_polyak=cfg.lateral_polyak,
            lateral_noise_sd=cfg.lateral_noise_sd,
            torus=cfg.lateral_torus,
        )
        # Frozen global observation normalizer (set after data generation).
        self.obs_mean: torch.Tensor | None = None
        self.obs_std: torch.Tensor | None = None

    def set_normalizer(self, mean: torch.Tensor, std: torch.Tensor):
        self.obs_mean, self.obs_std = mean, std

    def _norm(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.obs_mean) / self.obs_std

    def reset_optimizer_state(self):
        """Per-task Adam reset: stale momentum must not keep stepping the
        previous task's neurons once they leave the active mask."""
        self.opt = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.cfg.bc_lr,
            weight_decay=self.cfg.weight_decay,
        )

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, mask_a: torch.Tensor | None = None) -> np.ndarray:
        m = self.m_a if mask_a is None else mask_a
        obs = self._norm(torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0))
        was_training = self.actor.training
        self.actor.eval()
        a = self.actor(obs, m).squeeze(0).cpu().numpy()
        if was_training:
            self.actor.train()
        return np.clip(a, -1.0, 1.0).astype(np.float32)

    def reconfig_actor_mask(self, obs: torch.Tensor, act: torch.Tensor, commit: bool):
        """Search the k-hot actor mask that minimizes BC loss on the (already
        normalized) expert support batch via AnnealedSTE. `obs` is normalized."""
        cfg = self.cfg
        H, k = cfg.ftn_hidden_size, cfg.ftn_k
        init = self.m_a_logits if cfg.mask_warm_start else None

        def bc_loss(soft_m):
            return F.mse_loss(self.actor(obs, soft_m), act)

        m_new, logits_new, l_init, l_final = anneal_ste_mask(
            H,
            k,
            bc_loss,
            cfg.reconfig_steps,
            cfg.reconfig_lr,
            init_logits=init,
            noise_scale=cfg.mask_init_noise_scale,
        )
        # Lateral spatial smoothing: cluster the searched mask into a contiguous
        # blob on the square grid (no-op when smoothing is disabled).
        m_new = self.smoother.smooth(m_new)
        if commit:
            self.m_a = m_new
            self.m_a_logits = logits_new
        return m_new, {"bc_init": l_init, "bc_final": l_final}

    def bc_update(self, obs: torch.Tensor, act: torch.Tensor, recalled: list):
        """One BC gradient step: MSE on the current batch (under m_a) summed
        with MSE on each recalled past chunk (under the mask it was trained
        under). Shared neurons accumulate gradients from both — the cross-task
        consolidation that keeps past tasks intact while learning the new one.
        `obs` tensors are already normalized. Returns the current-task loss."""
        terms = [(obs, act, self.m_a)] + recalled
        total = None
        cur_loss_val = 0.0
        for o, a, m in terms:
            loss = F.mse_loss(self.actor(o, m), a)
            if m is self.m_a:
                cur_loss_val = float(loss.item())
            total = loss if total is None else total + loss
        self.opt.zero_grad()
        total.backward()
        self.opt.step()
        return cur_loss_val


# ---------------------------------------------------------------------------
# Mask-overlap-filtered support archive (actor-only analog of MaskedReplayBuffer)
# ---------------------------------------------------------------------------


class SupportArchive:
    """Per-completed-task store of (normalized obs, act, m_a). Filtered sampling
    returns chunks whose mask overlaps the current mask within (rho_low,
    rho_high] — the consolidation targets. No-mask (identical masks, overlap=1)
    and fixed-mask (disjoint, overlap=0) never qualify, so consolidation is a
    no-op for those baselines, exactly as in train.py."""

    def __init__(self):
        self.past: list[dict] = []

    def archive(self, obs: torch.Tensor, act: torch.Tensor, m_a: torch.Tensor):
        self.past.append(
            {
                "obs": obs.detach().clone(),
                "act": act.detach().clone(),
                "m_a": m_a.detach().clone(),
            }
        )

    @staticmethod
    def _overlap(m_cur: torch.Tensor, m_old: torch.Tensor) -> float:
        denom = float(m_cur.sum().clamp(min=1).item())
        return float((m_cur * m_old).sum().item()) / denom

    def sample_filtered_multi(self, m_a_cur, n_chunks, n_tuples, rho_low, rho_high):
        if not self.past:
            return []
        cands = [
            i
            for i, ph in enumerate(self.past)
            if rho_low < self._overlap(m_a_cur, ph["m_a"]) <= rho_high
        ]
        if not cands:
            return []
        chosen = np.random.choice(cands, size=min(n_chunks, len(cands)), replace=False)
        out = []
        for i in chosen:
            ph = self.past[int(i)]
            N = ph["obs"].shape[0]
            idx = torch.randint(0, N, (min(n_tuples, N),))
            out.append((ph["obs"][idx], ph["act"][idx], ph["m_a"]))
        return out


# ---------------------------------------------------------------------------
# Support / batch tensor helpers (normalized obs)
# ---------------------------------------------------------------------------


def make_support(demo: dict, agent: BCAgent, n_max: int):
    """Subsample up to n_max expert transitions → (normalized obs, act) tensors."""
    N = demo["obs"].shape[0]
    idx = np.random.choice(N, size=min(n_max, N), replace=False)
    obs = agent._norm(torch.tensor(demo["obs"][idx], dtype=torch.float32))
    act = torch.tensor(demo["act"][idx], dtype=torch.float32)
    return obs, act


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


def eval_task(
    env: StandardTaskEnv,
    agent: BCAgent,
    cfg: ExpertBCConfig,
    mask_a: torch.Tensor | None = None,
    reconfig_demo: dict | None = None,
    reconfig_every: int = 0,
) -> float:
    """Run cfg.eval_episodes greedy rollouts. With `reconfig_demo` (full FTN),
    recover task tj's mask from its expert support by a cold-started reconfig
    (commit=False): before episode 0, then again every `reconfig_every` episodes
    (reconfig_every=0 -> only once per cell, reused thereafter). Without
    `reconfig_demo`, roll out under the fixed `mask_a` (recall / no-mask)."""
    n_succ = 0
    for ep in range(cfg.eval_episodes):
        need_reconfig = reconfig_demo is not None and (
            ep == 0 or (reconfig_every >= 1 and ep % reconfig_every == 0)
        )
        if need_reconfig:
            s_obs, s_act = make_support(reconfig_demo, agent, cfg.reconfig_n)
            saved = agent.m_a_logits
            agent.m_a_logits = None  # cold-start this reconfig
            mask_a, _ = agent.reconfig_actor_mask(s_obs, s_act, commit=False)
            agent.m_a_logits = saved
        obs, _ = env.reset()
        for _ in range(cfg.eval_max_steps):
            a = agent.act(obs, mask_a=mask_a)
            obs, _, term, trunc, info = env.step(a)
            if float(info.get("success", 0.0)) > 0.5:
                n_succ += 1
                break
            if term or trunc:
                break
    return n_succ / cfg.eval_episodes


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: str,
    agent: BCAgent,
    cfg: ExpertBCConfig,
    obs_dim: int,
    act_dim: int,
    phases_done: int,
):
    """Save the FTN model to a single file (overwritten after each task). Holds
    the actor weights, the frozen obs normalizer, every per-task mask learned so
    far, and enough config to rebuild the agent for eval/visualization."""
    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "obs_mean": None
            if agent.obs_mean is None
            else agent.obs_mean.detach().cpu(),
            "obs_std": None if agent.obs_std is None else agent.obs_std.detach().cpu(),
            "m_a": agent.m_a.detach().cpu(),
            "stored_m_a": {i: m.detach().cpu() for i, m in agent.stored_m_a.items()},
            "cfg": asdict(cfg),
            "tasks": list(cfg.tasks),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "phases_done": phases_done,
        },
        path,
    )
    print(f"  [ckpt] saved model through phase {phases_done - 1} -> {path}")


def train(cfg: ExpertBCConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    envs = build_envs(cfg.tasks, cfg.seed)
    N = len(envs)
    obs_dim, act_dim = envs[0].obs_dim, envs[0].act_dim

    if cfg.fixed_mask and N * cfg.ftn_k > cfg.ftn_hidden_size:
        raise ValueError(
            f"fixed_mask needs N*k <= H for disjoint subsets, got "
            f"N={N}, k={cfg.ftn_k}, H={cfg.ftn_hidden_size}"
        )

    print("Generating / loading expert demonstrations...")
    demos = generate_or_load_demos(envs, cfg)
    obs_mean, obs_std = fit_normalizer(demos)

    agent = BCAgent(obs_dim, act_dim, cfg)
    agent.set_normalizer(obs_mean, obs_std)
    archive = SupportArchive()

    no_reconfig = cfg.fixed_mask or cfg.no_mask
    recall_eval = cfg.fixed_mask or cfg.stored_mask
    baseline_label = (
        "no-mask" if cfg.no_mask else ("fixed-mask" if cfg.fixed_mask else "")
    )

    matrix = np.full((N, N), np.nan, dtype=np.float32)
    t_start = time.time()

    for ti, env in enumerate(envs):
        print(f"\n=== Phase {ti}/{N - 1}: {env.name} ===")
        agent.reset_optimizer_state()

        if cfg.fixed_mask:
            agent.m_a = disjoint_task_mask(ti, cfg.ftn_hidden_size, cfg.ftn_k)
            print(f"  fixed-mask: neurons [{ti * cfg.ftn_k}, {(ti + 1) * cfg.ftn_k})")

        demo = demos[env.name]
        obs_full = agent._norm(torch.tensor(demo["obs"], dtype=torch.float32))
        act_full = torch.tensor(demo["act"], dtype=torch.float32)
        n_trans = obs_full.shape[0]

        # 1) Mask reconfig on the expert support (FTN / stored-mask only) ----
        if not no_reconfig:
            s_obs, s_act = make_support(demo, agent, cfg.reconfig_n)
            _, stats = agent.reconfig_actor_mask(s_obs, s_act, commit=True)
            print(f"  reconfig: BC {stats['bc_init']:.4f} -> {stats['bc_final']:.4f}")
        else:
            print(f"  {baseline_label} baseline: skipping reconfig")
        agent.stored_m_a[ti] = agent.m_a.detach().clone()

        # 2) Behavior cloning under the committed mask ----------------------
        self_bs = cfg.bc_batch_size
        batch_ctr = 0
        for epoch in range(cfg.bc_epochs):
            perm = torch.randperm(n_trans)
            ep_loss, n_b = 0.0, 0
            for start in range(0, n_trans, self_bs):
                bidx = perm[start : start + self_bs]
                batch_ctr += 1
                # Every reconfig_train'th minibatch, re-search the mask on this
                # minibatch (warm-started from the running logits) before the
                # step. reconfig_train=0 disables it (mask frozen post phase-start).
                if (
                    cfg.reconfig_train >= 1
                    and not no_reconfig
                    and batch_ctr % cfg.reconfig_train == 0
                ):
                    agent.reconfig_actor_mask(
                        obs_full[bidx], act_full[bidx], commit=True
                    )
                recalled = (
                    agent.sample_recalled(archive)
                    if (cfg.consolidate and not no_reconfig)
                    else []
                )
                loss = agent.bc_update(obs_full[bidx], act_full[bidx], recalled)
                ep_loss += loss
                n_b += 1
            if (epoch + 1) % cfg.log_every == 0 or epoch == cfg.bc_epochs - 1:
                elapsed = time.time() - t_start
                print(
                    f"  epoch {epoch + 1:>3d}/{cfg.bc_epochs}  "
                    f"bc_loss={ep_loss / max(1, n_b):.4f}  ({elapsed:.0f}s)"
                )

        agent.stored_m_a[ti] = agent.m_a.detach().clone()
        # 3) Archive this phase's support for cross-task consolidation -------
        if not no_reconfig:
            arch_obs, arch_act = make_support(demo, agent, cfg.replay_max_tuples * 8)
            archive.archive(arch_obs, arch_act, agent.m_a)

        # 4) Eval every task -------------------------------------------------
        print(f"  --- Eval after phase {ti} ---")
        for tj in range(N):
            tag = "<-" if tj == ti else "  "
            if recall_eval:
                if tj > ti:
                    matrix[ti, tj] = np.nan
                    print(f"    [{tag}] {envs[tj].name:25s} sr=  — (skipped)")
                    continue
                sr = eval_task(envs[tj], agent, cfg, mask_a=agent.stored_m_a[tj])
            elif cfg.no_mask:
                sr = eval_task(envs[tj], agent, cfg, mask_a=agent.m_a)  # all-ones
            else:
                # Full FTN: recover task tj's mask from its expert support,
                # reconfiguring per cfg.reconfig_eval (0 = once per cell).
                sr = eval_task(
                    envs[tj],
                    agent,
                    cfg,
                    reconfig_demo=demos[envs[tj].name],
                    reconfig_every=cfg.reconfig_eval,
                )
            matrix[ti, tj] = sr
            print(f"    [{tag}] {envs[tj].name:25s} sr={sr:.2f}")

        np.savez(
            cfg.output_path,
            matrix=matrix,
            tasks=np.array(cfg.tasks),
            phases_done=ti + 1,
        )

        # 5) Checkpoint the model (single file, overwritten each phase) ---------
        if cfg.ckpt_path:
            save_checkpoint(cfg.ckpt_path, agent, cfg, obs_dim, act_dim, ti + 1)

    print("\n=== Eval matrix (rows: training phase, cols: eval task) ===")
    for ti in range(N):
        row = " ".join(
            "  — " if np.isnan(matrix[ti, tj]) else f"{matrix[ti, tj]:.2f}"
            for tj in range(N)
        )
        print(f"  phase {ti} ({envs[ti].name}): {row}")

    plot_matrix(matrix, [e.name for e in envs], cfg.output_path.replace(".npz", ".png"))
    config_path = cfg.output_path.replace(".npz", ".config.json")
    with open(config_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    print(f"Saved config to {config_path}")
    return matrix


def plot_matrix(matrix: np.ndarray, names, out_path: str):
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * n + 2), max(5, 0.6 * n + 1.5)))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    ax.set_xticks(range(n), names, rotation=45, ha="right")
    ax.set_yticks(range(n), names)
    ax.set_xlabel("Eval task")
    ax.set_ylabel("Trained through task")
    ax.set_title("Sequential MT10 expert-BC: success rate matrix")
    for i in range(n):
        for j in range(n):
            if np.isnan(matrix[i, j]):
                continue
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


# Bound method shim: keep the recalled-chunk sampling on the agent so bc_update's
# caller reads cleanly, while the archive owns the data.
def _sample_recalled(self: BCAgent, archive: SupportArchive):
    return archive.sample_filtered_multi(
        self.m_a,
        self.cfg.replay_max_chunks,
        self.cfg.replay_max_tuples,
        self.cfg.rho_low,
        self.cfg.rho_high,
    )


BCAgent.sample_recalled = _sample_recalled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Tiny smoke-test config.")
    parser.add_argument(
        "--fixed-mask",
        action="store_true",
        help="Oracle disjoint allocation: task i uses neurons [i*k,(i+1)*k), "
        "recalled at eval. No reconfig/consolidation; forward-transfer skipped.",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="All-ones mask (full shared net). Catastrophic-forgetting reference.",
    )
    parser.add_argument(
        "--stored-mask",
        action="store_true",
        help="FTN training, but eval recalls each task's last training mask "
        "(forward-transfer cells skipped).",
    )
    parser.add_argument(
        "--n-demos", type=int, default=None, help="Expert episodes per task."
    )
    parser.add_argument(
        "--bc-epochs", type=int, default=None, help="BC epochs per task."
    )
    parser.add_argument(
        "--reconfig-train",
        type=int,
        default=None,
        help="Mask reconfig cadence during training, in minibatches (FTN / "
        "stored-mask only). 0 = only the phase-start reconfig; n>=1 = also "
        "re-search every n'th BC minibatch (1 = every batch).",
    )
    parser.add_argument(
        "--reconfig-eval",
        type=int,
        default=None,
        help="Mask reconfig cadence at eval, in episodes (full FTN only). "
        "0 = once per eval cell (reused); n>=1 = fresh cold-start reconfig every "
        "n'th episode (1 = every episode).",
    )
    parser.add_argument(
        "--lateral-steps",
        type=int,
        default=None,
        help="Lateral spatial-smoothing passes applied to each reconfigured mask. "
        "0 disables smoothing.",
    )
    parser.add_argument(
        "--lateral-polyak",
        type=float,
        default=None,
        help="Polyak coefficient mixing the binary mask with its smoothed grid "
        "each lateral step. 1.0 disables smoothing; <1 mixes the smoothed grid in.",
    )
    parser.add_argument(
        "--lateral-kernel",
        type=int,
        default=None,
        help="Side length of the uniform lateral averaging kernel.",
    )
    parser.add_argument(
        "--square-dim",
        type=int,
        default=None,
        help="Side of each square block in the mask grid (H must be a multiple "
        "of square-dim**2).",
    )
    parser.add_argument(
        "--no-ckpt",
        action="store_true",
        help="Disable per-task model checkpointing (on by default).",
    )
    parser.add_argument("--name", default="")
    parser.add_argument("--seed", default="0")
    args = parser.parse_args()

    if args.fixed_mask + args.no_mask + args.stored_mask > 1:
        parser.error(
            "--fixed-mask, --no-mask, and --stored-mask are mutually exclusive."
        )

    cfg = smoke_config() if args.smoke else ExpertBCConfig()
    cfg.fixed_mask = args.fixed_mask
    cfg.no_mask = args.no_mask
    cfg.stored_mask = args.stored_mask
    if args.n_demos is not None:
        cfg.n_demos = args.n_demos
    if args.bc_epochs is not None:
        cfg.bc_epochs = args.bc_epochs
    if args.reconfig_train is not None:
        cfg.reconfig_train = args.reconfig_train
    if args.reconfig_eval is not None:
        cfg.reconfig_eval = args.reconfig_eval
    if args.lateral_steps is not None:
        cfg.lateral_steps = args.lateral_steps
    if args.lateral_polyak is not None:
        cfg.lateral_polyak = args.lateral_polyak
    if args.lateral_kernel is not None:
        cfg.lateral_kernel = args.lateral_kernel
    if args.square_dim is not None:
        cfg.ftn_square_dim = args.square_dim

    if cfg.no_mask:
        suffix = "_nomask"
    elif cfg.fixed_mask:
        suffix = "_fixed"
    elif cfg.stored_mask:
        suffix = "_stored"
    else:
        suffix = "_ftn"
    tag = args.name + suffix
    cfg.output_path = cfg.output_path.replace(".", f"_{tag}_{args.seed}.")
    # Single model checkpoint, keyed off --name, overwritten after each task.
    cfg.ckpt_path = "" if args.no_ckpt else cfg.output_path.replace(".npz", ".pt")
    cfg.seed = int(args.seed)

    train(cfg)
