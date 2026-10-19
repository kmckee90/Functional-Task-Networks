"""JAX OptionsAgent: shared FTN backbone + per-option masks, sequential training.

Port of options/options_agent.py for train_options_2.py. ONE actor/critic backbone
(plus polyak targets); an option == a mask pair over it. The two hot phases are
fully jitted lax.scans built once in __init__ (cfg baked into the closures):

  * ``reconfig_option``  -- AnnealedSTE search of (m_c, m_a) on membership-relabeled
    MC-return data, then SmoothKWTA. One jit call (reconfig_steps fused).
  * ``train_option``     -- option_train_steps of DDPG over the masked subspace,
    summing in up to 3 mask-overlapping sibling options (the consolidation channel,
    exactly as torch _ddpg_step does with a list of terms). One jit call.

RECODE clustering, the FPS CentroidRegistry, footprints and graph membership are
host-side numpy (run once per round, cheap) -- only the per-step inner loops are
on-device.
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

import recode as recode_mod
from ftn import (actor_forward, actor_init, anneal_ste_mask, critic_forward,
                 critic_init, kwta, LateralSmoother, parallel_forward)


def _actor_rows(p, obs, mask_BH):
    """Actor forward with a PER-ROW mask (B, H) instead of one shared (H,)."""
    z = parallel_forward(p["parallel"], obs)
    return jnp.tanh((z * mask_BH) @ p["linear"])


def _critic_rows(p, obs, act, mask_BH):
    z = parallel_forward(p["parallel"], jnp.concatenate([obs, act], axis=-1))
    return ((z * mask_BH) @ p["linear"]).squeeze(-1)

_TERMS = 4  # 1 main option + up to 3 consolidation siblings (fixed for jit)


# ---------------------------------------------------------------------------
# Manual Adam (no optax dependency); matches torch.optim.Adam defaults.
# ---------------------------------------------------------------------------


def adam_init(params):
    z = lambda p: jnp.zeros_like(p)
    return {"m": jax.tree_util.tree_map(z, params),
            "v": jax.tree_util.tree_map(z, params),
            "t": jnp.int32(0)}


def adam_update(params, grads, st, lr, b1=0.9, b2=0.999, eps=1e-8):
    t = st["t"] + 1
    m = jax.tree_util.tree_map(lambda m, g: b1 * m + (1 - b1) * g, st["m"], grads)
    v = jax.tree_util.tree_map(lambda v, g: b2 * v + (1 - b2) * g * g, st["v"], grads)
    bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
    new = jax.tree_util.tree_map(
        lambda p, m, v: p - lr * (m / bc1) / (jnp.sqrt(v / bc2) + eps), params, m, v)
    return new, {"m": m, "v": v, "t": t}


def polyak(target, source, tau):
    return jax.tree_util.tree_map(lambda pt, p: (1 - tau) * pt + tau * p, target, source)


# ---------------------------------------------------------------------------
# Reward / membership (device, used inside the scans)
# ---------------------------------------------------------------------------


def membership_dev(obs, centroids):
    """Nearest-centroid id(s). obs (B,obs_dim), centroids (n,obs_dim) -> (B,)."""
    d = jnp.sum((obs[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return jnp.argmin(d, axis=1)


def reward_done(next_obs, cid, centroids, boundary, goal, shaping, step_cost, dist_reward=False):
    ids = membership_dev(next_obs, centroids)
    member = (ids == cid).astype(jnp.float32)
    dist = jnp.linalg.norm(next_obs - centroids[cid], axis=1)
    # dist_reward: dense negative distance-to-centroid; else sparse membership reward.
    reward = -dist if dist_reward else goal * member - shaping * dist - step_cost
    # Continuous-control semantics: membership is a recurring per-step reward that
    # BOOTSTRAPS (V_inside -> ~goal/(1-gamma)), not an absorbing terminal. Only the
    # world boundary (/timeout) ends the episode, so the option learns to stay in.
    done = boundary
    return reward, done


def compute_mc_returns(rewards, dones, gamma):
    """Reverse-scan MC returns (port of compute_mc_returns); resets at dones."""
    def f(g, x):
        r, d = x
        g = jnp.where(d > 0.5, 0.0, g)
        g = r + gamma * g
        return g, g
    _, out = jax.lax.scan(f, jnp.float32(0.0), (rewards, dones), reverse=True)
    return out


# ---------------------------------------------------------------------------
# Host-side stable centroid registry (FPS over RECODE atoms)
# ---------------------------------------------------------------------------


class CentroidRegistry:
    def __init__(self, obs_dim, cfg):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.centroids = np.zeros((0, obs_dim), np.float32)
        self.pos = np.zeros((0, 2), np.float32)
        self.pos_count = np.zeros((0,), np.float32)
        self.counts = np.zeros((0,), np.float32)

    @property
    def n(self):
        return self.centroids.shape[0]

    def sync(self, atoms: np.ndarray) -> int:
        """FPS-fold RECODE atoms into the stable centroid set (port of sync)."""
        if atoms.shape[0] == 0:
            return 0
        cfg = self.cfg
        if self.n > 0:  # re-anchor existing centroids onto the manifold (EMA)
            d = np.linalg.norm(self.centroids[:, None, :] - atoms[None, :, :], axis=2)
            nearest = atoms[np.argmin(d, axis=1)]
            e = cfg.centroid_anchor_ema
            self.centroids = (1 - e) * self.centroids + e * nearest
        added = 0
        target = min(cfg.target_centroids, cfg.max_centroids, atoms.shape[0])
        while self.n < target:
            if self.n == 0:
                self._append(atoms[0])
                added += 1
                continue
            d = np.min(np.linalg.norm(atoms[:, None, :] - self.centroids[None, :, :], axis=2), axis=1)
            j = int(np.argmax(d))
            if float(d[j]) < cfg.centroid_min_sep:
                break
            self._append(atoms[j])
            added += 1
        return added

    def _append(self, vec):
        self.centroids = np.concatenate([self.centroids, vec[None, :]], axis=0)
        self.pos = np.concatenate([self.pos, np.zeros((1, 2), np.float32)], axis=0)
        self.pos_count = np.concatenate([self.pos_count, np.zeros((1,), np.float32)], axis=0)
        self.counts = np.concatenate([self.counts, np.zeros((1,), np.float32)], axis=0)

    def membership_np(self, obs: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return np.full((obs.shape[0],), -1, dtype=np.int64)
        d = np.linalg.norm(obs[:, None, :] - self.centroids[None, :, :], axis=2)
        return np.argmin(d, axis=1)

    def update_footprints(self, next_obs: np.ndarray, next_pos: np.ndarray) -> None:
        ids = self.membership_np(next_obs)
        for i in range(self.n):
            m = ids == i
            c = int(m.sum())
            if c == 0:
                continue
            self.counts[i] += c
            mean_p = next_pos[m].mean(axis=0)
            n0 = self.pos_count[i]
            self.pos[i] = (self.pos[i] * n0 + mean_p * c) / (n0 + c)
            self.pos_count[i] += c

    def positions_np(self):
        return self.pos.copy()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class OptionsAgent:
    def __init__(self, cfg, seed=0, option_train_steps=1000):
        self.cfg = cfg
        self.option_train_steps = int(option_train_steps)
        key = jax.random.PRNGKey(seed)
        ka, kc = jax.random.split(key)
        H, L, IS = cfg.ftn_hidden_size, cfg.ftn_layers, cfg.ftn_inner_size
        # History-augmented obs: current raw obs + n previous (obs, action) pairs.
        # The networks see aug_obs_dim; RECODE/registry/membership keep raw obs_dim.
        self.raw_obs_dim = cfg.obs_dim
        self.n_hist = int(getattr(cfg, "history_len", 0))
        self.aug_obs_dim = cfg.obs_dim * (self.n_hist + 1) + cfg.act_dim * self.n_hist
        self.actor = actor_init(ka, self.aug_obs_dim, cfg.act_dim, H, L, IS)
        self.critic = critic_init(kc, self.aug_obs_dim, cfg.act_dim, H, L, IS)
        self.actor_t = jax.tree_util.tree_map(lambda x: x, self.actor)
        self.critic_t = jax.tree_util.tree_map(lambda x: x, self.critic)
        self.actor_opt = adam_init(self.actor)
        self.critic_opt = adam_init(self.critic)

        self.smoother = LateralSmoother(
            H, cfg.ftn_k, square_dim=cfg.lateral_square_dim, kernel_size=cfg.lateral_kernel_size,
            lateral_steps=cfg.lateral_steps, lateral_polyak=cfg.lateral_polyak,
            lateral_noise_sd=cfg.lateral_noise_sd, torus=cfg.lateral_torus)

        self.options: dict[int, dict] = {}  # cid -> {"m_a": jnp, "m_c": jnp}
        self.registry = CentroidRegistry(cfg.obs_dim, cfg)
        self.recode = recode_mod.init_state(cfg.recode_memory_size, cfg.obs_dim)
        self.key = jax.random.fold_in(key, 7)
        self._build_jitted()

    def _next_key(self):
        self.key, sub = jax.random.split(self.key)
        return sub

    # ---- RECODE ---------------------------------------------------------
    def recode_update(self, next_obs_jnp):
        c = self.cfg
        self.recode = recode_mod.update(
            self.recode, next_obs_jnp, c.recode_memory_size, c.recode_k, c.recode_kappa,
            c.recode_decay, c.recode_gamma, c.recode_insertion_probability, self._next_key())

    # ---- jitted reconfig + train ---------------------------------------
    def _build_jitted(self):
        cfg = self.cfg
        H, k = cfg.ftn_hidden_size, cfg.ftn_k
        rlr, rsteps = cfg.reconfig_lr, cfg.reconfig_steps
        smoother = self.smoother
        gamma, tau = cfg.gamma, cfg.tau
        alr, clr, B = cfg.actor_lr, cfg.critic_lr, cfg.batch_size
        goal, shaping, step_cost = cfg.option_goal_reward, cfg.option_shaping, cfg.option_step_cost
        dist_reward = bool(getattr(cfg, "dist_reward", False))
        cql_alpha, cql_n, act_dim = cfg.cql_alpha, cfg.cql_n_actions, cfg.act_dim
        nsteps = self.option_train_steps
        raw = self.raw_obs_dim  # membership/reward use the raw color (first raw dims of aug obs)

        # ---- reconfig: search (m_c, m_a) for one option -----------------
        def _reconfig(actor_p, critic_p, obs, act, returns, key):
            kc, ka = jax.random.split(key)

            def critic_loss(m):  # -cosine(Q(s,a), G_t)
                q = critic_forward(critic_p, obs, act, m)
                qc, rc = q - q.mean(), returns - returns.mean()
                cos = (qc * rc).sum() / (jnp.linalg.norm(qc) * jnp.linalg.norm(rc) + 1e-8)
                return -cos

            m_c_hard, _, c0, c1 = anneal_ste_mask(H, k, critic_loss, rlr, rsteps, kc)
            m_c = smoother.smooth(m_c_hard)

            def actor_loss(m):  # maximize Q under the new critic mask
                return -critic_forward(critic_p, obs, actor_forward(actor_p, obs, m), m_c).mean()

            m_a_hard, _, a0, a1 = anneal_ste_mask(H, k, actor_loss, rlr, rsteps, ka)
            m_a = smoother.smooth(m_a_hard)
            return m_a, m_c, c0, c1
        self._reconfig = jax.jit(_reconfig)

        # ---- membership-relabeled MC returns over the ordered buffer ----
        def _returns(nobs, bnd, cid, centroids):
            # nobs may be history-augmented; membership/reward use the raw color.
            r, d = reward_done(nobs[:, :raw], cid, centroids, bnd, goal, shaping, step_cost, dist_reward)
            return compute_mc_returns(r, d, gamma)
        self._returns = jax.jit(_returns)

        # ---- train: DDPG over the masked subspace + sibling terms -------
        def _train(actor_p, critic_p, actor_t, critic_t, a_os, c_os,
                   obs_b, act_b, nobs_b, bnd_b, centroids, masks_a, masks_c, cids, w, key):
            N = obs_b.shape[0]
            # Per-row term metadata, fixed across the scan: each of the _TERMS
            # options owns a contiguous block of B rows in the (_TERMS*B) batch.
            ma_rows = jnp.repeat(masks_a, B, axis=0)      # (_TERMS*B, H)
            mc_rows = jnp.repeat(masks_c, B, axis=0)
            cid_rows = jnp.repeat(cids, B)                # (_TERMS*B,)
            w_rows = jnp.repeat(w, B)

            def step(carry, _):
                ap, cp, at, ct, aos, cos_, key = carry
                # Extra subkey for CQL random action sampling only when enabled, so
                # the RNG stream (and thus results) is byte-identical to pure DDPG
                # when cql_alpha == 0.
                if cql_alpha != 0.0:
                    key, ik, qk = jax.random.split(key, 3)
                else:
                    key, ik = jax.random.split(key)
                # One pooled batch over all terms (main + siblings) -> a single
                # batch-(_TERMS*B) forward instead of _TERMS small ones.
                idx = jax.random.randint(ik, (_TERMS * B,), 0, N)
                o, a, no, bd = obs_b[idx], act_b[idx], nobs_b[idx], bnd_b[idx]
                # Membership-relabeled reward/done with per-row target centroid.
                # o/no are history-augmented; reward uses the raw color (first dims).
                no_raw = no[:, :raw]
                ids = membership_dev(no_raw, centroids)
                member = (ids == cid_rows).astype(jnp.float32)
                dist = jnp.linalg.norm(no_raw - centroids[cid_rows], axis=1)
                # dist_reward: dense negative distance-to-centroid; else membership.
                r = -dist if dist_reward else goal * member - shaping * dist - step_cost
                # Continuous-control: membership reward bootstraps; only the world
                # boundary terminates (see reward_done).
                d = bd

                # Sum-of-per-term-means == (1/B) * sum_rows w_row * loss_row, so the
                # per-row weighting below reproduces the unbatched loss exactly.
                def critic_loss(cp):
                    a_next = _actor_rows(at, no, ma_rows)
                    q_next = _critic_rows(ct, no, a_next, mc_rows)
                    target = jax.lax.stop_gradient(r + gamma * (1.0 - d) * q_next)
                    q = _critic_rows(cp, o, a, mc_rows)
                    loss = jnp.sum(w_rows * (q - target) ** 2) / B
                    if cql_alpha != 0.0:
                        # Conservative penalty: push Q DOWN on OOD/random actions and
                        # UP on the dataset action (q, reused from the TD term). Each
                        # row keeps its own option critic mask (mc_rows), so the
                        # conservatism is per-option. CQL(H) continuous approximation:
                        # logsumexp over cql_n uniform actions on the tanh box plus the
                        # current policy action ~ soft max_a Q(s,a). Actions are
                        # stop-grad'd -- only the critic is regularized, not the policy.
                        rows = o.shape[0]  # _TERMS * B
                        a_rand = jax.random.uniform(qk, (cql_n, rows, act_dim),
                                                    minval=-1.0, maxval=1.0)
                        q_rand = jax.vmap(lambda ar: _critic_rows(cp, o, ar, mc_rows))(a_rand)
                        a_pi = jax.lax.stop_gradient(_actor_rows(ap, o, ma_rows))
                        q_pi = _critic_rows(cp, o, a_pi, mc_rows)[None]    # (1, rows)
                        q_cat = jnp.concatenate([q_rand, q_pi], axis=0)    # (cql_n+1, rows)
                        # -log(M) keeps the scale ~invariant to the sample count, so
                        # cql_alpha's meaning is stable as cql_n_actions changes.
                        lse = logsumexp(q_cat, axis=0) - jnp.log(cql_n + 1.0)
                        loss = loss + cql_alpha * jnp.sum(w_rows * (lse - q)) / B
                    return loss
                cl, cg = jax.value_and_grad(critic_loss)(cp)
                cp, cos_ = adam_update(cp, cg, cos_, clr)

                def actor_loss(ap):
                    q = _critic_rows(cp, o, _actor_rows(ap, o, ma_rows), mc_rows)
                    return -jnp.sum(w_rows * q) / B
                al, ag = jax.value_and_grad(actor_loss)(ap)
                ap, aos = adam_update(ap, ag, aos, alr)

                at = polyak(at, ap, tau)
                ct = polyak(ct, cp, tau)
                return (ap, cp, at, ct, aos, cos_, key), (cl, al)

            init = (actor_p, critic_p, actor_t, critic_t, a_os, c_os, key)
            (ap, cp, at, ct, aos, cos_, _), (cls, als) = jax.lax.scan(
                step, init, None, length=nsteps)
            return ap, cp, at, ct, aos, cos_, cls[-1], als[-1]
        self._train = jax.jit(_train)

    # ---- option helpers -------------------------------------------------
    def ensure_option(self, cid):
        if cid not in self.options:
            H, k = self.cfg.ftn_hidden_size, self.cfg.ftn_k
            ka, kb = jax.random.split(self._next_key())
            self.options[cid] = {"m_a": self.smoother.smooth(kwta(jax.random.normal(ka, (H,)), k)),
                                 "m_c": self.smoother.smooth(kwta(jax.random.normal(kb, (H,)), k))}
        return self.options[cid]

    def reconfig_option(self, cid, buffer):
        """Search a fresh mask pair for option `cid` (jitted)."""
        cfg = self.cfg
        obs, act, nobs, bnd = buffer  # jnp arrays
        N = obs.shape[0]
        cents = jnp.asarray(self.registry.centroids)
        # MC returns over the ORDERED stream, then gather a random support batch.
        returns_full = self._returns(nobs, bnd, cid, cents)
        n_sup = min(cfg.reconfig_n, N)
        sup = jax.random.permutation(self._next_key(), N)[:n_sup]
        m_a, m_c, c0, c1 = self._reconfig(self.actor, self.critic, obs[sup], act[sup],
                                          returns_full[sup], self._next_key())
        self.options[cid] = {"m_a": m_a, "m_c": m_c}
        return float(c0), float(c1)

    def _siblings(self, cid):
        """Up to 3 options whose actor mask overlaps cid's within (0.1, 0.95)."""
        m_cur = np.asarray(self.options[cid]["m_a"])
        denom = max(m_cur.sum(), 1.0)
        sibs = []
        for sid, so in self.options.items():
            if sid == cid:
                continue
            rho = float((m_cur * np.asarray(so["m_a"])).sum()) / denom
            if 0.1 < rho < 0.95:
                sibs.append(sid)
        return sibs[:3]

    def train_option(self, cid, buffer, consolidate=True):
        """DDPG-train option cid's masked subspace + overlapping siblings (jitted)."""
        self.ensure_option(cid)
        obs, act, nobs, bnd = buffer
        cents = jnp.asarray(self.registry.centroids)
        sibs = self._siblings(cid) if consolidate else []
        # Build the fixed _TERMS-slot term arrays (main + siblings, zero-padded).
        ids = [cid] + sibs
        ma = [self.options[cid]["m_a"]] + [self.options[s]["m_a"] for s in sibs]
        mc = [self.options[cid]["m_c"]] + [self.options[s]["m_c"] for s in sibs]
        w = [1.0] * len(ids)
        while len(ids) < _TERMS:  # pad
            ids.append(cid)
            ma.append(self.options[cid]["m_a"])
            mc.append(self.options[cid]["m_c"])
            w.append(0.0)
        masks_a = jnp.stack(ma)
        masks_c = jnp.stack(mc)
        cids = jnp.asarray(ids, dtype=jnp.int32)
        wj = jnp.asarray(w, dtype=jnp.float32)
        (self.actor, self.critic, self.actor_t, self.critic_t, self.actor_opt,
         self.critic_opt, cl, al) = self._train(
            self.actor, self.critic, self.actor_t, self.critic_t, self.actor_opt,
            self.critic_opt, obs, act, nobs, bnd, cents, masks_a, masks_c, cids, wj,
            self._next_key())
        return float(cl), float(al)

    # ---- diagnostics ----------------------------------------------------
    def mean_overlap(self, which="m_a"):
        cids = sorted(self.options.keys())
        if len(cids) < 2:
            return 0.0
        M = np.stack([np.asarray(self.options[c][which]) for c in cids]).astype(np.float32)
        inter = (M @ M.T) / max(M[0].sum(), 1.0)
        n = len(cids)
        off = inter[~np.eye(n, dtype=bool)]
        return float(off.mean())
