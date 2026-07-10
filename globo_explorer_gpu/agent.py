"""GloboExplorerBranchingDQN — JAX/Flax port of ../globo_explorer/agent.py.

Faithful single-lifetime explorer:
  * frozen conv2d reservoir + frozen feedback projection encode the RGB obs
    (+ a feedback channel of prev action/reward/binned-novelty) into a hidden
    state carried across the lifetime;
  * two branching Q-nets (DeepConvDecoder) — an intrinsic stream (RECODE novelty,
    discount `gamma`) and a survival stream (env reward − death penalty, discount
    `gamma_env`) — act on the *combined* greedy action (NGU/Agent57-style) with
    independent Polyak target rates;
  * RECODE gives intrinsic novelty over the downscaled grayscale image, with an
    online (collection-time) term and an offline (recomputed) term;
  * a single-stream GPU replay buffer + goal buffer with prioritized trajectory
    replay pushes a sparse frontier bonus backward via repeated TD backups.

Everything is JAX on-device. Only the two Q-nets carry optimized params; the
reservoir/feedback/RECODE are non-trainable.
"""
from __future__ import annotations

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from models import Conv2dReservoir, DeepConvDecoder, FeedbackToChannel, downscale_embedding
from recode import RECODE, knn_reward


class ReplayBuffer:
    """Fixed-capacity ring of transitions, all fields as JAX arrays on-device."""

    def __init__(self, capacity: int, hidden_shape, emb_dim: int, n_heads: int):
        h, w, c = hidden_shape
        self.capacity = capacity
        self.z = jnp.zeros((capacity, h, w, c), jnp.float32)
        self.z_next = jnp.zeros((capacity, h, w, c), jnp.float32)
        self.emb = jnp.zeros((capacity, emb_dim), jnp.float32)
        self.action = jnp.zeros((capacity, n_heads), jnp.int32)
        self.env_r = jnp.zeros((capacity,), jnp.float32)
        self.online_nov = jnp.zeros((capacity,), jnp.float32)
        self.terminal = jnp.zeros((capacity,), jnp.float32)
        self.boundary = jnp.zeros((capacity,), jnp.float32)
        self.ptr = 0
        self.size = 0

    def add(self, z, z_next, emb, action, env_r, online_nov, terminal, boundary):
        i = self.ptr
        self.z = self.z.at[i].set(z)
        self.z_next = self.z_next.at[i].set(z_next)
        self.emb = self.emb.at[i].set(emb)
        self.action = self.action.at[i].set(action)
        self.env_r = self.env_r.at[i].set(env_r)
        self.online_nov = self.online_nov.at[i].set(online_nov)
        self.terminal = self.terminal.at[i].set(terminal)
        self.boundary = self.boundary.at[i].set(boundary)
        self.ptr = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def gather(self, idx):
        return dict(z=self.z[idx], z_next=self.z_next[idx], emb=self.emb[idx],
                    action=self.action[idx], env_r=self.env_r[idx],
                    online_nov=self.online_nov[idx], terminal=self.terminal[idx])

    def add_block(self, src: "ReplayBuffer", idx):
        """Copy a block of transitions (by index) from another buffer into this ring."""
        n = int(idx.shape[0])
        pos = (self.ptr + jnp.arange(n)) % self.capacity
        for field in ("z", "z_next", "emb", "action", "env_r", "online_nov", "terminal", "boundary"):
            setattr(self, field, getattr(self, field).at[pos].set(getattr(src, field)[idx]))
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)


class GloboExplorerBranchingDQN:
    def __init__(self, cfg: dict, head_sizes: tuple[int, ...], key):
        self.cfg = cfg
        self.head_sizes = tuple(head_sizes)
        self.n_heads = len(head_sizes)
        oh, ow = cfg["obs_height"], cfg["obs_width"]
        self.obs_h, self.obs_w = oh, ow

        # RECODE embedding dim.
        c_emb = 1 if not cfg["recode_rgb"] else 3
        self.emb_dim = cfg["recode_down_h"] * cfg["recode_down_w"] * c_emb
        self.nov_scale = 1.0 / math.sqrt(max(1, self.emb_dim))
        self.num_novelty_bins = cfg["num_novelty_bins"]
        self.feedback_size = sum(self.head_sizes) + 1 + self.num_novelty_bins

        # Frozen front-end.
        self.reservoir = Conv2dReservoir(
            in_channels=4, num_channels=cfg["reservoir_channels"],
            kernel_size=cfg["reservoir_kernel"], stride=cfg["reservoir_stride"],
            recur_kernel=cfg["reservoir_recur_kernel"], input_scale=cfg["reservoir_input_scale"],
            recur_scale=cfg["reservoir_recur_scale"], tc=cfg["reservoir_tc"],
            in_sparsity=cfg["reservoir_sparsity"], seed=cfg.get("seed", 0))
        self.feedback = FeedbackToChannel(self.feedback_size, oh, ow, seed=cfg.get("seed", 0) + 1)
        self.hidden_shape = self.reservoir.hidden_shape(oh, ow)  # (h,w,C)

        # Trained branching Q-nets (intrinsic + survival).
        self.qnet = DeepConvDecoder(out_sizes=self.head_sizes, layer_size=cfg["decoder_layer_size"])
        k1, k2 = jax.random.split(key)
        dummy = jnp.zeros((1,) + self.hidden_shape, jnp.float32)
        p_i = self.qnet.init(k1, dummy)
        p_e = self.qnet.init(k2, dummy)
        self.params = {"intr": p_i, "env": p_e}
        self.target = {"intr": p_i, "env": p_e}

        self.opt = optax.chain(
            optax.clip_by_global_norm(cfg["grad_clip"]),
            optax.rmsprop(cfg["lr"]),
        )
        self.opt_state = self.opt.init(self.params)

        # RECODE density.
        self.density = RECODE(
            observation_size=self.emb_dim, memory_size=cfg["recode_memory"],
            ema_k=cfg["recode_k"], reward_k=cfg["recode_knn_k"],
            relative_tolerance=cfg["recode_kappa"], decay_rate=cfg["recode_decay"],
            gamma=cfg["recode_gamma"], insertion_probability=cfg["recode_insertion_probability"])

        # Replay + goal buffers.
        self.replay = ReplayBuffer(cfg["replay_capacity"], self.hidden_shape, self.emb_dim, self.n_heads)
        self.goal = ReplayBuffer(cfg["goal_memory_length"], self.hidden_shape, self.emb_dim, self.n_heads)

        self.epsilon = cfg["epsilon_start"]
        self.cur_step = 0
        self._build_jitted()

    # ------------------------------------------------------------------ #
    # Encoding / embedding
    # ------------------------------------------------------------------ #
    def init_hidden(self):
        return self.reservoir.init_hidden(1, self.obs_h, self.obs_w)

    def build_feedback(self, prev_action, prev_reward, prev_novelty):
        """prev_action: (n_heads,) int or None; returns (1, feedback_size)."""
        parts = []
        for h, n in enumerate(self.head_sizes):
            oneh = jnp.zeros((n,), jnp.float32)
            if prev_action is not None:
                oneh = oneh.at[prev_action[h]].set(1.0)
            parts.append(oneh)
        parts.append(jnp.array([prev_reward], jnp.float32))
        # novelty -> tanh -> bin -> one-hot
        b = jnp.zeros((self.num_novelty_bins,), jnp.float32)
        binned = jnp.clip(((jnp.tanh(prev_novelty) * 0.5 + 0.5) * self.num_novelty_bins).astype(jnp.int32),
                          0, self.num_novelty_bins - 1)
        b = b.at[binned].set(1.0)
        parts.append(b)
        return jnp.concatenate(parts)[None, :]

    def encode(self, img, hidden, feedback_vec):
        """img (1,H,W,3), hidden (1,h,w,C), feedback_vec (1,fb) -> new hidden."""
        ch = self.feedback(feedback_vec)               # (1,H,W,1)
        x = jnp.concatenate([img, ch], axis=-1)        # (1,H,W,4)
        return self.reservoir(x, hidden)

    def embed(self, img):
        return downscale_embedding(img[0], self.cfg["recode_down_h"], self.cfg["recode_down_w"],
                                   grayscale=not self.cfg["recode_rgb"])

    def novelty(self, emb):
        return self.density.get_reward(emb[None] if emb.ndim == 1 else emb) * self.nov_scale

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #
    def _build_jitted(self):
        qnet = self.qnet
        res, fbproj = self.reservoir, self.feedback
        head_sizes, n_heads, nbins = self.head_sizes, self.n_heads, self.num_novelty_bins
        reward_k, nov_scale = self.cfg["recode_knn_k"], self.nov_scale
        rdh, rdw = self.cfg["recode_down_h"], self.cfg["recode_down_w"]
        gray = not self.cfg["recode_rgb"]

        def q_values(params, z):
            return qnet.apply(params, z)  # list of (B, head_size)

        def feedback_vec(prev_action, prev_r, prev_nov):
            parts = [jnp.zeros((head_sizes[h],), jnp.float32).at[prev_action[h]].set(1.0)
                     for h in range(n_heads)]
            parts.append(jnp.reshape(prev_r, (1,)))
            b = jnp.zeros((nbins,), jnp.float32)
            binned = jnp.clip(((jnp.tanh(prev_nov) * 0.5 + 0.5) * nbins).astype(jnp.int32), 0, nbins - 1)
            parts.append(b.at[binned].set(1.0))
            return jnp.concatenate(parts)[None, :]

        def eps_greedy(qi, qe, epsilon, key):
            keys = jax.random.split(key, n_heads * 2)
            out = []
            for h in range(n_heads):
                greedy = jnp.argmax(qi[h][0] + qe[h][0])
                rand = jax.random.randint(keys[h], (), 0, head_sizes[h])
                out.append(jnp.where(jax.random.uniform(keys[n_heads + h]) < epsilon,
                                     rand, greedy).astype(jnp.int32))
            return jnp.stack(out)

        @jax.jit
        def act_fn(params, z, epsilon, key):
            return eps_greedy(q_values(params["intr"], z), q_values(params["env"], z), epsilon, key)
        self._act_fn = act_fn

        @jax.jit
        def perceive_fn(params, hidden, img, prev_action, prev_r, prev_nov, epsilon, key, dmem, dcounts):
            """One fused, host-sync-free per-step compute: encode → act → embed → novelty."""
            ch = fbproj(feedback_vec(prev_action, prev_r, prev_nov))
            z = res(jnp.concatenate([img, ch], axis=-1), hidden)
            action = eps_greedy(q_values(params["intr"], z), q_values(params["env"], z), epsilon, key)
            emb = downscale_embedding(img[0], rdh, rdw, grayscale=gray)
            nov = knn_reward(emb[None], dmem, dcounts, reward_k)[0] * nov_scale
            return z, action, emb, nov
        self._perceive_fn = perceive_fn

        cfg = self.cfg
        gamma, gamma_env = cfg["gamma"], cfg["gamma_env"]
        env_coef, death_pen = cfg["env_reward_coef"], cfg["death_penalty"]
        online_coef, offline_coef = cfg["online_coef"], cfg["offline_coef"]
        use_on, use_off = cfg["use_online_novelty"], cfg["use_offline_novelty"]
        nov_scale, reward_k = self.nov_scale, cfg["recode_knn_k"]
        tau, tau_env = cfg["target_update_tau"], cfg["target_update_tau_env"]

        def td_loss(params, target, batch, dmem, dcounts, sparse_scale):
            z, z_next = batch["z"], batch["z_next"]
            action, env_r = batch["action"], batch["env_r"]
            terminal = batch["terminal"]
            not_term = 1.0 - terminal
            # intrinsic reward: online (stored) + offline (recomputed live) novelty
            r_intr = jnp.zeros_like(env_r)
            if use_on:
                r_intr = r_intr + online_coef * batch["online_nov"]
            if use_off:
                off = knn_reward(batch["emb"], dmem, dcounts, reward_k) * nov_scale
                r_intr = r_intr + offline_coef * off
            # sparse frontier bonus on the highest-novelty transition in the batch
            score = batch["online_nov"] if not use_off else off
            best = jnp.argmax(score)
            r_intr = r_intr.at[best].multiply(sparse_scale)
            r_env = env_coef * env_r - death_pen * terminal

            qi_next = q_values(target["intr"], z_next)
            qe_next = q_values(target["env"], z_next)
            qi = q_values(params["intr"], z)
            qe = q_values(params["env"], z)
            loss = 0.0
            for h in range(self.n_heads):
                a_next = jnp.argmax(qi_next[h] + qe_next[h], axis=1)
                bi = jnp.arange(z.shape[0])
                tgt_i = r_intr + gamma * qi_next[h][bi, a_next] * not_term
                tgt_e = r_env + gamma_env * qe_next[h][bi, a_next] * not_term
                qi_sa = qi[h][bi, action[:, h]]
                qe_sa = qe[h][bi, action[:, h]]
                loss = loss + jnp.mean((qi_sa - tgt_i) ** 2) + jnp.mean((qe_sa - tgt_e) ** 2)
            return loss / self.n_heads

        @jax.jit
        def update_fn(params, target, opt_state, batch, dmem, dcounts, sparse_scale):
            loss, grads = jax.value_and_grad(td_loss)(params, target, batch, dmem, dcounts, sparse_scale)
            updates, opt_state = self.opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            new_target = {
                "intr": optax.incremental_update(params["intr"], target["intr"], tau),
                "env": optax.incremental_update(params["env"], target["env"], tau_env),
            }
            return params, new_target, opt_state, loss
        self._update_fn = update_fn

    def act(self, hidden, key):
        return self._act_fn(self.params, hidden, jnp.float32(self.epsilon), key)

    def perceive(self, hidden, img, prev_action, prev_r, prev_nov, key):
        """Fused jitted step compute. prev_* are device arrays; returns (z, action, emb, nov)."""
        return self._perceive_fn(
            self.params, hidden, img, prev_action, prev_r, prev_nov,
            jnp.float32(self.epsilon), key,
            self.density.state.memory, self.density.state.counts)

    def q_head_values(self, hidden):
        """For inference/inspection: combined (intr+env) Q per head."""
        qi = self.qnet.apply(self.params["intr"], hidden)
        qe = self.qnet.apply(self.params["env"], hidden)
        return [qi[h] + qe[h] for h in range(self.n_heads)]

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def remember(self, z, z_next, emb, action, env_r, online_nov, terminal, boundary):
        self.replay.add(z, z_next, emb, action, env_r, online_nov, terminal, boundary)

    def update_density(self, emb, key):
        self.density.update(emb, key)

    def step_schedule(self):
        c = self.cfg
        frac = min(1.0, self.cur_step / max(1, c["epsilon_decay"]))
        self.epsilon = c["epsilon_start"] + frac * (c["epsilon_end"] - c["epsilon_start"])
        self.cur_step += 1

    def _sample_block_start(self, buf: ReplayBuffer, mode: str, key) -> int:
        L = self.cfg["batch"]
        hi = max(1, buf.size - L)
        if mode == "last":
            return max(0, buf.size - L)
        if mode == "goal":
            return max(0, buf.size - L)  # goal buffer: most-recent inserted trajectory
        return int(jax.random.randint(key, (), 0, hi))

    def train_round(self, iters: int, key):
        c = self.cfg
        L = c["batch"]
        if self.replay.size < max(c["warmup"], L):
            return None
        # --- offline frontier scan over the main buffer ---
        scan_n = min(self.replay.size, c["offline_scan_max"])
        emb_scan = self.replay.emb[:scan_n]
        nov = knn_reward(emb_scan, self.density.state.memory, self.density.state.counts,
                         c["recode_knn_k"]) * self.nov_scale
        frontier = int(jnp.argmax(nov))
        main_best = float(nov[frontier]) * c["main_sparse_reward_scale"]
        # --- move the frontier trajectory block into the goal buffer ---
        goal_best = 0.0
        if self.goal.size >= L:
            gnov = knn_reward(self.goal.emb[:self.goal.size], self.density.state.memory,
                              self.density.state.counts, c["recode_knn_k"]) * self.nov_scale
            goal_best = float(jnp.max(gnov)) * c["goal_sparse_reward_scale"]
        if main_best >= goal_best:
            start = max(0, min(frontier - L + 1, self.replay.size - L))
            self.goal.add_block(self.replay, jnp.arange(start, start + L))

        modes = ["goal", "last", "random"] if c["prioritized_replay"] else ["random"]
        losses = []
        for it in range(iters):
            key, sk, uk = jax.random.split(key, 3)
            mode = modes[it % len(modes)]
            buf = self.goal if (mode == "goal" and self.goal.size >= L) else self.replay
            m = mode if buf is self.goal else ("last" if mode == "goal" else mode)
            start = self._sample_block_start(buf, m, sk)
            idx = jnp.arange(start, start + L)
            batch = buf.gather(idx)
            sparse = c["goal_sparse_reward_scale"] if buf is self.goal else c["main_sparse_reward_scale"]
            self.params, self.target, self.opt_state, loss = self._update_fn(
                self.params, self.target, self.opt_state, batch,
                self.density.state.memory, self.density.state.counts, jnp.float32(sparse))
            losses.append(float(loss))
        return float(np.mean(losses)) if losses else None
