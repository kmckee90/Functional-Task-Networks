"""JAX FTN backbone + AnnealedSTE mask search + SmoothKWTA.

Port of rl/robotics/ftn.py. An FTN is a bank of H independent tiny MLPs read out
through a binary top-k (KWTA) mask; an *option* is a mask pair (m_a, m_c). Params
are plain pytrees (dicts of arrays) so everything jits/vmaps cleanly.

    parallel(x): z = relu-MLP-bank(x) -> (B, H)
    actor : tanh((z * m_a) @ W_a)          critic: (z' * m_c) @ W_c
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# ParallelMLP -- bank of H independent small MLPs
# ---------------------------------------------------------------------------


def parallel_init(key, input_size, output_size, hidden_size, layers, inner_size):
    """Replicates ParallelMLP._init_weights (uniform fan-in bounds, s=6, s_b=0.1)."""
    sizes = [input_size] + [inner_size] * layers + [output_size]
    s, s_b = 6.0, 0.1
    weights, biases = [], []
    for i in range(len(sizes) - 1):
        fan_in, fan_out = sizes[i], sizes[i + 1]
        bound = math.sqrt(s / fan_in) if i < layers else math.sqrt(s / (fan_in + fan_out))
        key, kw, kb = jax.random.split(key, 3)
        weights.append(jax.random.uniform(kw, (hidden_size, sizes[i], sizes[i + 1]),
                                           minval=-bound, maxval=bound))
        bb = s_b / math.sqrt(fan_in)
        biases.append(jax.random.uniform(kb, (hidden_size, sizes[i + 1]), minval=-bb, maxval=bb))
    return {"weights": weights, "biases": biases}


def parallel_forward(p, x):
    """x (B, D) -> (B, H). einsum 'bki,kij->bkj' per layer, relu on hidden layers.

    n_hidden = #matrices - 1 is read from the (static) list length so it survives
    jit -- storing an int inside the params pytree would turn it into a tracer."""
    h = x[:, None, :]  # (B, 1, D) broadcasts over H
    n = len(p["weights"]) - 1
    for i, (W, b) in enumerate(zip(p["weights"], p["biases"])):
        h = jnp.einsum("bki,kij->bkj", h, W) + b
        if i < n:
            h = jax.nn.relu(h)
    return h.squeeze(-1)  # (B, H)


# ---------------------------------------------------------------------------
# Actor / Critic (linear readout stored as (H, out) so y = (z*m) @ W)
# ---------------------------------------------------------------------------


def actor_init(key, obs_dim, act_dim, H, layers, inner_size):
    kp, kl = jax.random.split(key)
    par = parallel_init(kp, obs_dim, 1, H, layers, inner_size)
    bound = 1.0 / math.sqrt(H)  # torch nn.Linear default
    lin = jax.random.uniform(kl, (H, act_dim), minval=-bound, maxval=bound)
    return {"parallel": par, "linear": lin}


def critic_init(key, obs_dim, act_dim, H, layers, inner_size):
    kp, kl = jax.random.split(key)
    par = parallel_init(kp, obs_dim + act_dim, 1, H, layers, inner_size)
    bound = 1.0 / math.sqrt(H)
    lin = jax.random.uniform(kl, (H, 1), minval=-bound, maxval=bound)
    return {"parallel": par, "linear": lin}


def actor_forward(p, obs, mask):
    z = parallel_forward(p["parallel"], obs)
    return jnp.tanh((z * mask.reshape(1, -1)) @ p["linear"])  # (B, act_dim)


def critic_forward(p, obs, act, mask):
    x = jnp.concatenate([obs, act], axis=-1)
    z = parallel_forward(p["parallel"], x)
    return ((z * mask.reshape(1, -1)) @ p["linear"]).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# KWTA + square-grid lateral smoothing (SmoothKWTA)
# ---------------------------------------------------------------------------


def kwta(x, k):
    """Binary top-k mask over a 1D vector."""
    idx = jax.lax.top_k(x, k)[1]
    return jnp.zeros_like(x).at[idx].set(1.0)


def mask_to_grid(m, block_dim):
    """Flat mask -> (block_dim, n_blocks*block_dim) by laying blocks side by side."""
    bs = block_dim * block_dim
    n_blocks = m.shape[0] // bs
    blocks = [m[i * bs:(i + 1) * bs].reshape(block_dim, block_dim) for i in range(n_blocks)]
    return jnp.concatenate(blocks, axis=-1)


def mask_to_vec(grid, block_dim):
    n_blocks = grid.shape[-1] // block_dim
    return jnp.concatenate([grid[:, i * block_dim:(i + 1) * block_dim].reshape(-1)
                            for i in range(n_blocks)], axis=0)


class LateralSmoother:
    """SmoothKWTA: spatially cluster a k-hot mask into a contiguous blob.

    No-op (identity) unless lateral_steps>0 and lateral_polyak<1.0, matching the
    torch LateralSmoother. Uses an all-ones kernel (only relative ordering matters
    before the re-sparsifying top-k), circular padding when torus=True."""

    def __init__(self, hidden_size, k, square_dim=16, kernel_size=17, lateral_steps=0,
                 lateral_polyak=1.0, lateral_noise_sd=0.0, torus=False):
        self.H = hidden_size
        self.k = k
        self.square_dim = square_dim
        self.kernel_size = kernel_size
        self.lateral_steps = int(lateral_steps)
        self.lateral_polyak = float(lateral_polyak)
        self.lateral_noise_sd = float(lateral_noise_sd)
        self.torus = torus
        self.enabled = self.lateral_steps > 0 and self.lateral_polyak < 1.0
        if self.enabled and hidden_size % (square_dim * square_dim) != 0:
            raise ValueError(f"hidden_size={hidden_size} must be a multiple of square_dim**2")

    def _conv(self, g):
        p = self.kernel_size // 2
        mode = "wrap" if self.torus else "constant"
        gp = jnp.pad(g, ((p, p), (p, p)), mode=mode)
        ker = jnp.ones((self.kernel_size, self.kernel_size), g.dtype)
        out = jax.lax.conv_general_dilated(
            gp[None, None], ker[None, None], window_strides=(1, 1), padding="VALID")
        return out[0, 0]

    def smooth(self, mask, key=None):
        """Polyak-mix `lateral_steps` smoothing passes, then re-sparsify to top-k.
        Identity when disabled. `key` only needed if lateral_noise_sd>0."""
        if not self.enabled:
            return mask
        best = mask.astype(jnp.float32)
        for _ in range(self.lateral_steps):
            g = mask_to_grid(best, self.square_dim)
            g = self._conv(g)
            if self.lateral_noise_sd > 0 and key is not None:
                key, nk = jax.random.split(key)
                g = g + jax.random.normal(nk, g.shape) * self.lateral_noise_sd
            m_lat = mask_to_vec(g, self.square_dim)
            best = self.lateral_polyak * best + (1.0 - self.lateral_polyak) * m_lat
        return kwta(best, self.k)


# ---------------------------------------------------------------------------
# AnnealedSTE mask search
# ---------------------------------------------------------------------------


def anneal_ste_mask(H, k, loss_fn, lr, n_steps, key, noise_scale=0.1):
    """Search a k-hot mask minimizing loss_fn(soft_mask) via SGD on logits with a
    straight-through estimator and annealed exploration noise (port of
    ftn.anneal_ste_mask). Returns (best_hard_mask, final_logits, first_loss,
    best_loss). loss_fn maps (H,) soft mask -> scalar.

    NOT jitted itself (loss_fn is a Python closure over arrays); call it from
    inside an already-jitted function -- the lax.scan below compiles normally."""
    key, k0 = jax.random.split(key)
    logits0 = noise_scale * jax.random.normal(k0, (H,))
    denom = max(1.0, n_steps / 4)

    def masked_loss(logits, noise):
        sg = jax.lax.stop_gradient(logits)
        hard = kwta(sg + noise, k)
        mask_ste = hard + logits - sg  # STE: identity grad to logits, value = hard
        return loss_fn(mask_ste), hard

    def step(carry, i):
        logits, best, lowest, first, key = carry
        std = jnp.std(logits)
        scale = jnp.where(std > 1e-6, std, 0.1)
        tau = scale * jnp.exp(-i / denom)
        key, nk = jax.random.split(key)
        noise = jax.random.normal(nk, (H,)) * tau
        (loss, hard), grad = jax.value_and_grad(masked_loss, has_aux=True)(logits, noise)
        first = jnp.where(i == 0, loss, first)
        improved = loss < lowest
        best = jnp.where(improved, hard, best)
        lowest = jnp.where(improved, loss, lowest)
        logits = logits - lr * grad
        return (logits, best, lowest, first, key), None

    init = (logits0, jnp.zeros((H,)), jnp.float32(jnp.inf), jnp.float32(0.0), key)
    (logits, best, lowest, first, _), _ = jax.lax.scan(
        step, init, jnp.arange(n_steps))
    return best, logits, first, lowest
