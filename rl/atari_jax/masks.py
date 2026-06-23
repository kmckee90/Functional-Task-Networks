"""
masks.py - KWTA + lateral smoothing in pure JAX.

Port of the parent project's masks.py (kwta, SmoothKWTAFast.step_lateral,
and gradient-based mask reconfiguration).
"""

import jax
import jax.numpy as jnp
import optax


def kwta(x, k):
    """
    K-winner-take-all: return binary mask with 1s at top-k values.

    Args:
        x: input tensor, shape (n,)
        k: number of winners

    Returns:
        binary mask, shape (n,)
    """
    top_k_idx = jnp.argsort(-x)[:k]
    return jnp.zeros_like(x).at[top_k_idx].set(1.0)


def step_lateral(m, square_dim, kernel_size=17):
    """
    Apply lateral smoothing via uniform convolution on the 2D mask grid.

    Port of SmoothKWTAFast.step_lateral: reshape to (square_dim, square_dim),
    convolve with uniform kernel, reshape back to flat.

    Args:
        m: flat mask, shape (hidden_size,)
        square_dim: side length of square grid
        kernel_size: convolution kernel size

    Returns:
        smoothed flat mask, shape (hidden_size,)
    """
    # Reshape to 2D grid: (1, 1, H, W) for lax.conv
    grid = m.reshape(1, 1, square_dim, square_dim)

    # Uniform averaging kernel: (out_c, in_c, kH, kW)
    kernel = jnp.ones((1, 1, kernel_size, kernel_size))
    padding = kernel_size // 2

    smoothed = jax.lax.conv_general_dilated(
        grid,
        kernel,
        window_strides=(1, 1),
        padding=((padding, padding), (padding, padding)),
    )
    return smoothed.reshape(-1)


def reconfigure_mask(model, obs, cfg, key, m_init):
    """
    Gradient-based mask discovery for a game.

    1. Compute Q-targets with all-ones mask
    2. Optimize mask params via RMSprop to minimize MSE on Q-values
       (straight-through estimator for Bernoulli sampling)
    3. Apply lateral smoothing
    4. Apply KWTA sparsification

    Args:
        model: Conv2dDQNMinAtar (equinox module)
        obs: batch of preprocessed observations, shape (N, C, 10, 10)
        cfg: Config dataclass
        key: PRNG key
        m_init: initial mask logits, shape (hidden_size,)

    Returns:
        (binary_mask, final_logits) — mask shape (hidden_size,), logits shape (hidden_size,)
    """
    hidden_size = cfg.hidden_size
    n = min(cfg.reconfig_sample_size, obs.shape[0])
    obs = obs[:n]

    # Q-targets with all-ones mask
    ones_mask = jnp.ones(hidden_size)
    q_targets = jax.vmap(lambda o: model(o, ones_mask))(obs)
    q_targets = jax.lax.stop_gradient(q_targets)

    opt = optax.rmsprop(cfg.reconfig_lr)
    opt_state = opt.init(m_init)

    def loss_fn(m_params, key_step):
        sm = jax.nn.sigmoid(m_params)
        bernoulli = jax.random.bernoulli(key_step, sm).astype(jnp.float32)
        mask = bernoulli + sm - jax.lax.stop_gradient(sm)
        q_pred = jax.vmap(lambda o: model(o, mask))(obs)
        return jnp.mean((q_pred - q_targets) ** 2)

    grad_fn = jax.grad(loss_fn)

    def reconfig_step(carry, _):
        m_params, opt_state, rng = carry
        rng, step_key = jax.random.split(rng)
        grads = grad_fn(m_params, step_key)
        updates, opt_state = opt.update(grads, opt_state)
        m_params = optax.apply_updates(m_params, updates)
        return (m_params, opt_state, rng), None

    (m_final, _, _), _ = jax.lax.scan(
        reconfig_step, (m_init, opt_state, key), None, length=cfg.reconfig_steps
    )

    # Lateral smoothing
    m_smooth = m_final
    if cfg.lateral_polyak < 1.0 and cfg.lateral_steps > 0:

        def lateral_step(m_val, _):
            m_lat = step_lateral(m_val, cfg.square_dim, cfg.kernel_size)
            m_out = cfg.lateral_polyak * m_val + (1 - cfg.lateral_polyak) * m_lat
            return m_out, None

        m_smooth, _ = jax.lax.scan(
            lateral_step, m_smooth, None, length=cfg.lateral_steps
        )

    # KWTA sparsification
    m_binary = kwta(m_smooth, cfg.k)
    return m_binary, m_final
