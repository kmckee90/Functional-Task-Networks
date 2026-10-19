"""Vision models for the GPU globo explorer — JAX/Flax port of ../globo_explorer/models.py.

Same design as the craftium version (Fluid_1S conv2d reservoir + deep conv2d
decoder), re-expressed in JAX so the whole encode→act→learn path stays on-device:

  * ``Conv2dReservoir`` — a *fixed* (frozen) 2D echo-state network. Its weights are
    plain arrays held on the class (NOT flax params), so they never enter the
    optimizer. Exact per-weight values can't match torch's RNG, but the
    architecture, N(0,1) init, input sparsity mask, and update rule are faithful.
  * ``FeedbackToChannel`` — a frozen random projection of the feedback vector into
    one extra input channel (folded onto the RGB image before the reservoir).
  * ``DeepConvDecoder`` — the ONLY trained module (flax.linen): a shared 3-conv
    trunk with one Linear Q-head per action group (branching DQN).

Convention: images/hidden are NHWC (flax default), unlike the torch NCHW original.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn


# --------------------------------------------------------------------------- #
# Frozen reservoir front-end (plain arrays, not flax params)
# --------------------------------------------------------------------------- #
def _normal(rng: np.random.Generator, shape) -> jnp.ndarray:
    return jnp.asarray(rng.standard_normal(shape), dtype=jnp.float32)


class Conv2dReservoir:
    """Fixed 2D echo-state reservoir (frozen).

        h_{t+1} = (1-tc)*h_t + tc*tanh( conv1(input_scale*x) + conv2(recur_scale*h_t) )

    conv1: strided, non-overlapping downsampler (kernel=stride by default), N(0,1)
    weights masked to ``in_sparsity`` density. conv2: circular-padded recurrent
    conv, dense N(0,1). All weights frozen. Arrays are HWIO for lax.conv.
    """

    def __init__(self, in_channels: int, num_channels: int = 64, kernel_size: int = 7,
                 stride: int = 4, recur_kernel: int = 3, input_scale: float = 2.375 * 10,
                 recur_scale: float = 1.49, tc: float = 1.0, in_sparsity: float = 0.5,
                 seed: int = 0):
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.recur_kernel = recur_kernel
        self.input_scale = input_scale
        self.recur_scale = recur_scale
        self.tc = tc
        rng = np.random.default_rng(seed)
        # conv1 kernel HWIO = (k, k, in_channels, num_channels), sparsified.
        w1 = _normal(rng, (kernel_size, kernel_size, in_channels, num_channels))
        mask = jnp.asarray(rng.random(w1.shape) < in_sparsity, dtype=jnp.float32)
        self.w1 = w1 * mask
        # conv2 kernel HWIO = (rk, rk, num_channels, num_channels), dense.
        self.w2 = _normal(rng, (recur_kernel, recur_kernel, num_channels, num_channels))
        self._dn = jax.lax.conv_dimension_numbers((1, 1, 1, in_channels), self.w1.shape,
                                                   ("NHWC", "HWIO", "NHWC"))

    def hidden_shape(self, in_h: int, in_w: int) -> tuple[int, int, int]:
        out_h = (in_h - self.kernel_size) // self.stride + 1
        out_w = (in_w - self.kernel_size) // self.stride + 1
        return (out_h, out_w, self.num_channels)

    def init_hidden(self, batch: int, in_h: int, in_w: int) -> jnp.ndarray:
        h, w, c = self.hidden_shape(in_h, in_w)
        return jnp.zeros((batch, h, w, c), dtype=jnp.float32)

    def __call__(self, x: jnp.ndarray, hidden: jnp.ndarray) -> jnp.ndarray:
        """x: (B, H, W, C_in) in [0,1]; hidden: (B, h, w, num_channels)."""
        dn1 = jax.lax.conv_dimension_numbers(x.shape, self.w1.shape, ("NHWC", "HWIO", "NHWC"))
        pre = jax.lax.conv_general_dilated(
            self.input_scale * x, self.w1, window_strides=(self.stride, self.stride),
            padding="VALID", dimension_numbers=dn1)
        # circular ("wrap") padding for the recurrent conv → SAME spatial size.
        p = (self.recur_kernel - 1) // 2
        h_pad = jnp.pad(self.recur_scale * hidden, ((0, 0), (p, p), (p, p), (0, 0)), mode="wrap")
        dn2 = jax.lax.conv_dimension_numbers(h_pad.shape, self.w2.shape, ("NHWC", "HWIO", "NHWC"))
        pre = pre + jax.lax.conv_general_dilated(
            h_pad, self.w2, window_strides=(1, 1), padding="VALID", dimension_numbers=dn2)
        return (1.0 - self.tc) * hidden + self.tc * jnp.tanh(pre)


class FeedbackToChannel:
    """Frozen random projection of the feedback vector → one (H,W) channel."""

    def __init__(self, feedback_size: int, in_h: int, in_w: int, seed: int = 1):
        self.in_h, self.in_w = in_h, in_w
        rng = np.random.default_rng(seed)
        self.w = _normal(rng, (feedback_size, in_h * in_w))  # bias = 0 (as in torch)

    def __call__(self, feedback: jnp.ndarray) -> jnp.ndarray:
        """feedback: (B, feedback_size) → (B, in_h, in_w, 1)."""
        b = feedback.shape[0]
        y = jnp.tanh(feedback @ self.w)
        return y.reshape(b, self.in_h, self.in_w, 1)


# --------------------------------------------------------------------------- #
# Trained decoder (flax.linen) — the only module with optimized params
# --------------------------------------------------------------------------- #
class DeepConvDecoder(nn.Module):
    """3× (Conv 3x3 s1 same + ReLU) shared trunk → one Dense Q-head per group."""

    out_sizes: tuple[int, ...]
    layer_size: int = 256

    @nn.compact
    def __call__(self, hidden: jnp.ndarray) -> list[jnp.ndarray]:
        x = hidden
        for _ in range(3):
            x = nn.Conv(self.layer_size, (3, 3), strides=(1, 1), padding="SAME")(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        return [nn.Dense(n)(x) for n in self.out_sizes]


# --------------------------------------------------------------------------- #
# RECODE embedding
# --------------------------------------------------------------------------- #
def downscale_embedding(img_nhwc: jnp.ndarray, out_h: int, out_w: int,
                        grayscale: bool = True) -> jnp.ndarray:
    """Downscale image(s) to a flat RECODE embedding.

    img_nhwc: (H,W,C) or (B,H,W,C) float in [0,1]. Returns (D,) or (B,D) with
    D = out_h*out_w*(1 or C). Uses area-average resize (≈ torch adaptive_avg_pool2d).
    """
    single = img_nhwc.ndim == 3
    x = img_nhwc[None] if single else img_nhwc
    if grayscale and x.shape[-1] == 3:
        w = jnp.asarray([0.299, 0.587, 0.114], dtype=x.dtype)
        x = (x * w).sum(axis=-1, keepdims=True)
    b, _, _, c = x.shape
    x = jax.image.resize(x, (b, out_h, out_w, c), method="linear")
    emb = x.reshape(b, -1)
    return emb[0] if single else emb
