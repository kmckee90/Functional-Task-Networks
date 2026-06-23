"""
model.py - Equinox Conv2dDQNMinAtar for masked DQN on MinAtar.

Port of atari/dqn.py:Conv2dDQNMinAtar. Operates on single observations (no batch
dim) -- batching is handled externally via jax.vmap.
"""

import equinox as eqx
import jax
import jax.numpy as jnp


class Conv2dDQNMinAtar(eqx.Module):
    """
    Masked DQN for MinAtar using all-Conv2d hidden layers.

    Architecture:
        Conv2d layers (each: in -> hidden_size channels, 3x3, pad 1)
        Mask applied channel-wise: h = conv(x) * mask[:, None, None]
        Skip connection from first conv output
        Linear output head: flatten(h0 + h_final) -> max_actions

    Args:
        max_actions: max action space across all games
        hidden_size: number of conv filters per layer (square_dim^2)
        in_channels: max observation channels (padded)
        n_layers: number of conv layers
        key: PRNG key for initialization
        drop_p: dropout probability applied to mask during training
    """

    convs: list
    out_linear: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    max_actions: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    conv_flat_size: int = eqx.field(static=True)
    drop_p: float = eqx.field(static=True)

    def __init__(
        self, max_actions, hidden_size, in_channels, n_layers, *, key, drop_p=0.0
    ):
        self.hidden_size = hidden_size
        self.max_actions = max_actions
        self.n_layers = n_layers
        self.conv_flat_size = hidden_size * 10 * 10
        self.drop_p = drop_p

        keys = jax.random.split(key, n_layers + 1)
        convs = []
        for i in range(n_layers):
            in_c = in_channels if i == 0 else hidden_size
            convs.append(
                eqx.nn.Conv2d(in_c, hidden_size, 3, stride=1, padding=1, key=keys[i])
            )
        self.convs = convs
        self.out_linear = eqx.nn.Linear(self.conv_flat_size, max_actions, key=keys[-1])

    def drop_mask(self, mask, key):
        """Apply Bernoulli dropout to mask. No-op when drop_p == 0 or key is None."""
        k = jnp.sum(mask)
        if key is not None and self.drop_p > 0:
            keep = jax.random.bernoulli(key, 1.0 - self.drop_p, mask.shape)
            j = jnp.sum(keep)
            return mask * keep * (k / j)
        return mask

    def __call__(self, x, mask):
        """
        Forward pass on a single observation.

        Args:
            x: observation, shape (max_channels, 10, 10)
            mask: binary mask, shape (hidden_size,) — caller applies drop_mask for training

        Returns:
            Q-values, shape (max_actions,)
        """
        dm = mask.reshape(-1, 1, 1)  # (hidden_size, 1, 1)

        h0 = self.convs[0](x) * dm
        h = jax.nn.relu(h0)
        h0_flat = h0.reshape(-1)

        for conv in self.convs[1:]:
            h = conv(h) * dm
            h = jax.nn.relu(h) * dm

        q = self.out_linear(h0_flat + h.reshape(-1))
        return q
