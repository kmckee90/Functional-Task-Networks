"""Single-lifetime globo env on the warpcraft GPU engine (num_worlds=1).

GPU port of ../globo_explorer/env.py. Instead of a craftium/Luanti subprocess,
this drives warpcraft's globo world (Warp/torch on CUDA) at num_worlds=1 for one
continuous real-world-style lifetime. All observations/actions/rewards stay on
the GPU and are exposed to JAX zero-copy via Warp's dlpack bridge — no host sync
in the hot loop.

Action model (faithful analog of the original's `factored` branching control):
warpcraft's action tensor is [1,4] int32 over channels LOCOMOTION(10), LOOK(5),
INTERACT(5), CRAFT/inventory(7). The branching DQN gets one head per *active*
channel. Default = (LOCOMOTION, LOOK): a 10-way move primitive plus a concurrent
5-way look, which is the closest warpcraft allows to the craftium version's
simultaneous move+strafe+jump+look (warpcraft bundles move/strafe/jump into the
single 10-way LOCOMOTION channel). INTERACT/CRAFT can be enabled via config for
break/place/egg/sleep actions.

Obs pipeline mirrors the original: render at `render_*`, downscale to the agent's
`obs_*` (bilinear/area), feed RGB (+feedback channel, added by the agent) to the
reservoir; RECODE embeds a further-downscaled grayscale image.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import warp as wp

# Channel sizes from warpcraft.engine.contracts (verified via probe):
#   LOCOMOTION=10, LOOK=5, INTERACT=5, CRAFT/GloboInventoryAction=7
CHANNEL_SIZES = (10, 5, 5, 7)
CHANNEL_NAMES = ("locomotion", "look", "interact", "craft")


def _t2j(t):
    """torch (CUDA) -> jax, zero-copy via Warp dlpack."""
    return wp.to_jax(wp.from_torch(t))


class GloboGPUEnv:
    """num_worlds=1 warpcraft globo, JAX-facing. One continuous lifetime with
    auto-reset on death/timeout (episode boundaries within the lifetime)."""

    def __init__(
        self,
        seed: int = 7,
        obs_width: int = 48,
        obs_height: int = 24,
        render_width: int = 64,
        render_height: int = 64,
        recode_down_h: int = 32,
        recode_down_w: int = 48,
        recode_grayscale: bool = True,
        active_channels: tuple[int, ...] = (0, 1),  # locomotion + look
        max_episode_steps: int = 5000,
        gpu_id: int = 0,
        advance_weather: bool = True,
    ):
        from warpcraft.env.globo import create_globo, GloboConfig
        from warpcraft.env.globo import step as _globo_step

        self.obs_width, self.obs_height = obs_width, obs_height
        self.render_width, self.render_height = render_width, render_height
        self.recode_down_h, self.recode_down_w = recode_down_h, recode_down_w
        self.recode_grayscale = recode_grayscale
        self.active_channels = tuple(active_channels)
        self.head_sizes = tuple(CHANNEL_SIZES[c] for c in self.active_channels)
        self.head_names = tuple(CHANNEL_NAMES[c] for c in self.active_channels)
        self._advance_weather = advance_weather
        self._globo_step = _globo_step

        wp.init()
        cfg = GloboConfig(world_seed=int(seed), max_episode_steps=int(max_episode_steps))
        self.env = create_globo(
            cfg, num_worlds=1, exec_mode=1, gpu_id=gpu_id,
            render_width=render_width, render_height=render_height, auto_reset=True,
        )
        self._steps = 0

    # -- observation ------------------------------------------------------- #
    def _read_obs(self) -> dict:
        """Build the JAX obs from the freshly rendered frame (all on GPU)."""
        rgba = _t2j(self.env.frame.rgba)           # (1,H,W,4) uint8 NHWC
        img = rgba[..., :3].astype(jnp.float32) / 255.0
        # downscale render -> agent obs resolution (area-average)
        obs_img = jax.image.resize(img, (1, self.obs_height, self.obs_width, 3), method="linear")
        return {"image": obs_img, "render_rgb": img}

    def reset(self) -> dict:
        self.env.episode.reset[0] = 1
        self._step_engine(write_action=False)
        self._steps = 0
        return self._read_obs()

    # -- action ------------------------------------------------------------ #
    def write_action(self, head_indices: jnp.ndarray) -> None:
        """head_indices: (num_active_heads,) int — one chosen index per active channel.
        Assembled into the [4] engine action and written GPU-side (no host sync)."""
        act = jnp.zeros((4,), dtype=jnp.int32)
        for h, ch in enumerate(self.active_channels):
            act = act.at[ch].set(head_indices[h].astype(jnp.int32))
        self.env.episode.action[0, :] = wp.to_torch(wp.from_jax(act))

    def _step_engine(self, write_action: bool = True) -> None:
        if self._advance_weather:
            # module-level step advances weather + renders; action already in tensor.
            self._globo_step(self.env, None)
        else:
            self.env.step()
            self.env.render()

    def step(self, head_indices: jnp.ndarray) -> tuple[dict, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Apply action, step one engine tick, return (obs, env_reward, terminal, timeout)
        as JAX scalars. `terminal` = real episode end (death); `timeout` = truncation.
        auto_reset re-spawns the world on the *next* step after done."""
        self.write_action(head_indices)
        self._step_engine()
        self._steps += 1
        reward = _t2j(self.env.episode.reward)[0]                 # ()
        done = _t2j(self.env.episode.done)[0].astype(jnp.float32)
        tl = _t2j(self.env.episode.time_limit)[0].astype(jnp.float32) if hasattr(
            self.env.episode, "time_limit") else jnp.float32(0.0)
        timeout = done * tl
        terminal = done * (1.0 - tl)                              # death, not timeout
        return self._read_obs(), reward, terminal, timeout

    # -- convenience ------------------------------------------------------- #
    @property
    def player_health(self) -> float:
        return float(self.env.player.health[0])
