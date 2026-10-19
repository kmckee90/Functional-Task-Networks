"""
env_utils.py - Environment creation for Atari and MinAtar.

Atari: uses gymnasium's built-in AtariPreprocessing (grayscale, resize, frame
skip, noop reset) and FrameStackObservation for frame stacking.

MinAtar: wraps the native minatar.Environment into a gymnasium-compatible
interface with observations transposed to CHW and zero-padded to a fixed
channel count so the shared CNN works across all games.
"""

import gymnasium as gym
import ale_py
import numpy as np

gym.register_envs(ale_py)


# ═══════════════════════════════════════════════════════════════════════
# Atari (full)
# ═══════════════════════════════════════════════════════════════════════


def make_atari_env(game_name, cfg, seed=None):
    """
    Create a preprocessed Atari environment.

    Args:
        game_name: Short game name (e.g. "Pong", "Breakout")
        cfg: AtariConfig instance
        seed: Optional random seed

    Returns:
        Wrapped gymnasium environment with observations of shape
        (frame_stack, 84, 84) as uint8.
    """
    env_id = f"ALE/{game_name}-v5"
    env = gym.make(env_id, frameskip=1)  # frameskip=1 since AtariPreprocessing handles it

    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=cfg.FRAME_SKIP,
        screen_size=cfg.IMG_SIZE,
        terminal_on_life_loss=cfg.EPISODIC_LIFE,
        grayscale_obs=cfg.GRAYSCALE,
        scale_obs=False,  # keep uint8, we'll normalize in the model
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=cfg.FRAME_STACK)

    if seed is not None:
        env.reset(seed=seed)

    return env


# ═══════════════════════════════════════════════════════════════════════
# MinAtar
# ═══════════════════════════════════════════════════════════════════════


class MinAtarWrapper:
    """
    Gymnasium-compatible wrapper around minatar.Environment.

    Observations are transposed from HWC (10,10,C) to CHW (pad_channels,10,10)
    and stored as uint8 (0 or 255) so the model's /255 normalization yields
    0.0 / 1.0, matching the Atari path.
    """

    def __init__(self, game_name, pad_channels=None, seed=None):
        from minatar import Environment

        self.env = Environment(game_name)
        if seed is not None:
            self.env.seed(seed)
        self.env.reset()

        self.native_channels = self.env.n_channels
        self.pad_channels = pad_channels or self.native_channels
        self._n_actions = self.env.num_actions()

        # Gymnasium-compatible action_space
        self.action_space = _DiscreteSpace(self._n_actions)

    def _get_obs(self):
        state = self.env.state()               # (10, 10, C) bool
        obs = np.transpose(state, (2, 0, 1))   # (C, 10, 10)
        obs = obs.astype(np.uint8) * 255
        if self.pad_channels > self.native_channels:
            pad = np.zeros(
                (self.pad_channels - self.native_channels, 10, 10), dtype=np.uint8
            )
            obs = np.concatenate([obs, pad], axis=0)
        return obs

    def reset(self, seed=None):
        if seed is not None:
            self.env.seed(seed)
        self.env.reset()
        return self._get_obs(), {}

    def step(self, action):
        reward, terminated = self.env.act(action)
        obs = self._get_obs()
        return obs, reward, terminated, False, {}

    def close(self):
        pass


class _DiscreteSpace:
    """Minimal gymnasium-compatible discrete action space."""

    def __init__(self, n):
        self.n = n

    def sample(self):
        return np.random.randint(self.n)


def get_minatar_max_channels(game_names):
    """Return the maximum number of observation channels across MinAtar games."""
    from minatar import Environment

    max_c = 0
    for name in game_names:
        env = Environment(name)
        max_c = max(max_c, env.n_channels)
    return max_c


def make_minatar_env(game_name, cfg, seed=None, pad_channels=None):
    """
    Create a MinAtar environment with gymnasium-compatible interface.

    Args:
        game_name: MinAtar game name (e.g. "asterix", "breakout")
        cfg: AtariConfig instance (with USE_MINATAR=True)
        seed: Optional random seed
        pad_channels: Zero-pad observations to this many channels.
                      If None, uses the game's native channel count.

    Returns:
        MinAtarWrapper with observations of shape (pad_channels, 10, 10) uint8.
    """
    return MinAtarWrapper(game_name, pad_channels=pad_channels, seed=seed)


# ═══════════════════════════════════════════════════════════════════════
# Dispatch helpers
# ═══════════════════════════════════════════════════════════════════════


def make_env(game_name, cfg, seed=None, pad_channels=None):
    """Create the right environment based on cfg.USE_MINATAR."""
    if cfg.USE_MINATAR:
        return make_minatar_env(game_name, cfg, seed=seed, pad_channels=pad_channels)
    return make_atari_env(game_name, cfg, seed=seed)


def get_action_space_size(game_name, cfg):
    """Return the number of discrete actions for a given game."""
    if cfg.USE_MINATAR:
        from minatar import Environment
        env = Environment(game_name)
        return env.num_actions()
    env = make_atari_env(game_name, cfg)
    n = env.action_space.n
    env.close()
    return n


def get_max_action_space(game_names, cfg):
    """Return the maximum action space across all games in the sequence."""
    max_n = 0
    for name in game_names:
        n = get_action_space_size(name, cfg)
        max_n = max(max_n, n)
    return max_n
