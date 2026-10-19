"""
config.py - Configuration for JAX-based masked DQN continual learning on MinAtar.

Frozen dataclass mirroring atari/config.py with MinAtar defaults.
Game specs map gymnax environment names to (n_channels, n_actions).
"""

from dataclasses import dataclass

# gymnax MinAtar env name -> (obs_channels, n_actions)
GAME_SPECS = {
    "SpaceInvaders-MinAtar": (6, 4),
    "Breakout-MinAtar": (4, 3),
    "Asterix-MinAtar": (4, 5),
    "Freeway-MinAtar": (7, 3),
    "Seaquest-MinAtar": (10, 6),
}

DEFAULT_GAMES = (
    "SpaceInvaders-MinAtar",
    "Breakout-MinAtar",
    "SpaceInvaders-MinAtar",
    "Breakout-MinAtar",
    "SpaceInvaders-MinAtar",
    "Breakout-MinAtar",
    "SpaceInvaders-MinAtar",
    "Breakout-MinAtar",
    "SpaceInvaders-MinAtar",
    "Breakout-MinAtar",
    "SpaceInvaders-MinAtar",
    "Breakout-MinAtar",
)


@dataclass(frozen=True)
class Config:
    seed: int = 42
    game_sequence: tuple = DEFAULT_GAMES

    # Training
    max_steps_per_game: int = 10000
    warmup_steps: int = 0
    eval_episodes: int = 100
    eval_every_n_games: int = 1
    eval_mask_attempts: int = 8

    # DQN hyperparameters
    batch_size: int = 512
    gamma: float = 0.90
    lr: float = 1e-4
    target_update_freq: int = 10
    train_freq: int = 1  # env steps between training bursts
    gradient_steps: int = 1  # gradient updates per training burst
    grad_clip: float = 1.0

    # Epsilon-greedy schedule
    eps_start: float = 0.1
    eps_end: float = 0.1
    eps_decay_steps: int = 10_000

    # Replay buffer
    replay_capacity: int = 25_000
    min_replay_size: int = 1_000

    # Mask replay (cross-game)
    mask_replay_enabled: bool = True
    mask_replay_corr_low: float = 0.1
    mask_replay_corr_high: float = 0.9
    n_replay_samples: int = 1
    n_sample_per_game: int = 10_000
    mask_replay_capacity: int = 50_000

    # Model / masking
    square_dim: int = 16
    k: int = 32
    fc_layers: int = 2
    dropout: float = 0.0

    # Mask reconfiguration
    reconfig_steps: int = 1
    reconfig_lr: float = 1
    reconfig_sample_size: int = 2048
    discover_iterations: int = 2
    discover_epsilon: float = 0.1

    # Lateral smoothing
    kernel_size: int = 9
    lateral_steps: int = 2
    lateral_polyak: float = 0.0

    # Logging
    log_every_steps: int = 5_000
    loss_ema_smoothing: float = 0.99
    scan_chunk_size: int = 1_000

    @property
    def hidden_size(self) -> int:
        return self.square_dim**2

    @property
    def max_channels(self) -> int:
        return max(ch for ch, _ in GAME_SPECS.values())

    @property
    def max_actions(self) -> int:
        return max(na for _, na in GAME_SPECS.values())

    @property
    def n_games(self) -> int:
        return len(self.game_sequence)

    def game_channels(self, game: str) -> int:
        return GAME_SPECS[game][0]

    def game_actions(self, game: str) -> int:
        return GAME_SPECS[game][1]
