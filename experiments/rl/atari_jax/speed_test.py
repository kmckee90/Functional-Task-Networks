"""
speed_test.py - Benchmark comparison between PyTorch and JAX masked DQN on MinAtar.

Runs both implementations on the same game with *identical* hyperparameters and
reports wall-clock time, steps/sec, and speedup factor.
"""

import os
import subprocess
import sys
import textwrap
import time

# ── Shared hyperparameters (authoritative source for both runs) ──────
SHARED = dict(
    max_steps=10_000,
    batch_size=512,
    gamma=0.90,
    lr=1e-4,
    target_update_freq=10,
    train_freq=100,
    gradient_steps=100,
    grad_clip=1.0,
    eps_start=1.0,
    eps_end=0.1,
    eps_decay_steps=10_000,
    replay_capacity=25_000,
    min_replay_size=1_000,
    warmup_steps=1_000,
    square_dim=24,
    k=64,
    fc_layers=3,
)

# PyTorch minatar game name -> JAX gymnax env name
PT_TO_JAX = {
    "breakout": "Breakout-MinAtar",
    "space_invaders": "SpaceInvaders-MinAtar",
    "asterix": "Asterix-MinAtar",
    "freeway": "Freeway-MinAtar",
    "seaquest": "Seaquest-MinAtar",
}
JAX_TO_PT = {v: k for k, v in PT_TO_JAX.items()}


def resolve_names(game):
    """Return (pytorch_name, jax_name) for a game string."""
    g = game.lower().replace("-minatar", "")
    for pt_name, jax_name in PT_TO_JAX.items():
        if pt_name.startswith(g) or jax_name.lower().startswith(g):
            return pt_name, jax_name
    raise ValueError(f"Unknown game: {game}")


def print_config():
    print("Shared config:")
    for k, v in SHARED.items():
        print(f"  {k:24s} = {v}")
    print()


def run_pytorch(pt_game, steps):
    """Run PyTorch atari/ with exact shared config via inline script."""
    atari_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "atari")
    S = SHARED

    script = textwrap.dedent(f"""\
        import sys, os, time, copy
        _dir = {repr(os.path.abspath(atari_dir))}
        sys.path.insert(0, _dir)
        os.chdir(_dir)
        import numpy as np, torch
        from config import AtariConfig
        from dqn import Conv2dDQNMinAtar
        from env_utils import make_minatar_env, get_minatar_max_channels
        from replay_buffer import ReplayBuffer, MaskReplayBuffer
        from run_atari import train_game, evaluate_game

        cfg = AtariConfig()
        cfg.enable_minatar()
        # Override to match shared config exactly
        cfg.MAX_STEPS_PER_GAME = {steps}
        cfg.BATCH_SIZE = {S["batch_size"]}
        cfg.GAMMA = {S["gamma"]}
        cfg.LR = {S["lr"]}
        cfg.TARGET_UPDATE_FREQ = {S["target_update_freq"]}
        cfg.GRAD_CLIP = {S["grad_clip"]}
        cfg.EPS_START = {S["eps_start"]}
        cfg.EPS_END = {S["eps_end"]}
        cfg.EPS_DECAY_STEPS = {S["eps_decay_steps"]}
        cfg.REPLAY_CAPACITY = {S["replay_capacity"]}
        cfg.MIN_REPLAY_SIZE = {S["min_replay_size"]}
        cfg.WARMUP_STEPS = {S["warmup_steps"]}
        cfg.SQUARE_DIM = {S["square_dim"]}
        cfg.K = {S["k"]}
        cfg.FC_LAYERS = {S["fc_layers"]}
        cfg.TRAIN_FREQ = {S["train_freq"]}
        cfg.GRADIENT_STEPS = {S["gradient_steps"]}
        cfg.MASK_REPLAY_ENABLED = False

        game = "{pt_game}"
        torch.set_default_device("cuda")
        torch.manual_seed(cfg.SEED)
        np.random.seed(cfg.SEED)

        pad_channels = get_minatar_max_channels([game])
        cfg._pad_channels = pad_channels
        n_actions = make_minatar_env(game, cfg, pad_channels=pad_channels).action_space.n

        model = Conv2dDQNMinAtar(
            max_actions=n_actions, hidden_size=cfg.HIDDEN_SIZE,
            in_channels=pad_channels, **cfg.MODEL_PARS,
        )
        target_model = copy.deepcopy(model); target_model.eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR)
        mask = torch.ones(cfg.HIDDEN_SIZE)
        replay_buf = ReplayBuffer(cfg.REPLAY_CAPACITY)
        mask_replay = MaskReplayBuffer(max_per_game=0)

        outdir = os.path.join(_dir, "images")
        os.makedirs(outdir, exist_ok=True)

        model.train()
        t0 = time.time()
        losses = train_game(model, target_model, optimizer, game, 0, mask, cfg,
                            replay_buf, mask_replay, outdir)
        dt = time.time() - t0
        print(f"RESULT time={{dt:.4f}} episodes={{len(losses)}}")
    """)

    print(f"[PyTorch] Running: {pt_game}, steps={steps}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=atari_dir,
    )
    dt = time.time() - t0

    if result.returncode != 0:
        print(f"[PyTorch] FAILED (exit code {result.returncode})")
        stderr = result.stderr.strip()
        if stderr:
            for line in stderr.split("\n")[-15:]:
                print(f"  {line}")
        return None

    # Parse embedded timing from the subprocess
    for line in result.stdout.strip().split("\n"):
        if line.startswith("RESULT"):
            parts = dict(kv.split("=") for kv in line.split()[1:])
            inner_time = float(parts["time"])
            print(f"[PyTorch] Completed in {inner_time:.1f}s (wall {dt:.1f}s)")
            return inner_time

    # Fallback to wall-clock
    print(f"[PyTorch] Completed in {dt:.1f}s (wall)")
    return dt


def run_jax(jax_game, steps):
    """Run JAX atari_jax/ with exact shared config."""
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="scatter inputs have incompatible types",
        category=FutureWarning,
    )

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax
    from config import Config
    from model import Conv2dDQNMinAtar
    from replay_buffer import init_mask_buffer
    from run import train_game

    S = SHARED
    cfg = Config(
        game_sequence=(jax_game,),
        mask_replay_enabled=True,
        max_steps_per_game=steps,
        batch_size=S["batch_size"],
        gamma=S["gamma"],
        lr=S["lr"],
        target_update_freq=S["target_update_freq"],
        train_freq=S["train_freq"],
        gradient_steps=S["gradient_steps"],
        grad_clip=S["grad_clip"],
        eps_start=S["eps_start"],
        eps_end=S["eps_end"],
        eps_decay_steps=S["eps_decay_steps"],
        replay_capacity=S["replay_capacity"],
        min_replay_size=S["min_replay_size"],
        warmup_steps=S["warmup_steps"],
        square_dim=S["square_dim"],
        k=S["k"],
        fc_layers=S["fc_layers"],
    )

    key = jax.random.PRNGKey(cfg.seed)
    key, model_key = jax.random.split(key)
    model = Conv2dDQNMinAtar(
        max_actions=cfg.max_actions,
        hidden_size=cfg.hidden_size,
        in_channels=cfg.max_channels,
        n_layers=cfg.fc_layers,
        key=model_key,
    )
    target_model = model
    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    mask = jnp.ones(cfg.hidden_size)
    obs_shape = (cfg.max_channels, 10, 10)
    mask_replay = init_mask_buffer(cfg.mask_replay_capacity, obs_shape, cfg.hidden_size)

    print(f"[JAX] Running: {jax_game}, steps={steps}")

    t0 = time.time()
    model, target_model, opt_state, mask_replay, key, ep_rewards = train_game(
        model,
        target_model,
        opt_state,
        mask,
        jax_game,
        0,
        cfg,
        mask_replay,
        key,
    )
    jax.tree.map(lambda x: x.block_until_ready(), eqx.filter(model, eqx.is_array))
    dt = time.time() - t0

    print(f"[JAX] Completed in {dt:.1f}s ({len(ep_rewards)} episodes)")
    return dt


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Speed test: PyTorch vs JAX masked DQN"
    )
    parser.add_argument(
        "--game", type=str, default="breakout", help="Game name (default: breakout)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override step count (default: use SHARED config)",
    )
    parser.add_argument(
        "--train-freq",
        type=int,
        default=None,
        help="Env steps between training bursts (default: use SHARED config)",
    )
    parser.add_argument(
        "--gradient-steps",
        type=int,
        default=None,
        help="Gradient updates per training burst (default: use SHARED config)",
    )
    parser.add_argument(
        "--skip-pytorch", action="store_true", help="Skip PyTorch run (JAX only)"
    )
    parser.add_argument(
        "--skip-jax", action="store_true", help="Skip JAX run (PyTorch only)"
    )
    args = parser.parse_args()

    # Apply CLI overrides to SHARED
    if args.train_freq is not None:
        SHARED["train_freq"] = args.train_freq
    if args.gradient_steps is not None:
        SHARED["gradient_steps"] = args.gradient_steps

    steps = args.steps or SHARED["max_steps"]
    pt_game, jax_game = resolve_names(args.game)

    print("=" * 60)
    print("Speed Test: PyTorch vs JAX Masked DQN on MinAtar")
    print("=" * 60)
    print(f"Game: {pt_game} / {jax_game}")
    print(f"Steps: {steps:,}")
    print()
    print_config()

    jax_time = None
    pytorch_time = None

    if not args.skip_jax:
        print("-" * 40)
        jax_time = run_jax(jax_game, steps)
        print()

    if not args.skip_pytorch:
        print("-" * 40)
        pytorch_time = run_pytorch(pt_game, steps)
        print()

    # Summary
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Steps: {steps:,}  |  Game: {args.game}")
    print()

    if jax_time is not None:
        print(f"  JAX:     {jax_time:7.1f}s  ({steps / jax_time:,.0f} steps/sec)")
    if pytorch_time is not None:
        print(
            f"  PyTorch: {pytorch_time:7.1f}s  ({steps / pytorch_time:,.0f} steps/sec)"
        )
    if jax_time and pytorch_time:
        speedup = pytorch_time / jax_time
        print(
            f"\n  Speedup: {speedup:.2f}x {'(JAX faster)' if speedup > 1 else '(PyTorch faster)'}"
        )


if __name__ == "__main__":
    main()
