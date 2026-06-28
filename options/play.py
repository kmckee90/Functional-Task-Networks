#!/usr/bin/env python
"""Human play + diagnostic renderer for the survival grid world.

Turn-based: each keypress steps the env exactly once (the same Discrete action
the agent uses), then the terminal is cleared and redrawn. Both the full world
state and the agent's egocentric observation are shown so you can diagnose the
environment from either viewpoint.

Controls:
    W / A / S / D   move up / left / down / right
    space or .      no-op move (still advances time -> hunger/NPCs)
    r               reset
    q / ctrl-c      quit

Usage:
    uv run python play.py
    uv run python play.py --size 24 --npcs 6 --nutrients 8
"""

from __future__ import annotations

import argparse
import sys

from grid_world import SurvivalGridEnv

KEY_TO_ACTION = {"w": 0, "s": 1, "a": 2, "d": 3}
CLEAR = "\033[2J\033[H"  # clear screen + move cursor home


def _read_key() -> str:
    """Read a single keypress without waiting for Enter (POSIX terminal)."""
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _frame(env: SurvivalGridEnv, msg: str) -> str:
    legend = "@ you   N npc   # obstacle   * nutrient   . empty"
    controls = "[WASD] move   [r] reset   [q] quit"
    return "\n".join(
        [CLEAR, "SURVIVAL GRID", legend, "", env.render_full(), "",
         env.render_view(), "", controls, msg]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=20)
    p.add_argument("--npcs", type=int, default=4)
    p.add_argument("--nutrients", type=int, default=5)
    p.add_argument("--obstacle-density", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    env = SurvivalGridEnv(
        # size=args.size,
        # n_npcs=args.npcs,
        # n_nutrients=args.nutrients,
        # obstacle_density=args.obstacle_density,
        render_mode="human",
    )
    env.reset(seed=args.seed)
    msg = ""

    while True:
        print(_frame(env, msg), flush=True)
        key = _read_key().lower()

        if key in ("q", "\x03"):  # q or ctrl-c
            print(CLEAR + "bye")
            return
        if key == "r":
            env.reset(seed=args.seed)
            msg = "reset"
            continue
        if key in KEY_TO_ACTION:
            action = KEY_TO_ACTION[key]
        else:
            msg = f"unknown key {key!r} (use W/A/S/D, r, q)"
            continue

        _, reward, terminated, truncated, info = env.step(action)
        msg = f"reward {reward:+.0f}   nutrients {info['n_nutrients']}"

        if terminated or truncated:
            cause = "DIED" if terminated else "TIMEOUT"
            print(_frame(env, f"{cause} (reward {reward:+.0f}) -- press any key to restart, q to quit"))
            if _read_key().lower() in ("q", "\x03"):
                print(CLEAR + "bye")
                return
            env.reset(seed=args.seed)
            msg = "reset"


if __name__ == "__main__":
    main()
