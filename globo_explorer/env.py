"""Globo gym environment via craftium (headless, first-person RGB).

Globo ships only as a Minetest *game* (Lua mods). We run it inside craftium's
Luanti 5.12.0 build, which provides a gymnasium interface with programmatic
actions and RGB frame capture. This module builds the craftium ``env_dir`` for
globo and returns a discrete-action, headless env.

Key facts (see also the project README):
  * craftium returns observations as (H, W, 3) uint8.
  * Minetest actions are fundamentally discrete keys + mouse -> DQN.
  * sync_mode MUST be False (sync_mode=True crashes: "Failed to open semaphore A").
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENV_DIR = os.path.join(HERE, "craftium_globo_env")
DEFAULT_GLOBO_SRC = os.path.normpath(os.path.join(HERE, "..", "globo"))

# Discrete action set for exploration (DiscreteActionWrapper adds a no-op -> n+1).
DEFAULT_ACTIONS = [
    "forward", "backward", "left", "right", "jump",
    "mouse x+", "mouse x-", "mouse y+", "mouse y-",
]

_CRAFTIUM_ENV_INIT_LUA = """\
-- Craftium RL control mod for globo. Frames are captured by the engine.
-- Task reward is computed externally (RECODE over images), so in-game reward
-- stays 0; we only signal termination on player death.
minetest.register_on_dieplayer(function(player, reason)
    if set_termination ~= nil then set_termination() end
end)

minetest.register_on_joinplayer(function(player, _last_login)
    minetest.set_timeofday(0.5)  -- midday: stable lighting / vision
    player:hud_set_flags({
        hotbar = false, crosshair = false, healthbar = false,
        chat = false, minimap = false, breathbar = false, wielditem = false,
    })
end)
"""

_CRAFTIUM_ENV_MOD_CONF = (
    "name = craftium_env\n"
    "description = Craftium RL hooks for globo (obs+reset; reward via RECODE).\n"
)

_WORLD_MT = (
    "enable_damage = true\n"
    "creative_mode = false\n"
    "mod_storage_backend = sqlite3\n"
    "auth_backend = sqlite3\n"
    "player_backend = sqlite3\n"
    "backend = sqlite3\n"
    "gameid = globo\n"
    "world_name = world\n"
    "server_announce = false\n"
    "load_mod_craftium_env = mods/craftium_env\n"
)


def ensure_env_dir(env_dir: str = DEFAULT_ENV_DIR, globo_src: str = DEFAULT_GLOBO_SRC,
                   force: bool = False) -> str:
    """Build the craftium env_dir for globo if it does not already exist."""
    if os.path.isdir(os.path.join(env_dir, "games", "globo")) and not force:
        return env_dir
    if not os.path.isdir(globo_src):
        raise FileNotFoundError(
            f"globo game source not found at {globo_src}. Pass globo_src=... "
            "(the directory containing globo's game.conf)."
        )
    if force and os.path.isdir(env_dir):
        shutil.rmtree(env_dir)
    os.makedirs(os.path.join(env_dir, "games"), exist_ok=True)
    os.makedirs(os.path.join(env_dir, "mods", "craftium_env"), exist_ok=True)
    os.makedirs(os.path.join(env_dir, "worlds", "world"), exist_ok=True)

    game_dst = os.path.join(env_dir, "games", "globo")
    if not os.path.isdir(game_dst):
        shutil.copytree(globo_src, game_dst)
    for cruft in (".git", ".github", "rattler-build-recipe"):
        p = os.path.join(game_dst, cruft)
        if os.path.isdir(p):
            shutil.rmtree(p)

    with open(os.path.join(env_dir, "mods", "craftium_env", "init.lua"), "w") as f:
        f.write(_CRAFTIUM_ENV_INIT_LUA)
    with open(os.path.join(env_dir, "mods", "craftium_env", "mod.conf"), "w") as f:
        f.write(_CRAFTIUM_ENV_MOD_CONF)
    with open(os.path.join(env_dir, "worlds", "world", "world.mt"), "w") as f:
        f.write(_WORLD_MT)
    return env_dir


def make_globo_env(
    obs_width: int = 64,
    obs_height: int = 32,
    max_timesteps: int = 1000,
    init_frames: int = 200,
    actions: list[str] | None = None,
    mouse_mov: float = 0.5,
    seed: int | None = None,
    env_dir: str = DEFAULT_ENV_DIR,
    run_dir_prefix: str | None = None,
    pmul: int = 20,
    extra_minetest_conf: dict | None = None,
):
    """Construct a headless, discrete-action globo env. Returns a gymnasium env."""
    from craftium import CraftiumEnv, DiscreteActionWrapper

    ensure_env_dir(env_dir)
    if run_dir_prefix is None:
        run_dir_prefix = os.path.join(HERE, "runs")
    os.makedirs(run_dir_prefix, exist_ok=True)

    minetest_conf = {
        "node_highlighting": "none",   # remove the central selection box from the obs
        "show_entity_selectionbox": False,
        "enable_fog": False,
    }
    if extra_minetest_conf:
        minetest_conf.update(extra_minetest_conf)

    base = CraftiumEnv(
        env_dir=os.path.abspath(env_dir),
        game_id="globo",
        obs_width=obs_width,
        obs_height=obs_height,
        init_frames=init_frames,
        max_timesteps=max_timesteps,
        sync_mode=False,            # CRITICAL: True crashes ("semaphore A")
        seed=seed,
        pmul=pmul,
        run_dir_prefix=os.path.abspath(run_dir_prefix),
        mt_listen_timeout=120_000,
        minetest_conf=minetest_conf,
    )
    return DiscreteActionWrapper(base, actions=actions or DEFAULT_ACTIONS, mouse_mov=mouse_mov)


def obs_to_chw(obs: np.ndarray, device=None) -> torch.Tensor:
    """craftium (H, W, 3) uint8 -> torch (3, H, W) float in [0,1]."""
    t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device) / 255.0
    return t.permute(2, 0, 1).contiguous()
