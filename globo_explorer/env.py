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

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import MultiDiscrete

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENV_DIR = os.path.join(HERE, "craftium_globo_env")
# Vendored copy of the globo game lives inside this project, so globo_explorer is
# self-contained and does not depend on the repo's top-level /globo subtree.
DEFAULT_GLOBO_SRC = os.path.join(HERE, "globo_game")

# Discrete action set for exploration (DiscreteActionWrapper adds a no-op -> n+1).
DEFAULT_ACTIONS = [
    "forward", "backward", "left", "right", "jump",
    "mouse x+", "mouse x-", "mouse y+", "mouse y-",
]

# Factored ("branching") action scheme: independent, mutually-exclusive groups
# applied concurrently each step. Each option is a dict of engine effects:
#   movement key -> 1 (pressed); "mouse_x"/"mouse_y" -> +/-1 (scaled by mouse_mov).
FACTORED_GROUPS = [
    ("look_h", [{"mouse_x": -1}, {}, {"mouse_x": +1}]),   # left  / noop / right
    ("look_v", [{"mouse_y": +1}, {}, {"mouse_y": -1}]),   # up    / noop / down
    ("jump",   [{}, {"jump": 1}]),                         # noop  / jump
    ("move",   [{"forward": 1}, {}, {"backward": 1}]),     # fwd   / noop / back
    ("strafe", [{"left": 1}, {}, {"right": 1}]),           # left  / noop / right
]


class FactoredActionWrapper(gym.ActionWrapper):
    """Maps a MultiDiscrete action (one index per group) to craftium's native
    Dict action, so the agent can press several mutually-compatible controls at
    once (move + strafe + jump + look-h + look-v). Action space = MultiDiscrete
    of the per-group option counts, e.g. [3,3,2,3,3]."""

    def __init__(self, env, groups=FACTORED_GROUPS, mouse_mov: float = 0.5):
        super().__init__(env)
        self.groups = groups
        self.mouse_mov = mouse_mov
        self.group_names = [n for n, _ in groups]
        self.action_space = MultiDiscrete([len(opts) for _, opts in groups])

    def action(self, action):
        keys, mx, my = {}, 0.0, 0.0
        for (_name, opts), idx in zip(self.groups, action):
            for k, v in opts[int(idx)].items():
                if k == "mouse_x":
                    mx += v * self.mouse_mov
                elif k == "mouse_y":
                    my += v * self.mouse_mov
                else:
                    keys[k] = v
        keys["mouse"] = [mx, my]
        return keys

_CRAFTIUM_ENV_INIT_LUA = """\
-- Craftium RL control mod for globo. Frames are captured by the engine.
-- Task reward is computed externally (RECODE over images), so in-game reward
-- stays 0; we only signal termination on player death (unless invincible).
--
-- When `craftium_invincible` is set, the player cannot die: globo applies
-- starvation/thirst/temperature damage via direct player:set_hp(), which
-- bypasses the `immortal` armor group, so we top HP back up and pin the
-- exile-style survival stats every server step. This enables one long episode.
-- weather_seed: globo's `climate` mod picks its weather via math.random at load
-- time, and the engine seeds Lua's RNG from wall-clock time, so weather/clouds
-- differ every life. Reseeding here (before climate loads -- climate is made to
-- optionally depend on this mod so we load first) makes weather reproducible and
-- selectable: a given craftium_weather_seed always yields the same weather.
local _wseed = tonumber(minetest.settings:get("craftium_weather_seed"))
if _wseed ~= nil then
    math.randomseed(_wseed)
end

local invincible = minetest.settings:get_bool("craftium_invincible", false)
-- freeze_daylight: keep the world permanently at midday (no day/night cycle), so a
-- long episode's observations stay legible (night turns the view black).
local freeze_daylight = minetest.settings:get_bool("craftium_freeze_daylight", false)

minetest.register_on_dieplayer(function(player, reason)
    if (not invincible) and set_termination ~= nil then set_termination() end
    return false
end)

minetest.register_on_joinplayer(function(player, _last_login)
    minetest.set_timeofday(0.5)  -- midday: stable lighting / vision
    -- Deterministic initial facing so a fixed seed+spawn gives the same first view.
    player:set_look_horizontal(0.0)
    player:set_look_vertical(0.0)
    player:hud_set_flags({
        hotbar = false, crosshair = false, healthbar = false,
        chat = false, minimap = false, breathbar = false, wielditem = false,
    })
    if invincible then player:set_armor_groups({immortal = 1}) end
end)

if invincible or freeze_daylight then
    local acc = 0.0
    minetest.register_globalstep(function(dtime)
        if freeze_daylight then
            minetest.set_timeofday(0.5)  -- re-assert midday every step (cheap)
        end
        if not invincible then return end
        acc = acc + dtime
        if acc < 0.2 then return end  -- a few times per second is enough
        acc = 0.0
        for _, player in ipairs(minetest.get_connected_players()) do
            local props = player:get_properties()
            player:set_hp(props.hp_max or 20)
            player:set_armor_groups({immortal = 1})
            local meta = player:get_meta()
            -- pin exile-style survival stats to safe values (hunger/thirst rise = bad)
            meta:set_float("hunger", 0)
            meta:set_float("thirst", 0)
            local emax = tonumber(meta:get_string("energy_max"))
            if emax then meta:set_float("energy", emax) end
            meta:set_float("temp_internal", 37)
        end
    end)
end

minetest.register_on_mods_loaded(function()
    if _wseed ~= nil and climate ~= nil and climate.active_weather ~= nil then
        minetest.log("action", "[craftium_env] weather_seed=" .. _wseed ..
            " active_weather=" .. tostring(climate.active_weather.name))
    end
end)
"""

_CRAFTIUM_ENV_MOD_CONF = (
    "name = craftium_env\n"
    "description = Craftium RL hooks for globo (obs+reset; reward via RECODE).\n"
)

def _world_mt(invincible: bool) -> str:
    return (
        f"enable_damage = {'false' if invincible else 'true'}\n"
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
                   force: bool = False, invincible: bool = False) -> str:
    """Build/refresh the craftium env_dir for globo.

    The globo game is copied once; the small control mod + world.mt are always
    (re)written so template/option changes (e.g. ``invincible``) take effect.
    """
    game_dst = os.path.join(env_dir, "games", "globo")
    have_game = os.path.isdir(game_dst)
    if force and os.path.isdir(env_dir):
        shutil.rmtree(env_dir)
        have_game = False
    os.makedirs(os.path.join(env_dir, "games"), exist_ok=True)
    os.makedirs(os.path.join(env_dir, "mods", "craftium_env"), exist_ok=True)
    os.makedirs(os.path.join(env_dir, "worlds", "world"), exist_ok=True)

    if not have_game:
        if not os.path.isdir(globo_src):
            raise FileNotFoundError(
                f"globo game source not found at {globo_src}. Pass globo_src=... "
                "(the directory containing globo's game.conf)."
            )
        shutil.copytree(globo_src, game_dst)
        for cruft in (".git", ".github", "rattler-build-recipe"):
            p = os.path.join(game_dst, cruft)
            if os.path.isdir(p):
                shutil.rmtree(p)

    # Make globo's `climate` mod load AFTER craftium_env, so our RNG reseed (for
    # weather_seed) runs before climate picks its weather. Idempotent.
    climate_conf = os.path.join(game_dst, "mods", "climate", "mod.conf")
    if os.path.isfile(climate_conf):
        with open(climate_conf) as f:
            txt = f.read()
        if "craftium_env" not in txt:
            with open(climate_conf, "a") as f:
                f.write(("" if txt.endswith("\n") else "\n") + "optional_depends = craftium_env\n")

    with open(os.path.join(env_dir, "mods", "craftium_env", "init.lua"), "w") as f:
        f.write(_CRAFTIUM_ENV_INIT_LUA)
    with open(os.path.join(env_dir, "mods", "craftium_env", "mod.conf"), "w") as f:
        f.write(_CRAFTIUM_ENV_MOD_CONF)
    with open(os.path.join(env_dir, "worlds", "world", "world.mt"), "w") as f:
        f.write(_world_mt(invincible))
    return env_dir


def make_globo_env(
    obs_width: int = 64,
    obs_height: int = 32,
    max_timesteps: int = 1000,
    init_frames: int = 200,
    action_mode: str = "discrete",
    actions: list[str] | None = None,
    mouse_mov: float = 0.5,
    seed: int | None = None,
    env_dir: str = DEFAULT_ENV_DIR,
    run_dir_prefix: str | None = None,
    pmul: int = 20,
    invincible: bool = False,
    freeze_daylight: bool = False,
    spawn: tuple[float, float, float] | str | None = None,
    weather_seed: int | None = None,
    extra_minetest_conf: dict | None = None,
):
    """Construct a headless, discrete-action globo env. Returns a gymnasium env.

    If ``invincible`` is True the player cannot die (HP/stats pinned by the
    control mod), so a single long episode is possible. If ``freeze_daylight``
    is True the world stays permanently at midday (no day/night cycle), so
    observations remain legible over long runs. ``max_timesteps=None`` disables
    truncation (episode ends only on death). ``seed`` fixes the world+spawn, so
    every restart returns to the identical world and location.
    """
    from craftium import CraftiumEnv, DiscreteActionWrapper

    if action_mode not in ("discrete", "factored"):
        raise ValueError(f"action_mode must be 'discrete' or 'factored', got {action_mode!r}")

    ensure_env_dir(env_dir, invincible=invincible)
    if run_dir_prefix is None:
        run_dir_prefix = os.path.join(HERE, "runs")
    os.makedirs(run_dir_prefix, exist_ok=True)

    minetest_conf = {
        "node_highlighting": "none",   # remove the central selection box from the obs
        "show_entity_selectionbox": False,
        "enable_fog": False,
        "craftium_invincible": "true" if invincible else "false",
        "craftium_freeze_daylight": "true" if freeze_daylight else "false",
    }
    if freeze_daylight:
        minetest_conf["time_speed"] = 0      # stop the day/night clock engine-side
        minetest_conf["time_of_day"] = 6000  # start at midday (0.5 * 24000)
    if spawn is not None:
        # Force an identical spawn location every (re)start. globo's respawn
        # handler returns false, so the engine honors static_spawnpoint; without
        # it the engine's spawn search varies even at a fixed map seed.
        if not isinstance(spawn, str):
            spawn = f"{spawn[0]},{spawn[1]},{spawn[2]}"
        minetest_conf["static_spawnpoint"] = spawn
    if weather_seed is not None:
        minetest_conf["craftium_weather_seed"] = str(int(weather_seed))
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
    if action_mode == "factored":
        return FactoredActionWrapper(base, groups=FACTORED_GROUPS, mouse_mov=mouse_mov)
    return DiscreteActionWrapper(base, actions=actions or DEFAULT_ACTIONS, mouse_mov=mouse_mov)


def obs_to_chw(obs: np.ndarray, device=None) -> torch.Tensor:
    """craftium (H, W, 3) uint8 -> torch (3, H, W) float in [0,1]."""
    t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device) / 255.0
    return t.permute(2, 0, 1).contiguous()
