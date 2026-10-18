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
from gymnasium.spaces import MultiDiscrete

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENV_DIR = os.path.join(HERE, "craftium_globo_env")
# Vendored copy of the globo game lives inside this project, so globo_explorer is
# self-contained and does not depend on the repo's top-level /globo subtree.
DEFAULT_GLOBO_SRC = os.path.join(HERE, "globo_game")

# Reference obs width for normalizing turn magnitude (the agent's view width), so
# mouse_mov maps to a fixed yaw-per-step independent of the actual resolution.
REF_OBS_WIDTH = 64

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
-- clear_sky: visually disable weather. globo's `climate` mod runs a Markov chain over
-- ~20 weather states, each re-applying moving clouds / sky / sun / precipitation -- a
-- constantly-changing up-view that the explorer fixates on (it reads as novelty). This
-- forces a static, cloudless, featureless sky every step and no-ops the precipitation
-- particle spawners, WITHOUT pinning the weather state, so climate's temperature/damage
-- (survival) dynamics are left intact -- only the rendering is suppressed.
local clear_sky = minetest.settings:get_bool("craftium_clear_sky", false)

-- A fixed flat-blue dome: no clouds, no sun/moon/stars, uniform colour at all times of day.
local _flat = "#8cbafa"
local function apply_clear_sky(player)
    player:set_clouds({density = 0})
    player:set_sky({
        type = "regular",
        clouds = false,
        sky_color = {
            day_sky = _flat, day_horizon = _flat, dawn_sky = _flat, dawn_horizon = _flat,
            night_sky = _flat, night_horizon = _flat, indoors = _flat,
        },
    })
    player:set_sun({visible = false, sunrise_visible = false})
    player:set_moon({visible = false})
    player:set_stars({visible = false})
end

minetest.register_on_dieplayer(function(player, reason)
    if (not invincible) and set_termination ~= nil then set_termination() end
    return false
end)

minetest.register_on_joinplayer(function(player, _last_login)
    minetest.set_timeofday(0.5)  -- midday: stable lighting / vision
    -- Deterministic initial facing so a fixed seed+spawn gives the same first view.
    player:set_look_horizontal(0.0)
    player:set_look_vertical(0.0)
    -- Revoke fly/fast/noclip (singleplayer gets them by default) so the agent/human
    -- can't accidentally toggle fly/fast and launch into the sky.
    local _name = player:get_player_name()
    local _privs = minetest.get_player_privs(_name)
    _privs.fly = nil; _privs.fast = nil; _privs.noclip = nil
    minetest.set_player_privs(_name, _privs)
    player:hud_set_flags({
        hotbar = false, crosshair = false, healthbar = false,
        chat = false, minimap = false, breathbar = false, wielditem = false,
    })
    if invincible then player:set_armor_groups({immortal = 1}) end
    if clear_sky then apply_clear_sky(player) end
end)

if invincible or freeze_daylight or clear_sky then
    local acc = 0.0
    local acc_sky = 0.0
    minetest.register_globalstep(function(dtime)
        if freeze_daylight then
            minetest.set_timeofday(0.5)  -- re-assert midday every step (cheap)
        end
        if clear_sky then
            -- re-assert a few times/sec so we override climate whenever it changes weather
            acc_sky = acc_sky + dtime
            if acc_sky >= 0.2 then
                acc_sky = 0.0
                for _, player in ipairs(minetest.get_connected_players()) do
                    apply_clear_sky(player)
                end
            end
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
    -- Suppress precipitation visuals without changing the weather state: no-op the
    -- particle spawners (climate keeps its active_weather, so temperature/damage are unchanged).
    if clear_sky and climate ~= nil then
        if climate.add_particle ~= nil then climate.add_particle = function() end end
        if climate.add_blizzard_particle ~= nil then climate.add_blizzard_particle = function() end end
        minetest.log("action", "[craftium_env] clear_sky on: clouds/sky/sun/particles suppressed")
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
    # Refresh this narrow crash fix in older cached templates, at life boundaries.
    animal_api = os.path.join("mods", "animals", "api.lua")
    if have_game:
        with open(os.path.join(game_dst, animal_api)) as f:
            cached_api = f.read()
        if 'tgtobj:get_luaentity().height * 0.6' in cached_api:
            shutil.copyfile(os.path.join(globo_src, animal_api), os.path.join(game_dst, animal_api))
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
    invincible: bool = False,
    freeze_daylight: bool = False,
    clear_sky: bool = False,
    spawn: tuple[float, float, float] | str | None = None,
    weather_seed: int | None = None,
    # explicit, predictable physics (replaces craftium's opaque pmul speed inflation)
    frameskip: int = 1,            # engine frames advanced per env step (more game-time/step)
    move_speed: float = 8.0,       # movement_speed_walk (fixed velocity)
    move_accel: float = 500.0,     # movement_acceleration_default/_air (high -> snappy start/stop)
    climb_speed: float = 3.0,      # movement_speed_climb (small -> no climbable-plant launch)
    jump_speed: float = 6.5,       # movement_speed_jump
    gravity: float = 9.81,         # movement_gravity (normal)
    extra_minetest_conf: dict | None = None,
):
    """Construct a headless, discrete-action globo env. Returns a gymnasium env.

    If ``invincible`` is True the player cannot die (HP/stats pinned by the
    control mod), so a single long episode is possible. If ``freeze_daylight``
    is True the world stays permanently at midday (no day/night cycle), so
    observations remain legible over long runs. If ``clear_sky`` is True, weather
    is visually disabled (static cloudless sky, no sun/precipitation) while the
    climate state still drives temperature/damage. ``max_timesteps=None`` disables
    truncation (episode ends only on death). ``seed`` fixes the world+spawn, so
    every restart returns to the identical world and location.
    """
    from craftium import CraftiumEnv, DiscreteActionWrapper

    if action_mode not in ("discrete", "factored", "raw"):
        raise ValueError(f"action_mode must be 'discrete', 'factored', or 'raw', got {action_mode!r}")

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
        "craftium_clear_sky": "true" if clear_sky else "false",
        # explicit physics: high accel -> near-instant to a fixed speed + fast stop;
        # small climb speed -> climbable plants no longer launch you; normal gravity.
        "movement_speed_walk": move_speed,
        "movement_acceleration_default": move_accel,
        "movement_acceleration_air": move_accel,
        "movement_speed_climb": climb_speed,
        "movement_speed_jump": jump_speed,
        "movement_gravity": gravity,
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
        frameskip=frameskip,        # advance `frameskip` engine frames per env step
        pmul=1,                     # neutralize craftium's speed inflation; physics set above
        run_dir_prefix=os.path.abspath(run_dir_prefix),
        mt_listen_timeout=120_000,
        minetest_conf=minetest_conf,
    )
    # The engine turns by int(mouse * obs_width//2), so turn-per-step scales with
    # resolution. Normalize to a 64px reference so a given mouse_mov yields the same
    # yaw regardless of obs size (keeps the agent's turning fixed/predictable and
    # makes higher-res human play turn identically to the agent).
    mm = mouse_mov * (REF_OBS_WIDTH / obs_width)
    if action_mode == "raw":
        return base   # unwrapped: step() takes craftium's native Dict action (keys + mouse)
    if action_mode == "factored":
        return FactoredActionWrapper(base, groups=FACTORED_GROUPS, mouse_mov=mm)
    return DiscreteActionWrapper(base, actions=actions or DEFAULT_ACTIONS, mouse_mov=mm)


def obs_to_chw(obs: np.ndarray, device=None) -> torch.Tensor:
    """craftium (H, W, 3) uint8 -> torch (3, H, W) float in [0,1]."""
    import torch
    t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device) / 255.0
    return t.permute(2, 0, 1).contiguous()


def resolve_render_size(obs_w: int, obs_h: int, render_w=None, render_h=None) -> tuple[int, int]:
    """Engine render (capture) size. Falls back to the obs size when unset (or 0), so the
    default render==obs case needs no extra config."""
    return (int(obs_w) if not render_w else int(render_w),
            int(obs_h) if not render_h else int(render_h))


def downsample_obs(obs: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Downsample a rendered (H, W, 3) uint8 frame to the agent's (out_h, out_w) input.

    Returns the frame UNCHANGED when it already matches (render==obs fast path), so the
    high-res render pipeline costs nothing when render size == obs size. Bilinear (so it
    anti-aliases on downscale). This is the SINGLE shared downsample used by train.py, the
    --agent demo, and the live trainer, so the agent sees an identical input pipeline
    regardless of the render resolution it is being watched at.
    """
    a = np.asarray(obs)
    if a.shape[0] == out_h and a.shape[1] == out_w:
        return a
    from PIL import Image
    return np.asarray(Image.fromarray(a[:, :, :3]).resize((int(out_w), int(out_h)), Image.BILINEAR))
