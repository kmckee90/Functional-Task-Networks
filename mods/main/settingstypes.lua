-- For settings api, see: https://api.minetest.net/class-reference/#settings.
-- Setting defaults from settingstypes.txt do not apply, so we fill them in here. See https://github.com/minetest/minetest/issues/11736.
local function is_nan(value)
    return value ~= value
end

local function is_finite(value)
    return value ~= 1 / 0 and value ~= -1 / 0 and not is_nan(value)
end

local function is_number(value)
    return type(value) == 'number'
end

local function assert_finite_number(value)
    assert(is_number(value), "not a number")
    assert(is_finite(value), "not finite")
    return value
end

local function is_bool(value)
    return type(value) == 'boolean'
end

local function assert_bool(value)
    assert(is_bool(value), "not a bool")
    return value
end

local function is_string(value)
    return type(value) == 'string'
end

local function assert_string(value)
    assert(is_string(value), "not a string")
    return value
end

-- Use for floats or ints, minetest Lua has separate types internally but doesn't expose them. See https://www.lua.org/manual/5.1/manual.html#2.2 and https://github.com/Astera-org/minetest/blob/d11359409aafb1111dc0f8058524512382dca458/lib/lua/src/lua.h#L103. 
local function get_number(key, default)
    return assert_finite_number(
        minetest.settings:has(key) and tonumber(minetest.settings:get(key)) or default
    )
end

local function get_bool(key, default)
    return assert_bool(minetest.settings:get_bool(key, default))
end

local function get_string(key, default)
    return assert_string(minetest.settings:has(key) and minetest.settings:get(key) or default)
end

globo.HudSetting = {player = "player", agent = "agent", none = "none"}

local function get_hud(key, default)
    local value = get_string(key, default)
    if value == globo.HudSetting.player then
        return globo.HudSetting.player
    elseif value == globo.HudSetting.agent then
        return globo.HudSetting.agent
    elseif value == globo.HudSetting.none then
        return globo.HudSetting.none
    end
    assert(
        false,
            "expected one of " .. globo.HudSetting.player .. ", " .. globo.HudSetting.agent ..
                ", or " .. globo.HudSetting.none .. " but got " .. value
    )
end

-- See `../../settingstypes.txt` for the definition of these settings.
globo.HUD = get_hud('globo_hud', "player")
globo.LOG_MESSAGES = get_bool('globo_log_messages', true)
globo.ENERGY_RECOVERY_RATE_SLEEP = get_number('globo_energy_recovery_rate_sleep', 10.0)
globo.GAME_LENGTH = get_number('globo_game_length', 3600)
globo.GAME_SPEED = get_number('globo_game_speed', 1.0)
globo.HEAL_RATE = get_number('globo_heal_rate', 0.5)
globo.HEAL_RATE_SLEEP = get_number('globo_heal_rate_sleep', 1.0)
globo.INVENTORY_SIZE = get_number('globo_inventory_size', 4)
globo.PLAYER_MAX_HEALTH = get_number('globo_player_max_health', 1000)
