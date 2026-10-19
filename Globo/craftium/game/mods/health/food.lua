-- food.lua
-- handles the intake of food and drink
-- Modding info:
--[[
 To add new foods, define your node(s), then pass a table with food info to
exile_add_food(table). Its on_use will be set automatically.
 Make sure all your nodes have been defined BEFORE you send any tables!
 exile_add_food() will override a node's on_use.

 For cookable things, define the node, and a name_cooked/name_burned version,
then pass the cooking data to exile_add_bake(table). Don't forget to add the
cooked version to the food table.
 If the burned version is not also added to foods, it will be inedible.
 exile_add_bake() will override a node's on_construct and on_timer.

 If a food can only be cooked in a pot, don't define a name_cooked node,
but add it to the food table anyway. The cooking pot will make a soup using
the food table's data, but the food will not be able to bake in an oven or
over a fire.
#TODO: Test this ^^ after the cooking pot supports both tables
]] --
-- Internationalization
local S = HEALTH.S

dofile(minetest.get_modpath('health') .. '/data_food.lua')

-- Declare globals
HEALTH.food_harm_table = {} -- food_harm_table LATER: for now no harms

local function do_food_harm(user, nodename)
    if not HEALTH.food_harm_table[nodename] then
        return
    end
    local fht = HEALTH.food_harm_table[nodename]
    for i = 1, #fht do
        if math.random() < fht[i][2] then
            HEALTH.add_new_effect(user, {fht[i][1], fht[i][3]})
        end
    end
end

local function get_eat2x(player)
    return (player:get_meta():get_string("conf_eat2x") or
               minetest.settings:get("exile_eat_doubleclick") or "false") == "true"
end

local function set_eat2x(player, eat2x)
    player:get_meta():set_string("conf_eat2x", eat2x and "true" or "false")
end

local function toggle_eat2x(player)
    local eat2x = not get_eat2x(player)
    set_eat2x(player, eat2x)
    return eat2x
end

minetest.register_chatcommand(
    "eat2x", {
        params = "true | false",
        description = "Toggles double-click to eat",
        func = function(name, param)
            if param and param ~= "" then
                local wlist = "/eat2x:\n" .. "Toggle double-click to eat"
                return false, wlist
            end

            local player = minetest.get_player_by_name(name)
            local eat2x = toggle_eat2x(player)
            globo.chat_send_player(name, "Double-Click to eat: " .. tostring(eat2x))
        end,
    }
)

local function eat_ok(itemstack, user, pointed_thing)
    local pt_pos = minetest.get_pointed_thing_position(pointed_thing)
    local pname = user:get_player_name()
    local player = minetest.get_player_by_name(pname) -- is this the same as user?
    local eat2x = get_eat2x(player)
    if (not eat2x) or minimal.click_count_ready(pname, "eat2x", pt_pos, 2, 2) then
        return true
    else
        globo.chat_send_player(pname, S("double click to eat"))
    end
end

-- function exile_eatdrink_playermade(itemstack, user, pointed_thing)
--     local imeta = itemstack:get_meta()
--     local pname = user:get_player_name()
--     local t = minetest.deserialize(imeta:get_string("eat_value"))
--     if t == nil then
--         minetest.log("warning", pname .. " ate an invalid food of type " .. itemstack:get_name())
--         return
--     end
--     if eat_ok(itemstack, user, pointed_thing) then
--         return HEALTH.use_item(itemstack, user, t[1], t[2], t[3], t[4], t[5], t[6])
--     end
-- end

function HEALTH.exile_eatdrink(itemstack, user, pointed_thing)
    local name = itemstack:get_name()

    if minetest.registered_aliases[name] then
        name = minetest.registered_aliases[name]
    end
    if not HEALTH.food_table[name] then
        globo.chat_send_player(user:get_player_name(), S("This is inedible."))
        return
    end
    if eat_ok(itemstack, user, pointed_thing) then
        do_food_harm(user, name)
        local t = HEALTH.food_table[name]
        return HEALTH.use_item(itemstack, user, t[1], t[2], t[3], t[4], t[5], t[6])
    else
        return
    end
end

-- Overrides for edible and bakable nodes
local eat_redef = {
    on_use = function(itemstack, user, pointed_thing)
        return HEALTH.exile_eatdrink(itemstack, user, pointed_thing)
    end,
}

-- function exile_add_food(table)
--     -- Add new foods, mod must send a table in the food_data.lua format
--     for k, v in pairs(table) do
--         HEALTH.food_table[k] = v
--         if minetest.registered_nodes[k] then minetest.override_item(k, eat_redef) end
--     end
-- end

-- function exile_add_harm(table)
--     -- Add new food harm, mod must send a table in the food_data.lua format
--     for k, v in pairs(table) do HEALTH.food_harm_table[k] = v end
-- end

function HEALTH.exile_add_food_hooks(name)
    if (minetest.get_item_group(name, 'edible') > 0) or HEALTH.food_table[name] then
        minetest.override_item(name, eat_redef)
    end
end

-- Add food hooks to all nodes in edible group
minetest.register_on_mods_loaded(
    function()
        for name, _ in pairs(minetest.registered_nodes) do
            if minetest.get_item_group(name, 'edible') > 0 then
                HEALTH.exile_add_food_hooks(name)
            end
        end
    end
)

-- Finalized table list
-- Outputs a compilned list of all added foods to the minetest log, info level
minetest.after(
    1, function()
        minetest.log("info", "Finalized list of food_table entries:")
        for k, v in pairs(HEALTH.food_table) do
            if minetest.registered_nodes[k] then
                minetest.log("info", k)
            end
        end
        minetest.log("info", "-------")
        minetest.log("info", "Finalized list of bake_table entries:")
        for k, v in pairs(HEALTH.bake_table) do
            if not minetest.registered_nodes[k] then
                minetest.log("info", "Bake table contains an undefined node: " .. k)
            else
                if minetest.registered_nodes[k .. "_cooked"] then
                    minetest.log("info", k)
                else
                    minetest.log(
                        "info", "undefined node (cooking pot only entry): " .. k .. "_cooked"
                    )
                end
            end
        end
        minetest.log("info", "-------")
    end
)
