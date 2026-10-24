--[[
Implements all changes to the players:
 - needing certain nutrients. 
 - The hunger or thirst will increase till the player eats or drinks
]] --
local player_definition = {
    physical = true,
    collisionbox = {-0.3, 0.0, -0.3, 0.3, 1.7, 0.3},
    visual = "mesh",
    visual_size = {x = 1, y = 1},
    mesh = "character.b3d",
    textures = {"character.png"},
    view_range = 10,
    walk_velocity = 1,
    run_velocity = 1.6,
    hp_max = globo.PLAYER_MAX_HEALTH,
}

local get_gametime = std.minetest.get_gametime

minetest.register_entity("main:player", player_definition)

function globo.add_energy(player, amount)
    local meta = player:get_meta()
    local eCur = tonumber(meta:get_string("energy"))
    local eMax = tonumber(meta:get_string("energy_max"))
    eCur = eCur + amount
    if eCur < 0 then
        eCur = 0
    elseif eCur > eMax then
        eCur = eMax
    end
    meta:set_float("energy", eCur)
end

function globo.add_hp(player, amount)
    local new_health = player:get_hp() + amount

    if new_health < 0 then
        new_health = 0
    end

    -- if new_health > player:get_properties().hp_max

    player:set_hp(new_health)
end

function globo.add_nutrient(player, name, value)
    local meta = player:get_meta()
    local nut = meta:get_float(name)
    nut = nut + value
    if nut < 0 then
        nut = 0
    elseif nut > 1000 then
        nut = 1000
    end
    meta:set_float(name, nut)
end

local player_name_to_hud_name_to_hud_id = {}

function globo.allocate_hud_for_player(player)
    local player_name = player:get_player_name()
    assert(player_name_to_hud_name_to_hud_id[player_name] == nil, "hud already allocated")
    player_name_to_hud_name_to_hud_id[player_name] = {}
end

function globo.deallocate_hud_for_player(player)
    local player_name = player:get_player_name()
    player_name_to_hud_name_to_hud_id[player_name] = nil
end

function globo.hud_for_player(player)
    local player_name = player:get_player_name()
    local hud_name_to_hud_id = player_name_to_hud_name_to_hud_id[player_name]
    assert(hud_name_to_hud_id ~= nil, "hud not allocated")
    return hud_name_to_hud_id
end

function globo.hud_add(player, element)
    local hud_name = element.name
    assert(hud_name ~= nil, "hud element must have a name")

    local hud_name_to_hud_id = globo.hud_for_player(player)

    assert(hud_name_to_hud_id[hud_name] == nil, "hud element name already in use")
    hud_name_to_hud_id[hud_name] = player:hud_add(element)
end

local function hud_text(name, offset_y)
    return {
        name = name,
        type = "text",
        text = "",
        number = 0xFFFFFF,
        position = {x = 0, y = 1}, -- bottom left corner of screen
        offset = {x = 10, y = -10 + offset_y}, -- offset in pixels
        alignment = {x = 1, y = -1}, -- anchor at bottom left corner
        scale = {x = 100, y = 100},
    }
end

local function setup_hud_agent(player)
    local hud_names = {
        "hunger_rate", "thirst_rate", "energy_rate", "heal_rate", "score", "air", "health",
        "hunger", "thirst", "energy", "temp_internal", "temp_external",
    }

    local pixels_per_row = 20
    for index, name in ipairs(hud_names) do
        globo.hud_add(player, hud_text(name, (index - #hud_names) * pixels_per_row))
    end
end

local function setup_hud_player(player)
    local hud_names = {"score", "air", "health", "hunger", "thirst", "energy", "temp", "status"}

    local pixels_per_row = 20
    for index, name in ipairs(hud_names) do
        globo.hud_add(player, hud_text(name, (index - #hud_names) * pixels_per_row))
    end
end

local function update_hud(player)
    local player_pos = player:get_pos()
    player_pos.y = player_pos.y + 0.6 -- adjust to body height
    local external_temp = math.floor(climate.get_point_temp(player_pos, true))

    -- Necessary for communication with RL environments.
    local meta = player:get_meta()
    meta:set_int("temperature_external", external_temp)

    local hunger_level = meta:get_int("hunger")
    local thirst_level = meta:get_int("thirst")
    local energy_cur = meta:get_int("energy")
    local energy_max = meta:get_int("energy_max")
    local internal_temp = meta:get_int("temperature")
    local score = meta:get_int("score")
    local hun_rate = meta:get_int("hunger_rate")
    local t_rate = meta:get_int("thirst_rate")
    local r_rate = meta:get_int("recovery_rate")
    local h_rate = meta:get_int("heal_rate")

    local air = 1000
    local health = player:get_hp()

    local hud_ids = globo.hud_for_player(player)
    if globo.HUD == globo.HudSetting.agent then
        player:hud_change(hud_ids.score, "text", "Score: " .. score)
        player:hud_change(hud_ids.air, "text", "Air: " .. air)
        player:hud_change(hud_ids.health, "text", "Health: " .. health)
        player:hud_change(hud_ids.hunger, "text", "Hunger: " .. hunger_level)
        player:hud_change(hud_ids.thirst, "text", "Thirst: " .. thirst_level)
        player:hud_change(hud_ids.energy, "text", "Energy: " .. energy_cur)
        player:hud_change(hud_ids.temp_internal, "text", "Int Temp: " .. internal_temp)
        player:hud_change(hud_ids.temp_external, "text", "Ext Temp: " .. external_temp)
        player:hud_change(hud_ids.heal_rate, "text", "Health Rate: " .. h_rate)
        player:hud_change(hud_ids.hunger_rate, "text", "Hunger Rate: " .. -hun_rate)
        player:hud_change(hud_ids.thirst_rate, "text", "Thirst Rate: " .. -t_rate)
        player:hud_change(hud_ids.energy_rate, "text", "Energy Rate: " .. r_rate)
    elseif globo.HUD == globo.HudSetting.player then
        player:hud_change(hud_ids.score, "text", "Score: " .. score)
        player:hud_change(hud_ids.air, "text", "Air: " .. air)
        player:hud_change(hud_ids.health, "text", "Health: " .. health .. "+" .. h_rate)
        player:hud_change(hud_ids.hunger, "text", "Hunger: " .. hunger_level .. "-" .. -hun_rate)
        player:hud_change(hud_ids.thirst, "text", "Thirst: " .. thirst_level .. "-" .. -t_rate)
        player:hud_change(
            hud_ids.energy, "text", "Energy: " .. energy_cur .. " / " .. energy_max .. "+" .. r_rate
        )
        player:hud_change(hud_ids.temp, "text", "Temp: " .. internal_temp .. " / " .. external_temp)
        player:hud_change(hud_ids.status, "text", HEALTH.effects_list_str(player))
    end
end

local invTime = get_gametime()

local function inventoryEffects(player)
    if invTime < get_gametime() then
        invTime = get_gametime() + 1
        local inv = player:get_inventory()

        for _, stack in ipairs(inv:get_list("main")) do
            local name = stack:get_name()
            if name == "main:ember" then
                globo.add_hp(player, -50)
            end
            if name == "main:pulse_blossom" or name == "main:pusle_blossom_on" then
                globo.add_hp(player, -50)
            end
            if (name == "main:sun_berry" or name == "main:moon_berry") and math.random() < 0.001 then
                -- delete the item
                inv:remove_item("main", name)
            end
        end
    end
end

local function spawn_mob_at_position(pos)
    -- minimal.log("Rand spawning...") 
    -- make sure the node is air
    if minetest.get_node(pos).name ~= "air" then
        minimal.log(
            "Not air: " .. minetest.get_node(pos).name .. " at " .. "(" .. pos.x .. "," .. pos.y ..
                "," .. pos.z .. ")"
        )
        return
    end
    -- make sure the node under is soil
    local under = minetest.get_node({x = pos.x, y = pos.y - 1, z = pos.z})
    local soils_list = {
        "nodes_nature:marshland_soil", "nodes_nature:marshland_soil_wet",
        "nodes_nature:grassland_soil", "nodes_nature:grassland_soil_wet",
        "nodes_nature:highland_soil", "nodes_nature:highland_soil_wet",
        "nodes_nature:duneland_soil", "nodes_nature:duneland_soil_wet",
        "nodes_nature:woodland_soil", "nodes_nature:woodland_soil_wet", "nodes_nature:granite",
        "nodes_nature:basalt", "nodes_nature:silt", "nodes_nature:clay", "nodes_nature:sand",
        "nodes_nature:gravel", "nodes_nature:slope_outer_grassland_soil",
        "nodes_nature:slope_inner_grassland_soil", "nodes_nature:sari",
        "nodes_nature:slope_grassland_soil",
    }
    -- check if node.name is in the list
    if not globo.is_in_list(under.name, soils_list) then
        minimal.log(
            "Not soil: " .. under.name .. " at " .. "(" .. pos.x .. "," .. pos.y .. "," .. pos.z ..
                ")"
        )
        return
    end

    local moblist = {
        {"animals:firepuffin_spawn", .1}, {"animals:darkasthaan_spawn", .1},
        {"animals:wolf_spawn", .1}, {"animals:mongoose_spawn", .1}, {"animals:puma_spawn", .1},
    } -- these are herds "animals:gazelle_spawn","animals:cow_spawn"}

    local mob_index = math.random(1, #moblist)
    if moblist[mob_index][2] > math.random() then
        -- spawn the mob
        minetest.set_node(pos, {name = moblist[mob_index][1]})
        minimal.log(
            "Spawned " .. moblist[mob_index][1] .. " at " .. "(" .. pos.x .. "," .. pos.y .. "," ..
                pos.z .. ")"
        )
        -- else
        -- minimal.log("Not spawning "..moblist[mob_index][1].." at ".."("..pos.x..","..pos.y..","..pos.z..")")
    end
end

local function node_collisions_player(player)
    local player_pos = player:get_pos()
    -- minimal.log("on_step")
    -- Example: Check if the player is standing on a specific node
    local node_pos = {x = player_pos.x, y = player_pos.y - .25, z = player_pos.z}
    local node = minetest.get_node(node_pos)

    if node.name == "main:gumboline" then

        local velocity = player:get_velocity()
        velocity.x = velocity.x * 2
        velocity.y = 5
        velocity.z = velocity.z * 2
        minimal.log(
            "on gumboline!" .. "x:" .. velocity.x .. "y:" .. velocity.y .. "z:" .. velocity.z
        )
        player:add_velocity(velocity)
    end
end

local function step_player_walk_run(player)
    local ctrl = player:get_player_control()
    if ctrl.aux1 then -- aux1 is often mapped to the 'E' or 'Ctrl' key by default
        player_monoids.speed:add_change(player, player_definition.run_velocity, "main")
    else
        player_monoids.speed:add_change(player, player_definition.walk_velocity, "main")
    end
end

local function player_wake_up(player, msg)
    local meta = player:get_meta()
    if meta:get_int("sleeping") == 1 then
        meta:set_int("sleeping", 0)
        globo.chat_send_player(player:get_player_name(), msg)
        local ov = player:get_physics_override()
        ov.speed = player_definition.walk_velocity
        player:set_physics_override(ov)
    end
end

-- Manage sleeping
-- returns true if not sleeping
local function step_player_sleep(player, dtime)
    local meta = player:get_meta()

    if meta:get_int("sleeping") == 1 then

        --[[
        local eMax = meta:get_float("energy_max")
        local eCur = meta:get_float("energy")

        -- Increase energy and health while sleeping
        eMax = math.min(eMax + (ENERGY_RECOVERY_RATE_SLEEP * dtime), 1000)
        eCur = math.min(eCur + (ENERGY_RECOVERY_RATE_SLEEP * dtime), eMax) 
        meta:set_float("energy", eCur)
        meta:set_float("energy_max",eMax)
        changePlayerHP(player,HEAL_RATE_SLEEP * dtime)
]] --
        local controls = player:get_player_control()
        -- globo.chat_send_player(player:get_player_name(), dump(controls))
        -- if you move forward or jump, you wake up
        if controls.up or controls.jump then
            player_wake_up(player, "You wake up.")
            return true
        end

        local pos = player:get_pos()
        pos.y = pos.y + 0.5
        local node = minetest.get_node(pos)
        if node.name ~= "air" then
            player_wake_up(player, "You can't sleep here.")
            return true
        end

        local eCur = meta:get_float("energy")
        -- Wake up if fully rested
        if eCur == 1000 then
            player_wake_up(player, "You wake up feeling rested.")
        else
            return false
        end
    end
    return true
end

local timer = 0
minetest.register_globalstep(
    function(dtime)
        timer = timer + dtime
        if timer >= 20 then -- Adjust the interval as needed
            timer = 0
            for _, player in ipairs(minetest.get_connected_players()) do
                local pos = player:get_pos()
                local spawn_pos = globo.get_random_position_around_pos(pos, 100, 150)
                spawn_mob_at_position(spawn_pos)
            end
        end

        for _, player in ipairs(minetest.get_connected_players()) do

            local meta = player:get_meta()
            local timePlayed = meta:get_float("time") + dtime
            meta:set_float("time", timePlayed)
            if timePlayed > globo.GAME_LENGTH then
                local score = meta:get_int("score") or 0
                minimal.log("******************************************")
                minimal.log("Game over man! Game over!")
                minimal.log("Score: " .. score)
                minimal.log("******************************************")
                meta:set_float("time", 0)
            end

            local tod = minetest.get_timeofday()
            if tod > .82 or tod < .19 then
                player:override_day_night_ratio(0)
            else
                player:override_day_night_ratio(nil)
            end

            node_collisions_player(player)

            if step_player_sleep(player, dtime) then
                step_player_walk_run(player)
            end
            update_hud(player)
            inventoryEffects(player)

            globo.lay_egg_item_update(player)
        end
    end
)

minetest.register_on_joinplayer(
    function(player)
        local meta = player:get_meta()
        if meta:get_int("joined") == 0 then
            meta:set_int("joined", 1)
            meta:set_float("time", 0)

            player:set_properties({hp_max = globo.PLAYER_MAX_HEALTH})
            player:set_hp(globo.PLAYER_MAX_HEALTH)
            meta:set_int("energy", 1000)
            meta:set_int("energy_max", 1000)
            meta:set_int("hunger", 1000)
            meta:set_int("thirst", 1000)
            meta:set_int("score", 0)
        end

        -- "hotbar", "healthbar", "crosshair", "wielditem", "breathbar", "minimap", "minimap_radar", "basic_debug", "chat"
        local flags = player:hud_get_flags()
        flags["healthbar"] = false
        flags["breathbar"] = false
        player:hud_set_flags(flags)

        player:set_properties({hp_max = globo.PLAYER_MAX_HEALTH})

        local privs = minetest.get_player_privs("singleplayer")
        privs.fly = true
        privs.fast = true
        privs.settime = true
        privs.give = true
        privs.set_weather = true
        minetest.set_player_privs("singleplayer", privs)

        local inv = player:get_inventory()
        inv:set_size("main", globo.INVENTORY_SIZE)

        globo.allocate_hud_for_player(player)

        if globo.HUD == globo.HudSetting.agent then
            setup_hud_agent(player)
        elseif globo.HUD == globo.HudSetting.player then
            setup_hud_player(player)
        end
        -- player:override_day_night_ratio(.1)
        globo.lay_egg_item_init(player)
        globo.sleep_item_init(player)
    end
)

minetest.register_on_leaveplayer(
    function(player)
        globo.deallocate_hud_for_player(player)
    end
)

minetest.register_on_respawnplayer(
    function(player)
        local meta = player:get_meta()
        meta:set_int("energy", 1000)
        meta:set_int("energy_max", 1000)
        meta:set_int("hunger", 1000)
        meta:set_int("thirst", 1000)
        meta:set_int("sleeping", 0)
        local score = meta:get_int("score") or 0
        score = score - 6
        meta:set_int("score", score)

        -- get rid of past inventory
        local inv = player:get_inventory()
        inv:set_size("main", globo.INVENTORY_SIZE)
        inv:set_list("main", {})

        player:set_properties({hp_max = globo.PLAYER_MAX_HEALTH})
        player:set_hp(globo.PLAYER_MAX_HEALTH)

        globo.lay_egg_item_init(player)
        globo.sleep_item_init(player)

        return false
    end
)

function globo.player_sleep(player)
    local meta = player:get_meta()
    if meta:get_int("sleeping") == 1 then return end
    if meta:get_int("energy") >= 1000 then
        globo.chat_send_player(player:get_player_name(), "Already fully rested; sleep when energy is below 1000.")
        return
    end
    -- make sure the player isn't in water
    local pos = player:get_pos()
    pos.y = pos.y + 0.5
    local node = minetest.get_node(pos)
    if node.name ~= "air" then
        globo.chat_send_player(player:get_player_name(), "You cannot sleep here: " .. node.name)
        return
    end

    meta:set_int("sleeping", 1)
    local ov = player:get_physics_override()
    ov.speed = 0
    player:set_physics_override(ov) -- Player cannot move while sleeping
    globo.chat_send_player(player:get_player_name(), "You go to sleep.")
end

globo.LAY_EGG_HUNGER_COST = 120
globo.LAY_EGG_HUNGER_MIN = 500
globo.LAY_EGG_ENERGY_COST = 100
globo.LAY_EGG_ENERGY_MIN = 500
globo.LAY_EGG_DEFAULT_SLOT = 3 -- inventory slot

-- Returns the number of eggs the player can lay.
function globo.lay_egg_possible_count(player)
    local meta = player:get_meta()
    local ch = math.floor(
        (meta:get_int("hunger") - globo.LAY_EGG_HUNGER_MIN) / globo.LAY_EGG_HUNGER_COST
    )
    local ce = math.floor(
        (meta:get_int("energy") - globo.LAY_EGG_ENERGY_MIN) / globo.LAY_EGG_ENERGY_COST
    )
    return math.max(0, math.min(ch, ce))
end

function globo.lay_egg_is_item_name(name)
    return name == "main:lay_egg" or name == "main:lay_egg_disabled"
end

-- Returns true if the player was able to lay an egg.
function globo.lay_egg(player)
    local meta = player:get_meta()
    local energy = meta:get_int("energy")
    local hunger = meta:get_int("hunger")
    if hunger <= globo.LAY_EGG_HUNGER_MIN or energy <= globo.LAY_EGG_ENERGY_MIN then
        globo.chat_send_player(player:get_player_name(), "You are too weak to lay an egg.")
        return false
    end

    local pos = player:get_pos()
    pos.y = pos.y + 0.5
    local node = minetest.get_node(pos)
    if node.name ~= "air" then
        globo.chat_send_player(player:get_player_name(), "Unable to lay an egg here.")
        return false
    end

    minetest.add_node(pos, {name = "main:player_egg"})
    minetest.get_meta(pos):set_string("owner", player:get_player_name())
    globo.add_nutrient(player, "hunger", -globo.LAY_EGG_HUNGER_COST)
    globo.add_energy(player, -globo.LAY_EGG_ENERGY_COST)
    globo.chat_send_player(player:get_player_name(), "You laid an egg.")
    return true
end

-- Compute the desired item stack based on the possible number of eggs the
-- player can lay.
function globo.lay_egg_desired_stack(player)
    local possible_count = globo.lay_egg_possible_count(player)
    return possible_count < 1 and ItemStack({name = "main:lay_egg_disabled"}) or
               ItemStack({name = "main:lay_egg", count = possible_count})
end

-- Ensures that exactly one main:lay_egg or main:lay_egg_disabled item stack is
-- in the player inventory.
function globo.lay_egg_item_init(player)
    local inventory = player:get_inventory()

    local stack_index = nil
    for index, stack in ipairs(inventory:get_list("main")) do
        if globo.lay_egg_is_item_name(stack:get_name()) then
            if stack_index ~= nil then
                inventory:set_stack("main", index, nil)
            else
                stack_index = index
            end
        end
    end

    if stack_index == nil then
        inventory:set_stack("main", globo.LAY_EGG_DEFAULT_SLOT, globo.lay_egg_desired_stack(player))
    end
end

-- Updates the player's lay_egg item or lay_egg_disabled tool based on the
-- player's stats.
function globo.lay_egg_item_update(player)
    local desired_stack = globo.lay_egg_desired_stack(player)

    local inventory = player:get_inventory()

    -- find lay_egg item index
    local stack_index = nil
    for index, stack in ipairs(inventory:get_list("main")) do
        if globo.lay_egg_is_item_name(stack:get_name()) then
            stack_index = index
            break
        end
    end

    -- Should be guaranteed by `globo.lay_egg_item_init` and `globo.lay_egg_prevent_transfer`.
    assert(stack_index ~= nil, "player should have lay_egg or lay_egg_disabled item in inventory")

    local current_stack = inventory:get_stack("main", stack_index)
    if current_stack:get_name() ~= desired_stack:get_name() or current_stack:get_count() ~=
        desired_stack:get_count() then
        inventory:set_stack("main", stack_index, desired_stack)
    end
end

globo.SLEEP_DEFAULT_SLOT = 4

-- Ensures that exactly one main:lay_egg or main:lay_egg_disabled item stack is
-- in the player inventory.
function globo.sleep_item_init(player)
    local inventory = player:get_inventory()

    local stack_index = nil
    for index, stack in ipairs(inventory:get_list("main")) do
        if stack:get_name() == "main:sleep" then
            if stack_index ~= nil then
                inventory:set_stack("main", index, nil)
            else
                stack_index = index
            end
        end
    end

    if stack_index == nil then
        inventory:set_stack("main", globo.SLEEP_DEFAULT_SLOT, ItemStack({name = "main:sleep"}))
    end
end

function globo.is_prevent_transfer_item_name(name)
    return globo.lay_egg_is_item_name(name) or name == "main:sleep"
end

-- Prevents the lay egg item from transfering between inventories. See
-- `register_allow_player_inventory_action` in
-- https://api.minetest.net/core-namespace-reference/#global-callback-registration-functions
function globo.prevent_item_transfer(player, action, inventory, inventory_info)
    if action == "move" then
        -- inventory_info = {from_list=string, to_list=string, from_index=number, to_index=number, count=number}

        local inventory = player:get_inventory();
        local stack = inventory:get_stack(inventory_info.from_list, inventory_info.from_index)
        if globo.is_prevent_transfer_item_name(stack:get_name()) then
            if inventory_info.count ~= stack:get_count() then
                -- Disallow partially moving item.
                return 0
            end
            if inventory_info.from_list ~= inventory_info.to_list then
                -- Disallow moving items between inventories.
                return 0
            end
        end

        return inventory_info.count
    elseif action == "put" or action == "take" then
        -- inventory_info = {listname=string, index=number, stack=ItemStack}

        local stack = inventory_info.stack
        if globo.is_prevent_transfer_item_name(stack:get_name()) then
            -- Prevent dropping or picking up item.
            return 0
        end

        return stack:get_count()
    end
end

minetest.register_allow_player_inventory_action(globo.prevent_item_transfer)

-- Function to handle when a player is hurt
local function on_player_hpchange(player, hp_change, reason)
    local properties = player:get_properties()
    local max_hp = properties.hp_max
    minimal.log("Player hurt: " .. hp_change .. " reason: " .. reason.type .. " max: " .. max_hp)
    if reason.type == "fall" or reason.type == "drown" then
        hp_change = hp_change * 50
    end

    local meta = player:get_meta()
    if hp_change < 0 and meta:get_int("sleeping") == 1 then
        -- Player hurt while sleeping
        player_wake_up(player, "You were hurt and have woken up.")
    end
    return hp_change
end

-- Override Minetest's on_player_hpchange callback
minetest.register_on_player_hpchange(on_player_hpchange, true)
