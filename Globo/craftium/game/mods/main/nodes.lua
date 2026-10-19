-- Node Definitions
local function start_node_timer(pos)
    local timer = minetest.get_node_timer(pos)
    timer:start(60 * 1 * globo.GAME_SPEED)
end

minetest.register_node(
    "main:glow_stone", {
        description = "Glow Stone",
        drawtype = "normal",
        tiles = {"glow_stone.png"},
        light_source = 14,
        groups = {cracky = 3, falling_node = 1},
        on_punch = function(pos, node, player, pointed_thing)
            -- move this node 1 space in the direction the player is facing if possibl
            local dir = player:get_look_dir()
            -- globo.chat_send_player(player:get_player_name()," " .. dir.x .. " " .. dir.y .. " " .. dir.z)
            local newpos = {x = pos.x + dir.x, y = pos.y, z = pos.z + dir.z}
            if minetest.get_node(newpos).name ~= "air" then
                return
            end
            minetest.set_node(newpos, {name = "main:glow_stone"})
            minetest.remove_node(pos)
            minetest.check_for_falling(newpos)
        end,
    }
)

minetest.register_node(
    "main:brambles", {
        description = "Brambles",
        drawtype = "plantlike",
        tiles = {"brambles.png"},
        inventory_image = "brambles.png",
        wield_image = "brambles.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 3, flammable = 40, falling_node = 1, flora = 1},
        damage_per_second = 50,
    }
)

minetest.register_node(
    "main:pulse_blossom", {
        description = "Pulse Blossom",
        drawtype = "plantlike",
        tiles = {"pulse_blossom.png"},
        inventory_image = "pulse_blossom.png",
        wield_image = "pulse_blossom.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 1, flammable = 20, falling_node = 1, flora = 1},
        on_construct = function(pos)
            local timer = minetest.get_node_timer(pos)
            local t = (math.random(60, 270) + math.random(60, 270)) / 2
            timer:start(t * globo.GAME_SPEED)
        end,
        on_timer = function(pos, elapsed)
            minetest.set_node(pos, {name = "main:pulse_blossom_on"})
            return false
        end,
    }
)

minetest.register_node(
    "main:pulse_blossom_on", {
        description = "Pulse Blossom",
        drawtype = "plantlike",
        tiles = {"pulse_blossom_on.png"},
        inventory_image = "pulse_blossom.png",
        wield_image = "pulse_blossom.png",
        paramtype = "light",
        light_source = 10,
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 1, flammable = 20, falling_node = 1, flora = 1},
        on_construct = function(pos)
            local timer = minetest.get_node_timer(pos)
            timer:start(30 * globo.GAME_SPEED)
        end,
        on_timer = function(pos, elapsed)

            local node = minetest.get_node(pos)
            local p1 = node.param2 + 1
            if p1 == 1 then
                minetest.swap_node(pos, {name = "main:pulse_blossom_on", param2 = p1})
                local timer = minetest.get_node_timer(pos)
                timer:start(1)
                return false
            end

            if p1 > 20 then
                minetest.set_node(pos, {name = "main:pulse_blossom"})
                return false
            end

            -- cause damage to any creature around
            local objects = minetest.get_objects_inside_radius(pos, 4)
            for _, obj in ipairs(objects) do
                if obj:is_player() then
                    globo.chat_send_player(
                        obj:get_player_name(), "You have been damaged by a pulse blossom"
                    )
                    globo.add_hp(obj, -200)
                elseif obj:get_luaentity() ~= nil then
                    local mob = obj:get_luaentity()
                    mobkit.hurt(mob, 200)
                end
            end

            minetest.swap_node(pos, {name = "main:pulse_blossom_on", param2 = p1})
            return true
        end,
    }
)

minetest.register_node(
    "main:player_egg", {
        description = 'Player Egg',
        tiles = {"animals_gundu_eggs.png"},
        stack_max = minimal.stack_max_medium,
        drawtype = "nodebox",
        paramtype = "light",
        node_box = {type = "fixed", fixed = {-0.125, -0.5, -0.125, 0.125, -0.125, 0.125}},
        groups = {
            snappy = 3,
            falling_node = 1,
            dig_immediate = 3,
            flammable = 10,
            temp_pass = 1,
            edible = 1,
            egg = 1,
        },
        on_construct = function(pos)
            -- minimal.log("player egg on construct")
            minetest.get_node_timer(pos):start(600 * globo.GAME_SPEED)
        end,
        after_place_node = function(pos, placer, itemstack, pointed_thing)
            -- minimal.log("player egg after_place_node")
            minetest.get_meta(pos):set_string("owner", placer:get_player_name())
        end,
        on_timer = function(pos, elapsed)
            local playerName = minetest.get_meta(pos):get_string("owner")
            -- minimal.log("player egg on timer "..dump(playerName))

            if playerName ~= "" then
                local player = minetest.get_player_by_name(playerName)
                if player ~= nil then
                    local meta = player:get_meta()
                    local score = meta:get_int("score") or 0
                    score = score + 1
                    meta:set_int("score", score)
                end
            end
            -- remove egg
            minetest.remove_node(pos)
            return false
        end,
    }
)

minetest.register_node(
    "main:thorns", {
        description = "Thorns",
        drawtype = "plantlike",
        tiles = {"thorns_1.png"},
        inventory_image = "thorns_0.png",
        wield_image = "thorns_0.png",
        paramtype = "light",
        damage_per_second = 50,
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 1, flammable = 30, falling_node = 1, flora = 1},
        on_construct = function(pos)
            start_node_timer(pos)
        end,
        on_timer = function(pos, elapsed)
            if math.random(1, 1000) <= 60 then
                minetest.set_node(pos, {name = "main:thorns_fruit"})
            end
            return true -- Continue the cycle
        end,
    }
)

minetest.register_node(
    "main:thorns_fruit", {
        description = "Thorns",
        drawtype = "plantlike",
        tiles = {"thorns_3.png"},
        inventory_image = "thorns_0.png",
        wield_image = "thorns_0.png",
        paramtype = "light",
        damage_per_second = 50,
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 1, flammable = 30, falling_node = 1, flora = 2},
        on_punch = function(pos, node, player, pointed_thing)
            globo.add_nutrient(player, "hunger", 100)
            minetest.set_node(pos, {name = "main:thorns"})
        end,
    }
)

local function potatoes_spread(pos)
    -- Chance of spreading potatoes to adjacent positions
    local positions = minetest.find_nodes_in_area_under_air(
        {x = pos.x - 2, y = pos.y - 1, z = pos.z - 2},
            {x = pos.x + 2, y = pos.y + 1, z = pos.z + 2}, {"group:spreading"}
    )

    if #positions > 0 then
        local newPos = positions[math.random(#positions)]
        newPos.y = newPos.y + 1
        minetest.set_node(newPos, {name = "main:potatoes"})
        return true
    end
    return false
end

minetest.register_node(
    "main:potatoes", {
        description = "Potatoes",
        drawtype = "plantlike",
        tiles = {"potato_plant.png"},
        inventory_image = "potato.png",
        wield_image = "potato.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 3, flammable = 30, falling_node = 1, flora = 4},
        sounds = {},
        on_construct = function(pos)
            local timer = minetest.get_node_timer(pos)
            timer:start(600 * globo.GAME_SPEED)
            minetest.get_meta(pos):set_int("potato", 0)
        end,
        on_timer = function(pos, elapsed)
            local meta = minetest.get_meta(pos)
            local pCount = meta:get_int("potato") + 1
            if pCount > 2 then
                if potatoes_spread(pos) then
                    pCount = 0
                end
            end
            meta:set_int("potato", pCount)

            return true -- Continue the cycle
        end,
        on_use = function(itemstack, player, pointed_thing)
            globo.add_nutrient(player, "hunger", 20)
            itemstack:take_item()
            return itemstack
        end,
        on_punch = function(pos, node, player, pointed_thing)
            local meta = minetest.get_meta(pos)
            local pCount = meta:get_int("potato")
            if pCount > 0 then -- Check if potato
                local inv = player:get_inventory()
                if inv:room_for_item("main", "main:potatoes") then
                    -- globo.chat_send_player("singleplayer", "potato harvest taken")
                    inv:add_item("main", "main:potatoes")
                    meta:set_int("potato", 0)
                end
            end
        end,
    }
)

minetest.register_node(
    "main:sumac", {
        description = "Sumac",
        drawtype = "plantlike",
        tiles = {"sumac.png"},
        inventory_image = "sumac.png",
        wield_image = "sumac.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 3, flammable = 30, falling_node = 1, flora = 1},
        on_punch = function(pos, node, player, pointed_thing)
            globo.add_hp(player, -100)
        end,
    }
)

minetest.register_node(
    "main:goldcap", {
        description = "Goldcap",
        drawtype = "plantlike",
        tiles = {"goldcap.png"},
        inventory_image = "goldcap.png",
        wield_image = "goldcap.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 3, flammable = 10, falling_node = 1, flora = 1},
        on_use = function(itemstack, player, pointed_thing)
            globo.add_hp(player, 100) -- LATER: make it heal over time
            itemstack:take_item()
            return itemstack
        end,
    }
)

minetest.register_node(
    "main:moon_berry", {
        description = "Moon Berry",
        drawtype = "plantlike",
        tiles = {"moon_berry.png"},
        inventory_image = "moon_berry.png",
        wield_image = "moon_berry.png",
        paramtype = "light",
        light_source = 4,
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 3, flammable = 30, falling_node = 1, flora = 2},
    }
)

minetest.register_node(
    "main:sun_berry", {
        description = "Sun Berry",
        drawtype = "plantlike",
        tiles = {"sun_berry.png"},
        inventory_image = "sun_berry.png",
        wield_image = "sun_berry.png",
        paramtype = "light",
        light_source = 13,
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 3, flammable = 30, falling_node = 1, flora = 2},
    }
)

minetest.register_node(
    "main:coffee", {
        description = "Coffee",
        drawtype = "plantlike",
        tiles = {"coffee.png"},
        inventory_image = "coffee.png",
        wield_image = "coffee.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 3, flammable = 30, falling_node = 1, flora = 2},
        on_punch = function(pos, node, player, pointed_thing)
            globo.add_energy(player, 200)
        end,
    }
)

minetest.register_node(
    "main:barley", {
        drawtype = "plantlike",
        tiles = {"barley.png"},
        inventory_image = "barley.png",
        wield_image = "barley.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 3, flammable = 30, falling_node = 1, flora = 5},
        after_dig_node = function(pos, oldnode, oldmetadata, digger)
            if digger and digger:is_player() then
                local inv = digger:get_inventory()
                local stack = ItemStack(oldnode.name .. " 1")
                if inv then
                    inv:add_item("main", stack)
                end
            end
        end,
    }
)

local function grib_spread(pos)
    -- Chance of spreading grib weed to adjacent positions
    local positions = minetest.find_nodes_in_area_under_air(
        {x = pos.x - 1, y = pos.y - 1, z = pos.z - 1},
            {x = pos.x + 1, y = pos.y + 1, z = pos.z + 1}, {"group:flora"}
    )

    if #positions > 0 then
        local newPos = positions[math.random(#positions)]
        if minetest.get_node(newPos).name ~= "main:grib_weed" then
            minetest.set_node(newPos, {name = "main:grib_weed"})
            return
        end
    end

    positions = minetest.find_nodes_in_area_under_air(
        {x = pos.x - 1, y = pos.y - 1, z = pos.z - 1},
            {x = pos.x + 1, y = pos.y + 1, z = pos.z + 1}, {"group:spreading"}
    )

    if #positions > 0 then
        local newPos = positions[math.random(#positions)]
        newPos.y = newPos.y + 1
        minetest.set_node(newPos, {name = "main:grib_weed"})
    end
end

minetest.register_node(
    "main:grib_weed", {
        description = "Grib Weed",
        drawtype = "plantlike",
        tiles = {"grib_weed.png"},
        inventory_image = "grib_weed.png",
        wield_image = "grib_weed.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 3, flammable = 35, falling_node = 1, flora = 0},
        on_timer = function(pos, elapsed)
            if math.random() < .1 then
                grib_spread(pos)
            end
            return true
        end,
        on_construct = function(pos)
            -- minetest.log("action", "on_construct grib")
            start_node_timer(pos)
        end,
    }
)

local function dig_up_set(pos, mainName, names, digger)
    local np = {x = pos.x, y = pos.y + 1, z = pos.z}
    local unode = minetest.get_node(np)
    local count = 0
    while globo.is_in_list(unode.name, names) do
        count = count + 1
        minetest.set_node(np, {name = "air"})
        np.y = np.y + 1
        unode = minetest.get_node(np)
    end
    if count > 0 then
        if not digger or not digger.get_inventory then
            return
        else
            local inv = digger:get_inventory()
            local leftover = inv:add_item('main', mainName .. " " .. tostring(count))
            if leftover then
                minetest.add_item(pos, leftover)
            end
        end
    end
end

-- grows 8 tall
-- fruit appears at the top after sometime
-- if the bottom is destroyed, the thing goes away
minetest.register_node(
    "main:papaya", {
        description = "Papaya",
        drawtype = "plantlike",
        tiles = {"papaya.png"},
        inventory_image = "papaya.png",
        wield_image = "papaya.png",
        paramtype = "light",
        stack_max = 4,
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 3, flammable = 35, falling_node = 1, flora = 3},

        on_timer = function(pos, elapsed)
            -- grow the plant
            local startY = pos.y
            if minimal.get_daylight(pos) < 13 then
                return true
            end

            pos.y = pos.y + 1
            local above = minetest.get_node(pos)
            pos.y = pos.y - 1
            if above.name ~= "air" then
                return true
            end

            -- determine height of plant
            local height = 0
            pos.y = pos.y - 1
            local node = minetest.get_node(pos)
            while node.name == "main:papaya" do
                height = height + 1
                pos.y = pos.y - 1
                node = minetest.get_node(pos)
            end

            -- make sure plant is propperly rooted
            if minetest.get_item_group(node.name, "sediment") < 1 then
                return false
            end

            if height < 8 then
                pos.y = startY + 1
                minetest.set_node(pos, {name = "main:papaya"})
            else
                pos.y = startY
                minetest.set_node(pos, {name = "main:papaya_flower"})
            end

            --[[ LATER
        ---extreme stop growth
        local temp = climate.get_point_temp(pos)
        if temp < 10 or temp > 40 then
            return
        end
        ]] --

            return false
        end,
        on_construct = function(pos)
            local timer = minetest.get_node_timer(pos)
            local time = math.random(60, 270) + math.random(60, 270)
            timer:start(time * globo.GAME_SPEED * .3)
        end,
        after_dig_node = function(pos, node, metadata, digger)
            dig_up_set(
                pos, "main:papaya", {"main:papaya", "main:papaya_fruit", "main:papaya_flower"},
                    digger
            )
        end,
    }
)

minetest.register_node(
    "main:papaya_flower", {
        description = "Papaya",
        drawtype = "plantlike",
        tiles = {"papaya_flower.png"},
        inventory_image = "papaya_flower.png",
        wield_image = "papaya_flower.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        drop = "",
        groups = {snappy = 3, flammable = 35, flora = 3},
        on_construct = function(pos)
            local timer = minetest.get_node_timer(pos)
            local time = math.random(90, 300) + math.random(90, 300)
            timer:start(time * globo.GAME_SPEED * .3)
        end,
        on_timer = function(pos, elapsed)
            minetest.set_node(pos, {name = "main:papaya_fruit"})
        end,
    }
)

minetest.register_node(
    "main:papaya_fruit", {
        description = "Papaya",
        drawtype = "plantlike",
        tiles = {"papaya_fruit.png"},
        inventory_image = "papaya_fruit_only.png",
        wield_image = "papaya_fruit_only.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        drop = "",
        groups = {snappy = 3, flammable = 35, flora = 3},
        on_punch = function(pos, node, player, pointed_thing)
            local inv = player:get_inventory()
            if inv:room_for_item("main", "main:papaya_fruit") then
                -- globo.chat_send_player("singleplayer", "potato harvest taken")
                inv:add_item("main", "main:papaya_fruit")
                minetest.set_node(pos, {name = "main:papaya"})
            end
        end,
        on_use = function(itemstack, user, pointed_thing)
            return HEALTH.exile_eatdrink(itemstack, user, pointed_thing)
        end,
    }
)

minetest.register_node(
    "main:pond", {
        on_construct = function(pos)
            -- TODO: hack since mts files seem annoying

            local pondDepth = math.random(-11, -5)
            local pondDepth2 = math.random(-11, -5)
            pondDepth = math.max(pondDepth, pondDepth2)

            for x = -2, 2 do
                for z = -2, 2 do
                    if x == -2 or x == 2 or z == -2 or z == 2 and math.random() < .5 then
                        -- skip this column
                    else
                        ---@diagnostic disable-next-line: count-down-loop
                        for y = pondDepth + 1, 1 do
                            local p = {x = pos.x + x, y = pos.y + y, z = pos.z + z}
                            minetest.set_node(p, {name = "air"})
                        end

                        if x == -2 or x == 2 or z == -2 or z == 2 and math.random() < .5 then
                            -- skip this column
                        else
                            for y = pondDepth - 3, pondDepth do
                                local p = {x = pos.x + x, y = pos.y + y, z = pos.z + z}
                                minetest.set_node(p, {name = "nodes_nature:freshwater_source"})
                            end
                        end
                    end
                end
            end
        end,
    }
)

minetest.register_node(
    "main:mesa", {
        on_construct = function(pos)
            local timer = minetest.get_node_timer(pos)
            timer:start(1)
        end,
        on_timer = function(pos)
            -- TODO: hack since I don't want to figure out terrain
            -- move every block up 4 blocks in some random radius of the mesa block

            minetest.set_node(pos, {name = "air"})

            local grid_size = math.random(15, 25)
            -- local grid_size=math.random(2,4)

            local mesa_grid = {}
            for i = 1, grid_size do
                mesa_grid[i] = {}
                for j = 1, grid_size do
                    mesa_grid[i][j] = 0
                end
            end

            local center = {x = grid_size / 2, y = 0, z = grid_size / 2}
            local center_grid = math.floor(center.x)

            local max_distance = globo.get_distance({x = grid_size, y = 0, z = 0}, center)

            for x = 1, grid_size do
                for z = 1, grid_size do
                    local d = globo.get_distance({x = x, y = 0, z = z}, center)
                    local chance = ((max_distance - d) / max_distance) + .05
                    if math.random() < chance then
                        mesa_grid[x][z] = 1
                    end
                end
            end

            for x = 2, center_grid do
                for z = 2, center_grid do
                    if mesa_grid[x][z] == 0 then
                        if mesa_grid[x][z - 1] == 1 or mesa_grid[x - 1][z] == 1 then
                            mesa_grid[x][z] = 1
                        end
                    end
                end
            end

            for x = 2, center_grid do
                for z = grid_size - 1, center_grid, -1 do
                    if mesa_grid[x][z] == 0 then
                        if mesa_grid[x - 1][z] == 1 or mesa_grid[x][z + 1] == 1 then
                            mesa_grid[x][z] = 1
                        end
                    end
                end
            end

            for x = grid_size - 1, center_grid, -1 do
                for z = 2, center_grid do
                    if mesa_grid[x][z] == 0 then
                        if mesa_grid[x + 1][z] == 1 or mesa_grid[x][z - 1] == 1 then
                            mesa_grid[x][z] = 1
                        end
                    end
                end
            end

            for x = grid_size - 1, center_grid, -1 do
                for z = grid_size - 1, center_grid, -1 do
                    if mesa_grid[x][z] == 0 then
                        if mesa_grid[x + 1][z] == 1 or mesa_grid[x][z + 1] == 1 then
                            mesa_grid[x][z] = 1
                        end
                    end
                end
            end
            pos.x = pos.x - center_grid
            pos.z = pos.z - center_grid
            -- minimal.log("mesa start: "..pos.x..","..pos.y..","..pos.z)
            for x = 1, grid_size do
                for z = 1, grid_size do
                    if mesa_grid[x][z] == 1 then
                        -- raise up all the blocks
                        for y = 12, -10, -1 do
                            local node = minetest.get_node(
                                {x = pos.x + x, y = pos.y + y, z = pos.z + z}
                            )
                            if node and node.name and node.name ~= "ignore" then
                                local replace_node = minetest.get_node(
                                    {x = pos.x + x, y = pos.y + y + 4, z = pos.z + z}
                                )
                                if replace_node and replace_node.name and replace_node.name ~=
                                    "ignore" then
                                    minetest.set_node(
                                        {x = pos.x + x, y = pos.y + y + 4, z = pos.z + z}, node
                                    )
                                end
                            else
                                -- minimal.log("node nil: ".."(x="..pos.x+x..",y="..pos.y+y..",z="..pos.z+z..")")
                            end
                        end
                    end
                end
            end
        end,
    }
)

minetest.register_node(
    "main:dawn_cactus", {
        description = "Thorns",
        drawtype = "plantlike",
        tiles = {"dawn_cactus.png"},
        inventory_image = "dawn_cactus.png",
        wield_image = "dawn_cactus.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 1, flammable = 30, falling_node = 1, flora = 1},
        on_timer = function(pos, elapsed)
            -- check if game time is dawn
            local time_of_day = minetest.get_timeofday()
            -- minimal.log("dawn cactus timer: "..time_of_day)
            local node = minetest.get_node(pos)
            if node.param2 == 1 then
                if time_of_day > 0.34 or time_of_day < 0.22 then
                    -- minimal.log("cactus rest")
                    minetest.swap_node(pos, {name = "main:dawn_cactus", param2 = 0})
                    return true
                end
            else
                if time_of_day < 0.34 and time_of_day > 0.22 then
                    minetest.set_node(pos, {name = "main:dawn_cactus_fruit"})
                end
            end
            return true
        end,
        on_punch = function(pos, node, player, pointed_thing)
            globo.add_hp(player, -50)
        end,
        on_construct = function(pos)
            -- minimal.log("cactus on construct")
            local timer = minetest.get_node_timer(pos)
            timer:start(30 * globo.GAME_SPEED)
        end,
    }
)

minetest.register_node(
    "main:dawn_cactus_fruit", {
        description = "Thorns",
        drawtype = "plantlike",
        tiles = {"dawn_cactus_fruit.png"},
        inventory_image = "dawn_cactus.png",
        wield_image = "dawn_cactus.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 1, flammable = 30, falling_node = 1, flora = 1},
        on_timer = function(pos, elapsed)
            minetest.set_node(pos, {name = "main:dawn_cactus", param2 = 1})
        end,
        on_punch = function(pos, node, player, pointed_thing)
            globo.add_nutrient(player, "hunger", 100)
            globo.add_nutrient(player, "thirst", 50)
            minetest.set_node(pos, {name = "main:dawn_cactus", param2 = 1})
        end,
        on_construct = function(pos)
            local timer = minetest.get_node_timer(pos)
            timer:start(60 * globo.GAME_SPEED)
        end,
    }
)

-- Register the balloon entity
minetest.register_entity(
    "main:balloo_hyce", {
        initial_properties = {
            physical = true,
            collide_with_objects = true,
            collisionbox = {-0.5, 0, -0.5, 0.5, 1, 0.5},
            visual = "sprite",
            textures = {"balloo_hyce.png"},
            visual_size = {x = 1, y = 1},
            static_save = true,
        },
        on_activate = function(self, staticdata)
            self.object:set_velocity({x = 0, y = 0.5, z = 0}) -- Initial upward movement
            self.direction = 1 -- 1 for up, -1 for down
            self.max_height = self.object:get_pos().y + math.random(5, 20) -- Random max height
        end,
        on_step = function(self, dtime)
            local pos = self.object:get_pos()

            if self.direction == 1 and pos.y >= self.max_height then
                self.object:set_velocity({x = 0, y = -0.5, z = 0}) -- Start moving down
                self.direction = -1
            elseif self.direction == -1 then
                local node = minetest.get_node({x = pos.x, y = pos.y - 1, z = pos.z})
                if node.name ~= "air" then
                    -- remove the balloon
                    self.object:remove()
                    return
                end
            end

            -- Ensure the velocity is always set to prevent stopping
            local vel = self.object:get_velocity()
            if self.direction == 1 and vel.y <= 0 then
                self.object:set_velocity({x = 0, y = 0.5, z = 0})
            elseif self.direction == -1 and vel.y >= 0 then
                self.object:set_velocity({x = 0, y = -0.5, z = 0})
            end
        end,
    }
)

minetest.register_node(
    "main:balloo_hyce_plant", {

        drawtype = "plantlike",
        tiles = {"balloo_hyce_plant.png"},
        inventory_image = "balloo_hyce_plant.png",
        wield_image = "balloo_hyce_plant.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 3, flammable = 30, falling_node = 0, flora = 1},

        on_timer = function(pos, elapsed)
            -- if node under this is water, spawn the balloon
            local node = minetest.get_node({x = pos.x, y = pos.y - 1, z = pos.z})
            if node.name == "default:water_source" or node.name == "nodes_nature:salt_water_source" or
                node.name == "nodes_nature:salt_water_flowing" or node.name ==
                "nodes_nature:freshwater_source" then
                pos.y = pos.y + 0.5
                minetest.add_entity(pos, "main:balloo_hyce")
            end
        end,
        on_punch = function(pos, node, player, pointed_thing)
            -- move this node 1 space in the direction the player is facing if possibl
            local dir = player:get_look_dir()
            -- globo.chat_send_player(player:get_player_name()," " .. dir.x .. " " .. dir.y .. " " .. dir.z)
            local newpos = {x = pos.x + dir.x, y = pos.y, z = pos.z + dir.z}
            if minetest.get_node(newpos).name ~= "air" then
                return
            end
            minetest.set_node(newpos, {name = "main:balloo_hyce_plant"})
            minetest.remove_node(pos)
            minetest.check_for_falling(newpos)
        end,
        on_construct = function(pos)
            local timer = minetest.get_node_timer(pos)
            local ft = math.random(0, 12) - 6
            timer:start(60 * 2 * globo.GAME_SPEED + ft)
        end,
    }
)

minetest.register_node(
    "main:sea_beans", {
        description = "Thorns",
        drawtype = "plantlike",
        tiles = {"sea_beans.png"},
        inventory_image = "sea_beans.png",
        wield_image = "sea_beans.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 3, flammable = 30, falling_node = 0, flora = 1},
        on_construct = function(pos)
            local ft = math.random(150, 450) + math.random(150, 450)
            local timer = minetest.get_node_timer(pos)
            timer:start(ft * globo.GAME_SPEED)
        end,
        on_timer = function(pos, elapsed)
            local node = minetest.get_node({x = pos.x, y = pos.y - 1, z = pos.z})
            if node.name == "default:water_source" or node.name == "nodes_nature:salt_water_source" or
                node.name == "nodes_nature:salt_water_flowing" or node.name ==
                "nodes_nature:freshwater_source" then
                minetest.set_node(pos, {name = "main:sea_beans_fruit"})
            else
                minetest.set_node(pos, {name = "nodes_nature:salt_water_flowing"})
            end
        end,
    }
)

minetest.register_node(
    "main:sea_beans_fruit", {

        drawtype = "plantlike",
        tiles = {"sea_beans_fruit.png"},
        inventory_image = "sea_beans_fruit.png",
        wield_image = "sea_beans_fruit.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 1, flammable = 30, falling_node = 0, flora = 1},

        on_punch = function(pos, node, player, pointed_thing)
            globo.add_nutrient(player, "hunger", 100)
            minetest.set_node(pos, {name = "main:sea_beans"})
        end,
    }
)

-- Gumboline: anything that moves onto this is launched in the air in the direction it was going. 
-- Is breakable so you can pick them up and place them in different places
-- grows in the highlands
minetest.register_node(
    "main:gumboline", {
        tiles = {"gumboline.png"},
        drawtype = "nodebox",
        paramtype = "light",
        node_box = {type = "fixed", fixed = {-0.5, -0.5, -0.5, 0.5, 0, 0.5}},
        sunlight_propagates = true,
        walkable = true,
        groups = {snappy = 3, flammable = 0, falling_node = 1},

    }
)

-- Descor: a bush that grows into a small tree producing a few gourds. The gourds give you a lot of food but make you very thirsty
-- If you plant the gourds they will sprout into new trees
minetest.register_node(
    "main:descor", {
        drawtype = "plantlike",
        tiles = {"descor.png"},
        inventory_image = "descor.png",
        wield_image = "descor.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {snappy = 3, flammable = 30, falling_node = 1, flora = 1},
        on_construct = function(pos)
            local ft = math.random(150, 450) + math.random(150, 450)
            if minetest.get_node(pos).param2 == 1 then
                ft = 1
            end
            local timer = minetest.get_node_timer(pos)
            timer:start(ft * globo.GAME_SPEED)
        end,
        on_timer = function(pos, elapsed)
            -- make sure conditions are correct
            local under = minetest.get_node({x = pos.x, y = pos.y - 1, z = pos.z})
            if under.name ~= "nodes_nature:grassland_soil" then
                return false
            end

            -- conditions are correct so sprout the full tree
            minetest.set_node(pos, {name = "nodes_nature:maraka_log"})
            pos.y = pos.y + 1
            if minetest.get_node(pos).name == "air" then
                minetest.set_node(pos, {name = "main:descor_tree"})
                pos.y = pos.y + 1
                if minetest.get_node(pos).name == "air" then
                    minetest.set_node(pos, {name = "main:descor_tree"})
                    pos.x = pos.x + 1
                    if minetest.get_node(pos).name == "air" then
                        minetest.set_node(pos, {name = "main:descor_fruit"})
                    end
                    pos.x = pos.x - 2
                    if minetest.get_node(pos).name == "air" then
                        minetest.set_node(pos, {name = "main:descor_fruit"})
                    end
                    pos.x = pos.x + 1
                    pos.z = pos.z + 1
                    if minetest.get_node(pos).name == "air" then
                        minetest.set_node(pos, {name = "main:descor_fruit"})
                    end
                    pos.z = pos.z - 2
                    if minetest.get_node(pos).name == "air" then
                        minetest.set_node(pos, {name = "main:descor_fruit"})
                    end
                end
            end
        end,
    }
)

minetest.register_node(
    "main:descor_fruit", {
        drawtype = "plantlike",
        tiles = {"descor_fruit.png"},
        inventory_image = "descor_fruit.png",
        visual_scale = 1,
        paramtype2 = "meshoptions",
        place_param2 = 1,
        stack_max = 2,
        selection_box = {type = "fixed", fixed = {-0.1, -0.1, -0.1, 0.1, 0.5, 0.1}},
        wield_image = "descor_fruit.png",
        paramtype = "light",
        sunlight_propagates = true,
        walkable = false,
        groups = {
            dig_immediate = 3,
            flammable = 50,
            leafdecay = 3,
            falling_node = 1,
            leafdecay_drop = 1,
        },
        on_use = function(itemstack, user, pointed_thing)
            globo.add_nutrient(user, "thirst", -90)
            globo.add_nutrient(user, "hunger", 110)
            itemstack:take_item()
            return itemstack
        end,
        on_place = function(itemstack, placer, pointed_thing)
            if pointed_thing.type ~= "node" then
                return itemstack
            end

            local pos = pointed_thing.above
            local node_name = minetest.get_node_or_nil(pos).name
            if node_name ~= "air" then
                return itemstack
            end

            -- Place the node
            minetest.set_node(pos, {name = "main:descor"})

            itemstack:take_item()
            return itemstack
        end,
    }
)

minetest.register_node(
    "main:descor_tree", {
        on_place = minetest.rotate_node,
        tiles = {
            "nodes_nature_kagum_tree_top.png", "nodes_nature_kagum_tree_top.png",
            "nodes_nature_kagum_tree.png",
        },
        paramtype = "light",
        sunlight_propagates = true,
        walkable = true,
        drop = "main:descor_log",
        groups = {snappy = 3, flammable = 30, falling_node = 0, flora = 1},
    }
)

-- dropped log
minetest.register_node(
    "main:descor_log", {
        tiles = {
            "nodes_nature_kagum_log_top.png", "nodes_nature_kagum_log_top.png",
            "nodes_nature_kagum_log.png",
        },
        stack_max = minimal.stack_max_bulky,
        drawtype = "nodebox",
        paramtype = "light",
        node_box = {
            type = "fixed",
            fixed = {
                {-0.4375, -0.5, -0.4375, 0.4375, 0.5, 0.4375},
                {-0.375, -0.5, 0.4375, 0.375, 0.5, 0.5}, {-0.375, -0.5, -0.5, 0.375, 0.5, -0.4375},
                {0.4375, -0.5, -0.375, 0.5, 0.5, 0.375}, {-0.5, -0.5, -0.375, -0.4375, 0.5, 0.375},
            },
        },
        paramtype2 = "facedir",
        is_ground_content = false,
        groups = {log = 1, choppy = 1, flammable = 40},
        on_place = minetest.rotate_node,
    }
)

