local target = nil

local function nearest(pos)
    local nearby_objects = minetest.get_objects_inside_radius(pos, 10)
    -- minetest.chat_send_all("objs:"..dump(nearby_objects))
    local cobj = nil
    local dist = 64
    for _, obj in ipairs(nearby_objects) do
        local luaent = obj:get_luaentity()
        -- minetest.chat_send_all("luaent:"..luaent.name)
        if mobkit.is_alive(obj) and not obj:is_player() and luaent then
            local opos = obj:get_pos()
            local odist = math.abs(opos.x - pos.x) + math.abs(opos.z - pos.z)
            if odist < dist then
                dist = odist
                cobj = obj
            end
        end
    end
    return cobj
end

minetest.register_chatcommand(
    "mark", {
        description = "Mark the mob under your cursor as the target",
        params = "",
        privs = {fly = true}, -- Privileges required to use the command
        func = function(name, param)
            local player = minetest.get_player_by_name(name)
            local dir = player:get_look_dir()
            local pos = player:get_pos()
            local newpos = {x = pos.x + dir.x, y = pos.y + dir.y, z = pos.z + dir.z}

            local entity = nearest(newpos)
            if entity ~= nil then
                target = entity:get_luaentity()
                globo.chat_send_player(name, "Marking" .. target.name)
                return
            end

            globo.chat_send_player(name, "Nothing found to mark")
        end,
    }
)

minetest.register_chatcommand(
    "d", {
        description = "Dump stats of target mob",
        params = "",
        privs = {fly = true},
        func = function(name, param)
            if target then
                local h = mobkit.recall(target, 'hibernate')
                local s = mobkit.recall(target, 'sexual')
                local p = mobkit.recall(target, 'pregnant')
                local a = mobkit.recall(target, 'action')
                -- local yaw = target.object:get_yaw()
                globo.chat_send_player(name, target.name)
                globo.chat_send_player(name, "HP:" .. target.hp .. "/" .. target.max_hp)
                globo.chat_send_player(name, "Energy:" .. mobkit.recall(target, 'energy'))
                globo.chat_send_player(name, "Age:" .. mobkit.recall(target, 'age'))
                globo.chat_send_player(name, "on_ground:" .. dump(target.isonground))
                if h ~= nil then
                    globo.chat_send_player(name, "Hiber:" .. dump(h))
                end
                if s ~= nil then
                    globo.chat_send_player(name, "Sexual:" .. dump(s))
                end
                if p ~= nil then
                    globo.chat_send_player(name, "Preg:" .. dump(p))
                end
                if a ~= nil then
                    globo.chat_send_player(name, "Action:" .. a)
                end
                -- globo.chat_send_player(name,"Yaw:"..yaw)
                -- globo.chat_send_player(name,"HQ:"..dump(target.hqueue))
                -- globo.chat_send_player(name,"LQ:"..dump(target.lqueue))
                globo.chat_send_player(name, "Pri:" .. mobkit.get_queue_priority(target))

            else
                globo.chat_send_player(name, "No target set use /mark")
            end
        end,
    }
)

minetest.register_chatcommand(
    "go", {
        description = "Makes the marked mob go to the space under the player",
        params = "",
        privs = {fly = true},
        func = function(name, param)
            if target then
                local player = minetest.get_player_by_name(name)
                local pos = player:get_pos()
                local newpos = globo.get_pos_ground(pos)
                if newpos then
                    -- mobkit.clear_queue_high(target)
                    -- mobkit.clear_queue_low(target)
                    -- mobkit.clear_queue(target)
                    mobkit.hq_goto(target, 90, newpos)
                    globo.chat_send_player(
                        name, "Mob Going to " .. newpos.x .. "," .. newpos.y .. "," .. newpos.z
                    )
                else
                    globo.chat_send_player(name, "No pos found")
                end
            else
                globo.chat_send_player(name, "No target set use /mark")
            end
        end,
    }
)

minetest.register_chatcommand(
    "mob", {
        description = "create a mob in front of you",
        params = "<mob name>",
        privs = {fly = true},
        func = function(name, param)
            local player = minetest.get_player_by_name(name)
            local dir = player:get_look_dir()
            local pos = player:get_pos()
            local newpos = {x = pos.x + dir.x, y = pos.y + dir.y, z = pos.z + dir.z}
            minimal.log("mob:" .. param)
            -- minetest.add_entity(newpos, "animals:"..param)
            animals.hatch_egg(newpos, 'air', 'air', "animals:" .. param, 2000, 1)
            local entity = nearest(newpos)
            if entity ~= nil then
                target = entity:get_luaentity()
                globo.chat_send_player(name, "Marking " .. target.name)
                return
            else
                globo.chat_send_player(name, "Failed to create " .. param)
            end
        end,
    }
)

minetest.register_chatcommand(
    "max", {
        description = "give you the max stats again",
        params = "<mob name>",
        privs = {fly = true},
        func = function(name, param)
            local player = minetest.get_player_by_name(name)
            local meta = player:get_meta()
            meta:set_int("hunger", 1000)
            meta:set_int("thirst", 1000)
            meta:set_int("energy", 1000)
            globo.add_hp(player, 1000)
        end,
    }
)

minetest.register_chatcommand(
    "nstat", {
        description = "give you stats of the node in front of you",
        params = "",
        privs = {fly = true},
        func = function(name, param)
            local nodePos

            if param ~= "" then
                nodePos = minimal.str_to_pos(param)
            else
                -- get the node in front of the player
                local player = minetest.get_player_by_name(name)
                local dir = player:get_look_dir()
                local pos = player:get_pos()

                nodePos = {x = pos.x + dir.x, y = pos.y, z = pos.z + dir.z}
            end

            local node = minetest.get_node(nodePos)
            minimal.log(
                "node:" .. node.name .. " p1:" .. dump(node.param1) .. " p2:" .. dump(node.param2)
            )
            local node_def = minetest.registered_nodes[node.name]
            if node_def and node_def.groups then
                for group, value in pairs(node_def.groups) do
                    minimal.log(group .. ":" .. dump(value))
                end
            else
                minimal.log("no groups")
            end
        end,
    }
)

minetest.register_chatcommand(
    "pop", {
        description = "Show population of each type of mob",
        params = "<mob type>",
        privs = {fly = true},
        func = function(name, param)
            -- if <mob type> is set then show the stats for each mob that exists of that type
            minimal.log("pop: " .. param)
            local nearby_objects = minetest.get_objects_inside_radius({x = 0, y = 0, z = 0}, 1000)
            if param == "" then
                local pop = {}
                for _, obj in ipairs(nearby_objects) do
                    local luaent = obj:get_luaentity()
                    if mobkit.is_alive(obj) and not obj:is_player() and luaent then
                        local name = luaent.name
                        if pop[name] == nil then
                            pop[name] = 1
                        else
                            pop[name] = pop[name] + 1
                        end
                    end
                end
                for k, v in pairs(pop) do
                    minimal.log(k .. ":" .. v)
                end
            else
                for _, obj in ipairs(nearby_objects) do
                    local luaent = obj:get_luaentity()
                    if mobkit.is_alive(obj) and not obj:is_player() and luaent then
                        local name = luaent.name
                        if name == "animals:" .. param then
                            minimal.log(
                                " hp:" .. luaent.hp .. " energy:" .. mobkit.recall(luaent, 'energy') ..
                                    " age:" .. mobkit.recall(luaent, 'age')
                            )
                        end
                    end
                end
            end
        end,
    }
)

minetest.register_chatcommand(
    "test", {
        description = "run test functions",
        params = "",
        privs = {fly = true},
        func = function(name, param)
            -- tanglerTest()

            local player = minetest.get_player_by_name(name)
            local properties = player:get_properties()
            local max_hp = properties.hp_max

            local physics = player:get_physics_override()
            local speed = physics.speed
            local jump = physics.jump
            local meta = player:get_meta()
            minimal.log(
                "player maxhp" .. max_hp .. " speed:" .. speed .. " jump:" .. jump .. " time:" ..
                    meta:get_float("time")
            )

            local dir = player:get_look_dir()
            local pos = player:get_pos()

            local nodePos = {x = pos.x + dir.x, y = pos.y, z = pos.z + dir.z}

            local ratio = player:get_day_night_ratio()
            if ratio then
                minimal.log("ratio: " .. player:get_day_night_ratio())
            end
            local tod = minetest.get_timeofday()
            minimal.log("tod: " .. tod)

            -- grib_spread(nodePos)
        end,
    }
)

minetest.register_chatcommand(
    "egg", {
        description = "Drop an egg",
        params = "",
        privs = {},
        func = function(name, param)
            local player = minetest.get_player_by_name(name)
            globo.lay_egg(player)
        end,
    }
)

minetest.register_chatcommand(
    "sleep", {
        description = "Go to Sleep",
        params = "",
        privs = {},
        func = function(name, param)
            local player = minetest.get_player_by_name(name)
            globo.player_sleep(player)
        end,
    }
)

minetest.register_chatcommand(
    "node", {
        description = "create a node below you",
        params = "<node name>",
        privs = {fly = true},
        func = function(name, param)
            local player = minetest.get_player_by_name(name)
            local pos = player:get_pos()
            local newpos = globo.get_pos_ground(pos)
            assert(newpos ~= nil, "no ground below me")
            minimal.log(
                "node:" .. param .. " at " .. "(" .. newpos.x .. "," .. newpos.y .. "," .. newpos.z ..
                    ")"
            )
            minetest.set_node(newpos, {name = "main:" .. param})
        end,
    }
)

minetest.register_chatcommand(
    "oid", {
        params = "<object_id>",
        description = "Identify an object by its ID",
        privs = {fly = true},
        func = function(name, param)
            local object_id = tonumber(param)
            if not object_id then
                return false, "Invalid object ID. Please enter a numeric ID."
            end

            -- Iterate over all objects in the world
            for _, obj in pairs(minetest.get_objects_inside_radius({x = 0, y = 0, z = 0}, 32000)) do
                if obj:get_id() == object_id then
                    local luaentity = obj:get_luaentity()
                    if luaentity then
                        -- If the object is a Lua entity, provide its type
                        return true,
                            "Object ID " .. object_id .. " is a LuaEntity: " .. luaentity.name
                    elseif obj:is_player() then
                        -- If the object is a player, provide the player's name
                        return true, "Object ID " .. object_id .. " is a Player: " ..
                            obj:get_player_name()
                    end
                end
            end

            -- If no object with that ID is found
            return false, "No object found with ID " .. object_id
        end,
    }
)

