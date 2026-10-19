function globo.clamp(value, min, max)
    if (value < min) then
        return min
    end
    if (value > max) then
        return max
    end
    return value
end

function globo.is_in_list(target, list)
    for _, value in ipairs(list) do
        if value == target then
            return true
        end
    end
    return false
end

function globo.is_dropped_item_in_group_at_pos(group, pos, radius)
    local objects = minetest.get_objects_inside_radius(pos, radius or 0.6) -- Adjust radius as needed
    for _, obj in ipairs(objects) do
        if obj:get_luaentity() and obj:get_luaentity().name == "__builtin:item" then
            local itemstack = ItemStack(obj:get_luaentity().itemstring)
            if minetest.get_item_group(itemstack:get_name(), group) > 0 then
                return true -- Found a dropped item in the group
            end
        end
    end
    return false -- No dropped item found in the group
end

function globo.get_pos_ground(pos)
    -- Create a raycast from the position downward
    local ray = minetest.raycast(pos, {x = pos.x, y = pos.y - 200, z = pos.z}, false, true)
    for pointed_thing in ray do
        if pointed_thing.type == "node" then
            local node_pos = pointed_thing.under
            return node_pos
        end
    end
    return nil -- If no ground found
end

function globo.get_random_position_around_pos(pos, min_distance, max_distance)
    local angle = math.random() * 2 * math.pi
    local distance = math.random(min_distance, max_distance)
    local retPos = {
        x = pos.x + distance * math.cos(angle),
        y = pos.y + 100,
        z = pos.z + distance * math.sin(angle),
    }
    local rPos = globo.get_pos_ground(retPos)
    if rPos then
        rPos.y = rPos.y + 1
        return rPos
    end
    retPos.y = pos.y
    return retPos
end

function globo.get_distance(pos1, pos2)
    local dx = pos2.x - pos1.x
    local dy = pos2.y - pos1.y
    local dz = pos2.z - pos1.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)
end

if globo.LOG_MESSAGES then
    globo.chat_send_player = minetest.chat_send_player
else
    globo.chat_send_player = function(player_name, message)
    end
end
