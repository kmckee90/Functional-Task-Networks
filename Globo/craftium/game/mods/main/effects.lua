function globo.node_collisions(entity, velocity)
    local p = entity:get_pos()
    p.y = p.y - .25

    local node = minetest.get_node(p)

    if node.name == "main:gumboline" then
        velocity.x = velocity.x * 3
        velocity.y = 8
        velocity.z = velocity.z * 3
        -- minimal.log("on gumboline!".."x:"..velocity.x.."y:"..velocity.y.."z:"..velocity.z)
        entity:set_velocity(velocity)
    end

end

function globo.create_explosion(pos, radius, damage)
    -- Set the node at the given position
    minetest.set_node(pos, {name = "main:explosion"})

    -- Get the meta of the node at the position
    local meta = minetest.get_meta(pos)

    -- Set custom metadata
    meta:set_int("radius", radius)
    meta:set_int("damage", damage)
    local timer = minetest.get_node_timer(pos)
    timer:start(1)
end

-- Register the explosion node
minetest.register_node(
    "main:explosion", {
        description = "Explosion Node",
        drawtype = "plantlike",
        tiles = {"explosion.png"},
        light_source = minetest.LIGHT_MAX, -- Max light level
        walkable = false,
        pointable = false,
        diggable = false,
        buildable_to = true,
        groups = {not_in_creative_inventory = 1},
        on_timer = function(pos, elapsed)
            -- Handle explosion effects
            local meta = minetest.get_meta(pos)
            local radius = meta:get_int("radius")
            local damage = meta:get_int("damage")

            -- Hurt nearby creatures
            local objects = minetest.get_objects_inside_radius(pos, radius)
            for _, obj in ipairs(objects) do
                if obj:is_player() then
                    globo.add_hp(obj, -damage)
                elseif obj:get_luaentity() ~= nil then
                    local mob = obj:get_luaentity()
                    mobkit.hurt(mob, damage)
                end
            end

            -- Kill nearby plants
            local minp = vector.subtract(pos, radius)
            local maxp = vector.add(pos, radius)
            local nodes = minetest.find_nodes_in_area(
                minp, maxp, {"group:flora", "group:snappy", "group:flammable"}
            )
            for _, node_pos in ipairs(nodes) do
                minetest.remove_node(node_pos)
            end

            -- Create particle effect (optional)
            minetest.add_particlespawner(
                {
                    amount = 100,
                    time = 0.1,
                    minpos = vector.subtract(pos, 0.5),
                    maxpos = vector.add(pos, 0.5),
                    minvel = {x = -10, y = -10, z = -10},
                    maxvel = {x = 10, y = 10, z = 10},
                    minacc = {x = 0, y = 0, z = 0},
                    maxacc = {x = 0, y = 0, z = 0},
                    minexptime = 0.5,
                    maxexptime = 1,
                    minsize = 1,
                    maxsize = 3,
                    texture = "ember.png",
                }
            )
            -- Remove the node after effects
            minetest.remove_node(pos)
        end,
    }
)

