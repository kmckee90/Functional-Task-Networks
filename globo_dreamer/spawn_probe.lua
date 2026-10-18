-- Diagnostic-only local geometry. Never exposed to the learning observation.
local audit_clock = 0
local audited = false
minetest.register_globalstep(function(dt)
    audit_clock = audit_clock + dt
    if audited or audit_clock < 10 then return end
    local player = minetest.get_connected_players()[1]
    if not player then return end
    local nodes = {}
    local unknown = 0
    for x = -4, 20 do
        for z = 103, 127 do
            for y = 0, 14 do
                local pos = {x=x, y=y, z=z}
                local node = minetest.get_node_or_nil(pos)
                if not node or node.name == "ignore" then
                    unknown = unknown + 1
                elseif node.name ~= "air" then
                    local def = minetest.registered_nodes[node.name] or {}
                    nodes[#nodes+1] = {x=x, y=y, z=z, name=node.name,
                        damage=def.damage_per_second or 0, walkable=def.walkable ~= false}
                end
            end
        end
    end
    minetest.safe_file_write(minetest.get_worldpath() .. "/spawn_audit.json",
        minetest.write_json({nodes=nodes, unknown=unknown, player=player:get_pos(),
            spawn={x=8,y=8,z=115}, elapsed=audit_clock}))
    audited = true
end)
