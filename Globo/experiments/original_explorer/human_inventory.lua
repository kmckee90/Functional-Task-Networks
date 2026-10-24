-- Human-play diagnostics only; does not alter inventory or game mechanics.
local human_digs, human_places, human_elapsed = 0, 0, 0
minetest.register_on_dignode(function(_, _, digger)
    if digger and digger:is_player() then human_digs = human_digs + 1 end
end)
minetest.register_on_placenode(function(_, _, placer)
    if placer and placer:is_player() then human_places = human_places + 1 end
end)
minetest.register_globalstep(function(dt)
    human_elapsed = human_elapsed + dt
    if human_elapsed < 0.2 then return end
    human_elapsed = 0
    local p = minetest.get_connected_players()[1]
    if not p then return end
    local slots = {}
    for i, stack in ipairs(p:get_inventory():get_list("main") or {}) do
        local def = stack:get_definition()
        slots[i] = {slot=i, name=stack:get_name(), count=stack:get_count(),
            description=def and def.description or stack:get_name()}
    end
    local path = minetest.get_worldpath() .. "/human_inventory.json"
    local f = io.open(path .. ".tmp", "w")
    if f then
        f:write(minetest.write_json({slots=slots, selected=p:get_wield_index(),
            digs=human_digs, placements=human_places}))
        f:close()
        os.rename(path .. ".tmp", path)
    end
end)
