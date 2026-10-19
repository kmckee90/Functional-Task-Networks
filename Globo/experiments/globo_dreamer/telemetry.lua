-- Read-only bodily observations and diagnostic events for the Python adapter.
-- No shaping rewards, resource grants, teleportation, or changes to survival rules.
local elapsed = 0
local death_reason = ""
local last_damage_type = ""
local cumulative_damage = 0
local function emit(player)
    local meta = player:get_meta()
    local inv = player:get_inventory():get_list("main") or {}
    local items = {}
    local slots = {}
    for i, stack in ipairs(inv) do
        slots[i] = stack:get_count()
        if not stack:is_empty() then
            local name = stack:get_name()
            items[name] = (items[name] or 0) + stack:get_count()
        end
    end
    local record = {
        elapsed = elapsed, hp = player:get_hp(),
        hp_max = player:get_properties().hp_max or 20,
        hunger = meta:get_int("hunger"), thirst = meta:get_int("thirst"),
        energy = meta:get_int("energy"), energy_max = meta:get_int("energy_max"),
        temperature = meta:get_int("temperature"),
        temperature_external = meta:get_int("temperature_external"),
        breath = player:get_breath(), wield = player:get_wield_index(),
        items = items, slots = slots, death_reason = death_reason,
        last_damage_type = last_damage_type, cumulative_damage = cumulative_damage,
        look_pitch = player:get_look_vertical(), look_yaw = player:get_look_horizontal(),
    }
    minetest.safe_file_write(minetest.get_worldpath() .. "/agent_state.json",
        minetest.write_json(record))
end
minetest.register_on_player_hpchange(function(player, hp_change, reason)
    if hp_change < 0 then
        last_damage_type = reason and reason.type or "unknown"
        cumulative_damage = cumulative_damage - hp_change
    end
end, false)
minetest.register_on_dieplayer(function(player, reason)
    death_reason = reason and reason.type or "unknown"
    emit(player)
end)
minetest.register_globalstep(function(dtime)
    elapsed = elapsed + dtime
    for _, player in ipairs(minetest.get_connected_players()) do emit(player) end
end)
