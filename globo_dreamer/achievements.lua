-- Opt-in evaluation only. Loaded in the generated game at process launch.
-- Count successful callbacks, never button presses or passive rain hydration.
local counts, details, diagnostics = {}, {}, {}
local function player_ok(player)
    return player and player.is_player and player:is_player()
end
local function write()
    minetest.safe_file_write(minetest.get_worldpath() .. '/agent_achievements.json',
        minetest.write_json({version=2, counts=counts, details=details, diagnostics=diagnostics}))
end
local function record(kind, detail, player)
    if not player_ok(player) then return end
    counts[kind] = (counts[kind] or 0) + 1
    if detail then
        local key = kind .. '/' .. detail
        details[key] = (details[key] or 0) + 1
    end
    write()
end
write()
minetest.register_on_dignode(function(pos, node, digger)
    record('dig_block', node.name, digger)
end)
minetest.register_on_placenode(function(pos, node, placer)
    record('place_block', node.name, placer)
end)
minetest.register_on_mods_loaded(function()
    local lay = globo.lay_egg
    globo.lay_egg = function(player)
        local success = lay(player)
        diagnostics.egg_attempts=(diagnostics.egg_attempts or 0)+1
        diagnostics.egg_successes=(diagnostics.egg_successes or 0)+(success and 1 or 0)
        if success then record('lay_egg', nil, player) end
        if not success then write() end
        return success
    end
    -- Direct killing blows only: no credit for despawn, age, or other animals.
    for _, method in ipairs({'on_punch', 'on_punch_water'}) do
        local original = animals[method]
        if original then
            animals[method] = function(self, caps, puncher, ...)
                local before = self.hp or 0
                local result = original(self, caps, puncher, ...)
                if before > 0 and (self.hp or 0) <= 0 and not self._achievement_killed then
                    self._achievement_killed = true
                    local species = ({['animals:wolf']='wolf', ['animals:wolf_male']='wolf',
                        ['animals:cow']='cow', ['animals:cow_male']='cow'})[self.name]
                    record(species and ('kill_' .. species) or 'kill_other', self.name, puncher)
                end
                return result
            end
        end
    end
    local name = 'nodes_nature:freshwater_source'
    local drink = minetest.registered_nodes[name].on_rightclick
    minetest.override_item(name, {on_rightclick=function(pos, node, player, ...)
        local before = player_ok(player) and player:get_meta():get_int('thirst')
        local result = drink(pos, node, player, ...)
        if before and player:get_meta():get_int('thirst') > before then
            record('drink_water', 'freshwater', player)
        end
        return result
    end})
    local drop = minetest.registered_entities['nodes_nature:drop_water']
    local drink_drop = drop.on_punch
    drop.on_punch = function(self, player, ...)
        local before = player_ok(player) and player:get_meta():get_int('thirst')
        local result = drink_drop(self, player, ...)
        if before and player:get_meta():get_int('thirst') > before then
            record('drink_water', 'water_drop', player)
        end
        return result
    end
end)

-- Additional outcome hooks. No reward shaping or changes to game mechanics.
minetest.register_on_mods_loaded(function()
    local function wrap_node(name, method, after, before)
        local def = minetest.registered_items[name]
        local original = def and def[method]
        if not original then return end
        minetest.override_item(name, {[method]=function(...)
            local args = {...}
            local snapshot = before and before(unpack(args))
            local result = original(unpack(args))
            after(snapshot, result, unpack(args))
            return result
        end})
    end
    local function inventory_count(player, name)
        local count = 0
        for _, stack in ipairs(player:get_inventory():get_list('main') or {}) do
            if stack:get_name() == name then count = count + stack:get_count() end
        end
        return count
    end
    for name, kind in pairs({['main:potatoes']='harvest_potatoes', ['main:papaya_fruit']='harvest_papaya'}) do
        wrap_node(name, 'on_punch', function(before, result, pos, node, player)
            if player_ok(player) and inventory_count(player,name) > before then record(kind,name,player) end
        end, function(pos,node,player) return inventory_count(player,name) end)
    end
    for name, kind in pairs({['main:thorns_fruit']='eat_thorn_fruit', ['main:dawn_cactus_fruit']='eat_dawn_cactus'}) do
        wrap_node(name, 'on_punch', function(before,result,pos,node,player)
            if player_ok(player) and minetest.get_node(pos).name ~= name then
                record(kind,name,player); record('eat_food',name,player)
            end
        end)
    end
    wrap_node('main:coffee','on_punch',function(before,result,pos,node,player)
        if player:get_meta():get_float('energy') > before then record('coffee_energy',nil,player) end
    end,function(pos,node,player) return player:get_meta():get_float('energy') end)
    wrap_node('main:goldcap','on_use',function(before,result,stack,player)
        if player:get_hp() > before then record('heal_goldcap',nil,player) end
    end,function(stack,player) return player:get_hp() end)
    wrap_node('main:player_egg','on_timer',function(before)
        if before.player and before.player:get_meta():get_int('score') > before.score then
            record('egg_mature',nil,before.player)
        end
    end,function(pos)
        local p=minetest.get_player_by_name(minetest.get_meta(pos):get_string('owner'))
        return {player=p,score=p and p:get_meta():get_int('score') or 0}
    end)
    -- Item consumption is confirmed by stack consumption, not just clicking food.
    for name,def in pairs(minetest.registered_items) do
        if def.on_use and ((def.groups or {}).edible or name=='main:potatoes' or name=='main:papaya_fruit') then
            wrap_node(name,'on_use',function(before,result,stack,player)
                local remaining=result and result:get_count() or stack:get_count()
                if player_ok(player) and remaining < before then record('eat_food',name,player) end
            end,function(stack) return stack:get_count() end)
        end
    end
    local sleep=globo.player_sleep
    globo.player_sleep=function(player,...)
        local before=player:get_meta():get_int('sleeping')
        local result=sleep(player,...)
        if before~=1 and player:get_meta():get_int('sleeping')==1 then record('sleep',nil,player) end
        return result
    end
end)

-- Read-only milestones sampled in game time. Once per life unless stated otherwise.
local observations, elapsed = {}, 0
minetest.register_globalstep(function(dt)
    elapsed=elapsed+dt
    for _,p in ipairs(minetest.get_connected_players()) do
        local id=p:get_player_name(); local pos=p:get_pos(); local meta=p:get_meta()
        local s=observations[id]
        if not s or (s.dead and p:get_hp()>0) then
            s={y=pos.y,flags={},energy=meta:get_float('energy'),hp=p:get_hp()};observations[id]=s
        end
        s.dead=p:get_hp()<=0
        if not s.dead then
            local inventory={}
            for _,stack in ipairs(p:get_inventory():get_list('main') or {}) do
                local name=stack:get_name()
                if name~='' then inventory[name]=(inventory[name] or 0)+stack:get_count() end
            end
            if s.inventory then
                for name,count in pairs(inventory) do
                    if count>(s.inventory[name] or 0) and not globo.lay_egg_is_item_name(name) then
                        record('collect_resource',name,p)
                    end
                end
            end
            s.inventory=inventory
            for _,height in ipairs({5,10,20}) do
                local key='climb_'..height..'m'
                if pos.y-s.y>=height and not s.flags[key] then record(key,nil,p);s.flags[key]=true end
            end
            local node=minetest.get_node(pos)
            local wet=minetest.get_item_group(node.name,'water')>0
            if wet and not s.water then s.water={x=pos.x,z=pos.z} end
            if s.water and not wet then
                if (pos.x-s.water.x)^2+(pos.z-s.water.z)^2>=25 then record('cross_water',nil,p) end
                s.water=nil
            end
            local biome=minetest.get_biome_data(pos)
            if biome then
                local name=minetest.get_biome_name(biome.biome)
                if name and not s.flags['biome/'..name] then
                    record('visit_biome',name,p);s.flags['biome/'..name]=true
                end
            end
            if meta:get_int('sleeping')==1 then
                for key,increased in pairs({rest_energy=meta:get_float('energy')>s.energy,rest_health=p:get_hp()>s.hp}) do
                    if increased and not s.flags[key] then record(key,nil,p);s.flags[key]=true end
                end
            end
            if s.attack and elapsed-s.attack>=30 then record('survive_predator',nil,p);s.attack=nil end
        else s.attack=nil end
        s.energy=meta:get_float('energy');s.hp=p:get_hp()
    end
end)
minetest.register_on_player_hpchange(function(player,change,reason)
    if change>=0 or not reason or not reason.object or not reason.object.get_luaentity then return end
    local entity=reason.object:get_luaentity()
    if entity and entity.name and (entity.name:find('wolf') or entity.name:find('puma')) then
        local s=observations[player:get_player_name()]
        if s then s.attack=elapsed end
    end
end,false)
