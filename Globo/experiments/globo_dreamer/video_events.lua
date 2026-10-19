-- Evaluation-only event hooks, enabled solely for selected capture runs.
local events, sequence, clock, players = {}, 0, 0, {}
local function emit(kind,detail,p)
    sequence=sequence+1
    events[#events+1]={id=sequence,kind=kind,detail=detail,game_clock=clock,position=p:get_pos()}
    if #events>128 then table.remove(events,1) end
    minetest.safe_file_write(minetest.get_worldpath()..'/agent_video_events.json',minetest.write_json({events=events}))
end
local function is_tree(name)
    return minetest.get_item_group(name,'tree')>0 or minetest.get_item_group(name,'leaves')>0
end
local function is_ground(name)
    if is_tree(name) then return false end
    local def=minetest.registered_nodes[name]
    if not def or not def.walkable then return false end
    return minetest.get_item_group(name,'sediment')>0 or minetest.get_item_group(name,'crumbly')>0
        or minetest.get_item_group(name,'stone')>0 or minetest.get_item_group(name,'soil')>0
        or name:find('rock')~=nil or name:find('stone')~=nil
end
local valuable={['main:goldcap']=true,['main:coffee']=true}
minetest.register_on_mods_loaded(function()
    for _,tree in ipairs(nodes_nature.tree_list or {}) do
        if tree[3] then valuable['nodes_nature:'..tree[3]]=true end
    end
    for name,def in pairs(minetest.registered_items) do
        local g=def.groups or {};local food=(HEALTH.food_table or {})[name]
        if name:find('fruit') or name:find('berry') or name:find('_nut')
            or (g.rare or 0)>0 or (g.precious or 0)>0 or (food and (food[1] or 0)>0) then valuable[name]=true end
    end
end)
minetest.register_on_dignode(function(pos,node,p)
    if not p or not p:is_player() or not is_ground(node.name) then return end
    local s=players[p:get_player_name()];if not s then return end
    local feet=p:get_pos()
    if pos.y>feet.y+.5 then return end
    local digs={};local connected=false
    for _,d in ipairs(s.digs or {}) do
        if clock-d.time<10 and vector.distance(d.pos,pos)<4 then
            digs[#digs+1]=d
            local distance=math.abs(d.pos.x-pos.x)+math.abs(d.pos.y-pos.y)+math.abs(d.pos.z-pos.z)
            if distance>0 and distance<=2 then connected=true end
        end
    end
    digs[#digs+1]={pos=vector.new(pos),time=clock};s.digs=digs
    if connected and clock-(s.hole_time or -100)>7.5 then
        emit('dig_hole',node.name,p);s.hole_time=clock
    end
end)
minetest.register_globalstep(function(dt)
    clock=clock+dt
    for _,p in ipairs(minetest.get_connected_players()) do
        local id=p:get_player_name();local pos=p:get_pos();local s=players[id]
        if not s or (s.dead and p:get_hp()>0) then s={history={},inventory=nil};players[id]=s end
        s.dead=p:get_hp()<=0
        if not s.dead then
            local inventory={}
            for _,stack in ipairs(p:get_inventory():get_list('main') or {}) do
                local n=stack:get_name();if valuable[n] then inventory[n]=(inventory[n] or 0)+stack:get_count() end
            end
            if s.inventory then
                for name,n in pairs(inventory) do
                    if n>(s.inventory[name] or 0) then emit('collect_valuable',name,p) end
                end
            end
            s.inventory=inventory
            -- Require substantial ascent from actual nearby ground, not low roots.
            local support=nil
            local ground=false
            for _,dx in ipairs({-.25,0,.25}) do
                for _,dz in ipairs({-.25,0,.25}) do
                    local n=minetest.get_node({x=pos.x+dx,y=pos.y-.35,z=pos.z+dz}).name
                    if is_tree(n) then support=n end
                    if is_ground(n) then ground=true end
                end
            end
            if not s.sample or clock-s.sample>=.2 then
                s.sample=clock
                if ground and not support then
                    s.history[#s.history+1]={time=clock,y=pos.y,x=pos.x,z=pos.z}
                end
                while #s.history>0 and clock-s.history[1].time>30 do table.remove(s.history,1) end
            end
            if support then
                s.contact=s.contact or clock
                local low=pos.y
                for _,v in ipairs(s.history) do
                    if (v.x-pos.x)^2+(v.z-pos.z)^2<=36 then low=math.min(low,v.y) end
                end
                -- Ground beneath the supporting tree must also be at least five
                -- blocks lower: a hillside walk ending on a root does not qualify.
                local ground_y=nil
                for dy=1,24 do
                    local below={x=pos.x,y=pos.y-dy,z=pos.z}
                    if is_ground(minetest.get_node(below).name) then ground_y=below.y+.5;break end
                end
                local high=pos.y-low>=5 and ground_y and pos.y-ground_y>=5
                if high then s.high_contact=s.high_contact or clock else s.high_contact=nil end
                if not s.tree_captured and s.high_contact and clock-s.high_contact>=1 then
                    emit('climb_tree',support,p);s.tree_captured=true
                end
                s.off_tree=nil
            else
                s.contact=nil;s.high_contact=nil;s.off_tree=s.off_tree or clock
                if clock-s.off_tree>5 then s.tree_captured=false end
            end
        end
    end
end)
