-- Separate read-only mapper world: never loaded into a learning agent's world.
local root=minetest.settings:get('terrain_output')
local clock,busy=0,false
local ceilings={}
local function emit(tx,tz,ceiling)
    local x0,z0=tx*64,tz*64
    local vm=minetest.get_voxel_manip()
    local lo,hi=vm:read_from_map({x=x0,y=-64,z=z0},{x=x0+63,y=ceiling,z=z0+63})
    local area=VoxelArea:new({MinEdge=lo,MaxEdge=hi});local data=vm:get_data()
    local air=minetest.get_content_id('air');local ignore=minetest.get_content_id('ignore')
    local names,lookup,pixels={},{},{}
    local clipped=0
    for z=z0,z0+63,2 do
        for x=x0,x0+63,2 do
            local height,kind=nil,nil
            for y=ceiling,-64,-1 do
                local id=data[area:index(x,y,z)]
                if id~=air and id~=ignore then
                    local name=minetest.get_name_from_content_id(id)
                    local def=minetest.registered_nodes[name] or {}
                    if def.walkable~=false or (def.liquidtype and def.liquidtype~='none') or minetest.get_item_group(name,'leaves')>0 then
                        height=y;kind=name;if y==ceiling then clipped=clipped+1 end;break
                    end
                end
            end
            if kind then
                if not lookup[kind] then names[#names+1]=kind;lookup[kind]=#names end
                pixels[#pixels+1]={height,lookup[kind]}
            else pixels[#pixels+1]={-999,0} end
        end
    end
    if clipped>0 and ceiling<2048 then
        ceilings[tx..','..tz]=ceiling*2
        return
    end
    minetest.safe_file_write(root..'/tile_'..tx..'_'..tz..'.json',minetest.write_json({
        x=x0,z=z0,resolution=2,width=32,names=names,pixels=pixels,clipped=clipped}))
end
minetest.register_globalstep(function(dt)
    clock=clock+dt;if clock<1 or busy then return end;clock=0
    local f=io.open(root..'/request.json','r');if not f then return end
    local request=minetest.parse_json(f:read('*a'));f:close();if not request then return end
    for _,tile in ipairs(request.tiles or {}) do
        local tx,tz=tile[1],tile[2]
        local done=io.open(root..'/tile_'..tx..'_'..tz..'.json','r')
        if done then done:close() else
            busy=true
            local ceiling=ceilings[tx..','..tz] or 256
            minetest.emerge_area({x=tx*64,y=-64,z=tz*64},{x=tx*64+63,y=ceiling,z=tz*64+63},
                function(block,action,remaining)
                    if remaining==0 then
                        local ok,err=pcall(emit,tx,tz,ceiling)
                        if not ok then minetest.log('error','terrain probe: '..tostring(err)) end
                        busy=false
                    end
                end)
            return
        end
    end
end)
