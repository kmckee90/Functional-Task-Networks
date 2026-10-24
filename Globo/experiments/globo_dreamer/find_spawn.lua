-- Screening only. Find safe solid ground; never loaded during agent training.
local started = false
local ceiling = globo_spawn_scan_ceiling or 60
minetest.register_globalstep(function(dt)
  if started or #minetest.get_connected_players() == 0 then return end
  started = true
  minetest.emerge_area({x=-16,y=-16,z=96}, {x=40,y=ceiling+20,z=144}, function(_,_,remaining)
    if remaining ~= 0 then return end
    local candidates = {}
    local function safe_air(x,y,z)
      local n=minetest.get_node({x=x,y=y,z=z})
      local d=minetest.registered_nodes[n.name]
      return d and n.name ~= 'ignore' and not d.walkable and
        (d.liquidtype == nil or d.liquidtype == 'none') and (d.damage_per_second or 0)==0
    end
    for x=-8,32,2 do for z=104,136,2 do
      for y=ceiling,-8,-1 do
        local n=minetest.get_node({x=x,y=y,z=z})
        local d=minetest.registered_nodes[n.name]
        if d and d.walkable and (d.damage_per_second or 0)==0 then
          local good=true
          for dx=-1,1 do for dz=-1,1 do
            if not safe_air(x+dx,y+1,z+dz) or not safe_air(x+dx,y+2,z+dz) then good=false end
          end end
          if good then
            candidates[#candidates+1]={x=x,y=y+1.1,z=z,ground=n.name,
              rank=(x-11)^2+(z-121)^2+0.1*(y-4)^2}
          end
          break
        end
      end
    end end
    table.sort(candidates,function(a,b) return a.rank<b.rank end)
    local out={}
    for i=1,math.min(12,#candidates) do out[i]=candidates[i] end
    minetest.safe_file_write(minetest.get_worldpath()..'/spawn_candidates.json',minetest.write_json(out))
  end)
end)
