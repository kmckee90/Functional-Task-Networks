"""Run on pod with isolated testdeps/lupa; exercises Lua hooks without a game."""
from pathlib import Path
from lupa import LuaRuntime

lua=LuaRuntime()
lua.execute('''
events_saved={}; player={pos={x=0,y=0,z=0},items={}}
function player:get_player_name() return 'agent' end
function player:is_player() return true end
function player:get_pos() return self.pos end
function player:get_hp() return 20 end
function player:get_inventory() return {get_list=function() return player.items end} end
vector={new=function(p) return {x=p.x,y=p.y,z=p.z} end,
distance=function(a,b) return math.sqrt((a.x-b.x)^2+(a.y-b.y)^2+(a.z-b.z)^2) end}
nodes_nature={tree_list={{'maraka','tree','maraka_nut'}}};HEALTH={food_table={['test:medicine']={2}}}
minetest={registered_items={['test:medicine']={}},registered_nodes={ground={walkable=true},tree={walkable=true},leaves={walkable=true}},
get_worldpath=function() return '/tmp' end,write_json=function(v) return v end,
safe_file_write=function(path,v) events_saved=v.events end,
register_on_mods_loaded=function(f) mods=f end,register_on_dignode=function(f) dig=f end,
register_globalstep=function(f) tick=f end,get_connected_players=function() return {player} end,
get_node=function(p) return {name=(p.y<.5 and 'ground') or support or 'ground'} end,
get_item_group=function(name,group)
 if (name=='tree' and group=='tree') or (name=='leaves' and group=='leaves') or (name=='ground' and group=='crumbly') then return 1 end
 return 0 end}
''')
lua.execute(Path(__file__).with_name('video_events.lua').read_text())
lua.execute('''
mods();tick(.2)
-- Hill ascent alone must not count as climbing a tree.
player.pos.y=5;tick(.5);assert(#events_saved==0)
support='tree';player.pos.y=2;tick(.2);tick(1.1);assert(#events_saved==0)
support='leaves';player.pos.y=6;tick(.2);assert(#events_saved==0)
tick(1.1);assert(events_saved[1].kind=='climb_tree')
tick(.5);assert(#events_saved==1)
-- One ground removal, or leaf removal, is not a hole.
player.pos.y=0;dig({x=0,y=-1,z=0},{name='ground'},player);assert(#events_saved==1)
dig({x=1,y=-1,z=0},{name='leaves'},player);assert(#events_saved==1)
dig({x=1,y=-1,z=0},{name='ground'},player);assert(events_saved[2].kind=='dig_hole')
-- Fruit and medicinal items trigger only when inventory actually increases.
player.items={{get_name=function() return 'maraka_nut' end,get_count=function() return 1 end}}
tick(.2);assert(#events_saved==2)
player.items={{get_name=function() return 'nodes_nature:maraka_nut' end,get_count=function() return 1 end}}
tick(.2);assert(events_saved[3].kind=='collect_valuable')
tick(.2);assert(#events_saved==3)
player.items[#player.items+1]={get_name=function() return 'test:medicine' end,get_count=function() return 1 end}
tick(.2);assert(events_saved[4].kind=='collect_valuable')
''')
print('Lua tree, hole, fruit and medicinal collection checks passed')
