"""Pod-only isolated real-engine verification; never uses training worlds."""
import json
import os
from pathlib import Path
from .legacy.env import make_globo_env
ROOT=Path('/home/dev/globo_dreamer_state/diagnostics/achievement_v2')
ROOT.mkdir(parents=True,exist_ok=True)
env=make_globo_env(env_dir=str(ROOT/'template'),run_dir_prefix=str(ROOT/'engine'),
    action_mode='raw',obs_width=64,obs_height=64,seed=19,spawn='12,39.1,120',
    frameskip=8,init_frames=200,max_timesteps=None)
env.mt.proc_env.update(LIBGL_ALWAYS_SOFTWARE='1',GALLIUM_DRIVER='llvmpipe',LP_NUM_THREADS='2',
    SDL_HINT_EGL_DEVICE='0',SDL_VIDEO_EGL_DRIVER='/home/dev/globo_dreamer_state/software_egl.so',
    __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
control=Path(env.mt.run_dir)/'mods/craftium_env/init.lua'
with control.open('a') as f:
    f.write('\n'+Path(__file__).with_name('achievements.lua').read_text())
    f.write('''
local verified=false;local probe_frames=0
minetest.register_globalstep(function()
    if verified then return end
    local p=minetest.get_connected_players()[1]
    if not p then return end
    probe_frames=probe_frames+1;if probe_frames<2 then return end
    verified=true
    local pos=p:get_pos();pos.y=pos.y+0.5
    minetest.load_area({x=pos.x-5,y=pos.y-5,z=pos.z-5},{x=pos.x+5,y=pos.y+5,z=pos.z+5})
    minetest.set_node(pos,{name='air'})
    local m=p:get_meta();m:set_int('hunger',1000);m:set_int('energy',1000);m:set_int('energy_max',1000)
    assert(globo.lay_egg(p),'probe egg failed')
    minetest.registered_nodes['main:player_egg'].on_timer(pos,600)
    local farm={x=pos.x+3,y=pos.y,z=pos.z}
    minetest.set_node(farm,{name='main:potatoes'})
    minetest.get_meta(farm):set_int('potato',1)
    minetest.registered_nodes['main:potatoes'].on_punch(farm,{},p,{})
    minetest.set_node(farm,{name='main:papaya_fruit'})
    minetest.registered_nodes['main:papaya_fruit'].on_punch(farm,{},p,{})
    minetest.set_node(farm,{name='main:thorns_fruit'})
    minetest.registered_nodes['main:thorns_fruit'].on_punch(farm,{},p,{})
    m:set_int('energy',100)
    minetest.registered_nodes['main:coffee'].on_punch(farm,{},p,{})
    p:set_hp(1)
    minetest.registered_nodes['main:goldcap'].on_use(ItemStack('main:goldcap'),p,{})
    minetest.registered_nodes['main:potatoes'].on_use(ItemStack('main:potatoes'),p,{})
    minetest.safe_file_write(minetest.get_worldpath()..'/probe_ok','ok')
end)
''')
try:
    env.reset()
    for _ in range(200):
        if (Path(env.mt.run_dir)/'worlds/world/probe_ok').exists():break
        env.step({})
    else:raise AssertionError('Probe did not execute')
    data=json.loads((Path(env.mt.run_dir)/'worlds/world/agent_achievements.json').read_text())
    print(json.dumps(data,indent=2),flush=True)
    for key in ('lay_egg','egg_mature','harvest_potatoes','harvest_papaya','eat_thorn_fruit','coffee_energy','heal_goldcap','eat_food'):
        assert data['counts'].get(key,0)>0,key
    assert data['counts']['lay_egg']==1
    print('REAL ENGINE ACHIEVEMENT CHECK PASSED',flush=True)
finally:
    env.close()
