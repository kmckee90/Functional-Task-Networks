"""Pod-only isolated engine test: selection and placement in a controlled scene."""
import json
import os
import tempfile
from pathlib import Path
from env import make_globo_env
from PIL import Image
from play import HumanFrameRepeat

LUA = r'''
local placed=0
minetest.register_on_placenode(function() placed=placed+1 end)
local prepared=false
minetest.register_globalstep(function()
 local p=minetest.get_connected_players()[1]
 if not p then return end
 if not prepared then
  prepared=true
  minetest.load_area({x=-6,y=109,z=-6},{x=6,y=117,z=6})
  for x=-5,5 do for z=-5,5 do
   minetest.set_node({x=x,y=110,z=z},{name="nodes_nature:sandstone"})
   for y=111,116 do minetest.set_node({x=x,y=y,z=z},{name="air"}) end
  end end
  p:set_pos({x=0,y=111,z=0});p:set_look_horizontal(0);p:set_look_vertical(math.rad(60))
  local inv=p:get_inventory()
  inv:set_stack("main",2,"nodes_nature:sand 10")
 end
 minetest.set_node({x=0,y=112,z=2},{name="nodes_nature:sandstone"})
 p:set_pos({x=0,y=110.5,z=0});p:set_look_horizontal(0);p:set_look_vertical(0)
 local inv=p:get_inventory()
 local f=io.open(PROBE_PATH,"w")
 f:write(minetest.write_json({wield=p:get_wield_index(),item=p:get_wielded_item():to_string(),
  pos=p:get_pos(),pitch=p:get_look_vertical(),yaw=p:get_look_horizontal(),
  ground=minetest.get_node({x=0,y=110,z=0}).name,
  slots=inv:get_size("main"),placed=placed,controls=p:get_player_control()}));f:close()
end)
'''


def main():
    with tempfile.TemporaryDirectory(prefix="globo-placement-") as d:
        root=Path(d)
        env=make_globo_env(env_dir=str(root/'template'),run_dir_prefix=str(root/'runs'),
            action_mode='raw',obs_width=64,obs_height=64,init_frames=200,
            spawn='0,111,0',seed=19,invincible=True,freeze_daylight=True,frameskip=1,
            extra_minetest_conf={"debug_log_level":"info","safe_dig_and_place":"false","repeat_place_time":"0.01"})
        env.mt.proc_env.update(LIBGL_ALWAYS_SOFTWARE='1',GALLIUM_DRIVER='llvmpipe',LP_NUM_THREADS='2',
            SDL_HINT_EGL_DEVICE='0',SDL_VIDEO_EGL_DRIVER='/home/dev/globo_dreamer_state/software_egl.so',
            __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
        state=Path(env.mt.run_dir)/'worlds/world/placement_probe.json'
        with (Path(env.mt.run_dir)/'mods/craftium_env/init.lua').open('a') as f:
            f.write('\nlocal PROBE_PATH='+json.dumps(str(state))+'\n'+LUA)
            f.write('\n'+Path(__file__).with_name("human_inventory.lua").read_text())
        try:
            human = HumanFrameRepeat(env, 8)
            human.reset(seed=19)
            for label,action in [('initial',{}),('slot2',{'slot_2':1}),('place',{'place':1})]:
                for _ in range(40):obs,*_=human.step(action)
                Image.fromarray(obs).save('/tmp/globo-probe-'+label+'.png')
                print(label,state.read_text(),flush=True)
            print("human_inventory",(state.parent/'human_inventory.json').read_text(),flush=True)
            assert json.loads(state.read_text())["placed"] > 0, "Placement failed"
            for name in ('stderr.txt','debug.txt'):
                for log in Path(env.mt.run_dir).rglob(name):
                    print('\n'.join(line for line in log.read_text(errors='replace').splitlines()
                        if any(s in line.lower() for s in ('pointing','place button','interact','places node','too far','privilege')))[-6000:],flush=True)
        except Exception:
            print((Path(env.mt.run_dir)/'stderr.txt').read_text()[-6000:],flush=True)
            raise
        finally:
            try:env.close()
            except ConnectionError:pass


if __name__=='__main__':main()
