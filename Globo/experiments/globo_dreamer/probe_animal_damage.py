"""Pod-only controlled attack diagnostic; never modifies a training world."""
import json
import os
from pathlib import Path
import sys
import tempfile

STATE = Path(os.environ.get("GLOBO_STATE", "/home/dev/globo_dreamer_state"))
sys.path[:0] = [str(STATE / "vendor/dreamerv3"), str(STATE / "packages")]
from .env import Globo
from .legacy import env as legacy

LUA = r'''
local animal, player
local hits, lastdamage, death = 0, 0, false
minetest.register_on_mods_loaded(function()
for name, def in pairs(minetest.registered_entities) do
 if name:sub(1,8) == "animals:" and def.max_hp then
  assert(def.max_hp <= 20, name .. " exceeds HP cap")
 end
end
local original = animals.on_punch
animals.on_punch = function(self, caps, puncher, prty, chance)
 local before=self.hp
 original(self,caps,puncher,prty,chance)
 if self.object == animal and before > self.hp then
  hits=hits+1
  lastdamage=before-self.hp
 end
end
end)
minetest.register_globalstep(function()
 player=minetest.get_connected_players()[1]
 if not player then return end
 if not animal then
  minetest.load_area({x=-4,y=109,z=-4},{x=4,y=116,z=4})
  for x=-4,4 do for z=-4,4 do
   minetest.set_node({x=x,y=110,z=z},{name="nodes_nature:sandstone"})
   for y=111,116 do minetest.set_node({x=x,y=y,z=z},{name="air"}) end
  end end
  player:get_inventory():set_stack("main",1,"")
  animal=minetest.add_entity({x=0,y=112,z=1.5},"animals:gazelle")
  -- Immobilize only: keep the original punch handler and core death path.
  -- Disable regeneration here to measure the exact landed-hit requirement.
  local e=animal:get_luaentity()
  e.on_step=function(self)
   if not animals.core_hp(self) then death=true end
  end
  animal:set_properties({physical=false})
 end
 player:set_pos({x=0,y=110.5,z=0})
 player:set_look_horizontal(0);player:set_look_vertical(0)
 local e=animal:get_luaentity()
 if e and not death then
  animal:set_pos({x=0,y=112,z=1.5});animal:set_velocity({x=0,y=0,z=0})
 end
 minetest.safe_file_write(minetest.get_worldpath().."/attack_probe.json",
  minetest.write_json({hits=hits,hp=e and e.hp or 0,damage=lastdamage,
   death=death,item=player:get_wielded_item():to_string(),
   wield=player:get_wield_index()}))
end)
'''


def main():
    os.environ["GLOBO_SOFTWARE_RENDERER"] = "1"
    factory = legacy.make_globo_env
    with tempfile.TemporaryDirectory(prefix="globo-attack-", dir="/tmp") as tmp:
        os.environ["GLOBO_ENGINE_ROOT"] = str(Path(tmp) / "engine")
        def fixture(**kwargs):
            engine = factory(**kwargs)
            with (Path(engine.mt.run_dir) / "mods/craftium_env/init.lua").open("a") as f:
                f.write("\n" + LUA)
            return engine
        legacy.make_globo_env = fixture
        env = Globo(Path(tmp) / "environment", seed=19, episode_steps=0,
                    video_every=0, render_size=64, spawn="0,111,0",
                    memory_size=32, screen_invincible=True, bounded_artifacts=True)
        try:
            env.step(dict(action=0, reset=True))
            path = Path(env.engine.mt.run_dir) / "worlds/world/attack_probe.json"
            env.step(dict(action=16, reset=False))
            for i in range(1800):
                env.step(dict(action=12, reset=False))
                if i % 100 == 0:
                    print(i, path.read_text(), flush=True)
                data = json.loads(path.read_text())
                if data["death"]:
                    print("DEATH", i + 1, data, flush=True)
                    assert data["hits"] == 4 and data["damage"] == 5, data
                    break
            else:
                raise AssertionError(("No death", data))
        finally:
            env.close()
            legacy.make_globo_env = factory


if __name__ == "__main__":
    main()
