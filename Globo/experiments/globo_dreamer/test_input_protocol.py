"""Pod-only regression: real Globo adapter, isolated world, one place action."""
import json
import os
from pathlib import Path
import sys
import tempfile

STATE = Path(os.environ.get("GLOBO_STATE", "/home/dev/globo_dreamer_state"))
sys.path.insert(0, str(STATE / "vendor/dreamerv3"))
sys.path.insert(0, str(STATE / "packages"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "original_explorer"))

from probe_placement import LUA
from .env import Globo
from .legacy import env as legacy


def main():
    single_place = '--single-place' in sys.argv
    os.environ["GLOBO_SOFTWARE_RENDERER"] = "1"
    original = legacy.make_globo_env
    with tempfile.TemporaryDirectory(prefix="globo-input-", dir="/tmp") as tmp:
        os.environ["GLOBO_ENGINE_ROOT"] = str(Path(tmp) / "engine")

        def fixture(**kwargs):
            engine = original(**kwargs)
            path = Path(engine.mt.run_dir) / "worlds/world/placement_probe.json"
            with (Path(engine.mt.run_dir) / "mods/craftium_env/init.lua").open("a") as f:
                f.write("\nlocal PROBE_PATH=" + json.dumps(str(path)) + "\n"
                        + LUA.replace("nodes_nature:sand 10", "nodes_nature:silt_wet 10"))
                if single_place:
                    # Keep the target clear so a held button could repeat-place.
                    f.write('\nminetest.register_on_placenode(function(pos) minetest.remove_node(pos) end)\n')
                f.write(r'''
local uses = 0
minetest.register_craftitem("craftium_env:input_probe", {
 description="Input regression probe",
 on_use=function(stack)
  uses=uses+1
  minetest.safe_file_write(minetest.get_worldpath().."/primary_use_probe.json",
   minetest.write_json({uses=uses}))
  return stack
 end
})
local given=false
minetest.register_globalstep(function()
 local p=minetest.get_connected_players()[1]
 if p and not given then
  p:get_inventory():set_stack("main",1,"craftium_env:input_probe")
  given=true
 end
end)
''')
            return engine

        legacy.make_globo_env = fixture
        env = Globo(Path(tmp) / "environment", seed=19, episode_steps=0, video_every=0,
                    render_size=256, spawn="0,111,0", memory_size=32, bounded_artifacts=True,
                    human_place_test=single_place)
        try:
            env.step(dict(action=0, reset=True))
            path = Path(env.engine.mt.run_dir) / "worlds/world/placement_probe.json"
            def action(a):
                return env.step(dict(action=a, reset=False))
            # One inventory action, followed by idle steps to allow asynchronous
            # server acknowledgements; do not spam the selection command.
            action(17)
            for _ in range(40):
                action(0)
                if json.loads(path.read_text())["wield"] == 2:
                    break
            before = json.loads(path.read_text())
            assert before["wield"] == 2, before
            assert before["placed"] == 0, before
            action(13)  # Exactly ONE agent place action.
            for _ in range(40):
                action(0)
                if json.loads(path.read_text())["placed"] > 0:
                    break
            after = json.loads(path.read_text())
            assert after["placed"] > 0, after
            assert after["item"] != before["item"], (before, after)
            assert env.action_hist[13] == 1
            if single_place:
                for _ in range(20): action(0)
                assert json.loads(path.read_text())['placed'] == 1
                action(13)
                action(13)  # Consecutive commands must each get a fresh edge.
                for _ in range(40): action(0)
                assert json.loads(path.read_text())['placed'] == 3, path.read_text()
                print('PASS: one placement per command, including consecutive commands.', flush=True)
                return
            action(16)
            for _ in range(40):
                action(0)
                if json.loads(path.read_text())["wield"] == 1:
                    break
            assert json.loads(path.read_text())["wield"] == 1
            action(12)
            use_path = path.with_name("primary_use_probe.json")
            for _ in range(40):
                action(0)
                if use_path.exists():
                    break
            assert json.loads(use_path.read_text())["uses"] == 1
            print("PASS: one slot_2 action and one place_use action via Globo.step.", flush=True)
            print("PASS: one dig_use action invokes a primary-use callback exactly once.", flush=True)
            print(json.dumps(dict(before=before, after=after)), flush=True)
        finally:
            env.close()
            legacy.make_globo_env = original


if __name__ == "__main__":
    main()
