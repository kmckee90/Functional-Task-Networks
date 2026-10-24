"""Pod-only integration regression for late initial player telemetry."""
import os
from pathlib import Path
import sys
import tempfile

STATE = Path(os.environ.get('GLOBO_STATE', '/home/dev/globo_dreamer_state'))
sys.path[:0] = [str(STATE/'vendor/dreamerv3'), str(STATE/'packages')]
from .env import Globo
from .legacy import env as legacy

def main():
    factory = legacy.make_globo_env
    os.environ['GLOBO_SOFTWARE_RENDERER'] = '1'
    with tempfile.TemporaryDirectory(prefix='telemetry-start-', dir='/tmp') as tmp:
        os.environ['GLOBO_ENGINE_ROOT'] = str(Path(tmp)/'engine')
        def fixture(**kwargs):
            engine = factory(**kwargs)
            with (Path(engine.mt.run_dir)/'mods/craftium_env/init.lua').open('a') as f:
                f.write('''
local original_write = minetest.safe_file_write
local started = minetest.get_us_time()
minetest.safe_file_write = function(path, content)
 if path:match('agent_state.json$') and minetest.get_us_time()-started < 8000000 then
  return true
 end
 return original_write(path, content)
end
''')
            reset = engine.reset
            def reset_without_ready_telemetry(*args, **kwargs):
                result = reset(*args, **kwargs)
                # Guarantee the missing-file branch even on a slow renderer.
                (Path(engine.mt.run_dir)/'worlds/world/agent_state.json').unlink(missing_ok=True)
                return result
            engine.reset = reset_without_ready_telemetry
            return engine
        legacy.make_globo_env = fixture
        env = Globo(Path(tmp)/'environment', seed=19, render_size=64,
                    spawn='12,39.1,120', memory_size=32, video_every=0)
        try:
            env.step(dict(action=0, reset=True))
            assert env.telemetry.exists()
            assert env.total == 1 and env.episode_step == 0
            assert env.action_hist.sum() == 0
            print('PASS: late real telemetry; no startup policy actions counted.', flush=True)
        finally:
            env.close()
            legacy.make_globo_env = factory

if __name__ == '__main__':
    main()
