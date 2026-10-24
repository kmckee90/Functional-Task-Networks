"""One sequential Globo experience stream, with diagnostic-only world position."""
import json
import os
import base64
import time
from pathlib import Path

import av
import elements
import embodied
import numpy as np
from PIL import Image, ImageDraw

from .memory import Recode, embedding
from .frame_quality import frame_quality


# Motor primitives include common compound controls. No scripted navigation.
ACTIONS = [
    ('idle', {}), ('forward', {'forward': 1}), ('back', {'backward': 1}),
    ('left', {'left': 1}), ('right', {'right': 1}),
    ('jump_forward', {'forward': 1, 'jump': 1}),
    ('look_left', {'mouse': [-.25, 0]}), ('look_right', {'mouse': [.25, 0]}),
    ('look_up', {'mouse': [0, .15]}), ('look_down', {'mouse': [0, -.15]}),
    ('forward_left', {'forward': 1, 'mouse': [-.15, 0]}),
    ('forward_right', {'forward': 1, 'mouse': [.15, 0]}),
    ('dig_use', {'dig': 1}), ('place_use', {'place': 1}),
    ('sneak', {'sneak': 1}), ('sneak_use', {'sneak': 1, 'place': 1}),
] + [(f'slot_{i}', {f'slot_{i}': 1}) for i in range(1, 10)]


def runtime_path(path):
    path = Path(path).expanduser().resolve()
    source = Path(__file__).resolve().parent
    if path == source or source in path.parents:
        raise ValueError(f'Runtime files must be outside synced source: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, allow_nan=False, indent=2))
    temp.replace(path)


class Globo(embodied.Env):
    def __init__(self, root, seed=7, episode_steps=6000, video_every=4000,
                 memory_size=1024, **kwargs):
        self.root = runtime_path(root)
        self.seed, self.episode_steps, self.video_every = seed, episode_steps, video_every
        self.kwargs = kwargs
        self.lifetime_v2 = os.environ.get('GLOBO_LIFETIME_V2') == '1'
        self.actions = ACTIONS[:14] if kwargs.get('reduced_actions', False) else ACTIONS
        if self.lifetime_v2:
            self.actions = list(self.actions)
            self.actions[5] = ('jump', {'jump': 1})
        action_names = [name for name, _ in self.actions]
        action_file = self.root / 'actions.json'
        if action_file.exists() and json.loads(action_file.read_text()) != action_names:
            raise ValueError('Cannot change action space when resuming a run')
        if not action_file.exists() and (self.root / 'lifetime.json').exists() and kwargs.get('reduced_actions'):
            raise ValueError('Cannot resume a legacy full-action run with reduced actions')
        atomic_json(action_file, action_names)
        self.render_size = int(kwargs.get('render_size', 64))
        self.human_frame = None
        self.memory = Recode(capacity=memory_size, seed=seed)
        self.novelty_fn = None
        self.memory_size_fn = None
        if (self.root / 'memory.npz').exists():
            self.memory.load(self.root / 'memory.npz')
        self.total = self.memory.seen
        self.life = 0
        self.visited = set()
        if (self.root / 'lifetime.json').exists():
            state = json.loads((self.root / 'lifetime.json').read_text())
            self.life = state['life']
            self.total = state.get('total', self.total)
            self.visited = set(map(tuple, state['visited']))
        self.engine = None
        self.done = True
        self.video = None
        self.video_count = 0
        self.record = None
        self.wall_start = time.time()
        self.start_total = self.total
        self.episode_start = 0
        self.action_hist = np.zeros(len(self.actions), np.int64)
        self.achievements = None

    def enable_achievements(self):
        from .achievements import Achievements
        self.achievements=Achievements(self.root)

    @property
    def obs_space(self):
        from .hud_feedback import NAMES
        return dict(
            image=elements.Space(np.uint8, (64, 64, 3)),
            body=elements.Space(np.float32, (len(NAMES) if self.lifetime_v2 else 17,)),
            density=elements.Space(np.float32, (1,), 0, 1),
            reward=elements.Space(np.float32),
            is_first=elements.Space(bool), is_last=elements.Space(bool),
            is_terminal=elements.Space(bool),
        )

    @property
    def act_space(self):
        return dict(action=elements.Space(np.int32, (), 0, len(self.actions)),
                    reset=elements.Space(bool))

    def _start(self):
        from .legacy import env as legacy
        # Use the pinned input-fixed engine without replacing the wheel's binary
        # or touching already-running worlds. Missing builds fail explicitly.
        state_root = Path(os.environ.get('GLOBO_STATE', '/home/dev/globo_dreamer_state'))
        single_place = self.kwargs.get('human_place_test') or os.environ.get('GLOBO_SINGLE_PLACE') == '1'
        block_capacity = int(os.environ.get('GLOBO_BLOCK_STACK_CAPACITY', '0'))
        input_binary = Path(os.environ.get(
            'GLOBO_INPUT_BINARY', str(state_root / 'engines/input-v2/luanti')))
        if single_place:
            input_binary = state_root / 'engines/input-v2-single-place/luanti'
        if not input_binary.is_file():
            raise RuntimeError(
                f'Input-fixed Craftium engine missing: {input_binary}. '
                'Run globo_dreamer/build_input_engine.py on this pod first.')
        if self.engine is not None:
            old_run=Path(self.engine.mt.run_dir)
            self.engine.close()
            if self.kwargs.get('bounded_artifacts') and os.environ.get('GLOBO_ENGINE_ROOT'):
                import shutil
                if old_run.resolve().is_relative_to(Path(os.environ['GLOBO_ENGINE_ROOT']).resolve()/'engine'):
                    shutil.rmtree(old_run,ignore_errors=True)
        # Template remains outside sync. Only generated runtime files are patched.
        engine_root = Path(os.environ['GLOBO_ENGINE_ROOT']) if os.environ.get('GLOBO_ENGINE_ROOT') else self.root
        engine_root.mkdir(parents=True,exist_ok=True)
        self.engine = legacy.make_globo_env(
            env_dir=str(engine_root / 'template'), run_dir_prefix=str(engine_root / 'engine'),
            action_mode='raw', obs_width=self.render_size, obs_height=self.render_size, seed=self.seed,
            spawn=self.kwargs.get('spawn', '8,8,115'), frameskip=8, init_frames=200, max_timesteps=None,
            move_speed=float(os.environ.get('GLOBO_MOVE_SPEED','7.5')),
            move_accel=float(os.environ.get('GLOBO_MOVE_ACCEL','1000')), climb_speed=3, jump_speed=6.5,
            freeze_daylight=os.environ.get('GLOBO_NATURAL_CYCLE')!='1',
            clear_sky=os.environ.get('GLOBO_NATURAL_CYCLE')!='1', weather_seed=68,
            invincible=self.kwargs.get('screen_invincible', False),
            extra_minetest_conf={'globo_hud': 'none'})
        import shutil
        shutil.copy2(input_binary, Path(self.engine.mt.run_dir) / 'bin/luanti')
        if os.environ.get('GLOBO_HARD_SURVIVAL')=='1':
            from .hard_survival_game import apply
            apply(Path(self.engine.mt.run_dir)/'games/globo')
        atomic_json(self.root / 'input_protocol.json', dict(
            version=2, native_frameskip=8, binary=str(input_binary),
            click_edges=['dig', 'place'], held_through_skip=(['sneak'] if
                single_place else ['place', 'sneak']),
            block_stack_capacity=block_capacity, starting_blocks=4 if self.kwargs.get('human_place_test') else 0))
        if os.environ.get('GLOBO_SOFTWARE_RENDERER')=='1':
            # Craftium supplies a minimal child environment, dropping CUDA and
            # Mesa settings from the trainer. Set the renderer explicitly.
            self.engine.mt.proc_env.update(LIBGL_ALWAYS_SOFTWARE='1',
                GALLIUM_DRIVER='llvmpipe',LP_NUM_THREADS='2',
                SDL_HINT_EGL_DEVICE='0',
                SDL_VIDEO_EGL_DRIVER=os.environ.get('GLOBO_SOFTWARE_EGL_LIBRARY','/home/dev/globo_dreamer_state/software_egl.so'),
                __EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/50_mesa.json')
        # Craftium has already copied the template into its run directory.
        control = Path(self.engine.mt.run_dir) / 'mods/craftium_env/init.lua'
        with control.open('a') as f:
            if block_capacity:
                f.write('''
minetest.register_on_mods_loaded(function()
 for name, def in pairs(minetest.registered_nodes) do
  if (def.stack_max or 99) < %d then
   minetest.override_item(name, {stack_max=%d})
  end
 end
end)
''' % (block_capacity, block_capacity))
            if self.kwargs.get('human_place_test'):
                f.write('''
minetest.register_on_mods_loaded(function()
 minetest.override_item('nodes_nature:silt_wet', {stack_max=4})
end)
minetest.register_on_joinplayer(function(player)
 minetest.after(0, function()
  if player:is_player() then
   player:get_inventory():set_stack('main', 1, 'nodes_nature:silt_wet 4')
  end
 end)
end)
''')
            f.write('\n' + Path(__file__).with_name('telemetry.lua').read_text())
            if self.achievements is not None:
                f.write('\n' + Path(__file__).with_name('achievements.lua').read_text())
            if os.environ.get('GLOBO_EVENT_CAPTURE')=='1':
                f.write('\n' + Path(__file__).with_name('video_events.lua').read_text())
            if self.kwargs.get('audit_spawn'):
                f.write('\n' + Path(__file__).with_name('spawn_probe.lua').read_text())
            if self.kwargs.get('find_spawn'):
                f.write('\nglobo_spawn_scan_ceiling = '+str(int(self.kwargs.get('spawn_scan_ceiling',60)))+'\n')
                f.write('\n' + Path(__file__).with_name('find_spawn.lua').read_text())
        self.telemetry = Path(self.engine.mt.run_dir) / 'worlds/world/agent_state.json'
        image, info = self.engine.reset()
        # Rendering can finish Craftium's fixed warmup before the asynchronous
        # server has joined the player and emitted its first bodily observation.
        # Advance only startup no-ops until that real observation exists; these
        # are not policy interactions. Never substitute invented bodily values.
        deadline = time.monotonic() + 30
        while not self.telemetry.exists():
            if time.monotonic() >= deadline:
                log = Path(self.engine.mt.run_dir) / 'stderr.txt'
                detail = log.read_text(errors='replace')[-4000:] if log.exists() else ''
                raise RuntimeError(f'Globo player telemetry did not initialize within 30s. '
                                   f'Engine log: {log}\n{detail}')
            image, _, _, _, info = self.engine.step({})
            time.sleep(0.01)
        self.life += 1
        self.episode_step = 0
        self.episode_start = self.total
        self.episode_cells = set()
        self.episode_return = 0.
        self.game_time = 0.
        self.action_hist[:] = 0
        self.done = False
        return image, info

    def step(self, action):
        first = bool(action['reset']) or self.done
        aid = 0 if first else int(action['action'])
        if first:
            image, info = self._start()
            terminal = timeout = False
        else:
            controls = dict(self.actions[aid][1])
            if 'mouse' in controls:
                controls['mouse'] = [v * 64 / self.render_size for v in controls['mouse']]
            image, _, terminal, _, info = self.engine.step(controls)
            self.episode_step += 1
            self.action_hist[aid] += 1
            timeout = self.episode_steps > 0 and self.episode_step >= self.episode_steps
        if self.render_size != 64:
            self.human_frame = np.array(image, copy=True)
            image = np.asarray(Image.fromarray(image).resize((64,64), Image.Resampling.BOX))
        if not self.telemetry.exists():
            raise RuntimeError('Globo telemetry missing; bodily state must not be silently fabricated')
        state = json.loads(self.telemetry.read_text())
        if first:
            self.start_game_elapsed = float(state['elapsed'])
        terminal = bool(terminal or state['hp'] <= 0)
        self.done = bool(terminal or timeout)
        if self.novelty_fn is None:
            emb = embedding(image)
            novelty = self.memory.query(emb)
            self.memory.update(emb)
        else:
            novelty = self.novelty_fn(image)
        self.total += 1
        slots = list(state.get('slots') or [])[:9]
        slots += [0] * (9 - len(slots))
        body = np.array([
            state['hp'] / max(1, state['hp_max']), state['hunger'] / 1000,
            state['thirst'] / 1000, state['energy'] / max(1, state['energy_max']),
            (state['temperature'] - 37) / 20, state['breath'] / 10,
            state['wield'] / 9, sum(state['items'].values()) / 100,
            *[v / 99 for v in slots]], np.float32)
        body = np.clip(body, -10, 10)
        if self.lifetime_v2:
            from .hud_feedback import encode
            body = encode(state)
        self.game_time = max(0., float(state['elapsed']) - self.start_game_elapsed)
        pos = np.asarray(info['player_pos']).reshape(-1).tolist()
        cell = (int(np.floor(pos[0] / 4)), int(np.floor(pos[2] / 4)))
        new_spatial_cell = cell not in self.visited
        self.visited.add(cell)
        self.episode_cells.add(cell)
        reward = -float(terminal)
        self.episode_return += reward
        record = dict(step=self.total, life=self.life, life_step=self.episode_step,
            game_seconds=self.game_time, wall_seconds=time.time() - self.wall_start,
            density=novelty, memory_size=(self.memory_size_fn() if self.memory_size_fn
                                        else self.memory.size), action=self.actions[aid][0],
            position=pos, cells_lifetime=len(self.visited), cells_life=len(self.episode_cells),
            terminal=terminal, truncated=bool(timeout), reward=reward, **state)
        record.update(frame_quality(image))
        if os.environ.get('GLOBO_EVENT_CAPTURE')=='1':
            record['new_spatial_cell']=new_spatial_cell
            event_path=self.telemetry.with_name('agent_video_events.json')
            try:record['video_events']=json.loads(event_path.read_text()).get('events',[])
            except (OSError,ValueError):record['video_events']=[]
        if self.achievements is not None:
            path=self.telemetry.with_name('agent_achievements.json')
            data=json.loads(path.read_text())
            record.update(self.achievements.observe(data,str(self.engine.mt.run_dir),
                self.total,self.life,initial=first))
        if self.total % 10 == 0:
            # Fixed evaluation features, independent of policy/reward hyperparameters.
            view = np.asarray(image).reshape(8,8,8,8,3).mean(axis=(1,3)).astype(np.uint8)
            record['view_signature'] = base64.b64encode(view.tobytes()).decode('ascii')
        self.record = record
        if not self.kwargs.get('bounded_artifacts'):
            with (self.root / 'trace.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
        if first or self.total % 50 == 0 or self.done:
            atomic_json(self.root / 'latest.json', record)
        self._video(image, record)
        if self.total % 500 == 0 or self.done:
            self.save()
        if self.done:
            record['action_counts'] = dict(zip([a[0] for a in self.actions], self.action_hist.tolist()))
            with (self.root / 'episodes.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            self._close_video()
        return dict(image=np.asarray(image, np.uint8), body=body,
            density=np.array([novelty], np.float32), reward=np.float32(reward),
            is_first=np.bool_(first), is_last=np.bool_(self.done), is_terminal=np.bool_(terminal))

    def _video(self, image, rec):
        if not self.video_every:
            return
        if self.video is None and (self.total % self.video_every == 1 or self.total == self.start_total + 1):
            self.video_path = self.root / f'behavior_{self.total:09d}.mp4'
            self.video = av.open(str(self.video_path), 'w')
            self.stream = self.video.add_stream('libx264', rate=12)
            self.stream.width, self.stream.height = 384, 448
            self.stream.pix_fmt = 'yuv420p'
            self.video_count = 0
        if self.video is None or self.episode_step % 2:
            return
        canvas = Image.new('RGB', (384, 448), '#171c24')
        canvas.paste(Image.fromarray(image).resize((384, 384)), (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 386), f"step {self.total} life {self.life}: {rec['action']}", fill='white')
        draw.text((8, 403), f"HP {rec['hp']} food {rec['hunger']} water {rec['thirst']}", fill='white')
        draw.text((8, 420), f"novelty {rec['density']:.3f} cells {rec['cells_lifetime']}", fill='white')
        frame = av.VideoFrame.from_image(canvas)
        for packet in self.stream.encode(frame):
            self.video.mux(packet)
        self.video_count += 1
        if self.video_count >= 240:
            self._close_video()

    def _close_video(self):
        if self.video is not None:
            for packet in self.stream.encode():
                self.video.mux(packet)
            self.video.close()
            self.video = None

    def save(self):
        self.memory.save(self.root / 'memory.npz')
        atomic_json(self.root / 'lifetime.json', dict(life=self.life, total=self.total,
                    visited=sorted(self.visited)))

    def close(self):
        self._close_video()
        self.save()
        if self.engine is not None:
            self.engine.close()
            self.engine = None
