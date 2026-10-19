"""FTN-M2E runner. All outputs go to --root, outside the synchronized checkout."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import signal
import time

import av
import numpy as np
from PIL import Image, ImageDraw
import torch

from .doom import Doom, normalize_telemetry
from .model_free import ModelFreeBehavior
from .m2e_ftn import M2EFTN


def atomic_json(path, data):
    tmp = path.with_suffix('.pending')
    tmp.write_text(json.dumps(data, allow_nan=False))
    tmp.replace(path)


def movie(path, frames, fps):
    tmp = path.with_name(path.stem + '.pending.mp4')
    with av.open(str(tmp), 'w') as out:
        stream = out.add_stream('libx264', rate=fps)
        stream.width, stream.height = frames[0].size
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '23', 'preset': 'veryfast'}
        for image in frames:
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    tmp.replace(path)


def publish_video(root, rows, ftn, tics):
    from fractions import Fraction
    ftn.consolidate()
    frames, inputs = [], []
    for row in rows:
        canvas = Image.new('RGB', (1280, 504), '#171c24')
        canvas.paste(Image.fromarray(row['human']), (0, 24))
        mask = row['mask'][ftn.layout].reshape(*ftn.grid, 3)
        canvas.paste(Image.fromarray(mask).resize((320, 320), Image.Resampling.NEAREST), (640, 80))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 5), f"Step {row['step']:,} | {row['map_name']}", fill='white')
        draw.text((646, 5), 'FTN mask / EMA', fill='white')
        draw.text((966, 5), 'Offline frontier (density view)', fill='white')
        if row['goal'] is not None:
            canvas.paste(Image.fromarray(row['goal']).convert('RGB').resize((320, 160), Image.Resampling.NEAREST), (960, 140))
            draw.text((966, 320), f"Observed at step {row['goal_step']:,}", fill='white')
        else:
            draw.text((966, 150), 'Collecting initial experience', fill='white')
        draw.text((646, 450), f"Health {row['health']:.0f} | Selected ammo {row['ammo']:.0f}", fill='white')
        frames.append(canvas)
        inp = Image.new('RGB', (640, 264), '#171c24')
        inp.paste(Image.fromarray(row['input']), (0, 24))
        inp.paste(Image.fromarray(row['feedback']).convert('RGB'), (320, 24))
        d = ImageDraw.Draw(inp)
        d.text((5, 5), 'Actual RGB input (320 x 240)', fill='white')
        d.text((325, 5), 'Projected feedback channel', fill='white')
        inputs.append(inp)
    movie(root / 'obs.mp4', frames, Fraction(35, tics))
    movie(root / 'input.mp4', inputs, Fraction(35, tics))
    frames[-1].save(root / 'latest.jpg')
    atomic_json(root / 'video.json', dict(step=rows[-1]['step'], start=rows[0]['step'],
        seconds=len(rows) * tics / 35, updated=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wad', type=Path, default=Path('wad/Doom + Doom II Collection WAD files/doom.wad'))
    parser.add_argument('--seed', type=int, default=19)
    parser.add_argument('--steps', type=int, default=0, help='0 means indefinite; nonzero for validation')
    parser.add_argument('--replay', type=int, default=100000)
    parser.add_argument('--frontier', type=int, default=20000)
    parser.add_argument('--centroids', type=int, default=100000)
    parser.add_argument('--video-every', type=int, default=200)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.is_relative_to(Path.cwd().resolve()):
        raise ValueError('Runtime must be outside synchronized source')
    root.mkdir(parents=True, exist_ok=True)
    os.environ['GLOBO_MF_REPLAY_DTYPE'] = 'float16'
    os.environ['GLOBO_FRONTIER_ROTATION'] = '1'
    torch.set_num_threads(4)
    stop = False
    def request_stop(*_):
        nonlocal stop
        stop = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    saved = torch.load(root / 'checkpoint.pt', map_location='cpu', weights_only=False) if (root / 'checkpoint.pt').exists() else None
    wad_changed = bool(saved and saved.get('wad') != str(args.wad.resolve()))
    env = Doom(args.wad, args.seed, saved['map'] if saved and not wad_changed else 1)
    behavior = ModelFreeBehavior(device='cuda:0', seed=args.seed, dreamer_probability=0.,
        replay_capacity=args.replay, goal_capacity=args.frontier, recode_capacity=args.centroids,
        checkpoint_root=root / 'replay', epsilon_end=.015366900171141888,
        gamma=.9036665304295952, online_coef=.8728610896464566,
        frontier_scale=4.574998390699497, lr=.0001899492997635358,
        reservoir_channels=128, reservoir_variant='tuned_v1', num_actions=len(env.actions),
        in_h=240, in_w=320, reservoir_stride=16, telemetry_dims=2)
    behavior.ftn = M2EFTN(behavior.agent, 'conv')
    agent, ftn = behavior.agent, behavior.ftn
    agent.capture_input = True
    step, deaths, levels = 0, 0, 0
    if saved:
        behavior.load(saved['behavior'])
        step, deaths, levels = saved['step'], saved['deaths'], saved['levels']
        # Resume learning memory, but begin a new game trajectory at current map.
        for buf in (agent.memory, agent.goal):
            if buf.size:
                idx = (buf.ptr - 1) % buf.capacity
                buf.boundary[idx] = True
                buf.dirty_chunks.add(idx // behavior.store.chunk_size)
        del saved
    config = dict(seed=args.seed, wad=str(args.wad.resolve()), actions=env.action_names,
        action_buttons=env.actions, tics=env.tics, render=[640,480], input=[320,240],
        health_scale='clip(health/200,0,1)', ammo_scale='clip(log1p(selected_ammo)/log1p(600),0,1)',
        replay=args.replay, frontier=args.frontier, centroids=args.centroids,
        reservoir=agent.reservoir_config, ftn=ftn.q.spec, extrinsic_reward=0.,
        skill=3, initial_map=env.maps[0], resumed=step > 0, frontier_rotation=True,
        initial_step=step, initial_replay=agent.memory.size, initial_frontier=agent.goal.size,
        initial_centroids=agent.density.size, initial_updates=behavior.updates,
        initial_ftn_updates=ftn.updates, initial_epsilon=agent.epsilon)
    atomic_json(root / 'config.json', config)
    if wad_changed:
        with (root / 'events.jsonl').open('a') as out:
            out.write(json.dumps(dict(event='wad_change', step=step, time=time.time(),
                wad=str(args.wad.resolve()), map_name=env.map_name, retained_learning_state=True))+'\n')
    print(f'{"Resumed" if step else "Started fresh"} step={step} wad={args.wad.name} map={env.map_name}', flush=True)
    rows = deque(maxlen=105)
    history = deque(maxlen=2000)
    if (root / 'history.json').exists():
        history.extend(json.loads((root / 'history.json').read_text()))
    started, start_step = time.time(), step
    checkpoint_at = time.time()
    first = True
    def status(running=True):
        free, total = torch.cuda.mem_get_info()
        data = dict(step=step, map=env.map_number, map_name=env.map_name, deaths=deaths, levels_completed=levels,
            seed=args.seed, running=running, updated=time.time(), pid=os.getpid(),
            steps_per_second=(step-start_step)/max(1.,time.time()-started),
            replay=agent.memory.size, frontier=agent.goal.size, centroids=agent.density.size,
            gpu_free_gib=free/2**30, gpu_total_gib=total/2**30,
            gpu_allocated_gib=torch.cuda.memory_allocated()/2**30,
            extrinsic_reward=0., **{k:float(v) for k,v in behavior.metrics.items()})
        atomic_json(root / 'status.json', data)
        return data
    def checkpoint():
        nonlocal checkpoint_at
        state = dict(step=step, deaths=deaths, levels=levels, map=env.map_number, wad=str(args.wad.resolve()),
                     behavior=behavior.save())
        torch.save(state, root / 'checkpoint.pending.pt')
        (root / 'checkpoint.pending.pt').replace(root / 'checkpoint.pt')
        behavior.store.committed()
        checkpoint_at = time.time()
        print(f'Checkpoint committed step={step}', flush=True)
    try:
        status()
        while not stop and (not args.steps or step < args.steps):
            image, telemetry = env.observation()
            agent.telemetry.copy_(torch.from_numpy(normalize_telemetry(*telemetry)).to(agent.device))
            novelty = behavior.observe_frame(image)
            obs = dict(image=image[None], is_first=[first], is_last=[False], is_terminal=[False],
                       density=np.array([[novelty]]), reward=[0.])
            action = int(behavior.policy(obs, {'action':[0]})['action'][0])
            goal = getattr(agent, 'display_frontier', None)
            rows.append(dict(step=step, map=env.map_number, map_name=env.map_name, human=image,
                input=np.uint8(np.clip(agent.last_input[:3].transpose(1,2,0)*255,0,255)),
                feedback=np.uint8(np.clip((agent.last_input[3]+1)*127.5,0,255)),
                mask=ftn.frame().copy(), health=float(telemetry[0]), ammo=float(telemetry[1]),
                goal=None if goal is None else np.uint8(np.clip(goal.numpy().reshape(16,32)*255,0,255)),
                goal_step=getattr(agent,'display_frontier_step',-1)))
            ended, dead = env.step(action)
            step += 1
            first = False
            if ended:
                # ViZDoom gives no final screen after termination; the last visible
                # screen supplies the terminal next-state, whose Q is never bootstrapped.
                obs.update(is_first=[False], is_last=[True], is_terminal=[True])
                behavior.policy(obs, {'action':[0]})
                deaths += int(dead)
                completed = not dead and not env.game.is_episode_timeout_reached()
                levels += int(completed)
                env.reset(completed)
                first = True
            if step % 25 == 0:
                data = status()
                data.update(health=float(telemetry[0]), ammo=float(telemetry[1]))
                history.append(data)
                with (root / 'metrics.jsonl').open('a') as out:
                    out.write(json.dumps(data) + '\n')
                atomic_json(root / 'history.json', list(history))
            if step % args.video_every == 0:
                publish_video(root, rows, ftn, env.tics)
                print(f'Step={step} novelty={novelty:.5f} updates={behavior.updates}', flush=True)
            if step % 2000 == 0 or time.time() - checkpoint_at > 600:
                checkpoint()
        if rows:
            publish_video(root, rows, ftn, env.tics)
        checkpoint()
    finally:
        status(False)
        env.close()


if __name__ == '__main__':
    main()
