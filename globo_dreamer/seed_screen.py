"""Screen fixed seed/spawn pairs outside training, with safe-start and movement checks."""
import argparse
import json
import os
from pathlib import Path
import sys
STATE=Path(os.environ.get('GLOBO_STATE','/home/dev/globo_dreamer_state'))
sys.path.insert(0,str(STATE/'vendor/dreamerv3'))
from PIL import Image
from .env import Globo, atomic_json, runtime_path


def screen(seed):
    root=runtime_path(STATE/'seed_screen'/str(seed))
    scan=Globo(root/'scan',seed=seed,video_every=0,episode_steps=0,
               spawn='11,45,121',find_spawn=True,screen_invincible=True)
    candidates=[]
    try:
        for i in range(1000):
            scan.step(dict(reset=i==0,action=0))
            path=scan.telemetry.with_name('spawn_candidates.json')
            if path.exists():
                candidates=json.loads(path.read_text()) or []
                break
    finally: scan.close()
    if not candidates:
        result=dict(seed=seed,accepted=False,reason='No safe loaded spawn found')
        atomic_json(root/'result.json',result); return result
    c=candidates[0]; spawn=','.join(str(c[k]) for k in ('x','y','z'))
    env=Globo(root/'check',seed=seed,spawn=spawn,video_every=0,episode_steps=0)
    rows=[]
    try:
        for i in range(700):
            # Idle first, then fixed movement/turn/jump probes, identical per seed.
            act=0 if i<400 else [1,10,5,11,1,7][((i-400)//25)%6]
            obs=env.step(dict(reset=i==0,action=act)); rows.append(dict(env.record))
            if i in (100,399,699): Image.fromarray(obs['image']).save(root/f'frame_{i}.png')
            if obs['is_last']: break
    finally: env.close()
    idle=rows[:400]; motion=rows[400:]
    accepted=(len(rows)==700 and min(r['hp'] for r in idle)==1000 and
              rows[-1]['cells_lifetime']>=3 and
              sum(r.get('low_detail',False) for r in rows[100:])/max(1,len(rows)-100)<.7)
    result=dict(seed=seed,spawn=spawn,ground=c['ground'],accepted=accepted,
        idle_damage=1000-min(r['hp'] for r in idle),cells=rows[-1]['cells_lifetime'],
        low_detail_fraction=sum(r.get('low_detail',False) for r in rows)/len(rows),
        health_end=rows[-1]['hp'],observations=len(rows),candidates=candidates)
    atomic_json(root/'result.json',result); print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True)
    screen(p.parse_args().seed)
