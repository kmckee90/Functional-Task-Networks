"""Idle spawn probe; no policy training or changes to world mechanics. POD ONLY."""
import json
import argparse
import os
from pathlib import Path
import sys
STATE=Path(os.environ.get('GLOBO_STATE','/home/dev/globo_dreamer_state'))
sys.path.insert(0,str(STATE/'vendor/dreamerv3'))
from PIL import Image
from .env import Globo, runtime_path, atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--spawn',default='8,8,115')
    p.add_argument('--name',default='spawn_check')
    args=p.parse_args()
    root=runtime_path(STATE/'diagnostics'/args.name)
    env=Globo(root/'environment',audit_spawn=True,video_every=0,spawn=args.spawn)
    changes=[]
    prev_hp=None
    try:
        for i in range(600):
            obs=env.step(dict(reset=i==0,action=0))
            r=env.record
            if r['hp']!=prev_hp:
                changes.append({k:r.get(k) for k in ('step','game_seconds','hp','position','last_damage_type')})
            prev_hp=r['hp']
            audit=env.telemetry.with_name('spawn_audit.json')
            if audit.exists() and not (root/'geometry.json').exists():
                atomic_json(root/'geometry.json',json.loads(audit.read_text()))
            if i in (100,500):
                Image.fromarray(obs['image']).resize((512,512)).save(root/f'idle_{i}.png')
            if obs['is_last']: break
        atomic_json(root/'result.json',dict(health_changes=changes,last=env.record))
        print(json.dumps(dict(health_changes=changes,steps=env.total)),flush=True)
    finally:
        env.close()


if __name__=='__main__': main()
