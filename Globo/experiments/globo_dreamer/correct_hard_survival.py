"""Archive the mistaken new-world cohort and export the four seed-matched explorers."""
import argparse,os,signal,shutil,time
from pathlib import Path
from .phase2 import STATE,read,atomic,stop_supervisor
ROOT=Path('/mnt/obelisk-shared/kmckee90/globo_survival_hard_s19')
def stop():
    stop_supervisor()
    path=STATE/'continuous_runs/state.json';trials=read(path,[])
    for t in trials:
        assert t.get('phase')=='survival_hard'
        proc=Path(f'/proc/{t.get("pid")}/cmdline')
        if proc.exists() and t['name'].encode() in proc.read_bytes():os.kill(t['pid'],signal.SIGTERM)
        t.update(status='paused',pause_reason='Incorrect world-seed interpretation; replaced with world19 agent-seed forks')
    atomic(path,trials)
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:cmd=p.read_bytes()
        except OSError:continue
        if b'globo_dreamer.survival_hard' in cmd and (b'publish' in cmd or b'monitor' in cmd):os.kill(int(p.parent.name),signal.SIGTERM)
        if b'globo_dreamer.terrain_worker' in cmd:os.kill(int(p.parent.name),signal.SIGTERM)
    atomic(Path('/mnt/obelisk-shared/kmckee90/globo_survival_hard')/'pods'/f'{args.pod}.json',dict(time=time.time(),pod=args.pod,trials=trials))
    print('Archived and requested saved stop',flush=True)
def export():
    for i in range(1,5):
        name=f'lifetime2_14_conv_r{i}_s19';src=STATE/'runs'/name;dst=ROOT/'sources'/name
        if (dst/'READY').exists():continue
        assert read(src/'status.json')['running'] is False
        latest=(src/'ckpt/latest').read_text().strip();assert (src/'ckpt'/latest/'done').exists()
        dst.mkdir(parents=True,exist_ok=True)
        shutil.copytree(src/'ckpt'/latest,dst/'ckpt'/latest,dirs_exist_ok=True)
        (dst/'ckpt/latest').write_text(latest)
        shutil.copytree(src/'mf_replay',dst/'mf_replay',dirs_exist_ok=True)
        shutil.copy2(src/'status.json',dst/'status.json')
        (dst/'environment').mkdir(exist_ok=True);(dst/'live').mkdir(exist_ok=True)
        for name2 in ('actions.json','input_protocol.json','lifetime.json','memory.npz','achievements.json'):
            if (src/'environment'/name2).exists():shutil.copy2(src/'environment'/name2,dst/'environment'/name2)
        if (src/'live/visitation.json').exists():shutil.copy2(src/'live/visitation.json',dst/'live/visitation.json')
        atomic(dst/'READY',dict(source=name,checkpoint=latest));print('EXPORTED',name,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['stop','export']);p.add_argument('--pod');args=p.parse_args()
    stop() if args.command=='stop' else export()
