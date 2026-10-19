"""Twelve fresh Plan2Explore replicates, distinct learner seeds, world 19."""
import argparse,os,time
from pathlib import Path
from .phase2 import STATE,SHARED as PHASE2,read,atomic,detach,stop_supervisor,publish

SHARED=Path('/mnt/obelisk-shared/kmckee90/globo_phase5')
SEEDS=tuple(range(1901,1913))

def experiments():
    trials=[]
    for i,seed in enumerate(SEEDS):
        pod='globo-dreamer' if i<8 else ('globo1' if i<10 else 'globo2')
        t=dict(name=f'phase5_p2e_r{i+1}_s19',model='Plan2Explore',phase='phase5',pair=i+1,
            seed=19,agent_seed=seed,pod=pod,gpu=i if i<8 else (i-8)%2,
            spawn='12.0,39.1,120.0',status='pending',attempts=0,achievements=True,
            render_size=256,software_renderer=True,weight=0.,visitation=True,
            plan2explore=True,dreamer_size='size12m',decoder_depth=16,p2e_reward_scale=10000.)
        trials.append(t)
    return trials

def launch(pod):
    SHARED.mkdir(parents=True,exist_ok=True)
    # Reuse the original terrain cache and its CPU mapper, never training worlds.
    terrain=SHARED/'terrain'
    if not terrain.exists():
        try:terrain.symlink_to(PHASE2/'terrain',target_is_directory=True)
        except FileExistsError:pass
    settings=SHARED/'clip_settings.json'
    if pod=='globo-dreamer' and not settings.exists():atomic(settings,dict(frames=192))
    stop_supervisor();path=STATE/'continuous_runs/state.json';trials=read(path,[])
    for t in experiments():
        if t['pod']!=pod or any(x['name']==t['name'] for x in trials):continue
        if (STATE/'runs'/t['name']).exists():raise RuntimeError('Fresh trial already has a run directory')
        trials.append(t)
    atomic(path,trials)
    print('supervisor',detach('globo_dreamer.continuous_runs','continuous_runs/supervisor.log'),flush=True)
    p=STATE/'phase5_publisher.json';old=read(p);proc=Path(f'/proc/{old.get("pid")}/cmdline')
    if not proc.exists() or b'globo_dreamer.phase5' not in proc.read_bytes():
        pid=detach('globo_dreamer.phase5','phase5_publisher.log','publish','--pod',pod)
        atomic(p,dict(pid=pid,pod=pod));print('publisher',pid,flush=True)
    if pod=='globo-dreamer':atomic(SHARED/'experiments.json',experiments())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('launch','publish'));p.add_argument('--pod',required=True)
    a=p.parse_args()
    if a.command=='launch':launch(a.pod)
    else:publish(a.pod,'phase5',SHARED)
