"""Paired six-replicate M2E/convFTN experiment, fresh weights, world 19."""
import argparse,os,time
from pathlib import Path
from .phase2 import STATE,SHARED as PHASE2,read,atomic,detach,stop_supervisor,publish

SHARED=Path('/mnt/obelisk-shared/kmckee90/globo_phase4')
SEEDS=(1901,1902,1903,1904,1905,1906)

def experiments():
    trials=[]
    for i,seed in enumerate(SEEDS):
        for variant in (0,1):
            if variant:
                pod='globo3' if i<4 else 'globo4'
                gpu=i if i<4 else i-4
            else:
                pod=('globo-dreamer','globo1','globo2')[i//2]
                gpu=6+i if i<2 else i%2
            t=dict(name=f'phase4_{"conv" if variant else "m2e"}_r{i+1}_s19',
                model='M2E+FTN' if variant else 'M2E',phase='phase4',pair=i+1,
                seed=19,agent_seed=seed,pod=pod,gpu=gpu,
                spawn='12.0,39.1,120.0',status='pending',attempts=0,frontier_rotation=False,achievements=True,
                render_size=256,software_renderer=True,weight=0.,visitation=True,
                model_free_only=True,scaled=True,reservoir_variant='tuned_v1',mf_replay_dtype='float16')
            if variant:t['m2e_ftn']='conv'
            if variant and i<4:t['event_capture']=True
            trials.append(t)
    return trials

def launch(pod):
    SHARED.mkdir(parents=True,exist_ok=True)
    # Reuse the original terrain cache and its CPU mapper, never training worlds.
    terrain=SHARED/'terrain'
    if not terrain.exists():terrain.symlink_to(PHASE2/'terrain',target_is_directory=True)
    settings=SHARED/'clip_settings.json'
    if not settings.exists():atomic(settings,dict(frames=192))
    stop_supervisor();path=STATE/'continuous_runs/state.json';trials=read(path,[])
    for t in experiments():
        if t['pod']!=pod or any(x['name']==t['name'] for x in trials):continue
        if (STATE/'runs'/t['name']).exists():raise RuntimeError('Fresh trial already has a run directory')
        trials.append(t)
    atomic(path,trials)
    print('supervisor',detach('globo_dreamer.continuous_runs','continuous_runs/supervisor.log'),flush=True)
    p=STATE/'phase4_publisher.json';old=read(p);proc=Path(f'/proc/{old.get("pid")}/cmdline')
    if not proc.exists() or b'globo_dreamer.phase4' not in proc.read_bytes():
        pid=detach('globo_dreamer.phase4','phase4_publisher.log','publish','--pod',pod)
        atomic(p,dict(pid=pid,pod=pod));print('publisher',pid,flush=True)
    if pod=='globo-dreamer':atomic(SHARED/'experiments.json',experiments())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('launch','publish'));p.add_argument('--pod',required=True)
    a=p.parse_args()
    if a.command=='launch':launch(a.pod)
    else:publish(a.pod,'phase4',SHARED)
