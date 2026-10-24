"""Phase 2 deployment and pod-owned artifact publication. Run only on pods."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

STATE=Path('/home/dev/globo_dreamer_state')
SHARED=Path('/mnt/obelisk-shared/kmckee90/globo_phase2')
RUNTIME=Path('/mnt/obelisk-shared/kmckee90/globo_runtime_v1')
KEEP={'m2eftn_flat_s19','m2eftn_conv_s19','m2eftn_conv_s101','m2eftn_waypoints_s101'}
MODELS=('Dreamer','Plan2Explore','M2E','M2E+convFTN')

def read(p,default=None):
    try:return json.loads(Path(p).read_text())
    except (OSError,ValueError):return {} if default is None else default

def atomic(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def experiments():
    out=[]
    assignments={'Dreamer':('globo-dreamer',(1,4)),'Plan2Explore':('globo-dreamer',(5,6)),
                 'M2E':('globo1',(0,1)),'M2E+convFTN':('globo2',(0,1))}
    for model,(pod,gpus) in assignments.items():
        for seed,gpu in zip((19,101),gpus):
            slug={'Dreamer':'dreamer','Plan2Explore':'p2e','M2E':'m2e','M2E+convFTN':'conv','M2E+mlpFTN':'flat'}[model]
            t=dict(name=f'phase2_{slug}_s{seed}',pod=pod,model=model,gpu=gpu,seed=seed,
                spawn='12.0,39.1,120.0' if seed==19 else '16.0,58.1,118.0',
                phase='phase2',status='pending',attempts=0,achievements=True,render_size=256,
                software_renderer=True,weight=0.,visitation=True)
            if model=='Dreamer':t.update(dreamer_baseline=True,scaled=True,decoder_depth=64)
            elif model=='Plan2Explore':t.update(plan2explore=True,dreamer_size='size12m',decoder_depth=16,p2e_reward_scale=10000.)
            else:t.update(model_free_only=True,scaled=True,reservoir_variant='tuned_v1',mf_replay_dtype='float16')
            if model.endswith('convFTN'):t['m2e_ftn']='conv'
            if model.endswith('mlpFTN'):t['m2e_ftn']='flat'
            out.append(t)
    return out

def python_env():
    return dict(os.environ,PYTHONPATH=f'{STATE}/packages:{STATE}/vendor/dreamerv3',
        PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',XLA_PYTHON_CLIENT_PREALLOCATE='false')

def detach(module,log,*args):
    with (STATE/log).open('a') as f:
        proc=subprocess.Popen([str(STATE/'venv/bin/python'),'-u','-m',module,*args],cwd=STATE,
            env=python_env(),stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True)
    return proc.pid

def stop_supervisor():
    p=STATE/'continuous_runs/supervisor.pid'
    if not p.exists():return
    pid=int(p.read_text());cmd=Path(f'/proc/{pid}/cmdline')
    if cmd.exists() and b'globo_dreamer.continuous_runs' in cmd.read_bytes():
        # SIGTERM would stop every child in finally; retain the chosen workers.
        os.kill(pid,signal.SIGKILL);time.sleep(1)

def preserve_phase1():
    root=STATE/'continuous_runs';stop_supervisor();trials=read(root/'state.json',[])
    backup=root/'phase1_before_phase2.json'
    if not backup.exists():atomic(backup,trials)
    for t in trials:
        if t.get('phase')=='phase2':continue
        if t['name'] not in KEEP and t['status'] in ('running','pending'):
            t['status']='paused'
            pid=t.get('pid');proc=Path(f'/proc/{pid}/cmdline')
            if pid and proc.exists() and t['name'].encode() in proc.read_bytes():os.kill(pid,signal.SIGTERM)
        elif t['name'] in KEEP:t['phase']='phase1_retained'
    atomic(root/'state.json',trials)
    atomic(root/'phase.json',dict(title='Phase 1 — retained explorers',phase=1))
    atomic(SHARED/'experiments.json',experiments())
    pid=detach('globo_dreamer.continuous_runs','continuous_runs/supervisor.log')
    print('Phase 1 supervisor',pid,'retaining',sorted(KEEP),flush=True)

def install():
    if not (RUNTIME/'READY').exists():raise RuntimeError('Shared runtime copy not ready')
    STATE.mkdir(exist_ok=True)
    for name in ('venv','python','vendor','software_egl.so'):
        p=STATE/name
        if not p.exists():p.symlink_to(RUNTIME/name,target_is_directory=name!='software_egl.so')
    (STATE/'packages').mkdir(exist_ok=True)
    p=STATE/'packages/globo_dreamer'
    if not p.exists():p.symlink_to(Path(__file__).resolve().parent,target_is_directory=True)
    print('Runtime installed',flush=True)

def launch(pod):
    root=STATE/'continuous_runs';root.mkdir(parents=True,exist_ok=True)
    stop_supervisor();trials=read(root/'state.json',[])
    for t in experiments():
        if t['pod']!=pod or any(x['name']==t['name'] for x in trials):continue
        if (STATE/'runs'/t['name']).exists():raise RuntimeError('Fresh run directory already exists: '+t['name'])
        trials.append(t)
    atomic(root/'state.json',trials)
    print('Supervisor',detach('globo_dreamer.continuous_runs','continuous_runs/supervisor.log'),flush=True)
    p=STATE/'phase2_publisher.pid';old=read(STATE/'phase2_publisher.json')
    proc=Path(f'/proc/{old.get("pid")}/cmdline')
    if not proc.exists() or b'globo_dreamer.phase2' not in proc.read_bytes():
        pid=detach('globo_dreamer.phase2','phase2_publisher.log','publish','--pod',pod)
        atomic(STATE/'phase2_publisher.json',dict(pid=pid,pod=pod))

PUBLISH_FILES=('status.json','architecture.json','launch.json','environment/achievements.json',
    'live/survival.json','live/history.jsonl','live/progress.json','live/video.json','live/obs.mp4','live/bw.mp4',
    'live/mask.html','live/mask_video.json','live/clip_consolidation.json','live/visitation.json',
    'live/human.mp4','live/agent_input.mp4','live/recent.mp4','live/mask_status.json','live/goal_status.json','live/active_goal.jpg','live/frontier_goal.jpg','live/offline_goal.mp4','live/events.jsonl')

def publish(pod,phase='phase2',shared=None):
    from .consolidate_clip import process
    import fcntl
    shared=SHARED if shared is None else shared
    with (STATE/f'{phase}_publish.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signatures={}
        while True:
            trials=[t for t in read(STATE/'continuous_runs/state.json',[]) if t.get('phase')==phase]
            for t in trials:
                run=STATE/'runs'/t['name'];target=shared/'runs'/t['name']
                if t.get('m2e_ftn') and (run/'live/video.json').exists():
                    try:
                        process(run/'live')
                        from .mask_diagnostics import publish as masks
                        masks(run/'live',read(run/'live/video.json'),t['model'],read(run/'status.json'))
                    except Exception as exc:print('Clip',t['name'],repr(exc),flush=True)
                elif t.get('model_free_only') and read(run/'live/video.json').get('exact_inputs'):
                    try:
                        from .split_human_video import publish as split
                        split(run/'live',read(run/'live/video.json'))
                    except Exception as exc:print('Split',repr(exc),flush=True)
                from .survival import summary
                try:atomic(run/'live/survival.json',summary(run))
                except Exception as exc:print('Survival',t['name'],repr(exc),flush=True)
                files=list(PUBLISH_FILES)
                if phase=='lifetime_v2' and pod=='globo1':
                    files.extend(str(p.relative_to(run)) for p in (run/'event_clips').glob('*')
                                 if p.suffix in ('.json','.mp4'))
                for rel in files:
                    src=run/rel;dst=target/rel
                    if src.resolve()==dst.resolve():continue
                    try:
                        stat=src.stat();signature=(stat.st_mtime_ns,stat.st_size)
                        if signatures.get(str(src))==signature:continue
                        dst.parent.mkdir(parents=True,exist_ok=True)
                        temp=dst.with_suffix(dst.suffix+'.tmp');shutil.copyfile(src,temp);temp.replace(dst)
                        signatures[str(src)]=signature
                    except FileNotFoundError:pass
                    except OSError as exc:print(str(src),repr(exc),flush=True)
            atomic(shared/'pods'/f'{pod}.json',dict(time=time.time(),pod=pod,trials=trials))
            time.sleep(20)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('preserve','install','launch','publish'));p.add_argument('--pod')
    args=p.parse_args()
    if args.command=='preserve':preserve_phase1()
    elif args.command=='install':install()
    elif args.command=='launch':launch(args.pod)
    else:publish(args.pod)
