"""Fresh hard-survival FTN-M2E, natural day/night, slow movement; two worlds."""
import argparse,time,os
from pathlib import Path
from .phase2 import STATE,read,atomic,detach,publish
from .lifetime_comparison import experiments as templates
from . import action_comparison as base
SHARED=Path('/mnt/obelisk-shared/kmckee90/globo_natural_survival')
PODS={'globo-actions':19,'globo1':101}

def experiments(pod):
    world=PODS[pod]
    return [dict(t,name=f'natural_hard_w{world}_a{2901+i}',pod=pod,gpu=i,
        seed=world,agent_seed=2901+i,pair=i+1,
        spawn='12.0,39.1,120.0' if world==19 else '16.0,58.1,118.0',
        model=f'FTN-M2E · world {world}',phase='natural_survival',status='pending',attempts=0,
        mf_gamma=.90,mf_offline_updates=50,survival=True,survival_novelty_scale=1.,fork_step=0,
        hard_survival=True,natural_cycle=True,move_speed=2.5,move_accel=2.,life_capture=True,
        artifact_root=str(SHARED),clip_settings=str(SHARED/'clip_settings.json'),
        life_settings=str(SHARED/'life_settings.json')) for i,t in enumerate(templates())]

def launch(pod):
    specs=experiments(pod)
    for t in specs:
        assert not (STATE/'runs'/t['name']).exists()
        assert not (SHARED/'runs'/t['name']).exists()
    old=read(STATE/'continuous_runs/state.json',[])
    for t in old:
        proc=Path(f'/proc/{t.get("pid")}/cmdline')
        assert not proc.exists() or t['name'].encode() not in proc.read_bytes(),'Old learner still active'
    pidfile=STATE/'continuous_runs/supervisor.pid'
    if pidfile.exists():
        proc=Path('/proc')/pidfile.read_text().strip()/'cmdline'
        assert not proc.exists() or b'globo_dreamer.continuous_runs' not in proc.read_bytes()
    for t in specs:
        root=SHARED/'runs'/t['name'];root.mkdir(parents=True)
        (STATE/'runs'/t['name']).symlink_to(root,target_is_directory=True)
        atomic(root/'initialization.json',dict(fresh=True,checkpoint=None,world_seed=t['seed'],agent_seed=t['agent_seed']))
    if not (SHARED/'clip_settings.json').exists():atomic(SHARED/'clip_settings.json',dict(frames=192))
    if not (SHARED/'life_settings.json').exists():atomic(SHARED/'life_settings.json',dict(minimum_minutes=0))
    atomic(SHARED/'specs'/f'{pod}.json',specs)
    path=STATE/'continuous_runs/state.json'
    atomic(path.with_name(f'archived_before_natural_{int(time.time())}.json'),old)
    atomic(path,specs)
    print('supervisor',detach('globo_dreamer.continuous_runs','continuous_runs/supervisor.log'),flush=True)
    print('publisher',detach('globo_dreamer.natural_survival','natural_publish.log','publish','--pod',pod),flush=True)
    if pod=='globo-actions':
        # Existing terrain tiles are seed-matched and independent of agents.
        terrain=SHARED/'terrain'
        if not terrain.exists():terrain.symlink_to(Path('/mnt/obelisk-shared/kmckee90/globo_lifetime_v2/terrain'),target_is_directory=True)
        os.environ.update(GLOBO_TERRAIN_SHARED=str(SHARED),GLOBO_TERRAIN_SEEDS='19,101',GLOBO_TERRAIN_DIRECTORY='terrain_natural_survival',GLOBO_TERRAIN_PORT_BASE='32000')
        print('terrain',detach('globo_dreamer.terrain_worker','natural_terrain.log'),flush=True)
        print('monitor',detach('globo_dreamer.natural_survival','natural_monitor.log','monitor'),flush=True)

def monitor(port):
    base.SHARED=SHARED;base.OUT=SHARED/'monitor';base.archived=lambda:[]
    base.COLORS={'FTN-M2E · world 19':'#7de1bf','FTN-M2E · world 101':'#d695ff'}
    def current():
        result=[]
        for pod in PODS:
            actual={t['name']:t for t in read(SHARED/'pods'/f'{pod}.json').get('trials',[])}
            result.extend(dict(t,**actual.get(t['name'],{})) for t in read(SHARED/'specs'/f'{pod}.json',[]))
        return result
    base.CURRENT_TRIALS=current
    base.INTRO='''<h1>FTN-M2E · natural-cycle hard survival</h1>
<p>Eight fresh agents: seeds 2901–2904 on each of worlds 19 and 101. No checkpoint
initialization; weights start untrained and replay, frontier and density memories start empty.</p>
<p>γ=.90 in both Q streams; 50 offline updates per round. Original explorer novelty
scales and goal bonus are retained, with an additional death penalty equal to the
unscaled goal bonus. Hard survival: original animal HP, 4× hunger/thirst depletion
and starvation/dehydration damage. Energy rates and recovery gains remain unchanged.</p>
<p>Clouds and weather visuals enabled, day/night cycle restored. Each life starts at
midday. Walk speed 2.5, ground/air acceleration 2. Same 14 actions, stationary jump,
single-click placement, four-block stacks, empty starting block inventory and
HUD feedback. Memory and TD values continue across lives. No step limit.</p>
<p>Curves show per-world means and min/max ranges; lifespans show raw individual
completed lives. Use the entire-life video setting below to choose the minimum
retained game-time duration in integer minutes. Zero keeps every life.</p>
<p><a href="http://localhost:8016/">Previous survival sweep</a> ·
<a href="http://localhost:8017/">Previous hard-survival experiment</a></p>'''
    base.monitor(port)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['launch','publish','monitor'])
    p.add_argument('--pod',choices=list(PODS));p.add_argument('--port',type=int,default=8018);a=p.parse_args()
    if a.command=='launch':launch(a.pod)
    elif a.command=='publish':publish(a.pod,'natural_survival',SHARED)
    else:monitor(a.port)
