"""Fresh 18-action survival explorers; previous natural-cycle runs are references."""
import argparse
import os
import signal
import time
from pathlib import Path
from .phase2 import STATE, read, atomic, detach, publish
from . import natural_survival as previous
from . import action_comparison as base

SHARED = Path('/mnt/obelisk-shared/kmckee90/globo_ability_survival')
PODS = previous.PODS

def experiments(pod):
    return [dict(t, name=t['name'].replace('natural_hard_', 'abilities18_'),
        model=f'18 actions · world {t["seed"]}', phase='ability_survival',
        reduced_actions=False, inventory_actions=True, block_stack_capacity=0,
        collected_stack_capacity=2, artifact_root=str(SHARED),
        clip_settings=str(SHARED/'clip_settings.json'),
        life_settings=str(SHARED/'life_settings.json')) for t in previous.experiments(pod)]

def references():
    result=[]
    for pod in PODS:
        for t in read(previous.SHARED/'specs'/f'{pod}.json',[]):
            result.append(dict(t, model=f'Previous 14 actions · world {t["seed"]}', status='archived'))
    return result

def launch(pod):
    old=read(STATE/'continuous_runs/state.json',[])
    for t in old:
        proc=Path('/proc')/str(t.get('pid'))/'cmdline'
        assert not proc.exists() or t['name'].encode() not in proc.read_bytes(), 'Old learner still saving'
    specs=experiments(pod)
    # Freeze the prior publisher before replacing the authoritative run list.
    for proc in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=proc.read_bytes().split(b'\0')
            if b'globo_dreamer.natural_survival' in args and b'publish' in args:
                os.kill(int(proc.parent.name),signal.SIGTERM)
        except (OSError,ProcessLookupError):pass
    snapshot=read(previous.SHARED/'pods'/f'{pod}.json')
    snapshot['trials']=old
    atomic(previous.SHARED/'pods'/f'{pod}.json',snapshot)
    for t in specs:
        assert not (STATE/'runs'/t['name']).exists()
        assert not (SHARED/'runs'/t['name']).exists()
    for t in specs:
        root=SHARED/'runs'/t['name'];root.mkdir(parents=True)
        (STATE/'runs'/t['name']).symlink_to(root,target_is_directory=True)
        atomic(root/'initialization.json',dict(fresh=True,checkpoint=None,world_seed=t['seed'],agent_seed=t['agent_seed']))
    atomic(SHARED/'specs'/f'{pod}.json',specs)
    path=STATE/'continuous_runs/state.json'
    atomic(path.with_name(f'archived_before_abilities_{int(time.time())}.json'),old)
    if pod=='globo-actions':
        for name in ('clip_settings.json','life_settings.json'):
            atomic(SHARED/name,read(previous.SHARED/name))
        (SHARED/'terrain').symlink_to(previous.SHARED/'terrain',target_is_directory=True)
        for t in references():
            (SHARED/'runs'/t['name']).symlink_to(previous.SHARED/'runs'/t['name'],target_is_directory=True)
    atomic(path,specs)
    print('supervisor',detach('globo_dreamer.continuous_runs','continuous_runs/supervisor.log'),flush=True)
    print('publisher',detach('globo_dreamer.ability_survival','ability_publish.log','publish','--pod',pod),flush=True)
    if pod=='globo-actions':
        print('monitor',detach('globo_dreamer.ability_survival','ability_monitor.log','monitor'),flush=True)

def monitor(port):
    base.SHARED=SHARED;base.OUT=SHARED/'monitor';base.archived=references
    base.COLORS={'18 actions · world 19':'#7de1bf','18 actions · world 101':'#d695ff',
        'Previous 14 actions · world 19':'#568d80','Previous 14 actions · world 101':'#826a99'}
    def current():
        result=[]
        for pod in PODS:
            actual={t['name']:t for t in read(SHARED/'pods'/f'{pod}.json').get('trials',[])}
            result.extend(dict(t,**actual.get(t['name'],{})) for t in read(SHARED/'specs'/f'{pod}.json',[]))
        return result
    base.CURRENT_TRIALS=current
    base.INTRO='''<h1>FTN-M2E · egg laying, sleep and four-slot inventory</h1>
<p>Eight fresh agents: agent seeds 2901–2904 on each of worlds 19 and 101,
matched against the previous natural-cycle runs. No checkpoint initialization.</p>
<p>18 actions: the previous 14 plus selection of slots 1–4. Slots 1–2 hold
collected items (maximum two per stack; single-item tools remain single).
Slot 3 is egg laying; slot 4 is sleep. Both abilities accept dig/use and
place/use. Normal resource and location requirements apply. No sneak actions.</p>
<p>Otherwise unchanged: γ=.90, 50 offline updates, original novelty scales,
death penalty equal to the goal bonus, hard survival, original animal HP,
natural day/night and clouds, speed 2.5, acceleration 2, stationary jump,
single-click placement, empty collected-item inventory. No step limit.</p>
<p>Means and min/max bounds compare the new runs with the saved previous batch.
Life-video thresholds apply only to newly completed recordings; saved videos
remain available. <a href="http://localhost:8018/">Previous experiment</a>.</p>'''
    base.monitor(port)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['launch','publish','monitor'])
    p.add_argument('--pod',choices=list(PODS));p.add_argument('--port',type=int,default=8019)
    a=p.parse_args()
    if a.command=='launch':launch(a.pod)
    elif a.command=='publish':publish(a.pod,'ability_survival',SHARED)
    else:monitor(a.port)
