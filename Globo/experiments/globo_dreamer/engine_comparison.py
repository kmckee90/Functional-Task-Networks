"""Fresh corrected-engine experiment vs the archived four 14-action runs."""
import argparse
from . import action_comparison as base
from .phase2 import atomic, detach, publish, STATE

OLD = base.SHARED
old_experiments = base.experiments
base.SHARED = OLD.with_name('globo_engine_v2')
base.OUT = base.SHARED / 'monitor'
base.FULL = 'FTN-M2E · previous engine'
base.REDUCED = 'FTN-M2E · corrected inputs + HP20'
base.COLORS = {base.FULL:'#72b7ff', base.REDUCED:'#7de1bf'}
base.INTRO = '''<h1>FTN-M2E · corrected engine and animal health</h1>
<p>Four completely fresh runs versus the previous four 14-action runs.
World 19, matched agent seeds 1901–1904 and spawn. New runs use corrected
dig/place input events and an animal health ceiling of 20. Both groups use
the same FTN-M2E architecture and learning settings; no Dreamer or extrinsic reward.</p>
<p>Weights, optimizer, replay, frontier and density memory start empty/untrained.
Runs have no step limit. Charts show mean and min/max, clipped to the shortest
trajectory across both groups. This compares the combined environment changes,
not an isolated health or input ablation. Animal kills trigger saved videos.</p>
<p>The runs were subsequently checkpointed and resumed with single-click placement
and room for four bulky blocks per stack. No starting blocks are granted.
Each run records its changeover step in single_place_restart.json.</p>
<p><a href="previous/">Previous 14-action experiment (archived monitor)</a></p>'''

def experiments():
    return [dict(t, name=f'enginev2_14_conv_r{i+1}_s19', model=base.REDUCED,
                 single_place=True, block_stack_capacity=4,
                 phase='engine_v2', clip_settings=str(base.SHARED/'clip_settings.json'))
            for i,t in enumerate(old_experiments())]

def archived():
    return [dict(t, model=base.FULL, status='archived') for t in old_experiments()]

base.experiments = experiments
base.archived = archived

def launch():
    shared=base.SHARED
    shared.mkdir(parents=True,exist_ok=True)
    (shared/'runs').mkdir(exist_ok=True)
    base.OUT.mkdir(exist_ok=True)
    previous=base.OUT/'previous'
    if not previous.exists():previous.symlink_to(OLD/'monitor',target_is_directory=True)
    terrain=shared/'terrain'
    if not terrain.exists():terrain.symlink_to(OLD/'terrain',target_is_directory=True)
    for t in archived():
        dst=shared/'runs'/t['name']
        if not dst.exists():dst.symlink_to(OLD/'runs'/t['name'],target_is_directory=True)
    path=STATE/'continuous_runs/state.json'
    if path.exists():raise RuntimeError('Archive old supervisor state before fresh launch')
    if any((STATE/'runs'/t['name']).exists() for t in experiments()):
        raise RuntimeError('Fresh experiment directories already exist')
    atomic(shared/'experiments.json',dict(current=experiments(),reference=archived()))
    atomic(shared/'clip_settings.json',dict(frames=192))
    atomic(path,experiments())
    for mod,args,log in [('globo_dreamer.continuous_runs',(),'continuous_runs/supervisor.log'),
                        ('globo_dreamer.engine_comparison',('publish',),'engine_publish.log'),
                        ('globo_dreamer.engine_comparison',('monitor',),'engine_monitor.log')]:
        print(mod,detach(mod,log,*args),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('launch','publish','monitor'))
    p.add_argument('--port',type=int,default=8014);a=p.parse_args()
    if a.command=='launch':launch()
    elif a.command=='publish':publish('globo-actions','engine_v2',base.SHARED)
    else:base.monitor(a.port)
