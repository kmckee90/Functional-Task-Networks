"""Fresh HUD/continuing-life runs vs the previous corrected-engine cohort."""
import argparse
from . import action_comparison as base
from .phase2 import STATE,read,atomic,detach,publish

OLD=base.SHARED.with_name('globo_engine_v2')
seed_configs=base.experiments
base.SHARED=OLD.with_name('globo_lifetime_v2')
base.OUT=base.SHARED/'monitor'
base.FULL='FTN-M2E · previous cohort'
base.REDUCED='FTN-M2E · γ=0.90367 · 50 updates'
HIGH50='FTN-M2E · γ=0.99 · 50 updates'
HIGH200='FTN-M2E · γ=0.99 · 200 updates'
SURVIVAL='Survival · γ=0.99 · 50 updates'
base.COLORS={base.REDUCED:'#7de1bf',HIGH50:'#ffbd66',HIGH200:'#d695ff',SURVIVAL:'#ff6e82'}
base.INTRO='''<h1>FTN-M2E · HUD feedback and learning across lives</h1>
<p>Four fresh runs on world 19, seeds 1901–1904, matched to the previous four runs.
Action 5 is nondirectional jump. Normalized HUD and inventory-count feedback
enters the reservoir. Death and respawn preserve reservoir, feedback and FTN state;
novelty value bootstraps across lives. No extrinsic reward or death penalty.</p>
<p>Single-click placement, four-block stack capacity, 20-HP animals and empty
starting block slots are retained. All weights, replay, frontier and density
memories start fresh. No step limit. All plot axes extend to the furthest run.
Each condition's mean and min/max bands stop at its shortest run; individual traces retain their full extent.</p>
<p><a href="previous/">Previous experiment</a> · <a href="changes.md">Experiment changes</a></p>'''
base.INTRO += '''<p>Discount/update comparison: the four original runs continue unchanged
(γ=0.90367, 50 offline updates). Four fresh runs on globo1 use γ=0.99:
two with 50 updates and two with 200 updates per training round. Rounds still occur
every 50 interactions after 500 transitions. Each new condition uses agent seeds
1901 and 1902 in world 19. Colors identify conditions; thick lines are means and
bands are min/max. The previous experiment link retains the older engine comparison.</p>'''

def experiments():
    return [dict(t,name=f'lifetime2_14_conv_r{i+1}_s19',model=base.REDUCED,
                 mf_gamma=0.9036665,mf_offline_updates=50,
                 phase='lifetime_v2',lifetime_v2=True,single_place=True,
                 block_stack_capacity=4,clip_settings=str(base.SHARED/'clip_settings.json'))
            for i,t in enumerate(seed_configs())]

def archived():
    return [dict(t,name=f'enginev2_14_conv_r{i+1}_s19',model=base.FULL,status='archived')
            for i,t in enumerate(seed_configs())]
base.experiments=experiments;base.archived=lambda:[]

def discount_experiments():
    templates=experiments()
    return [dict(templates[i%2],name=f'lifetime2_g99_u{updates}_r{i%2+1}_s19',
                 model=HIGH50 if updates==50 else HIGH200,pod='globo1',gpu=i,
                 mf_gamma=.99,mf_offline_updates=updates)
            for i,updates in enumerate((50,50,200,200))]

def current_trials():
    trials=[]
    for pod,expected in [('globo-actions',experiments()),('globo1',discount_experiments())]:
        actual={t['name']:t for t in read(base.SHARED/'pods'/f'{pod}.json').get('trials',[])}
        for t in expected:
            merged=dict(t,**actual.get(t['name'],{}));merged['model']=t['model']
            if merged.get('status')=='paused':continue
            trials.append(merged)
        for t in actual.values():
            if t.get('survival'):
                trials.append(dict(t,model=t.get('model',SURVIVAL)))
    palette=['#ff6e82','#8cbcff','#7de1bf','#ffbd66','#d695ff','#d5e76c','#ff96d3','#65d4e8']
    for i,model in enumerate(dict.fromkeys(t['model'] for t in trials if t.get('survival'))):
        base.COLORS[model]=palette[i%len(palette)]
    return trials

base.CURRENT_TRIALS=current_trials
base.INTRO = """<h1>FTN-M2E · survival reward experiments</h1>
<p>Eight survival-reward runs on world 19, forked from their corresponding explorer
checkpoints and agent seeds. Original explorer checkpoints are archived separately.
The two original γ=.99 / 50-update survival runs continue unchanged; six new forks
vary updates, discount, learning rate and ordinary novelty reward scale.</p>
<p>The goal reward retains its scale; ordinary novelty is multiplied by ¼, ⅛ or ¹⁄₁₆.
Death receives a negative target matching the unscaled batch goal reward.
Values bootstrap across lives. New forks apply their discount to both Q streams.
Historical exploration replay is retained; death labels start at each fork.
LR×.25 means 4.7487e-5, versus the original 1.89949e-4.</p>
<p>Spatial/achievement counts include inherited progress. Lifespans show individual
completed lives, not averages. Separate survival plots use observations since each
fork. Whole-life videos are kept only above 10 game minutes.</p>"""


def launch_discount():
    import time
    from pathlib import Path
    trials=discount_experiments()
    if any((STATE/'runs'/t['name']).exists() for t in trials):
        raise RuntimeError('Fresh discount runs already exist; resume their supervisor instead')
    pidfile=STATE/'continuous_runs/supervisor.pid'
    if pidfile.exists():
        cmd=Path(f'/proc/{pidfile.read_text().strip()}/cmdline')
        if cmd.exists() and b'globo_dreamer.continuous_runs' in cmd.read_bytes():
            raise RuntimeError('Existing supervisor must be stopped before replacing run list')
    path=STATE/'continuous_runs/state.json'
    if path.exists():path.rename(path.with_name(f'before_discount_{int(time.time())}.json'))
    atomic(path,trials)
    atomic(base.SHARED/'experiments.json',dict(current=experiments()+trials,reference=archived()))
    print('supervisor',detach('globo_dreamer.continuous_runs','continuous_runs/supervisor.log'),flush=True)
    print('publisher',detach('globo_dreamer.lifetime_comparison','lifetime_discount_publish.log',
                            'publish','--pod','globo1'),flush=True)

def launch():
    shared=base.SHARED;shared.mkdir(parents=True,exist_ok=True)
    (shared/'runs').mkdir(exist_ok=True);base.OUT.mkdir(exist_ok=True)
    for dst,src in [(shared/'terrain',OLD/'terrain'),(base.OUT/'previous',OLD/'monitor')]:
        if not dst.exists():dst.symlink_to(src,target_is_directory=True)
    for t in archived():
        dst=shared/'runs'/t['name']
        if not dst.exists():dst.symlink_to(OLD/'runs'/t['name'],target_is_directory=True)
    path=STATE/'continuous_runs/state.json'
    if path.exists() or any((STATE/'runs'/t['name']).exists() for t in experiments()):
        raise RuntimeError('Archive supervisor state; new run directories must not exist')
    from pathlib import Path
    (base.OUT/'changes.md').write_text(Path(__file__).with_name('LIFETIME_V2.md').read_text())
    atomic(shared/'experiments.json',dict(current=experiments(),reference=archived()))
    atomic(shared/'clip_settings.json',dict(frames=192));atomic(path,experiments())
    for mod,args,log in [('globo_dreamer.continuous_runs',(),'continuous_runs/supervisor.log'),
                        ('globo_dreamer.lifetime_comparison',('publish',),'lifetime_publish.log'),
                        ('globo_dreamer.lifetime_comparison',('monitor',),'lifetime_monitor.log')]:
        print(mod,detach(mod,log,*args),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('launch','launch-discount','publish','monitor'))
    p.add_argument('--pod',default='globo-actions')
    p.add_argument('--port',type=int,default=8016);a=p.parse_args()
    if a.command=='launch':launch()
    elif a.command=='launch-discount':launch_discount()
    elif a.command=='publish':publish(a.pod,'lifetime_v2',base.SHARED)
    else:base.monitor(a.port)
