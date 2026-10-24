"""Four agent-seed survival forks on world 19, with harder resource dynamics."""
import argparse
import os
import shutil
import time
from pathlib import Path
from .phase2 import STATE, read, atomic, detach, publish

SHARED=Path('/mnt/obelisk-shared/kmckee90/globo_survival_hard_s19')
WORLDS={'globo2':(19,),'globo3':(19,)}
AGENT_SEEDS={'globo2':(1901,1902),'globo3':(1903,1904)}

def screen(pod):
    import sys
    sys.path.insert(0,str(STATE/'vendor/dreamerv3'))
    from .env import Globo
    os.environ.update(GLOBO_SOFTWARE_RENDERER='1',GLOBO_SINGLE_PLACE='1',
        GLOBO_LIFETIME_V2='1',GLOBO_BLOCK_STACK_CAPACITY='4',GLOBO_HARD_SURVIVAL='1')
    for seed in WORLDS[pod]:
        root=SHARED/'screen'/str(seed)
        if read(root/'result.json').get('accepted'):continue
        scan=Globo(root/'scan',seed=seed,video_every=0,episode_steps=0,
            spawn='11,45,121',find_spawn=True,spawn_scan_ceiling=256,screen_invincible=True,reduced_actions=True)
        candidates=[]
        try:
            for i in range(1000):
                scan.step(dict(reset=i==0,action=0))
                candidate_file=scan.telemetry.with_name('spawn_candidates.json')
                candidates=read(candidate_file,[])
                if candidate_file.exists():break
        finally:scan.close()
        if not candidates:raise RuntimeError(f'No safe spawn for {seed}')
        c=candidates[0];spawn=','.join(str(c[k]) for k in ('x','y','z'))
        env=Globo(root/'check',seed=seed,spawn=spawn,video_every=0,episode_steps=0,
            render_size=256,reduced_actions=True)
        rows=[]
        try:
            for i in range(100):
                obs=env.step(dict(reset=i==0,action=0));rows.append(dict(env.record))
                if obs['is_last']:break
            from PIL import Image
            Image.fromarray(env.human_frame).save(root/'spawn.png')
            manifest=read(Path(env.engine.mt.run_dir)/'games/globo/survival_variant.json')
            assert manifest['food_water_depletion_multiplier']==4
        finally:env.close()
        result=dict(seed=seed,spawn=spawn,ground=c['ground'],observations=len(rows),
            initial_hp=rows[0]['hp'],final_hp=rows[-1]['hp'],variant=manifest,
            accepted=len(rows)==100 and min(r['hp'] for r in rows)>900)
        atomic(root/'result.json',result);print(result,flush=True)
        if not result['accepted']:raise RuntimeError(f'Unsafe start: {result}')

def trials(pod):
    from .lifetime_comparison import experiments
    from .search60 import FIXED
    templates=experiments()
    out=[]
    for i,agent_seed in enumerate(AGENT_SEEDS[pod]):
        template=templates[agent_seed-1901]
        t=dict(template,name=f'hard_survival_s19_a{agent_seed}',pod=pod,gpu=i,
            seed=19,agent_seed=agent_seed,phase='survival_hard',status='pending',attempts=0,
            model=f'Survival · agent {agent_seed}',survival=True,survival_novelty_scale=.25,
            mf_gamma=.99,mf_offline_updates=50,mf_lr=FIXED['mf_lr'],
            hard_survival=True,life_capture=True,artifact_root=str(SHARED),
            source_run=template['name'],clip_settings=str(SHARED/'clip_settings.json'))
        out.append(t)
    return out

def launch(pod):
    specs=trials(pod)
    pidfile=STATE/'continuous_runs/supervisor.pid'
    if pidfile.exists():
        proc=Path('/proc')/pidfile.read_text().strip()/'cmdline'
        assert not proc.exists() or b'globo_dreamer.continuous_runs' not in proc.read_bytes(),'Supervisor already active'
    for t in specs:
        assert not (SHARED/'runs'/t['name']).exists(),f'Fork already exists: {t["name"]}'
        assert read(SHARED/'sources'/t['source_run']/'READY'),'Export not ready'
    for t in specs:
        src=SHARED/'sources'/t['source_run'];dst=SHARED/'runs'/t['name']
        dst.mkdir(parents=True);latest=(src/'ckpt/latest').read_text().strip()
        assert (src/'ckpt'/latest/'done').exists()
        shutil.copytree(src/'mf_replay',dst/'mf_replay',copy_function=os.link)
        shutil.copytree(src/'ckpt'/latest,dst/'ckpt'/latest)
        (dst/'ckpt/latest').write_text(latest)
        step=read(src/'status.json')['step'];t['fork_step']=step
        (dst/'environment').mkdir();(dst/'live').mkdir()
        for file in ('actions.json','input_protocol.json','lifetime.json','memory.npz','achievements.json'):
            if (src/'environment'/file).exists():shutil.copy2(src/'environment'/file,dst/'environment'/file)
        if (src/'live/visitation.json').exists():shutil.copy2(src/'live/visitation.json',dst/'live/visitation.json')
        atomic(dst/'fork.json',dict(source=t['source_run'],checkpoint=latest,step=step,
            source_world=19,target_world=19,agent_seed=t['agent_seed'],reset_world_metrics=False,time=time.time(),
            gamma=.99,updates=50,lr=t['mf_lr'],novelty_scale=.25,
            source_is_exploration=True,animal_hp='original',resource_loss_multiplier=4,
            shortage_damage_multiplier=4,replay='retained immutable chunks; private future writes',
            checkpoint_directory=str(dst/'ckpt')))
        alias=STATE/'runs'/t['name'];alias.parent.mkdir(exist_ok=True)
        alias.symlink_to(dst,target_is_directory=True)
        print('Forked',t['source_run'],'to',t['name'],step,flush=True)
    path=STATE/'continuous_runs/state.json';path.parent.mkdir(exist_ok=True)
    if path.exists():path.rename(path.with_name(f'before_hard_survival_{int(time.time())}.json'))
    atomic(path,specs);atomic(SHARED/'specs'/f'{pod}.json',specs)
    if not (SHARED/'clip_settings.json').exists():atomic(SHARED/'clip_settings.json',dict(frames=192))
    print('supervisor',detach('globo_dreamer.continuous_runs','continuous_runs/supervisor.log'))
    print('publisher',detach('globo_dreamer.survival_hard','hard_publish.log','publish','--pod',pod))
    os.environ.update(GLOBO_TERRAIN_SHARED=str(SHARED),
        GLOBO_TERRAIN_SEEDS=','.join(map(str,WORLDS[pod])),GLOBO_TERRAIN_DIRECTORY='terrain_hard_survival')
    if pod=='globo2':print('terrain',detach('globo_dreamer.terrain_worker','hard_terrain.log'))

def monitor(port):
    from . import action_comparison as base
    from . import phase4_monitor as charts
    base.SHARED=SHARED;base.OUT=SHARED/'monitor';base.archived=lambda:[]
    history=charts.history
    def local_history(t):
        fork=t.get('fork_step',0)
        return [dict(r,step=r['step']-fork) for r in history(t) if r['step']>=fork]
    charts.history=local_history
    def current():
        result=[]
        for pod in WORLDS:
            expected=read(SHARED/'specs'/f'{pod}.json',[])
            actual={t['name']:t for t in read(SHARED/'pods'/f'{pod}.json').get('trials',[])}
            result.extend(dict(t,**actual.get(t['name'],{})) for t in expected)
        for t,color in zip(result,['#7de1bf','#ffbd66','#d695ff','#8cbcff']):base.COLORS[t['model']]=color
        return result
    base.CURRENT_TRIALS=current
    base.INTRO='''<h1>FTN-M2E · harder survival on world 19</h1>
<p>Four independent exploration checkpoints, agent seeds 1901–1904, all continuing on the
same world 19 and original spawn. These are the four original 14-action exploration agents.
No trained survival checkpoint is used. γ=.99, 50 offline updates, original learning rate,
¼ ordinary novelty reward; goal reward is unchanged and death cost matches its scale.</p>
<p>Animals retain their original species health. Hunger/thirst depletion and resource-shortage
damage run at 4× the previous rates; resource gains and other damage are unchanged.
The 14 actions, single-click placement, four-block stack capacity, HUD feedback and memory across lives remain.</p>
<p>Weights, exploration replay/frontier/density memories, world coverage and achievement
accounting are retained. All plots use observations since the fork; run cards and
video labels retain the inherited training counter. Lifespans are individual completed lives.
Entire-life videos are retained only above 10 game minutes.</p>
<p><a href="http://localhost:8016/">Previous survival experiment</a></p>'''
    base.monitor(port)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['screen','launch','publish','monitor'])
    p.add_argument('--pod',choices=list(WORLDS));p.add_argument('--port',type=int,default=8017);a=p.parse_args()
    if a.command=='screen':screen(a.pod)
    elif a.command=='launch':launch(a.pod)
    elif a.command=='publish':publish(a.pod,'survival_hard',SHARED)
    else:monitor(a.port)
