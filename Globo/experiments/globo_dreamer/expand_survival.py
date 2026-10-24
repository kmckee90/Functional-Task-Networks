"""Checkpoint explorer sources, fork reward variants, preserve original runs."""
import argparse
import os
import shutil
import signal
import time
from pathlib import Path
from .phase2 import STATE,read,atomic,detach,stop_supervisor
from .search60 import FIXED

def conditions(pod):
    # (source, gamma, updates, LR multiplier, ordinary novelty multiplier)
    if pod=='globo-actions':
        return [(f'lifetime2_14_conv_r{i+1}_s19',*v) for i,v in enumerate([
            (.999,200,.25,.125),(.999,200,.25,.0625),
            (.995,200,1.,.25),(.999,50,.25,.125)])]
    return [('lifetime2_g99_u50_r1_s19',.995,50,.25,.0625),
            ('lifetime2_g99_u50_r2_s19',.999,200,1.,.25)]

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['pause','fork']);p.add_argument('--pod',required=True,choices=['globo-actions','globo1']);a=p.parse_args()
    path=STATE/'continuous_runs/state.json';trials=read(path);byname={t['name']:t for t in trials}
    specs=conditions(a.pod)
    if a.action=='pause':
        atomic(path.with_name(f'before_survival_expansion_{int(time.time())}.json'),trials)
        stop_supervisor()
        for name,*_ in specs:
            t=byname[name];assert not t.get('survival')
            proc=Path(f'/proc/{t.get("pid")}/cmdline')
            if proc.exists() and name.encode() in proc.read_bytes():os.kill(t['pid'],signal.SIGTERM)
            t.update(status='paused',paused_at=time.time(),pause_reason='Archived explorer source for survival fork')
        atomic(path,trials);print('Requested saved stops:',[s[0] for s in specs]);return
    for source,*_ in specs:
        t=byname[source];assert t['status']=='paused'
        proc=Path(f'/proc/{t.get("pid")}/cmdline')
        assert not proc.exists() or not proc.read_bytes(),f'Still saving {source}'
        assert read(STATE/'runs'/source/'status.json')['running'] is False
        assert not (STATE/'runs'/('survival_sweep_'+source)).exists()
    for source,gamma,updates,lr_mult,novelty in specs:
        t=byname[source];name='survival_sweep_'+source
        src=STATE/'runs'/source;dst=STATE/'runs'/name;dst.mkdir()
        latest=(src/'ckpt/latest').read_text().strip()
        assert (src/'ckpt'/latest/'done').exists()
        shutil.copytree(src/'mf_replay',dst/'mf_replay',copy_function=os.link)
        shutil.copytree(src/'ckpt'/latest,dst/'ckpt'/latest)
        (dst/'ckpt/latest').write_text(latest)
        (dst/'environment').mkdir();(dst/'live').mkdir()
        for file in ('actions.json','input_protocol.json','lifetime.json','memory.npz','achievements.json'):
            if (src/'environment'/file).exists():shutil.copy2(src/'environment'/file,dst/'environment'/file)
        if (src/'live/visitation.json').exists():shutil.copy2(src/'live/visitation.json',dst/'live/visitation.json')
        step=read(src/'status.json')['step'];lr=FIXED['mf_lr']*lr_mult
        label=f'Survival · γ={gamma:g} · {updates} updates · LR×{lr_mult:g} · novelty 1/{round(1/novelty)}'
        trial=dict(t,name=name,model=label,survival=True,source_run=source,fork_step=step,
            mf_gamma=gamma,mf_offline_updates=updates,mf_lr=lr,lr_multiplier=lr_mult,
            survival_novelty_scale=novelty,status='pending',attempts=0,retry_after=0,life_capture=True)
        for key in ('pid','paused_at','pause_reason','watchdog_term_at','watchdog_kill_at'):trial.pop(key,None)
        atomic(dst/'fork.json',dict(source=source,checkpoint=latest,step=step,time=time.time(),
            gamma=gamma,gamma_env=gamma,updates=updates,lr=lr,novelty_scale=novelty,
            death_cost='unscaled batch goal reward',goal_reward='unchanged',
            death_labels='new deaths only',checkpoint_directory=str(dst/'ckpt'),
            replay='immutable chunks hard-linked; independent manifests and future writes'))
        trials.append(trial);print(label,source,'->',name,'step',step,flush=True)
    atomic(path,trials)
    print('Supervisor',detach('globo_dreamer.continuous_runs','continuous_runs/supervisor.log'))

if __name__=='__main__':main()
