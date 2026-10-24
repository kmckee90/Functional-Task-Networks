"""Fork two stopped, consistent exploration snapshots; run only on globo1."""
import os
import shutil
import time
from pathlib import Path
from .phase2 import STATE,read,atomic,detach

def main():
    path=STATE/'continuous_runs/state.json';trials=read(path)
    assert len(trials)==4 and all(t['status']=='paused' for t in trials)
    for t in trials:
        proc=Path(f'/proc/{t.get("pid")}/cmdline')
        assert not proc.exists() or not proc.read_bytes(),f'Still saving: {t["name"]}'
        assert read(STATE/'runs'/t['name']/'status.json')['running'] is False
    sources=[t for t in trials if t['mf_offline_updates']==50]
    assert len(sources)==2
    for i,t in enumerate(sources):
        name=f'lifetime2_survival_r{i+1}_s19'
        src=STATE/'runs'/t['name'];dst=STATE/'runs'/name
        assert not dst.exists(),f'Fork already exists: {dst}'
        dst.mkdir()
        latest=(src/'ckpt/latest').read_text().strip()
        # Replay chunks are immutable; separate hard links survive each branch's
        # garbage collection. Mutable metadata and checkpoint pointers are copied.
        shutil.copytree(src/'mf_replay',dst/'mf_replay',copy_function=os.link)
        shutil.copytree(src/'ckpt'/latest,dst/'ckpt'/latest)
        (dst/'ckpt/latest').write_text(latest)
        (dst/'environment').mkdir()
        for file in ('actions.json','input_protocol.json','lifetime.json','memory.npz','achievements.json'):
            if (src/'environment'/file).exists():shutil.copy2(src/'environment'/file,dst/'environment'/file)
        (dst/'live').mkdir()
        if (src/'live/visitation.json').exists():shutil.copy2(src/'live/visitation.json',dst/'live/visitation.json')
        fork_step=read(src/'status.json')['step']
        atomic(dst/'fork.json',dict(source=t['name'],checkpoint=latest,step=fork_step,time=time.time(),
            replay='immutable hard-linked chunks; independent manifests',death_labels='from fork onward',
            novelty_scale=0.25,goal_reward='unchanged',death_cost='batch unscaled goal reward magnitude'))
        trial=dict(t,name=name,gpu=i+2,model='Survival · γ=0.99 · 50 updates',survival=True,
                   source_run=t['name'],fork_step=fork_step,status='pending',attempts=0,retry_after=0)
        for key in ('pid','paused_at','pause_reason','watchdog_term_at','watchdog_kill_at'):trial.pop(key,None)
        trials.append(trial)
        t.update(status='pending',retry_after=0)
        for key in ('pid','paused_at','pause_reason','watchdog_term_at','watchdog_kill_at'):t.pop(key,None)
        print('Forked',t['name'],'->',name,'at',fork_step,flush=True)
    atomic(path,trials)
    print('Supervisor',detach('globo_dreamer.continuous_runs','continuous_runs/supervisor.log'))

if __name__=='__main__':main()
