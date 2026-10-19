"""Detached three-seed repair/evaluation supervisor; runs only on pod."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('/home/dev/sit_state')
    jobs=[]
    for gpu,seed in enumerate([123,124,125]):
        name=f'bayes_replay_s{seed}'
        env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4')
        log=(root/f'repair_{seed}.log').open('a',buffering=1)
        command=[sys.executable,'-u','-m','continual.repair_eval','--name',name]
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        jobs.append({'name':name,'proc':proc,'log':log,'env':env,'phase':'repair'})
    while True:
        for job in jobs:
            proc=job['proc']
            if proc.poll()==0 and job['phase']=='repair':
                job['phase']='quality'
                job['proc']=subprocess.Popen([sys.executable,'-u','-m','continual.repair_quality','--name',job['name']],
                    env=job['env'],stdout=job['log'],stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
        status={j['name']:{'pid':j['proc'].pid,'phase':j['phase'],'exit_code':j['proc'].poll()} for j in jobs}
        temp=root/'repair_status.tmp'; temp.write_text(json.dumps(status,indent=2)); temp.replace(root/'repair_status.json')
        if all(j['proc'].poll() is not None for j in jobs):
            break
        time.sleep(20)
    if any(j['proc'].returncode for j in jobs):
        raise RuntimeError('Repair worker failure; inspect repair_status.json')


if __name__=='__main__':
    main()
