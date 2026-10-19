"""User-authorized SD3/SD1 detector ablations; original SD5 stays untouched."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root=Path('/home/dev/sit_state')
    protocol=json.loads((root/'reactive_protocol.json').read_text())
    protocol.update(thresholds=[3.,1.],seeds=[123,124,125],
        authorization='User asked to lower the detector standard-deviation threshold after SD5 missed both boundaries.',
        selection='Both thresholds fixed before these runs; report every seed/threshold, no best-run selection.',
        source_reference='Reactive_Med uses SD3; Reactive_High uses SD1.',
        changes='Only detector SD threshold changes relative to the completed ReactiveLow SD5 runs.')
    (root/'reactive_sweep_protocol.json').write_text(json.dumps(protocol,indent=2))
    specs=[(6,3,123),(7,1,123),(0,3,124),(1,1,124),(2,3,125),(3,1,125)]
    jobs=[]
    for gpu,sd,seed in specs:
        jobs.append({'gpu':gpu,'sd':sd,'seed':seed,'name':f'bayes_reactive_low_sd{sd}_s{seed}','phase':'queued','proc':None})
    while True:
        baseline=json.loads((root/'reactive_status.json').read_text())
        base_done=all(v['exit_code']==0 for v in baseline.values())
        for j in jobs:
            if j['phase']=='queued' and (j['gpu']>=6 or base_done):
                env=os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(j['gpu']),OMP_NUM_THREADS='4',HF_HOME=str(root/'hf'))
                j['env']=env; j['log']=(root/f"{j['name']}.log").open('a',buffering=1)
                cmd=[sys.executable,'-u','-m','continual.run','--name',j['name'],'--seed',str(j['seed']),
                     '--steps','1500','--bayes-preset','reactive_low','--reactive-sd',str(j['sd']),'--ic','--resume']
                j['proc']=subprocess.Popen(cmd,env=env,stdout=j['log'],stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                j['phase']='train'
            elif j['phase']=='train' and j['proc'].poll()==0:
                j['proc']=subprocess.Popen([sys.executable,'-u','-m','continual.reactive_quality','--name',j['name'],'--single-gpu'],
                    env=j['env'],stdout=j['log'],stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                j['phase']='quality'
        status={j['name']:{'phase':j['phase'],'gpu':j['gpu'],'pid':j['proc'].pid if j['proc'] else None,
                 'exit_code':j['proc'].poll() if j['proc'] else None} for j in jobs}
        temp=root/'reactive_sweep_status.tmp'; temp.write_text(json.dumps(status,indent=2)); temp.replace(root/'reactive_sweep_status.json')
        if all(j['proc'] is not None and j['proc'].poll() is not None for j in jobs):
            break
        time.sleep(20)
    if any(j['proc'].returncode for j in jobs):
        raise RuntimeError('Threshold sweep worker failed; inspect status/logs')


if __name__=='__main__':
    main()
