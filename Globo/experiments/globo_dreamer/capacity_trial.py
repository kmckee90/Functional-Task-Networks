"""Pod-only measured capacity trial; full replay allocation, bounded duration."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .search60 import FIXED

STATE=Path(os.environ.get('GLOBO_STATE','/home/dev/globo_dreamer_state'))
name='capacity_r128_h1024_s128c32_e2048_buffers2m_s7'
root=STATE/'runs'/name
root.mkdir(parents=True,exist_ok=True)
cmd=[sys.executable,'-u','-m','globo_dreamer.train','--name',name,'--seed','7',
    '--spawn','12,5.1,120','--steps','3000','--model-free','--world-model-only',
    '--train-ratio','256','--model-feedback','--model-reward-weight','1',
    '--continuous-monitor','--checkpoint-seconds','600','--reservoir-channels','128',
    '--rssm-hidden','1024','--rssm-stoch','128','--rssm-classes','32','--encoder-depth','32',
    '--mf-replay-capacity','2000000','--mf-goal-capacity','500000','--recode-capacity','100000',
    '--dreamer-replay-capacity','100000']
for k,v in dict(FIXED,dreamer_action_probability=0.).items():cmd.extend(['--'+k.replace('_','-'),str(v)])
env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='4',LP_NUM_THREADS='2',
    PYTHONDONTWRITEBYTECODE='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',
    GLOBO_ENGINE_ROOT=f'/tmp/globo_capacity/{name}')
cpus=sorted(os.sched_getaffinity(0))[12:16]
if len(cpus)==4:cmd=['taskset','-c',','.join(map(str,cpus)),*cmd]
start=time.time();samples=[]
with (root/'console.log').open('a') as log:
    child=subprocess.Popen(cmd,env=env,cwd=STATE,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
    (root/'benchmark_pid.json').write_text(json.dumps(dict(wrapper=os.getpid(),train=child.pid)))
    with (root/'gpu_samples.jsonl').open('a') as out:
        while child.poll() is None:
            result=subprocess.check_output(['nvidia-smi','-i','3',
                '--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            used,free,util=map(int,result.strip().split(','))
            row=dict(time=time.time(),used_mib=used,free_mib=free,util=util)
            try:
                status=json.loads((root/'status.json').read_text())
                row.update(step=status['step'],updates=status['updates'])
            except (OSError,ValueError):pass
            samples.append(row);out.write(json.dumps(row)+'\n');out.flush()
            time.sleep(2)
summary=dict(exit_code=child.returncode,wall_seconds=time.time()-start,
    peak_gpu_used_gib=max(r['used_mib'] for r in samples)/1024,
    minimum_gpu_free_gib=min(r['free_mib'] for r in samples)/1024,samples=len(samples))
(root/'benchmark.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
