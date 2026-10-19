"""Pod-only: replace only the supervisor, preserving all four live workers."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from .continuous_runs import ROOT,STATE,read
from .search import atomic,alive


def main():
    path=ROOT/'state.json'
    if any(t.get('frontier_ftn') for t in read(path)):
        raise RuntimeError('Frontier run set already exists; do not recreate it')
    for gpu in (4,5,6,7):
        free=int(subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True).strip())
        if free<85000:raise RuntimeError(f'Insufficient free memory on GPU {gpu}: {free} MiB')
    pid=int((ROOT/'supervisor.pid').read_text())
    cmd=__import__('pathlib').Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if b'globo_dreamer.continuous_runs' not in cmd:raise RuntimeError('Supervisor PID identity mismatch')
    # SIGTERM deliberately terminates workers in this supervisor's finally.
    # SIGKILL affects only the verified supervisor PID; workers are detached.
    os.kill(pid,signal.SIGKILL)
    with (ROOT/'lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        trials=read(path)
        atomic(ROOT/f'before_frontier_{int(time.time())}.json',trials)
        preserved=[t['pid'] for t in trials if t['status']=='running' and alive(t)]
        if len(preserved)!=4:raise RuntimeError(f'Expected four retained live workers, found {preserved}')
        for gpu,(mode,seed) in enumerate((('frontier',19),('frontier',101),('waypoints',19),('waypoints',101)),start=4):
            trials.append(dict(name=f'm2eftn_{mode}_s{seed}',gpu=gpu,seed=seed,
                spawn='12.0,39.1,120.0' if seed==19 else '16.0,58.1,118.0',
                scaled=True,model_free_only=True,m2e_ftn='conv',frontier_ftn=mode,
                reservoir_variant='tuned_v1',mf_replay_dtype='float16',software_renderer=True,
                weight=0.,status='pending',attempts=0,achievements=True,phase='m2e_ftn',render_size=256))
        atomic(path,trials)
        atomic(ROOT/'phase.json',dict(title='FTN–M2E · policy decoders and retained frontier options',time=time.time()))
    with (ROOT/'supervisor.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','globo_dreamer.continuous_runs'],
            cwd=STATE,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),
            stdout=log,stderr=log,stdin=subprocess.DEVNULL,start_new_session=True)
    print(json.dumps(dict(supervisor=p.pid,preserved_worker_pids=preserved,new_gpus=[4,5,6,7])),flush=True)


if __name__=='__main__':main()
