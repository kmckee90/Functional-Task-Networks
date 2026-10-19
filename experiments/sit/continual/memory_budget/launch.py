import fcntl,json,os,subprocess,sys,time
from .common import ROOT,ART,ASSETS,BUDGETS,atomic

def main():
    ROOT.mkdir(exist_ok=True);ART.mkdir(exist_ok=True)
    lock=(ROOT/'launcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert json.loads((ART/'data_check.json').read_text())['passed']
    assert json.loads((ART/'preflight.json').read_text())['passed']
    assert json.loads((ART/'integration.json').read_text())['passed']
    active={};done=[];failed=[];last_count=-1
    for gpu,b in enumerate(BUDGETS):
        tag=f'replay_p{b:02d}_s125'
        if (ART/tag/'complete.json').exists():done.append(tag);continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TORCH_HOME=str(ASSETS/'ic4_recall/torch'),OMP_NUM_THREADS='4')
        f=(ROOT/f'{tag}.log').open('a',buffering=1)
        p=subprocess.Popen([sys.executable,'-u','-m','continual.memory_budget.training','--fraction',str(b)],env=env,stdout=f,stderr=subprocess.STDOUT)
        active[gpu]=(p,f,tag)
    while active:
        for gpu,(p,f,tag) in list(active.items()):
            if p.poll() is not None:
                f.close();(done if p.returncode==0 else failed).append(tag);del active[gpu]
        atomic(ART/'status.json',dict(active=[dict(gpu=g,pid=p.pid,run=t) for g,(p,f,t) in active.items()],done=done,failed=failed,time=time.time()))
        count=len(list(ART.glob('*/*/summary.json')))
        if count!=last_count:
            # Reporting failures must not prevent worker monitoring.
            result=subprocess.run([sys.executable,'-m','continual.memory_budget.report'])
            if result.returncode==0:last_count=count
        if active:time.sleep(20)
    if failed:raise RuntimeError(failed)
    subprocess.run([sys.executable,'-m','continual.memory_budget.audit'],check=True)
    subprocess.run([sys.executable,'-m','continual.memory_budget.report'],check=True)
    atomic(ART/'complete.json',dict(complete=True,runs=4,blocks=11,checkpoints_evaluated=48))

if __name__=='__main__':main()
