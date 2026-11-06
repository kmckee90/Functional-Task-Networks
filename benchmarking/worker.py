"""One GPU worker. NFS advisory locks prevent duplicate jobs across pods."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback

from jobs import jobs
from storage import WORKSPACE,bulk,write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pod',choices=['bm1','bm2','bm3'],required=True)
    p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--focus-dataset',default='tinyimagenet')
    p.add_argument('--suite', choices=['default', 'core50'], default='default')
    a=p.parse_args()
    if a.suite == 'core50':
        from core50_jobs import jobs as suite_jobs
        manifest = suite_jobs()
    else:
        manifest=[job for job in jobs() if job['dataset']==a.focus_dataset]
    worker_name=f'{a.pod}_gpu{a.gpu}'
    status=bulk('runs','workers',worker_name+'.json')
    while True:
        first_ready=all(bulk('runs',j['id'],'result.json').exists() for j in manifest if j['id'].endswith('/seed0'))
        unfinished=0
        worked=False
        for job in manifest:
            if not first_ready and not job['id'].endswith('/seed0'):
                continue
            result=bulk('runs',job['id'],'result.json')
            if result.exists():
                data=json.loads(result.read_text())
                if data.get('status')=='complete' and not data.get('smoke'):
                    continue
            unfinished+=1
            if job.get('pod',a.pod)!=a.pod:
                continue
            if not bulk('data',job['dataset'],'ready.json').exists():
                continue
            if job.get('needs_features'):
                seed=int(job['command'][job['command'].index('--seed')+1])
                path=bulk('data','features',f'{job["dataset"]}_vit_run{seed if job["dataset"]=="core50" else 0}.pt')
                if not path.exists(): continue
            if not bulk('validation',job['gate']+'.json').exists():
                continue
            failure=bulk('runs',job['id'],'failure.json')
            if failure.exists():
                continue  # inspection/fix required; never silently substitute a smaller run
            lock_path=bulk('runs',job['id'],'job.lock')
            with lock_path.open('a') as lock:
                try:
                    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                if result.exists():
                    continue
                write_json(status,{'worker':worker_name,'status':'running','job':job['id'],'started_unix':time.time()})
                environment=os.environ.copy()
                environment['CUDA_VISIBLE_DEVICES']=str(a.gpu)
                logfile=bulk('runs',job['id'],'stdout.log')
                start=time.perf_counter()
                with logfile.open('w') as log:
                    done=subprocess.run([sys.executable,'-u',*job['command']],cwd=WORKSPACE,env=environment,stdout=log,stderr=subprocess.STDOUT)
                if done.returncode or not result.exists():
                    write_json(failure,{'returncode':done.returncode,'job':job,'worker':worker_name,
                        'elapsed_seconds':time.perf_counter()-start,'log':str(logfile),'finished_unix':time.time()})
                else:
                    data=json.loads(result.read_text())
                    if data.get('status')!='complete' or data.get('smoke'):
                        write_json(failure,{'reason':'Missing complete production result','job':job})
                worked=True
                break
        if unfinished==0:
            write_json(status,{'worker':worker_name,'status':'complete'})
            return
        if not worked:
            write_json(status,{'worker':worker_name,'status':'waiting','unfinished_jobs':unfinished,'updated_unix':time.time()})
            time.sleep(15)


if __name__=='__main__': main()
