"""Pod-resident asynchronous evolutionary search, one sequential agent per GPU.

Durable JSON state + event journal; restart adopts live workers. Candidates are
ranked only after all screened seeds finish. No finite search queue or generations.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np
from .search_score import score_run

BASE=dict(mf_gamma=.9,mf_epsilon_end=.025,mf_online_coef=1.,
          mf_frontier_scale=5.,dreamer_action_probability=.1,mf_lr=3e-4)
# Mutations are in normalized coordinates; positive scales are logarithmic.
BOUNDS=dict(horizon=(8.,100.),mf_epsilon_end=(.01,.25),mf_online_coef=(.1,1.5),
            mf_frontier_scale=(1.,6.),dreamer_action_probability=(.025,.25),mf_lr=(1e-4,5e-4))


def atomic(path,data):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(data,indent=2,allow_nan=False)); tmp.replace(path)


def alive(trial):
    if not trial.get('pid'): return False
    try:
        data=Path(f'/proc/{trial["pid"]}/cmdline').read_bytes().split(b'\0')
        return b'globo_dreamer.train' in data and trial['name'].encode() in data
    except FileNotFoundError: return False


def uninterruptible_kill_pending(pid):
    try:
        fields=dict(line.split(':',1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
        pending=int(fields.get('SigPnd','0').strip(),16)|int(fields.get('ShdPnd','0').strip(),16)
        return fields['State'].strip().startswith('D') and bool(pending & (1 << (signal.SIGKILL-1)))
    except (FileNotFoundError,KeyError): return False


def gpu_has_headroom(gpu):
    try:
        out=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.free',
                                     '--format=csv,noheader,nounits'],text=True,timeout=10)
        return int(out.strip())>=40000
    except (OSError,ValueError,subprocess.SubprocessError): return False


def propose(rng,parent=None,sigma=.18):
    params=dict(parent or BASE)
    keys=list(BOUNDS)
    chosen=keys if parent is None else list(rng.choice(keys,size=int(rng.integers(1,3)),replace=False))
    for key in chosen:
        lo,hi=map(math.log,BOUNDS[key])
        current=1/(1-params['mf_gamma']) if key=='horizon' else params[key]
        z=rng.random() if parent is None else (math.log(current)-lo)/(hi-lo)+rng.normal(0,sigma)
        # Reflect at bounds rather than accumulating probability at clipping edges.
        z=z%2; z=2-z if z>1 else z
        value=math.exp(lo+z*(hi-lo))
        params['mf_gamma' if key=='horizon' else key]=1-1/value if key=='horizon' else value
    return params


class Search:
    prefix = 's40'
    objective_version = 1
    propose = staticmethod(propose)
    score = staticmethod(score_run)

    def initial_configs(self):
        return [BASE,{**BASE,'mf_gamma':.97},{**BASE,'mf_gamma':.97,'mf_epsilon_end':.1},
                {**BASE,'mf_gamma':.97,'mf_frontier_scale':2.},
                {**BASE,'mf_gamma':.97,'mf_online_coef':.25}]

    def __init__(self,args):
        self.args=args; self.root=Path(args.state).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        self.state_root=Path(os.environ.get('GLOBO_STATE','/home/dev/globo_dreamer_state')).resolve()
        if self.root.is_relative_to(Path(__file__).resolve().parent): raise ValueError('Search state must be outside sync')
        self.lock=(self.root/'search.lock').open('w'); fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.lock.write(str(os.getpid())); self.lock.flush()
        self.rng=np.random.default_rng(args.search_seed); self.children={}; self.stopping=False
        self.path=self.root/'state.json'; self.gpus=list(range(args.gpus))
        if self.path.exists():
            self.db=json.loads(self.path.read_text()); self.rng.bit_generator.state=self.db['rng']
            if self.db['steps']!=args.steps: raise ValueError('Cannot change trial budget in existing search')
            for t in self.db['trials']:
                if t['status']=='paused': t['status']='pending'
        else:
            seeds=json.loads(Path(args.seeds).read_text())
            assert len(seeds)==3 and len({x['seed'] for x in seeds})==3
            digest=hashlib.sha256()
            for p in sorted(Path(__file__).resolve().parent.rglob('*.py')): digest.update(p.read_bytes())
            self.db=dict(version=1,steps=args.steps,seeds=seeds,configs=[],trials=[],rng=None,
                         source_digest=digest.hexdigest(),failures=0,cooldown_until=0,created=time.time())
            initial=self.initial_configs()
            for params in initial: self.add_config(params,None,.18,'initial hypothesis')
        signal.signal(signal.SIGTERM,lambda *_: setattr(self,'stopping',True))
        signal.signal(signal.SIGINT,lambda *_: setattr(self,'stopping',True))
        self.save()

    def event(self,kind,**data):
        with (self.root/'events.jsonl').open('a') as f: f.write(json.dumps(dict(time=time.time(),kind=kind,**data))+'\n')

    def save(self):
        self.db['rng']=self.rng.bit_generator.state; self.db['updated']=time.time()
        self.db['supervisor_pid']=os.getpid(); atomic(self.path,self.db)

    def add_config(self,params,parent,sigma,reason):
        idx=len(self.db['configs'])
        self.db['configs'].append(dict(id=idx,params=params,parent=parent,sigma=sigma,reason=reason))
        for s in self.db['seeds']:
            self.db['trials'].append(dict(name=f'{self.prefix}_c{idx:06d}_s{s["seed"]}',config=idx,
                seed=s['seed'],spawn=s['spawn'],status='pending',attempts=0,created=time.time()))
        self.event('candidate',config=idx,params=params,parent=parent,reason=reason)

    def rankings(self):
        output=[]
        base={t['seed']:t['result']['score'] for t in self.db['trials'] if t['config']==0 and t['status']=='done'}
        if len(base)!=3: return []
        for c in self.db['configs']:
            trials=[t for t in self.db['trials'] if t['config']==c['id']]
            if len(trials)!=3 or any(t['status']!='done' for t in trials): continue
            values=[t['result']['score']-base.get(t['seed'],0.) for t in trials]
            output.append(dict(config=c['id'],score=float(np.mean(values)-.25*np.std(values)),
                mean=float(np.mean(values)),seed_scores={t['seed']:t['result']['score'] for t in trials},params=c['params']))
        return sorted(output,key=lambda x:x['score'],reverse=True)

    def evolve(self):
        ranks=self.rankings()
        if not ranks or self.rng.random()<.2:
            self.add_config(self.propose(self.rng),None,.18,'global random exploration'); return
        elite=ranks[:max(1,math.ceil(len(ranks)*.25))]
        winner=elite[int(self.rng.integers(len(elite)))]; parent=self.db['configs'][winner['config']]
        self.add_config(self.propose(self.rng,parent['params'],parent['sigma']),parent['id'],parent['sigma'],
                        'mutate one or two parameters of a three-seed elite')

    def launch(self,t,gpu):
        run=self.state_root/'runs'/t['name']; run.mkdir(parents=True,exist_ok=True)
        c=self.db['configs'][t['config']]
        cmd=[sys.executable,'-u','-m','globo_dreamer.train','--name',t['name'],'--steps',str(self.args.steps),
             '--seed',str(t['seed']),'--spawn',t['spawn'],'--model-free','--train-ratio','64',
             '--mf-replay-capacity','1000000','--mf-goal-capacity','100000','--recode-capacity','10000',
             '--dreamer-replay-capacity','50000','--checkpoint-seconds','900','--video-every','10000']
        for key,value in c['params'].items(): cmd.extend(['--'+key.replace('_','-'),str(value)])
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',
                 XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='4',LP_NUM_THREADS='2')
        cpus=sorted(os.sched_getaffinity(0))
        if shutil.which('taskset') and len(cpus)>=self.args.gpus*4:
            assigned=cpus[gpu*4:(gpu+1)*4]
            cmd=['taskset','-c',','.join(map(str,assigned)),*cmd]
        atomic(run/'search_trial.json',dict(config=c,seed=t['seed'],spawn=t['spawn'],
               objective_version=self.objective_version,source_digest=self.db['source_digest']))
        with (run/'console.log').open('a') as log:
            child=subprocess.Popen(cmd,cwd=self.state_root,env=env,stdout=log,stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL,start_new_session=True)
        self.children[t['name']]=child
        t.update(status='running',pid=child.pid,gpu=gpu,started=time.time(),attempts=t['attempts']+1,
                 last_progress=time.time(),last_step=-1)
        t.pop('term_sent',None)
        self.event('launch',name=t['name'],gpu=gpu,pid=child.pid,attempt=t['attempts']); self.save()

    def finish(self,t):
        run=self.state_root/'runs'/t['name']
        try:
            status=json.loads((run/'status.json').read_text())
            if status['running'] or status['step']<self.args.steps: raise ValueError('Trial exited before budget')
            ck=run/'ckpt'; latest=ck/(ck/'latest').read_text().strip()
            if not (latest/'done').exists(): raise ValueError('Incomplete final checkpoint')
            result=self.score(run,self.args.steps); atomic(run/'summary.json',result)
            t.update(status='done',finished=time.time(),result={k:v for k,v in result.items() if k!='windows'})
            self.db['failures']=0
            self.event('complete',name=t['name'],result=t['result'])
            ranks={x['config']:x['score'] for x in self.rankings()}; c=self.db['configs'][t['config']]
            if c['id'] in ranks and c['parent'] in ranks and not c.get('adapted'):
                parent=self.db['configs'][c['parent']]
                success=ranks[c['id']]>ranks[c['parent']]+.02
                parent['sigma']=float(np.clip(parent['sigma']*(1.15 if success else .85),.04,.4))
                c['adapted']=True
        except Exception as exc:
            t.update(status='pending' if t['attempts']<3 else 'failed',error=str(exc))
            self.db['failures']+=1
            if self.db['failures']>=8: self.db['cooldown_until']=time.time()+300
            self.event('failure',name=t['name'],error=str(exc),status=t['status'])
        self.save(); self.prune()

    def prune(self):
        # Only this search's finished runs. Preserve winners, recent results, and all active trials.
        ranks=self.rankings(); best={r['config'] for r in ranks[:2]}
        done=sorted([t for t in self.db['trials'] if t['status'] in ('done','failed')],key=lambda t:t.get('finished',t['created']),reverse=True)
        recent={t['name'] for t in done[:6]}
        for t in done:
            if t['config'] in best or t['name'] in recent or t.get('pruned'): continue
            siblings=[x for x in self.db['trials'] if x['config']==t['config']]
            if any(x['status'] not in ('done','failed') for x in siblings): continue
            run=self.state_root/'runs'/t['name']
            for name in ('ckpt','mf_replay','replay','environment','agent_feedback.jsonl'):
                path=run/name
                if path.is_dir(): shutil.rmtree(path)
                elif path.exists(): path.unlink()
            t['pruned']=True; self.event('prune',name=t['name'])
        self.save()

    def poll(self):
        for t in self.db['trials']:
            if t['status']!='running': continue
            child=self.children.get(t['name'])
            if child is not None: child.poll()  # reap finished children
            if not alive(t): self.finish(t); continue
            run=self.state_root/'runs'/t['name']
            try:
                status=json.loads((run/'status.json').read_text()); step=status['step']
                if step!=t['last_step']: t.update(last_step=step,last_progress=time.time())
            except (FileNotFoundError,json.JSONDecodeError): pass
            if time.time()-t['last_progress']>1200:
                if not t.get('term_sent'):
                    os.kill(t['pid'],signal.SIGTERM); t['term_sent']=time.time()
                    self.event('stalled_worker',name=t['name'])
                elif time.time()-t['term_sent']>120:
                    try: os.killpg(t['pid'],signal.SIGKILL)
                    except ProcessLookupError: continue
                    if uninterruptible_kill_pending(t['pid']):
                        # Kernel-blocked NFS tasks cannot exit until the server responds.
                        # Never touch/reuse that attempt's directory or count it as done.
                        old=dict(t)
                        self.db.setdefault('quarantined',[]).append(old)
                        t.update(name=f'{old["name"]}_fresh{old["attempts"]+1}',
                            status='pending' if old['attempts']<3 else 'failed',
                            error='Uninterruptible kernel I/O with SIGKILL pending; fresh attempt required')
                        self.event('quarantine',name=old['name'],pid=old['pid'],gpu=old['gpu'],replacement=t['name'])
                        self.save()
        occupied={t['gpu'] for t in self.db['trials'] if t['status']=='running'}
        if not self.stopping and time.time()>=self.db['cooldown_until']:
            for gpu in self.gpus:
                if gpu in occupied: continue
                if self.db.get('quarantined') and not gpu_has_headroom(gpu):
                    self.event('insufficient_gpu_headroom',gpu=gpu); continue
                # Avoid filling the volume if unrelated files have consumed its free space.
                if shutil.disk_usage(self.state_root).free<15*2**30:
                    self.event('low_disk_space'); break
                pending=next((t for t in self.db['trials'] if t['status']=='pending'),None)
                if pending is None:
                    self.evolve(); pending=next(t for t in self.db['trials'] if t['status']=='pending')
                self.launch(pending,gpu)
        atomic(self.root/'leaderboard.json',dict(time=time.time(),rankings=self.rankings(),
            active=[{k:t.get(k) for k in ('name','gpu','last_step','attempts','last_progress')} for t in self.db['trials'] if t['status']=='running'],
            completed=sum(t['status']=='done' for t in self.db['trials']),failed=sum(t['status']=='failed' for t in self.db['trials'])))
        self.save()

    def run(self):
        self.event('supervisor_start',pid=os.getpid())
        try:
            while not self.stopping:
                self.poll(); time.sleep(self.args.poll_seconds)
        finally:
            for t in self.db['trials']:
                if t['status']=='running' and alive(t): os.kill(t['pid'],signal.SIGTERM)
            deadline=time.time()+180
            while time.time()<deadline and any(t['status']=='running' and alive(t) for t in self.db['trials']): time.sleep(1)
            for t in self.db['trials']:
                if t['status']=='running' and not alive(t): t['status']='paused'
            self.save(); self.event('supervisor_stop')


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--state',default='/home/dev/globo_dreamer_state/search40k')
    p.add_argument('--seeds',default='/home/dev/globo_dreamer_state/seed_screen/selected.json')
    p.add_argument('--steps',type=int,default=40000)
    p.add_argument('--gpus',type=int,default=8)
    p.add_argument('--search-seed',type=int,default=20261007)
    p.add_argument('--poll-seconds',type=float,default=20)
    args=p.parse_args()
    if args.steps%1000 or args.steps<10000: p.error('Budget must be >=10K and divisible by 1K')
    Search(args).run()


if __name__=='__main__': main()
