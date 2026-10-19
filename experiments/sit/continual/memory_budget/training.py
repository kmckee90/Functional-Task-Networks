"""Full active-task training, strict retained-only access after each block."""
import argparse
import fcntl
import gc
import json
import math
import time
import torch
from . import common as C
from .common import ROOT,ART,NAMES,FILTERS,atomic
from .selection import retain
from .replay import CoverageReplay,pool_entries
from ..identity_distinct.representative.sampling import storage_steps
from .evaluate import stage_eval
from ..model import transport_batch


def main():
    p=argparse.ArgumentParser();p.add_argument('--fraction',type=int,choices=C.BUDGETS,required=True)
    p.add_argument('--seed',type=int,default=125);a=p.parse_args()
    tag=f'replay_p{a.fraction:02d}_s{a.seed}';folder=ROOT/'runs'/tag;out=ART/tag
    folder.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'complete.json').exists():return
    cfg=dict(fraction=a.fraction,seed=a.seed,H=1024,k=256,rank=1,steps=1500,batch=16,replay_batch=4)
    atomic(out/'config.json',cfg)
    m,ds=C.model(a.seed),C.data();replay=CoverageReplay(a.seed+1000)
    dg=torch.Generator(device='cuda').manual_seed(a.seed+20);tg=torch.Generator(device='cuda').manual_seed(a.seed+10)
    saved={};mu,var,since,start=None,0.,0,0
    ck=None
    if (folder/'latest.pt').exists():
        ck=torch.load(folder/'latest.pt',map_location='cpu',weights_only=False);assert ck['config']==cfg
        m.load_adapter(ck['adapter']);saved={n:v.cuda() for n,v in ck['saved'].items()}
        C.MEMORY.update(ck['memory']);replay.restore(ck['replay'])
        start=ck['completed'];mu,var,since=ck['detector']
        dg.set_state(ck['dg']);tg.set_state(ck['tg'])
        torch.set_rng_state(ck['cpu_rng']);torch.cuda.set_rng_state(ck['cuda_rng'])
    def checkpoint(completed,stage):
        assert C.ACTIVE is None
        for e in C.MEMORY.values():e.pop('_replay_pool',None)
        obj=dict(config=cfg,completed=completed,stage=stage,seed=a.seed,adapter=m.adapter_state(),
                 saved={n:v.cpu() for n,v in saved.items()},memory=C.MEMORY,replay=replay.state(),
                 detector=(mu,var,since),dg=dg.get_state(),tg=tg.get_state(),
                 cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state())
        temp=folder/'latest.tmp';torch.save(obj,temp);temp.replace(folder/'latest.pt');return obj
    def evaluate(ck):
        C.attach_retained(ds)
        stage_eval(m,ds,ck,out,True,ck['completed']==11)
        atomic(out/ck['stage']/'data_access.json',C.ACCESSES)
        C.ACCESSES.clear()
    if ck is None:ck=checkpoint(0,'initial')
    evaluate(ck);del ck
    logpath=out/'train.jsonl'
    if logpath.exists():
        rows=[s for s in logpath.read_text().splitlines() if json.loads(s)['block']<start]
        logpath.write_text(''.join(s+'\n' for s in rows))
    with logpath.open('a',buffering=1) as log:
        for block in range(start,len(NAMES)):
            n=NAMES[block];train,fs=C.open_block(n)
            opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=3e-4)
            trajectory=[];transient=[];schedule=set(storage_steps(a.seed,block));tic=time.time()
            for step in range(1500):
                ids=torch.randint(len(train['z']),(16,),device='cuda',generator=dg)
                x,target=transport_batch(train['z'][ids],train['y'][ids],tg,dropout=.1 if n in FILTERS else 0.)
                with torch.no_grad():observed=m.loss(m(x),target).item()
                threshold=None if mu is None else mu+3*math.sqrt(max(var,0))
                cp=mu is None or (since>=math.ceil(1/max(1-.9,1e-8)) and observed>threshold)
                if cp:m.reconfigure_changepoint(x,target);since=0
                else:m.reconfigure(x,target)
                opt.zero_grad(set_to_none=True);loss=m.loss(m(x),target);value=loss.item()
                if cp or mu is None:mu,var=value,0.
                else:
                    diff=value-mu;mu,var=.9*mu+.1*value,.9*var+.1*diff*diff
                since+=1
                if step in schedule:
                    transient.append(dict(z=train['z'][ids[:4]].detach().cpu().clone(),
                        y=train['y'][ids[:4]].cpu().clone(),masks=m.m.detach().cpu().clone()[None],
                        entries=[dict(mask=0,indices=list(range(4)))]))
                loss.backward()
                pool=pool_entries(list(C.MEMORY.values())+transient)
                rep=replay.backward(m,pool);opt.step()
                assert math.isfinite(value)
                if step in [799,899,999,1099,1199,1299,1399,1499]:trajectory.append(m.m.detach().cpu().clone())
                log.write(json.dumps(dict(block=block,domain=n,step=step,loss=value,changepoint=cp,replay=rep))+'\n')
                if step%100==0:
                    print('TRAIN',tag,n,step,value,'seconds',time.time()-tic,flush=True)
                    atomic(out/'progress.json',dict(phase='train',task=n,block=block+1,step=step))
            opt.zero_grad(set_to_none=True);del opt
            saved[n]=m.m.detach().clone()
            atomic(out/'progress.json',dict(phase='select_memory',task=n,block=block+1))
            entry,selection=retain(m,n,train,fs,a.fraction,trajectory,C.MEMORY)
            C.close_block(n,entry)
            del train,fs,trajectory,entry,transient,pool,x,target,loss,ids
            replay.prune(pool_entries(list(C.MEMORY.values())))
            gc.collect()
            atomic(out/f'memory_{block+1:02d}_{n}.json',selection)
            print('MEMORY',tag,n,selection['retained'],selection['mask_coverage'],flush=True)
            stage=f'after_{block+1}_{n}';ck=checkpoint(block+1,stage)
            atomic(out/'progress.json',dict(phase='evaluate',stage=stage))
            evaluate(ck);del ck
    atomic(out/'complete.json',dict(complete=True,blocks=11,replay_hits=replay.hits,replay_attempts=replay.attempts))
    print('RUN_COMPLETE',tag,flush=True)

if __name__=='__main__':main()
