"""Sequential FTN identity learning with paired support inference and no identity labels."""
import argparse
from collections import defaultdict
import json
import math
import time
import numpy as np
import torch
from ..model import InputPrior, transport_batch
from ..run import Replay, query_loss
from .common import ROOT, atomic, names, model, data


def evaluate(m, ds, router, saved, stage):
    state=m.get_reconfig_state(); row={'stage':stage,'domains':{}}
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        for index,n in enumerate(names()):
            support,test=ds[n]['support'],ds[n]['test']
            query=[transport_batch(test['z'],test['y'],torch.Generator(device='cuda').manual_seed(1900+r)) for r in range(4)]
            qx=torch.cat([x for x,y in query]); qy=torch.cat([y for x,y in query])
            candidates={'base':torch.zeros_like(m.m)}; timings={}
            if n in saved: candidates['stored']=saved[n]
            for draw in range(3):
                g=torch.Generator(device='cuda').manual_seed(501+draw)
                ids=torch.randperm(len(support['z']),generator=g,device='cuda')
                sx,sy=transport_batch(support['z'][ids],support['y'][ids],g)
                for budget in ([2,4,16] if draw==0 else [2,4]):
                    torch.manual_seed(9900+index*100+draw*20+budget); tic=time.time()
                    m.cold_reconfigure(sx[:budget],sy[:budget],steps=30,take_best=False,restarts=1)
                    key=f'cold{budget}_d{draw}'; candidates[key]=m.m.clone();timings[key]=time.time()-tic
                torch.manual_seed(9900+index*100+draw*20+4)
                router.infer(m,support['z'][ids[:4]],sx[:4],sy[:4],False)
                candidates[f'ic_direct4_d{draw}']=m.m.clone()
                torch.manual_seed(9900+index*100+draw*20+4)
                router.infer(m,support['z'][ids[:4]],sx[:4],sy[:4],True)
                candidates[f'ic_refine4_d{draw}']=m.m.clone()
            results={}
            for key,mask in candidates.items():
                r=query_loss(m,qx,qy,mask)
                # Four corruptions of the same image are not four independent images.
                r['per_image']=np.array(r.pop('per_example')).reshape(4,-1).mean(0).tolist()
                r['se']=float(np.std(r['per_image'],ddof=1)/math.sqrt(len(r['per_image'])))
                r['mask']=mask.cpu().tolist(); results[key]=r
            row['domains'][n]=results
            print('EVAL',stage,n,{k:round(v['loss'],5) for k,v in results.items() if k.endswith('d0')},flush=True)
    m.set_reconfig_state(state)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',type=int,default=123)
    p.add_argument('--replay',action='store_true');p.add_argument('--plain',action='store_true')
    p.add_argument('--only',default=None);p.add_argument('--steps',type=int,default=1500)
    a=p.parse_args(); folder=ROOT/'runs'/a.name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    atomic(folder/'config.json',vars(a));m=model(a.seed,a.plain);ds=data()
    if a.plain:m.m.fill_(1)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=3e-4)
    replay=Replay(a.seed+1000);router=InputPrior();saved={}; history=[]
    dg=torch.Generator(device='cuda').manual_seed(a.seed+20)
    tg=torch.Generator(device='cuda').manual_seed(a.seed+10)
    mu,var,since=None,0.,0; start=0
    order=[a.only] if a.only else names()
    if (folder/'latest.pt').exists():
        ck=torch.load(folder/'latest.pt',map_location='cuda',weights_only=False);m.load_adapter(ck['adapter'])
        router.entries=ck['router'];saved=ck['saved'];history=ck['history'];start=ck['completed']
        replay.entries=ck['replay']; replay.rng.bit_generator.state=ck['replay_rng']
        replay.generator.set_state(ck['replay_generator'].cpu());dg.set_state(ck['dg'].cpu());tg.set_state(ck['tg'].cpu())
        torch.cuda.set_rng_state(ck['cuda_rng'].cpu());torch.set_rng_state(ck['cpu_rng'].cpu());mu,var,since=ck['detector']
    else:
        torch.save(m.adapter_state(),folder/'initial_adapter.pt')
        history=[evaluate(m,ds,router,saved,'initial')];atomic(folder/'lifecycle.json',history)
    with (folder/'train.jsonl').open('a',buffering=1) as log:
        for block,n in enumerate(order):
            if block<start:continue
            opt.state=defaultdict(dict);train=ds[n]['train'];tic=time.time()
            for step in range(a.steps):
                ids=torch.randint(len(train['z']),(16,),device='cuda',generator=dg)
                z,y=train['z'][ids],train['y'][ids];x,target=transport_batch(z,y,tg)
                with torch.no_grad():observed=m.loss(m(x),target).item()
                threshold=None if mu is None else mu+3*math.sqrt(max(var,0))
                cp=not a.plain and (mu is None or (since>=math.ceil(1/max(1-.9,1e-8)) and observed>threshold))
                if not a.plain:
                    if cp:m.reconfigure_changepoint(x,target);since=0
                    else:m.reconfigure(x,target)
                opt.zero_grad(set_to_none=True);loss=m.loss(m(x),target);value=loss.item()
                if cp or mu is None:mu,var=value,0.
                else:
                    diff=value-mu;mu,var=.9*mu+.1*value,.9*var+.1*diff*diff
                since+=1
                if a.replay and step<32:replay.add(z,y,m.m)
                loss.backward();rep=replay.backward(m) if a.replay else None;opt.step()
                assert math.isfinite(value)
                r={'block':block,'domain':n,'step':step,'loss':value,'detector_loss':observed,
                   'threshold':threshold,'changepoint':cp,'replay_loss':rep,'mask':m.m.cpu().tolist()}
                log.write(json.dumps(r)+'\n')
                if step%100==0:print('TRAIN',a.name,n,step,value,'CP',cp,'seconds',time.time()-tic,flush=True)
            saved[n]=m.m.clone();router.archive(train['z'],m.mask_logits)
            stage=f'after_{block+1}_{n}';history.append(evaluate(m,ds,router,saved,stage));atomic(folder/'lifecycle.json',history)
            ck={'adapter':m.adapter_state(),'router':router.entries,'saved':saved,'history':history,'completed':block+1,
                'replay':replay.entries,'replay_rng':replay.rng.bit_generator.state,'replay_generator':replay.generator.get_state(),
                'detector':(mu,var,since),'dg':dg.get_state(),'tg':tg.get_state(),
                'cuda_rng':torch.cuda.get_rng_state(),'cpu_rng':torch.get_rng_state()}
            torch.save(ck,folder/f'{stage}.pt');torch.save(ck,folder/'latest.tmp');(folder/'latest.tmp').replace(folder/'latest.pt')
    atomic(folder/'complete.json',{'complete':True,'blocks':len(order),'replay_hits':replay.hits})


if __name__=='__main__':main()
