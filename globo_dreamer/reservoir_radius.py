"""Constrained fine-radius search. Run only on reserved pod GPU."""
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from .reservoir_tune import ROOT as SOURCE,D,noise,realistic,make,evaluate,roll,readout
from .legacy.models import FeedbackToChannel

ROOT=SOURCE.parent/'reservoir_radius'
ROOT.mkdir(exist_ok=True)


def save(name,data):
    path=ROOT/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def raw_data(data,start,stop):
    result=list(realistic(data,start,stop,0))
    result[1]=data[1][start:stop].clone()
    result[2][...,17:18]=result[1][...,-1:]
    return result


def run(c,p,seed,nd,rd,raw,train=8,test=None):
    r=evaluate(c,p,seed,nd,rd,train,test,stability=True)
    res=make(c,seed,p);projection=FeedbackToChannel(35,32,64,seed=seed+1).to(D)
    h,sat,valid=roll(res,projection,raw)
    r['real_raw']=dict(scores=readout(h,raw,valid,train,test),saturation=sat)
    # Optimize the longer delays; current decoding is a separate hard constraint.
    r['long_memory']=float(.5*np.mean([r['noise']['scores'][f'{k}_{l}']['r2']
        for k in ('image','density','tag','action') for l in (16,32)])+
        .5*np.mean([r['real_tagged']['scores'][f'tag_{l}']['r2'] for l in (16,32)]))
    return r


def feasible(row,reference):
    for domain,key,tolerance in [('noise','image',.002),('noise','tag',.005),
            ('noise','density',.005),('noise','action',.005),('real_raw','image',.005)]:
        if row[domain]['scores'][f'{key}_0']['r2']<reference[domain]['scores'][f'{key}_0']['r2']-tolerance:
            return False
    return row['echo'][-1]['rms']<.001


def main():
    data=tuple(x.to(D) for x in torch.load(SOURCE/'real_data.pt',weights_only=True))
    prior=json.loads((SOURCE/'complete.json').read_text())['winners']
    nd=noise();rd=realistic(data,0,12,1234);raw=raw_data(data,0,12)
    winners=[];selection=[]
    for previous in prior:
        c=previous['channels'];p=previous['params']
        refs={seed:run(c,p,seed,nd,rd,raw) for seed in (7,19,101)}
        candidates=[]
        for rho,gain in itertools.product((.98,.985,.99,.995,.9975,1.,1.0025),(.005,.01,.02,.04,.06,.08,.1,.15)):
            params=dict(p,rho=rho,input=gain)
            row=run(c,params,7,nd,rd,raw)
            row['feasible']=feasible(row,refs[7]);candidates.append(row)
            with (ROOT/'trials.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(c,params,'memory',round(row['long_memory'],4),'feasible',row['feasible'],flush=True)
        confirm=[]
        best=sorted([r for r in candidates if r['feasible']],key=lambda r:r['long_memory'],reverse=True)[:8]
        for row in best:
            rows=[row]+[run(c,row['params'],seed,nd,rd,raw) for seed in (19,101)]
            ok=all(feasible(r,refs[r['seed']]) for r in rows)
            summary=dict(params=row['params'],rows=rows,feasible=ok,
                score=float(np.mean([r['long_memory'] for r in rows])))
            confirm.append(summary)
        base=dict(params=p,rows=list(refs.values()),feasible=True,
            score=float(np.mean([r['long_memory'] for r in refs.values()])))
        winner=max([base]+[r for r in confirm if r['feasible']],key=lambda r:r['score'])
        winners.append(dict(channels=c,previous=p,params=winner['params'],validation=winner['score'],previous_validation=base['score']))
        selection.append(dict(channels=c,baseline=base,confirmed=confirm))
        save('selection.json',selection);save('winners.json',winners)
        print('WINNER',winners[-1],flush=True)
    # Fresh noise/tag values; whole held-out streams and disjoint replay chunks.
    nd=noise(batch=32,length=464,seed=20261009)
    rd=realistic(data,12,44,20261009);raw=raw_data(data,12,44)
    results=[]
    for winner in winners:
        c=winner['channels']
        for seed in (7,19,101,37,71):
            for mode in ('previous','tuned'):
                p=winner['previous' if mode=='previous' else 'params']
                r=run(c,p,seed,nd,rd,raw,24,28);r['mode']=mode;results.append(r)
                save('heldout.json',results)
                print('HELDOUT',c,seed,mode,r['long_memory'],flush=True)
                if mode=='tuned':
                    res=make(c,seed,p)
                    torch.save(dict(params=p,channels=c,seed=seed,state_dict={k:v.cpu() for k,v in res.state_dict().items()}),ROOT/f'winner_c{c}_seed{seed}.pt')
    save('complete.json',dict(winners=winners,heldout=results,
        protocol='Fixed connectivity from previous winners. Optimize delays 16/32 subject to per-seed current-decoding tolerances (noise image .002; other noise targets and real image .005 R2) and initial-state RMS <.001 at step256. Selection uses seeds7/19/101; heldout additionally tests unseen reservoir seeds37/71. All candidate selection excludes heldout streams/chunks.'))


if __name__=='__main__':main()
