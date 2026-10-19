"""Extend gain boundary and confirm parameter selection across reservoir seeds."""
import itertools
import json
import numpy as np
import torch
from .reservoir_tune import ROOT,D,noise,realistic,make,echo,evaluate,atomic
from .legacy.models import FeedbackToChannel


def main():
    old=json.loads((ROOT/'complete.json').read_text())
    atomic('initial_complete.json',old)
    data=tuple(x.to(D) for x in torch.load(ROOT/'real_data.pt',weights_only=True))
    nd=noise();rd=realistic(data,0,12,1234);winners=[]
    for channels in (64,128):
        candidates=[]
        for density,gain,rho in itertools.product((.01,.02,.05,.1,.2,.3,1.),(.01,.03,.06,.1,.2),(.95,1.,1.05)):
            p=dict(density=density,input=gain,rho=rho,leak=1.)
            r=evaluate(channels,p,7,nd,rd,stability=True)
            candidates.append(r)
            with (ROOT/'refinement.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
            print(channels,p,round(r['objective'],4),r['echo'][-1]['rms'],flush=True)
        sparse=[r for r in candidates if r['params']['density']<1 and r['echo'][-1]['rms']<.001]
        confirmed=[]
        for r in sorted(sparse,key=lambda r:r['objective'],reverse=True)[:6]:
            rows=[r]+[evaluate(channels,r['params'],seed,nd,rd,stability=True) for seed in (19,101)]
            if any(x['echo'][-1]['rms']>=.001 for x in rows):continue
            r['validation_seeds']=rows[1:]
            r['selection_mean']=float(np.mean([x['objective'] for x in rows]))
            confirmed.append(r)
        if not confirmed:raise RuntimeError('No candidate stable on all validation seeds')
        best=max(confirmed,key=lambda r:r['selection_mean']);winners.append(best)
        atomic('winners.json',winners)
        print('WINNER',channels,best['params'],best['selection_mean'],flush=True)
    # New IID inputs for final reporting; parameter choice never consults test scores.
    nd=noise(batch=32,length=464,seed=9090);rd=realistic(data,12,44,9090)
    results=[]
    for winner in winners:
        for seed in (7,19,101):
            for baseline in (True,False):
                row=evaluate(winner['channels'],winner['params'],seed,nd,rd,
                    train_n=24,test_n=28,baseline=baseline,stability=True)
                results.append(row);atomic('heldout.json',results)
                print('HELDOUT',winner['channels'],seed,baseline,row['objective'],flush=True)
    atomic('complete.json',dict(winners=winners,heldout=results,input_seed=9090,
        protocol='Parameters selected using search streams and first 12 real chunks only, stability checked on three reservoir seeds. Test uses fresh IID seed 9090 and 32 disjoint real chunks. Independent tag replaces model feedback in real-input memory test.'))


if __name__=='__main__':main()
