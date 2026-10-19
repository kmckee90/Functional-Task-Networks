"""Held-out dense ablation and unmodified replay checks for selected reservoirs."""
import json
import torch
from .reservoir_tune import ROOT,D,noise,realistic,make,roll,readout,echo,evaluate,atomic
from .legacy.models import Conv2dReservoir,FeedbackToChannel


def main():
    winners=json.loads((ROOT/'winners.json').read_text())
    data=tuple(x.to(D) for x in torch.load(ROOT/'real_data.pt',weights_only=True))
    input_seed=json.loads((ROOT/'complete.json').read_text()).get('input_seed',5678)
    nd=noise(batch=32,length=464,seed=input_seed)
    tagged=realistic(data,12,44,input_seed)
    real=list(realistic(data,12,44,5678))
    real[1]=data[1][12:44].clone()
    real[2][...,17:18]=real[1][...,-1:]
    results=[]
    for winner in winners:
        channels=winner['channels'];params=winner['params']
        for seed in (7,19,101):
            for mode in ('original','sparse','normalized_dense'):
                p=dict(params,density=1.) if mode=='normalized_dense' else params
                res=Conv2dReservoir(4,num_channels=channels,seed=seed).to(D) if mode=='original' else make(channels,seed,p)
                proj=FeedbackToChannel(35,32,64,seed=seed+1).to(D)
                h,sat,valid=roll(res,proj,real)
                row=dict(channels=channels,seed=seed,mode=mode,params=p,saturation=sat,
                    real_scores=readout(h,real,valid,24,28),echo=echo(res,proj,real))
                if mode=='normalized_dense':
                    row['controlled']=evaluate(channels,p,seed,nd,tagged,24,28)
                if mode=='sparse':
                    torch.save(dict(params=p,channels=channels,seed=seed,
                        reservoir={k:v.cpu() for k,v in res.state_dict().items()},
                        projection={k:v.cpu() for k,v in proj.state_dict().items()}),
                        ROOT/f'winner_c{channels}_seed{seed}.pt')
                results.append(row);atomic('verification.json',results)
                print(channels,seed,mode,'current image',row['real_scores']['image_0']['r2'],
                    'echo',row['echo'][-1]['rms'],flush=True)
    atomic('verified.json',dict(results=results,
        note='Recorded images, actions and density; model feedback uses source replay approximation of normalization. Real-image lag decoding can benefit from temporal autocorrelation. Dense controls share winning gains and leak, without sparse recurrence.'))


if __name__=='__main__':main()
