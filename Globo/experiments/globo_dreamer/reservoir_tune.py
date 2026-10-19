"""Isolated validation-driven sparse reservoir tuning. All artifacts on pod."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from .legacy.models import Conv2dReservoir,FeedbackToChannel
from .reservoir_probe import real_inputs

ROOT=Path('/home/dev/globo_dreamer_state/diagnostics/reservoir_tuning')
ROOT.mkdir(parents=True,exist_ok=True)
D='cuda:0';LAGS=(0,1,2,4,8,16,32,64);WARM=64
torch.set_num_threads(4);torch.set_grad_enabled(False)
CACHE={}


def atomic(name,data):
    p=ROOT/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)


def noise(batch=12,length=256,seed=1234):
    gen=torch.Generator(device=D).manual_seed(seed)
    img=torch.rand(batch,length,16,device=D,generator=gen)
    tex=torch.rand(1,3,32,64,device=D,generator=gen)*.2
    x=(F.interpolate(img.reshape(-1,1,4,4),size=(32,64),mode='nearest')*.6+.1+tex).reshape(batch,length,3,32,64)
    den=torch.rand(batch,length,1,device=D,generator=gen)*.2
    mod=torch.rand(batch,length,1,device=D,generator=gen)
    act=F.one_hot(torch.randint(25,(batch,length),device=D,generator=gen),25).float()
    fb=torch.cat([act,den,F.one_hot((den[...,0].tanh()*8).long(),8).float(),mod],-1)
    y=torch.cat([img,den,mod,act],-1)
    reset=torch.zeros(batch,length,dtype=torch.bool,device=D);reset[:,0]=True
    return x,fb,y,reset


def realistic(data,start,stop,tag_seed):
    image,feedback,_,reset=[x[start:stop].clone() for x in data]
    # A random memory tag carried in the model-feedback slot prevents apparent
    # memory from being explained by temporal similarity of real images.
    gen=torch.Generator(device=D).manual_seed(tag_seed)
    tag=torch.rand(*feedback.shape[:2],1,device=D,generator=gen)
    feedback[...,-1:]=tag
    coarse=F.adaptive_avg_pool2d(image.flatten(0,1).mean(1,keepdim=True),(4,4)).reshape(*image.shape[:2],16)
    targets=torch.cat([coarse,feedback[...,25:26],tag,feedback[...,:25]],-1)
    return image,feedback,targets,reset


def make(channels,seed,params):
    key=(channels,seed,params['density'])
    if key not in CACHE:
        gen=torch.Generator(device=D).manual_seed(seed)
        # Nested masks and identical Gaussian draws across connection densities.
        w1=torch.randn(channels,4,7,7,device=D,generator=gen)
        w1*=torch.rand(w1.shape,device=D,generator=gen)<.5
        w1/=w1.flatten(1).norm(dim=1).clamp_min(1e-8)[:,None,None,None]
        w2=torch.randn(channels,channels,3,3,device=D,generator=gen)
        w2*=torch.rand(w2.shape,device=D,generator=gen)<params['density']
        w2/=w2.flatten(1).norm(dim=1).clamp_min(1e-8)[:,None,None,None]
        # Exact linear recurrent spectral radius on the actual circular 4x9 grid.
        kernel=torch.zeros(channels,channels,4,9,device=D)
        for i,j in itertools.product(range(3),repeat=2):kernel[:,:,(i-1)%4,(j-1)%9]+=w2[:,:,i,j]
        frequency=torch.fft.fft2(kernel).permute(2,3,0,1)
        radius=float(torch.linalg.eigvals(frequency).abs().max())
        CACHE[key]=(w1,w2/radius,radius)
    w1,w2,_=CACHE[key]
    r=Conv2dReservoir(4,num_channels=channels,seed=seed,input_scale=params['input'],
        recur_scale=params['rho'],tc=params['leak']).to(D)
    r.cnn1.weight.copy_(w1);r.cnn2.weight.copy_(w2)
    return r


def roll(res,projection,data):
    pixels,feedback,target,reset=data;b,t=pixels.shape[:2]
    h=res.init_hidden(b,32,64,D);states=[];sat=[];age=torch.zeros(b,device=D,dtype=torch.long);valid=[]
    for i in range(t):
        h[reset[:,i]]=0;age[reset[:,i]]=0
        h=res(torch.cat([pixels[:,i],projection(feedback[:,i])],1),h)
        if i>=WARM:
            states.append(h.flatten(1).clone());sat.append((h.abs()>.99).float().mean())
            valid.append(age>=max(LAGS))
        age+=1
    return torch.stack(states,1),float(torch.stack(sat).mean()),torch.stack(valid,1)


def readout(features,data,valid,train_n,test_n=None):
    y=data[2];t=y.shape[1]
    target=torch.cat([y[:,WARM-l:t-l] for l in LAGS],-1)
    a=train_n;b=test_n or len(features)
    # Entire streams/chunks split; omit windows crossing an environment reset.
    xtr=features[:a][valid[:a]];xva=features[a:b][valid[a:b]]
    ytr=target[:a][valid[:a]];yva=target[a:b][valid[a:b]]
    xte=features[b:][valid[b:]] if test_n else xva
    yte=target[b:][valid[b:]] if test_n else yva
    xm=xtr.mean(0);xs=xtr.std(0).clamp_min(.001)
    xtr=(xtr-xm)/xs;xva=(xva-xm)/xs;xte=(xte-xm)/xs
    ym=ytr.mean(0);ys=ytr.std(0).clamp_min(.001)
    yt=(ytr-ym)/ys;yv=(yva-ym)/ys;ye=(yte-ym)/ys
    # Dual ridge when there are fewer examples than features.
    dual=len(xtr)<xtr.shape[-1]
    gram=(xtr@xtr.T if dual else xtr.T@xtr)/len(xtr)
    rhs=yt if dual else xtr.T@yt/len(xtr)
    eye=torch.eye(len(gram),device=D);best={}
    for reg in (.00001,.0001,.001,.01,.1,1.):
        sol=torch.linalg.solve(gram+reg*eye,rhs)
        weights=xtr.T@sol/len(xtr) if dual else sol
        pv=xva@weights;pe=xte@weights
        for j,l in enumerate(LAGS):
            for name,lo,hi in [('image',0,16),('density',16,17),('tag',17,18),('action',18,43)]:
                lo+=43*j;hi+=43*j;k=f'{name}_{l}'
                loss=float((pv[:,lo:hi]-yv[:,lo:hi]).square().mean())
                if k not in best or loss<best[k]['val_mse']:
                    ev=(pe[:,lo:hi]-ye[:,lo:hi]).square().mean(0)
                    variance=(ye[:,lo:hi]-ye[:,lo:hi].mean(0)).square().mean(0).clamp_min(1e-8)
                    row=dict(r2=float((1-ev/variance).mean()),val_mse=loss,alpha=reg)
                    if name=='action':
                        row['accuracy']=float(((pe[:,lo:hi]*ys[lo:hi]+ym[lo:hi]).argmax(-1)==yte[:,lo:hi].argmax(-1)).float().mean())
                    best[k]=row
    return best


def echo(res,projection,data):
    pixels,fb,_,_=data;b,t=pixels.shape[:2]
    gen=torch.Generator(device=D).manual_seed(123)
    h=res.init_hidden(b,32,64,D);g=torch.rand(h.shape,device=D,generator=gen)*2-1
    values=[]
    for i in range(t):
        x=torch.cat([pixels[:,i],projection(fb[:,i])],1)
        h=res(x,h);g=res(x,g)
        if i in (0,7,31,63,127,t-1):values.append(dict(step=i+1,rms=float((h-g).square().mean().sqrt())))
    return values


def evaluate(channels,params,seed,nd,rd,train_n=8,test_n=None,baseline=False,stability=False):
    start=time.time();torch.manual_seed(seed)
    res=Conv2dReservoir(4,num_channels=channels,seed=seed).to(D) if baseline else make(channels,seed,params)
    projection=FeedbackToChannel(35,32,64,seed=seed+1).to(D)
    row=dict(channels=channels,seed=seed,params=params,baseline=baseline)
    for name,data in [('noise',nd),('real_tagged',rd)]:
        h,sat,valid=roll(res,projection,data)
        row[name]=dict(saturation=sat,scores=readout(h,data,valid,train_n,test_n))
    ns=row['noise']['scores'];rs=row['real_tagged']['scores']
    memory=np.mean([ns[f'{kind}_{lag}']['r2'] for kind in ('image','tag','action') for lag in (4,8,16,32)])
    real_memory=np.mean([rs[f'tag_{lag}']['r2'] for lag in (4,8,16,32)])
    # Prevent buying memory by throwing away the current image or feedback.
    penalty=2*max(0,.80-ns['image_0']['r2'])+max(0,.85-ns['tag_0']['r2'])+max(0,.75-rs['image_0']['r2'])
    row['objective']=float(.5*memory+.5*real_memory-penalty)
    if stability:row['echo']=echo(res,projection,rd)
    row['seconds']=time.time()-start
    return row


def emit(row):
    with (ROOT/'trials.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps({k:row[k] for k in ('channels','seed','params','objective','seconds')}),flush=True)


def search():
    torch.save(tuple(x.cpu() for x in real_inputs()),ROOT/'real_data.pt')
    atomic('real_dataset.json',json.loads((ROOT.parent/'reservoir_capacity/real_dataset.json').read_text()))
    data=tuple(x.to(D) for x in torch.load(ROOT/'real_data.pt',weights_only=True))
    nd=noise();rd=realistic(data,0,12,1234);winners=[]
    for channels in (64,128):
        baseline=evaluate(channels,{},7,nd,rd,baseline=True,stability=True);emit(baseline)
        candidates=[];seen=set()
        def trial(p):
            key=tuple(sorted(p.items()))
            if key in seen:return
            seen.add(key);r=evaluate(channels,p,7,nd,rd);candidates.append(r);emit(r)
        for density,gain,rho in itertools.product((.01,.02,.05,.1,.2,.3),(.1,.4,1.),(.85,1.1)):
            trial(dict(density=density,input=gain,rho=rho,leak=1.))
        parents=[]
        for r in sorted(candidates,key=lambda x:x['objective'],reverse=True):
            if not any(p['params']['density']==r['params']['density'] for p in parents):parents.append(r)
            if len(parents)==2:break
        for parent in parents:
            for rho,leak in itertools.product((.75,.9,1.,1.1,1.25),(.25,.5,1.)):
                trial(dict(parent['params'],rho=rho,leak=leak))
        # Dense controls distinguish benefits of gain normalization from sparsity.
        for parent in sorted(candidates,key=lambda x:x['objective'],reverse=True)[:2]:
            trial(dict(parent['params'],density=1.))
        stable=[]
        for row in sorted([c for c in candidates if c['params']['density']<1],key=lambda x:x['objective'],reverse=True)[:8]:
            res=make(channels,7,row['params']);proj=FeedbackToChannel(35,32,64,seed=8).to(D)
            row['echo']=echo(res,proj,rd)
            if row['echo'][-1]['rms']<.001:stable.append(row)
        if not stable:raise RuntimeError('No stable top candidate; inspect before deployment')
        best=stable[0];winners.append(best);atomic('winners.json',winners)
    # Fresh random streams and unused real chunks; test scores never select params.
    nd=noise(batch=32,length=464,seed=5678);rd=realistic(data,12,44,5678)
    results=[]
    for winner in winners:
        for seed in (7,19,101):
            for baseline in (True,False):
                row=evaluate(winner['channels'],winner['params'],seed,nd,rd,
                    train_n=24,test_n=28,baseline=baseline,stability=True)
                results.append(row)
                atomic('heldout.json',results)
                print('HELDOUT',winner['channels'],seed,baseline,row['objective'],flush=True)
    atomic('complete.json',dict(time=time.time(),winners=winners,heldout=results,
        protocol='Validation-only selection. Heldout uses fresh IID streams and disjoint real chunks. Real images and action/density feedback are recorded; independent tag replaces model feedback to measure memory without visual autocorrelation.'))


if __name__=='__main__':search()
