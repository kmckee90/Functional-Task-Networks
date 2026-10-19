"""Pod-only matched-reservoir diagnostics. Does not load/mutate live agents."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn.functional as F
from .legacy.models import Conv2dReservoir, FeedbackToChannel

ROOT=Path('/home/dev/globo_dreamer_state/diagnostics/reservoir_capacity')
ROOT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(4)
torch.set_grad_enabled(False)
DEVICE='cuda:0'
LAGS=[0,1,2,4,8,16,32,64]
B,T,WARM=32,464,64


def save(name,data):
    path=ROOT/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)


def inputs():
    gen=torch.Generator(device=DEVICE).manual_seed(1234)
    image=torch.rand(B,T,16,device=DEVICE,generator=gen)
    # Low-frequency image variations around a fixed textured background.
    texture=torch.rand(1,3,32,64,device=DEVICE,generator=gen)*.2
    pixels=F.interpolate(image.reshape(-1,1,4,4),size=(32,64),mode='nearest')*.6+.1+texture
    pixels=pixels.reshape(B,T,3,32,64)
    density=torch.rand(B,T,1,device=DEVICE,generator=gen)*.2
    model=torch.rand(B,T,1,device=DEVICE,generator=gen)
    action=torch.randint(25,(B,T),device=DEVICE,generator=gen)
    bins=F.one_hot((density[...,0].tanh()*8).long(),8).float()
    onehot=F.one_hot(action,25).float()
    feedback=torch.cat([onehot,density,bins,model],-1)
    target=torch.cat([image,density,model,onehot],-1)
    return pixels,feedback,target


def ridge(features,targets,groups,split=(24,28)):
    # Whole independent streams are assigned to train/validation/test.
    a,b=split
    xx=[v.reshape(-1,v.shape[-1]) for v in (features[:a],features[a:b],features[b:])]
    yy=[v.reshape(-1,v.shape[-1]) for v in (targets[:a],targets[a:b],targets[b:])]
    mean=xx[0].mean(0);sd=xx[0].std(0).clamp_min(.05)
    xx=[(v-mean)/sd for v in xx]
    ym=yy[0].mean(0);ys=yy[0].std(0).clamp_min(.001)
    y=[(v-ym)/ys for v in yy]
    gram=xx[0].T@xx[0]/len(xx[0]);rhs=xx[0].T@y[0]/len(xx[0])
    eye=torch.eye(gram.shape[0],device=DEVICE)
    best={}
    for reg in (.0001,.001,.01,.1,1.):
        weights=torch.linalg.solve(gram+reg*eye,rhs)
        val=xx[1]@weights;pred=xx[2]@weights
        for name,lo,hi,kind in groups:
            score=float(((val[:,lo:hi]-y[1][:,lo:hi])**2).mean())
            if name not in best or score<best[name]['validation_mse']:
                err=((pred[:,lo:hi]-y[2][:,lo:hi])**2).mean(0)
                var=((y[2][:,lo:hi]-y[2][:,lo:hi].mean(0))**2).mean(0).clamp_min(1e-8)
                row=dict(r2=float((1-err/var).mean()),alpha=reg,validation_mse=score)
                if kind=='action':
                    logits=pred[:,lo:hi]*ys[lo:hi]+ym[lo:hi]
                    row['accuracy']=float((logits.argmax(-1)==yy[2][:,lo:hi].argmax(-1)).float().mean())
                best[name]=row
    return best


def synthetic(seed,channels,pixels,feedback,target):
    torch.manual_seed(seed)
    reservoir=Conv2dReservoir(4,num_channels=channels,seed=seed).to(DEVICE)
    projection=FeedbackToChannel(35,32,64,seed=seed+1).to(DEVICE)
    h=reservoir.init_hidden(B,32,64,DEVICE)
    states=[];saturation=[];derivative=[];drive=[]
    for t in range(T):
        x=torch.cat([pixels[:,t],projection(feedback[:,t])],1)
        h=reservoir(x,h)
        if t>=WARM:
            states.append(h.flatten(1).clone())
            saturation.append((h.abs()>.99).float().mean())
            derivative.append((1-h.square()).mean())
            if t==WARM:
                drive=[float(reservoir.cnn1(reservoir.input_scale*x).std()),
                       float(reservoir.cnn2(reservoir.recur_scale*h).std())]
    states=torch.stack(states,1)
    targets=torch.cat([target[:,WARM-lag:T-lag] for lag in LAGS],-1)
    groups=[]
    for i,lag in enumerate(LAGS):
        for name,lo,hi in [('image',0,16),('density',16,17),('model',17,18),('action',18,43)]:
            groups.append((f'{name}_lag{lag}',i*43+lo,i*43+hi,name))
    start=time.time();scores=ridge(states,targets,groups)
    if seed==7 and os.environ.get('PROBE_NONLINEAR'):
        nonlinear_scores=nonlinear(states,targets,groups)
        save(f'nonlinear_c{channels}.json',nonlinear_scores)
    result=dict(seed=seed,channels=channels,features=channels*36,
        saturation_099=float(torch.stack(saturation).mean()),mean_tanh_derivative=float(torch.stack(derivative).mean()),
        constant_feature_fraction=float((states.reshape(-1,states.shape[-1]).std(0)<.01).float().mean()),
        input_drive_std=drive[0],recurrent_drive_std=drive[1],scores=scores,probe_seconds=time.time()-start)
    # Same continuing input, different initial conditions: persistence != recall.
    gen=torch.Generator(device=DEVICE).manual_seed(42)
    ha=torch.zeros_like(h);hb=torch.rand(h.shape,device=DEVICE,generator=gen)*2-1
    divergence=[]
    for t in range(128):
        x=torch.cat([pixels[:,t],projection(feedback[:,t])],1)
        ha=reservoir(x,ha);hb=reservoir(x,hb)
        if t in (0,1,3,7,15,31,63,127):
            divergence.append(dict(step=t+1,rms=float((ha-hb).square().mean().sqrt()),
                sign_disagreement=float((ha.sign()!=hb.sign()).float().mean())))
    result['initial_state_divergence']=divergence
    save(f'iid_seed{seed}_c{channels}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('scores','initial_state_divergence')}),flush=True)
    print(json.dumps(scores),flush=True)
    return result


def nonlinear(features,targets,groups):
    """Independent-stream nonlinear readout; validation selects epoch per task."""
    torch.manual_seed(77)
    xx=[v.reshape(-1,v.shape[-1]) for v in (features[:24],features[24:28],features[28:])]
    yy=[v.reshape(-1,v.shape[-1]) for v in (targets[:24],targets[24:28],targets[28:])]
    mean=xx[0].mean(0);sd=xx[0].std(0).clamp_min(.05);xx=[(v-mean)/sd for v in xx]
    ym=yy[0].mean(0);ys=yy[0].std(0).clamp_min(.001);yy=[(v-ym)/ys for v in yy]
    net=torch.nn.Sequential(torch.nn.Linear(xx[0].shape[-1],512),torch.nn.ReLU(),
        torch.nn.Linear(512,512),torch.nn.ReLU(),torch.nn.Linear(512,yy[0].shape[-1])).to(DEVICE)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    best={}
    for epoch in range(30):
        order=torch.randperm(len(xx[0]),device=DEVICE)
        with torch.enable_grad():
            for idx in order.split(512):
                opt.zero_grad(set_to_none=True)
                loss=(net(xx[0][idx])-yy[0][idx]).square().mean()
                loss.backward();opt.step()
        val=net(xx[1]);pred=net(xx[2])
        for name,lo,hi,kind in groups:
            score=float((val[:,lo:hi]-yy[1][:,lo:hi]).square().mean())
            if name not in best or score<best[name]['validation_mse']:
                err=(pred[:,lo:hi]-yy[2][:,lo:hi]).square().mean(0)
                var=(yy[2][:,lo:hi]-yy[2][:,lo:hi].mean(0)).square().mean(0).clamp_min(1e-8)
                row=dict(r2=float((1-err/var).mean()),epoch=epoch+1,validation_mse=score)
                if kind=='action':
                    logits=pred[:,lo:hi]*ys[lo:hi]+ym[lo:hi]
                    truth=yy[2][:,lo:hi]*ys[lo:hi]+ym[lo:hi]
                    row['accuracy']=float((logits.argmax(-1)==truth.argmax(-1)).float().mean())
                best[name]=row
    print('NONLINEAR',features.shape[-1],json.dumps(best),flush=True)
    return best


def real_inputs():
    # One identical replay dataset for both reservoirs. Random order of disjoint
    # chunks avoids systematic train/test ordering; reset at every chunk.
    run=Path('/home/dev/globo_dreamer_state/runs/continuous_fb_w1p0_s7')
    files=sorted((run/'replay').glob('*.npz'),reverse=True)
    rows=[];sources=[]
    progress=json.loads((run/'live/progress.json').read_text())
    scale=max(.001,progress['model_input_scale'])
    for path in files:
        try:
            with np.load(path) as ar:
                if len(ar['image'])!=256:continue
                rows.append({k:ar[k].copy() for k in ('image','action','density','is_first','action_disagreement')})
                sources.append(path.name)
        except (OSError,ValueError,KeyError):continue
        if len(rows)==48:break
    if len(rows)<48:raise RuntimeError('Need 48 complete saved chunks')
    order=np.random.default_rng(7).permutation(48)
    data={k:torch.as_tensor(np.stack([rows[i][k] for i in order]),device=DEVICE) for k in rows[0]}
    images=data['image'].permute(0,1,4,2,3).float()/255
    image=F.interpolate(images.flatten(0,1),size=(32,64),mode='area').reshape(48,256,3,32,64)
    density=data['density'].float()
    action=torch.roll(data['action'].long(),1,1)
    raw=data['action_disagreement'].gather(-1,data['action'].long()[...,None])
    model=torch.tanh(torch.roll(raw,1,1)/(3*scale))
    reset=data['is_first'].bool();reset[:,0]=True
    action[reset]=0;model[reset]=0
    reward=density.clone();reward[reset]=0
    feedback=torch.cat([F.one_hot(action,25).float(),reward,
        F.one_hot((density[...,0].tanh().clamp(0,1-1e-6)*8).long(),8).float(),model],-1)
    target=F.adaptive_avg_pool2d(image.flatten(0,1), (4,4)).reshape(48,256,48)
    save('real_dataset.json',dict(run=run.name,chunks=[sources[i] for i in order],
        note='Model feedback approximated using current RMS, not historical normalization; reset per chunk.',scale=scale))
    return image,feedback,target,reset


def real_probe(channels,pixels,feedback,target,reset):
    seed=7;torch.manual_seed(seed)
    res=Conv2dReservoir(4,num_channels=channels,seed=seed).to(DEVICE)
    proj=FeedbackToChannel(35,32,64,seed=seed+1).to(DEVICE)
    h=res.init_hidden(48,32,64,DEVICE);states=[];sat=[]
    for t in range(256):
        h[reset[:,t]]=0
        h=res(torch.cat([pixels[:,t],proj(feedback[:,t])],1),h)
        if t>=64:states.append(h.flatten(1).clone());sat.append((h.abs()>.99).float().mean())
    features=torch.stack(states,1)
    result=dict(channels=channels,saturation_099=float(torch.stack(sat).mean()),
        constant_feature_fraction=float((features.flatten(0,1).std(0)<.01).float().mean()),
        image_decoding=ridge(features,target[:,64:],[('image',0,48,'image')],split=(32,40)))
    save(f'real_c{channels}.json',result);print('REAL',json.dumps(result),flush=True)
    return result


def main():
    pixels,feedback,target=inputs();results=[]
    for seed in ((7,) if os.environ.get('PROBE_NONLINEAR') else (7,19,101)):
        for channels in (64,128):
            results.append(synthetic(seed,channels,pixels,feedback,target))
    del pixels,feedback,target
    real=real_inputs()
    rr=[real_probe(channels,*real) for channels in (64,128)]
    save('nonlinear_summary.json' if os.environ.get('PROBE_NONLINEAR') else 'summary.json',dict(iid=results,real=rr,lags=LAGS,
        protocol='Independent random inputs; 24 train / 4 validation / 4 test streams, 400 scored steps each; ridge regularization selected by validation per target and lag. No live agent mutation.'))


if __name__=='__main__':main()
