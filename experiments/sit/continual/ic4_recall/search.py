"""Training-only mask-search pilot on frozen legacy FTN banks.

This produces search supervision and a diagnostic ceiling, not a finite-adapter
selector. Candidate masks are combinations of the same rank-one atoms. Neither
the independent evaluator nor support/test images select the candidates.
"""
import argparse
import hashlib
import os
import time

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from ..model import transport_batch
from .common import ASSETS, ROOT, ART, SUITES, atomic, build_model, load_data, run_path
from .recognition import style_features


def dino_model():
    os.environ['TORCH_HOME'] = str(ROOT/'torch')
    return torch.hub.load(str(ROOT/'torch/hub/facebookresearch_dinov2_main'),
                          'dinov2_vitb14', source='local').cuda().eval().requires_grad_(False)


@torch.no_grad()
def encode_images(net, images):
    prep = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                               transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    pil = [to_pil_image(im) for im in images]
    chunks = []
    for start in range(0,len(pil),32):
        batch = torch.stack([prep(im) for im in pil[start:start+32]]).cuda()
        chunks.append(F.normalize(net(batch).float(),dim=1).cpu())
    return dict(dino=torch.cat(chunks), style=style_features(pil))


def mmd(x,y,scale):
    x,y=x/scale,y/scale
    def kernel(a,b):
        distance=torch.cdist(a,b).square()/a.shape[1]
        return sum(torch.exp(-distance/band) for band in [.25,1.,4.])/3
    # Biased MMD is stable for the small development sample batches.
    return float(kernel(x,x).mean()+kernel(y,y).mean()-2*kernel(x,y).mean())


def generate(suite,m,vae,mask,count,seed):
    if suite=='filters':
        from ..run import generate as sample
        return sample(m,vae,mask,count=count,seed=seed,steps=50,batch=16)
    from ..identity_distinct.common import generate as sample
    return sample(m,vae,mask,count=count,seed=seed)


def search_mask(m,z,y,seed,initial=None,steps=30):
    """Bayesian STE top-k search; fixed bank, fresh multi-time transport draws."""
    torch.manual_seed(seed)
    gen=torch.Generator(device='cuda').manual_seed(seed)
    # Same underlying images, four independent corruptions: not more support.
    xy=[transport_batch(z,y,gen) for _ in range(4)]
    x,target=torch.cat([v[0] for v in xy]),torch.cat([v[1] for v in xy])
    saved=m.get_reconfig_state()
    try:
        if initial is None:
            m.cold_reconfigure(x,target,steps=steps,take_best=False)
        else:
            # A softened prior allows moves that saturated archived logits resist.
            m.mask_logits.copy_(torch.where(initial>.5,.5,-.5))
            m._posterior_update(x,target,steps=steps,lr=.3,sample_size=None,
                kl_weight=.05,temperature=1.,prior_decay=1.,stochastic_mask=True,changepoint=False)
            m.m.copy_(m._topk_mask(m.mask_logits))
        return m.m.detach().clone()
    finally:
        m.set_reconfig_state(saved)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite',choices=SUITES,required=True)
    p.add_argument('--arm',choices=['replay','noreplay'],required=True)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--episodes',type=int,default=3)
    p.add_argument('--count',type=int,default=20)
    a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    cfg=SUITES[a.suite];run=run_path(a.suite,a.arm,a.seed)
    work=ROOT/'search'/a.suite/f'{a.arm}_s{a.seed}'
    out=ART/a.suite/f'{a.arm}_s{a.seed}'/'search'
    work.mkdir(parents=True,exist_ok=True);out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=(work/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=run/'latest.pt';source_stat=(path.stat().st_size,path.stat().st_mtime_ns)
    ck=torch.load(path,map_location='cpu',weights_only=False)
    m=build_model(a.suite,a.seed);m.load_adapter(ck['adapter'])
    original_saved=ck.get('saved',ck.get('saved_masks'))
    assert original_saved is not None
    # Freeze weights explicitly. Mask gradients still pass through _run.
    m.requires_grad_(False)
    versions=[p._version for p in m.parameters()]
    ds=load_data(a.suite,splits=('train',))
    vae=AutoencoderKL.from_pretrained(ASSETS/'vae').cuda().eval().requires_grad_(False)
    net=dino_model()
    cache=ROOT/'features'/a.suite
    features={n:{k:torch.load(cache/f'{n}_train_{k}.pt',weights_only=True) for k in ['dino','style']} for n in cfg['names']}
    scale={k:torch.cat([features[n][k] for n in cfg['names']]).std(0).clamp_min(.01) for k in ['dino','style']}
    records=[];started=time.time()
    for domain_index,n in enumerate(cfg['names']):
        train=ds[n]['train'];gen=torch.Generator().manual_seed(6100+domain_index)
        order=torch.randperm(len(train['z']),generator=gen)
        val_count=max(16,len(order)//5);validation=order[-val_count:];fit=order[:-val_count]
        candidates=[dict(kind='archived',mask=original_saved[n].cpu(),fit_indices=[])]
        for episode in range(a.episodes):
            ids=fit[torch.randperm(len(fit),generator=gen)[:min(32,len(fit))]]
            for kind in ['cold','warm']:
                maskfile=work/f'{n}_{kind}_{episode}.pt'
                if maskfile.exists():
                    mask=torch.load(maskfile,weights_only=True)
                else:
                    mask=search_mask(m,train['z'][ids].cuda(),train['y'][ids].cuda(),
                        8100+domain_index*100+episode,original_saved[n].cuda() if kind=='warm' else None).cpu()
                    torch.save(mask,maskfile)
                candidates.append(dict(kind=f'{kind}{episode}',mask=mask,fit_indices=ids.tolist()))
        seen={}
        for candidate in candidates:
            mask=candidate.pop('mask');sig=hashlib.sha256(mask.numpy().tobytes()).hexdigest()
            if sig in seen:
                records.append(dict(seen[sig],kind=candidate['kind'],duplicate_of=seen[sig]['kind']))
                continue
            name=f'{n}_{candidate["kind"]}'
            featfile=work/f'{name}_generated_features.pt'
            if featfile.exists():
                generated=torch.load(featfile,weights_only=True)
            else:
                images=generate(a.suite,m,vae,mask.cuda(),a.count,seed=70231)
                generated=encode_images(net,images)
                torch.save(generated,featfile)
                save_image(images[:20],out/f'{name}.png',nrow=5)
            distances={k:mmd(generated[k],features[n][k][validation],scale[k]) for k in ['dino','style']}
            score=distances['style']+.25*distances['dino'] if a.suite=='filters' else distances['dino']+.1*distances['style']
            record=dict(domain=n,**candidate,mask=mask.tolist(),signature=sig,distances=distances,
                        selection_loss=score,grid=f'{name}.png',validation_indices=validation.tolist(),count=a.count)
            seen[sig]=record;records.append(record)
            atomic(out/'candidates.json',records)
            print('CANDIDATE',a.suite,a.arm,n,candidate['kind'],score,distances,flush=True)
        atomic(out/'candidates.json',records)
        best=min((r for r in records if r['domain']==n),key=lambda r:r['selection_loss'])
        print('TRAIN_VALIDATED_BEST',n,best['kind'],best['selection_loss'],flush=True)
    assert versions==[p._version for p in m.parameters()]
    assert source_stat==(path.stat().st_size,path.stat().st_mtime_ns)
    atomic(out/'complete.json',dict(complete=True,suite=a.suite,arm=a.arm,seed=a.seed,seconds=time.time()-started,
        original_checkpoint_unchanged=True,weights_unchanged=True,selection_data='internal training validation only',
        independent_evaluator_used_for_selection=False,support_or_test_used=False,
        best={n:min((r for r in records if r['domain']==n),key=lambda r:r['selection_loss'])['kind'] for n in cfg['names']}))


if __name__=='__main__':main()
