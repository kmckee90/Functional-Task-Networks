"""Stage-local unified direct4, with no fitting on future/support/test images."""
import hashlib
import json
import time
import torch
from diffusers import AutoencoderKL
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from .common import *
from ..ic4_recall.search import search_mask,dino_model,encode_images,mmd
from ..ic4_recall.atom_conditioner import AtomReadout
from ..model import transport_batch
from ..run import query_loss,balanced_indices

_cache={}

def load(path,default=None):return json.loads(path.read_text()) if path.exists() else default

def components():
    if not _cache:
        from ..union11.metrics import Metrics
        _cache['vae']=AutoencoderKL.from_pretrained(ASSETS/'vae').cuda().eval().requires_grad_(False)
        _cache['dino']=dino_model()
        _cache['metric']=Metrics()
    return _cache

def repair(m,ds,ck,folder,c):
    seen=NAMES[:ck['completed']]
    if not seen:return {}
    cache={n:{b:features(n,'train',b) for b in ['dino','style']} for n in seen}
    scale={b:torch.cat([cache[n][b] for n in seen]).std(0).clamp_min(.01) for b in ['dino','style']}
    teachers={}
    for j,n in enumerate(seen):
        output=folder/f'search_{n}.json'
        if output.exists():
            rr=load(output);teachers[n]=min(rr,key=lambda r:r['selection_loss']);continue
        train=ds[n]['train'];gen=torch.Generator().manual_seed(6100+j)
        order=torch.randperm(len(train['z']),generator=gen)
        val=order[-max(1,len(order)//5):];fit=order[:-len(val)]
        candidates=[('archived',ck['saved'][n].cuda(),[])]
        for episode in range(3):
            ids=fit[torch.randperm(len(fit),generator=gen)[:32]]
            for kind in ['cold','warm']:
                mask=search_mask(m,train['z'][ids],train['y'][ids],8100+j*100+episode,
                                 ck['saved'][n].cuda() if kind=='warm' else None)
                candidates.append((f'{kind}{episode}',mask,ids.tolist()))
        records=[];scored={}
        for kind,mask,ids in candidates:
            sig=hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
            if sig not in scored:
                ims=generate(m,c['vae'],mask,20,70231,domain=n)
                fs=encode_images(c['dino'],ims)
                distances={b:mmd(fs[b],cache[n][b][val],scale[b]) for b in ['dino','style']}
                scored[sig]=distances
                save_image(ims,folder/f'search_{n}_{kind}.png',nrow=5)
            distances=scored[sig]
            score=distances['style']+.25*distances['dino'] if n in FILTERS else distances['dino']+.1*distances['style']
            records.append(dict(domain=n,kind=kind,mask=mask.cpu().tolist(),selection_loss=score,
                                distances=distances,fit_indices=ids,validation_indices=val.tolist()))
        atomic(output,records);teachers[n]=min(records,key=lambda r:r['selection_loss'])
        print('REPAIR',ck['stage'],n,teachers[n]['kind'],teachers[n]['selection_loss'],flush=True)
    return teachers

def fit(teachers,folder):
    dim=unified(NAMES[0],'support').shape[1]
    gen=torch.Generator().manual_seed(93811)
    projection=torch.randn(dim,128,generator=gen).cuda()
    model=AtomReadout(projection,torch.zeros(dim+129,1024,device='cuda'))
    gram=torch.eye(dim+129,device='cuda',dtype=torch.float64)*.01
    rhs=torch.zeros(dim+129,1024,device='cuda',dtype=torch.float64)
    for n in teachers:
        # Rebuild only from retained images, including all four at 5%.
        # Repair holdout is internal to mask selection, not a readout test set.
        x=unified(n,'train')
        ids=torch.stack([torch.randperm(len(x),generator=gen)[:4] for _ in range(512)])
        design=model.features(x[ids].cuda()).double()
        target=(torch.tensor(teachers[n]['mask'],device='cuda')*2-1).double().expand(512,-1)
        gram+=design.T@design;rhs+=design.T@target
    if teachers:model.weight.copy_(torch.linalg.solve(gram,rhs).float())
    # Before any task, fixed seed-only mask; no fit on any future task.
    prior=torch.zeros(1024,device='cuda');prior[torch.randperm(1024,generator=gen)[:256]]=1
    torch.save(dict(projection=projection.cpu(),weight=model.weight.cpu(),teachers=teachers),folder/'conditioner.pt')
    return model,prior

def stage_eval(m,ds,ck,out,do_repair,final):
    marker=out/f'{ck["stage"]}_complete.json'
    if marker.exists():return
    folder=out/ck['stage'];folder.mkdir(exist_ok=True)
    started=time.time();state=m.get_reconfig_state();controller=m.controller.mask
    flags=[p.requires_grad for p in m.parameters()];versions=[p._version for p in m.parameters()]
    cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
    m.requires_grad_(False)
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            c=components();teachers=repair(m,ds,ck,folder,c);conditioner,prior=fit(teachers,folder)
            atomic(folder/'teachers.json',teachers)
            results=load(folder/'results.json',[])
            done={(r['domain'],r['draw']) for r in results};generation={}
            for j,n in enumerate(NAMES):
                support=unified(n,'support');support_images=None
                test=ds[n]['test'];qi=balanced_indices(test['y'],40) if n in FILTERS else torch.arange(len(test['z']),device='cuda')
                queries=[transport_batch(test['z'][qi],test['y'][qi],torch.Generator(device='cuda').manual_seed(1900+r)) for r in range(4)]
                qx=torch.cat([x for x,y in queries]);qy=torch.cat([y for x,y in queries])
                query_cache={}
                stored_mse=None
                if n in ck['saved']:stored_mse=query_loss(m,qx,qy,ck['saved'][n].cuda())['loss']
                for draw in range(10):
                    if (n,draw) in done:continue
                    ids=torch.randperm(len(support),generator=torch.Generator().manual_seed(15001+draw))[:4]
                    with torch.no_grad():mask=conditioner.mask(support[ids].cuda()[None],256)[0] if teachers else prior.clone()
                    assert int(mask.sum())==256
                    sig=hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
                    family='filters' if n in FILTERS else 'other';count=200 if n in FILTERS else 64
                    key=f'{family}_{sig[:16]}'
                    cachefolder=folder/'generated'/key;cachefolder.mkdir(parents=True,exist_ok=True)
                    metricfile=cachefolder/'metrics.json'
                    if key not in generation:
                        if metricfile.exists():generation[key]=load(metricfile)
                        else:
                            ims=generate(m,c['vae'],mask,count,99233,domain=n)
                            sampledir=cachefolder/'samples';sampledir.mkdir(exist_ok=True)
                            for i,im in enumerate(ims):save_image(im,sampledir/f'{i:04d}.png')
                            save_image(ims[:20 if n in FILTERS else 16],cachefolder/'grid.png',nrow=5 if n in FILTERS else 8)
                            generation[key]=c['metric'](ims,sampledir,n in FILTERS)
                            atomic(metricfile,generation[key])
                    if sig not in query_cache:query_cache[sig]=query_loss(m,qx,qy,mask)['loss']
                    target=torch.tensor(teachers[n]['mask'],device='cuda') if n in teachers else None
                    record=dict(stage=ck['stage'],completed=ck['completed'],domain=n,draw=draw,
                        relation='forward' if j>=ck['completed'] else 'current' if j==ck['completed']-1 else 'backward',
                        support_indices=ids.tolist(),mask=mask.cpu().tolist(),signature=sig,count=count,
                        metrics=generation[key][n],query_mse=query_cache[sig],stored_query_mse=stored_mse,
                        exact_teacher=bool(torch.equal(mask,target)) if target is not None else None,
                        teacher_overlap=float((mask*target).sum()/256) if target is not None else None,
                        grid=str((cachefolder/'grid.png').relative_to(out)))
                    results.append(record);atomic(folder/'results.json',results)
                    if draw==0:
                        if support_images is None:support_images=images(n,'support')
                        save_image(torch.stack([to_tensor(support_images[i]) for i in ids]),folder/f'{n}_support4.png',nrow=4)
                        print('DIRECT4',ck['stage'],n,record['metrics'],record['exact_teacher'],flush=True)
            assert len(results)==110
            atomic(folder/'summary.json',dict(complete=True,completed=ck['completed'],stage=ck['stage'],
                targets={n:dict(metrics={key:sum(r['metrics'][key] for r in results if r['domain']==n)/10
                                        for key in next(r for r in results if r['domain']==n)['metrics']},
                    query_mse=sum(r['query_mse'] for r in results if r['domain']==n)/10,
                    stored_query_mse=next(r['stored_query_mse'] for r in results if r['domain']==n),
                    exact_teacher=sum(r['exact_teacher'] for r in results if r['domain']==n)/10 if n in teachers else None,
                    unique_masks=len({r['signature'] for r in results if r['domain']==n})) for n in NAMES},
                seconds=time.time()-started,training_tasks=list(teachers),support=4,refinement=False,
                future_data_used_for_fitting=False))
    finally:
        m.set_reconfig_state(state);m.controller.mask=controller
        for p,flag in zip(m.parameters(),flags):p.requires_grad_(flag)
    assert versions==[p._version for p in m.parameters()]
    assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
    for key in ['m','mask_logits']:assert torch.equal(state[key],m.get_reconfig_state()[key])
    atomic(marker,dict(complete=True,weights_unchanged=True,rng_restored=True,posterior_restored=True))
    print('STAGE_COMPLETE',ck['stage'],'seconds',time.time()-started,flush=True)
