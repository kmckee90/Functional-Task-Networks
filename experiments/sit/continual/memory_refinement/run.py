"""Rerunnable final-checkpoint comparison; never modifies the source experiments."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from ..memory_budget import common as C
from ..memory_budget.evaluate import components
from ..ic4_recall.atom_conditioner import AtomReadout
from ..ic4_recall.search import search_mask,encode_images
from ..model import transport_batch
from ..run import query_loss,balanced_indices

ROOT=C.ASSETS/'memory_refinement_v1'
ART=Path(__file__).parent/'artifacts'
MODES=[('direct',0,0),('refine10_s0',10,92000),('refine30_s0',30,92000),('refine30_s1',30,92001),('refine30_s2',30,92002)]

def read(p):return json.loads(p.read_text())

def pictures(folder):
    return torch.stack([to_tensor(Image.open(p).convert('RGB')) for p in sorted(folder.glob('*.png'))])

@torch.no_grad()
def distances(generated,real,retained):
    # Each branch is normalized separately; these are descriptive distance proxies.
    out={}
    for branch in ['dino','style']:
        g=torch.nn.functional.normalize(generated[branch].float(),dim=1)
        r=torch.nn.functional.normalize(real[branch].float(),dim=1)
        t=torch.nn.functional.normalize(retained[branch].float(),dim=1)
        d=torch.cdist(r,g)
        out[branch+'_real_to_generated_nn']=float(d.min(1).values.mean())
        out[branch+'_generated_to_real_nn']=float(d.min(0).values.mean())
        out[branch+'_pair_distance']=float(torch.pdist(g).mean())
        out[branch+'_generated_to_retained_nn']=float(torch.cdist(g,t).min(1).values.mean())
    return out

def paired_queries(split,ids,seed):
    pairs=[transport_batch(split['z'][ids].cuda(),split['y'][ids].cuda(),
            torch.Generator(device='cuda').manual_seed(seed+r)) for r in range(4)]
    return torch.cat([x for x,y in pairs]),torch.cat([y for x,y in pairs])

def worker(b):
    tag=f'replay_p{b:02d}_s125';source=C.ART/tag;out=ART/tag;work=ROOT/tag
    if not (source/'complete.json').exists():
        print('SKIP_UNFINISHED',tag,flush=True);return
    out.mkdir(parents=True,exist_ok=True);work.mkdir(parents=True,exist_ok=True)
    lock=(work/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (out/'complete.json').exists():return
    checkpoint=C.ROOT/'runs'/tag/'latest.pt'
    source_stat=(checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ck['completed']==11 and ck['stage']=='after_11_BMW_i8'
    stage=source/ck['stage'];original=read(stage/'results.json')
    m=C.model(125);m.load_adapter(ck['adapter']);m.requires_grad_(False)
    versions=[p._version for p in m.parameters()]
    condition=torch.load(stage/'conditioner.pt',map_location='cpu',weights_only=False)
    mapper=AtomReadout(condition['projection'].cuda(),condition['weight'].cuda())
    c=components();rows=read(out/'results.json') if (out/'results.json').exists() else []
    config=dict(source_checkpoint=str(checkpoint),source_size=source_stat[0],source_mtime_ns=source_stat[1],
        completed=11,support_draw=0,support_seed=15001,support_size=4,generation_seed=99233,
        modes=MODES,selection='all prespecified modes reported; no evaluator-based winner selection',
        refinement='same softened-mask Bayesian search used by repair: lr .3, KL .05, temperature1, stochastic masks; four support images with four transport corruptions',
        readout_fitted=False,bank_frozen=True,training_images_used_for_refinement=False,
        feature_diagnostics='DINO/style L2-normalized pairwise Euclidean distances; proxies, not calibrated precision/recall; held-out reference counts 40 filters, 24 faces/BMW, 23 anteater',
        source_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if (out/'protocol.json').exists():assert read(out/'protocol.json')==json.loads(json.dumps(config))
    else:C.atomic(out/'protocol.json',config)
    # Start with the suspected low-memory failure, then cover every target.
    order=['George_W_Bush']+[n for n in C.NAMES if n!='George_W_Bush']
    for n in order:
        direct_record=next(r for r in original if r['domain']==n and r['draw']==0)
        ids=torch.tensor(direct_record['support_indices'])
        with torch.no_grad():direct=mapper.mask(C.unified(n,'support')[ids].cuda()[None],256)[0]
        assert direct.cpu().tolist()==direct_record['mask']
        support=torch.load(C.DATA[C.SOURCE[n]]/f'{n}_support.pt',weights_only=True,map_location='cpu')
        test=torch.load(C.DATA[C.SOURCE[n]]/f'{n}_test.pt',weights_only=True,map_location='cpu')
        qi=balanced_indices(test['y'],40) if n in C.FILTERS else torch.arange(len(test['z']))
        qx,qy=paired_queries(test,qi,1900);sx,sy=paired_queries(support,ids,39100)
        directory=out/n;directory.mkdir(exist_ok=True)
        refpath=work/f'{n}_reference.pt'
        if refpath.exists():real=torch.load(refpath,weights_only=True)
        else:
            ims=C.images(n,'test');ri=qi.tolist()
            if n=='Giant_anteater':ri=[i for i in ri if i!=23]
            real=encode_images(c['dino'],torch.stack([to_tensor(ims[i]) for i in ri]));torch.save(real,refpath)
        retained=ck['memory'][n]['features']
        grids=[]
        for mode,steps,seed in MODES:
            previous=next((r for r in rows if r['target']==n and r['mode']==mode),None)
            if previous is not None:
                grids.append(pictures(Path(previous['sampledir']))[:8]);continue
            begin=time.time()
            if steps:
                mask=search_mask(m,support['z'][ids].cuda(),support['y'][ids].cuda(),seed,direct,steps=steps)
            else:mask=direct.clone()
            assert int(mask.sum())==256
            sig=hashlib.sha256(mask.cpu().numpy().tobytes()).hexdigest()
            count=direct_record['count']
            if torch.equal(mask,direct):
                sampledir=source/Path(direct_record['grid']).parent/'samples'
                ims=pictures(sampledir);metrics=direct_record['metrics']
            else:
                cache=work/'generated'/n/sig;sampledir=cache/'samples'
                cache.mkdir(parents=True,exist_ok=True)
                if (cache/'metrics.json').exists():metrics=read(cache/'metrics.json');ims=pictures(sampledir)
                else:
                    ims=C.generate(m,c['vae'],mask,count,99233,domain=n)
                    sampledir.mkdir(exist_ok=True)
                    for i,im in enumerate(ims):save_image(im,sampledir/f'{i:04d}.png')
                    metrics=c['metric'](ims,sampledir,n in C.FILTERS)[n]
                    C.atomic(cache/'metrics.json',metrics)
                    ims=pictures(sampledir)
            assert len(ims)==count
            features=encode_images(c['dino'],ims)
            diag=distances(features,real,retained)
            mse=query_loss(m,qx,qy,mask)['loss'];support_mse=query_loss(m,sx,sy,mask)['loss']
            if not steps:assert abs(mse-direct_record['query_mse'])<1e-4
            grid=directory/f'{mode}.png';save_image(ims[:16],grid,nrow=8)
            grids.append(ims[:8])
            record=dict(budget=b,target=n,mode=mode,steps=steps,refinement_seed=seed,
                support_indices=ids.tolist(),mask=mask.cpu().tolist(),mask_overlap=float((mask*direct).sum()/256),
                atoms_replaced=int(256-(mask*direct).sum()),signature=sig,count=count,metrics=metrics,
                query_mse=mse,support_mse=support_mse,diagnostics=diag,
                sampledir=str(sampledir),grid=str(grid.relative_to(out)),seconds=time.time()-begin)
            rows.append(record);C.atomic(out/'results.json',rows)
            print('RESULT',tag,n,mode,metrics,'replaced',record['atoms_replaced'],'coverage',diag['dino_real_to_generated_nn'],flush=True)
            assert versions==[p._version for p in m.parameters()]
        save_image(torch.cat(grids),directory/'paired_comparison.png',nrow=8)
        support_images=C.images(n,'support')
        save_image(torch.stack([to_tensor(support_images[i]) for i in ids]),directory/'support4.png',nrow=4)
    assert len(rows)==55
    assert source_stat==(checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)
    C.atomic(out/'complete.json',dict(complete=True,conditions=55,targets=11,weights_unchanged=True,source_checkpoint_unchanged=True))


def launch():
    ROOT.mkdir(exist_ok=True);ART.mkdir(exist_ok=True)
    lock=(ROOT/'launcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    active={};skipped=[];done=[];failed=[];last=-1
    for gpu,b in enumerate([50,25,10,5]):
        tag=f'replay_p{b:02d}_s125'
        if (ART/tag/'complete.json').exists():done.append(tag);continue
        if not (C.ART/tag/'complete.json').exists():skipped.append(tag);continue
        f=(ROOT/f'{tag}.log').open('a',buffering=1)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),TORCH_HOME=str(C.ASSETS/'ic4_recall/torch'),OMP_NUM_THREADS='4')
        p=subprocess.Popen([sys.executable,'-u','-m','continual.memory_refinement.run','--fraction',str(b)],env=env,stdout=f,stderr=subprocess.STDOUT)
        active[tag]=(p,f,gpu)
    while active:
        for tag,(p,f,gpu) in list(active.items()):
            if p.poll() is not None:
                (done if p.returncode==0 else failed).append(tag);f.close();del active[tag]
        C.atomic(ART/'status.json',dict(active=[dict(run=t,pid=p.pid,gpu=g) for t,(p,f,g) in active.items()],done=done,skipped_unfinished=skipped,failed=failed,time=time.time()))
        count=sum(len(read(p)) for p in ART.glob('*/results.json'))
        if count!=last:
            result=subprocess.run([sys.executable,'-m','continual.memory_refinement.report'])
            if result.returncode==0:last=count
        if active:time.sleep(20)
    if failed:raise RuntimeError(failed)
    subprocess.run([sys.executable,'-m','continual.memory_refinement.report'],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--fraction',type=int,choices=[50,25,10,5]);a=p.parse_args()
    if a.fraction:worker(a.fraction)
    else:launch()
