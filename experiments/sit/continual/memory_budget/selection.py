"""Nested image coverage first; quality-constrained trajectory-mask coverage second."""
import numpy as np
import torch
from ..model import transport_batch
from .common import FILTERS

@torch.no_grad()
def facility_order(features,labels,count):
    """Greedy facility-location, with balanced class prefixes and deterministic ties."""
    x=torch.cat([torch.nn.functional.normalize(features[b],dim=1) for b in ['dino','style','face']],1).cuda()/(3**.5)
    sim=((x@x.T)+1).clamp(0,2)/2
    coverage=torch.zeros(len(x),device='cuda');selected=torch.zeros(len(x),dtype=torch.bool,device='cuda')
    labels=labels.to('cuda');classes=labels.unique(sorted=True);counts={int(c):0 for c in classes}
    order=[]
    for _ in range(count):
        gains=(sim-coverage[:,None]).clamp_min(0).mean(0)
        # At each prefix, counts across the original classes differ by at most one.
        minimum=min(counts.values());eligible=torch.zeros_like(selected)
        for c in classes:
            if counts[int(c)]==minimum:eligible|=labels==c
        gains[~eligible|selected]=-torch.inf
        idx=int(gains.argmax());assert torch.isfinite(gains[idx])
        order.append(idx);counts[int(labels[idx])]+=1;selected[idx]=True
        coverage=torch.maximum(coverage,sim[:,idx])
    # Diagnostic only: same-budget, same-class-count random subset coverage.
    gen=torch.Generator().manual_seed(85109);random_ids=[]
    for c in classes:
        members=torch.where(labels.cpu()==int(c))[0]
        random_ids.extend(members[torch.randperm(len(members),generator=gen)[:counts[int(c)]]].tolist())
    random_coverage=sim[:,random_ids].max(1).values
    return order,dict(mean_similarity=float(coverage.mean()),worst_similarity=float(coverage.min()),
                      random_mean_similarity=float(random_coverage.mean()),
                      random_worst_similarity=float(random_coverage.min()),class_counts=counts)

@torch.no_grad()
def associate(model,train,ids,candidates,previous):
    """Only retained images are scored; masks come from the training trajectory, not a task library."""
    masks=[]
    for mask in candidates:
        mask=mask.detach().cpu().float()
        if not any(torch.equal(mask,m) for m in masks):masks.append(mask)
    masks=torch.stack(masks)
    z,y=train['z'][ids],train['y'][ids]
    losses=torch.zeros(len(masks),len(ids))
    for repeat in range(2):
        gen=torch.Generator(device='cuda').manual_seed(70091+repeat)
        for start in range(0,len(ids),32):
            x,target=transport_batch(z[start:start+32],y[start:start+32],gen)
            for j,mask in enumerate(masks):
                pred=model(x,m=mask.cuda())
                losses[j,start:start+len(x)]+=(pred-target).square().flatten(1).mean(1).cpu()/2
    quality=losses.mean(1)
    good=quality<=quality.min()*1.25
    losses[~good]=torch.inf
    best=losses.argmin(0)
    usage=torch.zeros(1024)
    previous_atoms=torch.zeros(1024)
    for e in previous.values():previous_atoms+=e['masks'].sum(0)
    atom_weight=1+(previous_atoms>0).float()
    # One best association, plus one near-best association if it covers different atoms.
    associations=[]
    for i,j in enumerate(best.tolist()):
        associations.append((i,j));usage+=masks[j]
    for i in range(len(ids)):
        eligible=(losses[:,i]<=losses[best[i],i]*1.10)&good
        eligible[best[i]]=False
        if eligible.any():
            gains=(masks*atom_weight/(1+usage)).sum(1);gains[~eligible]=-torch.inf
            j=int(gains.argmax());associations.append((i,j));usage+=masks[j]
    used=sorted(set(j for i,j in associations));mapping={j:k for k,j in enumerate(used)}
    entries=[dict(mask=mapping[j],indices=[i for i,k in associations if k==j],baseline=float(losses[j,[i for i,k in associations if k==j]].mean())) for j in used]
    return masks[used],entries,dict(candidate_masks=len(masks),quality_eligible=int(good.sum()),
              retained_masks=len(used),associations=len(associations),best_velocity_mse=float(quality.min()))

def retain(model,n,train,features,fraction,candidates,previous):
    total=len(train['z']);count=total*fraction//100
    assert count>=4
    ids,coverage=facility_order(features,train['y'],count)
    masks,entries,mask_stats=associate(model,train,ids,candidates,previous)
    entry=dict(ids=ids,z=train['z'][ids].detach().cpu().clone(),y=train['y'][ids].cpu().clone(),
               features={b:x[ids].clone() for b,x in features.items()},masks=masks,entries=entries,
               source_count=total,fraction=fraction)
    report=dict(task=n,source_count=total,retained=count,ids=ids,coverage=coverage,mask_coverage=mask_stats)
    return entry,report
