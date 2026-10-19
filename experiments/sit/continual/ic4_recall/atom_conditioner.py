"""Sequential atom-wise regression on a frozen shared nonlinear feature map.

This diagnostic removes complete-mask nearest-prototype lookup. Each atom has
its own scalar linear readout of fixed image features and fixed ReLU features.
Running second-order sufficient statistics preserve the full earlier training
objective without retaining old image episodes. This does NOT guarantee old
test predictions are unchanged; insertion-stage recall is measured explicitly.
Offline repaired masks remain supervision, so this is not four-image discovery.
"""
import argparse
import json

import torch
from torch import nn

from .common import ROOT, ART, SUITES, atomic
from .fixed_conditioner import fixed_features


class AtomReadout(nn.Module):
    def __init__(self, projection, weight):
        super().__init__()
        self.register_buffer('projection', projection)
        self.register_buffer('weight', weight)

    def features(self, x):
        x=x.mean(-2)
        nonlinear=torch.relu(x@self.projection)/self.projection.shape[1]**.5
        return torch.cat([x, nonlinear, torch.ones_like(x[...,:1])],-1)

    def forward(self,x):
        return self.features(x)@self.weight

    def mask(self,x,k):
        scores=self(x)
        return torch.zeros_like(scores).scatter_(-1,scores.topk(k,dim=-1).indices,1.)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--suite',choices=SUITES,required=True)
    p.add_argument('--arm',choices=['noreplay','replay'],required=True)
    p.add_argument('--seed',type=int,default=123)
    p.add_argument('--variant',default='fixed_atoms',choices=['fixed_atoms'])
    a=p.parse_args();torch.set_num_threads(4)
    cfg=SUITES[a.suite];out=ART/a.suite/f'{a.arm}_s{a.seed}'
    records=json.loads((out/'search/candidates.json').read_text())
    teachers={n:min((r for r in records if r['domain']==n),key=lambda r:r['selection_loss']) for n in cfg['names']}
    train=fixed_features(a.suite,'train','fixed_multiview')
    support=fixed_features(a.suite,'support','fixed_multiview')
    dim=next(iter(train.values())).shape[1]
    gen=torch.Generator().manual_seed(93811)
    projection=torch.randn(dim,128,generator=gen,device='cpu').cuda()
    model=AtomReadout(projection,torch.zeros(dim+129,cfg['H'],device='cuda'))
    gram=torch.eye(dim+129,device='cuda',dtype=torch.float64)*.01
    rhs=torch.zeros(dim+129,cfg['H'],device='cuda',dtype=torch.float64)
    stages=[];saved_designs=[];saved_targets=[]
    for n in cfg['names']:
        val=set(teachers[n]['validation_indices'])
        fit=train[n][[i for i in range(len(train[n])) if i not in val]]
        # Four distinct training images per episode; no support/test fitting.
        indices=torch.stack([torch.randperm(len(fit),generator=gen)[:4] for _ in range(512)])
        design=model.features(fit[indices].cuda()).double()
        target=(torch.tensor(teachers[n]['mask'],device='cuda')*2-1).double().expand(len(design),-1)
        gram+=design.T@design;rhs+=design.T@target
        model.weight.copy_(torch.linalg.solve(gram,rhs).float())
        # Retained only within this audit for verifying the sufficient-statistic
        # solution against the equivalent joint objective, never for prediction.
        saved_designs.append(design);saved_targets.append(target)
        joint_x=torch.cat(saved_designs);joint_y=torch.cat(saved_targets)
        assert torch.allclose(gram,joint_x.T@joint_x+torch.eye(dim+129,device='cuda',dtype=torch.float64)*.01,
                              atol=1e-8,rtol=1e-8)
        assert torch.allclose(rhs,joint_x.T@joint_y,atol=1e-8,rtol=1e-8)
        residual=(gram@model.weight.double()-rhs).norm()/rhs.norm().clamp_min(1e-12)
        assert float(residual)<1e-5
        recall={}
        for domain in cfg['names'][:len(stages)+1]:
            target_mask=torch.tensor(teachers[domain]['mask'],device='cuda')
            ids=torch.stack([torch.randperm(len(support[domain]),generator=torch.Generator().manual_seed(15001+d))[:4] for d in range(100)])
            masks=model.mask(support[domain][ids].cuda(),cfg['k'])
            recall[domain]=dict(exact_teacher_rate=float((masks==target_mask).all(1).float().mean()),
                mean_overlap=float((masks*target_mask).sum(1).mean()/cfg['k']))
        stages.append(dict(after=n,recall=recall,normal_equation_residual=float(residual),
            joint_training_mse=float((joint_x@model.weight.double()-joint_y).square().mean())))
        print('ATOM_INSERTION',a.suite,a.arm,a.seed,stages[-1],flush=True)
    path=ROOT/'conditioners_fixed_atoms'/a.suite/f'{a.arm}_s{a.seed}.pt'
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(projection=projection.cpu(),weight=model.weight.cpu(),gram=gram.cpu(),rhs=rhs.cpu(),teachers=teachers),path)
    atomic(out/'conditioner_fixed_atoms_fit.json',dict(stages=stages,encoder_frozen=True,
        complete_mask_lookup=False,regularization=.01,episodes_per_task=512,feature_seed=93811,
        prediction_parameters=model.weight.numel(),sufficient_statistics_bytes=(gram.numel()+rhs.numel())*8,
        scope='Sequential atom regression with offline repaired-mask supervision; no full generator lifecycle or unseen-mask evidence'))


def load(suite,arm,seed):
    ck=torch.load(ROOT/'conditioners_fixed_atoms'/suite/f'{arm}_s{seed}.pt',weights_only=False)
    return AtomReadout(ck['projection'],ck['weight']).cuda().eval(),ck


if __name__=='__main__':main()
