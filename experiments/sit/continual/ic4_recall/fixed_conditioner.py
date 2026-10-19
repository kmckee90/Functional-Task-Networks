"""Fixed representations with immutable, append-only atom-prior associations.

No global router is trained or re-fitted when another concept arrives. Each
entry stores a compressed feature mean and an H-dimensional atom prior, never
raw images. Four-image features yield a kernel-weighted atom prior followed by
top-k over the original shared FTN bank. There is no generator-loss enumeration
of complete adapters. This is learned-domain recall, not unseen-task discovery.
"""
import argparse
import hashlib
import json

import torch
from torch import nn

from .common import ROOT,ART,SUITES,atomic


def fixed_features(suite,split,variant='fixed'):
    cfg=SUITES[suite];cache=ROOT/'features'/suite
    if variant in ('fixed_unified','fixed_multiview'):
        # Identical transform for every suite; no task-dependent scaling or PCA.
        # Equal unit-norm branches preserve appearance alongside semantics.
        branches=('dino','style','face') if variant=='fixed_multiview' else ('dino','style')
        return {n:torch.cat([torch.nn.functional.normalize(torch.load(
            cache/f'{n}_{split}_{branch}.pt',weights_only=True),dim=1)
            for branch in branches],dim=1)/len(branches)**.5 for n in cfg['names']}
    branch='face' if variant=='fixed_face' else 'style' if suite=='filters' else 'dino'
    # Feature transform and distance do not depend on the encountered classes.
    # Semantic embeddings are already unit-normalized by the frozen encoder.
    return {n:torch.load(cache/f'{n}_{split}_{branch}.pt',weights_only=True) for n in cfg['names']}


class FixedAtomPrior(nn.Module):
    def __init__(self,keys,values,temperature=.001):
        super().__init__()
        self.register_buffer('keys',keys)
        self.register_buffer('values',values)
        self.temperature=temperature

    def forward(self,features):
        means=features.mean(-2)
        distances=torch.cdist(means,self.keys).square()
        weights=torch.softmax(-distances/self.temperature,dim=-1)
        return weights@self.values

    def mask(self,features,k):
        logits=self(features)
        return torch.zeros_like(logits).scatter_(-1,logits.topk(k,dim=-1).indices,1.)


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=SUITES,required=True)
    p.add_argument('--arm',choices=['replay','noreplay'],required=True);p.add_argument('--seed',type=int,default=123)
    p.add_argument('--variant',choices=['fixed','fixed_face','fixed_unified','fixed_multiview'],default='fixed')
    a=p.parse_args();torch.set_num_threads(4)
    cfg=SUITES[a.suite];out=ART/a.suite/f'{a.arm}_s{a.seed}'
    candidates=json.loads((out/'search'/'candidates.json').read_text())
    teachers={n:min((r for r in candidates if r['domain']==n),key=lambda r:r['selection_loss']) for n in cfg['names']}
    train=fixed_features(a.suite,'train',a.variant);support=fixed_features(a.suite,'support',a.variant)
    keys=[];values=[];stages=[];hashes=[]
    for index,n in enumerate(cfg['names']):
        # Fit the association key on training-fit images only; internal validation
        # images selected the mask value but do not fit recognition keys.
        vi=set(teachers[n]['validation_indices'])
        ids=torch.tensor([i for i in range(len(train[n])) if i not in vi])
        keys.append(train[n][ids].mean(0))
        values.append(torch.tensor(teachers[n]['mask'])*8-4)
        serialized=[hashlib.sha256(torch.cat([k,v]).numpy().tobytes()).hexdigest() for k,v in zip(keys,values)]
        assert serialized[:-1]==hashes
        hashes=serialized
        model=FixedAtomPrior(torch.stack(keys),torch.stack(values))
        row={}
        for domain in cfg['names'][:index+1]:
            target=torch.tensor(teachers[domain]['mask'])
            exact=[];overlaps=[]
            for draw in range(100):
                si=torch.randperm(len(support[domain]),generator=torch.Generator().manual_seed(15001+draw))[:4]
                mask=model.mask(support[domain][si][None],cfg['k'])[0]
                exact.append(torch.equal(mask,target));overlaps.append(float((mask*target).sum()/cfg['k']))
            row[domain]=dict(exact_teacher_rate=sum(exact)/len(exact),mean_overlap=sum(overlaps)/len(overlaps))
        stages.append(dict(after=n,associations=len(keys),key_value_hashes=list(hashes),recall=row))
    ck=dict(keys=torch.stack(keys),values=torch.stack(values),temperature=.001,k=cfg['k'],teachers=teachers,
            suite=a.suite,arm=a.arm,seed=a.seed,variant=a.variant,entry_hashes=hashes)
    path=ROOT/('conditioners_'+a.variant)/a.suite/f'{a.arm}_s{a.seed}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(ck,path)
    atomic(out/f'conditioner_{a.variant}_fit.json',dict(stages=stages,parameters=0,temperature=.001,
        prior_entries_immutable=True,encoder_frozen=True,feature_transform_independent_of_task_sequence=True,
        recognition_fit_split='training-fit only',support_results_are_diagnostic_not_selection=True,
        scope='Sequential association insertion on frozen final FTN bank; not yet full generator lifecycle evaluation',
        teachers={n:teachers[n]['kind'] for n in cfg['names']}))
    print('FIXED_ASSOCIATION',a.suite,a.arm,a.seed,stages[-1]['recall'],flush=True)


def load(suite,arm,seed,variant='fixed'):
    ck=torch.load(ROOT/('conditioners_'+variant)/suite/f'{arm}_s{seed}.pt',weights_only=False,map_location='cpu')
    return FixedAtomPrior(ck['keys'],ck['values'],ck['temperature']).cuda().eval(),ck


if __name__=='__main__':main()
