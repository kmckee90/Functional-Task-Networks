"""Mechanism checks on pod; intentionally independent of the full experiment."""
import json
import tempfile
from pathlib import Path
import numpy as np
import torch
from . import common as C
from .selection import facility_order
from .replay import CoverageReplay,pool_entries

class Toy:
    def __init__(self):
        self.m=torch.zeros(1024,device='cuda');self.m[:256]=1
        self.w=torch.nn.Parameter(torch.tensor(.3,device='cuda'))
    def __call__(self,x,m=None):return self.w*x[:,:-2].reshape(-1,4,32,32)
    def loss(self,p,y):return (p-y).square().mean()

def main():
    torch.set_num_threads(4)
    gen=torch.Generator().manual_seed(73)
    f={b:torch.randn(80,24,generator=gen) for b in ['dino','style','face']}
    y=torch.arange(80)%10
    a,ra=facility_order(f,y,20);b,rb=facility_order(f,y,40)
    assert a==b[:20] and len(set(b))==40
    assert set(ra['class_counts'].values())=={2}
    for name in C.NAMES:
        try:C.features(name,'train','dino')
        except AssertionError:pass
        else:raise AssertionError('Historical data guard failed')
    m=Toy();mask=m.m.cpu().clone();eligible=mask.clone();eligible[128:256]=0;eligible[256:384]=1
    memory={'monochrome':dict(ids=[0,1,2,3],z=torch.randn(4,4,32,32,generator=gen),y=torch.ones(4,dtype=torch.long),
             masks=torch.stack([mask,eligible]),entries=[dict(mask=0,indices=[0,1,2,3]),dict(mask=1,indices=[0,1,2,3])])}
    replay=CoverageReplay(91)
    pool=pool_entries(list(memory.values()))
    r=replay.backward(m,pool)
    assert r['overlap']==.5 and m.w.grad is not None
    assert CoverageReplay(91).backward(m,pool[:1]) is None
    renamed={'arbitrary label':memory['monochrome']}
    r2=CoverageReplay(91).backward(m,pool_entries(list(renamed.values())))
    assert r==r2, 'Task labels changed replay'
    # Two eligible masks and reversed storage order: same anonymous selection.
    second=eligible.clone();second[100:128]=0;second[400:428]=1
    memory['another']=dict(z=memory['monochrome']['z'],y=memory['monochrome']['y'],masks=second[None],entries=[dict(mask=0,indices=[0,1,2,3])])
    pool=pool_entries(list(memory.values()))
    assert CoverageReplay(8).backward(m,pool)==CoverageReplay(8).backward(m,list(reversed(pool)))
    for shared,allowed in [(51,False),(52,True),(217,True),(218,False)]:
        gate_mask=torch.zeros(1024);gate_mask[:shared]=1;gate_mask[256:512-shared]=1
        e=dict(z=memory['monochrome']['z'],y=memory['monochrome']['y'],masks=gate_mask[None],entries=[dict(mask=0,indices=[0,1,2,3])])
        assert (CoverageReplay(12).backward(m,pool_entries([e])) is not None)==allowed
    state=replay.state();other=CoverageReplay(0);other.restore(state)
    assert replay.rng.random()==other.rng.random()
    assert torch.equal(replay.generator.get_state(),other.generator.get_state())
    # Exercise four-retained-image ridge fitting; no full-data source can be read.
    from .evaluate import fit
    C.MEMORY['monochrome']=dict(ids=[0,1,2,3],features={b:C.base.features('monochrome','support',b)[:4] for b in ['dino','style','face']})
    teachers={'monochrome':dict(mask=mask.tolist(),validation_indices=[3])}
    with tempfile.TemporaryDirectory(dir=C.ROOT) as d:
        readout,prior=fit(teachers,Path(d))
        support=C.unified('monochrome','train').cuda()[None]
        assert torch.equal(readout.mask(support,256)[0].cpu(),mask)
    C.MEMORY.clear()
    C.atomic(C.ART/'preflight.json',dict(passed=True,checks=['nested facility subsets','balanced prefixes','historical full-data guard','task-label and storage-order invariant replay','100-percent overlap excluded','50-percent overlap eligible','replay RNG resume','four-retained-image conditioner']))
    print('PREFLIGHT_PASSED',flush=True)

if __name__=='__main__':main()
