"""Pod-only contract checks for replay repair and append-only routing."""
from types import SimpleNamespace
import torch
from .model import InputPrior


class ControlledModel:
    def __init__(self, improve):
        self.m=torch.tensor([1.,0.],device='cuda')
        self.mask_logits=self.m.clone()
        self.controller=SimpleNamespace(mask=self.m.clone())
        self.improve=improve
    def get_reconfig_state(self):
        return {'m':self.m.clone(),'logits':self.mask_logits.clone()}
    def set_reconfig_state(self,s):
        self.m.copy_(s['m']); self.mask_logits.copy_(s['logits'])
    def _topk_mask(self,logits):
        return torch.zeros_like(logits).scatter_(0,logits.topk(1).indices,1.)
    def __call__(self,x,m):
        return m[0]
    def loss(self,pred,target):
        return pred
    def cold_reconfigure(self,x,y,**kwargs):
        self.m.copy_(torch.tensor([0.,1.] if self.improve else [1.,0.],device='cuda'))


def main():
    for improve in [False,True]:
        router=InputPrior(rank=1)
        z=torch.zeros(8,4,32,32,device='cuda')
        z[:,0]=torch.arange(1,9,device='cuda')[:,None,None]
        router.archive(z,torch.tensor([4.,-4.],device='cuda'))
        old={k:v.clone() for k,v in router.entries[0].items()}
        model=ControlledModel(improve)
        cpu_rng=torch.get_rng_state().clone(); gpu_rng=torch.cuda.get_rng_state().clone()
        replay=[(z,torch.zeros(8,device='cuda',dtype=torch.long),model.m),
                (z,torch.zeros(8,device='cuda',dtype=torch.long),model.m)]
        result=router.repair_from_replay(model,replay,fit_rows=4,validation_rows=4,steps=1,restarts=1)
        assert result[0]['unique_rows']==8
        assert not set(result[0]['fit_hashes']) & set(result[0]['validation_hashes'])
        assert result[0]['accepted']==improve
        assert len(router.entries)==1+int(improve)
        for k,v in old.items():
            assert torch.equal(v,router.entries[0][k])
        assert router.retrieve(z[:4])[1]==int(improve)
        assert torch.equal(model.m,torch.tensor([1.,0.],device='cuda'))
        assert torch.equal(cpu_rng,torch.get_rng_state())
        assert torch.equal(gpu_rng,torch.cuda.get_rng_state())
    print('PASS: accept/reject, append-only projectors, latest retrieval, duplicate-safe split, state/RNG restoration')


if __name__=='__main__':
    main()
