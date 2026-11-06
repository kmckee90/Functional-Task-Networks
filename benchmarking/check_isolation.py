"""Assert gradient, optimizer-state, and function isolation for FTN banks."""
import argparse
import copy

import torch

from ftn_models import VARIANTS,build,load_family
from isolation import IsolationAdam,atom_axes
from storage import bulk,write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--vit',action='store_true')
    a=p.parse_args()
    torch.set_default_device('cuda')
    torch.set_num_threads(4)
    torch.manual_seed(1907)
    cfg,runner=load_family(a.variant)
    cfg.SQUARE_DIM,cfg.K=4,4
    cfg.LAYERS,cfg.INNER_DIM=2,4
    cfg.LORA_LAYERS,cfg.LORA_WIDTH=2,16
    cfg.RECONFIG_STEPS=3
    cfg.BAYES_RECONFIG_STEPS=2
    cfg.BAYES_CHANGEPOINT_RECONFIG_STEPS=3
    model=build(cfg,4 if not a.vit else 1,3,vit=a.vit,pretrained=False)
    optimizer=IsolationAdam(model,lr=1e-3)
    x=torch.randn(4,3,224,224) if a.vit else torch.randn(8,4)
    y=torch.nn.functional.one_hot(torch.arange(len(x))%3,3).float()
    mask_a=torch.zeros_like(model.m); mask_a[:cfg.K]=1
    mask_b=torch.zeros_like(model.m); mask_b[cfg.K:2*cfg.K]=1
    axes=atom_axes(model)
    def update(masks):
        optimizer.zero_grad(set_to_none=True)
        for mask in masks:
            loss=model.loss(model(x,m=mask),y)
            loss.backward()
        union=torch.stack(masks).bool().any(0)
        for param,(name,axis) in axes.items():
            if param.grad is not None:
                inactive=param.grad.movedim(axis,0)[~union]
                assert torch.count_nonzero(inactive)==0, f'Inactive gradient leaked into {name}'
        optimizer.step()
    # Warm Adam moments on assembly A, then stop replaying it while continuing B.
    # Even with no intervening reconfiguration, stale A moments must not move A.
    model.m.copy_(mask_a)
    update([mask_a]); update([mask_a,mask_b])
    model.eval()
    with torch.no_grad():
        predictions_a=model(x,m=mask_a).clone()
    model.train()
    before={p:p.detach().clone() for p in axes}
    update([mask_b])
    for param,(name,axis) in axes.items():
        torch.testing.assert_close(param.movedim(axis,0)[~mask_b.bool()],
            before[param].movedim(axis,0)[~mask_b.bool()],rtol=0,atol=0,msg=f'Inactive optimizer drift in {name}')
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(x,m=mask_a),predictions_a,rtol=0,atol=0)
    model.train()
    # A real native mask search must reset weight Adam, leave weights intact,
    # and leave no weight gradients. Bayesian posterior state is not discarded.
    optimizer.zero_grad(set_to_none=True)
    before={p:p.detach().clone() for p in axes}
    assert len(optimizer.state)>0
    count=optimizer.reconfiguration_resets
    model._reconfig(x,y)
    assert len(optimizer.state)==0
    assert optimizer.reconfiguration_resets==count+1
    assert all(p.grad is None for p in model.parameters())
    for param in axes:
        torch.testing.assert_close(param,before[param],rtol=0,atol=0)
    # Evaluation searches must preserve training optimizer state.
    update([model.m.clone()])
    state_before=copy.deepcopy(optimizer.state_dict())
    model.eval()
    model._reconfig(x,y)
    state_after=optimizer.state_dict()
    for key,state in state_before['state'].items():
        for name,value in state.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(value,state_after['state'][key][name],rtol=0,atol=0)
    assert all(group['weight_decay']==0 for group in optimizer.param_groups)
    result={'variant':a.variant,'vit':a.vit,'status':'pass',
            'assertions':['no shared trainable upstream parameters','zero inactive weight gradients',
              'inactive parameter values bitwise unchanged after warm Adam and replay',
              'disjoint assembly function bitwise unchanged','real reconfiguration clears Adam moments and counters',
              'native mask search leaves all weight gradients None','evaluation preserves training optimizer state'],
            'policy':optimizer.policy()}
    write_json(bulk('validation','isolation',a.variant+('_vit' if a.vit else '')+'.json'),result)
    print('PASS isolation:',a.variant,'ViT' if a.vit else 'MLP',flush=True)


if __name__=='__main__': main()
