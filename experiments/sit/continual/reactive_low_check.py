"""Pod-only checks of exact stable-batch and change-point preset semantics."""
import torch
from .model import BayesSiT,transport_batch
from .bayesian_adv_config_snapshot import BayesLoRAReactiveLow


def main():
    torch.set_num_threads(2); torch.manual_seed(12)
    model=BayesSiT(None,H=8,k=2,model_name='SiT-S/2',amp=False,reactive_low=True).cuda()
    for key,value in BayesLoRAReactiveLow().BAYES_FTN_PARS.items():
        assert getattr(model,key)==value,(key,getattr(model,key),value)
    assert model.bayes_reconfig_steps==1 and model.bayes_reconfig_lr==0
    assert not model.bayes_stochastic_mask
    assert model.bayes_changepoint_reconfig_steps==30
    assert model.bayes_changepoint_stochastic_mask
    assert model.bayes_changepoint_prior_decay==0 and model.bayes_changepoint_kl_weight==0
    with torch.no_grad():
        model.backbone.final_layer.linear.weight.normal_(0,.01)
        for block in model.backbone.blocks:
            block.adaLN_modulation[-1].weight.normal_(0,.01)
    z=torch.randn(2,4,32,32,device='cuda')
    x,y=transport_batch(z,torch.zeros(2,device='cuda',dtype=torch.long))
    model.reconfigure_changepoint(x,y)
    logits=model.mask_logits.clone(); expected=model._topk_mask(logits)
    model.reconfigure(x,y)
    assert torch.equal(model.mask_logits,logits) and torch.equal(model.m,expected)
    x,y=transport_batch(z*2,torch.ones(2,device='cuda',dtype=torch.long))
    model.reconfigure(x,y)
    assert torch.equal(model.mask_logits,logits) and torch.equal(model.m,expected)
    assert all(p.grad is None for p in model.parameters())
    print('PASS: ReactiveLow stable posterior unchanged, deterministic mask, unchanged 30-step stochastic change-point search, no weight gradients')


if __name__=='__main__':
    main()
