"""Run on the pod: numerical and inference-contract checks before XL training."""
import json
import torch
from torch import nn
from .model import BankedLinear, Controller, BayesSiT, pack


def main():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    c = Controller(8, 2)
    layer = BankedLinear(nn.Linear(7, 9), c, rank=2).cuda()
    x = torch.randn(3, 5, 7, device='cuda')
    c.mask = torch.tensor([1., 1., 0, 0, 0, 0, 0, 0], device='cuda', requires_grad=True)
    out = layer(x)
    ref = layer.base(x) + 2 * (
        torch.einsum('bti,hir,h,hro->bto', x, layer.A, c.mask, layer.B)
        + torch.einsum('h,ho->o', c.mask, layer.bias_bank))
    torch.testing.assert_close(out, ref, atol=1e-6, rtol=1e-5)
    out.square().mean().backward()
    assert layer.A.grad[2:].abs().max() == 0
    assert layer.B.grad[2:].abs().max() == 0
    assert c.mask.grad[2:].abs().sum() > 0
    assert layer.base.weight.grad is None
    layer.zero_grad(set_to_none=True)
    c.mask = c.mask.detach().requires_grad_()
    c.detach_atoms = True
    layer(x).square().mean().backward()
    assert layer.A.grad is None and layer.B.grad is None
    assert c.mask.grad.abs().sum() > 0
    c.mask = torch.zeros(8, device='cuda')
    torch.testing.assert_close(layer(x), layer.base(x), atol=0, rtol=0)

    # Small SiT only for a fast API and backward compatibility check.
    model = BayesSiT(None, H=8, k=2, model_name='SiT-S/2', amp=False).cuda()
    # SiT constructor zeroes output, so randomize that head for useful gradients.
    with torch.no_grad():
        model.backbone.final_layer.linear.weight.normal_(0, .01)
        for block in model.backbone.blocks:
            block.adaLN_modulation[-1].weight.normal_(0, .01)
    z = torch.randn(2, 4, 32, 32, device='cuda')
    packed = pack(z, torch.ones(2, device='cuda') * .5, torch.zeros(2, device='cuda'))
    model.bayes_reconfig_steps = 2
    model.reconfigure(packed, z)
    assert model.m.sum() == 2 and torch.isfinite(model.mask_logits).all()
    old = model.get_reconfig_state()
    model.cold_reconfigure(packed, z, steps=2, take_best=False)
    torch.testing.assert_close(model.mask_logits, old['mask_logits'])
    model.set_reconfig_state(old)
    torch.testing.assert_close(model.m, old['m'])
    assert all(p.grad is None for p in model.parameters())
    print(json.dumps({'checks': 'passed', 'dense_gradient': True,
                      'inactive_atom_grad_zero': True, 'frozen_base': True,
                      'zero_mask_exact_base': True, 'cold_preserves_logits': True}))


if __name__ == '__main__':
    main()
