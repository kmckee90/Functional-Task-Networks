"""Validate the ViT adaptation with native FTN mask-search methods."""
import argparse
import torch
from ftn_models import build, load_family

p = argparse.ArgumentParser()
p.add_argument('--variant', choices=['LoRA','BayesLoRA'], required=True)
p.add_argument('--pretrained', action='store_true')
a = p.parse_args()
torch.set_default_device('cuda')
torch.set_num_threads(4)
torch.manual_seed(812)
cfg, runner = load_family(a.variant)
cfg.SQUARE_DIM, cfg.K = 4, 4
cfg.LORA_WIDTH, cfg.LORA_LAYERS = 16, 1
cfg.RECONFIG_STEPS, cfg.RECONFIG_SAMPLE_SIZE = 2, 2
cfg.BAYES_RECONFIG_STEPS = 2
model = build(cfg, 1, 3, vit=True, pretrained=a.pretrained)
assert len(model.insertion_points) == 49
x = torch.randn(2,3,224,224)
y = torch.nn.functional.one_hot(torch.tensor([0,1]),3).float()
model._reconfig(x,y)
assert int(model.m.sum()) == cfg.K
assert all(p.grad is None for p in model.parameters())
pred = model(x)
assert pred.shape == (2,3)
model.loss(pred,y).backward()
assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in model.named_parameters() if '.A' in n or '.B' in n)
assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
print('PASS:',a.variant,'ViT-B/16, 49 insertion points, native search, frozen backbone, atom weight gradients',flush=True)
