"""Pretrained integrity and runtime measurement on one pod GPU."""
import json
import time
from pathlib import Path
import torch
from .model import BayesSiT, transport_batch
from models import SiT_models


def main():
    torch.set_num_threads(4)
    torch.manual_seed(123)
    root = Path('/home/dev/sit_state')
    model = BayesSiT(root / 'SiT-XL-2-256.pt').cuda().eval()
    d = torch.load(root / 'monochrome_train.pt', weights_only=True)
    x, target = transport_batch(d['z'][:16].cuda(), d['y'][:16].cuda())
    original = SiT_models['SiT-XL/2'](input_size=32, num_classes=1000, learn_sigma=True)
    original.load_state_dict(torch.load(root/'SiT-XL-2-256.pt', map_location='cpu', weights_only=False))
    original = original.cuda().eval().requires_grad_(False)
    with torch.no_grad():
        base = model(x)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            reference = original(x[:, :-2].reshape(-1, 4, 32, 32), x[:, -2], x[:, -1].long()).float()
        torch.testing.assert_close(base, reference, atol=0, rtol=0)
        model.m.copy_(model._topk_mask(torch.randn_like(model.m)))
        initial = model(x)
    del original
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-4)
    timings = []
    for i in range(4):
        torch.cuda.synchronize()
        start = time.time()
        model.reconfigure(x, target)
        opt.zero_grad(set_to_none=True)
        loss = model.loss(model(x), target)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        timings.append(time.time()-start)
    result = {'base_loss': model.loss(base, target).item(),
              'initial_mask_loss': model.loss(initial, target).item(),
              'initial_delta_rms': (initial-base).square().mean().sqrt().item(),
              'steps_seconds': timings, 'peak_GB': torch.cuda.max_memory_allocated()/1e9,
              'trainable': sum(p.numel() for p in params), 'last_loss': loss.item()}
    print(json.dumps(result), flush=True)
    (root/'benchmark.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
