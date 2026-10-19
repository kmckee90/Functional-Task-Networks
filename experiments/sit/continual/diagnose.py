"""Post-hoc inference diagnostics on frozen final adapters, never weight tuning."""
import argparse
import json
from pathlib import Path
import time

import torch

from .model import BayesSiT, InputPrior, transport_batch
from .run import balanced_indices, query_loss, atomic_json
from .prepare import DOMAINS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--root', default='/home/dev/sit_state')
    args = p.parse_args()
    torch.set_num_threads(4)
    root = Path(args.root)
    run = root/'runs'/args.name
    cfg = json.loads((run/'config.json').read_text())
    model = BayesSiT(root/'SiT-XL-2-256.pt', cfg['H'], cfg['k'], cfg['rank'], cfg['alpha']).cuda().eval()
    ck = torch.load(run/'latest.pt', map_location='cuda', weights_only=False)
    model.load_adapter(ck['adapter'])
    router = InputPrior()
    router.entries = ck['router']
    saved = ck['saved_masks']
    output = {}
    for domain in DOMAINS:
        support = {k: v.cuda() for k,v in torch.load(root/f'{domain}_support.pt', weights_only=True).items()}
        test = {k: v.cuda() for k,v in torch.load(root/f'{domain}_test.pt', weights_only=True).items()}
        si = balanced_indices(support['y'], 80)
        qi = balanced_indices(test['y'], 200)
        sx, sy = transport_batch(support['z'][si], support['y'][si], torch.Generator(device='cuda').manual_seed(189))
        qx, qy = transport_batch(test['z'][qi], test['y'][qi], torch.Generator(device='cuda').manual_seed(901))
        candidates, metadata = {}, {}
        for budget, steps in [(4,30), (40,30), (80,30), (40,120)]:
            name = f'cold_{budget}_{steps}'
            torch.manual_seed(7801 + budget + DOMAINS.index(domain))
            tic = time.time()
            model.cold_reconfigure(sx[:budget], sy[:budget], steps=steps, take_best=False)
            candidates[name] = model.m.clone()
            metadata[name] = {'seconds': time.time()-tic}
        for refine in [False, True]:
            name = 'ic_refine4' if refine else 'ic_direct4'
            metadata[name] = router.infer(model, support['z'][si[:4]], sx[:4], sy[:4], refine)
            candidates[name] = model.m.clone()
        # Stored-mask retrieval is input/support driven: no domain ID selection.
        for budget in [4,40]:
            with torch.no_grad():
                losses = {d: model.loss(model(sx[:budget], m=m), sy[:budget]).item() for d,m in saved.items()}
            chosen = min(losses, key=losses.get)
            name = f'retrieve_stored{budget}'
            candidates[name] = saved[chosen]
            metadata[name] = {'chosen_domain_for_diagnostic_only': chosen, 'support_losses': losses}
        # Random-search comparator gets same 30 support evaluations, no gradients.
        torch.manual_seed(971)
        with torch.no_grad():
            random_masks = [model._topk_mask(torch.randn_like(model.m)) for _ in range(30)]
            losses = [model.loss(model(sx[:40], m=m), sy[:40]).item() for m in random_masks]
        candidates['random_search30'] = random_masks[min(range(30), key=losses.__getitem__)]
        if domain in saved:
            candidates['oracle_stored_domain_id'] = saved[domain]
        output[domain] = {}
        for name, mask in candidates.items():
            result = query_loss(model, qx, qy, mask)
            result['metadata'] = metadata.get(name, {})
            result['mask'] = mask.cpu().tolist()
            output[domain][name] = result
        atomic_json(run/'diagnostics.json', output)
        print(json.dumps({'domain': domain, 'losses': {n: r['loss'] for n,r in output[domain].items()}}), flush=True)


if __name__ == '__main__':
    main()
