"""Generated-distribution check using CleanFID Inception features and KID.

KID is an unbiased finite-sample MMD estimate and can be slightly negative.
This is a small-sample diagnostic, not an ImageNet FID-50K benchmark.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image
from diffusers import AutoencoderKL
from cleanfid import fid

from .model import BayesSiT
from .run import generate, atomic_json
from .prepare import DOMAINS


def unbiased_kid(a, b):
    d = a.shape[1]
    aa = (a @ a.T / d + 1) ** 3
    bb = (b @ b.T / d + 1) ** 3
    ab = (a @ b.T / d + 1) ** 3
    return float((aa.sum()-np.trace(aa))/(len(a)*(len(a)-1)) +
                 (bb.sum()-np.trace(bb))/(len(b)*(len(b)-1)) - 2*ab.mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    p.add_argument('--count', type=int, default=200)
    p.add_argument('--all-stages', action='store_true')
    p.add_argument('--root', default='/home/dev/sit_state')
    args = p.parse_args()
    torch.set_num_threads(4)
    root = Path(args.root)
    run = root/'runs'/args.name
    cfg = json.loads((run/'config.json').read_text())
    history = json.loads((run/'lifecycle.json').read_text())
    torch.manual_seed(cfg['seed'])
    H, k, rank, alpha = cfg['H'], cfg['k'], cfg['rank'], cfg['alpha']
    if cfg['method'] == 'plain':
        H, k, rank, alpha = 1, 1, k*rank, alpha/k
    model = BayesSiT(root/'SiT-XL-2-256.pt', H, k, rank, alpha).cuda().eval()
    initial = model.adapter_state()
    vae = AutoencoderKL.from_pretrained(root/'vae').cuda().eval().requires_grad_(False)
    extractor = fid.build_feature_extractor('clean', device=torch.device('cuda'), use_dataparallel=False)
    real = {}
    for domain in DOMAINS:
        cached = root/f'inception_real_{domain}.npy'
        if cached.exists():
            real[domain] = np.load(cached)
        else:
            real[domain] = fid.get_folder_features(str(root/'real'/domain), model=extractor,
                num_workers=2, batch_size=32, device=torch.device('cuda'), mode='clean', verbose=False)
            # Unique temporary filename: multiple jobs may populate the same cache.
            temp = root/f'inception_real_{domain}_{args.name}.npy'
            np.save(temp, real[domain])
            temp.replace(cached)
    records = []
    if (run/'quality.json').exists():
        records = json.loads((run/'quality.json').read_text())
    completed = {(r['stage'], r['domain'], r['mode']) for r in records}
    stages = history if args.all_stages else history[-1:]
    for row in stages:
        stage = row['stage']
        if stage == 'initial':
            model.load_adapter(initial)
        else:
            ck = torch.load(run/f'{stage}.pt', map_location='cpu', weights_only=False)
            model.load_adapter(ck['adapter'])
        for domain in DOMAINS:
            modes = [f'cold{cfg["support"]}']
            if args.all_stages and stage == history[-1]['stage'] and 'stored' in row['domains'][domain]:
                modes.append('stored')
            if stage == 'initial' and domain == 'photo':
                modes.append('base')
            for mode in modes:
                if (stage, domain, mode) in completed:
                    continue
                mask = torch.tensor(row['domains'][domain][mode]['mask'], device='cuda')
                folder = run/'quality_samples'/stage/f'{domain}_{mode}'
                folder.mkdir(parents=True, exist_ok=True)
                if len(list(folder.glob('*.png'))) != args.count:
                    images = generate(model, vae, mask, count=args.count,
                                      steps=cfg['sample_steps'], seed=99233)
                    for i, im in enumerate(images):
                        save_image(im, folder/f'{i:05d}.png')
                feats = fid.get_folder_features(str(folder), model=extractor, num_workers=2,
                    batch_size=32, device=torch.device('cuda'), mode='clean', verbose=False).astype(np.float64)
                np.save(folder/'features.npy', feats)
                ref = real[domain].astype(np.float64)
                value = unbiased_kid(feats, ref)
                # Subsample variation is descriptive; it is not a run-level CI.
                rng = np.random.default_rng(420)
                subsets = [unbiased_kid(feats[rng.choice(len(feats), 100, replace=False)],
                                       ref[rng.choice(len(ref), 100, replace=False)]) for _ in range(20)]
                record = {'stage': stage, 'domain': domain, 'mode': mode, 'kid': value,
                          'subset_sd': float(np.std(subsets)), 'generated_count': len(feats),
                          'reference_count': len(ref), 'feature_model': 'CleanFID Inception clean'}
                records.append(record)
                atomic_json(run/'quality.json', records)
                print(json.dumps(record), flush=True)
    print('QUALITY_COMPLETE', args.name, flush=True)


if __name__ == '__main__':
    main()
