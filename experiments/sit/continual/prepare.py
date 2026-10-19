"""Download official pretrained assets and prepare disjoint Imagenette style domains."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.request

import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import torch
from diffusers import AutoencoderKL
from torchvision import transforms
from torchvision.utils import save_image

CLASSES = {'n01440764': 0, 'n02102040': 217, 'n02979186': 482,
           'n03000684': 491, 'n03028079': 497, 'n03394916': 566,
           'n03417042': 569, 'n03425413': 571, 'n03445777': 574,
           'n03888257': 701}
DOMAINS = ['photo', 'monochrome', 'poster', 'sketch']
MODEL_URL = 'https://www.dl.dropboxusercontent.com/scl/fi/as9oeomcbub47de5g4be0/SiT-XL-2-256.pt?rlkey=uxzxmpicu46coq3msb17b9ofa&dl=0'
DATA_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'


def download(url, dest):
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + '.partial')
    print('Downloading', url, flush=True)
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def styled(im, domain):
    im = ImageOps.fit(im.convert('RGB'), (256, 256), method=Image.Resampling.LANCZOS)
    if domain == 'monochrome':
        im = ImageOps.grayscale(im).convert('RGB')
    elif domain == 'poster':
        im = ImageOps.posterize(ImageEnhance.Color(im).enhance(1.5), 2)
    elif domain == 'sketch':
        im = ImageOps.grayscale(im).filter(ImageFilter.GaussianBlur(1.0))
        im = ImageOps.invert(im.filter(ImageFilter.FIND_EDGES))
        # Remove FIND_EDGES' one-pixel boundary artifact before resizing.
        im = ImageOps.autocontrast(im.crop((2, 2, 254, 254))).resize((256, 256)).convert('RGB')
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='/home/dev/sit_state')
    p.add_argument('--assets-only', action='store_true')
    args = p.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    download(MODEL_URL, root / 'SiT-XL-2-256.pt')
    download(DATA_URL, root / 'imagenette2-320.tgz')
    if not (root / 'imagenette2-320').exists():
        with tarfile.open(root / 'imagenette2-320.tgz') as archive:
            archive.extractall(root, filter='data')
    vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse')
    vae.save_pretrained(root / 'vae')
    if args.assets_only:
        return
    torch.set_num_threads(4)
    vae = vae.cuda().eval().requires_grad_(False)
    rng = np.random.default_rng(20260929)
    splits = {'train': [], 'support': [], 'test': []}
    for wnid, label in CLASSES.items():
        train = sorted((root / 'imagenette2-320' / 'train' / wnid).glob('*.JPEG'))
        val = sorted((root / 'imagenette2-320' / 'val' / wnid).glob('*.JPEG'))
        rng.shuffle(train)
        rng.shuffle(val)
        splits['train'].extend((str(f), label) for f in train[:256])
        splits['support'].extend((str(f), label) for f in val[:8])
        splits['test'].extend((str(f), label) for f in val[8:48])
    manifest = {'classes': CLASSES, 'domains': DOMAINS, 'splits': splits,
                'data_url': DATA_URL, 'model_url': MODEL_URL,
                'latent': 'posterior mode, sd-vae-ft-mse, scale 0.18215',
                'model_sha256': hashlib.sha256((root / 'SiT-XL-2-256.pt').read_bytes()).hexdigest()}
    (root / 'data_manifest.json').write_text(json.dumps(manifest, indent=2))
    to_tensor = transforms.ToTensor()
    demos = []
    for domain in DOMAINS:
        for split, rows in splits.items():
            out = root / f'{domain}_{split}.pt'
            if out.exists():
                print('Already prepared', out, flush=True)
                continue
            latents, labels = [], []
            for start in range(0, len(rows), 32):
                batch_rows = rows[start:start + 32]
                ims = [styled(Image.open(f), domain) for f, _ in batch_rows]
                if split == 'test':
                    image_dir = root / 'real' / domain
                    image_dir.mkdir(parents=True, exist_ok=True)
                    for i, im in enumerate(ims):
                        im.save(image_dir / f'{start+i:05d}.png')
                x = torch.stack([to_tensor(im) * 2 - 1 for im in ims]).cuda()
                with torch.no_grad():
                    z = vae.encode(x).latent_dist.mode() * 0.18215
                latents.append(z.cpu().half())
                labels.extend(y for _, y in batch_rows)
            torch.save({'z': torch.cat(latents), 'y': torch.tensor(labels)}, out)
            print(domain, split, len(labels), flush=True)
        demos.extend(to_tensor(styled(Image.open(splits['test'][i*40][0]), domain)) for i in range(10))
    save_image(torch.stack(demos), root / 'domain_examples.png', nrow=10)


if __name__ == '__main__':
    main()
