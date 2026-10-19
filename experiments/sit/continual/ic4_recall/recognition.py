"""Diagnose recognition without changing generator weights or consulting test data.

Features use original images, not evaluator scores or identity-name prompts.
All fitted keys use training images. Support episodes contain four unique images.
"""
import argparse
import os
import time

import torch
import torch.nn.functional as F
from torchvision import transforms

from ..model import InputPrior
from .common import ROOT, ART, SUITES, atomic, image_rows, load_data, run_path


def style_features(images):
    x = torch.stack([transforms.ToTensor()(im.resize((128, 128))) for im in images])
    gray = x.mean(1, keepdim=True)
    chroma = x - gray
    dx, dy = gray[..., 1:] - gray[..., :-1], gray[..., 1:, :] - gray[..., :-1, :]
    blocks = [x.flatten(2), chroma.abs().flatten(2), dx.abs().flatten(2), dy.abs().flatten(2)]
    quantiles = torch.tensor([.05, .25, .5, .75, .95])
    pieces = []
    for b in blocks:
        pieces.extend([b.mean(-1), b.std(-1), torch.quantile(b, quantiles, dim=-1).permute(1, 2, 0).flatten(1)])
    for c in range(3):
        hist = torch.stack([torch.histc(row[c], bins=16, min=0, max=1) for row in x])
        pieces.append(hist / hist.sum(1, keepdim=True))
    return torch.cat(pieces, dim=1)


class FeatureRouter:
    """Nearest class mean with train-only per-coordinate shrinkage scaling.

    Class indices address archived entries; no semantic name reaches retrieval.
    Scaling suppresses nuisance variation without discarding minority components.
    """
    def __init__(self, groups, whiten=True):
        self.mean = torch.stack([g.mean(0) for g in groups])
        residual = torch.cat([g - m for g, m in zip(groups, self.mean)])
        variance = residual.square().mean(0)
        self.scale = (variance + .1 * variance.mean()).clamp_min(1e-6).sqrt() if whiten else torch.ones_like(variance)

    def scores(self, x):
        return -(((x[:, None] - self.mean[None]) / self.scale)**2).mean(-1)

    def predict(self, x):
        return int(self.scores(x).mean(0).argmax())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--suite', choices=SUITES, required=True)
    p.add_argument('--features', choices=['style', 'dino', 'face'], default='style')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(7301)
    cfg = SUITES[a.suite]
    out = ART/a.suite
    if not (cfg['data']/f'{cfg["names"][0]}_train.pt').exists():
        atomic(out/'unavailable.json', dict(reason='Original data unavailable on SiT pod', path=str(cfg['data'])))
        raise FileNotFoundError(cfg['data'])
    cache = ROOT/'features'/a.suite
    cache.mkdir(parents=True, exist_ok=True)
    ds = load_data(a.suite)
    network = None
    detector = None
    if a.features == 'dino':
        os.environ['TORCH_HOME'] = str(ROOT/'torch')
        network = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True).cuda().eval().requires_grad_(False)
    if a.features == 'face':
        from facenet_pytorch import MTCNN,InceptionResnetV1
        os.environ['TORCH_HOME']=str(ROOT/'torch')
        detector=MTCNN(image_size=160,margin=14,keep_all=False,device='cuda')
        # Independent weights from the VGGFace2 network used for final evaluation.
        network=InceptionResnetV1(pretrained='casia-webface').cuda().eval().requires_grad_(False)
    prep = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                               transforms.Normalize((.485, .456, .406), (.229, .224, .225))])
    features = {}
    started = time.time()
    for n in cfg['names']:
        features[n] = {}
        for split in ['train', 'support']:
            path = cache/f'{n}_{split}_{a.features}.pt'
            if path.exists():
                features[n][split] = torch.load(path, weights_only=True)
                continue
            images = image_rows(a.suite, n, split)
            if a.features == 'face':
                faces=[];detected=[]
                for im in images:
                    face=detector(im)
                    detected.append(face is not None)
                    if face is None:
                        face=(transforms.ToTensor()(im.resize((160,160))).cuda()*255-127.5)/128
                    faces.append(face.cpu())
                with torch.no_grad():
                    values=torch.cat([network(torch.stack(faces[i:i+32]).cuda()).cpu() for i in range(0,len(faces),32)])
                atomic(cache/f'{n}_{split}_face_detection.json',dict(detected=detected,encoder='CASIA-WebFace FaceNet; independent of VGGFace2 evaluation weights'))
            elif network is None:
                values = style_features(images)
            else:
                chunks = []
                with torch.no_grad():
                    for start in range(0, len(images), 32):
                        batch = torch.stack([prep(im) for im in images[start:start+32]]).cuda()
                        chunks.append(F.normalize(network(batch).float(), dim=1).cpu())
                values = torch.cat(chunks)
            torch.save(values, path)
            features[n][split] = values
            print('FEATURES', a.suite, n, split, list(values.shape), flush=True)
    groups = [features[n]['train'] for n in cfg['names']]
    methods = {a.features+'_nearest_mean': FeatureRouter(groups, False),
               a.features+'_whitened_mean': FeatureRouter(groups, True)}
    if a.features == 'dino' and all((cache/f'{n}_train_style.pt').exists() for n in cfg['names']):
        # Normalize each feature branch separately using training variance.
        branches = []
        for n in cfg['names']:
            for split in ['train', 'support']:
                semantic = features[n][split]
                style = torch.load(cache/f'{n}_{split}_style.pt', weights_only=True)
                features[n][split+'_combined'] = torch.cat([semantic, style], 1)
            branches.append(features[n]['train_combined'])
        methods['semantic_style_whitened'] = FeatureRouter(branches, True)
    results = {}
    for name, router in methods.items():
        episodes = []
        for index, n in enumerate(cfg['names']):
            split = 'support_combined' if name.startswith('semantic_style') else 'support'
            support = features[n][split]
            for draw in range(100):
                ids = torch.randperm(len(support), generator=torch.Generator().manual_seed(15001+draw))[:4]
                prediction = router.predict(support[ids])
                episodes.append(dict(target=index, prediction=prediction, correct=prediction==index,
                                     draw=draw, support_indices=ids.tolist()))
        results[name] = dict(accuracy=sum(e['correct'] for e in episodes)/len(episodes), episodes=episodes)
    # Original projector vs actual checkpoint masks: diagnosis only, no fitting to support.
    pca = InputPrior()
    for n in cfg['names']:
        pca.archive(ds[n]['train']['z'], torch.zeros(1))
    episodes = []
    for i, n in enumerate(cfg['names']):
        for draw in range(100):
            ids = torch.randperm(len(ds[n]['support']['z']), generator=torch.Generator().manual_seed(15001+draw))[:4]
            logits, entry, error = pca.retrieve(ds[n]['support']['z'][ids])
            episodes.append(dict(target=i, prediction=entry, hit=logits is not None,
                                 correct=entry==i and logits is not None, draw=draw, error=error))
    results['original_pca'] = dict(accuracy=sum(e['correct'] for e in episodes)/len(episodes), episodes=episodes)
    atomic(out/f'recognition_{a.features}.json', dict(suite=a.suite, feature=a.features,
        training_only_keys=True, support_per_episode=4, episodes_share_support_pool=True,
        seconds=time.time()-started, results=results))
    print('RECOGNITION', a.suite, {k:v['accuracy'] for k,v in results.items()}, flush=True)


if __name__ == '__main__':
    main()
