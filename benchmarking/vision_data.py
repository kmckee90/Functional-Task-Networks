"""Shared benchmark splits, disjoint support/query sets, remote feature caches."""
import fcntl
import pickle
import bisect
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

from storage import bulk, write_json

N_CLASSES = {'cifar10': 10, 'cifar100': 100, 'tinyimagenet': 200, 'core50': 50}
N_TASKS = {'cifar10': 5, 'cifar100': 10, 'tinyimagenet': 10, 'core50': 9}


class FeatureDataset(Dataset):
    def __init__(self,x,y):
        self.data=x
        self.targets=y.cpu().numpy().astype(np.int64)
    def __len__(self): return len(self.targets)
    def __getitem__(self,index): return self.data[index],int(self.targets[index])


def feature_datasets(name,seed=0):
    path=bulk('data','features',f'{name}_vit_run{seed if name=="core50" else 0}.pt')
    if not path.exists():
        raise RuntimeError(f'Frozen vision feature cache is not ready: {path}')
    cache=torch.load(path,map_location='cpu',weights_only=True)
    train=FeatureDataset(cache['train_x'],cache['train_y'])
    test=FeatureDataset(cache['test_x'],cache['test_y'])
    if name=='core50':
        import pickle
        with bulk('data','core50','labels.pkl').open('rb') as f:
            labels=pickle.load(f)['nc'][seed%10]
        groups=[sorted(set(map(int,labels[i]))) for i in range(9)]
    else:
        order=np.random.default_rng(seed).permutation(N_CLASSES[name])
        groups=[list(map(int,g)) for g in np.array_split(order,N_TASKS[name])]
    return train,test,groups,cache.get('encoder_parameters',0),cache['encoder']


class ImageList(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.targets = np.array([label for _, label in samples], dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as im:
            x = self.transform(im.convert('RGB'))
        return x, label


class ProcessedTiny(Dataset):
    """Read the official Mammoth distribution without loading all float arrays."""
    def __init__(self, root, train, transform):
        split = 'train' if train else 'val'
        self.data = [np.load(root/f'x_{split}_{i:02d}.npy', mmap_mode='r') for i in range(1,21)]
        self.targets = np.concatenate([np.load(root/f'y_{split}_{i:02d}.npy') for i in range(1,21)]).astype(np.int64)
        self.ends = np.cumsum([len(x) for x in self.data]).tolist()
        self.transform = transform
        assert len(self.targets) == (100000 if train else 10000)
        assert len(np.unique(self.targets)) == 200

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        shard = bisect.bisect_right(self.ends, index)
        offset = index - (self.ends[shard-1] if shard else 0)
        image = Image.fromarray(np.uint8(255 * self.data[shard][offset]))
        return self.transform(image), int(self.targets[index])


def image_transform(size=224, prompt=False):
    if prompt:
        return transforms.Compose([transforms.Resize(256,interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),transforms.ToTensor()])
    return transforms.Compose([transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.ToTensor(), transforms.Normalize((.485, .456, .406), (.229, .224, .225))])


def raw_datasets(name, seed=0, size=224, prompt=False):
    directory = bulk('data', name)
    if not (directory / 'ready.json').exists():
        raise RuntimeError(f'Dataset {name} is not prepared.')
    tf = image_transform(size,prompt)
    if name in ('cifar10', 'cifar100'):
        cls = datasets.CIFAR10 if name == 'cifar10' else datasets.CIFAR100
        train = cls(str(directory), train=True, download=False, transform=tf)
        test = cls(str(directory), train=False, download=False, transform=tf)
        classes = list(np.random.default_rng(seed).permutation(N_CLASSES[name]).astype(int))
        groups = [list(map(int, x)) for x in np.array_split(classes, N_TASKS[name])]
    elif name == 'tinyimagenet':
        base = directory / 'tiny-imagenet-200'
        if base.exists():
            train = datasets.ImageFolder(str(base / 'train'), transform=tf)
            labels = dict(train.class_to_idx)
            rows = [line.split('\t') for line in (base / 'val/val_annotations.txt').read_text().splitlines()]
            test = ImageList([(base / 'val/images' / row[0], labels[row[1]]) for row in rows], tf)
        else:
            train = ProcessedTiny(directory/'mammoth/processed', True, tf)
            test = ProcessedTiny(directory/'mammoth/processed', False, tf)
        classes = list(np.random.default_rng(seed).permutation(200).astype(int))
        groups = [list(map(int, x)) for x in np.array_split(classes, 10)]
    elif name == 'core50':
        # Official NC runs (10 classes then 8 batches of 5), official held-out
        # sessions. LUP indexes and labels are paired within each batch.
        with (directory / 'paths.pkl').open('rb') as f:
            paths = pickle.load(f)
        with (directory / 'LUP.pkl').open('rb') as f:
            lup = pickle.load(f)['nc'][seed % 10]
        with (directory / 'labels.pkl').open('rb') as f:
            targets = pickle.load(f)['nc'][seed % 10]
        groups = [sorted(set(map(int, targets[i]))) for i in range(9)]
        assert [len(g) for g in groups]==[10]+[5]*8
        assert len(set(c for g in groups for c in g))==50
        assert set(j for batch in lup[:-1] for j in batch).isdisjoint(lup[-1])
        train_rows = [(directory / 'core50_128x128' / paths[j], int(y)) for i in range(9) for j, y in zip(lup[i], targets[i])]
        test_rows = [(directory / 'core50_128x128' / paths[j], int(y)) for j, y in zip(lup[-1], targets[-1])]
        train, test = ImageList(train_rows, tf), ImageList(test_rows, tf)
    else:
        raise ValueError(name)
    return train, test, groups


def split_indices(targets, groups, seed):
    targets = np.asarray(targets)
    rng = np.random.default_rng(seed + 81091)
    training, calibration = [], []
    for group in groups:
        ti, ci = [], []
        for label in group:
            indexes = np.flatnonzero(targets == label)
            rng.shuffle(indexes)
            n = max(1, int(round(len(indexes) * .1)))
            ci.extend(indexes[:n].tolist())
            ti.extend(indexes[n:].tolist())
        training.append(np.array(ti, dtype=np.int64))
        calibration.append(np.array(ci, dtype=np.int64))
    return training, calibration


def cache_features(name, encoder='resnet18', run=0, batch_size=256):
    # CIFAR/Tiny source order does not depend on class-order seed. CORe50
    # official NC metadata changes the sample order and label mapping per run.
    cache_run = run if name == 'core50' else 0
    dest = bulk('data', 'features', f'{name}_{encoder}_run{cache_run}.pt')
    with dest.with_suffix('.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if dest.exists():
            return dest
        if encoder == 'resnet18':
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            model.fc = torch.nn.Identity()
            weight_id = 'torchvision.ResNet18_Weights.IMAGENET1K_V1'
        else:
            from ftn_models import pretrained_vit,VIT_CHECKPOINT_URL
            model=pretrained_vit(0)
            weight_id=VIT_CHECKPOINT_URL
        model = model.cuda().eval().requires_grad_(False)
        train, test, _ = raw_datasets(name, cache_run,prompt=encoder=='vit')
        result = {'encoder': weight_id,'encoder_parameters':sum(p.numel() for p in model.parameters())}
        for split, dataset in [('train', train), ('test', test)]:
            feats, labels = [], []
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
            with torch.inference_mode():
                for i, (x, y) in enumerate(loader):
                    feats.append(model(x.cuda()).cpu())
                    labels.append(y)
                    if i % 50 == 0:
                        print(name, encoder, split, i, '/', len(loader), flush=True)
            result[split + '_x'] = torch.cat(feats)
            result[split + '_y'] = torch.cat(labels)
        temporary = dest.with_suffix('.partial')
        torch.save(result, temporary)
        temporary.replace(dest)
        write_json(dest.with_suffix('.json'), {'dataset': name, 'encoder': weight_id,
                   'shapes': {k: list(v.shape) for k, v in result.items() if isinstance(v, torch.Tensor)},
                   'note': 'Frozen ImageNet-pretrained features; no benchmark training or test-label fitting.'})
        return dest


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=N_CLASSES, required=True)
    p.add_argument('--encoder', choices=['resnet18', 'vit'], default='resnet18')
    p.add_argument('--run', type=int, default=0)
    a = p.parse_args()
    print(cache_features(a.dataset, a.encoder, a.run))
