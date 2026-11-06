"""Download standard vision datasets directly to remote shared storage."""
import argparse
import fcntl
import hashlib
import json
import pickle
import zipfile
from pathlib import Path

import requests
from torchvision.datasets import CIFAR10, CIFAR100
from tqdm import tqdm

from storage import WORKSPACE, bulk, write_json
from parallel_download import download_ranges, checksum


def download(url, dest):
    if dest.exists():
        return
    partial = dest.with_suffix(dest.suffix + '.partial')
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with partial.open('wb') as f, tqdm(total=int(response.headers.get('content-length', 0)), unit='B', unit_scale=True, desc=dest.name) as bar:
            for block in response.iter_content(4 * 1024 * 1024):
                f.write(block)
                bar.update(len(block))
    partial.replace(dest)


def prepare(name):
    directory = bulk('data', name)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'prepare.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = directory / 'ready.json'
        if ready.exists():
            print(f'{name}: already prepared', flush=True)
            return
        if name in ('cifar10', 'cifar100'):
            cls = CIFAR10 if name == 'cifar10' else CIFAR100
            archive=directory/cls.filename
            if archive.exists() and checksum(archive) != cls.tgz_md5:
                archive.rename(archive.with_suffix(archive.suffix+'.incomplete'))
            download_ranges(cls.url,archive,md5=cls.tgz_md5)
            for train in (True, False):
                cls(str(directory), train=train, download=True)
        elif name == 'tinyimagenet':
            archive = directory / 'tiny-imagenet-200.zip'
            try:
                download('https://cs231n.stanford.edu/tiny-imagenet-200.zip', archive)
            except requests.RequestException:
                # Distribution maintained by the DER++/Mammoth authors.
                from huggingface_hub import HfApi, snapshot_download
                info = HfApi().repo_info('aimagelab-ta/tinyimg', repo_type='dataset')
                snapshot_download('aimagelab-ta/tinyimg', repo_type='dataset', revision=info.sha,
                    local_dir=str(directory / 'mammoth'), allow_patterns=['processed/*.npy'], max_workers=8)
                write_json(directory/'mirror.json', {'repo':'aimagelab-ta/tinyimg', 'revision':info.sha})
            if archive.exists():
                with zipfile.ZipFile(archive) as z:
                    z.extractall(directory)
        elif name == 'core50':
            archive = directory / 'core50_128x128.zip'
            download_ranges('http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip', archive,
                            md5='745f3373fed08d69343f1058ee559e13')
            core_digest = hashlib.md5()
            with archive.open('rb') as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    core_digest.update(block)
            if core_digest.hexdigest() != '745f3373fed08d69343f1058ee559e13':
                raise ValueError('CORe50 archive does not match the official checksum.')
            with zipfile.ZipFile(archive) as z:
                z.extractall(directory)
            # Keep even the official split metadata out of source sync.
            for filename in ('LUP.pkl', 'labels.pkl', 'paths.pkl'):
                download('https://raw.githubusercontent.com/vlomonaco/core50/master/extras/' + filename, directory / filename)
        else:
            raise ValueError(name)
        write_json(ready, {'dataset': name, 'path': str(directory), 'source': 'official dataset distribution'})
        print(f'{name}: ready at {directory}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='+', choices=['cifar10', 'cifar100', 'tinyimagenet', 'core50'])
    args = parser.parse_args()
    for dataset in args.datasets:
        prepare(dataset)
