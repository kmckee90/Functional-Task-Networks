"""Resumable HTTP range download; bulk parts stay beside the remote archive."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import math
import time

import requests


def download_ranges(url, dest, md5=None, workers=16):
    if dest.exists():
        if md5 is None or checksum(dest) == md5:
            return
        raise ValueError(f'Existing archive checksum mismatch: {dest}')
    with requests.get(url, headers={'Range':'bytes=0-0'}, timeout=(30,120), stream=True) as r:
        r.raise_for_status()
        if r.status_code != 206 or not r.headers.get('Content-Range'):
            raise RuntimeError('Server does not support validated HTTP ranges.')
        total = int(r.headers['Content-Range'].rsplit('/',1)[1])
    chunk = 8 * 1024 * 1024
    parts = dest.parent / (dest.name + '.parts')
    parts.mkdir(exist_ok=True)
    n = math.ceil(total/chunk)

    def get_part(i):
        begin, end = i*chunk, min((i+1)*chunk, total)-1
        out = parts / f'{i:05d}'
        if out.exists() and out.stat().st_size == end-begin+1:
            return i
        for trial in range(5):
            try:
                with requests.get(url, headers={'Range':f'bytes={begin}-{end}'}, timeout=(30,120), stream=True) as r:
                    r.raise_for_status()
                    if r.status_code != 206 or r.headers.get('Content-Range') != f'bytes {begin}-{end}/{total}':
                        raise ValueError('Invalid Content-Range received.')
                    temp = out.with_suffix('.partial')
                    with temp.open('wb') as f:
                        for block in r.iter_content(1024*1024):
                            f.write(block)
                    if temp.stat().st_size != end-begin+1:
                        raise ValueError('Truncated range response.')
                    temp.replace(out)
                    return i
            except requests.RequestException:
                if trial == 4:
                    raise
                time.sleep(2**trial)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for completed, _ in enumerate(pool.map(get_part, range(n)), 1):
            if completed % 10 == 0 or completed == n:
                print(f'{dest.name}: {completed}/{n} ranges ready', flush=True)
    temp = dest.with_suffix(dest.suffix+'.assembled')
    digest = hashlib.md5()
    with temp.open('wb') as f:
        for i in range(n):
            with (parts / f'{i:05d}').open('rb') as part:
                for block in iter(lambda:part.read(8*1024*1024),b''):
                    f.write(block)
                    digest.update(block)
    if md5 is not None and digest.hexdigest() != md5:
        raise ValueError(f'Download checksum mismatch: {dest}')
    temp.replace(dest)


def checksum(path):
    h=hashlib.md5()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()
