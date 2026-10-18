"""NumPy single-observation RECODE, following globo_explorer/recode/recode.py.

Fixed pixel features avoid representation drift. Memory persists across lives.
The weighted kth-neighbor reward is measured BEFORE insertion, in RMS units.
"""
import json
from pathlib import Path

import numpy as np


def embedding(image):
    # The adapter uses 64x64. 8x8 RGB means retain color at modest memory cost.
    h, w, c = image.shape
    return image.reshape(8, h // 8, 8, w // 8, c).mean((1, 3)).reshape(-1).astype(np.float32) / 255


class Recode:
    def __init__(self, capacity=1024, dim=192, k=10, seed=0):
        self.memory = np.zeros((capacity, dim), np.float32)
        self.counts = np.zeros(capacity, np.float32)
        self.nearest = np.full(capacity, np.inf, np.float32)
        self.size, self.scale, self.k, self.seen = 0, 0., k, 0
        self.rng = np.random.default_rng(seed)

    def query(self, x):
        if not self.size:
            return 0.
        ds = np.mean((self.memory[:self.size] - x) ** 2, axis=-1)
        k = min(self.k, self.size)
        ids = np.argsort(ds)[:k]
        idx = min(np.searchsorted(np.cumsum(self.counts[ids]), k), k - 1)
        return float(np.sqrt(ds[ids[idx]]))

    def update(self, x):
        self.seen += 1
        if self.size < len(self.memory):
            self.memory[self.size] = x
            self.counts[self.size] = 1
            self.size += 1
            if self.size > self.k:
                ds = np.mean((self.memory[:self.size] - x) ** 2, axis=-1)
                ds[ds == 0] = np.inf
                self._scale(ds)
            return
        ds = np.mean((self.memory - x) ** 2, axis=-1)
        self._scale(ds)
        self.counts *= .9995
        j = int(np.argmin(ds))
        if ds[j] > .2 * self.scale and self.rng.random() < .1:
            if np.isinf(self.nearest).all():
                for i, point in enumerate(self.memory):
                    dist = np.mean((self.memory - point) ** 2, axis=-1)
                    dist[dist == 0] = np.inf
                    self.nearest[i] = dist.min()
            old = int(self.nearest.argmin())
            dist = np.mean((self.memory - self.memory[old]) ** 2, axis=-1)
            dist[old] = np.inf
            near = int(dist.argmin())
            total = self.counts[near] + self.counts[old]
            self.memory[near] = (self.counts[near] * self.memory[near] +
                                 self.counts[old] * self.memory[old]) / total
            self.counts[near] = total
            self.memory[old], self.counts[old], self.nearest[old] = x, 1, ds[j]
        else:
            self.memory[j] = (self.counts[j] * self.memory[j] + x) / (self.counts[j] + 1)
            self.counts[j] += 1

    def _scale(self, ds):
        val = np.sort(ds)[:self.k].mean()
        if np.isfinite(val):
            self.scale = .9995 * self.scale + .0005 * val if self.scale else float(val)

    def save(self, path):
        path = Path(path)
        temp = path.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, memory=self.memory, counts=self.counts,
                nearest=self.nearest, size=self.size, scale=self.scale,
                seen=self.seen, rng=json.dumps(self.rng.bit_generator.state))
        temp.replace(path)

    def load(self, path):
        with np.load(path, allow_pickle=False) as data:
            if data['memory'].shape != self.memory.shape:
                raise ValueError('Density memory shape does not match configuration')
            self.memory[:] = data['memory']
            self.counts[:] = data['counts']
            self.nearest[:] = data['nearest']
            self.size, self.scale, self.seen = int(data['size']), float(data['scale']), int(data['seen'])
            self.rng.bit_generator.state = json.loads(str(data['rng']))
