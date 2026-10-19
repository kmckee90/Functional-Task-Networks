"""Run on pod: python -m unittest globo_dreamer.test_memory."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .memory import Recode, embedding


class MemoryTests(unittest.TestCase):
    def test_weighted_knn_and_query_before_update(self):
        memory = Recode(capacity=3, dim=1, k=2)
        for x in (0., .5, 1.):
            memory.update(np.array([x], np.float32))
        self.assertAlmostEqual(memory.query(np.array([0.])), .5)
        memory.counts[0] = 3
        self.assertEqual(memory.query(np.array([0.])), 0.)

    def test_restart_preserves_novelty_and_future_updates(self):
        memory = Recode(capacity=12, dim=3, k=3, seed=3)
        xs = np.random.default_rng(8).random((70, 3)).astype(np.float32)
        for x in xs[:40]:
            memory.update(x)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'memory.npz'
            memory.save(path)
            restored = Recode(capacity=12, dim=3, k=3)
            restored.load(path)
        for x in xs[40:]:
            self.assertAlmostEqual(memory.query(x), restored.query(x))
            memory.update(x)
            restored.update(x)
        np.testing.assert_array_equal(memory.memory, restored.memory)
        np.testing.assert_array_equal(memory.counts, restored.counts)

    def test_embedding_is_stable_and_resolution_normalized(self):
        a = np.full((64, 64, 3), 127, np.uint8)
        b = np.full((128, 128, 3), 127, np.uint8)
        np.testing.assert_array_equal(embedding(a), embedding(b))
        self.assertEqual(embedding(a).shape, (192,))


if __name__ == '__main__':
    unittest.main()
