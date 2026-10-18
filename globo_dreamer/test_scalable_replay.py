"""Run on pod: python -m unittest globo_dreamer.test_scalable_replay."""
import tempfile
import unittest
import torch
from .model_free import ModelFreeBehavior
from .legacy.agent import ReplayBuffer
from .scalable_replay import ChunkStore, trajectory


class ReplayTests(unittest.TestCase):
    def make(self, capacity=5):
        return ReplayBuffer(capacity, 'cpu', emb_dim=1)

    def add(self, buf, number, boundary=False):
        z = torch.full((1, 1, 1), float(number))
        buf.add(z, [number], 0., 0., z.flatten(), z, False, boundary)

    def test_ring_and_boundaries(self):
        buf = self.make()
        for i in range(8):
            self.add(buf, i, boundary=i == 5)
        self.assertEqual(buf.action[trajectory(buf, 2, 10), 0].tolist(), [6, 7])
        self.assertEqual(buf.action[trajectory(buf, 0, 10), 0].tolist(), [3, 4, 5])
        self.assertEqual(buf.action[trajectory(buf, 3, 10), 0].tolist(), [3])

    def test_incremental_checkpoint_commit_and_resume(self):
        with tempfile.TemporaryDirectory() as root:
            buf = self.make()
            store = ChunkStore(root, dict(memory=buf))
            for i in range(8): self.add(buf, i)
            first = store.save()
            self.add(buf, 8)
            second = store.save()
            # Previous checkpoint remains readable until the new one commits.
            restored = self.make()
            reader = ChunkStore(root, dict(memory=restored))
            reader.load(first)
            self.assertEqual(restored.ptr, 3)
            reader.load(second)
            self.assertEqual(restored.ptr, 4)
            torch.testing.assert_close(restored.z, buf.z)
            store.committed()
            self.assertEqual(len(list(store.root.glob('*.pt'))), 1)
            self.assertEqual(store.save(), second)

    def test_frontier_trajectory_ends_at_frontier(self):
        buf = self.make(10)
        for i in range(8): self.add(buf, i)
        self.assertEqual(buf.action[trajectory(buf, 4, 3), 0].tolist(), [2, 3, 4])

    def test_training_and_behavior_resume_after_wrap(self):
        with tempfile.TemporaryDirectory() as root:
            behavior = ModelFreeBehavior(device='cpu', replay_capacity=8,
                goal_capacity=8, recode_capacity=32, checkpoint_root=root)
            agent = behavior.agent
            agent.minibatch_size = 4
            image = torch.rand(3, 32, 64)
            for i in range(20):
                z = agent.encode(image, torch.tensor(.1))
                agent.update_density(image)
                agent.remember(z, i % 25, 0., .1, agent.embed(image), z, False, i % 7 == 0)
            self.assertTrue(torch.isfinite(torch.tensor(agent.train_round(3))))
            state = behavior.save()
            behavior.store.committed()
            restored = ModelFreeBehavior(device='cpu', replay_capacity=8,
                goal_capacity=8, recode_capacity=32, checkpoint_root=root)
            restored.load(state)
            self.assertEqual(restored.agent.memory.ptr, agent.memory.ptr)
            torch.testing.assert_close(restored.agent.memory.z, agent.memory.z)
            torch.testing.assert_close(restored.agent.goal.boundary, agent.goal.boundary)


if __name__ == '__main__': unittest.main()
