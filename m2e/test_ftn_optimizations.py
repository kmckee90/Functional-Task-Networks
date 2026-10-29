"""Numerical equivalence checks for the FTN performance optimizations."""
from copy import deepcopy
import unittest

import numpy as np
import torch

from agents.ftn_support.spatial_permute import spatial_permutation, _geometry
from ftn_diagnostics import isolated_rng
from test_agent_ftn import make_agent
from test_ftn_diagnostics import assert_tree_equal


def scalar_eligible(agent, masks):
    result = []
    for i, entry in enumerate(agent.ftn_archive):
        overlaps = [float((entry["masks"][k].to(m.device) * m).sum()) / agent.posteriors[k].k
                    for k, m in masks.items()]
        if any(agent.config["ftn_overlap_low"] <= v <= agent.config["ftn_overlap_high"] for v in overlaps):
            result.append(i)
    return result


class FTNOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_archive_thresholds_order_and_rng(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            agent = make_agent(device, ftn_k=3)
            masks = agent._masks()
            self.assertEqual(agent._eligible_archive_indices(masks), [])
            agent.ftn_archive = [dict(masks={k: torch.zeros(8).scatter_(0, torch.randperm(8)[:3], 1.)
                                            for k in masks}) for _ in range(32)]
            # Include exact boundary cases that float32 division can misclassify.
            for low, high in [(0., 0.), (0., 1.), (1/3, 1/3),
                              (np.nextafter(1/3, 1.), 1.), (.2, .8)]:
                agent.config.update(ftn_overlap_low=low, ftn_overlap_high=high)
                before = torch.get_rng_state()
                self.assertEqual(agent._eligible_archive_indices(masks), scalar_eligible(agent, masks))
                torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)

    def test_training_update_matches_scalar_archive_scan(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            fast = make_agent(device)
            d = dict(o=torch.randn(4, fast.rnn_size_o, device=device),
                     f=torch.randn(4, fast.rnn_size_f, device=device),
                     no=torch.randn(4, fast.rnn_size_o, device=device),
                     nf=torch.randn(4, fast.rnn_size_f, device=device),
                     action=torch.randn(4, 2, device=device), ro=torch.ones(4, 1, device=device),
                     rf=torch.ones(4, 1, device=device), done=torch.zeros(4, 1, dtype=torch.bool, device=device), goal=False)
            fast.ftn_archive = [dict(data=fast._trim(d, cpu=True),
                                     masks={k: m.cpu() for k, m in fast._masks().items()})]
            control = deepcopy(fast)
            control._eligible_archive_indices = lambda masks: scalar_eligible(control, masks)
            with isolated_rng():control._train_support(d)
            fast._train_support(d)
            assert_tree_equal(self, control.state_dict(), fast.state_dict())
            for k in ("a", "o", "f"):
                assert_tree_equal(self, getattr(control, "optimizer_" + k).state_dict(),
                                  getattr(fast, "optimizer_" + k).state_dict())

    def test_coactivation_preserves_objective_and_history(self):
        rng = np.random.default_rng(12)
        shape = (4, 5)
        for count in (1, 17, 100):
            masks = rng.random((count, 20)) < .3
            original = masks.copy()
            previous = rng.permutation(20)
            options = dict(initialization="random_blobs", seed=7, sweeps=4,
                           roundness=2., radius=3, center=.25, temperature=0.)
            left, right, weights = _geometry(shape, 3, 2.)
            y, x = np.indices(shape, dtype=float)
            distance = ((y - 1.5)**2 + (x - 2.)**2).ravel()
            central = .25 * distance / distance.max()

            def score(order):
                total = sum(w * np.count_nonzero(masks[:, order[a]] & masks[:, order[b]]) / count
                            for a, b, w in zip(left, right, weights))
                return total - central @ masks[:, order].any(0)

            state = np.random.get_state()
            order = spatial_permutation(masks, shape, previous, **options)
            self.assertGreaterEqual(score(order) + 1e-9, score(previous))
            np.testing.assert_array_equal(np.sort(order), np.arange(20))
            np.testing.assert_array_equal(masks, original)
            values = masks.astype(int)
            np.testing.assert_array_equal(values[:, order] @ values[:, order].T, values @ values.T)
            np.testing.assert_array_equal(order, spatial_permutation(masks, shape, previous, **options))
            # Repeating the full history changes neither mean co-activation nor
            # the objective. This also tests counts beyond the uint8 range.
            np.testing.assert_array_equal(order, spatial_permutation(np.tile(masks, (4, 1)), shape, previous, **options))
            assert_tree_equal(self, state, np.random.get_state())


if __name__ == "__main__":unittest.main()
