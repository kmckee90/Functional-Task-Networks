"""Checks for history-decoding experiments; execute in the base pod."""

from types import SimpleNamespace
import unittest

import torch

from linear_tests import collect, decode


class LinearHistoryTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            train_streams=2, val_streams=1, test_streams=1, size=8,
            burn_in=2, delays=[-1, 0, 1], samples_per_stream=16, ks=[4],
            padding='circular', readout='both', amplitude=.025, interval=4,
            alphas=[1e-4, .01, 1., 10.])

    def test_input_and_history_alignment(self):
        # rho=0 erases state between injections: current masked state must
        # equal the current target exactly, and all other pixels remain zero.
        splits, mask, _ = collect(self.args, 4, 0., 123)
        for x, y in splits:
            torch.testing.assert_close(x[:, mask], .025*y[:, 1],
                                       atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(y[:15, 0], y[1:16, 1])
            torch.testing.assert_close(y[1:16, 2], y[:15, 1])
            unmasked = x.clone()
            unmasked[:, mask] = 0
            self.assertEqual(torch.count_nonzero(unmasked).item(), 0)
        self.assertFalse(torch.equal(splits[0][1][:16], splits[1][1]))

    def test_ridge_recovers_known_linear_signal(self):
        device = self.args.device
        generator = torch.Generator(device=device).manual_seed(34)
        weights = torch.randn(5, 3, device=device, generator=generator)
        splits = []
        for n in [1000, 500, 500]:
            x = torch.randn(n, 5, device=device, generator=generator)
            signal = x @ weights + 2  # also verifies fitted intercept
            independent = torch.randn(n, 3, device=device, generator=generator)
            splits.append((x, torch.stack([independent, signal], dim=1)))
        self.args.delays = [-1, 0]
        scores = decode(self.args, splits)
        self.assertGreater(scores[1]['test_r2'], .999)
        self.assertLess(scores[0]['test_r2'], .05)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
