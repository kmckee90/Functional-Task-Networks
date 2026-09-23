"""Run in the base pod: python -m unittest test_nonlinear_tests.py"""
from types import SimpleNamespace
import unittest
import torch

from nonlinear_tests import ResidualMLP, prepare, train_mlp, predict
from linear_tests import channel_r2


class NonlinearDecoderTests(unittest.TestCase):
    def test_recovers_nonlinear_signal_and_zero_skip_initialization(self):
        device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(42)
        splits = []
        for n in (2048, 512, 512):
            x = torch.randn(n, 4, device=device)
            y = (x[:, 0]*x[:, 1]).reshape(n, 1, 1)
            splits.append((x, y))
        xs, ys, base, _ = prepare(splits, [1e-4, .1, 10.])
        args = SimpleNamespace(widths=[64], epochs=100, batch_size=256,
                               lr=.003, weight_decay=.001, eval_every=20,
                               patience=5)
        model = ResidualMLP(4, 64, 1).to(device)
        self.assertEqual(model(xs[0]).abs().max().item(), 0.)
        # No test split is supplied during optimization or checkpoint choice.
        best, scale, _ = train_mlp(xs[:2], ys[:2], base[:2], args, 4)
        model.load_state_dict(best[0]['state'])
        with torch.no_grad():
            output = base[2] + predict(model, xs[2]).reshape(-1, 1, 1)*scale
            linear_r2 = channel_r2(base[2][:,0],ys[2][:,0]).item()
            nonlinear_r2 = channel_r2(output[:,0],ys[2][:,0]).item()
        self.assertLess(linear_r2, .1)
        self.assertGreater(nonlinear_r2, .9)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
