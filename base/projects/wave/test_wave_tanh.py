"""Run with: python -m unittest test_wave_tanh.py"""
import unittest
from itertools import product

import torch

from wave_tanh import WaveTanh


class WaveTanhTests(unittest.TestCase):
    def test_full_zero_state_jacobian_spectrum(self):
        # Different boundaries and rectangular sheets: no FFT assumptions
        # in this check, construct the actual full Jacobian by autograd.
        for padding, (height, width, rho) in product(
                ("zeros", "circular", "reflect"), ((3, 4, 0.7), (5, 5, 0.985))):
            model = WaveTanh(rho=rho, padding=padding).double()
            x = torch.zeros(2*height*width, dtype=torch.float64)
            jac = torch.autograd.functional.jacobian(
                lambda z: model(z.reshape(1, 2, height, width)).flatten(), x)
            radii = torch.linalg.eigvals(jac).abs()
            torch.testing.assert_close(radii, torch.full_like(radii, rho),
                                       atol=2e-6, rtol=2e-6)
            self.assertEqual(list(model.parameters()), [])

    def test_local_input_propagates_and_decays(self):
        model = WaveTanh(rho=0.97)
        state = torch.zeros(1, 2, 48, 48)
        impulse = torch.zeros_like(state[:, :1])
        impulse[0, 0, 24, 24] = 1
        state = model(state, impulse)
        self.assertEqual(torch.count_nonzero(state).item(), 1)
        next_state = model(state)
        self.assertEqual(next_state[0, 0, 24, 26].item(), 0)
        self.assertGreater(next_state[0, 0, 24, 25].item(), 0)
        reached = 0.
        for t in range(400):
            state = model(state)
            reached = max(reached, abs(state[0, 0, 24, 34].item()))
        self.assertGreater(reached, 0.001)
        self.assertLess(state.abs().max().item(), 1e-5)

    def test_boundary_behavior(self):
        state = torch.zeros(1, 2, 5, 6)
        state[0, 0, 0, 0] = 0.1
        for padding in ("zeros", "circular", "reflect"):
            out = WaveTanh(padding=padding)(state)
            self.assertEqual(out.shape, state.shape)
            self.assertEqual(out[0, 0, -1, -1].item() > 0,
                             padding == "circular")
        # Reflection duplicates the neighbor one pixel inside the boundary.
        state.zero_()
        state[0, 0, 1, 2] = 0.1
        zero = WaveTanh(padding="zeros")(state)[0, 0, 0, 2]
        reflected = WaveTanh(padding="reflect")(state)[0, 0, 0, 2]
        torch.testing.assert_close(reflected, torch.tanh(2 * torch.atanh(zero)))

    def test_invalid_padding(self):
        with self.assertRaises(ValueError):
            WaveTanh(padding="invalid")
        with self.assertRaisesRegex(ValueError, "height and width"):
            WaveTanh(padding="reflect")(torch.zeros(1, 2, 1, 5))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
