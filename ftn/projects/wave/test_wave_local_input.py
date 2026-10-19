"""Run in base pod: python -m unittest projects.wave.test_wave_local_input."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from config import Default, Wave2
from ftn import ParallelMLP
from wave_ftn import WaveFTN


class LocalInputTests(unittest.TestCase):
    def model(self, **kwargs):
        return WaveFTN(2, 1, 16, 4, k=2, layers=1, inner_size=3, **kwargs)

    def test_patch_values_and_gradient_support(self):
        for padding in ('zeros', 'circular', 'reflect'):
            for size in (1, 3, 5):
                with self.subTest(padding=padding, size=size):
                    model = self.model(wave_padding=padding, wave_input_kernel_size=size)
                    r = torch.arange(32.).reshape(1,2,4,4).requires_grad_()
                    x = torch.tensor([[100., 200.]])
                    actual = model._cell_input(x, r)
                    expected = []
                    support = torch.zeros_like(r)
                    def index(i):
                        if 0 <= i < 4:
                            return i
                        if padding == 'zeros':
                            return None
                        if padding == 'circular':
                            return i % 4
                        return -i if i < 0 else 6-i
                    for row in range(4):
                        for col in range(4):
                            patch = []
                            for dy in range(-size//2+1, size//2+1):
                                for dx in range(-size//2+1, size//2+1):
                                    y, z = index(row+dy), index(col+dx)
                                    patch.append(0. if y is None or z is None else float(y*4+z))
                                    if row == 0 and col == 0 and y is not None and z is not None:
                                        support[0,0,y,z] += 1
                            expected.append([100.,200.] + patch)
                    torch.testing.assert_close(actual, torch.tensor([expected]))
                    actual[0,0,2:].sum().backward()
                    torch.testing.assert_close(r.grad, support)
                    self.assertEqual(model.cell.weights[0].shape, (16,2+size**2,3))

    def test_parallel_mlp_shared_input_unchanged(self):
        cell = ParallelMLP(5,1,16,2,3)
        x = torch.randn(2,5)
        torch.testing.assert_close(cell(x), cell(x[:,None].expand(-1,16,-1)))

    def test_config_and_validation(self):
        for cls in (Default, Wave2):
            cfg = cls()
            self.assertEqual(cfg.WAVE_INPUT_KERNEL_SIZE, 3)
            cfg.WAVE_INPUT_KERNEL_SIZE = 5
            self.assertEqual(cfg.WAVE_PARS['wave_input_kernel_size'], 5)
        for size in (0, -1, 2, 3.5, True, 9):
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.model(wave_input_kernel_size=size)


if __name__ == '__main__':
    unittest.main()
