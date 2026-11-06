"""Exercise optimizer selection through real mask searches; run on the pod."""
import unittest
from unittest.mock import patch
import torch
from config import Default, Adam, SGD, RMSprop
from ftn import FTN
from reconfig_optimizer import make_reconfig_optimizer
from test_overlap_regularizer import CASES, make


class Tests(unittest.TestCase):
    def test_config(self):
        for cls, name in [(Default, 'Adam'), (Adam, 'Adam'), (SGD, 'SGD'), (RMSprop, 'RMSprop')]:
            cfg = cls()
            for prop in ('FTN_PARS', 'QUAD_PARS', 'LORA_PARS', 'LORARNN_PARS', 'STACK_PARS', 'HML_PARS', 'WAVE_PARS'):
                self.assertEqual(getattr(cfg, prop)['reconfig_optimizer'], name)
            if cls is not Default:
                self.assertEqual({k for k in vars(cls) if not k.startswith('__')}, {'RECONFIG_OPTIMIZER'})

    def test_searches(self):
        x = torch.randn(8, 3)
        y = torch.eye(2)[torch.arange(8) % 2]
        for name in ('Adam', 'SGD', 'RMSprop'):
            for cls, extra in CASES:
                with self.subTest(optimizer=name, model=cls.__module__, extra=extra):
                    model = make(cls, dict(extra, reconfig_optimizer=name))
                    constructor = getattr(torch.optim, name)
                    with patch.object(torch.optim, name, wraps=constructor) as factory:
                        for method in ('reconfigure', 'fast_reconfigure'):
                            factory.reset_mock()
                            result = getattr(model, method)(x, y)
                            self.assertTrue(factory.called)
                            self.assertTrue(torch.isfinite(result).all())
                            self.assertTrue(torch.allclose(result, result.round(), atol=1e-6))
                            sums = result.reshape(-1, 16).sum(-1)
                            self.assertTrue(torch.allclose(sums, torch.full_like(sums, 4)))
                        if extra.get('use_second_mask'):
                            factory.reset_mock()
                            model.dual_reconfigure(x, y)
                            self.assertGreaterEqual(factory.call_count, 2)

    def test_validation_and_expansion(self):
        with self.assertRaises(ValueError):
            make(FTN, dict(reconfig_optimizer='bogus'))
        with self.assertRaises(ValueError):
            make_reconfig_optimizer([torch.nn.Parameter(torch.zeros(1))], lr=.3, name='bogus')
        for name in ('Adam', 'SGD', 'RMSprop'):
            self.assertEqual(make(FTN, dict(reconfig_optimizer=name)).expand(32).reconfig_optimizer, name)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
