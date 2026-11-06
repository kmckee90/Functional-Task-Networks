"""Run on the base actl pod: neuron decorrelation math and training isolation."""

import contextlib
import copy
import io
import unittest
from unittest.mock import patch

import torch

import config
import decorrelation_regularizer as reg
import run_experiments as runner
from ftn import FTN
from stack_ftn import StackFTN
from wave_ftn import WaveFTN
from test_overlap_regularizer import CASES, make, seed_mask


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(8, 3)
        self.y = torch.eye(2)[torch.arange(8) % 2]

    def test_matches_pearson_lower_triangle_and_gradients(self):
        z = torch.randn(12, 7, dtype=torch.float64, requires_grad=True)
        corr = torch.corrcoef(z.T)
        indices = torch.tril_indices(7, 7, offset=-1)
        expected = corr[indices[0], indices[1]].square().mean()
        actual = reg.neuron_decorrelation(z)
        self.assertTrue(torch.allclose(actual, expected))
        grad = torch.autograd.grad(actual, z, retain_graph=True)[0]
        reference = torch.autograd.grad(expected, z)[0]
        self.assertTrue(torch.allclose(grad, reference))
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_known_values_and_scale_shift_invariance(self):
        a = torch.tensor([-1., -1., 1., 1.])
        b = torch.tensor([-1., 1., -1., 1.])
        z = torch.stack([a, -a, b], dim=1)
        self.assertAlmostEqual(reg.neuron_decorrelation(z).item(), 1 / 3)
        self.assertAlmostEqual(reg.neuron_decorrelation(z[:, :2]).item(), 1.)
        self.assertAlmostEqual(reg.neuron_decorrelation(z[:, [0, 2]]).item(), 0.)
        transformed = z * torch.tensor([2., -3., 4.]) + torch.tensor([3., 7., -2.])
        self.assertTrue(torch.allclose(reg.neuron_decorrelation(z),
                                       reg.neuron_decorrelation(transformed)))

    def test_constant_singleton_and_low_precision(self):
        for shape in [(8, 3), (1, 3), (8, 1)]:
            z = torch.ones(shape, requires_grad=True)
            penalty = reg.neuron_decorrelation(z)
            self.assertEqual(penalty.item(), 0.)
            penalty.backward()
            self.assertTrue(torch.isfinite(z.grad).all())
        z = torch.randn(8, 4, dtype=torch.float16, requires_grad=True)
        penalty = reg.neuron_decorrelation(z)
        self.assertEqual(penalty.dtype, torch.float32)
        penalty.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_all_neurons_and_training_gradients(self):
        model = make(FTN, {})
        seed_mask(model)
        pred, penalty = reg.weight_training_forward(model, self.x, coefficient=.3)
        expected = .3 * reg.neuron_decorrelation(model.parallel(self.x))
        self.assertTrue(torch.allclose(penalty, expected))
        self.assertTrue(torch.equal(pred, model(self.x)))
        penalty.backward()
        # Inactive neurons still receive decorrelation gradients.
        self.assertGreater(model.parallel.weights[0].grad[4:].abs().sum().item(), 0)
        self.assertIsNone(model.linear.weight.grad)
        self.assertFalse(model.parallel._forward_hooks)

    def test_search_and_eval_never_compute_penalty(self):
        model = make(FTN, {})
        baseline = copy.deepcopy(model)
        search = model._reconfig

        def check_search(*args):
            self.assertFalse(model.parallel._forward_hooks)
            return search(*args)

        with patch.object(model, '_reconfig', side_effect=check_search):
            torch.manual_seed(15)
            pred, penalty = reg.weight_training_forward(model, self.x, self.y, coefficient=.2)
        torch.manual_seed(15)
        expected = baseline(self.x, self.y)
        self.assertTrue(torch.equal(model.m, baseline.m))
        self.assertTrue(torch.equal(pred, expected))
        (model.loss(pred, self.y) + penalty).backward()
        with patch.object(reg, 'neuron_decorrelation', side_effect=AssertionError):
            model(self.x)
            model.reconfigure(self.x, self.y)
            model.fast_reconfigure(self.x, self.y)

    def test_disabled_is_identical_and_validates_coefficient(self):
        model = make(FTN, {})
        seed_mask(model)
        with patch.object(reg, 'neuron_decorrelation', side_effect=AssertionError):
            pred, penalty = reg.weight_training_forward(model, self.x)
        self.assertTrue(torch.equal(pred, model(self.x)))
        self.assertEqual(penalty.item(), 0.)
        for coefficient in [-1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                reg.weight_training_forward(model, self.x, coefficient=coefficient)

    def test_parallel_backbone_variants(self):
        for cls, extra in CASES:
            if cls not in (FTN, StackFTN, WaveFTN):
                continue
            with self.subTest(cls=cls.__name__, extra=extra):
                model = make(cls, extra)
                seed_mask(model)
                pred, penalty = reg.weight_training_forward(model, self.x, coefficient=.1)
                self.assertTrue(torch.isfinite(penalty))
                self.assertTrue(torch.equal(pred, model(self.x)))
                (model.loss(pred, self.y) + penalty).backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                    if p.grad is not None))

    def test_training_schedules_replay_and_probes(self):
        for reactive in [False, True]:
            cfg = config.Default()
            cfg.DECORRELATION_COEF = .1
            cfg.TRAIN_DURATION = 2
            cfg.RENDER_STEPS = False
            cfg.EVAL_BACKWARDS = False
            cfg.RECONFIG_REACTIVE = reactive
            cfg.RECONFIG_REACTIVE_SD = -1e6
            cfg.RECONFIG_REACTIVE_EMA = 0.
            cfg.N_SAMPLE_PER_TASK = 0
            cfg.N_REPLAY_SAMPLES = 1
            cfg.NOISE_PROBE = False
            model = make(FTN, {})
            seed_mask(model)
            optimizer = torch.optim.SGD(model.parameters(), lr=.001)
            replay = runner.ReplayBuffer(maxlen=1)
            # A mask with 50% overlap ensures replay/probe admission.
            other_mask = torch.zeros_like(model.m)
            other_mask[[0, 1, 4, 5]] = 1
            replay.add((self.x, self.y), other_mask)
            probes = runner.NoiseProbeBuffer(maxlen=1)
            probes.add((self.x, model(self.x).detach()), other_mask)
            with patch.object(runner, 'Config', cfg, create=True), \
                    patch.object(model, '_reconfig', return_value=model.m), \
                    patch.object(reg, 'neuron_decorrelation', wraps=reg.neuron_decorrelation) as calc, \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.train(model, lambda _: (self.x, self.y), [0], 1, 'test',
                             optimizer, replay, save_viz=False)
                self.assertGreaterEqual(calc.call_count, 4)  # current + replay each step
                before = calc.call_count
                runner.noise_probe_step(model, probes)
                self.assertGreater(calc.call_count, before)

    def test_presets(self):
        self.assertEqual(config.Default.DECORRELATION_COEF, 0.)
        presets = [config.DecorrelationWeak, config.DecorrelationMedium,
                   config.DecorrelationStrong, config.DecorrelationVeryStrong]
        self.assertEqual([cls.DECORRELATION_COEF for cls in presets], [.001, .01, .1, 1.])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
