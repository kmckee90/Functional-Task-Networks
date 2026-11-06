"""Target-free mask inference and support/query protocol regression tests."""
import unittest
from unittest.mock import patch

import torch

if __package__:
    from . import config, run_experiments as runner
    from .bayes_lora_ftn import BayesLoRAFTN
    from .input_conditioned import InputConditionedRouting
else:
    import config
    import run_experiments as runner
    from bayes_lora_ftn import BayesLoRAFTN
    from input_conditioned import InputConditionedRouting


class EntropyReconfigTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.cfg = type('EntropySmoke', (config.EntropyReconfig, config.Smoke), {})()
        self.model = BayesLoRAFTN(2, 2, self.cfg.HIDDEN_SIZE,
            **self.cfg.LORA_FTN_PARS, **self.cfg.BAYES_FTN_PARS)
        self.x = torch.randn(16, 2)

    def test_entropy_formula_and_target_independence(self):
        pred = torch.tensor([[0., 0.], [1., -1.]], requires_grad=True)
        entropy = self.model.mask_search_loss(pred, object(), 'entropy')
        expected = torch.distributions.Categorical(logits=pred).entropy().mean()
        torch.testing.assert_close(entropy, expected)
        entropy.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(pred.grad.abs().sum().item(), 0)
        before = {n: p.clone() for n, p in self.model.named_parameters()}
        state = self.model.get_reconfig_state()
        for labels in (None, object()):
            torch.manual_seed(19)
            self.model.cold_reconfigure(self.x, labels, steps=5,
                                        objective='entropy', take_best=True)
            if labels is None:
                mask = self.model.m.clone()
            else:
                torch.testing.assert_close(mask, self.model.m)
        self.assertEqual(self.model.m.sum().item(), self.cfg.K)
        torch.testing.assert_close(state['mask_logits'], self.model.mask_logits)
        for n, p in self.model.named_parameters():
            torch.testing.assert_close(p, before[n])
            self.assertIsNone(p.grad)
        with torch.no_grad():
            actual = self.model.mask_search_loss(self.model(self.x), objective='entropy').item()
        self.assertAlmostEqual(actual, self.model.last_cold_reconfig_stats['support_loss'], places=6)

    def test_training_entropy_updates_ignore_targets_and_leave_weights_trainable(self):
        self.assertFalse(self.model.train_reconfig_entropy)
        self.cfg.TRAIN_RECONFIG_ENTROPY = True
        model = BayesLoRAFTN(2, 2, self.cfg.HIDDEN_SIZE,
            **self.cfg.LORA_FTN_PARS, **self.cfg.BAYES_FTN_PARS)
        initial = model.get_reconfig_state()
        before = {n: p.clone() for n, p in model.named_parameters()}
        for method in (model.reconfigure, model.reconfigure_changepoint):
            for labels in (None, object()):
                model.set_reconfig_state(initial)
                torch.manual_seed(13)
                method(self.x, labels)
                state = model.get_reconfig_state()
                if labels is None:
                    expected = state
                else:
                    for key in expected:
                        torch.testing.assert_close(state[key], expected[key])
                self.assertEqual(model.last_reconfig_stats['mask_objective'], 'entropy')
                self.assertEqual(model.m.sum().item(), self.cfg.K)
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before[n])
            self.assertIsNone(p.grad)
        labels = torch.nn.functional.one_hot((self.x[:, 0] > 0).long(), 2).float()
        model.loss(model(self.x), labels).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.parameters() if p.requires_grad))

    def test_supervised_training_objective_remains_default(self):
        with self.assertRaisesRegex(ValueError, 'requires support targets'):
            self.model.cold_reconfigure(self.x, steps=1)
        self.model.classification = False
        with self.assertRaisesRegex(ValueError, 'requires classification'):
            self.model.cold_reconfigure(self.x, steps=1, objective='entropy')

    def test_routed_entropy_refinement_does_not_read_labels(self):
        cfg = type('RoutedEntropy', (config.InputConditioned, type(self.cfg)), {})()
        cfg.IC_AE_RANK = 2
        cfg.IC_EVAL_STRATEGY = 'direct'
        cls = type('Routed', (InputConditionedRouting, BayesLoRAFTN), {})
        model = cls(2, 2, cfg.HIDDEN_SIZE, **cfg.LORA_FTN_PARS, **cfg.BAYES_FTN_PARS)
        model.configure_input_routing(cfg)
        model.ic_observe(self.x)
        state = model.get_reconfig_state()
        model.cold_reconfigure(self.x, object(), objective='entropy')
        self.assertEqual(model.ic_stats['eval_hits'], 1)
        self.assertFalse(model.last_cold_reconfig_stats['direct'])
        self.assertEqual(model.last_cold_reconfig_stats['objective'], 'entropy')
        torch.testing.assert_close(model.mask_logits, state['mask_logits'])
        model.ic_cov = None
        model.ic_samples = 0
        model.cold_reconfigure(self.x, object(), steps=2, objective='entropy')
        self.assertEqual(model.ic_stats['eval_misses'], 1)

    def test_support_query_switch_and_label_boundary(self):
        cfg = self.cfg
        cfg.EVAL_RECONFIG_SAMPLE_SIZE = 4
        for same in (True, False):
            cfg.EVAL_SAME_SUPPORT = same
            calls = []
            searches = []
            def generate(task):
                batch = (torch.full((10, 2), float(len(calls))), torch.ones(10, 2))
                calls.append(batch)
                return batch
            class Spy:
                def cold_reconfigure(self, x, y, **kwargs):
                    searches.append(x)
                    if y is not None:
                        raise AssertionError('Targets leaked to entropy search')
                    if kwargs['objective'] != 'entropy':
                        raise AssertionError('Wrong mask objective')
            with patch.object(runner, 'Config', cfg, create=True):
                x, y = runner.evaluation_trial(Spy(), generate, 0)
            self.assertEqual(len(calls), 1 if same else 2)
            self.assertEqual(searches[0].shape[0], 4)
            self.assertEqual(x.shape[0], 4 if same else 10)
            self.assertEqual(y.shape[0], x.shape[0])
            if same:
                self.assertIs(x, searches[0])
            else:
                self.assertIs(x, calls[1][0])


if __name__ == '__main__':
    unittest.main()
