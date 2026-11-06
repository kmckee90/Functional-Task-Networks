"""Regression tests and small end-to-end artifact checks (CPU sufficient)."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

if __package__:
    from . import config
    from .bayes_lora_ftn import BayesLoRAFTN
    from .input_conditioned import InputConditionedRouting
    from .viz import MaskAnimator, task_color
else:
    import config
    from bayes_lora_ftn import BayesLoRAFTN
    from input_conditioned import InputConditionedRouting
    from viz import MaskAnimator, task_color


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.cfg = config.Smoke()
        self.model = BayesLoRAFTN(2, 2, self.cfg.HIDDEN_SIZE,
                                 **self.cfg.LORA_FTN_PARS, **self.cfg.BAYES_FTN_PARS)
        self.x = torch.randn(16, 2)
        self.y = torch.nn.functional.one_hot((self.x[:, 0] > 0).long(), 2).float()

    def test_search_detaches_atoms_and_training_freezes_backbone(self):
        model = self.model
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        model.reconfigure_changepoint(self.x, self.y)
        self.assertEqual(model.m.sum().item(), self.cfg.K)
        self.assertTrue(torch.isfinite(model.mask_logits).all())
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before[n])
            self.assertIsNone(p.grad)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.01)
        model.loss(model(self.x), self.y).backward()
        opt.step()
        changed = False
        for n, p in model.named_parameters():
            if '.base.' in n:
                self.assertFalse(p.requires_grad)
                torch.testing.assert_close(p, before[n])
            else:
                changed |= not torch.equal(p, before[n])
        self.assertTrue(changed)

    def test_cold_eval_leaves_posterior_and_can_restore_mask(self):
        model = self.model
        model.reconfigure(self.x, self.y)
        state = model.get_reconfig_state()
        model.cold_reconfigure(self.x, self.y, steps=2)
        torch.testing.assert_close(model.mask_logits, state['mask_logits'])
        self.assertEqual(model.m.sum().item(), self.cfg.K)
        model.set_reconfig_state(state)
        torch.testing.assert_close(model.m, state['m'])

    def test_input_router_recognizes_disjoint_subspaces(self):
        cfg = type('Routing', (config.InputConditioned, config.Smoke), {})()
        cfg.IC_AE_RANK = 1
        Routed = type('Routed', (InputConditionedRouting, BayesLoRAFTN), {})
        model = Routed(2, 2, cfg.HIDDEN_SIZE, **cfg.LORA_FTN_PARS, **cfg.BAYES_FTN_PARS)
        model.configure_input_routing(cfg)
        x = torch.zeros(16, 2)
        x[:, 0] = torch.arange(1, 17)
        model.ic_observe(x)
        model._ic_archive()
        entry, error = model._ic_match(x)
        self.assertIsNotNone(entry)
        self.assertLess(error, 1e-6)
        entry, _ = model._ic_match(x.flip(1))
        self.assertIsNone(entry)


class LayoutTests(unittest.TestCase):
    def test_square_display_preserves_masks_rng_and_intersections(self):
        cfg = config.Default()
        self.assertTrue(cfg.SPATIAL_PERMUTE)
        self.assertEqual(cfg.SPATIAL_PERMUTE_INIT, 'random_blobs')
        self.assertEqual(cfg.SPATIAL_PERMUTE_CENTER, .25)
        with tempfile.TemporaryDirectory() as root:
            animator = MaskAnimator(64, 8, 20, 1, out_dir=root, spatial_permute=True,
                spatial_options=dict(initialization=cfg.SPATIAL_PERMUTE_INIT,
                                     center=cfg.SPATIAL_PERMUTE_CENTER, sweeps=1))
            masks = (torch.arange(64)[None, :] % torch.tensor([2, 3, 5])[:, None] == 0).float()
            original = masks.clone()
            state = torch.random.get_rng_state().clone()
            for t, mask in enumerate(masks):
                animator.record_step(mask)
                animator.record_task(t, mask, 'test')
            torch.testing.assert_close(masks, original)
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            order = animator.permutation
            np.testing.assert_array_equal(np.sort(order), np.arange(64))
            displayed = masks[:, order]
            torch.testing.assert_close(displayed @ displayed.T, masks @ masks.T)
            colors = torch.tensor([task_color(t) for t in range(3)])
            expected = (displayed.T @ colors.float()) / displayed.sum(0).clamp_min(1)[:, None]
            np.testing.assert_array_equal(np.asarray(animator.rgb_frames[-1]), expected.byte().numpy().reshape(8, 8, 3))
            self.assertEqual(animator.frames[-1].size, (8, 8))
            # Zero sweeps on subsequent blocks must retain the previous layout.
            animator.spatial_options['sweeps'] = 0
            animator.record_task(0, masks[0], 'test')
            np.testing.assert_array_equal(animator.permutation, order)


class ArtifactTests(unittest.TestCase):
    def run_smoke(self, extra, metrics_only=False, same_support=False, train_entropy=False):
        with tempfile.TemporaryDirectory() as root:
            cmd = [sys.executable, str(Path(__file__).with_name('run_experiments.py')),
                   '--config', 'Smoke', *extra, '--name', 'check', '--out-root', root]
            if train_entropy:
                cmd += ["--train-reconfig-entropy", "1"]
            if same_support:
                cmd += ["--eval-same-support", "1"]
            if metrics_only:
                cmd.append('--metrics-only')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg',
                       OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = Path(root) / 'check'
            expected = {'config_check.txt', 'metrics_check.txt'}
            if not metrics_only:
                expected |= {f'{p}_check.png' for p in ('rgb', 'loss', 'task_vis', 'xfer',
                                                       'entropy_acc', 'entropy_std', 'entropy_share')}
                expected |= {'bw_check.gif', 'rgb_check.gif'}
            self.assertEqual({p.name for p in output.iterdir()}, expected)
            if not metrics_only:
                for filename in ['bw_check.gif', 'rgb_check.gif', 'rgb_check.png']:
                    with Image.open(output / filename) as image:
                        self.assertEqual(image.size, (8, 8))
            if 'InputConditioned' in extra:
                metrics = (output / 'metrics_check.txt').read_text()
                fields = dict(line.split(' = ', 1) for line in metrics.splitlines()
                              if ' = ' in line and not line.startswith('#'))
                self.assertGreater(int(fields['ic_eval_hits']), 0)
                self.assertGreater(int(fields['ic_repair_attempts']), 0)
            saved_config = (output / 'config_check.txt').read_text()
            self.assertIn(f'TRAIN_RECONFIG_ENTROPY = {train_entropy}', saved_config)
            self.assertIn(f'train_reconfig_entropy = {train_entropy}',
                          (output / 'metrics_check.txt').read_text())
            self.assertNotIn('RECONFIG_OVERLAP_', saved_config)

    def test_complete_artifacts(self):
        self.run_smoke([])

    def test_input_conditioned_artifacts(self):
        self.run_smoke(['DomainShift', 'NoConceptShift', 'InputConditioned', 'InputConditionedSmoke'])

    def test_entropy_reconfig_separate_artifacts(self):
        self.run_smoke(['EntropyReconfig'])

    def test_entropy_reconfig_routed_same_support_artifacts(self):
        self.run_smoke(['DomainShift', 'NoConceptShift', 'InputConditioned',
                        'InputConditionedSmoke', 'EntropyReconfig'], same_support=True)

    def test_training_entropy_scheduled(self):
        self.run_smoke(['PerTask'], train_entropy=True)

    def test_training_entropy_routed_reactive(self):
        self.run_smoke(['DomainShift', 'NoConceptShift', 'InputConditioned',
                        'InputConditionedSmoke', 'EntropyReconfig'], train_entropy=True)

    def test_metrics_only(self):
        self.run_smoke(['PerTask'], metrics_only=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
