"""Display-layout regression checks; run with python -m unittest on the pod."""

import tempfile
import unittest

import numpy as np
import torch

from config import Default, SpatialPermute, SpatialPermuteRandom
from ftn import mask_to_grid
from spatial_permute import spatial_permutation
from viz import MaskAnimator, task_color


def adjacency(masks, order, shape):
    grids = masks[:, order].reshape(-1, *shape).astype(np.int64)
    return ((grids[:, 1:] * grids[:, :-1]).sum()
            + (grids[:, :, 1:] * grids[:, :, :-1]).sum())


class SpatialPermutationTests(unittest.TestCase):
    def test_random_blobs_and_continued_optimization(self):
        masks = np.random.default_rng(7).random((4, 128)) < .25
        options = dict(initialization='random_blobs', seed=19,
                       roundness=0, center=0, temperature=0)
        state = np.random.get_state()
        initial = spatial_permutation(masks[:1], (8, 16), sweeps=0, **options)
        np.testing.assert_array_equal(initial, spatial_permutation(
            masks[:1], (8, 16), sweeps=0, **options))
        other = spatial_permutation(masks[:1], (8, 16), sweeps=0,
                                    **dict(options, seed=20))
        self.assertFalse(np.array_equal(initial, other))
        self.assertGreater(adjacency(masks[:1], initial, (8, 16)),
                           adjacency(masks[:1], np.arange(128), (8, 16)))
        # New tasks must not silently replace the existing layout with a seed.
        np.testing.assert_array_equal(initial, spatial_permutation(
            masks, (8, 16), initial, sweeps=0, **options))
        refined = spatial_permutation(masks, (8, 16), initial, **options)
        self.assertGreater(adjacency(masks, refined, (8, 16)),
                           adjacency(masks, initial, (8, 16)))
        np.testing.assert_array_equal(np.sort(refined), np.arange(128))
        values = masks.astype(int)
        np.testing.assert_array_equal(values[:, refined] @ values[:, refined].T,
                                      values @ values.T)
        after = np.random.get_state()
        for a, b in zip(state, after):
            np.testing.assert_equal(a, b)

    def test_random_animator_starts_once(self):
        with tempfile.TemporaryDirectory() as out_dir:
            options = dict(initialization='random_blobs', sweeps=0, seed=12, center=0)
            animator = MaskAnimator(64, 8, 20, 1, out_dir=out_dir,
                                    spatial_permute=True, spatial_options=options)
            mask = (torch.arange(64) % 3 == 0).float()
            animator.record_task(0, mask, 'random')
            order = animator.permutation.copy()
            self.assertFalse(np.array_equal(order, np.arange(64)))
            animator.record_task(1, 1 - mask, 'random')
            np.testing.assert_array_equal(order, animator.permutation)
        self.assertTrue(SpatialPermuteRandom.SPATIAL_PERMUTE)
        self.assertEqual(SpatialPermuteRandom.SPATIAL_PERMUTE_INIT, 'random_blobs')
        self.assertEqual(SpatialPermuteRandom.SPATIAL_PERMUTE_CENTER, 0)

    def test_consolidation_and_invariants(self):
        rng = np.random.default_rng(12)
        masks = rng.random((7, 128)) < .3
        original = masks.copy()
        previous = rng.permutation(128)
        options = dict(roundness=0, center=0, temperature=0)
        order = spatial_permutation(masks, (8, 16), previous, **options)
        np.testing.assert_array_equal(np.sort(order), np.arange(128))
        np.testing.assert_array_equal(masks, original)
        np.testing.assert_array_equal(masks[:, order].sum(1), masks.sum(1))
        values = masks.astype(np.int64)
        np.testing.assert_array_equal(values[:, order] @ values[:, order].T,
                                      values @ values.T)
        self.assertGreater(adjacency(masks, order, (8, 16)),
                           adjacency(masks, np.arange(128), (8, 16)))
        self.assertGreaterEqual(adjacency(masks, order, (8, 16)),
                                adjacency(masks, previous, (8, 16)))
        np.testing.assert_array_equal(
            order, spatial_permutation(masks, (8, 16), previous, **options))

    def test_blob_objective_and_all_intersections(self):
        rng = np.random.default_rng(33)
        masks = rng.random((4, 80)) < .25
        previous = rng.permutation(80)
        # Independently evaluate all pixel pairs, including boundary effects.
        y, x = np.indices((8, 10), dtype=float)
        xy = np.stack((y.ravel(), x.ravel()), axis=1)
        distances = ((xy[:, None] - xy[None, :])**2).sum(2)
        kernel = np.exp(-2 * distances / 9) * ((distances > 0) & (distances <= 9))
        normalizer = sum(np.exp(-2 * (dy*dy + dx*dx) / 9)
                         for dy in range(-3, 4) for dx in range(-3, 4)
                         if 0 < dy*dy + dx*dx <= 9)
        weights = (distances == 1) + 2 * 4 * kernel / normalizer
        center_cost = ((xy - np.array([3.5, 4.5]))**2).sum(1)
        center_cost *= .4 / center_cost.max()

        def score(order):
            values = masks[:, order].astype(float)
            return (weights * (values.T @ values)).sum() / (2 * len(masks)) - center_cost @ values.any(0)

        seed = spatial_permutation(masks, (8, 10), previous, sweeps=0)
        order = spatial_permutation(masks, (8, 10), previous)
        self.assertGreaterEqual(score(order) + 1e-8, score(seed))
        self.assertGreaterEqual(score(order) + 1e-8, score(previous))
        self.assertGreater(score(order), score(np.arange(80)))
        np.testing.assert_array_equal(np.sort(order), np.arange(80))
        # The full signature histogram proves all higher-order overlaps survive.
        np.testing.assert_array_equal(
            np.unique(masks.T, axis=0, return_counts=True)[1],
            np.unique(masks[:, order].T, axis=0, return_counts=True)[1])
        np.testing.assert_array_equal(order, spatial_permutation(masks, (8, 10), previous))

    def test_single_task_is_central_and_compact(self):
        mask = np.zeros((1, 256), dtype=bool)
        mask[0, np.random.default_rng(4).choice(256, 48, replace=False)] = True
        order = spatial_permutation(mask, (16, 16))
        y, x = np.indices((16, 16))
        distance = ((x - 7.5)**2 + (y - 7.5)**2).ravel()
        self.assertLess(distance[mask[0, order]].mean(), distance[mask[0]].mean() * .4)
        self.assertGreater(adjacency(mask, order, (16, 16)),
                           adjacency(mask, np.arange(256), (16, 16)))

    def test_invalid_controls(self):
        for options in [dict(radius=0), dict(sweeps=-1), dict(center=-1),
                        dict(roundness=float('nan')), dict(temperature=-.1)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                spatial_permutation(np.ones((1, 16)), (4, 4), **options)

    def test_degenerate_masks(self):
        for masks, shape in [(np.zeros((2, 16)), (4, 4)),
                             (np.ones((2, 16)), (4, 4)),
                             (np.ones((1, 1)), (1, 1)),
                             (np.zeros((0, 16)), (4, 4))]:
            order = spatial_permutation(masks, shape)
            np.testing.assert_array_equal(np.sort(order), np.arange(masks.shape[1]))

    def test_animator_mapping_ema_and_outputs(self):
        # Two side-by-side blocks catch flat-vector vs raster indexing errors.
        with tempfile.TemporaryDirectory() as out_dir:
            animator = MaskAnimator(32, 4, 20, 1, out_dir=out_dir,
                                    spatial_permute=True)
            mask = (torch.arange(32) % 3 == 0).float()
            original = mask.clone()
            state = torch.random.get_rng_state().clone()
            animator.record_step(mask)
            ema = animator.m_ema.clone()
            animator.record_task(0, mask, "test")
            self.assertTrue(torch.equal(mask, original))
            self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
            self.assertTrue(torch.equal(ema, animator.m_ema))
            order = animator.permutation.copy()
            expected = mask_to_grid(mask, 4).numpy().ravel()[order].reshape(4, 8)
            rgb = np.asarray(animator.rgb_frames[-1])
            np.testing.assert_array_equal(rgb, expected[..., None] * task_color(0))
            np.testing.assert_array_equal(
                np.asarray(animator.frames[-1]),
                (ema.numpy().reshape(-1, 3)[order].reshape(4, 8, 3) * 255).astype(np.uint8))
            animator.record_step(1 - mask)
            np.testing.assert_array_equal(order, animator.permutation)
            old_frames = len(animator.frames)
            animator.record_task(1, 1 - mask, "test", snapshot=False)
            self.assertEqual(len(animator.frames), old_frames)
            animator.record_task(0, 1 - mask, "test", snapshot=False)
            self.assertEqual(len(animator.task_masks), 2)
            animator.maybe_save_gifs(0, "test")

    def test_disabled_and_config_composition(self):
        self.assertFalse(Default.SPATIAL_PERMUTE)
        combined = type("Combined", (SpatialPermute, Default), {})()
        self.assertTrue(combined.SPATIAL_PERMUTE)
        animator = MaskAnimator(16, 4, 20, 1)
        mask = (torch.arange(16) % 2).float()
        animator.record_step(mask)
        animator.record_task(0, mask, "", snapshot=False)
        np.testing.assert_array_equal(animator.permutation, np.arange(16))
        expected = np.repeat(mask.numpy().reshape(4, 4, 1), 3, axis=2) * 255
        np.testing.assert_array_equal(np.asarray(animator.frames[-1]), expected)
        self.assertEqual(len(animator.frames), 3)


if __name__ == "__main__":
    unittest.main()
