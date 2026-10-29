"""Spawned CPU/CUDA worker regression tests; run only on the m2e pod."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from envs.continuous_maze import ContinuousMaze
from ftn_diagnostic_worker import AsyncFTNRouteDiagnostic
from ftn_diagnostics import FTNRouteDiagnostic
from test_agent_ftn import make_agent
from test_ftn_diagnostics import assert_tree_equal, install, record_episode


class FTNAsyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_worker_uploads_only_png(self):
        from ftn_diagnostic_worker import _evaluate_job
        agent = make_agent()
        env = ContinuousMaze(size=13, old_walls=True, max_steps=1)
        env.reset()
        with tempfile.TemporaryDirectory() as out:
            diagnostic = FTNRouteDiagnostic(out, 4., lambda x: x,
                                            block_size=1, rollouts=1, spatial_sweeps=0)
            record_episode(diagnostic, agent, env, 1)
            job = dict(output_dir=out, obs_scale=4., normalization="identity",
                       block_size=1, rollouts=1, seed=123, spatial_sweeps=0,
                       episodes=diagnostic.pending, masks=diagnostic.masks,
                       policy=(deepcopy(agent.act_pred), deepcopy(agent.state_model_o),
                               deepcopy(agent.state_model_f), deepcopy(agent.density_estimator)),
                       cfg=agent.config, max_reward=1., training_step=100, device="cpu",
                       run_id="png-only-test", tracking_uri="databricks")
            with patch("ftn_diagnostic_worker._worker_diagnostic", None), \
                 patch("mlflow.MlflowClient") as client:
                result = _evaluate_job(job)
            upload = client.return_value.log_artifact
            upload.assert_called_once_with("png-only-test", str(result["paths"][0]),
                                           artifact_path="ftn_diagnostics")
            self.assertEqual([path.suffix for path in result["paths"]], [".png", ".npz", ".json"])
            self.assertTrue(all(path.is_file() for path in result["paths"]))

    def test_frozen_snapshot_same_device_ordered_layouts_and_bounded_queue(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as out:
                agent = make_agent(device)
                env = ContinuousMaze(size=13, old_walls=True, max_steps=3)
                env.reset()
                options = dict(block_size=1, rollouts=2, spatial_sweeps=1)
                sync = FTNRouteDiagnostic(Path(out) / "sync", 4., lambda x: x, **options)
                logged = []
                with AsyncFTNRouteDiagnostic(Path(out) / "async", 4., lambda x: x,
                                            artifact_logger=logged.append, max_pending=2, **options) as worker:
                    for episode in range(1, 4):
                        with torch.no_grad():
                            for p in agent.act_pred.parameters():p.fill_(float(episode - 1))
                        install(agent, 2 * (episode - 1))
                        agent.cur_step = episode * 100
                        for diagnostic in (sync, worker):
                            record_episode(diagnostic, agent, env, episode)
                        sync.evaluate(agent, 1.)
                        state = deepcopy(agent.state_dict())
                        cpu_rng, py_rng, np_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
                        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                        worker.submit(agent, 1.)
                        self.assertLessEqual(len(worker._futures), 2)
                        self.assertFalse(worker.pending)
                        assert_tree_equal(self, state, agent.state_dict())
                        assert_tree_equal(self, cpu_rng, torch.get_rng_state())
                        assert_tree_equal(self, np_rng, np.random.get_state())
                        self.assertEqual(py_rng, random.getstate())
                        assert_tree_equal(self, cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else [])
                    # Mutations happen before queued jobs finish: their results
                    # must still match the synchronous block-boundary snapshots.
                    with torch.no_grad():
                        for p in agent.act_pred.parameters():p.fill_(-100.)
                self.assertEqual(len(worker.completed), 3)
                self.assertEqual(len(logged), 3)
                self.assertTrue(all(path.suffix == ".png" for path in logged))
                worker_pids = set()
                for episode in range(1, 4):
                    stem = f"episodes_{episode:04d}_{episode:04d}"
                    metadata = json.loads((Path(out) / "async" / f"{stem}.json").read_text())
                    self.assertEqual(metadata["training_step"], episode * 100)
                    self.assertEqual(metadata["execution"]["mode"], "asynchronous")
                    self.assertEqual(metadata["execution"]["device"], str(agent.act_pred.mask.device))
                    worker_pids.add(metadata["execution"]["worker_pid"])
                    with np.load(Path(out) / "sync" / f"{stem}.npz") as expected, \
                         np.load(Path(out) / "async" / f"{stem}.npz") as actual:
                        self.assertEqual(expected.files, actual.files)
                        for key in expected.files:
                            np.testing.assert_allclose(actual[key], expected[key], rtol=0, atol=1e-6)
                self.assertEqual(len(worker_pids), 1)
                self.assertNotIn(os.getpid(), worker_pids)

    def test_worker_failure_propagates_on_drain(self):
        agent = make_agent()
        agent.config.pop("extra_zeros")
        env = ContinuousMaze(size=13, old_walls=True, max_steps=1)
        env.reset()
        with tempfile.TemporaryDirectory() as out:
            worker = AsyncFTNRouteDiagnostic(out, 4., lambda x: x,
                                            block_size=1, rollouts=1, spatial_sweeps=0)
            record_episode(worker, agent, env, 1)
            worker.submit(agent, 1.)
            with self.assertRaisesRegex(KeyError, "extra_zeros"):
                worker.close()
            self.assertTrue(worker._closed)
            worker.close()  # Idempotent, no orphaned executor.

    def test_async_experiment_preserves_training_continuation(self):
        import experiment_ftn as experiment
        from agents.agent_ftn import RDDPG_Dual
        small = replace(experiment.Hyperparameters(), ftn_k=2, ftn_hidden_size=8,
                        ftn_width=3, ftn_depth=2, hidden_size=8, minibatch_size=8,
                        memory_length=64, goal_memory_length=64, sample_memory_length=16,
                        n_optimizer_steps=1, n_optimizer_epochs=1, ftn_support_size=4)
        states = []
        with tempfile.TemporaryDirectory() as out:
            for enabled in (False, True):
                made = []

                def factory(*args, **kwargs):
                    agent = RDDPG_Dual(*args, **kwargs)
                    made.append(agent)
                    return agent

                args = SimpleNamespace(seed=123, render=False, log=False, run_name="async-integration",
                                       notes="", algo="RDDPG_Dual", task="Fixed", device="cpu",
                                       no_FCP=False, no_OCP=False, load=None, gif=None, save=False,
                                       no_ftn_diagnostics=not enabled, ftn_diagnostic_dir=out)
                with patch.object(experiment, "NUM_EPISODES", 20), \
                     patch.object(experiment, "Hyperparameters", return_value=small), \
                     patch.object(experiment, "Config", return_value=replace(experiment.Config(), gather_interval=0)), \
                     patch.object(experiment, "LGE_Maze", return_value=replace(experiment.LGE_Maze(), max_steps=3)), \
                     patch("agents.agent_ftn.RDDPG_Dual", side_effect=factory):
                    experiment.main(args)
                states.append(deepcopy(made[0].state_dict()))
            assert_tree_equal(self, *states)
            metadata = json.loads((Path(out) / "async-integration/episodes_0001_0020.json").read_text())
            self.assertEqual(metadata["execution"]["mode"], "asynchronous")
            self.assertEqual(metadata["rollouts_per_mask"], 32)


if __name__ == "__main__":unittest.main()
