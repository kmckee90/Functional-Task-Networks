"""Run on the m2e pod: uv run python -m unittest test_ftn_diagnostics -v."""
from copy import deepcopy
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from envs.continuous_maze import ContinuousMaze
from ftn_diagnostics import FTNRouteDiagnostic, BANKS, isolated_rng, spatial_permutation
from test_agent_ftn import make_agent


def assert_tree_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for k in left:
            assert_tree_equal(test, left[k], right[k])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


def install(agent, actor, critic=None):
    masks = {k: torch.zeros_like(getattr(agent, agent._names(k)[0]).mask) for k in BANKS}
    for k in BANKS:
        start = actor if k == "a" or critic is None else critic
        masks[k][start:start + 2] = 1
    agent._install(masks)
    agent.ftn_searches += 1


def record_episode(diagnostic, agent, env, number):
    diagnostic.begin_episode(number, env)
    diagnostic.record_step(agent)
    diagnostic.end_episode()


class FTNDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_most_used_actor_and_within_episode_changes(self):
        agent = make_agent()
        env = ContinuousMaze(size=13, old_walls=True, max_steps=3)
        env.reset()
        diagnostic = FTNRouteDiagnostic("unused", 4., lambda x: x)
        diagnostic.begin_episode(1, env)
        for actor, critic, steps in [(0, 0, 2), (2, 2, 3), (0, 4, 3)]:
            install(agent, actor, critic)
            for _ in range(steps):
                diagnostic.record_step(agent)
        diagnostic.end_episode()
        episode = diagnostic.pending[0]
        self.assertEqual(episode["steps"], 8)
        self.assertEqual(episode["representative_steps"], 5)
        self.assertEqual(len(episode["segments"]), 3)
        self.assertEqual(episode["mask_ids"]["a"], 0)
        self.assertEqual(episode["mask_ids"]["o"], 2)
        self.assertEqual(len(diagnostic.masks["a"]), 2)
        np.testing.assert_array_equal(env.pos, episode["start_pos"])

    def test_cumulative_layout_refines_previous_with_padding(self):
        agent = make_agent(ftn_hidden_size=7)
        env = ContinuousMaze(size=13, old_walls=True, max_steps=2)
        env.reset()
        diagnostic = FTNRouteDiagnostic("unused", 4., lambda x: x, block_size=1, spatial_sweeps=1)
        install(agent, 0)
        record_episode(diagnostic, agent, env, 1)
        diagnostic.consolidate()
        previous = {k: v.copy() for k, v in diagnostic.layouts.items()}
        diagnostic.pending.clear()
        install(agent, 2)
        record_episode(diagnostic, agent, env, 2)
        with patch("ftn_diagnostics.spatial_permutation", wraps=spatial_permutation) as permute:
            diagnostic.consolidate()
        for k, call in zip(BANKS, permute.call_args_list):
            np.testing.assert_array_equal(call.kwargs["previous"], previous[k])
            self.assertEqual(call.args[0].shape, (2, 8))
            self.assertEqual(call.kwargs["initialization"], "random_blobs")
            self.assertEqual(call.kwargs["center"], .25)
            order = diagnostic.layouts[k]
            np.testing.assert_array_equal(np.sort(order), np.arange(8))
            masks = np.pad(np.stack(diagnostic.masks[k]), ((0, 0), (0, 1))).astype(int)
            np.testing.assert_array_equal(masks[:, order] @ masks[:, order].T, masks @ masks.T)
        with patch("ftn_diagnostics.spatial_permutation") as permute:
            diagnostic.consolidate()
            permute.assert_not_called()

    def test_full_grid_and_live_training_state_isolation(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as out:
                agent = make_agent(device, ftn_action_reconfig_interval=0)
                agent.epsilon = 0.
                models = (agent.state_model_o, agent.state_model_f,
                          agent.state_model_o_targ, agent.state_model_f_targ)
                for model in models:
                    model.hidden.copy_(torch.rand_like(model.hidden))
                agent.density_estimator.update(torch.rand(8, 2, device=device))
                env = ContinuousMaze(size=13, old_walls=True, max_steps=3)
                env.reset()
                env.pos = np.array([1., 1.])
                env_before = (env.walls.copy(), env.pos.copy(), list(env._all_pos), env._cur_step)
                artifact_logger = unittest.mock.Mock()
                diagnostic = FTNRouteDiagnostic(out, 4., lambda x: x, spatial_sweeps=1,
                                                artifact_logger=artifact_logger)
                # Twenty source episodes, with repeated masks deliberately kept
                # as separate cells; all 32 diagnostic episodes run to completion.
                for number in range(1, 21):
                    install(agent, 2 * (number % 3))
                    record_episode(diagnostic, agent, env, number)
                self.assertTrue(diagnostic.ready)
                before = deepcopy(agent.state_dict())
                optimizers = {k: deepcopy(getattr(agent, "optimizer_" + k).state_dict()) for k in BANKS}
                baseline = deepcopy(agent)
                flags = [module.training for module in agent.modules()]
                cpu_rng = torch.random.get_rng_state().clone()
                cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                numpy_rng, python_rng = np.random.get_state(), random.getstate()
                with patch.object(agent, "forward", side_effect=AssertionError("Live agent used")), \
                     patch.object(agent, "_search", side_effect=AssertionError("Live mask search used")):
                    paths = diagnostic.evaluate(agent, max_reward=1.)
                self.assertFalse(diagnostic.pending)
                assert_tree_equal(self, before, agent.state_dict())
                for k, state in optimizers.items():
                    assert_tree_equal(self, state, getattr(agent, "optimizer_" + k).state_dict())
                self.assertEqual(flags, [module.training for module in agent.modules()])
                assert_tree_equal(self, cpu_rng, torch.random.get_rng_state())
                assert_tree_equal(self, cuda_rng, torch.cuda.get_rng_state_all() if cuda_rng else [])
                assert_tree_equal(self, numpy_rng, np.random.get_state())
                self.assertEqual(python_rng, random.getstate())
                assert_tree_equal(self, env_before, (env.walls, env.pos, env._all_pos, env._cur_step))
                self.assertEqual(artifact_logger.call_count, 1)
                self.assertEqual(artifact_logger.call_args.args[0].suffix, ".png")
                self.assertEqual([p.suffix for p in paths], [".png", ".npz", ".json"])
                with Image.open(paths[0]) as image:
                    self.assertEqual(image.size, (3000, 1920))
                metadata = json.loads(paths[2].read_text())
                self.assertEqual(len(metadata["episodes"]), 20)
                self.assertEqual(metadata["rollouts_per_mask"], 32)
                with np.load(paths[1], allow_pickle=False) as data:
                    for ep in range(1, 21):
                        for r in range(32):
                            visits = data[f"visits_{ep}_{r}"]
                            self.assertEqual(visits.shape, (4, 2))
                            np.testing.assert_array_equal(visits[0], env.pos)
                # The next live action and reservoir states match a control
                # that never ran diagnostics, proving continuity on resumption.
                obs = torch.ones(2, device=device)
                feedback = torch.ones(agent.feedback_size, device=device)
                with torch.no_grad(), isolated_rng():
                    expected = baseline(obs, feedback)
                with torch.no_grad():
                    actual = agent(obs, feedback)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for name in ("state_model_o", "state_model_f"):
                    torch.testing.assert_close(getattr(agent, name).hidden,
                                               getattr(baseline, name).hidden, rtol=0, atol=0)

    def test_rng_restored_on_evaluation_failure(self):
        agent = make_agent()
        env = ContinuousMaze(size=13, old_walls=True)
        env.reset()
        diagnostic = FTNRouteDiagnostic("unused", 4., lambda x: x, block_size=1)
        record_episode(diagnostic, agent, env, 1)
        before = deepcopy(agent.state_dict())
        cpu_rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()

        def fail(*args):
            random.random()
            np.random.rand()
            torch.rand(1)
            raise RuntimeError("intentional diagnostic failure")

        with patch.object(diagnostic, "consolidate"), patch.object(diagnostic, "_rollout", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "intentional"):
                diagnostic.evaluate(agent, 1.)
        assert_tree_equal(self, before, agent.state_dict())
        assert_tree_equal(self, cpu_rng, torch.get_rng_state())
        assert_tree_equal(self, numpy_rng, np.random.get_state())
        self.assertEqual(python_rng, random.getstate())

    def test_experiment_logging_schedule_and_training_continuation(self):
        import experiment_ftn as experiment
        from agents.agent_ftn import RDDPG_Dual

        small = replace(experiment.Hyperparameters(), ftn_k=2, ftn_hidden_size=8,
                        ftn_width=3, ftn_depth=2, hidden_size=8, minibatch_size=8,
                        memory_length=64, goal_memory_length=64, sample_memory_length=16,
                        n_optimizer_steps=1, n_optimizer_epochs=1, ftn_support_size=4)
        results = []
        with tempfile.TemporaryDirectory() as out:
            for enabled in (False, True):
                made_agents = []

                def factory(*args, **kwargs):
                    agent = RDDPG_Dual(*args, **kwargs)
                    made_agents.append(agent)
                    return agent

                args = SimpleNamespace(seed=123, render=False, log=True, run_name="integration",
                                       experiment_name="diagnostic-test", notes="", algo="RDDPG_Dual",
                                       task="Fixed", device="cpu", no_FCP=False, no_OCP=False,
                                       load=None, gif=None, save=False, no_ftn_diagnostics=not enabled,
                                       ftn_diagnostic_dir=out, ftn_diagnostic_sync=True)
                with patch.object(experiment, "NUM_EPISODES", 20), \
                     patch.object(experiment, "Hyperparameters", return_value=small), \
                     patch.object(experiment, "Config", return_value=replace(experiment.Config(), gather_interval=0)), \
                     patch.object(experiment, "LGE_Maze", return_value=replace(experiment.LGE_Maze(), max_steps=4)), \
                     patch("agents.agent_ftn.RDDPG_Dual", side_effect=factory), \
                     patch.object(experiment.mlflow, "set_tracking_uri"), \
                     patch.object(experiment.mlflow, "set_experiment"), \
                     patch.object(experiment.mlflow, "start_run", return_value=nullcontext()), \
                     patch.object(experiment.mlflow, "active_run", return_value=SimpleNamespace(info=SimpleNamespace(run_id="integration"))), \
                     patch.object(experiment.mlflow, "set_tag"), \
                     patch.object(experiment.mlflow, "log_params"), \
                     patch.object(experiment.mlflow, "log_metric"), \
                     patch.object(experiment.mlflow, "log_artifact") as log_artifact, \
                     patch.object(experiment, "save_image"), \
                     patch.object(experiment, "save_density_estimator"):
                    experiment.main(args)
                results.append(deepcopy(made_agents[0].state_dict()))
                self.assertEqual(log_artifact.call_count, 1 if enabled else 0)
                for call in log_artifact.call_args_list:
                    self.assertEqual(Path(call.args[0]).suffix, ".png")
                    self.assertEqual(call.kwargs["artifact_path"], "ftn_diagnostics")
                    self.assertTrue(Path(call.args[0]).is_file())
            # Includes the training episode AFTER the diagnostic, with real
            # Bayesian searches, replay, density updates and gradient steps.
            assert_tree_equal(self, results[0], results[1])
            metadata = json.loads((Path(out) / "integration/episodes_0001_0020.json").read_text())
            self.assertEqual([ep["episode"] for ep in metadata["episodes"]], list(range(1, 21)))
            self.assertEqual(metadata["rollouts_per_mask"], 32)


if __name__ == "__main__":
    unittest.main()
