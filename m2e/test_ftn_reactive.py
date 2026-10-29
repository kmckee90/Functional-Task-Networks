"""Reactive_Low scheduling and NoReplay ablation checks for the m2e pod."""
from copy import deepcopy
import io
import math
import unittest
from unittest.mock import patch

import torch

from experiment_ftn import ftn_setting_overrides
from test_agent_ftn import make_agent


class ReactiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def data(self, agent, device="cpu"):
        return dict(o=torch.randn(4, agent.rnn_size_o, device=device),
                    f=torch.randn(4, agent.rnn_size_f, device=device),
                    no=torch.randn(4, agent.rnn_size_o, device=device),
                    nf=torch.randn(4, agent.rnn_size_f, device=device),
                    action=torch.zeros(4, 2, device=device), ro=torch.ones(4, 1, device=device),
                    rf=torch.ones(4, 1, device=device), done=torch.zeros(4, 1, dtype=torch.bool, device=device), goal=False)

    def test_presets_and_no_replay_scope(self):
        values = ftn_setting_overrides("Reactive_Low", True)
        self.assertTrue(values["ftn_reactive"])
        self.assertEqual(values["ftn_reactive_sd"], 5.)
        self.assertEqual(values["ftn_reactive_ema"], .9)
        self.assertEqual(values["ftn_replay_capacity"], 0)
        self.assertNotIn("memory_length", values)
        self.assertNotIn("prioritized_replay", values)
        self.assertNotIn("ftn_reactive", ftn_setting_overrides())
        with self.assertRaises(ValueError):ftn_setting_overrides("unknown")

    def test_initial_search_stability_spike_cooldown_and_checkpoint(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            with self.subTest(device=device):
                agent = make_agent(device, **ftn_setting_overrides("Reactive_Low", True))
                d = self.data(agent, device)
                warmup = math.ceil(1 / (1 - .9))
                with patch.object(agent, "_critic_error", return_value=1.):
                    self.assertTrue(agent._search(d))
                    self.assertEqual(agent.ftn_searches, 1)
                    masks = agent._masks()
                    rng = torch.get_rng_state()
                    for _ in range(warmup):self.assertFalse(agent._search(d))
                    self.assertEqual(agent.ftn_searches, 1)
                    torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
                    for k, mask in masks.items():torch.testing.assert_close(mask, agent.posteriors[k].m, rtol=0, atol=0)
                with patch.object(agent, "_critic_error", side_effect=[100., 1.]):
                    self.assertTrue(agent._search(d))
                self.assertEqual(agent.ftn_searches, 2)
                self.assertEqual(agent.ftn_changepoints, 1)
                self.assertEqual(agent.ftn_reactive_mean, 1.)
                self.assertEqual(agent.ftn_reactive_var, 0.)
                self.assertEqual(agent.ftn_reactive_since, 1)
                with patch.object(agent, "_critic_error", return_value=100.):
                    self.assertFalse(agent._search(d))  # Cooldown suppresses repeated spikes.
                self.assertEqual(agent.ftn_searches, 2)
                saved = io.BytesIO()
                torch.save(agent.state_dict(), saved)
                saved.seek(0)
                restored = make_agent(device, **ftn_setting_overrides("Reactive_Low", True))
                restored.load_state_dict(torch.load(saved, weights_only=False, map_location=device))
                for name in ("ftn_reactive_mean", "ftn_reactive_var", "ftn_reactive_since", "ftn_reactive_checks"):
                    self.assertEqual(getattr(agent, name), getattr(restored, name))

    def test_no_archive_replay_but_weights_still_train(self):
        agent = make_agent(**ftn_setting_overrides("Reactive_Low", True))
        d = self.data(agent)
        before = [p.detach().clone() for p in agent.act_pred.parameters()]
        for _ in range(12):agent._train_support(d)
        self.assertEqual(agent.ftn_updates, 12)
        self.assertEqual(agent.ftn_archive, [])
        self.assertEqual(agent.ftn_replay_updates, 0)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.act_pred.parameters())))


if __name__ == "__main__":unittest.main()
