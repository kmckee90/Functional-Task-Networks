import unittest
from dataclasses import asdict
from unittest.mock import patch

import torch
from torch import nn

from agents.agent_recode import RDDPG_Dual
from experiment_recode import Config, Hyperparameters


class GoalReturnTests(unittest.TestCase):
    def test_returns_across_terminals_and_truncated_batch(self):
        agent = RDDPG_Dual.__new__(RDDPG_Dual)
        nn.Module.__init__(agent)
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            reward = torch.tensor([[1.], [2.], [3.], [4.]], device=device, requires_grad=True)
            done = torch.tensor([[False], [True], [False], [False]], device=device)
            for gamma, expected in [(0., [1., 2., 3., 4.]),
                                    (0.5, [2., 2., 5., 4.]),
                                    (1., [3., 2., 7., 4.])]:
                with self.subTest(device=device, gamma=gamma):
                    agent.config = {"returns_gamma": gamma}
                    returns = agent._empirical_returns(reward, done)
                    torch.testing.assert_close(returns, reward.new_tensor(expected))
                    self.assertFalse(returns.requires_grad)
            single = agent._empirical_returns(reward[:1], done[:1])
            torch.testing.assert_close(single, reward.detach()[:1, 0])

    def test_invalid_hyperparameters(self):
        for name in ("a", "returns_gamma"):
            for value in (-0.1, 1.1, float("nan")):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        RDDPG_Dual(2, 2, (torch.zeros(2), torch.ones(2)), **{name: value})

    def test_replay_blends_only_goal_ocp_targets(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for a in (0., 0.25, 1.):
                with self.subTest(device=device, a=a):
                    self.check_replay(device, a)

    def check_replay(self, device, a):
        cfg = {**asdict(Config()), **asdict(Hyperparameters())}
        cfg.update(device=device, a=a, returns_gamma=0.5, minibatch_size=4,
                   memory_length=32, goal_memory_length=32, hidden_size=8,
                   n_optimizer_epochs=1, n_optimizer_steps=2, main_sparse_reward_scale=0.)
        bounds = (torch.full((2,), -1., device=device), torch.ones(2, device=device))
        agent = RDDPG_Dual(2, 2, bounds, **cfg).to(device)
        for buffer, length in [(agent.memory, 8), (agent.goal_memory, 24)]:
            for i in range(length):
                buffer.remember(
                    torch.zeros(2, device=device),
                    torch.zeros(agent.feedback_size, device=device),
                    torch.ones(2, device=device),
                    torch.ones(agent.feedback_size, device=device),
                    torch.zeros(2, device=device),
                    torch.tensor([7.], device=device),
                    torch.tensor([i % 2 == 1], device=device),
                )

        idx = torch.arange(4, device=device)
        original_loss = torch.nn.functional.mse_loss
        with patch.object(agent, "get_reward", side_effect=lambda obs: torch.ones(len(obs), device=device)), \
             patch.object(agent, "_prioritized_recall", side_effect=lambda buf, mode: buf.recall(idx)), \
             patch.object(agent, "_q_targets_o", side_effect=lambda rew, obs, done: rew.new_full((4,), 10.)) as ocp, \
             patch.object(agent, "_q_targets_f", side_effect=lambda rew, obs, done: rew.new_full((4,), 20.)) as fcp, \
             patch("torch.nn.functional.mse_loss", wraps=original_loss) as loss:
            agent.replay()

        # Goal rewards, rather than stored exploration rewards, feed goal OCP TD.
        goal_rewards = torch.tensor([[5.], [1.], [1.], [1.]], device=device)
        torch.testing.assert_close(ocp.call_args_list[0].args[0], goal_rewards)
        expected_returns = torch.tensor([5.5, 1., 1.5, 1.], device=device)
        torch.testing.assert_close(loss.call_args_list[0].args[1], (1 - a) * 10 + a * expected_returns)
        # The next update is main-buffer replay: ordinary TD, regardless of a.
        stored_rewards = torch.full((4, 1), 7., device=device)
        torch.testing.assert_close(ocp.call_args_list[1].args[0], stored_rewards)
        torch.testing.assert_close(loss.call_args_list[2].args[1], torch.full((4,), 10., device=device))
        # FCP always keeps its stored rewards and TD targets, including goal replay.
        for call in fcp.call_args_list:
            torch.testing.assert_close(call.args[0], stored_rewards)
        for i in (1, 3):
            torch.testing.assert_close(loss.call_args_list[i].args[1], torch.full((4,), 20., device=device))


if __name__ == "__main__":
    torch.set_num_threads(4)
    unittest.main()
