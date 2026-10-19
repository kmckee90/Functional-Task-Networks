from collections.abc import Iterator
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor

from assets.reservoir_local_connectivity import ReservoirLocalConnectivityBatchSeq
from assets.tensor_circular_buffer import TimeFirstCircularBuffer

import random

class ExplorerPolicy(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        normalize_output=False,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )
        if normalize_output:
            self.model = nn.Sequential(self.model, nn.Tanh())

    def forward(self, x):
        return self.model(x)


class RDDPGMemory(nn.Module):
    def __init__(self, length: int = 10000):
        super().__init__()
        self.obs = TimeFirstCircularBuffer(capacity=length)
        self.next_obs = TimeFirstCircularBuffer(capacity=length)
        self.reward = TimeFirstCircularBuffer(capacity=length)
        self.action = TimeFirstCircularBuffer(capacity=length)
        self.done = TimeFirstCircularBuffer(capacity=length)
        self.feedback = TimeFirstCircularBuffer(capacity=length)
        self.next_feedback = TimeFirstCircularBuffer(capacity=length)

    def remember(
        self,
        obs: Tensor,
        feedback: Tensor,
        next_obs: Tensor,
        next_feedback: Tensor,
        action: Tensor,
        reward: Tensor,
        done: Tensor,
    ):
        """Stores an experience in the replay buffer."""
        self.obs.append(obs)
        self.feedback.append(feedback)
        self.next_obs.append(next_obs)
        self.next_feedback.append(next_feedback)
        self.action.append(action)
        self.reward.append(reward)
        self.done.append(done)

    def recall(self, idx: Tensor):
        return (
            self.obs[idx],
            self.feedback[idx],
            self.next_obs[idx],
            self.next_feedback[idx],
            self.action[idx],
            self.reward[idx],
            self.done[idx],
            idx,
        )

    def length(self):
        return len(self.obs)


class RDDPG_Dual(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        action_bounds: tuple[torch.Tensor, torch.Tensor],
        **kwargs,
    ):
        super().__init__()
        self.config = kwargs
        self.action_size = action_size
        self.action_bounds = action_bounds
        self.cur_step = 0
        self.feedback_size = self.action_size + 1 + kwargs["num_negdensity_bins"] + kwargs["extra_zeros"]
        self.input_size = observation_size + self.feedback_size

        self.state_model_o = ReservoirLocalConnectivityBatchSeq(
            observation_size, num_unique=40, num_shared=20, spectral_radius=kwargs["spectral_radius"]
        )
        self.state_model_f = ReservoirLocalConnectivityBatchSeq(
            self.feedback_size, num_unique=40, num_shared=20, spectral_radius=kwargs["spectral_radius"]
        )
        self.state_model_o_targ = deepcopy(self.state_model_o)
        self.state_model_f_targ = deepcopy(self.state_model_f)

        self.rnn_size_o = self.state_model_o.hidden_size
        self.rnn_size_f = self.state_model_f.hidden_size

        self.q_pred_o = ExplorerPolicy(self.rnn_size_o + action_size, kwargs["hidden_size"], 1)
        self.q_pred_f = ExplorerPolicy(self.rnn_size_f + action_size, kwargs["hidden_size"], 1)
        self.act_pred = ExplorerPolicy(
            self.rnn_size_o + self.rnn_size_f, kwargs["hidden_size"], action_size, normalize_output=True
        )
        self.q_target_o = deepcopy(self.q_pred_o)
        self.q_target_f = deepcopy(self.q_pred_f)
        self.act_target = deepcopy(self.act_pred)

        self.optimizer_o = torch.optim.RMSprop(self.q_pred_o.parameters(), lr=kwargs["lr"])
        self.optimizer_f = torch.optim.RMSprop(self.q_pred_f.parameters(), lr=kwargs["lr"])
        self.optimizer_a = torch.optim.RMSprop(self.act_pred.parameters(), lr=kwargs["lr"])

        with torch.no_grad():
            for param in self.act_target.parameters():
                param.requires_grad = False
            for param in self.q_target_o.parameters():
                param.requires_grad = False
            for param in self.q_target_f.parameters():
                param.requires_grad = False

        self.memory = RDDPGMemory(length=kwargs["memory_length"])
        self.goal_memory = RDDPGMemory(length=kwargs["goal_memory_length"])
        self.obs_samples = TimeFirstCircularBuffer(capacity=kwargs["sample_memory_length"])

        self.epsilon = kwargs["epsilon"]
        self.device = "cpu"

    def reset_memory(self, **kwargs):
        self.memory = RDDPGMemory(length=kwargs["memory_length"])
        self.goal_memory = RDDPGMemory(length=kwargs["goal_memory_length"])
        self.obs_samples = TimeFirstCircularBuffer(capacity=kwargs["sample_memory_length"])

    def _distance(self, x, samples, k):
        cdist = torch.cdist(x, samples)
        dist_to_kst = cdist.topk(min(k, len(samples)), largest=False).values[:, -1]
        return dist_to_kst

    def forward(self, x: Tensor, f: Tensor) -> Tensor:
        self.device = x.device
        if torch.rand(1) <= self.epsilon:
            return (
                torch.rand(self.action_size, device=x.device) * (self.action_bounds[1] - self.action_bounds[0])
                + self.action_bounds[0]
            )
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(f.shape) == 1:
            f = f.unsqueeze(0)

        zo = self.state_model_o(x)
        zf = self.state_model_f(f)
        z = torch.cat((zo, zf), -1)
        act_values = self.act_pred(z)  # Actor chooses action instead of critic
        return act_values

    def _discretize_unit_intervals(self, x: Tensor):
        cutpoints = torch.tensor(range(1, self.config["num_negdensity_bins"] + 1)) / self.config["num_negdensity_bins"]
        cutpoints = cutpoints.to(x.device)
        b = x.unsqueeze(-1) <= cutpoints
        c = torch.argmax(b.int(), dim=-1)
        y = torch.nn.functional.one_hot(c, num_classes=self.config["num_negdensity_bins"])
        return y

    def _q_values_o(self, obs: Tensor, action: Tensor) -> Tensor:
        obs_action = torch.cat([obs, action], dim=1)
        q_values = self.q_pred_o(obs_action).squeeze()
        return q_values

    def _q_values_f(self, obs: Tensor, action: Tensor) -> Tensor:
        obs_action = torch.cat([obs, action], dim=1)
        q_values = self.q_pred_f(obs_action).squeeze()
        return q_values

    def _q_targets_o(self, reward: Tensor, next_obs: Tensor, done: Tensor) -> Tensor:
        max_q_values_next = self.q_target_o(next_obs).detach().squeeze()
        targets = reward.squeeze() + self.config["gamma"] * max_q_values_next * ~done.squeeze()
        return targets

    def _q_targets_f(self, reward: Tensor, next_obs: Tensor, done: Tensor) -> Tensor:
        max_q_values_next = self.q_target_f(next_obs).detach().squeeze()
        targets = reward.squeeze() + self.config["gamma"] * max_q_values_next * ~done.squeeze()
        return targets

    def _polyak_update(self, target_parameters: Iterator[nn.Parameter], pred_parameters: Iterator[nn.Parameter]):
        with torch.no_grad():
            assert isinstance(self.config["polyak_beta"], float)
            for target, pred in zip(target_parameters, pred_parameters, strict=True):
                target.mul_(1 - self.config["polyak_beta"]).add_(self.config["polyak_beta"] * pred)

    def _prioritized_recall(self, buffer: RDDPGMemory, recall_type: str = "random"):
        """Returns a sequence mini-batch of experiences and feedback"""
        dones_idx = torch.where(buffer.done[:])[0][:-1] + 1  # type: ignore
        if recall_type in ["goal", "total_goal"]:
            densities = self._distance(buffer.obs[:], self.obs_samples[:], self.config["k"])  # type: ignore
            best = densities.argmax()
            start_idx = (best < dones_idx).int().argmax() - 1
        elif recall_type == "last":
            start_idx = -1
        else:
            start_idx = torch.randint(0, len(dones_idx), (1,))

        minibatch_start_idx = dones_idx[start_idx]
        minibatch_end_idx = int(min(minibatch_start_idx.item() + self.config["minibatch_size"], buffer.length()))
        idx = torch.tensor(range(minibatch_start_idx, minibatch_end_idx))
        return buffer.recall(idx)

    def replay(self):
        if self.memory.length() < self.config["minibatch_size"]:
            return

        # Save the real-time context to reinstate later.
        hidden_f = self.state_model_f.hidden.clone()
        hidden_o = self.state_model_o.hidden.clone()

        # Calculate the reward function for main memories:
        main_reward = self._distance(self.memory.next_obs[:], self.obs_samples[:], self.config["k"])  # type: ignore
        main_best = main_reward.argmax()
        main_reward[main_best] *= self.config["main_sparse_reward_scale"]
        best_main_reward = main_reward[main_best]

        # Calculate the reward function for goal memories:
        if self.goal_memory.length() > 3 * self.config["minibatch_size"]:
            goal_reward = self._distance(self.goal_memory.next_obs[:], self.obs_samples[:], self.config["k"])  # type: ignore
            goal_best = goal_reward.argmax()
            goal_reward[goal_best] *= self.config["goal_sparse_reward_scale"]
            best_goal_reward = goal_reward[goal_best]

            # Keep the goal location for rendering
            self.goal_location = self.goal_memory.next_obs[goal_best]
        else:
            best_goal_reward = 0

        # If a better reward was computed from main, move that episode to goals
        if best_main_reward > best_goal_reward:
            obs, fb, next_obs, next_fb, act, rew, done, _ = self._prioritized_recall(self.memory, "goal")
            for i in range(len(obs)):
                self.goal_memory.remember(obs[i], fb[i], next_obs[i], next_fb[i], act[i], rew[i], done[i])

            # Have to redo this because goal memory changed
            goal_reward = self._distance(self.goal_memory.next_obs[:], self.obs_samples[:], self.config["k"])  # type: ignore
            goal_best = goal_reward.argmax()
            goal_reward[goal_best] *= self.config["goal_sparse_reward_scale"]
            best_goal_reward = goal_reward[goal_best]

            # Keep the goal location for rendering
            self.goal_location = self.goal_memory.next_obs[goal_best]

        # Prioritized replay
        if self.config["prioritized_replay"]:
            replay_types = ["total_goal", "goal", "last", "random", "total_goal"]
        else:
            replay_types = ["random"]
        for _ in range(self.config["n_optimizer_epochs"]):
            for step in range(self.config["n_optimizer_steps"]):
                if step < len(replay_types):
                    recall_mode = replay_types[step]
                else:
                # recall_mode = random.choice(replay_types)
                    recall_mode = "random"

                # Train on either the goal buffer or the main buffer
                if self.goal_memory.length() > 4 * self.config["minibatch_size"] and recall_mode == "total_goal":
                    obs, feedback, next_obs, next_feedback, action, reward, done, idx = self._prioritized_recall(
                        self.goal_memory, "goal"
                    )
                    rewards_o = goal_reward[idx].reshape(-1, 1)  # type: ignore
                else:
                    obs, feedback, next_obs, next_feedback, action, reward, done, idx = self._prioritized_recall(
                        self.memory, recall_mode
                    )
                    rewards_o = main_reward[idx].reshape(-1, 1)  # type: ignore

                states_o = self.state_model_o(obs)
                next_states_o = self.state_model_o_targ(next_obs)

                states_f = self.state_model_f(feedback)
                next_states_f = self.state_model_f_targ(next_feedback)

                states = torch.cat((states_o, states_f), -1)
                next_states = torch.cat((next_states_o, next_states_f), -1)

                # Critic loss (obs)
                q_values_o = self._q_values_o(states_o, action)
                next_action = self.act_target(next_states)
                targets_o = self._q_targets_o(rewards_o, torch.cat([next_states_o, next_action], dim=1).detach(), done)
                critic_loss_o = torch.nn.functional.mse_loss(q_values_o, targets_o)

                # Critic loss (feedback)
                rewards_f = reward
                q_values_f = self._q_values_f(states_f, action)
                next_action = self.act_target(next_states)
                targets_f = self._q_targets_f(rewards_f, torch.cat([next_states_f, next_action], dim=1).detach(), done)
                critic_loss_f = torch.nn.functional.mse_loss(q_values_f, targets_f)

                # Actor loss
                action = self.act_pred(states)
                obs_action_o = torch.cat([states_o, action], dim=1)
                obs_action_f = torch.cat([states_f, action], dim=1)
                q_values_f = self.q_target_f(obs_action_f).squeeze()
                q_values_o = self.q_target_o(obs_action_o).squeeze()
                actor_loss = -(q_values_o.mean() * self.config["OCP"] + q_values_f.mean() * self.config["FCP"])

                # Update
                if self.config["OCP"]:
                    self.optimizer_o.zero_grad()
                    critic_loss_o.backward()
                    self.optimizer_o.step()

                if self.config["FCP"]:
                    self.optimizer_f.zero_grad()
                    critic_loss_f.backward()
                    self.optimizer_f.step()

                self.optimizer_a.zero_grad()
                actor_loss.backward()
                self.optimizer_a.step()

                if self.config["OCP"]:
                    self._polyak_update(self.q_target_o.parameters(), self.q_pred_o.parameters())
                if self.config["FCP"]:
                    self._polyak_update(self.q_target_f.parameters(), self.q_pred_f.parameters())
                self._polyak_update(self.act_target.parameters(), self.act_pred.parameters())

        # Reinstate real-world context
        self.state_model_f.hidden = hidden_f
        self.state_model_o.hidden = hidden_o

    def step(self):
        self.cur_step += 1
