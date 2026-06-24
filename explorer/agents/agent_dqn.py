from collections.abc import Iterator
from copy import deepcopy

import torch
import torch.nn as nn
from torch import Tensor

from assets.reservoir_local_connectivity import ReservoirLocalConnectivityBatchSeq
from assets.tensor_circular_buffer import TimeFirstCircularBuffer


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
        return(
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


class RDQN(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        **kwargs,
    ):
        super().__init__()
        self.config = kwargs
        self.action_size = action_size
        self.cur_step = 0
        self.feedback_size = self.action_size + 1 + kwargs["num_negdensity_bins"] + kwargs["extra_zeros"]
        self.input_size = observation_size + self.feedback_size

        self.state_model = ReservoirLocalConnectivityBatchSeq(
            self.input_size, num_unique=40, num_shared=20, spectral_radius=kwargs["spectral_radius"]
        )
        self.state_model_o_targ = deepcopy(self.state_model)
        self.rnn_size = self.state_model.hidden_size
        self.q_pred = ExplorerPolicy(self.rnn_size, kwargs["hidden_size"], self.action_size)
        self.q_target = deepcopy(self.q_pred)
        self.optimizer = torch.optim.RMSprop(self.q_pred.parameters(), lr=kwargs["lr"])

        with torch.no_grad():
            for param in self.q_target.parameters():
                param.requires_grad = False

        self.memory = RDDPGMemory(length=kwargs["memory_length"])
        self.goal_memory = RDDPGMemory(length=kwargs["goal_memory_length"])
        self.obs_samples = TimeFirstCircularBuffer(capacity=kwargs["sample_memory_length"])

        self.epsilon = kwargs["epsilon"]
        self.device = "cpu"

    # def _distance(self, x, samples, k):
    #     cdist = torch.cdist(x, samples)
    #     dist_to_kst = cdist.topk(min(k, len(samples)), largest=False).values[:, -1]
    #     return dist_to_kst

    def _distance(self, x, samples, k):
        cdist = torch.cdist(x, samples)
        dist_to_kst = cdist.topk(min(k, len(samples)), largest=False).values[:, -1]
        # x = (cdist < 1e-6).float().sum(-1)
        # prob_mass = 1/(1 + (x < 1e-6).sum(-1) ) #Just use the count
        # prob_mass = torch.exp(-x)-x/15.0+1
        # prob_mass = 4.0*torch.exp(-x/4.0)
        # breakpoint()
        return dist_to_kst

    def forward(self, x: Tensor, f: Tensor) -> tuple[Tensor, Tensor]:
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(f.shape) == 1:
            f = f.unsqueeze(0)
            
        x = torch.cat((x,f),-1)        
        z = self.state_model(x)
        q = self.q_pred(z)  # Actor chooses action instead of critic
        action = q.argmax()

        if torch.rand(1) <= self.epsilon:
            action = torch.randint(low=0, high=self.action_size, size=(1,), device=x.device)
        return q, action

    def _discretize_unit_intervals(self, x: Tensor):
        cutpoints = torch.tensor(range(1, self.config["num_negdensity_bins"] + 1)) / self.config["num_negdensity_bins"]
        cutpoints = cutpoints.to(x.device)
        b = x.unsqueeze(-1) <= cutpoints
        c = torch.argmax(b.int(), dim=-1)
        y = torch.nn.functional.one_hot(c, num_classes=self.config["num_negdensity_bins"])
        return y

    def _polyak_update(self, target_parameters: Iterator[nn.Parameter], pred_parameters: Iterator[nn.Parameter]):
        with torch.no_grad():
            assert isinstance(self.config["polyak_beta"], float)
            for target, pred in zip(target_parameters, pred_parameters, strict=True):
                target.mul_(1 - self.config["polyak_beta"]).add_(self.config["polyak_beta"] * pred)

    def _prioritized_recall(self, buffer: RDDPGMemory, recall_type: str = "random"):
        """Returns a sequence mini-batch of experiences and feedback"""
        dones_idx = torch.where(buffer.done[:])[0][:-1] + 1  # type: ignore
        if recall_type == "goal":
            densities = self._distance(buffer.obs[:], self.obs_samples[:], self.config["k"])  # type: ignore
            best = densities.argmax()
            start_idx =(best < dones_idx).int().argmax() - 1
        elif recall_type == "last":
            start_idx = -1
        else:
            start_idx = torch.randint(0, len(dones_idx),(1,))

        minibatch_start_idx = dones_idx[start_idx]
        minibatch_end_idx = int(min(minibatch_start_idx.item() + self.config["minibatch_size"], buffer.length()))
        idx = torch.tensor(range(minibatch_start_idx, minibatch_end_idx))
        return buffer.recall(idx)

    def replay(self):
        if self.memory.length() < self.config["minibatch_size"]:
            return

        # Save the real-time context to reinstate later.
        hidden = self.state_model.hidden.clone()

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
            replay_types = ["goal", "last", "random"]
        else:
            replay_types = ["random"]
        for _ in range(self.config["n_optimizer_epochs"]):
            for step in range(self.config["n_optimizer_steps"]):
                if step < len(replay_types):
                    recall_mode = replay_types[step]
                else:
                    recall_mode = "random"

                # Train on either the goal buffer or the main buffer
                if self.goal_memory.length() > 4 * self.config["minibatch_size"] and recall_mode == "goal":
                    obs, feedback, next_obs, next_feedback, action, reward, done, idx = self._prioritized_recall(
                        self.goal_memory, "goal"
                    )
                    reward = reward + goal_reward[idx].reshape(-1, 1)  # type: ignore
                else:
                    obs, feedback, next_obs, next_feedback, action, reward, done, idx = self._prioritized_recall(
                        self.memory, recall_mode
                    )
                    reward = reward + main_reward[idx].reshape(-1, 1)  # type: ignore

                obs = torch.cat((obs, feedback),-1)
                next_obs = torch.cat((next_obs, next_feedback),-1)

                states = self.state_model(obs)
                next_states = self.state_model_o_targ(next_obs)
                
                q_values_next = self.q_target(next_states).detach()
                max_q_values_next = torch.max(q_values_next, dim=1)[0]
                targets = reward.squeeze() + self.config["gamma"] * (max_q_values_next * ~done.squeeze())
                q_values = self.q_pred(states)                
                q_values = q_values.gather(1, action).squeeze(-1)
                critic_loss = torch.nn.functional.mse_loss(q_values, targets)

                # Update
                self.optimizer.zero_grad()
                critic_loss.backward()
                self.optimizer.step()

                self._polyak_update(self.q_target.parameters(), self.q_pred.parameters())

        # Reinstate real-world context
        self.state_model.hidden = hidden

    def step(self):
        self.cur_step += 1
