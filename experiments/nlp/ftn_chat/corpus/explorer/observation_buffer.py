from typing import Any

import torch

from recode.density_estimator import DensityEstimator
from assets.tensor_circular_buffer import TimeFirstCircularBuffer


class ObservationBuffer(DensityEstimator):
    def __init__(
        self,
        memory_size: int,
        k: int,
        memory_threshold: float,
        memory_epsilon: float,
        device: torch.device | None = None,
    ):
        super().__init__(device)
        self.k = k
        self.memory_threshold = memory_threshold
        self.memory_epsilon = memory_epsilon
        self.memory_size = memory_size
        self.memory = TimeFirstCircularBuffer(capacity=self.memory_size)

    @property
    def all_pos(self) -> torch.Tensor:
        if self.memory.buffer is None:
            return torch.tensor([], device=self.device)
        return self.memory.buffer[: len(self.memory)]

    @property
    def state(self) -> dict[str, Any]:
        return {
            "size": self.memory.size,
            "index": self.memory.index,
            "batch_size": self.memory.batch_size,
            "memory": self.memory.buffer if self.memory.buffer is not None else torch.tensor([], device=self.device),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if "size" in state:
            self.memory.size = state["size"]
        if "index" in state:
            self.memory.index = state["index"]
        if "batch_size" in state:
            self.memory.batch_size = state["batch_size"]
        if "memory" in state:
            self.memory.buffer = state["memory"]

    def get_reward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.memory.buffer is None:
            return torch.zeros(1, device=self.device)
        k = min(self.k, len(self.memory))
        dist = torch.cdist(embeddings, self.memory[:]) # type: ignore
        dist_to_kst = dist.topk(k, largest=False).values[:, -1]
        return dist_to_kst

    def update(self, embeddings: torch.Tensor) -> None:
        reward = self.get_reward(embeddings.unsqueeze(0))
        self.update_with_reward(embeddings, reward)

    def update_with_reward(self, embeddings: torch.Tensor, reward: torch.Tensor) -> None:
        if reward > self.memory_threshold or len(self.memory) < 2 or torch.rand(1) < self.memory_epsilon:
            self.memory.append(embeddings.clone())

    def step(self) -> None:
        pass
    
    def reset(self) -> None:
        self.memory = TimeFirstCircularBuffer(capacity=self.memory_size)