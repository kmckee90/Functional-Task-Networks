from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import torch


class DensityEstimator(ABC):
    def __init__(self, device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self._metrics: defaultdict[str, float] = defaultdict(float)

    @property
    def metrics(self) -> dict[str, float]:
        return self._metrics

    @property
    @abstractmethod
    def all_pos(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def state(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def set_state(self, state: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_reward(self, embeddings: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def update(self, embeddings: torch.Tensor) -> None:
        pass

    @abstractmethod
    def step(self) -> None:
        pass

    def reset_metrics(self) -> None:
        self._metrics.clear()
    