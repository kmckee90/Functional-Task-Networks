from collections.abc import Callable
from typing import Any

import torch

from .density_estimator import DensityEstimator


def knn_reward(
    embeddings: torch.Tensor, memory: torch.Tensor, counts: torch.Tensor, d2_ema: torch.Tensor, k: int = 15
) -> torch.Tensor:
    """Calculates the reward for each embedding using the k-th nearest neighbor

    Args:
        embeddings (torch.Tensor): Embeddings to calculate the reward for
            shape: (batch_size, embedding_size)
        memory (torch.Tensor): Memory of embeddings
            shape: (memory_size, embedding_size)
        counts (torch.Tensor): Counts for each memory element
            shape: (memory_size,)
        d2_ema (torch.Tensor): Exponential moving average of the squared distance to the k-th nearest neighbor
        k (int): Number of nearest neighbors to consider
    """
    k = min(k, len(memory))
    distances = torch.cdist(embeddings, memory)  # type: ignore
    min_distances, min_dist_idxs = torch.topk(distances, k, dim=1, largest=False)
    min_dist_counts = counts[min_dist_idxs]
    cum_counts = torch.cumsum(min_dist_counts, dim=1)
    k_tensor = torch.full_like(cum_counts[:, 0], k)
    kth_idxs = torch.searchsorted(cum_counts, k_tensor.unsqueeze(1), side="left")
    kth_idxs = torch.clamp(kth_idxs, max=k - 1)
    batch_idxs = torch.arange(embeddings.shape[0], device=embeddings.device)
    rewards = min_distances[batch_idxs, kth_idxs.squeeze(1)]
    return rewards


def adaptive_knn_reward(
    embeddings: torch.Tensor, memory: torch.Tensor, counts: torch.Tensor, d2_ema: torch.Tensor
) -> torch.Tensor:
    """Calculates the reward for each embedding using the adaptive k-th nearest neighbor

    Args:
        embeddings (torch.Tensor): Embeddings to calculate the reward for
            shape: (batch_size, embedding_size)
        memory (torch.Tensor): Memory of embeddings
            shape: (memory_size, embedding_size)
        counts (torch.Tensor): Counts for each memory element
            shape: (memory_size,)
    """
    k = 64 if torch.sum(counts) > 50000 else 16
    return knn_reward(embeddings, memory, counts, d2_ema, k)


def rbf_kernel(
    embeddings: torch.Tensor, memory: torch.Tensor, d2_ema: torch.Tensor, sigma2: float = 0.5
) -> torch.Tensor:
    """RBF Kernel function

    Args:
        embeddings (torch.Tensor): Embeddings to calculate the kernel for
            shape: (batch_size, embedding_size)
        memory (torch.Tensor): Memory of embeddings
            shape: (memory_size, embedding_size)
        d2_ema (torch.Tensor): Exponential moving average of the squared distance to the k-th nearest neighbor
        sigma2 (float): Variance of the Gaussian kernel
            default value = 0.5

    Returns:
        torch.Tensor: Kernel values for each embedding
            shape: (batch_size, memory_size)
    """
    diff = embeddings[:, None, :] - memory[None, :, :]
    norm_sq = torch.norm(diff, dim=-1) ** 2
    return torch.exp(-(norm_sq / d2_ema) / (2 * sigma2))


def inverse_kernel(
    embeddings: torch.Tensor, memory: torch.Tensor, d2_ema: torch.Tensor, epsilon: float = 1e-3
) -> torch.Tensor:
    """Kernel function for RECODE

    Args:
        embeddings (torch.Tensor): Embeddings to calculate the kernel for
            shape: (batch_size, embedding_size)
        memory (torch.Tensor): Memory of embeddings
            shape: (memory_size, embedding_size)
        d2_ema (torch.Tensor): Exponential moving average of the squared distance to the k-th nearest neighbor
        epsilon (float): Small constant to avoid division by zero
            default value from https://arxiv.org/pdf/2002.06038 (3)

    Returns:
        torch.Tensor: Kernel values for each embedding
            shape: (batch_size, memory_size)
    """
    diff = embeddings[:, None, :] - memory[None, :, :]
    norm_sq = torch.norm(diff, dim=-1) ** 2
    kernel = 1 / (1 + norm_sq / (epsilon * d2_ema))
    threshold = (norm_sq < d2_ema).float()
    return kernel * threshold


def soft_visitation_reward(
    embeddings: torch.Tensor,
    memory: torch.Tensor,
    counts: torch.Tensor,
    d2_ema: torch.Tensor,
    n_0: float = 0.01,
    kernel_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = inverse_kernel,
) -> torch.Tensor:
    kernel_values = kernel_func(embeddings, memory, d2_ema)
    n = torch.sum((1 + counts) * kernel_values, dim=-1)
    reward = 1 / (torch.sqrt(n) + n_0)
    return reward


class RECODE(DensityEstimator):
    """Implementation of RECODE (https://arxiv.org/pdf/2305.01521)

    Args:
        observation_size (int): Size of the observation space
        memory_size (int): Maximum number of elements in the memory
        k (int): Number of nearest neighbors to consider
        relative_tolerance (float): Relative tolerance for the distance to the k-th nearest neighbor
        decay_rate (float): Decay rate for the exponential moving average
        gamma (float): Discount factor for the counts
        insertion_probability (float): Probability of inserting a new element into the memory
        n_0 (float): Small constant to avoid division by zero
        d2_ema (float): Exponential moving average of the squared distance to the k-th nearest neighbor
        kernel_func (Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]): Kernel function to use
    """

    def __init__(
        self,
        observation_size: int,
        memory_size: int,
        k: int = 20,
        relative_tolerance: float = 0.2,
        decay_rate: float = 0.9999,
        gamma: float = 1.0,
        insertion_probability: float = 0.10,
        d2_ema: float = 0,
        device: torch.device | None = None,
        reward_func: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] = knn_reward,
    ):
        self.obs_size = observation_size
        self.memory_size = memory_size
        self.k = k
        self.relative_tolerance = relative_tolerance
        self.decay_rate = decay_rate
        self.gamma = gamma
        self.insertion_probability = insertion_probability
        self.d2_ema = torch.tensor(d2_ema, device=device)
        self.reward_func = reward_func
        super().__init__(device)

        self.device=device
        self.memory = torch.zeros(self.memory_size, observation_size, device=device, dtype=torch.float32)
        self.counts = torch.zeros(self.memory_size, device=device, dtype=torch.float32)
        self.d2_nearest = torch.full_like(self.counts, float("inf"), device=device, dtype=torch.float32)
        self.size = 0

    @property
    def all_pos(self) -> torch.Tensor:
        return self.memory[: self.size]

    @property
    def state(self) -> dict[str, Any]:
        return {
            "memory": self.memory,
            "counts": self.counts,
            "d2_nearest": self.d2_nearest,
            "size": self.size,
            "d2_ema": self.d2_ema,
        }

    def reset(self):
        self.memory = torch.zeros(self.memory_size, self.obs_size, device=self.device, dtype=torch.float32)
        self.counts = torch.zeros(self.memory_size, device=self.device, dtype=torch.float32)
        self.d2_nearest = torch.full_like(self.counts, float("inf"), device=self.device, dtype=torch.float32)
        self.size = 0

    def set_state(self, state: dict[str, Any]) -> None:
        if "memory" in state:
            if state["memory"].shape != self.memory.shape:
                raise ValueError("Memory shape mismatch")
            self.memory = state["memory"]
        if "counts" in state:
            if state["counts"].shape != self.counts.shape:
                raise ValueError(f"Counts shape mismatch: {state['counts'].shape} != {self.counts.shape}")
            self.counts = state["counts"]
        if "d2_nearest" in state:
            self.d2_nearest = state["d2_nearest"]
        if "size" in state:
            self.size = state["size"]
        if "d2_ema" in state:
            self.d2_ema = state["d2_ema"]

    def get_reward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Calculates the reward for each embedding

        Args:
            embeddings (torch.Tensor): Embeddings to calculate the reward for
                shape: (batch_size, embedding_size)
        """
        if self.size == 0:
            return torch.zeros(embeddings.shape[0], device=self.device)
        return self.reward_func(embeddings, self.memory[: self.size], self.counts[: self.size], self.d2_ema)

    def update(self, embeddings: torch.Tensor) -> None:
        """Updates the memory with new embeddings

        Args:
            embeddings (torch.Tensor): Embeddings to update the memory with
                shape: (batch_size, embedding_size)
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.unsqueeze(0)
        free_size = self.memory_size - self.size
        batch_size = embeddings.shape[0]
        insert_size = min(batch_size, free_size)
        if insert_size > 0:
            self.memory[self.size : self.size + insert_size] = embeddings[:insert_size]
            self.counts[self.size : self.size + insert_size] = 1
            self.size += insert_size
            self._metrics["Updates/vacant_insertions"] += insert_size
        if self.size > self.k:
            for embedding in embeddings[:insert_size]:
                d2 = torch.cdist(embedding.unsqueeze(0), self.memory[: self.size]).squeeze().pow(2)
                d2 = torch.where(d2 == 0, torch.tensor(float("inf")), d2)  # distance to self is 0
                self._update_d2_ema(d2)
        for embedding in embeddings[insert_size:]:
            self._merge_update(embedding)
        self._metrics["Distances/d2_ema"] = self.d2_ema.item()

    def _update_d2_nearest(self) -> None:
        for i, embedding in enumerate(self.memory[: self.size]):
            d2 = torch.cdist(embedding.unsqueeze(0), self.memory[: self.size]).squeeze().pow(2)
            d2 = torch.where(d2 == 0, torch.tensor(float("inf")), d2)  # distance to self is 0
            self.d2_nearest[i] = d2.min()

    def _update_d2_ema(self, d2: torch.Tensor) -> None:
        d2_topk_mean = d2.topk(self.k, largest=False).values.mean()
        if d2_topk_mean != float("inf"):
            if self.d2_ema == 0:
                self.d2_ema = d2_topk_mean
            else:
                self.d2_ema = self.decay_rate * self.d2_ema + (1 - self.decay_rate) * d2_topk_mean

    def _merge_update(self, embedding: torch.Tensor) -> None:
        d2 = torch.cdist(embedding.unsqueeze(0), self.memory).squeeze().pow(2)
        self._update_d2_ema(d2)
        self.counts *= self.gamma
        nearest_atom_idx = torch.argmin(d2)
        self._metrics["Distances/d2_nearest_atom"] = d2[nearest_atom_idx].item()
        if d2[nearest_atom_idx] > self.relative_tolerance * self.d2_ema and torch.rand(1) < self.insertion_probability:
            if torch.all(self.d2_nearest == float("inf")):
                self._update_d2_nearest()
            # select atom to remove
            remove_idx = self._get_remove_idx()
            # find nearest atom to removed atom
            d2_removed = torch.cdist(self.memory[remove_idx].unsqueeze(0), self.memory).squeeze().pow(2)
            d2_removed[remove_idx] = float("inf")
            redistribute_idx = torch.argmin(d2_removed)
            self._metrics["Distances/d2_removed"] = d2_removed[redistribute_idx].item()
            # redistruibute removed atom counts to nearest atom
            self.memory[redistribute_idx] = (
                self.counts[redistribute_idx] * self.memory[redistribute_idx]
                + self.counts[remove_idx] * self.memory[remove_idx]
            ) / (self.counts[redistribute_idx] + self.counts[remove_idx])
            self.counts[redistribute_idx] += self.counts[remove_idx]
            # replace removed atom with new atom
            self.memory[remove_idx] = embedding
            self.counts[remove_idx] = 1
            # replace log_d2_nearest entry
            self.d2_nearest[remove_idx] = d2[nearest_atom_idx]
            self._metrics["Updates/eviction_insertions"] += 1
        else:
            # merge embedding into nearest atom
            self.memory[nearest_atom_idx] = (
                self.counts[nearest_atom_idx] * self.memory[nearest_atom_idx] + embedding
            ) / (self.counts[nearest_atom_idx] + 1)
            self.counts[nearest_atom_idx] += 1
            self._metrics["Updates/merges"] += 1

    def _get_remove_idx(self) -> torch.Tensor:
        return torch.argmin(self.d2_nearest)

    def step(self) -> None:
        pass
