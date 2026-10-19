import torch
from torch.utils.data import Dataset


class _TensorCircularBuffer(Dataset):
    """Vectorized circular buffer class for efficient storage and retrieval of torch tensors.

    Arguments:
        capacity (int): The maximum amount of tensors that the buffer can hold at any moment in time.
    """

    def __init__(self, *, capacity: int):
        self.capacity = capacity
        self.clear()

    def clear(self):
        self.size = 0
        self.index = 0
        self.buffer: torch.Tensor | None = None
        self.device: torch.device | None = None
        self.batch_size: int | None = None

    def append(self, observation: torch.Tensor):
        # Once we push our first observation, we are bound to stay consistent with that
        # observation shape
        if self.buffer is None:
            self.device = observation.device
            self.buffer = torch.zeros(self.capacity, *observation.shape, device=self.device, dtype=observation.dtype)
            self.batch_size = observation.shape[0]

        self.buffer[self.index] = observation
        self.index += 1

        if self.index == self.capacity:
            self.index = 0

        self.size = min(self.capacity, self.size + 1)

    def __len__(self):
        return self.size

    def last_batch(self):
        assert self.buffer is not None
        return self.buffer[self.index - 1]

    def observation_shape(self):
        if self.buffer is None:
            raise RuntimeError("An observation must be appended before shape can be known.")

        return self.buffer[0].shape


class BatchFirstCircularBuffer(_TensorCircularBuffer):
    """Vectorized circular buffer class for efficient storage and retrieval of torch tensors. Indexing
    is performed along the batch dimension.

    buffer = [(0, 1), (2, 3), (4, 5)]
    buffer[[0, 1]] == [(0, 3)]
    buffer[[2, 0]] == [(4, 1)]
    """

    def append(self, observation: torch.Tensor):
        if len(observation.shape) < 2:
            raise ValueError("The shape of 'observation' must be (batch_size, *feature_shape)")

        super().append(observation)

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        if self.buffer is None:
            raise RuntimeError("An observation must be appended before acessing the buffer.")

        if len(idx) != self.batch_size:
            raise ValueError("Length of 'idx' must match self.batch_size")

        # (capacity, batch, feature) -> (buff size, batch, feature)
        sub_buffer = self.buffer[: self.size]
        # (buff size, batch, feature) -> (batch, buff size, feature) -> (batch, feature)
        return sub_buffer.transpose(1, 0)[torch.arange(len(idx)), idx]


class TimeFirstCircularBuffer(_TensorCircularBuffer):
    """Vectorized circular buffer class for efficient storage and retrieval of torch tensors. Indexing
    is performed along the time dimension.

    buffer = [(0, 1), (2, 3), (4, 5)]
    buffer[[0, 1]] == [(0, 1), (2, 3)]
    buffer[[2, 0]] == [(4, 5), (0, 1)]

    """

    def __getitem__(self, idx: torch.Tensor) -> torch.Tensor:
        if self.buffer is None:
            raise RuntimeError("An observation must be appended before acessing the buffer.")

        # (capacity, batch, feature) -> (buff size, batch, feature)
        sub_buffer = self.buffer[: self.size]
        # (buff size, batch, feature) -> (num timestamps, feature)
        return sub_buffer[idx]
