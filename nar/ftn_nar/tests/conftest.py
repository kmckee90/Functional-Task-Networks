import torch

# Tiny tensors + many-core pod: intra-op thread pools oversubscribe catastrophically.
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
