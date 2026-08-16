import torch
from collections import deque

class BlockPool:
    def __init__(self, num_layers, num_blocks, num_key_value_heads, block_size, head_dim, device, dtype):
        shape = (num_layers, num_blocks, num_key_value_heads, block_size, head_dim)
        self.k_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.v_cache = torch.zeros(shape, device=device, dtype=dtype)
        self.free_list = deque(range(self.k_cache.shape[1]))

    def allocate(self, n):
        if len(self.free_list) < n:
            raise ValueError("Not enough storage available")
        return [self.free_list.popleft() for _ in range(n)]

    def free(self, ids):
        self.free_list.extend(ids)
