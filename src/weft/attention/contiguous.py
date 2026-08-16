from dataclasses import dataclass

import torch

from weft.attention.base import AttentionBackend


@dataclass
class ContiguousMetadata:
    n_computed_tokens: int


class ContiguousBackend(AttentionBackend):
    def __init__(self, num_heads: int, num_key_value_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

    @staticmethod
    def kv_cache_shape(batch_size: int, num_kv_heads: int, max_seq_len: int, head_dim: int) -> tuple[int, int, int, int]:
        return (batch_size, num_kv_heads, max_seq_len, head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, metadata: ContiguousMetadata) -> torch.Tensor:
        # q -> (batch, num_heads, new_seq_len, head_dim)
        # k,v -> # (batch, num_key_value_heads, new_seq_len, head_dim)
        T = k.shape[-2]
        start = metadata.n_computed_tokens
        end = start + T
        k_cache[..., start:end, :] = k
        v_cache[..., start:end, :] = v
        # (batch, num_heads, seq_len, head_dim)
        num_heads_per_group = self.num_heads // self.num_key_value_heads
        k_full, v_full = k_cache[..., :end, :].repeat_interleave(num_heads_per_group, dim=1), v_cache[..., :end, :].repeat_interleave(num_heads_per_group, dim=1)
        # (batch, num_heads, new_seq_len, seq_len)
        scores = q @ k_full.transpose(-2, -1) / self.head_dim**0.5
        scores += torch.full((T, end), float('-inf'), device=scores.device, dtype=scores.dtype).triu(diagonal=start+1)
        attn = scores.softmax(dim=-1)
        return attn @ v_full
