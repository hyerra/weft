from dataclasses import dataclass

import torch

from weft.attention.base import AttentionBackend


@dataclass
class PagedMetadata:
    n_computed_tokens: torch.Tensor # [B]
    slot_mapping: torch.Tensor # [T_total]
    block_table: torch.Tensor # [B, max_blocks]


class PagedGatherBackend(AttentionBackend):
    def __init__(self, num_heads: int, num_key_value_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

    @staticmethod
    def kv_cache_shape(num_blocks: int, block_size: int, num_key_value_heads: int, head_dim: int) -> tuple[int, int, int, int]:
        return (num_blocks, block_size, num_key_value_heads, head_dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, metadata: PagedMetadata) -> torch.Tensor:
        # q -> (batch, num_heads, new_seq_len, head_dim)
        # k,v -> # (batch, num_key_value_heads, new_seq_len, head_dim)
        N = k_cache.shape[-4]
        P = k_cache.shape[-3]
        H_kv = k_cache.shape[-2]
        D = k_cache.shape[-1]
        k_flat = k_cache.view(N * P, H_kv, D)
        k_flat[metadata.slot_mapping] = k.permute(0, 2, 1, 3).reshape(-1, H_kv, D)
        v_flat = v_cache.view(N * P, H_kv, D)
        v_flat[metadata.slot_mapping] = v.permute(0, 2, 1, 3).reshape(-1, H_kv, D)
        T = k.shape[-2]
        end = metadata.n_computed_tokens + T
        max_end = int(end.max())
        # (batch, num_heads, max_end, head_dim)
        num_heads_per_group = self.num_heads // self.num_key_value_heads
        k_full = k_cache[metadata.block_table].flatten(1, 2)[:, :max_end].transpose(1, 2).repeat_interleave(num_heads_per_group, dim=1)
        v_full = v_cache[metadata.block_table].flatten(1, 2)[:, :max_end].transpose(1, 2).repeat_interleave(num_heads_per_group, dim=1)
        # (batch, num_heads, new_seq_len, max_end)
        scores = q @ k_full.transpose(-2, -1) / self.head_dim**0.5
        scores += torch.stack([torch.full((T, max_end), float('-inf'), device=scores.device, dtype=scores.dtype).triu(diagonal=s+1) for s in metadata.n_computed_tokens.tolist()]).unsqueeze(1)
        attn = scores.softmax(dim=-1)
        # (batch, num_heads, new_seq_len, head_dim)
        return attn @ v_full
