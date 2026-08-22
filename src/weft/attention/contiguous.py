from dataclasses import dataclass

import torch

from weft.attention.base import AttentionBackend


@dataclass
class ContiguousMetadata:
    cursor: int
    seq_len: torch.Tensor
    rows: list[int]


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
        # q -> (B, H_q, T, D)
        # k,v -> # (B, H_kv, T, D)
        D = self.head_dim
        T = k.shape[-2]
        C = metadata.cursor
        S_max = C + T
        k_cache[metadata.rows, :, C:S_max, :] = k
        v_cache[metadata.rows, :, C:S_max, :] = v
        # (B, H_kv, 1, S_max, D)
        H_kv = self.num_key_value_heads
        H_g = self.num_heads // self.num_key_value_heads
        k_full, v_full = k_cache[metadata.rows, :, :S_max, :].unsqueeze(2), v_cache[metadata.rows, :, :S_max, :].unsqueeze(2)
        # (B, H_kv, H_g, T, D)
        q = q.reshape(-1, H_kv, H_g, T, D)
        # (B, H_kv, H_g, T, S_max)
        scores = q @ k_full.transpose(-2, -1) / D**0.5
        causal_mask = torch.full((T, S_max), torch.finfo(scores.dtype).min, device=scores.device, dtype=scores.dtype).triu(diagonal=C+1)
        pad_mask = torch.stack([
            torch.cat([
                torch.full((T, S_max - n), torch.finfo(scores.dtype).min, device=scores.device, dtype=scores.dtype),
                torch.zeros((T, n), device=scores.device, dtype=scores.dtype),
            ], dim=1)
            for n in metadata.seq_len.tolist()
        ]).unsqueeze(1).unsqueeze(1)
        scores += causal_mask + pad_mask
        attn = scores.softmax(dim=-1)
        # (B, H_q, T, D)
        return (attn @ v_full).flatten(1, 2)
