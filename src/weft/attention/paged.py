from dataclasses import dataclass

import torch

from weft.attention.base import AttentionBackend


@dataclass
class PagedMetadata:
    n_computed_tokens: torch.Tensor # [B]
    slot_mapping: torch.Tensor # [T_total]
    block_table: torch.Tensor # [B, max_blocks]
    cu_tokens: torch.Tensor # [B+1]


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
        # q -> (1, H_q, T_total, D)
        # k,v -> # (1, H_kv, T_total, D)
        N = k_cache.shape[-4]
        P = k_cache.shape[-3]
        H_kv = k_cache.shape[-2]
        D = k_cache.shape[-1]
        k_flat = k_cache.view(N * P, H_kv, D)
        k_flat[metadata.slot_mapping] = k.permute(0, 2, 1, 3).reshape(-1, H_kv, D)
        v_flat = v_cache.view(N * P, H_kv, D)
        v_flat[metadata.slot_mapping] = v.permute(0, 2, 1, 3).reshape(-1, H_kv, D)
        T_b = (metadata.cu_tokens[1:] - metadata.cu_tokens[:-1])
        S = (metadata.n_computed_tokens + T_b).tolist()
        T_b = T_b.tolist()
        cu_tokens = metadata.cu_tokens.tolist()
        H_g = self.num_heads // self.num_key_value_heads
        n_computed_tokens = metadata.n_computed_tokens.tolist()
        out_parts = []
        for b in range(metadata.block_table.shape[0]):
            # (H_kv, 1, S, D)
            k_full = k_cache[metadata.block_table[b]].flatten(0, 1)[:S[b]].transpose(0, 1).unsqueeze(1)
            v_full = v_cache[metadata.block_table[b]].flatten(0, 1)[:S[b]].transpose(0, 1).unsqueeze(1)
            # (H_kv, H_g, T, D)
            q_b = q[0, :, cu_tokens[b]:cu_tokens[b+1], :].reshape(H_kv, H_g, -1, D)
            # (H_kv, H_g, T, S)
            scores = q_b @ k_full.transpose(-2, -1) / D**0.5
            causal_mask = torch.full((T_b[b], S[b]), float('-inf'), device=scores.device, dtype=scores.dtype) \
                .triu(diagonal=n_computed_tokens[b]+1) \
                .unsqueeze(0)
            scores += causal_mask
            attn = scores.softmax(dim=-1)
            # (H_q, T, D)
            out = (attn @ v_full).flatten(0, 1)
            out_parts.append(out)

        # (1, H_q, T, D)
        return torch.cat(out_parts, dim=1).unsqueeze(0)
