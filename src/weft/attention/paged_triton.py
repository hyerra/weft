import torch
import triton
import triton.language as tl

from weft.attention.paged import PagedMetadata, PagedGatherBackend


@triton.jit
def _block_scores(
    bid,
    q,
    k_cache_ptr,
    kvn_stride,
    kvp_stride,
    kvh_stride,
    kvd_stride,
    pid_h,
    head_dim,
    p_offsets,
    d_mask,
    d_offsets,
    seq_mask,
):
    k_ptrs = k_cache_ptr + bid * kvn_stride + p_offsets * kvp_stride + pid_h * kvh_stride
    k_tile_ptrs = k_ptrs[:, None] + d_offsets[None, :] * kvd_stride
    k = tl.load(k_tile_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0) # [P, D]

    # matmul trick because we can't use tl.dot for small dimension sizes
    scores = tl.sum((q[:, None, :] * k[None, :, :]).to(tl.float32), axis=2) / tl.sqrt(head_dim.to(tl.float32)) # [H_g, P]
    scores = tl.where(seq_mask[None, :], scores, float('-inf'))
    return scores


@triton.jit
def paged_kernel(
    q_ptr,
    qb_stride,
    qh_stride,
    qd_stride,
    k_cache_ptr,
    v_cache_ptr,
    kvn_stride,
    kvp_stride,
    kvh_stride,
    kvd_stride,
    bt_ptr,
    bt_stride,
    head_dim,
    num_queries_per_group,
    n_computed_ptr,
    output_ptr,
    oh_stride,
    ob_stride,
    od_stride,
    H_g: tl.constexpr,
    P: tl.constexpr,
    D: tl.constexpr,
):
    pid_b = tl.program_id(axis=0) # which request
    pid_h = tl.program_id(axis=1) # which kv-head

    S_b = tl.load(n_computed_ptr + pid_b) + 1
    num_blocks_b = tl.cdiv(S_b, P)
    p_offsets = tl.arange(0, P)

    d_offsets = tl.arange(0, D)
    d_mask = d_offsets < head_dim

    q_group_offsets = tl.arange(0, H_g)
    q_ptrs = q_ptr + pid_h * num_queries_per_group * qh_stride + qh_stride * q_group_offsets + pid_b * qb_stride
    q_tile_ptrs = q_ptrs[:, None] + d_offsets[None, :] * qd_stride
    q_mask = (q_group_offsets < num_queries_per_group)[:, None] & d_mask[None, :]
    q = tl.load(q_tile_ptrs, mask=q_mask, other=0.0) # [H_g, D]

    m = tl.full((H_g,), value=float('-inf'), dtype=tl.float32)
    for i in range(num_blocks_b):
        bid = tl.load(bt_ptr + pid_b * bt_stride + i)
        seq_mask = i * P + p_offsets < S_b
        scores = _block_scores(bid, q, k_cache_ptr, kvn_stride, kvp_stride,
                               kvh_stride, kvd_stride, pid_h, head_dim, p_offsets, d_mask,
                               d_offsets, seq_mask) # [H_g, P]
        m = tl.maximum(m, tl.max(scores, axis=1))

    acc = tl.full((H_g, D), value=0.0, dtype=tl.float32)
    l = tl.full((H_g,), value=0.0, dtype=tl.float32)
    for i in range(num_blocks_b):
        bid = tl.load(bt_ptr + pid_b * bt_stride + i)
        seq_mask = i * P + p_offsets < S_b
        scores = _block_scores(bid, q, k_cache_ptr, kvn_stride, kvp_stride,
                               kvh_stride, kvd_stride, pid_h, head_dim, p_offsets, d_mask,
                               d_offsets, seq_mask) # [H_g, P]

        v_ptrs = v_cache_ptr + bid * kvn_stride + p_offsets * kvp_stride + pid_h * kvh_stride
        v_tile_ptrs = v_ptrs[:, None] + d_offsets[None, :] * kvd_stride
        v = tl.load(v_tile_ptrs, mask=seq_mask[:, None] & d_mask[None, :], other=0.0) # [P, D]

        p = tl.exp(scores - m[:, None]) # [H_g, P]
        acc += tl.sum(p[:, :, None] * v[None, :, :].to(tl.float32), axis=1) # [H_g, D]
        l += tl.sum(p, axis=1) # [H_g]
    
    out = acc / l[:, None]
    out_mask = (q_group_offsets < num_queries_per_group)[:, None] & d_mask[None, :]
    output_ptr = output_ptr + pid_h * num_queries_per_group * oh_stride + pid_b * ob_stride
    output_tile_ptrs = output_ptr + q_group_offsets[:, None] * oh_stride + d_offsets[None, :] * od_stride
    tl.store(output_tile_ptrs, out.to(q.dtype), mask=out_mask)


class PagedTritonBackend(PagedGatherBackend):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, metadata: PagedMetadata) -> torch.Tensor:
        # q -> (1, H_q, T_total, D)
        # k,v -> # (1, H_kv, T_total, D)
        if q.shape[2] != metadata.n_computed_tokens.shape[0]:
            # T_total == B <-> every request contributed exactly 1 token <-> pure decode request
            return super().forward(q, k, v, k_cache, v_cache, metadata)

        self._store(k, v, k_cache, v_cache, metadata)
        # (1, H_q, T_total, D)
        output = torch.empty_like(q)

        num_queries_per_group = triton.cdiv(q.shape[1], k.shape[1])
        grid = (q.shape[2], k.shape[1])

        paged_kernel[grid](
            q, q.stride(2), q.stride(1), q.stride(3), k_cache, v_cache,
            k_cache.stride(-4), k_cache.stride(-3), k_cache.stride(-2), k_cache.stride(-1),
            metadata.block_table, metadata.block_table.stride(0), q.shape[-1],
            num_queries_per_group, metadata.n_computed_tokens, output, output.stride(1),
            output.stride(2), output.stride(3),
            triton.next_power_of_2(num_queries_per_group),
            triton.next_power_of_2(k_cache.shape[-3]), triton.next_power_of_2(q.shape[-1])
        )
        return output
