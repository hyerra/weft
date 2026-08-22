from dataclasses import dataclass
from typing import Protocol

import torch

from weft.attention.base import AttentionBackend, StepContext
from weft.attention.contiguous import ContiguousMetadata
from weft.attention.paged import PagedMetadata


@dataclass
class ScheduledRequest:
    request_id: int
    input_ids: torch.Tensor
    num_computed_tokens: int
    block_table: list[int]


class Runner(Protocol):
    def step(self, scheduled: list[ScheduledRequest]) -> torch.Tensor: ...


class ContiguousRunner(Runner):
    def __init__(self, model, backend: AttentionBackend, batch_size: int, max_seq_len: int):
        L = model.config.num_hidden_layers
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
        self.backend = backend
        shape = type(backend).kv_cache_shape(batch_size, model.config.num_key_value_heads, max_seq_len, model.config.head_dim)
        self.slab = torch.zeros((2, L, *shape), device=device, dtype=dtype)
        self.model = model
        self.cursor = 0

    def step(self, scheduled: list[ScheduledRequest]) -> torch.Tensor:
        # Trick to allow new batches once previous batch is finished.
        fresh = [r.num_computed_tokens == 0 for r in scheduled]
        assert all(fresh) or not any(fresh), \
             "contiguous runner serves whole cohorts: got a fresh request mixed into a running batch"
        if all(fresh):
            self.cursor = 0
        S = torch.tensor([
            r.num_computed_tokens + r.input_ids.shape[0]
            for r in scheduled
        ], device=self.slab.device)
        input_ids = torch.nn.utils.rnn.pad_sequence([r.input_ids for r in scheduled], batch_first=True, padding_side="left")
        metadata = ContiguousMetadata(self.cursor, S, [r.block_table[0] for r in scheduled])
        position_ids = torch.arange(self.cursor, self.cursor + input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        ctx = StepContext(self.backend, self.slab[0], self.slab[1], metadata)
        out = self.model(input_ids, ctx, position_ids=position_ids)[:, -1]
        self.cursor += input_ids.shape[1]
        return out


class PagedRunner(Runner):
    def __init__(self, model, backend: AttentionBackend, num_blocks: int, block_size: int):
        L = model.config.num_hidden_layers
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
        self.backend = backend
        shape = type(backend).kv_cache_shape(num_blocks, block_size, model.config.num_key_value_heads, model.config.head_dim)
        self.slab = torch.zeros((2, L, *shape), device=device, dtype=dtype)
        self.model = model

    def step(self, scheduled: list[ScheduledRequest]) -> torch.Tensor:
        num_computed_tokens = torch.tensor([r.num_computed_tokens for r in scheduled], device=self.slab.device)
        T_b = torch.tensor([r.input_ids.shape[0] for r in scheduled], device=self.slab.device)
        cu_tokens = torch.cat([T_b.new_zeros(1), T_b.cumsum(0)])
        block_size = self.slab.shape[-3]
        pos_chunks, slot_chunks = [], []
        for r in scheduled:
            tok_pos = r.num_computed_tokens + torch.arange(r.input_ids.shape[0], device=self.slab.device)
            bt = torch.tensor(r.block_table, device=self.slab.device)
            pos_chunks.append(tok_pos)
            slot_chunks.append(bt[tok_pos // block_size] * block_size + (tok_pos % block_size))

        input_ids = torch.cat([r.input_ids for r in scheduled]).unsqueeze(0)
        block_table = [torch.tensor(r.block_table, device=input_ids.device) for r in scheduled]
        block_table = torch.nn.utils.rnn.pad_sequence(block_table, batch_first=True, padding_value=0, padding_side="right")
        slot_mapping = torch.cat(slot_chunks)
        metadata = PagedMetadata(num_computed_tokens, slot_mapping, block_table, cu_tokens)
        position_ids = torch.cat(pos_chunks).unsqueeze(0)
        ctx = StepContext(self.backend, self.slab[0], self.slab[1], metadata)
        out = self.model(input_ids, ctx, position_ids=position_ids)[:, cu_tokens[1:]-1].squeeze(dim=0)
        return out
