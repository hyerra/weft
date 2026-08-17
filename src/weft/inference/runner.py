import math
from collections import deque
from typing import Protocol

import torch

from weft.attention.base import AttentionBackend, StepContext
from weft.attention.contiguous import ContiguousMetadata
from weft.attention.paged import PagedMetadata
from weft.cache.state import RequestState


class Runner(Protocol):
    def add_request(self, request_id: int): ...
    def step(self, request_ids: list[int], input_ids: torch.Tensor) -> torch.Tensor: ...
    def finish_request(self, request_id: int): ...


class ContiguousRunner(Runner):
    def __init__(self, model, backend: AttentionBackend, batch_size: int, max_seq_len: int):
        L = model.config.num_hidden_layers
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
        self.backend = backend
        shape = type(backend).kv_cache_shape(batch_size, model.config.num_key_value_heads, max_seq_len, model.config.head_dim)
        self.slab = torch.zeros((2, L, *shape), device=device, dtype=dtype)
        self.states: dict[int, RequestState] = {}
        self.model = model
        self.row_of: dict[int, int] = {}
        self.free_list = deque(range(self.slab.shape[2]))
        self.cursor = 0

    def add_request(self, request_id: int):
        if len(self.states) > self.slab.shape[2]:
            raise ValueError("Reached maximum capacity for runner")
        self.states[request_id] = RequestState()
        self.row_of[request_id] = self.free_list.popleft()

    def step(self, request_ids: list[int], input_ids: list[torch.Tensor]) -> torch.Tensor:
        S = torch.tensor([
            self.states[r].n_computed_tokens + i.shape[0]
            for r, i in zip(request_ids, input_ids)
        ], device=self.slab.device)
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_side="left")
        metadata = ContiguousMetadata(self.cursor, S)
        position_ids = torch.arange(self.cursor, self.cursor + input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        ctx = StepContext(self.backend, self.slab[0], self.slab[1], metadata)
        out = self.model(input_ids, ctx, position_ids=position_ids)[:, -1]
        for r, l in zip(request_ids, S.tolist()):
            self.states[r].n_computed_tokens = l
        self.cursor += input_ids.shape[1]
        return out

    def finish_request(self, request_id: int):
        self.free_list.append(self.row_of[request_id])
        del self.row_of[request_id]
        del self.states[request_id]


class PagedRunner(Runner):
    def __init__(self, model, backend: AttentionBackend, num_blocks: int, block_size: int):
        L = model.config.num_hidden_layers
        device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
        self.backend = backend
        shape = type(backend).kv_cache_shape(num_blocks, block_size, model.config.num_key_value_heads, model.config.head_dim)
        self.slab = torch.zeros((2, L, *shape), device=device, dtype=dtype)
        self.states: dict[int, RequestState] = {}
        self.model = model
        self.block_table: dict[int, list[int]] = {}
        self.free_list = deque(range(self.slab.shape[2]))

    def add_request(self, request_id):
        if not self.free_list:
            raise ValueError("Reached maximum capacity for runner")
        self.states[request_id] = RequestState()
        self.block_table[request_id] = []

    def step(self, request_ids: list[int], input_ids: list[torch.Tensor]) -> torch.Tensor:
        n_computed_tokens = torch.tensor([self.states[r].n_computed_tokens for r in request_ids], device=self.slab.device)
        T_b = torch.tensor([i.shape[0] for i in input_ids], device=self.slab.device)
        S = n_computed_tokens + T_b
        cu_tokens = torch.cat([T_b.new_zeros(1), T_b.cumsum(0)])
        pos_chunks, slot_chunks = [], []
        for r, input_id in zip(request_ids, input_ids):
            block_size = self.slab.shape[-3]
            n_needed_tokens = self.states[r].n_computed_tokens + input_id.shape[0]
            n_current_tokens = len(self.block_table[r]) * block_size
            needed_blocks = math.ceil((n_needed_tokens - n_current_tokens) / block_size)
            for _ in range(needed_blocks):
                try:
                    self.block_table[r].append(self.free_list.popleft())
                except IndexError:
                    raise ValueError("Reached maximum capacity for runner")

            tok_pos = self.states[r].n_computed_tokens + torch.arange(input_id.shape[0], device=self.slab.device)
            bt = torch.tensor(self.block_table[r], device=self.slab.device)
            pos_chunks.append(tok_pos)
            slot_chunks.append(bt[tok_pos // block_size] * block_size + (tok_pos % block_size))

        input_ids = torch.cat(input_ids).unsqueeze(0)
        block_table = [torch.tensor(self.block_table[r], device=input_ids.device) for r in request_ids]
        block_table = torch.nn.utils.rnn.pad_sequence(block_table, batch_first=True, padding_value=0, padding_side="right")
        slot_mapping = torch.cat(slot_chunks)
        metadata = PagedMetadata(n_computed_tokens, slot_mapping, block_table, cu_tokens)
        position_ids = torch.cat(pos_chunks).unsqueeze(0)
        ctx = StepContext(self.backend, self.slab[0], self.slab[1], metadata)
        out = self.model(input_ids, ctx, position_ids=position_ids)[:, cu_tokens[1:]-1].squeeze(dim=0)
        for r, l in zip(request_ids, S.tolist()):
            self.states[r].n_computed_tokens = l
        return out

    def finish_request(self, request_id):
        for block in self.block_table[request_id]:
            self.free_list.append(block)
        del self.states[request_id]
        del self.block_table[request_id]
