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

    def add_request(self, request_id: int):
        if len(self.states) > self.slab.shape[2]:
            raise ValueError("Reached maximum capacity for runner")
        self.states[request_id] = RequestState()
        self.row_of[request_id] = self.free_list.popleft()

    def step(self, request_ids: list[int], input_ids: torch.Tensor) -> torch.Tensor:
        lens = {self.states[request_id].n_computed_tokens for request_id in request_ids}
        assert len(lens) == 1, f"contiguous runner requires uniform length, got {lens} distinct lengths"
        metadata = ContiguousMetadata(next(iter(lens)))
        seq_len = metadata.n_computed_tokens + input_ids.shape[1]
        position_ids = torch.arange(metadata.n_computed_tokens, seq_len, device=input_ids.device).unsqueeze(0)
        ctx = StepContext(self.backend, self.slab[0], self.slab[1], metadata)
        out = self.model(input_ids, ctx, position_ids=position_ids)
        for r in request_ids:
            self.states[r].n_computed_tokens = seq_len
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

    def step(self, request_ids: list[int], input_ids: torch.Tensor) -> torch.Tensor:
        n_computed_tokens = torch.tensor([self.states[r].n_computed_tokens for r in request_ids], device=input_ids.device)
        pos_chunks, slot_chunks = [], []
        for r in request_ids:
            block_size = self.slab.shape[-3]
            n_needed_tokens = self.states[r].n_computed_tokens + input_ids.shape[1]
            n_current_tokens = len(self.block_table[r]) * block_size
            needed_blocks = math.ceil((n_needed_tokens - n_current_tokens) / block_size)
            for _ in range(needed_blocks):
                try:
                    self.block_table[r].append(self.free_list.popleft())
                except IndexError:
                    raise ValueError("Reached maximum capacity for runner")

            tok_pos = self.states[r].n_computed_tokens + torch.arange(input_ids.shape[1], device=input_ids.device)
            bt = torch.tensor(self.block_table[r], device=input_ids.device)
            pos_chunks.append(tok_pos)
            slot_chunks.append(bt[tok_pos // block_size] * block_size + (tok_pos % block_size))

        block_table = torch.tensor([self.block_table[r] for r in request_ids], device=input_ids.device)
        block_table = torch.nn.utils.rnn.pad_sequence(block_table, batch_first=True, padding_value=0, padding_side="right")
        slot_mapping = torch.cat(slot_chunks)
        metadata = PagedMetadata(n_computed_tokens, slot_mapping, block_table)
        seq_len = metadata.n_computed_tokens + input_ids.shape[1]
        position_ids = torch.stack(pos_chunks)
        ctx = StepContext(self.backend, self.slab[0], self.slab[1], metadata)
        out = self.model(input_ids, ctx, position_ids=position_ids)
        for i, r in enumerate(request_ids):
            self.states[r].n_computed_tokens = seq_len[i].item()
        return out

    def finish_request(self, request_id):
        for block in self.block_table[request_id]:
            self.free_list.append(block)
        del self.states[request_id]
        del self.block_table[request_id]
