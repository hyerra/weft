import math

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weft.attention.contiguous import ContiguousBackend
from weft.attention.paged import PagedGatherBackend
from weft.inference.engine import Engine
from weft.inference.runner import ContiguousRunner, PagedRunner
from weft.inference.scheduler import BlockManager, ContinuousScheduler, StaticScheduler

DEFAULT_BLOCK_SIZE = 8


def load_model(model_id: str, dtype: torch.dtype, device: str):
    from weft.models.qwen3 import Qwen3

    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    config = AutoConfig.from_pretrained(model_id)
    hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    mine = Qwen3.from_config(config, dtype)
    mine.load_state_dict(hf.state_dict(), strict=False)
    eos = hf.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else {eos}
    return tok, hf, mine.to(device), eos_ids


def blocks_for(n_requests: int, max_seq_len: int, block_size: int = DEFAULT_BLOCK_SIZE) -> int:
    return n_requests * (math.ceil(max_seq_len / block_size) + 1)


def make_contiguous_engine(model, eos_ids, batch_size, max_seq_len, max_tokens,
                           ignore_eos=False):
    config = model.config
    manager = BlockManager(num_blocks=batch_size, block_size=max_seq_len)
    backend = ContiguousBackend(config.num_attention_heads, config.num_key_value_heads, config.head_dim)
    runner = ContiguousRunner(model, backend, batch_size=manager.num_blocks, max_seq_len=manager.block_size)
    scheduler = StaticScheduler(manager, max_seqs=batch_size, max_tokens=batch_size * max_seq_len)
    return Engine(runner, scheduler, set() if ignore_eos else eos_ids, max_tokens)


def make_paged_engine(model, eos_ids, num_blocks, max_tokens,
                      backend_cls=PagedGatherBackend, ignore_eos=False,
                      block_size=DEFAULT_BLOCK_SIZE):
    config = model.config
    manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
    backend = backend_cls(config.num_attention_heads, config.num_key_value_heads, config.head_dim)
    runner = PagedRunner(model, backend, num_blocks=manager.num_blocks, block_size=manager.block_size)
    scheduler = ContinuousScheduler(manager, max_seqs=num_blocks, max_tokens=num_blocks * block_size)
    return Engine(runner, scheduler, set() if ignore_eos else eos_ids, max_tokens)


def make_triton_engine(model, eos_ids, num_blocks, max_tokens,
                       ignore_eos=False, block_size=DEFAULT_BLOCK_SIZE):
    from weft.attention.paged_triton import PagedTritonBackend

    return make_paged_engine(model, eos_ids, num_blocks, max_tokens,
                             backend_cls=PagedTritonBackend, ignore_eos=ignore_eos,
                             block_size=block_size)
