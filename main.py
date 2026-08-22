import math
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weft.attention.contiguous import ContiguousBackend
from weft.attention.paged import PagedGatherBackend
from weft.inference.engine import Engine
from weft.inference.generate import generate
from weft.inference.runner import ContiguousRunner, PagedRunner
from weft.inference.scheduler import BlockManager, ContinuousScheduler, StaticScheduler
from weft.models.qwen3 import Qwen3

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 100
BLOCK_SIZE = 8
DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

MESSAGES = [
    "Hi.",
    "What is 2 + 2?",
    "Tell me a joke about a cat who is learning to play the piano.",
]


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    config = AutoConfig.from_pretrained(MODEL_ID)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    mine = Qwen3.from_config(config, DTYPE)
    mine.load_state_dict(hf.state_dict(), strict=False)
    return tok, config, hf, mine.to(DEVICE)


def tokenize(tok, msgs):
    # Unpadded, one 1-D tensor per prompt: the ragged ingestion form.
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": m}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        )["input_ids"][0].to(DEVICE)
        for m in msgs
    ]


@torch.inference_mode()
def run_mine(make_engine, prompts, max_new_tokens, arrivals=None):
    engine = make_engine(len(prompts), max(p.shape[0] for p in prompts) + max_new_tokens, max_new_tokens)
    return generate(engine, prompts, arrivals)  # list[list[int]], EOS-terminated


@torch.inference_mode()
def run_hf(hf, tok, msgs, max_new_tokens, force_full_length=False):
    batch = tok.apply_chat_template(
        [[{"role": "user", "content": m}] for m in msgs],
        add_generation_prompt=True, tokenize=True, return_dict=True,
        padding=True, return_tensors="pt",
    ).to(DEVICE)
    kwargs = dict(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens)
    if force_full_length:
        kwargs["min_new_tokens"] = max_new_tokens
    out = hf.generate(**batch, **kwargs)
    return out[:, batch["input_ids"].shape[1]:]


def check(tok, hf_out, eos_ids, make_engine, prompts, msgs, label, arrivals=None):
    my_out = run_mine(make_engine, prompts, MAX_NEW_TOKENS, arrivals)

    ok = True
    for b, msg in enumerate(msgs):
        hf_row, my_row = hf_out[b].tolist(), my_out[b]
        # Our stack stops at EOS now, like HF: rows should match exactly,
        # length included, once HF's post-EOS padding is stripped.
        stop = next((i + 1 for i, t in enumerate(hf_row) if t in eos_ids), len(hf_row))
        hf_row = hf_row[:stop]
        if hf_row != my_row:
            ok = False
            div = next(
                (i for i, (a, c) in enumerate(zip(hf_row, my_row)) if a != c),
                min(len(hf_row), len(my_row)),  # lengths differ, contents agree
            )
            print(
                f"[FAIL] {label} | {msg[:32]!r}: diverges at generated token {div} "
                f"(hf len {len(hf_row)}, mine len {len(my_row)})\n"
                f"       hf:   {tok.decode(hf_row[:div + 3])!r}\n"
                f"       mine: {tok.decode(my_row[:div + 3])!r}"
            )
        else:
            print(f"[OK]   {label} | {msg[:32]!r}: {len(my_row)} tokens identical")
    return ok


def bench(tok, hf, make_engine, prompts, msgs, label):
    for name, fn, count in [
        ("hf", lambda n: run_hf(hf, tok, msgs, n, force_full_length=True),
         lambda out: out.shape[0] * out.shape[1]),
        (label, lambda n: run_mine(make_engine, prompts, n),
         lambda out: sum(len(r) for r in out)),
    ]:
        fn(8)  # warmup
        sync()
        t0 = time.perf_counter()
        out = fn(MAX_NEW_TOKENS)
        sync()
        dt = time.perf_counter() - t0
        n_tokens = count(out)
        print(f"{name:>12}: {dt:6.2f}s  {n_tokens / dt:8.1f} tok/s  ({n_tokens} tokens)")


def main():
    print(f"device={DEVICE} dtype={DTYPE}")
    tok, config, hf, mine = load()
    prompts = tokenize(tok, MESSAGES)
    print(f"{len(prompts)} ragged prompts, lengths: {[p.shape[0] for p in prompts]}")

    eos = hf.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else {eos}

    def make_contiguous_engine(batch_size, max_seq_len, max_tokens):
        # Rows are blocks with P = max_seq_len: one block per request.
        manager = BlockManager(num_blocks=batch_size, block_size=max_seq_len)
        backend = ContiguousBackend(config.num_attention_heads, config.num_key_value_heads, config.head_dim)
        runner = ContiguousRunner(mine, backend, batch_size=manager.num_blocks, max_seq_len=manager.block_size)
        scheduler = StaticScheduler(manager, max_seqs=batch_size,
                                    max_tokens=batch_size * max_seq_len)
        return Engine(runner, scheduler, eos_ids, max_tokens)

    def make_paged_engine(batch_size, max_seq_len, max_tokens):
        num_blocks = batch_size * (math.ceil(max_seq_len / BLOCK_SIZE) + 1)
        manager = BlockManager(num_blocks=num_blocks, block_size=BLOCK_SIZE)
        backend = PagedGatherBackend(config.num_attention_heads, config.num_key_value_heads, config.head_dim)
        runner = PagedRunner(mine, backend, num_blocks=manager.num_blocks, block_size=manager.block_size)
        scheduler = ContinuousScheduler(manager, max_seqs=batch_size,
                                        max_tokens=batch_size * max_seq_len)
        return Engine(runner, scheduler, eos_ids, max_tokens)

    engines = [
        ("contiguous", make_contiguous_engine),
        ("paged", make_paged_engine),
    ]

    print("\n--- correctness: token-exact vs HF generate (ragged batch, engine) ---")
    hf_out = run_hf(hf, tok, MESSAGES, MAX_NEW_TOKENS)
    passed = [(label, f) for label, f in engines if check(tok, hf_out, eos_ids, f, prompts, MESSAGES, label)]

    # Continuous batching verification: the last request is admitted mid-generation
    check(tok, hf_out, eos_ids, make_paged_engine, prompts, MESSAGES, "paged+midflight",
          arrivals=[0, 0, 10])

    if not passed:
        print("\nno engine passed — benchmarking a wrong implementation is meaningless, stopping.")
        return

    print("\n--- benchmark: greedy decode ---")
    for label, factory in passed:
        bench(tok, hf, factory, prompts, MESSAGES, label)


if __name__ == "__main__":
    main()
