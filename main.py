import math
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weft.attention.contiguous import ContiguousBackend
from weft.attention.paged import PagedGatherBackend
from weft.inference.generate import generate
from weft.inference.runner import ContiguousRunner, PagedRunner
from weft.models.qwen3 import Qwen3

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 100
BLOCK_SIZE = 8
DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Deliberately different lengths — ragged is the point now.
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
def run_mine(make_runner, prompts, max_new_tokens):
    runner = make_runner(len(prompts), max(p.shape[0] for p in prompts) + max_new_tokens)
    return torch.cat(list(generate(runner, prompts, max_new_tokens)), dim=1)  # [B, N]


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


def check(tok, hf_out, eos_ids, make_runner, prompts, msgs, label):
    my_out = run_mine(make_runner, prompts, MAX_NEW_TOKENS)

    ok = True
    for b, msg in enumerate(msgs):
        hf_row, my_row = hf_out[b].tolist(), my_out[b].tolist()
        # Compare ids up to and including HF's EOS.
        stop = next((i + 1 for i, t in enumerate(hf_row) if t in eos_ids), len(hf_row))
        hf_row, my_row = hf_row[:stop], my_row[:stop]
        if hf_row != my_row:
            ok = False
            div = next(i for i, (a, c) in enumerate(zip(hf_row, my_row)) if a != c)
            print(
                f"[FAIL] {label} | {msg[:32]!r}: diverges at generated token {div}: "
                f"hf={hf_row[div]} mine={my_row[div]}\n"
                f"       hf:   {tok.decode(hf_row[:div + 3])!r}\n"
                f"       mine: {tok.decode(my_row[:div + 3])!r}"
            )
        else:
            print(f"[OK]   {label} | {msg[:32]!r}: {stop} tokens identical")
    return ok


def bench(tok, hf, make_runner, prompts, msgs, label):
    for name, fn in [
        ("hf", lambda n: run_hf(hf, tok, msgs, n, force_full_length=True)),
        (label, lambda n: run_mine(make_runner, prompts, n)),
    ]:
        fn(8)  # warmup
        sync()
        t0 = time.perf_counter()
        out = fn(MAX_NEW_TOKENS)
        sync()
        dt = time.perf_counter() - t0
        n_tokens = out.shape[0] * out.shape[1]
        print(f"{name:>12}: {dt:6.2f}s  {n_tokens / dt:8.1f} tok/s  ({out.shape[0]}x{out.shape[1]} tokens)")


def main():
    print(f"device={DEVICE} dtype={DTYPE}")
    tok, config, hf, mine = load()
    prompts = tokenize(tok, MESSAGES)
    print(f"{len(prompts)} ragged prompts, lengths: {[p.shape[0] for p in prompts]}")

    def make_contiguous(batch_size, max_seq_len):
        backend = ContiguousBackend(config.num_attention_heads, config.num_key_value_heads, config.head_dim)
        return ContiguousRunner(mine, backend, batch_size=batch_size, max_seq_len=max_seq_len)

    def make_paged(batch_size, max_seq_len):
        backend = PagedGatherBackend(config.num_attention_heads, config.num_key_value_heads, config.head_dim)
        # One sequence's worth of blocks per row, plus slack for partial blocks.
        num_blocks = batch_size * (math.ceil(max_seq_len / BLOCK_SIZE) + 1)
        return PagedRunner(mine, backend, num_blocks=num_blocks, block_size=BLOCK_SIZE)

    runners = [
        ("contiguous", make_contiguous),
        ("paged", make_paged)
    ]

    print("\n--- correctness: token-exact vs HF generate (ragged batch) ---")
    hf_out = run_hf(hf, tok, MESSAGES, MAX_NEW_TOKENS)
    eos = hf.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else {eos}
    passed = [(label, f) for label, f in runners if check(tok, hf_out, eos_ids, f, prompts, MESSAGES, label)]

    if not passed:
        print("\nno runner passed — benchmarking a wrong implementation is meaningless, stopping.")
        return

    print("\n--- benchmark: greedy decode ---")
    for label, factory in passed:
        bench(tok, hf, factory, prompts, MESSAGES, label)


if __name__ == "__main__":
    main()
