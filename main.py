import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from weft.inference.generate import generate
from weft.models.qwen3 import Qwen3

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 100
DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

CANDIDATE_MESSAGES = [
    "Who are you?",
    "What is 2 + 2?",
    "Tell me a joke.",
    "Name a big city.",
    "What color is the sky?",
    "Give me one word.",
]


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def load():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    config = AutoConfig.from_pretrained(MODEL_ID)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=DTYPE).to(DEVICE).eval()
    mine = Qwen3.from_config(config, DTYPE)
    mine.load_state_dict(hf.state_dict(), strict=False)
    mine = mine.to(DEVICE)

    # from weft.attention.base import StepContext
    # from weft.attention.paged import PagedMetadata, PagedGatherBackend
    # with torch.inference_mode():
    #     shape = ContiguousBackend.kv_cache_shape(1, config.num_key_value_heads, 100, config.head_dim)
    #     slab = torch.zeros((2, config.num_hidden_layers, *shape), device=DEVICE, dtype=DTYPE)
    #     md = PagedMetadata(0, torch.)
    #     md = ContiguousMetadata(0)
    #     ctx = StepContext(PagedGatherBackend(config.num_attention_heads, config.num_key_value_heads, config.head_dim), slab[0], slab[1], md)
    #     hf_res = hf(input_ids=torch.tensor([[100, 120, 185]], device=DEVICE)).logits
    #     mine_res = mine(torch.tensor([[100, 120, 185]], device=DEVICE), ctx, position_ids=torch.tensor([[0, 1, 2]], device=DEVICE))
    #     print("mine", mine_res, flush=True)
    #     print("hf", hf_res, flush=True)
    #     print("diff", (mine_res - hf_res).abs().max())


    return tok, config, hf, mine


def pick_prompts(tok):
    by_len = {}
    for msg in CANDIDATE_MESSAGES:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": msg}],
            add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        )["input_ids"][0]
        by_len.setdefault(ids.shape[0], []).append((msg, ids))
    msgs, prompts = zip(*max(by_len.values(), key=len))
    assert len(prompts) >= 2, "no two candidates tokenize to equal length; adjust CANDIDATE_MESSAGES"
    return list(msgs), [p.to(DEVICE) for p in prompts]


@torch.inference_mode()
def run_mine(make_runner, prompts, max_new_tokens):
    runner = make_runner(200, 8)
    return torch.cat(list(generate(runner, prompts, max_new_tokens)), dim=1)  # [B, N]


@torch.inference_mode()
def run_hf(hf, tok, msgs, max_new_tokens, force_full_length=False):
    batch = tok.apply_chat_template(
        [[{"role": "user", "content": m}] for m in msgs],
        add_generation_prompt=True, tokenize=True, return_dict=True,
        padding=True, padding_side="left", return_tensors="pt",
    ).to(DEVICE)
    kwargs = dict(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens)
    if force_full_length:
        kwargs["min_new_tokens"] = max_new_tokens
    out = hf.generate(**batch, **kwargs)
    return out[:, batch["input_ids"].shape[1]:]


def check(tok, hf, make_runner, prompts, msgs):
    hf_out = run_hf(hf, tok, msgs, MAX_NEW_TOKENS)
    my_out = run_mine(make_runner, prompts, MAX_NEW_TOKENS)

    eos_ids = hf.generation_config.eos_token_id
    eos_ids = set(eos_ids) if isinstance(eos_ids, list) else {eos_ids}

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
                f"[FAIL] {msg!r}: diverges at generated token {div}: hf={hf_row[div]} mine={my_row[div]}\n"
                f"       hf:   {tok.decode(hf_row[:div + 3])!r}\n"
                f"       mine: {tok.decode(my_row[:div + 3])!r}"
            )
        else:
            print(f"[OK]   {msg!r}: {stop} tokens identical -> {tok.decode(my_row, skip_special_tokens=True)[:60]!r}")
    return ok


def bench(tok, hf, make_runner, prompts, msgs):
    results = {}
    for name, fn in [
        ("hf", lambda n: run_hf(hf, tok, msgs, n, force_full_length=True)),
        ("mine", lambda n: run_mine(make_runner, prompts, n)),
    ]:
        fn(8)  # warmup
        sync()
        t0 = time.perf_counter()
        out = fn(MAX_NEW_TOKENS)
        sync()
        dt = time.perf_counter() - t0
        n_tokens = out.shape[0] * out.shape[1]
        results[name] = (dt, n_tokens / dt)
        print(f"{name:>5}: {dt:6.2f}s  {n_tokens / dt:8.1f} tok/s  ({out.shape[0]}x{out.shape[1]} tokens)")
    return results


def main():
    print(f"device={DEVICE} dtype={DTYPE}")
    tok, config, hf, mine = load()
    msgs, prompts = pick_prompts(tok)
    print(f"batch of {len(prompts)} equal-length prompts ({prompts[0].shape[0]} tokens): {msgs}")

    def make_paged(num_blocks, block_size):
        from weft.attention.paged import PagedGatherBackend
        from weft.inference.runner import PagedRunner
        backend = PagedGatherBackend(config.num_attention_heads, config.num_key_value_heads, config.head_dim)
        return PagedRunner(mine, backend, num_blocks=num_blocks, block_size=block_size)

    print("\n--- correctness: token-exact vs HF generate ---")
    if not check(tok, hf, make_paged, prompts, msgs):
        print("\ncheck FAILED — benchmark of a wrong implementation is meaningless, stopping.")
        return

    print("\n--- benchmark: greedy decode ---")
    bench(tok, hf, make_paged, prompts, msgs)


if __name__ == "__main__":
    main()
