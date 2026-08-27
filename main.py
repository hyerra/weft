import argparse
import functools
import json
import subprocess
import time
from pathlib import Path

import modal
import torch

from weft.benchmark.factories import blocks_for, load_model, make_contiguous_engine, make_paged_engine, make_triton_engine
from weft.inference.generate import generate

MODEL_ID = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 500
DTYPE = torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

MESSAGES = [
    "Hi.",
    "What is 2 + 2?",
    "Tell me a joke about a cat who is learning to play the piano.",
]

modal_app = modal.App("weft-gpu")
modal_image = (modal.Image.debian_slim().uv_sync()
               .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
               .add_local_python_source("weft"))
hf_cache = modal.Volume.from_name("weft-hf-cache", create_if_missing=True)


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
def run_hf(hf, tok, msgs, max_new_tokens):
    batch = tok.apply_chat_template(
        [[{"role": "user", "content": m}] for m in msgs],
        add_generation_prompt=True, tokenize=True, return_dict=True,
        padding=True, return_tensors="pt",
    ).to(DEVICE)
    out = hf.generate(**batch, do_sample=False, num_beams=1, max_new_tokens=max_new_tokens)
    return out[:, batch["input_ids"].shape[1]:]


def check(tok, hf_out, eos_ids, make_engine, prompts, msgs, label, arrivals=None):
    my_out = run_mine(make_engine, prompts, MAX_NEW_TOKENS, arrivals)

    ok = True
    for b, msg in enumerate(msgs):
        hf_row, my_row = hf_out[b].tolist(), my_out[b]
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


def run_gate():
    print(f"device={DEVICE} dtype={DTYPE}")
    tok, hf, mine, eos_ids = load_model(MODEL_ID, DTYPE, DEVICE)
    prompts = tokenize(tok, MESSAGES)
    print(f"{len(prompts)} ragged prompts, lengths: {[p.shape[0] for p in prompts]}")

    def paged(n, s, t, **kw):
        return make_paged_engine(mine, eos_ids, blocks_for(n, s), t, **kw)

    def triton(n, s, t, **kw):
        return make_triton_engine(mine, eos_ids, blocks_for(n, s), t, **kw)

    engines = [
        ("contiguous", functools.partial(make_contiguous_engine, mine, eos_ids)),
        ("paged", paged),
    ]
    if torch.cuda.is_available():
        engines.append(("triton", triton))

    print("\n--- correctness: token-exact vs HF generate (ragged batch, engine) ---")
    hf_out = run_hf(hf, tok, MESSAGES, MAX_NEW_TOKENS)
    ok = all([check(tok, hf_out, eos_ids, f, prompts, MESSAGES, label) for label, f in engines])

    # Continuous batching verification: the last request is admitted mid-generation,
    ok &= check(tok, hf_out, eos_ids, paged, prompts, MESSAGES, "paged+midflight",
                arrivals=[0, 0, 10])
    return ok


@modal_app.function(
    image=modal_image, gpu="A10G", timeout=1800,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def gate_remote():
    return run_gate()


@modal_app.function(
    image=modal_image, gpu="L40S", timeout=5400,
    volumes={"/root/.cache/huggingface": hf_cache},
)
def collect_remote(names: list[str]) -> dict:
    from weft.benchmark.experiments import collect

    return collect(names)


def main():
    parser = argparse.ArgumentParser(description="weft correctness check + experiment collection")
    from weft.benchmark.experiments import EXPERIMENTS

    parser.add_argument("--modal", action="store_true", help="run the gate on a Modal GPU")
    parser.add_argument("--collect", nargs="+", metavar="EXP", choices=[*EXPERIMENTS, "all"],
                        help="run experiments on Modal and write results/*.json (e.g. --collect e1 e3, or all)")
    args = parser.parse_args()

    if args.collect:
        names = list(EXPERIMENTS) if "all" in args.collect else args.collect
        with modal.enable_output(), modal_app.run():
            results = collect_remote.remote(names)
        stamp = time.strftime("%Y%m%d")
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
        for stem, payload in results.items():
            payload["meta"]["git_sha"] = sha
            path = Path("results") / f"{stem}_{stamp}.json"
            path.write_text(json.dumps(payload, indent=1))
            print(f"wrote {path}")
    elif args.modal:
        with modal.enable_output(), modal_app.run():
            gate_remote.remote()
    else:
        run_gate()


if __name__ == "__main__":
    main()
