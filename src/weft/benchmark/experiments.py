import math
import time

import torch

from weft.attention.paged import PagedGatherBackend, PagedMetadata
from weft.benchmark import factories, metrics, workloads
from weft.benchmark.factories import blocks_for, make_contiguous_engine, make_paged_engine, make_triton_engine
from weft.benchmark.metrics import cuda_time_ms, run_timed

MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"]
DTYPE = torch.bfloat16
PEAK_GBPS = {"NVIDIA A10G": 600.0, "NVIDIA L40S": 864.0}
SEED = 0


def _peak_gbps():
    return PEAK_GBPS[torch.cuda.get_device_name()]

def _engines(model, eos, n, max_seq_len, max_tokens, block_size=None):
    kw = {"block_size": block_size} if block_size else {}
    nb = blocks_for(n, max_seq_len, block_size or 8)
    return {
        "contiguous": lambda: make_contiguous_engine(model, eos, n, max_seq_len, max_tokens, ignore_eos=True),
        "paged": lambda: make_paged_engine(model, eos, nb, max_tokens, ignore_eos=True, **kw),
        "triton": lambda: make_triton_engine(model, eos, nb, max_tokens, ignore_eos=True, **kw),
    }


@torch.inference_mode()
def _hf_row(hf, prompts, new_tokens, device):
    input_ids = torch.stack(prompts)
    attn = torch.ones_like(input_ids)

    def prefill():
        hf(input_ids=input_ids, attention_mask=attn)

    metrics.sync(device)
    t0 = time.perf_counter()
    prefill()
    metrics.sync(device)
    ttft_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    hf.generate(input_ids=input_ids, attention_mask=attn, do_sample=False, num_beams=1,
                max_new_tokens=new_tokens, min_new_tokens=new_tokens)
    metrics.sync(device)
    total_s = time.perf_counter() - t0
    n = len(prompts) * new_tokens
    return {"ttft_s": ttft_s, "total_s": total_s, "output_tps": n / total_s}


def e1_backend_sweep(ctx):
    device, model, hf, eos = ctx["device"], ctx["model"], ctx["hf"], ctx["eos_ids"]
    vocab = model.config.vocab_size
    new_tokens, batch = 128, 4
    out = {"prompt_lengths": [128, 512, 1024, 2048], "batch_size": batch,
           "new_tokens": new_tokens, "runs": []}

    for p_len in out["prompt_lengths"]:
        prompts = workloads.synthetic_prompts([p_len] * batch, vocab, device, seed=SEED)
        row = {"prompt_len": p_len, "backends": {}}
        for name, make in _engines(model, eos, batch, p_len + new_tokens, new_tokens).items():
            run_timed(make(), prompts[:1], device)
            engine = make()
            trace = run_timed(engine, prompts, device)
            row["backends"][name] = trace.to_json() | {
                "ttft_s": trace.tick_s[0],
                "input_tps": batch * p_len / trace.tick_s[0],
            }
            del engine, trace
            row["backends"][name]["mem_gb"] = metrics.reclaim_mem_gb(device)
            print(f"[mem] e1 p_len={p_len} {name}: {row['backends'][name]['mem_gb']:.2f} GB live", flush=True)
        row["backends"]["hf"] = _hf_row(hf, prompts, new_tokens, device)
        print(f"[mem] e1 p_len={p_len} hf: {torch.cuda.memory_allocated() / 1e9:.2f} GB live", flush=True)
        out["runs"].append(row)
    return out


def e2_batch_sweep(ctx):
    device, model, hf, eos = ctx["device"], ctx["model"], ctx["hf"], ctx["eos_ids"]
    vocab = model.config.vocab_size
    p_len, new_tokens = 256, 128
    out = {"batch_sizes": [1, 2, 4, 8, 16, 32], "prompt_len": p_len,
           "new_tokens": new_tokens, "runs": []}

    for batch in out["batch_sizes"]:
        prompts = workloads.synthetic_prompts([p_len] * batch, vocab, device, seed=SEED)
        row = {"batch_size": batch, "backends": {}}
        for name, make in _engines(model, eos, batch, p_len + new_tokens, new_tokens).items():
            run_timed(make(), prompts[:1], device)
            engine = make()
            trace = run_timed(engine, prompts, device)
            row["backends"][name] = {"output_tps": trace.output_tps, "total_s": trace.total_s}
            del engine, trace
            row["backends"][name]["mem_gb"] = metrics.reclaim_mem_gb(device)
            print(f"[mem] e2 B={batch} {name}: {row['backends'][name]['mem_gb']:.2f} GB live", flush=True)
        row["backends"]["hf"] = _hf_row(hf, prompts, new_tokens, device)
        out["runs"].append(row)
    return out


def e3_attend_microbench(ctx):
    from weft.attention.paged_triton import PagedTritonBackend

    device = ctx["device"]
    H_q, H_kv, D, P, B = 16, 8, 128, 8, 8
    bytes_per_token = 2 * H_kv * D * DTYPE.itemsize
    peak = _peak_gbps()

    # achievable-bandwidth reference: a pure d2d copy, plotted as the ceiling
    a = torch.empty(256 * 2**20, dtype=torch.uint8, device=device)
    b = torch.empty_like(a)
    copy_ms = cuda_time_ms(lambda: b.copy_(a))
    memcpy_gbps = 2 * a.numel() / (copy_ms / 1e3) / 1e9
    del a, b

    out = {"S": [512, 1024, 2048, 4096, 8192, 16384, 32768], "B": B,
           "peak_gbps": peak, "memcpy_gbps": memcpy_gbps,
           "memcpy_pct_peak": 100 * memcpy_gbps / peak, "runs": []}

    gather = PagedGatherBackend(H_q, H_kv, D)
    triton_b = PagedTritonBackend(H_q, H_kv, D)

    for S in out["S"]:
        blocks_per_req = math.ceil(S / P)
        N = B * blocks_per_req + 1
        g = torch.Generator(device="cpu").manual_seed(SEED)
        k_cache = torch.randn(N, P, H_kv, D, generator=g, dtype=DTYPE).to(device)
        v_cache = torch.randn(N, P, H_kv, D, generator=g, dtype=DTYPE).to(device)
        q = torch.randn(1, H_q, B, D, generator=g, dtype=DTYPE).to(device)
        k = torch.randn(1, H_kv, B, D, generator=g, dtype=DTYPE).to(device)
        v = torch.randn(1, H_kv, B, D, generator=g, dtype=DTYPE).to(device)

        tables = torch.arange(B * blocks_per_req, dtype=torch.int64, device=device).reshape(B, blocks_per_req)
        n_computed = torch.full((B,), S - 1, dtype=torch.int64, device=device)
        slots = tables[:, -1] * P + (S - 1) % P
        cu = torch.arange(B + 1, dtype=torch.int64, device=device)
        md = PagedMetadata(n_computed, slots, tables, cu)

        row = {"S": S}
        for name, backend in [("gather", gather), ("triton", triton_b)]:
            ms = cuda_time_ms(lambda: backend.forward(q, k, v, k_cache, v_cache, md))
            gbps = (B * S * bytes_per_token) / (ms / 1e3) / 1e9
            row[name] = {"ms": ms, "gbps": gbps, "pct_peak": 100 * gbps / peak}
        out["runs"].append(row)
    return out


def _arrival_configs(model, eos):
    max_seqs = 8
    new_tokens = 64
    s_max = max(workloads.PROMPT_MIXES["mixed"]) + new_tokens
    blocks = blocks_for(max_seqs, s_max)
    return {
        "static_contiguous": lambda: make_contiguous_engine(
            model, eos, max_seqs, s_max, new_tokens, ignore_eos=True),
        "continuous_gather": lambda: make_paged_engine(
            model, eos, blocks, new_tokens, ignore_eos=True),
        "continuous_triton": lambda: make_triton_engine(
            model, eos, blocks, new_tokens, ignore_eos=True),
    }, new_tokens


def e4_arrival_stream(ctx):
    device, model, eos = ctx["device"], ctx["model"], ctx["eos_ids"]
    vocab = model.config.vocab_size
    wl = workloads.poisson_arrivals(n=64, mean_interarrival_ticks=2.0, mix="mixed", seed=SEED)
    prompts = workloads.synthetic_prompts(wl.lengths, vocab, device, seed=SEED)
    configs, new_tokens = _arrival_configs(model, eos)

    out = {"n_requests": len(prompts), "lengths": wl.lengths, "arrivals": wl.arrivals,
           "new_tokens": new_tokens, "configs": {}}
    for name, make in configs.items():
        run_timed(make(), prompts[:2], device, arrivals=[0, 1])  # warmup
        trace = run_timed(make(), prompts, device, arrivals=wl.arrivals)
        out["configs"][name] = trace.to_json()
    return out


def e5_preemption(ctx):
    device, model, eos = ctx["device"], ctx["model"], ctx["eos_ids"]
    vocab = model.config.vocab_size
    wl = workloads.poisson_arrivals(n=32, mean_interarrival_ticks=1.0, mix="mixed", seed=SEED)
    prompts = workloads.synthetic_prompts(wl.lengths, vocab, device, seed=SEED)

    new_tokens = 64
    s_max = max(workloads.PROMPT_MIXES["mixed"]) + new_tokens
    full_pool = blocks_for(len(prompts), s_max)

    out = {"pool_fractions": [1.0, 0.75, 0.5, 0.35, 0.25, 0.2], "full_pool_blocks": full_pool, "runs": []}
    assert out["pool_fractions"][0] == 1.0, "unpressured run must come first to compute reference_tokens"
    reference_tokens = None
    run_timed(make_triton_engine(model, eos, full_pool, new_tokens, ignore_eos=True),
              prompts[:2], device, arrivals=[0, 1])  # warmup
    for frac in out["pool_fractions"]:
        engine = make_triton_engine(model, eos, max(1, int(full_pool * frac)), new_tokens,
                                    ignore_eos=True)
        trace = run_timed(engine, prompts, device, arrivals=wl.arrivals)
        sched = engine.scheduler
        tokens = {rid: r.tokens for rid, r in trace.requests.items()}
        if frac == 1.0:
            reference_tokens = tokens
        arrival_order = sorted(trace.requests, key=lambda rid: (trace.requests[rid].arrival_tick, rid))
        out["runs"].append({
            "fraction": frac,
            "output_tps": trace.output_tps,
            "total_s": trace.total_s,
            "preemptions": sched.total_preemptions,
            "recomputed_tokens": sched.total_recomputed_tokens,
            "recompute_overhead_pct": 100 * sched.total_recomputed_tokens / trace.total_generated,
            "tokens_identical_to_unpressured": tokens == reference_tokens,
            "completion_follows_arrival": trace.completion_order
                == sorted(trace.completion_order, key=arrival_order.index),
        })
    return out


def e6_block_size_sweep(ctx):
    device, model, eos = ctx["device"], ctx["model"], ctx["eos_ids"]
    vocab = model.config.vocab_size
    batch, p_len, new_tokens = 8, 256, 128
    seq_len = p_len + new_tokens
    p_values = [8, 16, 32, 64, 128, 256]

    prompts = workloads.synthetic_prompts([p_len] * batch, vocab, device, seed=SEED)
    out = {"block_sizes": p_values, "batch_size": batch, "seq_len_per_req": seq_len, "runs": []}
    for P in p_values:
        waste_tokens = math.ceil(seq_len / P) * P - seq_len
        make = lambda: make_triton_engine(model, eos, blocks_for(batch, seq_len, P), new_tokens,
                                          ignore_eos=True, block_size=P)
        run_timed(make(), prompts[:1], device)  # warmup
        trace = run_timed(make(), prompts, device)
        out["runs"].append({"block_size": P, "waste_pct": 100 * waste_tokens / seq_len,
                            "output_tps": trace.output_tps})
    return out


def e7_tick_profile(ctx):
    device, model, eos = ctx["device"], ctx["model"], ctx["eos_ids"]
    vocab = model.config.vocab_size
    batch, p_len, new_tokens = 8, 256, 128
    prompts = workloads.synthetic_prompts([p_len] * batch, vocab, device, seed=SEED)
    engine = make_triton_engine(model, eos, blocks_for(batch, p_len + new_tokens), new_tokens, ignore_eos=True)

    for p in prompts:
        engine.add_request(p)
    for _ in range(6):  # warmup
        engine.step()

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            engine.step()
        metrics.sync(device)

    rows = sorted(prof.key_averages(), key=lambda r: r.self_device_time_total, reverse=True)
    return {"n_ticks": 20, "batch_size": batch, "ops": [
        {"name": r.key, "count": r.count,
         "self_cpu_us": r.self_cpu_time_total,
         "self_cuda_us": r.self_device_time_total}
        for r in rows[:30]
    ]}


EXPERIMENTS = {
    "e1": (e1_backend_sweep, MODELS),
    "e2": (e2_batch_sweep, MODELS),
    "e3": (e3_attend_microbench, [MODELS[0]]),
    "e4": (e4_arrival_stream, [MODELS[0]]),
    "e5": (e5_preemption, [MODELS[0]]),
    "e6": (e6_block_size_sweep, [MODELS[0]]),
    "e7": (e7_tick_profile, [MODELS[0]]),
}


def collect(names: list[str], device: str = "cuda") -> dict[str, dict]:
    results = {}
    for model_id in MODELS:
        todo = [n for n in names if model_id in EXPERIMENTS[n][1]]
        if not todo:
            continue
        tok, hf, mine, eos_ids = factories.load_model(model_id, DTYPE, device)
        ctx = {"tok": tok, "hf": hf, "model": mine, "eos_ids": eos_ids, "device": device}
        for name in todo:
            fn, _ = EXPERIMENTS[name]
            print(f"[collect] {name} on {model_id}")
            payload = fn(ctx)
            payload["meta"] = {
                "experiment": name, "model": model_id, "dtype": str(DTYPE), "device": device,
                "gpu": torch.cuda.get_device_name() if device == "cuda" else device,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            stem = f"{name}_{model_id.split('/')[-1].lower()}"
            results[stem] = payload
        ctx.clear()
        del hf, mine
        metrics.reclaim_mem_gb(device)
    return results
