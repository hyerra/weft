import gc
import time
from dataclasses import dataclass, field

import numpy as np
import torch

PERCENTILES = [50, 90, 95, 99]


def reclaim_mem_gb(device: str) -> float:
    # gc before empty_cache: cycle-held tensors must be freed for their segments
    # to be releasable. Return value is the leak detector: live bytes after full
    # teardown should be just the resident weights, flat across rows.
    gc.collect()
    if device != "cuda":
        return 0.0
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated() / 1e9


def sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@dataclass
class RequestTrace:
    arrival_tick: int
    arrival_s: float = 0.0
    first_token_tick: int | None = None
    first_token_s: float | None = None
    completion_tick: int | None = None
    completion_s: float | None = None
    n_generated: int = 0
    tokens: list[int] = field(default_factory=list)

    @property
    def ttft_s(self) -> float:
        return self.first_token_s - self.arrival_s


@dataclass
class EngineTrace:
    tick_s: list[float] = field(default_factory=list)
    requests: dict[int, RequestTrace] = field(default_factory=dict)
    completion_order: list[int] = field(default_factory=list)

    @property
    def total_s(self) -> float:
        return float(sum(self.tick_s))

    @property
    def total_generated(self) -> int:
        return sum(r.n_generated for r in self.requests.values())

    @property
    def output_tps(self) -> float:
        return self.total_generated / self.total_s

    def ttft_percentiles(self) -> dict[str, float]:
        ttfts = [r.ttft_s for r in self.requests.values()]
        return {f"p{p}": float(np.percentile(ttfts, p)) for p in PERCENTILES}

    def itl_percentiles(self) -> dict[str, float]:
        return {f"p{p}": float(np.percentile(self.tick_s[1:], p)) for p in PERCENTILES}

    def to_json(self) -> dict:
        return {
            "tick_s": self.tick_s,
            "total_s": self.total_s,
            "total_generated": self.total_generated,
            "output_tps": self.output_tps,
            "ttft": self.ttft_percentiles(),
            "itl": self.itl_percentiles(),
            "completion_order": self.completion_order,
            "requests": {
                str(rid): {
                    "arrival_tick": r.arrival_tick, "arrival_s": r.arrival_s,
                    "first_token_tick": r.first_token_tick, "first_token_s": r.first_token_s,
                    "completion_tick": r.completion_tick, "completion_s": r.completion_s,
                    "ttft_s": r.ttft_s, "n_generated": r.n_generated,
                }
                for rid, r in self.requests.items()
            },
        }


@torch.inference_mode()
def run_timed(engine, prompts: list[torch.Tensor], device: str,
              arrivals: list[int] | None = None) -> EngineTrace:
    arrivals = arrivals or [0] * len(prompts)
    trace = EngineTrace()
    n_admitted = 0
    tick = 0
    clock = 0.0

    sync(device)
    while n_admitted < len(prompts) or engine.has_unfinished():
        for i, arrival in enumerate(arrivals):
            if arrival == tick:
                rid = engine.add_request(prompts[i])
                trace.requests[rid] = RequestTrace(arrival_tick=tick, arrival_s=clock)
                n_admitted += 1

        t0 = time.perf_counter()
        outs = engine.step()
        sync(device)
        dt = time.perf_counter() - t0
        clock += dt
        trace.tick_s.append(dt)

        for out in outs:
            r = trace.requests[out.request_id]
            if out.token is not None:
                r.n_generated += 1
                r.tokens.append(out.token)
                if r.first_token_tick is None:
                    r.first_token_tick, r.first_token_s = tick, clock
            if out.finish_reason is not None:
                r.completion_tick, r.completion_s = tick, clock
                trace.completion_order.append(out.request_id)
        tick += 1

    return trace


def cuda_time_ms(fn, warmup: int = 5, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return float(np.median(times))
