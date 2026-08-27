from dataclasses import dataclass

import numpy as np
import torch

PROMPT_MIXES = {
    "uniform": [256, 256, 256],
    "mixed": [64, 256, 1024],
    "extreme": [16, 128, 2048],
}


def synthetic_prompts(lengths: list[int], vocab_size: int, device, seed: int = 0) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab_size, (n,), generator=g).to(device) for n in lengths]


def sample_mix(mix: str, n: int, seed: int = 0) -> list[int]:
    rng = np.random.default_rng(seed)
    return [int(rng.choice(PROMPT_MIXES[mix])) for _ in range(n)]


@dataclass
class ArrivalWorkload:
    lengths: list[int]
    arrivals: list[int]  # tick index per request


def poisson_arrivals(n: int, mean_interarrival_ticks: float, mix: str = "mixed",
                     seed: int = 0) -> ArrivalWorkload:
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(mean_interarrival_ticks, size=n)
    ticks = np.floor(np.cumsum(gaps)).astype(int)
    ticks[0] = 0  # someone arrives at the start so tick 0 has work
    return ArrivalWorkload(lengths=sample_mix(mix, n, seed=seed + 1), arrivals=ticks.tolist())
