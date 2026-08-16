from dataclasses import dataclass
from typing import Protocol, Any

import torch


class AttentionBackend(Protocol):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, metadata: Any) -> torch.Tensor:
        ...


@dataclass
class StepContext:
    backend: AttentionBackend
    k_view: torch.Tensor
    v_view: torch.Tensor
    metadata: Any
