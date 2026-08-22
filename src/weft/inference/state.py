from dataclasses import dataclass
from enum import Enum

import torch


class FinishReason(Enum):
    EOS = 1
    MAX_TOKENS = 2
    CAPACITY = 3


class RequestStatus(Enum):
    WAITING = 1
    RUNNING = 2
    FINISHED = 3
    FAILED = 4


@dataclass
class Request:
    id: int
    prompt: torch.Tensor # [T_i] - T_p = number of input tokens
    output_ids: list[int] # [T_o] - T_o = number of output tokens
    status: RequestStatus
    num_computed_tokens: int
    finish_reason: FinishReason | None = None

    @property
    def num_tokens(self) -> int:
        return self.prompt.shape[0] + len(self.output_ids)

    @property
    def uncached_tokens(self) -> torch.Tensor:
        k = self.num_computed_tokens - len(self.prompt)
        output_ids = self.prompt.new_tensor(self.output_ids)
        if k >= 0:
            return output_ids[k:]
        return torch.cat([self.prompt, output_ids])[self.num_computed_tokens:]
