from dataclasses import dataclass

import torch

from weft.inference.runner import Runner
from weft.inference.scheduler import Scheduler
from weft.inference.state import Request, RequestStatus, FinishReason


@dataclass
class RequestOutput:
    request_id: int
    token: int | None
    finish_reason: FinishReason | None


class Engine:
    def __init__(self, runner: Runner, scheduler: Scheduler, eos_ids: set[int], max_tokens: int):
        self._requests: dict[int, Request] = {}
        self._next_id = 0
        self.runner = runner
        self.scheduler = scheduler
        self._eos_ids = eos_ids
        self._max_tokens = max_tokens

    def add_request(self, prompt: torch.Tensor) -> int:
        id = self._next_id
        self._next_id += 1
        req = Request(id, prompt, [], RequestStatus.WAITING, 0)
        self._requests[id] = req
        self.scheduler.add(req)
        return id

    def step(self) -> list[RequestOutput]:
        out: list[RequestOutput] = []
        requests = self.scheduler.schedule()
        for failed in requests.failed:
            req = self._requests[failed.id]
            req.finish_reason = failed.finish_reason
            req.status = RequestStatus.FAILED
            out.append(
                RequestOutput(
                    failed.id, None, failed.finish_reason
                )
            )
        if not requests.scheduled:
            return out
        toks = self.runner.step(requests.scheduled).argmax(dim=-1)
        for scheduled, tok in zip(requests.scheduled, toks.tolist()):
            req = self._requests[scheduled.request_id]
            status = RequestStatus.RUNNING
            finish_reason: FinishReason | None = None
            req.num_computed_tokens += len(scheduled.input_ids)
            req.output_ids.append(tok)
            if tok in self._eos_ids:
                finish_reason = FinishReason.EOS
            elif len(req.output_ids) >= self._max_tokens:
                finish_reason = FinishReason.MAX_TOKENS
            req.finish_reason = finish_reason
            if req.finish_reason is not None:
                self.scheduler.finish(req)
                del self._requests[req.id]
            out.append(
                RequestOutput(scheduled.request_id, tok, req.finish_reason)
            )
        return out

    def has_unfinished(self) -> bool:
        return bool(self._requests)
