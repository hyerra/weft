import math
from collections import deque
from dataclasses import dataclass
from typing import Protocol

from weft.inference.state import Request, RequestStatus, FinishReason
from weft.inference.runner import ScheduledRequest


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self._num_blocks = num_blocks
        self.block_size = block_size
        self._free_blocks = deque(range(num_blocks))
        self._table: dict[int, list[int]] = {}

    @property
    def num_blocks(self):
        return self._num_blocks

    @property
    def free_blocks(self) -> int: 
        return len(self._free_blocks)

    def _blocks_needed(self, request_id, total_tokens) -> int:
        return max(0, math.ceil(total_tokens / self.block_size) - len(self._table.get(request_id, [])))

    def can_allocate(self, request_id: int, total_tokens: int) -> bool:
        return self.free_blocks >= self._blocks_needed(request_id, total_tokens)

    def allocate(self, request_id: int, total_tokens: int) -> list[int]:
        if not self.can_allocate(request_id, total_tokens):
            raise ValueError("Not enough free space available.")
        blocks_needed = self._blocks_needed(request_id, total_tokens)
        for _ in range(blocks_needed):
            if request_id not in self._table:
                self._table[request_id] = []
            self._table[request_id].append(self._free_blocks.popleft())
        return self._table[request_id]

    def table(self, request_id: int) -> list[int]:
        return self._table[request_id]

    def free(self, request_id: int) -> None:
        for block in self._table[request_id]:
            self._free_blocks.append(block)
        del self._table[request_id]


@dataclass
class SchedulingBudget:
    max_seqs: int
    max_tokens: int
    seqs: int = 0
    tokens: int = 0

    def try_take(self, num_tokens: int) -> bool:
        if self.seqs + 1 > self.max_seqs or self.tokens + num_tokens > self.max_tokens:
            return False
        self.seqs += 1
        self.tokens += num_tokens
        return True


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledRequest]
    failed: list[Request]


def _add(request: Request, max_tokens: int, waiting: deque[Request]) -> None:
    if request.prompt.shape[0] > max_tokens:
        raise ValueError("Prompt doesn't fit within max_num_batched_tokens!")
    request.status = RequestStatus.WAITING
    waiting.append(request)


def _fail_if_infeasible(req: Request, block_manager: BlockManager, running: list[Request], waiting: deque[Request]) -> bool:
    if math.ceil(req.num_tokens / block_manager.block_size) <= block_manager.num_blocks:
        return False
    req.finish_reason = FinishReason.CAPACITY
    if req.status is RequestStatus.RUNNING:
        block_manager.free(req.id)
        running.remove(req)
    else:
        waiting.remove(req)
    req.status = RequestStatus.FAILED
    return True

def _preempt(victim: Request, block_manager: BlockManager, running: list[Request], waiting: deque[Request]) -> None:
    block_manager.free(victim.id)
    victim.status = RequestStatus.WAITING
    victim.num_computed_tokens = 0
    running.remove(victim)
    waiting.appendleft(victim)


def _try_schedule(req: Request, budget: SchedulingBudget, block_manager: BlockManager, scheduled: list[ScheduledRequest]) -> bool:
    if not (block_manager.can_allocate(req.id, req.num_tokens) and budget.try_take(req.uncached_tokens.shape[0])):
        return False
    block_manager.allocate(req.id, req.num_tokens)
    scheduled.append(ScheduledRequest(
        req.id, req.uncached_tokens, req.num_computed_tokens, block_manager.table(req.id)
    ))
    return True


def _admit_waiting(budget: SchedulingBudget, waiting: deque[Request], running: list[Request], block_manager: BlockManager, scheduled: list[ScheduledRequest], failed: list[Request]) -> None:
    while waiting:
        req = waiting[0]
        if _fail_if_infeasible(req, block_manager, running, waiting):
            failed.append(req)
            continue
        if not _try_schedule(req, budget, block_manager, scheduled):
            break
        running.append(waiting.popleft())
        req.status = RequestStatus.RUNNING


def _schedule_running(running: list[Request], waiting: deque[Request], budget: SchedulingBudget, block_manager: BlockManager, scheduled: list[ScheduledRequest], failed: list[Request]) -> None:
    for req in list(running):
        if _fail_if_infeasible(req, block_manager, running, waiting):
            failed.append(req)
        if req.status is not RequestStatus.RUNNING:
            continue
        while not block_manager.can_allocate(req.id, req.num_tokens):
            victim = running[-1]
            _preempt(victim, block_manager, running, waiting)
            if victim is req:
                # No emit can follow because freeing our own blocks
                # raises blocks needed by exactly what it frees.
                # Thus, can_allocate stays false and _try_schedule
                # below is guaranteed a no-op.
                break
        _try_schedule(req, budget, block_manager, scheduled)


def _finish(request: Request, block_manager: BlockManager, running: list[Request], waiting: deque[Request]) -> None:
    if request.status in (RequestStatus.FINISHED, RequestStatus.FAILED):
        return
    if request.status is RequestStatus.RUNNING:
        running.remove(request)
        block_manager.free(request.id)
    else:
        waiting.remove(request)
    request.status = RequestStatus.FINISHED


class Scheduler(Protocol):
    def add(self, request: Request) -> None: ...
    def schedule(self) -> SchedulerOutput: ...
    def finish(self, request: Request) -> None: ...


class StaticScheduler(Scheduler):
    def __init__(self, block_manager: BlockManager, max_seqs: int, max_tokens: int):
        self._block_manager = block_manager
        self._max_seqs = max_seqs
        self._max_tokens = max_tokens
        self._running: list[Request] = []
        self._waiting: deque[Request] = deque()

    def add(self, request: Request) -> None:
        _add(request, self._max_tokens, self._waiting)

    def schedule(self) -> SchedulerOutput:
        scheduled: list[ScheduledRequest] = []
        failed: list[Request] = []

        budget = SchedulingBudget(self._max_seqs, self._max_tokens)
        # Create a new batch of requests
        if not self._running:
            _admit_waiting(budget, self._waiting, self._running, self._block_manager, scheduled, failed)
            return SchedulerOutput(scheduled, failed)
        # Otherwise, only handle in-flight requests
        _schedule_running(self._running, self._waiting, budget, self._block_manager, scheduled, failed)
        return SchedulerOutput(scheduled, failed)

    def finish(self, request: Request) -> None:
        _finish(request, self._block_manager, self._running, self._waiting)


class ContinuousScheduler(Scheduler):
    def __init__(self, block_manager: BlockManager, max_seqs: int, max_tokens: int):
        self._block_manager = block_manager
        self._max_seqs = max_seqs
        self._max_tokens = max_tokens
        self._running: list[Request] = []
        self._waiting: deque[Request] = deque()

    def add(self, request: Request) -> None:
        _add(request, self._max_tokens, self._waiting)

    def schedule(self) -> SchedulerOutput:
        scheduled: list[ScheduledRequest] = []
        failed: list[Request] = []

        budget = SchedulingBudget(self._max_seqs, self._max_tokens)

        _schedule_running(self._running, self._waiting, budget, self._block_manager, scheduled, failed)
        _admit_waiting(budget, self._waiting, self._running, self._block_manager, scheduled, failed)
        return SchedulerOutput(scheduled, failed)

    def finish(self, request: Request) -> None:
        _finish(request, self._block_manager, self._running, self._waiting)
