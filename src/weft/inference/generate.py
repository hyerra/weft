import torch

from weft.inference.engine import Engine


@torch.inference_mode()
def generate(engine: Engine, prompts: list[torch.Tensor], arrivals: list[int] | None = None) -> list[list[int]]:
    arrivals = arrivals or [0] * len(prompts)
    ids: dict[int, int] = {}  # prompt index -> request id
    outputs: dict[int, list[int]] = {}
    tick = 0
    while len(ids) < len(prompts) or engine.has_unfinished():
        for i, arrival in enumerate(arrivals):
            if arrival == tick:
                ids[i] = engine.add_request(prompts[i])
                outputs[ids[i]] = []
        for out in engine.step():
            if out.token is not None:
                outputs[out.request_id].append(out.token)
        tick += 1
    return [outputs[ids[i]] for i in range(len(prompts))]
