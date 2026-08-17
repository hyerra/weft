from typing import Iterator

import torch

from weft.inference.runner import Runner

@torch.inference_mode()
def generate(runner: Runner, prompts: list[torch.Tensor], max_new_tokens: int) -> Iterator[torch.Tensor]:
    request_ids = list(range(len(prompts)))
    next_input = prompts
    for i in range(len(prompts)):
        runner.add_request(i)

    for _ in range(max_new_tokens):
        logits = runner.step(request_ids, next_input)
        next_tok = logits.argmax(-1, keepdim=True)
        next_input = list(next_tok.unbind(dim=0))
        yield next_tok

    for i in range(len(prompts)):
        runner.finish_request(i)
