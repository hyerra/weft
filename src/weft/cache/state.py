from dataclasses import dataclass

@dataclass
class RequestState:
    n_computed_tokens: int = 0
