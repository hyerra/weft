# weft

A KV cache and attention stack for LLM inference, built from first principles: a
contiguous baseline as a correctness oracle, then paged attention on top of it, with
each step verified token-for-token against HuggingFace `generate`.

## Shapes and symbols

| Symbol | Meaning |
|---|---|
| `B` | requests in the current step |
| `E` | hidden / embedding size (`E = H_q · D` for queries) |
| `H_q`, `H_kv` | query heads, key-value heads (`H_q / H_kv` = GQA group size) |
| `D` | head dimension |
| `L` | transformer layers |
| `V` | vocabulary size |
| `T` | new tokens this step, per request |
| `T_total` | new tokens this step, summed across requests |
| `S_max` | max sequence length a contiguous row can hold |
| `N` | blocks in the paged pool |
| `P` | block size (tokens per block) |

## Cache layouts

Each layer gets a view of one preallocated slab. The two backends deliberately
disagree about layout, because each is what its kernel wants:

| | Contiguous | Paged |
|---|---|---|
| slab | `[2, L, B, H_kv, S_max, D]` | `[2, L, N, P, H_kv, D]` |
| per-layer view | `[B, H_kv, S_max, D]` | `[N, P, H_kv, D]` |
| write | slice-assign at `cursor` | scatter via `slot_mapping` |
| read | slice `[:, :, :end, :]` | gather by `block_table` |
| ownership | static: request `b` owns row `b` | dynamic: block tables |

## Architecture

- `attention/` — backends: layout plus the kernels that read it. Stateless (immutable config only), model-agnostic, no allocation.
- `cache/` — the block pool: integer ids, free list, refcounts. No tensors.
- `inference/` — runners own the slab, allocation, and per-step metadata;
  `generate` owns sampling and the feedback loop.
- `models/` — Qwen3, which delegates all attention to whichever backend the runner
  hands it via `StepContext`.
