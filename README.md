# weft

weft is a from-scratch LLM serving stack built from first principles.
It supports multiple scheduling policies and attention backends each benchmarked
in analysis.ipynb. Correctness is verified against HuggingFace `generate`.

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

The cache layout depends on the attention backend:

| | Contiguous | Paged |
|---|---|---|
| slab | `[2, L, B, H_kv, S_max, D]` | `[2, L, N, P, H_kv, D]` |
| per-layer view | `[B, H_kv, S_max, D]` | `[N, P, H_kv, D]` |
| write | slice-assign at `cursor` | scatter via `slot_mapping` |
| read | slice `[:, :, :end, :]` | gather by `block_table` |
| ownership | static: request `b` owns row `b` | dynamic: blocks given per request |
