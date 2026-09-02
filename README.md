# weft

weft is a from-scratch LLM serving stack built from first principles.
It supports multiple scheduling policies and attention backends each benchmarked
in analysis.ipynb. Correctness is verified against HuggingFace `generate`.

## Quickstart

This runs the token-exact correctness check against HuggingFace:
```bash
uv run main.py
```

This runs the token-exact correctness check against HuggingFace in Modal:
```bash
uv run main.py --modal
```

This reproduces all the experiments in Modal:
```bash
uv run main.py --collect all
```

## Architecture

- `/models`: common models reimplemented in PyTorch and written to make use of weft's infra
- `/layers`: reusable components that are shared between models
- `/attention`: multiple implementations of attention. defines the shape of the kv-cache along with the attention computation
- `/inference`: the infrastructure to perform inference. scheduler decides what requests
run on each tick and allocates space for it, runner executes a forward pass of the model
for the scheduled requests
- `/benchmark`: sets up the experiments and metrics that the analysis notebook depends on

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

## Performance Analysis

You can find detailed visualizations around the performance of weft in the `analysis.ipynb` notebook.

Some important results include:
1. The p50 TTFT drops (64 requests, mixed length) drops from 10s -> 1.5s when switching from static scheduling and contiguous attention to the continuous scheduling with triton kernel.
2. The same 64 request workload finishes in 12 seconds vs 28 seconds when comparing continuous_triton vs static_contiguous.
3. The triton kernel scales much better than any of the other attention backends.
4. Triton kernel ITL p50 is 36ms vs 47ms contiguous vs 59 ms paged-gather.
5. The throughput degrades gracefully as we reduce the pool size.
