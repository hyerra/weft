import torch

from cache import KVCache


@torch.inference_mode()
def generate(model, input_ids, attn_mask, new_tokens, use_cache=True):
    B, prompt_tokens = input_ids.shape
    cache = KVCache.for_model(model, B, prompt_tokens + new_tokens) if use_cache else None

    for _ in range(new_tokens):
        logits = model(input_ids, attention_mask=attn_mask, past_key_values=cache)
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)

        # Cached, the keys/values for the prefix are already stored, so only the new
        # token is fed back. Uncached, the whole prefix is recomputed every step.
        input_ids = next_id if use_cache else torch.cat([input_ids, next_id], dim=-1)
        attn_mask = torch.cat([attn_mask, attn_mask.new_ones((B, 1))], dim=-1)

        yield next_id
