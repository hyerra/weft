import torch


@torch.inference_mode()
def generate(model, input_ids, attn_mask, new_tokens):
    B = input_ids.shape[0]

    for _ in range(new_tokens):
        logits = model(input_ids, attention_mask=attn_mask)
        next_id = logits[:, -1, :].argmax(-1, keepdim=True)

        input_ids = torch.cat([input_ids, next_id], dim=-1)
        attn_mask = torch.cat([attn_mask, attn_mask.new_ones((B, 1))], dim=-1)

        yield next_id
