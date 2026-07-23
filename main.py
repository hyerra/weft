import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 1000

@torch.inference_mode()
def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", device_map="auto", dtype="auto")

    messages = [
        [
            {"role": "user", "content": "Who are you?"}
        ],
        [
            {"role": "user", "content": "What is your best advice?"}
        ],
    ]
    B = len(messages)
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        padding=True,
        padding_side="left",
        return_tensors="pt"
    ).to(model.device)
    input_ids = inputs["input_ids"]
    attn_mask = inputs["attention_mask"]

    finished = torch.zeros((B, 1), dtype=torch.bool, device=model.device)
    stop_ids = torch.as_tensor(model.generation_config.eos_token_id, device=model.device)

    prompt_len = input_ids.shape[1]

    for _ in range(MAX_NEW_TOKENS):
        out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False).logits
        next_id = out[:, -1, :].argmax(-1, keepdim=True).masked_fill(finished, tokenizer.pad_token_id)
        finished |= torch.isin(next_id, stop_ids)
        input_ids = torch.cat([input_ids, next_id], dim=-1)
        attn_mask = torch.cat([attn_mask, attn_mask.new_ones((B, 1))], dim=-1)
        if finished.all():
            break

    responses = tokenizer.decode(input_ids[:, prompt_len:], skip_special_tokens=True)
    print(responses)


if __name__ == "__main__":
    main()
