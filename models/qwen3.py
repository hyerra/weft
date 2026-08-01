import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from layers.rope import rope


class Qwen3Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_attention_heads, num_key_value_heads, bias, rope_theta, rms_norm_eps):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads_per_group = num_attention_heads // num_key_value_heads
        self.rope_theta = rope_theta
        self.q_proj = nn.Linear(hidden_size, head_dim * num_attention_heads, bias=bias)
        self.k_proj = nn.Linear(hidden_size, head_dim * num_key_value_heads, bias=bias)
        self.v_proj = nn.Linear(hidden_size, head_dim * num_key_value_heads, bias=bias)
        self.o_proj = nn.Linear(head_dim * num_attention_heads, hidden_size, bias=bias)
        self.q_norm = nn.RMSNorm((head_dim), eps=rms_norm_eps)
        self.k_norm = nn.RMSNorm((head_dim), eps=rms_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        # In: (batch, seq_len, hidden_size)
        # Out: (batch, seq_len, hidden_size)
        input_shape = hidden_states.shape[:-1]

        # (batch, seq_len, num_heads, head_dim)
        hidden_shape = (*input_shape, -1, self.head_dim)

        # (batch, num_heads, seq_len, head_dim)
        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        q, k = rope(q, self.head_dim, base=1/self.rope_theta), rope(k, self.head_dim, base=1/self.rope_theta)

        k = k.repeat_interleave(self.num_heads_per_group, dim=1)
        v = v.repeat_interleave(self.num_heads_per_group, dim=1)

        # (batch, num_heads, seq_len, seq_len)
        scores = q @ k.transpose(-2, -1) / self.head_dim**0.5
        if attention_mask is not None:
            scores = scores + attention_mask
        attn = torch.softmax(scores, dim=-1)

        # (batch, num_heads, seq_len, head_dim)
        attn_output = attn @ v

        # (batch, seq_len, hidden_size)
        out = self.o_proj(attn_output.transpose(1, 2).reshape(*input_shape, -1))
        return out


class Qwen3MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, hidden_states):
        return self.down_proj(self.up_proj(hidden_states) * self.activation(self.gate_proj(hidden_states)))


class Qwen3DecoderBlock(nn.Module):
    def __init__(self, hidden_size, head_dim, num_attention_heads, num_key_value_heads, intermediate_size, attention_bias, rope_theta, rms_norm_eps):
        super().__init__()
        self.input_layernorm = nn.RMSNorm((hidden_size), eps=rms_norm_eps)
        self.self_attn = Qwen3Attention(hidden_size, head_dim, num_attention_heads, num_key_value_heads, attention_bias, rope_theta, rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm((hidden_size), eps=rms_norm_eps)
        self.mlp = Qwen3MLP(hidden_size, intermediate_size)

    def forward(self, x, attention_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), attention_mask=attention_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, vocab_size, pad_token_id, hidden_size, head_dim, num_attention_heads, num_key_value_heads, intermediate_size, num_hidden_layers, attention_bias, rope_theta, rms_norm_eps):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.layers = nn.ModuleList([Qwen3DecoderBlock(hidden_size, head_dim, num_attention_heads, num_key_value_heads, intermediate_size, attention_bias, rope_theta, rms_norm_eps) for _ in range(num_hidden_layers)])
        self.norm = nn.RMSNorm((hidden_size), eps=rms_norm_eps)

    def forward(self, x, attention_mask=None):
        # In: (batch, seq_len)
        # Out: (batch, seq_len, hidden_size)
        x = self.embed_tokens(x)
        seq_len = x.shape[-2]
        causal_mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device).tril().unsqueeze(0)
        # (batch, heads (broadcasted), seq_len, seq_len)
        mask = (attention_mask[:, None, :].bool() & causal_mask if attention_mask is not None else causal_mask).unsqueeze(1)
        mask = torch.zeros_like(mask, dtype=x.dtype).masked_fill(~mask.bool(), torch.finfo(x.dtype).min)
        for l in self.layers:
            x = l(x, attention_mask=mask)
        x = self.norm(x)
        return x


class Qwen3(nn.Module):
    def __init__(self, vocab_size, pad_token_id, hidden_size, head_dim, num_attention_heads, num_key_value_heads, intermediate_size, num_hidden_layers, attention_bias, rope_theta, rms_norm_eps):
        super().__init__()
        self.model = Qwen3Model(vocab_size, pad_token_id, hidden_size, head_dim, num_attention_heads, num_key_value_heads, intermediate_size, num_hidden_layers, attention_bias, rope_theta, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @classmethod
    def from_config(cls, config, dtype=torch.float32):
        return cls(config.vocab_size, config.pad_token_id, config.hidden_size, config.head_dim, config.num_attention_heads, config.num_key_value_heads, config.intermediate_size, config.num_hidden_layers, config.attention_bias, config.rope_parameters['rope_theta'], config.rms_norm_eps).to(dtype).eval()

    @classmethod
    def from_pretrained(cls, model_id="Qwen/Qwen3-0.6B", dtype=torch.float32):
        model = cls.from_config(AutoConfig.from_pretrained(model_id), dtype)
        hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        model.load_state_dict(hf.state_dict(), strict=False)
        return model

    def forward(self, x, attention_mask=None):
        # In: (batch, seq_len)
        # Out: (batch, seq_len, vocab_size)
        x = self.model(x, attention_mask=attention_mask)
        x = self.lm_head(x)
        return x


if __name__ == "__main__":
    DTYPE = torch.float32
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
    print(config)

    hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=DTYPE)
    for k, v in hf.state_dict().items():
        print(k, tuple(v.shape))

    mine = Qwen3.from_config(config, DTYPE)
    missing, unexpected = mine.load_state_dict(hf.state_dict(), strict=False)
    print("missing:", missing); print("unexpected:", unexpected)
