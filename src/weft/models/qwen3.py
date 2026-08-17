"""Qwen3, with attention delegated to whichever backend the runner supplies.

Shapes follow weft.attention.base: B requests, T new tokens per request, E hidden
size, H_q / H_kv heads, D head dimension.
"""

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from weft.attention.base import StepContext
from weft.layers.rope import rope


class Qwen3Attention(nn.Module):
    def __init__(self, layer_idx, config):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads_per_group = config.num_attention_heads // config.num_key_value_heads
        self.rope_theta = config.rope_parameters["rope_theta"]
        self.q_proj = nn.Linear(config.hidden_size, config.head_dim * config.num_attention_heads, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.head_dim * config.num_key_value_heads, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.head_dim * config.num_key_value_heads, bias=config.attention_bias)
        self.o_proj = nn.Linear(config.head_dim * config.num_attention_heads, config.hidden_size, bias=config.attention_bias)
        self.q_norm = nn.RMSNorm((config.head_dim), eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm((config.head_dim), eps=config.rms_norm_eps)

    def forward(self, hidden_states, context: StepContext, position_ids: torch.Tensor):
        # In: (B, T, E)
        # Out: (B, T, E)
        input_shape = hidden_states.shape[:-1]

        # (B, T, H_q, D)
        hidden_shape = (*input_shape, -1, self.head_dim)

        # (B, H_q, T, D)
        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        # (B, H_kv, T, D)
        k_new = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v_new = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        q, k_new = rope(q, self.head_dim, base=1/self.rope_theta, position_ids=position_ids), rope(k_new, self.head_dim, base=1/self.rope_theta, position_ids=position_ids)

        attn_output = context.backend.forward(q, k_new, v_new, context.k_view[self.layer_idx], context.v_view[self.layer_idx], context.metadata)

        # (B, T, E)
        out = self.o_proj(attn_output.transpose(1, 2).reshape(*input_shape, -1))
        return out


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation = nn.SiLU()

    def forward(self, hidden_states):
        return self.down_proj(self.up_proj(hidden_states) * self.activation(self.gate_proj(hidden_states)))


class Qwen3DecoderBlock(nn.Module):
    def __init__(self, layer_idx, config):
        super().__init__()
        self.input_layernorm = nn.RMSNorm((config.hidden_size), eps=config.rms_norm_eps)
        self.self_attn = Qwen3Attention(layer_idx, config)
        self.post_attention_layernorm = nn.RMSNorm((config.hidden_size), eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, x, context: StepContext, position_ids: torch.Tensor):
        x = x + self.self_attn(self.input_layernorm(x), context, position_ids=position_ids)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([Qwen3DecoderBlock(layer_idx, config) for layer_idx in range(config.num_hidden_layers)])
        self.norm = nn.RMSNorm((config.hidden_size), eps=config.rms_norm_eps)

    def forward(self, x, context: StepContext, position_ids: torch.Tensor):
        # In: (B, T)
        # Out: (B, T, E)
        x = self.embed_tokens(x)
        for l in self.layers:
            x = l(x, context, position_ids=position_ids)
        x = self.norm(x)
        return x


class Qwen3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_config(cls, config, dtype=torch.float32):
        return cls(config).to(dtype).eval()

    @classmethod
    def from_pretrained(cls, model_id="Qwen/Qwen3-0.6B", dtype=torch.float32):
        model = cls.from_config(AutoConfig.from_pretrained(model_id), dtype)
        hf = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        model.load_state_dict(hf.state_dict(), strict=False)
        return model

    def forward(self, x, context: StepContext, position_ids: torch.Tensor):
        # In: (B, T)
        # Out: (B, T, V)
        x = self.model(x, context, position_ids=position_ids)
        x = self.lm_head(x)
        return x
