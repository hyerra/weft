import torch

class KVCache:
    def __init__(self, num_hidden_layers, num_key_value_heads, max_seq_len, head_dim, device, dtype, batch_size=1):
        shape = (batch_size, num_key_value_heads, max_seq_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_hidden_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(num_hidden_layers)]
        self.length = 0

    @classmethod
    def for_model(cls, model, batch_size, max_seq_len):
        attn = model.model.layers[0].self_attn
        return cls(
            num_hidden_layers=len(model.model.layers),
            num_key_value_heads=attn.k_proj.out_features // attn.head_dim,
            max_seq_len=max_seq_len,
            head_dim=attn.head_dim,
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
            batch_size=batch_size,
        )

    def update(self, layer_idx, k_new, v_new):
        n = k_new.shape[-2]
        self.k[layer_idx][..., self.length:self.length+n, :] = k_new
        self.v[layer_idx][..., self.length:self.length+n, :] = v_new
        return (self.k[layer_idx][..., :self.length+n, :], self.v[layer_idx][..., :self.length+n, :])

    def advance(self, n):
        self.length += n
