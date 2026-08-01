import torch


def rope(x, d, base=1e-4):
    # IN: (batch, seq_len, d_model)
    # OUT: (batch, seq_len, d_model)
    x_rope, x_pass = x[..., :d], x[..., d:]
    d_2 = d // 2
    neg_x_rope = torch.cat([-x_rope[..., d_2:], x_rope[..., :d_2]], dim=-1)
    seq_len = x.shape[-2]
    omega = base ** (torch.arange(0, d, 2) / d).to(x.device)
    seq_idx = torch.arange(0, seq_len).to(x.device)
    theta = torch.einsum('i,j->ij', seq_idx, omega).to(x.device)
    theta = torch.cat([theta, theta], dim=-1).to(x.device)

    x_rope = x_rope * theta.cos().to(x.dtype) + neg_x_rope * theta.sin().to(x.dtype)
    return torch.cat([x_rope, x_pass], dim=-1)
