import torch
from torch import nn
from torch.nn import functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    tensor: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "bshd",
    fused: bool = False,
) -> torch.Tensor:
    del tensor_format, fused
    if freqs is None:
        return tensor

    original_shape = tensor.shape
    if tensor.ndim < 4:
        raise ValueError(f"Expected rotary tensor with at least 4 dims, got {original_shape}")

    batch = tensor.shape[0]
    heads = tensor.shape[-2]
    head_dim = tensor.shape[-1]
    seq_shape = tensor.shape[1:-2]
    flat = tensor.reshape(batch, -1, heads, head_dim)

    angles = freqs.reshape(freqs.shape[0], -1, freqs.shape[-1])
    if angles.shape[1] != 1:
        angles = angles[:, :1]
    angles = angles[: flat.shape[1]].to(device=tensor.device, dtype=tensor.dtype)

    cos = angles.cos().unsqueeze(0)
    sin = angles.sin().unsqueeze(0)
    out = (flat * cos) + (_rotate_half(flat) * sin)
    return out.reshape(batch, *seq_shape, heads, head_dim)


class DotProductAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads: int | None = None,
        kv_channels: int | None = None,
        *args,
        qkv_format: str = "bshd",
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        del num_attention_heads, kv_channels, args, kwargs
        self.qkv_format = qkv_format
        self.attention_dropout = attention_dropout

    def set_context_parallel_group(self, *args, **kwargs) -> None:
        del args, kwargs

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        q_shape = q.shape
        k_shape = k.shape
        batch = q_shape[0]
        q_seq_shape = q_shape[1:-2]
        q_flat = q.reshape(batch, -1, q_shape[-2], q_shape[-1]).transpose(1, 2)
        k_flat = k.reshape(batch, -1, k_shape[-2], k_shape[-1]).transpose(1, 2)
        v_flat = v.reshape(batch, -1, k_shape[-2], k_shape[-1]).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q_flat,
            k_flat,
            v_flat,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous()
        return out.reshape(batch, *q_seq_shape, q_shape[-2], q_shape[-1])


__all__ = ["DotProductAttention", "apply_rotary_pos_emb"]
