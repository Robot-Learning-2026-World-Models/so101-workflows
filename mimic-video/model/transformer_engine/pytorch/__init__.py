import torch
from torch import nn

from . import distributed
from .attention import DotProductAttention


class RMSNorm(nn.Module):
    def __init__(self, dim: int | None = None, eps: float = 1e-5, **kwargs):
        super().__init__()
        dim = dim if dim is not None else kwargs["hidden_size"]
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return output * self.weight


class LayerNorm(nn.LayerNorm):
    def __init__(self, hidden_size: int, eps: float = 1e-5, **kwargs):
        del kwargs
        super().__init__(hidden_size, eps=eps)


class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *args,
        bias: bool = True,
        return_bias: bool = False,
        init_method=None,
        **kwargs,
    ):
        del args, kwargs
        super().__init__(in_features, out_features, bias=bias)
        self.return_bias = return_bias
        self.use_bias = bias
        self.parallel_mode = None
        self.tp_size = 1
        if init_method is not None:
            init_method(self.weight)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        del args, kwargs
        out = super().forward(x)
        if self.return_bias:
            return out, self.bias
        return out


class LayerNormLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *args,
        eps: float = 1e-5,
        bias: bool = True,
        return_bias: bool = False,
        init_method=None,
        normalization: str = "LayerNorm",
        **kwargs,
    ):
        del args, kwargs
        super().__init__()
        norm_cls = RMSNorm if normalization == "RMSNorm" else LayerNorm
        self.layer_norm = norm_cls(in_features, eps=eps)
        self.linear = Linear(
            in_features,
            out_features,
            bias=bias,
            return_bias=return_bias,
            init_method=init_method,
        )
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        self.return_bias = return_bias
        self.use_bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = 1

    def forward(self, x: torch.Tensor, *args, **kwargs):
        del args, kwargs
        return self.linear(self.layer_norm(x))


class GroupedLinear(nn.Module):
    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        *args,
        bias: bool = True,
        return_bias: bool = False,
        init_method=None,
        **kwargs,
    ):
        del args, kwargs
        super().__init__()
        self.num_gemms = num_gemms
        self.layers = nn.ModuleList(
            Linear(in_features, out_features, bias=bias, return_bias=return_bias, init_method=init_method)
            for _ in range(num_gemms)
        )
        self.return_bias = return_bias
        self.use_bias = bias

    def forward(self, x: torch.Tensor, m_splits=None, *args, **kwargs):
        del args, kwargs
        if m_splits is None:
            return self.layers[0](x)
        chunks = torch.split(x, list(m_splits), dim=0)
        outputs = [layer(chunk) for layer, chunk in zip(self.layers, chunks, strict=False)]
        if self.return_bias:
            values = [item[0] if isinstance(item, tuple) else item for item in outputs]
            biases = [item[1] if isinstance(item, tuple) else None for item in outputs]
            return torch.cat(values, dim=0), biases
        return torch.cat(outputs, dim=0)


__all__ = [
    "DotProductAttention",
    "GroupedLinear",
    "LayerNorm",
    "LayerNormLinear",
    "Linear",
    "RMSNorm",
    "distributed",
]
