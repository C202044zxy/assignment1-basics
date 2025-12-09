import torch
import torch.nn as nn
from einops import einsum


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ):
        super().__init__()
        self._eps = eps
        self._d_model = d_model
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        sqr = x**2
        rms = torch.sqrt(sqr.mean(-1, keepdim=True) + self._eps)
        x_normed = x / rms * self.g

        return x_normed.to(in_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(d_ff, d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.ones(d_model, d_ff, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.ones(d_ff, d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = einsum(x, self.w1, "... d_model, d_ff d_model -> ... d_ff")
        x1 = x1 * torch.sigmoid(x1)
        x2 = einsum(x, self.w3, "... d_model, d_ff d_model -> ... d_ff")
        return einsum(x1 * x2, self.w2, "... d_ff, d_model d_ff -> ... d_model")