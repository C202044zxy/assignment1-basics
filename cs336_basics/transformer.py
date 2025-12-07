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
